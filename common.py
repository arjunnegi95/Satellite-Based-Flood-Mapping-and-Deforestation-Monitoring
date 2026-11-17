"""Shared config, data, model, loss, metrics and training code for the whole project."""
import os, glob, random
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
import albumentations as A
import segmentation_models_pytorch as smp

# ---------------- paths ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
FLOOD_DIR = os.path.join(DATA_DIR, 'sen1floods11')          # contains S2/ and LBL/
MODEL_DIR = os.path.join(ROOT, 'models')
OUT_DIR = os.path.join(ROOT, 'outputs')
SAMPLE_DIR = os.path.join(ROOT, 'samples')
for _d in (DATA_DIR, MODEL_DIR, OUT_DIR, SAMPLE_DIR):
    os.makedirs(_d, exist_ok=True)

NPZ = os.path.join(DATA_DIR, 'deforestation_patches.npz')
CKPT_FLOOD = os.path.join(MODEL_DIR, 'best_flood_unet.pt')
CKPT_CD = os.path.join(MODEL_DIR, 'best_change_unet.pt')

# ---------------- config ----------------
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS = 0 if os.name == 'nt' else 2                    # 0 is safest on Windows
GEE_PROJECT = os.environ.get('GEE_PROJECT', 'sattelite-monitoring')      # your Earth Engine / Cloud project ID

BANDS = [1, 2, 3, 7, 11, 12]       # B2,B3,B4,B8,B11,B12 from the 13-band Sen1Floods11 chip
S2_SCALE = 0.3                     # flood: reflectance / 10000 / 0.3, clipped to [0,1]
CD_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11']
CD_SCALE = 0.5                     # deforestation: reflectance / 0.5, clipped to [0,1]

# ---------------- class imbalance weight ----------------
# ~4.6% of pixels are deforestation → weight positives ~20x more
DEFO_POS_WEIGHT = 20.0


def seed_everything(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


seed_everything()

# ---------------- flood data ----------------
def load_s2(path):
    """Sentinel-2 GeoTIFF -> float32 (C,H,W) in [0,1]."""
    with rasterio.open(path) as src:
        img = src.read().astype('float32')
    if img.shape[0] == 13:
        img = img[BANDS]
    return np.clip(np.nan_to_num(img) / 10000.0 / S2_SCALE, 0, 1)


def list_flood_ids():
    ids = []
    for p in sorted(glob.glob(os.path.join(FLOOD_DIR, 'S2', '*_S2Hand.tif'))):
        n = os.path.basename(p).replace('_S2Hand.tif', '')
        if os.path.exists(os.path.join(FLOOD_DIR, 'LBL', f'{n}_LabelHand.tif')):
            ids.append(n)
    return ids


def split_flood_ids():
    from sklearn.model_selection import train_test_split
    return train_test_split(list_flood_ids(), test_size=0.2, random_state=SEED)   # 80/20


class FloodDS(Dataset):
    """Label file: -1 no data, 0 land, 1 water -> stored internally as 0/1/2 (uint8)."""
    def __init__(self, ids, train):
        self.ids = ids
        self.aug = A.Compose([A.RandomCrop(height=256, width=256),
                              A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90()]) if train else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        n = self.ids[i]
        x = load_s2(os.path.join(FLOOD_DIR, 'S2', f'{n}_S2Hand.tif')).transpose(1, 2, 0)
        with rasterio.open(os.path.join(FLOOD_DIR, 'LBL', f'{n}_LabelHand.tif')) as src:
            y = (src.read(1) + 1).astype('uint8')
        if self.aug:
            o = self.aug(image=x, mask=y); x, y = o['image'], o['mask']
        valid = (y > 0); y = (y == 2)
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()).float(),
                torch.from_numpy(y.astype('float32')), torch.from_numpy(valid.astype('float32')))


# ---------------- deforestation feature engineering ----------------
def cd_features(x):
    """
    Raw float32 (10, H, W) patch (5 bands × 2 dates: B2,B3,B4,B8,B11 for year1 then year2)
    -> 13-channel tensor:
        channels  0-9  : 10 scaled reflectance bands (clipped to [0,1])
        channel  10    : NDVI year1   = (B8_1 - B4_1) / (B8_1 + B4_1), mapped to [0,1]
        channel  11    : NDVI year2   = (B8_2 - B4_2) / (B8_2 + B4_2), mapped to [0,1]
        channel  12    : ΔNDVI        = NDVI2 - NDVI1,  mapped to [0,1] (drop = negative values)
    Adding NDVI explicitly gives the model the same signal the rule-based baseline uses.
    """
    b = np.clip(x / CD_SCALE, 0, 1)
    # year1: B2=0, B3=1, B4=2, B8=3, B11=4  |  year2: B2=5, B3=6, B4=7, B8=8, B11=9
    n1 = (x[3] - x[2]) / (x[3] + x[2] + 1e-6)   # NDVI year 1
    n2 = (x[8] - x[7]) / (x[8] + x[7] + 1e-6)   # NDVI year 2
    f1 = np.clip(n1, -1, 1) * 0.5 + 0.5           # rescale to [0,1]
    f2 = np.clip(n2, -1, 1) * 0.5 + 0.5
    fd = np.clip(n2 - n1, -0.5, 0.5) + 0.5        # ΔNDVI rescaled to [0,1]
    return np.concatenate([b, f1[None], f2[None], fd[None]]).astype('float32')


# ---------------- deforestation data ----------------
def split_defo_idx(n):
    perm = np.random.RandomState(SEED).permutation(n); k = int(0.8 * n)
    return perm[:k], perm[k:]


class ArrDS(Dataset):
    """
    Deforestation patch dataset.
    Uses 13-channel engineered features (cd_features) instead of raw 10 bands,
    and applies stronger augmentation during training.
    """
    def __init__(self, X, Y, train):
        self.X, self.Y = X, Y
        if train:
            self.aug = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomCrop(height=192, width=192, p=0.5),   # random sub-crop
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0, p=0.4),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
                # simulate cloud-masked gaps (like in real satellite imagery)
                A.CoarseDropout(max_holes=4, max_height=16, max_width=16,
                                min_holes=1, min_height=4, min_width=4,
                                fill_value=0, p=0.3),
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        # Build 13-channel engineered features
        x = cd_features(self.X[i].astype('float32'))   # (13, H, W)
        y = self.Y[i]                                   # (H, W) uint8
        # albumentations expects (H, W, C)
        x = x.transpose(1, 2, 0)
        if self.aug:
            o = self.aug(image=x, mask=y); x, y = o['image'], o['mask']
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()).float(),
                torch.from_numpy(y.astype('float32')), torch.ones(y.shape))


# ---------------- model, loss, metrics ----------------
def build_unet(in_ch, pretrained=True, arch='unet'):
    """
    Build a segmentation model.
    arch choices: 'unet' (default), 'unet++' (better skip connections),
                  'deeplabv3+' (good for sparse targets).
    """
    kwargs = dict(encoder_name='resnet34',
                  encoder_weights='imagenet' if pretrained else None,
                  in_channels=in_ch,
                  classes=1)
    if arch == 'unet++':
        return smp.UnetPlusPlus(**kwargs)
    if arch == 'deeplabv3+':
        return smp.DeepLabV3Plus(**kwargs)
    return smp.Unet(**kwargs)


def masked_loss(logits, y, valid, eps=1e-6, pos_weight=DEFO_POS_WEIGHT):
    """
    Weighted BCE + Dice loss, ignoring no-data pixels.

    pos_weight: scalar weight applied to positive (deforestation) pixels in BCE.
                Set to ~20 for the 4.6% positive-fraction dataset so the model
                is penalised 20x more for missing a deforestation pixel than for
                a false alarm.  Pass pos_weight=1.0 to restore the original behaviour.
    """
    logits = logits.squeeze(1); valid = valid.float()

    # Weighted BCE — penalises false negatives heavily
    pw = torch.tensor([pos_weight], device=logits.device)
    bce = (F.binary_cross_entropy_with_logits(
               logits, y, pos_weight=pw, reduction='none') * valid
           ).sum() / valid.sum().clamp(min=1)

    # Dice loss — directly optimises the overlap metric
    p = torch.sigmoid(logits) * valid; yy = y * valid
    dice = 1 - (2 * (p * yy).sum() + eps) / (p.sum() + yy.sum() + eps)

    return bce + dice


def conf(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    return (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()      # tp, fp, fn


def prf(tp, fp, fn):
    e = 1e-9; p = tp / (tp + fp + e); r = tp / (tp + fn + e)
    return dict(iou=tp / (tp + fp + fn + e), dice=2 * tp / (2 * tp + fp + fn + e),
                precision=p, recall=r, f1=2 * p * r / (p + r + e))


def ndvi_baseline(x, drop=0.2):
    """Classical NDVI differencing on a raw-reflectance patch (10,H,W): year1 ch 0-4, year2 ch 5-9 (B2,B3,B4,B8,B11)."""
    n1 = (x[3] - x[2]) / (x[3] + x[2] + 1e-6)
    n2 = (x[8] - x[7]) / (x[8] + x[7] + 1e-6)
    return ((n2 - n1) < -drop) & (n1 > 0.5)


@torch.no_grad()
def evaluate(model, dl, thr=0.5):
    model.eval(); tp = fp = fn = 0
    for x, y, v in dl:
        x, y, v = x.to(DEVICE), y.to(DEVICE), v.to(DEVICE).bool()
        pred = torch.sigmoid(model(x)).squeeze(1) > thr
        gt = y > 0.5
        tp += (pred & gt & v).sum().item(); fp += (pred & ~gt & v).sum().item(); fn += (~pred & gt & v).sum().item()
    return prf(tp, fp, fn)


def run_epoch(model, dl, opt=None):
    train = opt is not None
    model.train(train); tot = 0
    for x, y, v in dl:
        x, y, v = x.to(DEVICE), y.to(DEVICE), v.to(DEVICE)
        with torch.set_grad_enabled(train):
            loss = masked_loss(model(x), y, v)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * x.size(0)
    return tot / len(dl.dataset)


def fit(model, train_dl, val_dl, epochs, lr, ckpt, patience=10):
    """
    Train with Adam + CosineAnnealingLR.
    patience increased to 10 (was 6) to avoid stopping too early with imbalanced data.
    """
    from torch.optim.lr_scheduler import CosineAnnealingLR
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.01)
    best, bad, hist = -1, 0, []
    for ep in range(1, epochs + 1):
        tr = run_epoch(model, train_dl, opt); va = run_epoch(model, val_dl); m = evaluate(model, val_dl)
        scheduler.step()
        hist.append(dict(epoch=ep, train_loss=tr, val_loss=va, **m))
        print(f"ep{ep:02d} train {tr:.3f} val {va:.3f} | IoU {m['iou']:.3f} Dice {m['dice']:.3f} "
              f"P {m['precision']:.3f} R {m['recall']:.3f}", flush=True)
        if m['iou'] > best:
            best, bad = m['iou'], 0; torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
            if bad >= patience:
                print('Early stopping'); break
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return hist


def set_encoder_trainable(model, flag):
    for p in model.encoder.parameters():
        p.requires_grad = flag
