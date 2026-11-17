"""Compare ALL models on their held-out validation data and print the best one for each task.

Flood mapping       : U-Net (trained)  vs  NDWI rule  vs  MNDWI rule  (+ "predict no water" reference)
Deforestation       : U-Net (trained)  vs  NDVI-differencing rule (3 thresholds)  (+ "predict no loss" reference)

Usage:
    python 07_compare_models.py                       # rank by IoU (recommended)
    python 07_compare_models.py --rank-by f1          # or: iou, f1, dice, precision, recall, accuracy
    python 07_compare_models.py --thr 0.4             # decision threshold of the U-Nets (default 0.5)

Writes: outputs/model_comparison.csv, outputs/model_comparison.txt, outputs/best_models.json
Works with both the 10-channel and the 13-channel deforestation checkpoint (the channel count is read from the file).
"""
import argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import rasterio
from common import (BANDS, CD_BANDS, CD_SCALE, CKPT_CD, CKPT_FLOOD, DEVICE, FLOOD_DIR, NPZ, OUT_DIR,
                    build_unet, load_s2, split_defo_idx, split_flood_ids)
import os

REFERENCE = 'All-negative reference (not a model)'


# ----------------------------------------------------------------- pure numpy helpers
def metrics_from_counts(tp, fp, fn, tn):
    e = 1e-9
    prec, rec = tp / (tp + fp + e), tp / (tp + fn + e)
    return dict(accuracy=(tp + tn) / max(tp + tn + fp + fn, 1), precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec + e), iou=tp / (tp + fp + fn + e), dice=2 * tp / (2 * tp + fp + fn + e))


class Counter:
    def __init__(self):
        self.tp = self.fp = self.fn = self.tn = 0

    def add(self, pred, gt, valid):
        pred, gt, valid = pred.astype(bool), gt.astype(bool), valid.astype(bool)
        self.tp += int((pred & gt & valid).sum()); self.fp += int((pred & ~gt & valid).sum())
        self.fn += int((~pred & gt & valid).sum()); self.tn += int((~pred & ~gt & valid).sum())

    def row(self, name):
        m = metrics_from_counts(self.tp, self.fp, self.fn, self.tn)
        m.update(model=name, TP=self.tp, FP=self.fp, FN=self.fn, TN=self.tn)
        return m


def ndvi_rule(x, drop=0.2):
    """x: raw reflectance (10,H,W); year 1 = channels 0-4, year 2 = channels 5-9 (B2,B3,B4,B8,B11)."""
    n1 = (x[3] - x[2]) / (x[3] + x[2] + 1e-6)
    n2 = (x[8] - x[7]) / (x[8] + x[7] + 1e-6)
    return ((n2 - n1) < -drop) & (n1 > 0.5)


def water_index(green, other):
    """(Green - other) / (Green + other) > 0 : NDWI when other = NIR, MNDWI when other = SWIR."""
    return ((green - other) / (green + other + 1e-6)) > 0


def cd_features(x):
    """Raw (10,H,W) -> 13-channel input: 10 scaled bands + NDVI before, NDVI after, NDVI change."""
    b = np.clip(x / CD_SCALE, 0, 1)
    n1 = (x[3] - x[2]) / (x[3] + x[2] + 1e-6)
    n2 = (x[8] - x[7]) / (x[8] + x[7] + 1e-6)
    f1 = np.clip(n1, -1, 1) * 0.5 + 0.5
    f2 = np.clip(n2, -1, 1) * 0.5 + 0.5
    fd = np.clip(n2 - n1, -0.5, 0.5) + 0.5
    return np.concatenate([b, f1[None], f2[None], fd[None]]).astype('float32')


# ----------------------------------------------------------------- model helpers
def load_unet(ckpt):
    sd = torch.load(ckpt, map_location=DEVICE)
    in_ch = int(sd['encoder.conv1.weight'].shape[1])          # channel count stored in the checkpoint
    model = build_unet(in_ch, pretrained=False).to(DEVICE)
    model.load_state_dict(sd)
    return model.eval(), in_ch


@torch.no_grad()
def predict(model, x, thr):
    """x: float32 (C,H,W) already scaled. Pads to a multiple of 32, returns a boolean mask (H,W)."""
    H, W = x.shape[1:]
    t = torch.from_numpy(x)[None].to(DEVICE)
    ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode='reflect')
    p = torch.sigmoid(model(t))[0, 0, :H, :W].cpu().numpy()
    return p > thr


# ----------------------------------------------------------------- the two comparisons
def compare_flood(thr):
    rows = []
    if not os.path.isdir(os.path.join(FLOOD_DIR, 'S2')):
        print('[flood] skipped: dataset not found in', FLOOD_DIR); return rows
    _, va_ids = split_flood_ids()
    have_model = os.path.exists(CKPT_FLOOD)
    if have_model:
        model, in_ch = load_unet(CKPT_FLOOD)
    cu, cn, cm, c0 = Counter(), Counter(), Counter(), Counter()
    for n in va_ids:
        s2 = os.path.join(FLOOD_DIR, 'S2', f'{n}_S2Hand.tif')
        with rasterio.open(os.path.join(FLOOD_DIR, 'LBL', f'{n}_LabelHand.tif')) as src:
            lab = src.read(1)
        valid, gt = lab >= 0, lab == 1
        if have_model:
            cu.add(predict(model, load_s2(s2), thr), gt, valid)
        with rasterio.open(s2) as src:
            raw = np.nan_to_num(src.read().astype('float32')) / 10000.0
        if raw.shape[0] == 13:                                   # B3 = 2, B8 = 7, B11 = 11
            cn.add(water_index(raw[2], raw[7]), gt, valid)
            cm.add(water_index(raw[2], raw[11]), gt, valid)
        c0.add(np.zeros_like(gt), gt, valid)
    if have_model:
        rows.append(cu.row(f'U-Net ResNet34 ({in_ch} channels)'))
    else:
        print('[flood] U-Net skipped: checkpoint not found')
    rows += [cn.row('NDWI rule (green, NIR)'), cm.row('MNDWI rule (green, SWIR)'), c0.row(REFERENCE)]
    return rows


def compare_defo(thr):
    rows = []
    if not os.path.exists(NPZ):
        print('[deforestation] skipped: dataset not found:', NPZ); return rows
    d = np.load(NPZ); X, Y = d['X'], d['Y']
    _, va_idx = split_defo_idx(len(X))
    have_model = os.path.exists(CKPT_CD)
    if have_model:
        model, in_ch = load_unet(CKPT_CD)
        if in_ch not in (10, 13):
            raise SystemExit(f'Unexpected number of input channels in {CKPT_CD}: {in_ch}')
    cu = Counter(); rules = {0.1: Counter(), 0.2: Counter(), 0.3: Counter()}; c0 = Counter()
    for i in va_idx:
        x = X[i].astype('float32'); gt = Y[i] > 0; valid = np.ones_like(gt)
        if have_model:
            xin = cd_features(x) if in_ch == 13 else np.clip(x / CD_SCALE, 0, 1).astype('float32')
            cu.add(predict(model, xin, thr), gt, valid)
        for drop, c in rules.items():
            c.add(ndvi_rule(x, drop), gt, valid)
        c0.add(np.zeros_like(gt), gt, valid)
    if have_model:
        rows.append(cu.row(f'U-Net ResNet34 ({in_ch} channels)'))
    else:
        print('[deforestation] U-Net skipped: checkpoint not found')
    for drop, c in rules.items():
        rows.append(c.row(f'NDVI differencing rule (drop {drop})'))
    rows.append(c0.row(REFERENCE))
    return rows


# ----------------------------------------------------------------- reporting
def report(task, rows, rank_by, lines):
    if not rows:
        return None
    df = pd.DataFrame(rows).set_index('model')
    cols = ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice', 'TP', 'FP', 'FN', 'TN']
    df = df[cols]
    models = df.drop(index=REFERENCE, errors='ignore')                    # the reference is not a candidate
    best = models[rank_by].idxmax()
    lines.append(f'\n==================== {task.upper()} ====================')
    lines.append(f'(validation pixels: {int(df.iloc[0][["TP", "FP", "FN", "TN"]].sum()):,}, '
                 f'positive share: {(df.iloc[0]["TP"] + df.iloc[0]["FN"]) / df.iloc[0][["TP", "FP", "FN", "TN"]].sum():.2%})')
    show = df.sort_values(rank_by, ascending=False)
    lines.append(show.to_string(float_format=lambda v: f'{v:.4f}', formatters={c: '{:,}'.format for c in ['TP', 'FP', 'FN', 'TN']}))
    lines.append('')
    lines.append(f'>>> BEST MODEL by {rank_by.upper()}: {best}   ({rank_by} = {models.loc[best, rank_by]:.4f})')
    acc_best = models['accuracy'].idxmax(); prec_best = models['precision'].idxmax()
    lines.append(f'    highest accuracy : {acc_best} ({models.loc[acc_best, "accuracy"]:.4f})')
    lines.append(f'    highest precision: {prec_best} ({models.loc[prec_best, "precision"]:.4f})')
    ref_acc = df.loc[REFERENCE, 'accuracy'] if REFERENCE in df.index else None
    if ref_acc is not None:
        lines.append(f'    note: predicting "nothing" everywhere already gives accuracy {ref_acc:.4f}, which is why '
                     f'accuracy alone cannot pick the best model; IoU / F1 are used for the ranking.')
    df.insert(0, 'task', task)
    return df, best


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank-by', default='iou', choices=['iou', 'f1', 'dice', 'precision', 'recall', 'accuracy'])
    ap.add_argument('--thr', type=float, default=0.5)
    a = ap.parse_args()
    lines = [f'Device: {DEVICE} | U-Net threshold: {a.thr} | ranking metric: {a.rank_by}']
    frames, best = [], {}
    for task, fn in (('flood mapping', compare_flood), ('deforestation', compare_defo)):
        out = report(task, fn(a.thr), a.rank_by, lines)
        if out:
            frames.append(out[0]); best[task] = out[1]
    text = '\n'.join(lines)
    print(text)
    if frames:
        pd.concat(frames).to_csv(os.path.join(OUT_DIR, 'model_comparison.csv'))
        json.dump({'ranked_by': a.rank_by, 'best': best}, open(os.path.join(OUT_DIR, 'best_models.json'), 'w'), indent=2)
        open(os.path.join(OUT_DIR, 'model_comparison.txt'), 'w').write(text + '\n')
        print('\nSaved model_comparison.csv, model_comparison.txt and best_models.json to', OUT_DIR)
