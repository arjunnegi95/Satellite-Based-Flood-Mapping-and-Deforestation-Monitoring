"""Train the flood U-Net (ResNet34): phase 1 frozen encoder, phase 2 fine-tuning."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from common import *

if __name__ == '__main__':
    tr_ids, va_ids = split_flood_ids()
    print(len(tr_ids), 'train chips,', len(va_ids), 'val chips | device:', DEVICE)
    train_dl = DataLoader(FloodDS(tr_ids, True), batch_size=16, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    val_dl = DataLoader(FloodDS(va_ids, False), batch_size=4, shuffle=False, num_workers=NUM_WORKERS)

    model = build_unet(len(BANDS)).to(DEVICE)
    set_encoder_trainable(model, False)
    h1 = fit(model, train_dl, val_dl, epochs=10, lr=1e-3, ckpt=CKPT_FLOOD)
    set_encoder_trainable(model, True)
    h2 = fit(model, train_dl, val_dl, epochs=20, lr=1e-4, ckpt=CKPT_FLOOD)

    final = evaluate(model, val_dl)
    print({k: round(v, 4) for k, v in final.items()})
    json.dump({k: float(v) for k, v in final.items()}, open(os.path.join(OUT_DIR, 'flood_metrics.json'), 'w'), indent=2)

    hist = h1 + h2
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot([h['train_loss'] for h in hist], label='train'); ax[0].plot([h['val_loss'] for h in hist], label='val')
    ax[0].set_title('Loss (BCE+Dice)'); ax[0].legend()
    ax[1].plot([h['iou'] for h in hist], label='IoU'); ax[1].plot([h['dice'] for h in hist], label='Dice')
    ax[1].set_title('Validation metrics'); ax[1].legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'training_curves.png'), dpi=150); plt.close()

    x, y, v = next(iter(val_dl))
    with torch.no_grad():
        pr = (torch.sigmoid(model(x.to(DEVICE))).squeeze(1) > 0.5).cpu()
    fig, ax = plt.subplots(3, 4, figsize=(12, 9))
    for i in range(min(4, len(x))):
        ax[0, i].imshow(x[i][[2, 1, 0]].permute(1, 2, 0).clamp(0, 1)); ax[0, i].set_title('RGB')
        ax[1, i].imshow(y[i], cmap='Blues'); ax[1, i].set_title('Ground truth')
        ax[2, i].imshow(pr[i], cmap='Blues'); ax[2, i].set_title('Prediction')
    for a in ax.ravel():
        a.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, 'predictions.png'), dpi=150); plt.close()
    print('Saved model ->', CKPT_FLOOD)
