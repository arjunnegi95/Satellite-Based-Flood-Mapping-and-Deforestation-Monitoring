"""
Train the deforestation U-Net on the saved patches.

Improvements over v1:
  - 13-channel input  : 10 raw bands + NDVI₁ + NDVI₂ + ΔNDVI  (cd_features in common.py)
  - pos_weight=20     : weighted BCE penalises missing deforestation pixels 20× more
  - WeightedRandomSampler : oversample patches that actually contain loss pixels
  - stronger augmentation : see ArrDS in common.py
  - more epochs       : 15 (frozen) + 35 (full fine-tune)   was 6 + 14
  - CosineAnnealingLR : smooth decay, implemented inside fit()
  - patience=10       : allows more room to improve before early stopping
"""
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from common import *

# ---- training schedule (lower if running on CPU without a GPU) ----
EPOCHS_P1 = 15    # frozen encoder  (was 6)
EPOCHS_P2 = 35    # full fine-tune  (was 14)

if __name__ == '__main__':
    d = np.load(NPZ)
    X, Y = d['X'], d['Y']
    tr_idx, va_idx = split_defo_idx(len(X))
    print(len(tr_idx), 'train patches,', len(va_idx), 'val patches '
          '| mean loss fraction', Y.mean().round(4), '| device', DEVICE)

    # ----------------------------------------------------------------
    # Weighted sampler: patches with ≥5 % loss pixels get 5× higher
    # sampling probability so the model sees more positive examples.
    # ----------------------------------------------------------------
    frac = Y[tr_idx].reshape(len(tr_idx), -1).mean(axis=1)
    sample_weights = np.where(frac >= 0.05, 5.0, 1.0)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dl = DataLoader(
        ArrDS(X[tr_idx], Y[tr_idx], train=True),
        batch_size=8,
        sampler=sampler,          # replaces shuffle=True
        drop_last=True,
        num_workers=NUM_WORKERS
    )
    val_dl = DataLoader(
        ArrDS(X[va_idx], Y[va_idx], train=False),
        batch_size=8,
        num_workers=NUM_WORKERS
    )

    # ----------------------------------------------------------------
    # 13-channel model: 10 raw bands + NDVI₁ + NDVI₂ + ΔNDVI
    # ----------------------------------------------------------------
    IN_CH = 13   # was: 2 * len(CD_BANDS) = 10
    model = build_unet(IN_CH).to(DEVICE)

    # Phase 1 — train only the decoder (encoder stays ImageNet-pretrained)
    print(f'\n=== Phase 1: decoder only, {EPOCHS_P1} epochs ===')
    set_encoder_trainable(model, False)
    fit(model, train_dl, val_dl, epochs=EPOCHS_P1, lr=1e-3, ckpt=CKPT_CD)

    # Phase 2 — full fine-tune at lower LR
    print(f'\n=== Phase 2: full fine-tune, {EPOCHS_P2} epochs ===')
    set_encoder_trainable(model, True)
    fit(model, train_dl, val_dl, epochs=EPOCHS_P2, lr=1e-4, ckpt=CKPT_CD)

    # ----------------------------------------------------------------
    # Threshold sweep on validation set
    # ----------------------------------------------------------------
    print('\n=== Threshold sweep on validation set ===')
    best_thr, best_iou = 0.5, 0.0
    for thr in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        m = evaluate(model, val_dl, thr=thr)
        print(f'  thr={thr:.2f}  IoU={m["iou"]:.4f}  F1={m["f1"]:.4f}  '
              f'P={m["precision"]:.4f}  R={m["recall"]:.4f}')
        if m['iou'] > best_iou:
            best_iou, best_thr = m['iou'], thr
    print(f'\n  >> Best threshold: {best_thr}  (IoU {best_iou:.4f})')

    # ----------------------------------------------------------------
    # Final evaluation: U-Net vs NDVI baseline
    # ----------------------------------------------------------------
    unet = evaluate(model, val_dl, thr=best_thr)
    tp = fp = fn = 0
    for i in va_idx:
        a, b, c = conf(ndvi_baseline(X[i].astype('float32')), Y[i])
        tp += a; fp += b; fn += c
    res = pd.DataFrame({
        f'U-Net 13-ch (thr={best_thr})': unet,
        'NDVI differencing (rule)': prf(tp, fp, fn)
    }).T.round(3)
    print('\n', res)
    res.to_csv(os.path.join(OUT_DIR, 'deforestation_results.csv'))
    print('Saved model ->', CKPT_CD)
