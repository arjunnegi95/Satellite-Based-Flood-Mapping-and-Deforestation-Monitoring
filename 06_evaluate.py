"""Evaluate the SAVED models on their held-out validation data (no training).
Usage:
    python 06_evaluate.py                 # both models, threshold 0.5
    python 06_evaluate.py --only flood    # or: --only defo
    python 06_evaluate.py --sweep         # also show scores at thresholds 0.3 ... 0.7
Writes: outputs/flood_metrics.json, outputs/deforestation_results.csv, outputs/evaluation_report.txt
"""
import argparse, json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from common import *


def load_model(in_ch, ckpt):
    m = build_unet(in_ch, pretrained=False).to(DEVICE)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    return m.eval()


@torch.no_grad()
def collect(model, dl):
    """Run the model once; keep probabilities, labels and valid-pixel masks as numpy arrays."""
    P, G, V = [], [], []
    for x, y, v in dl:
        p = torch.sigmoid(model(x.to(DEVICE))).squeeze(1).cpu().numpy()
        P.append(p.astype('float16')); G.append((y.numpy() > 0.5)); V.append((v.numpy() > 0.5))
    return np.concatenate(P), np.concatenate(G), np.concatenate(V)


def scores(P, G, V, thr):
    pred = P.astype('float32') > thr
    tp = int((pred & G & V).sum()); fp = int((pred & ~G & V).sum())
    fn = int((~pred & G & V).sum()); tn = int((~pred & ~G & V).sum())
    m = prf(tp, fp, fn)
    m = {k: float(v) for k, v in m.items()}
    m['accuracy'] = (tp + tn) / max(tp + tn + fp + fn, 1)       # pixel accuracy (inflated by the easy background)
    return m, dict(tp=tp, fp=fp, fn=fn, tn=tn)


def show(title, m, c, lines):
    lines += [f'\n=== {title} ===',
              f"IoU {m['iou']:.4f} | Dice {m['dice']:.4f} | F1 {m['f1']:.4f} | Precision {m['precision']:.4f} | "
              f"Recall {m['recall']:.4f} | Pixel accuracy {m['accuracy']:.4f}",
              f"Pixels: TP {c['tp']:,}  FP {c['fp']:,}  FN {c['fn']:,}  TN {c['tn']:,}"]


def sweep(P, G, V, lines):
    lines.append('threshold sweep:  thr   IoU    Dice   Prec   Recall')
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        m, _ = scores(P, G, V, t)
        lines.append(f"                  {t:.1f}  {m['iou']:.3f}  {m['dice']:.3f}  {m['precision']:.3f}  {m['recall']:.3f}")


def eval_flood(thr, do_sweep, lines):
    if not os.path.exists(CKPT_FLOOD):
        lines.append(f'\n[flood] skipped: {CKPT_FLOOD} not found'); return
    _, va_ids = split_flood_ids()
    dl = DataLoader(FloodDS(va_ids, False), batch_size=4, shuffle=False, num_workers=NUM_WORKERS)
    model = load_model(len(BANDS), CKPT_FLOOD)
    P, G, V = collect(model, dl)
    m, c = scores(P, G, V, thr)
    show(f'FLOOD U-Net on {len(va_ids)} validation chips (thr {thr})', m, c, lines)
    if do_sweep:
        sweep(P, G, V, lines)
    json.dump(m, open(os.path.join(OUT_DIR, 'flood_metrics.json'), 'w'), indent=2)


def eval_defo(thr, do_sweep, lines):
    if not (os.path.exists(CKPT_CD) and os.path.exists(NPZ)):
        lines.append(f'\n[deforestation] skipped: need {CKPT_CD} and {NPZ}'); return
    d = np.load(NPZ); X, Y = d['X'], d['Y']
    _, va_idx = split_defo_idx(len(X))
    dl = DataLoader(ArrDS(X[va_idx], Y[va_idx], False), batch_size=8, num_workers=NUM_WORKERS)
    model = load_model(2 * len(CD_BANDS), CKPT_CD)
    P, G, V = collect(model, dl)
    m, c = scores(P, G, V, thr)
    show(f'DEFORESTATION U-Net on {len(va_idx)} validation patches (thr {thr})', m, c, lines)
    if do_sweep:
        sweep(P, G, V, lines)

    # classical NDVI-differencing baseline on the same patches
    BP = np.stack([ndvi_baseline(X[i].astype('float32')) for i in va_idx]).astype('float16')
    bm, bc = scores(BP, G, V, 0.5)
    show('NDVI differencing baseline (same patches)', bm, bc, lines)
    df = pd.DataFrame({'U-Net (trained)': m, 'NDVI differencing (rule)': bm}).T.round(3)
    df.to_csv(os.path.join(OUT_DIR, 'deforestation_results.csv'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['flood', 'defo'])
    ap.add_argument('--thr', type=float, default=0.5)
    ap.add_argument('--sweep', action='store_true')
    a = ap.parse_args()
    lines = [f'Device: {DEVICE}']
    if a.only in (None, 'flood'):
        eval_flood(a.thr, a.sweep, lines)
    if a.only in (None, 'defo'):
        eval_defo(a.thr, a.sweep, lines)
    report = '\n'.join(lines)
    print(report)
    open(os.path.join(OUT_DIR, 'evaluation_report.txt'), 'w').write(report + '\n')
    print('\nSaved to', OUT_DIR)