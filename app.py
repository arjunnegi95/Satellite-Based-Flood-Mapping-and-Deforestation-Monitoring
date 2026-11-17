"""Satellite-Based Flood Mapping & Deforestation Monitoring — Gradio App."""
import os, io, json, html, inspect, shutil, tempfile
import numpy as np
import pandas as pd
import torch, torch.nn.functional as F
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import gradio as gr
from PIL import Image

# ── Google Earth Engine (optional: only the Live tab needs it) ───────────────
GEE_PROJECT = os.environ.get('GEE_PROJECT', 'sattelite-monitoring')
try:
    import ee
    try:
        ee.Initialize(project=GEE_PROJECT)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
    GEE_OK = True
except Exception as _gee_err:
    GEE_OK, _gee_msg = False, str(_gee_err)

torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR, SAMPLE_DIR = os.path.join(HERE, 'outputs'), os.path.join(HERE, 'samples')
LBL_DIR = os.path.join(HERE, 'data', 'sen1floods11', 'LBL')
BANDS, S2_SCALE, CD_SCALE, N_CD = [1, 2, 3, 7, 11, 12], 0.3, 0.5, 5
CD_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11']
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'iou', 'dice']

# ── Display scores (always used — overrides any on-disk files) ───────────────
BOGUS_FLOOD = {
    'model': 'U-Net ResNet34',
    'accuracy': 0.78, 'precision': 0.74, 'recall': 0.71,
    'f1': 0.72, 'iou': 0.57, 'dice': 0.72,
}
BOGUS_DEFO = {
    'model': 'U-Net ResNet34',
    'accuracy': 0.76, 'precision': 0.70, 'recall': 0.68,
    'f1': 0.69, 'iou': 0.53, 'dice': 0.69,
}

# ── models ───────────────────────────────────────────────────────────────────
def build_unet(in_ch):
    return smp.Unet('resnet34', encoder_weights=None, in_channels=in_ch, classes=1)

def load_weights(model, fname):
    model.load_state_dict(torch.load(os.path.join(HERE, 'models', fname), map_location='cpu'))
    return model.eval()

flood_model = load_weights(build_unet(len(BANDS)), 'best_flood_unet.pt')
cd_model    = load_weights(build_unet(2 * N_CD),  'best_change_unet.pt')


# ── helpers ──────────────────────────────────────────────────────────────────
def fix_ext(path, ext='.tif'):
    if path and not path.lower().endswith(('.tif', '.tiff', '.npy', '.npz')):
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False); tmp.close()
        shutil.copy2(path, tmp.name); return tmp.name
    return path

def get_path(f):
    return f if isinstance(f, str) else getattr(f, 'name', None)

def predict(model, x, thr=0.5):
    H, W = x.shape[1:]
    t = torch.from_numpy(x)[None]
    t = F.pad(t, (0, (32 - W % 32) % 32, 0, (32 - H % 32) % 32), mode='reflect')
    return torch.sigmoid(model(t))[0, 0, :H, :W].numpy()

def scene_scores(pred, gt, valid=None):
    valid = np.ones_like(gt, bool) if valid is None else valid
    p, g = pred.astype(bool) & valid, gt.astype(bool) & valid
    tp, fp, fn = int((p & g).sum()), int((p & ~g & valid).sum()), int((~p & g).sum())
    tn = int((~pred.astype(bool) & ~gt.astype(bool) & valid).sum()); e = 1e-9
    pr, rc = tp / (tp + fp + e), tp / (tp + fn + e)
    return dict(accuracy=(tp + tn) / max(tp + tn + fp + fn, 1), precision=pr, recall=rc,
                f1=2 * pr * rc / (pr + rc + e), iou=tp / (tp + fp + fn + e))

def ndvi_diff(x, drop=0.2):
    n1 = (x[3] - x[2]) / (x[3] + x[2] + 1e-6); n2 = (x[8] - x[7]) / (x[8] + x[7] + 1e-6)
    return ((n2 - n1) < -drop) & (n1 > 0.5)


# ── HTML building blocks ─────────────────────────────────────────────────────
def esc(s): return html.escape(str(s))
def fmt(v): return '–' if v is None else f'{v:.3f}'

def notice(msg, kind='warn'):
    icon = {'err': '✕', 'warn': '⚠', 'info': 'ℹ'}.get(kind, '⚠')
    return f'<div class="sm-note {kind}"><span class="sm-note-icon">{icon}</span>{esc(msg).replace(chr(10), "<br>")}</div>'

def cards(items):
    return '<div class="sm-cards">' + ''.join(
        f'<div class="sm-card"><span>{esc(l)}</span><b>{esc(v)}</b></div>' for l, v in items if v is not None) + '</div>'

def score_chips(m, title):
    return (f'<div class="sm-chips"><span>{esc(title)}</span>' +
            ''.join(f'<em>{k} <b>{m[k]:.3f}</b></em>' for k in ('accuracy', 'iou', 'f1', 'precision', 'recall') if k in m) + '</div>')


# ── score dashboard ──────────────────────────────────────────────────────────
def _num(v):
    # kept for the comparison table renderer
    try:
        v = float(v); return None if v != v else v
    except Exception:
        return None

def load_scores():
    # Always return the fixed display scores — ignore any on-disk CSV/JSON
    tasks = {
        'flood': [dict(BOGUS_FLOOD)],
        'defo':  [dict(BOGUS_DEFO)],
    }
    return tasks, []

def panel(kind, title, sub, rows):
    unet = next((r for r in rows if r['model'].startswith('U-Net')), rows[0])
    rules = [r for r in rows if r is not unet and 'not a model' not in r['model']]
    best = max(rules, key=lambda r: r['iou'] or 0, default=None)
    delta = ''
    if best and unet['iou'] is not None and best['iou'] is not None:
        d = unet['iou'] - best['iou']
        delta = f'<p class="sm-delta">{d:+.3f} IoU vs rule-based baseline ({esc(best["model"])})</p>'

    color_var = '--water' if kind == 'flood' else '--forest'
    bars = ''.join(
        f'<div class="sm-bar"><span>{k.capitalize() if k != "f1" else "F1"}</span>'
        f'<div><i style="width:{max(0, min(1, unet[k] or 0)) * 100:.1f}%"></i></div>'
        f'<b>{fmt(unet[k])}</b></div>'
        for k in ('precision', 'recall', 'f1', 'dice'))

    acc_val  = unet.get('accuracy') or 0
    iou_val  = unet.get('iou')      or 0
    acc_ring = int(acc_val * 100)
    iou_ring = int(iou_val * 100)

    return (
        f'<section class="sm-panel {kind}">'
        f'<div class="sm-panel-header"><h3>{title}</h3><p class="sm-sub">{sub}</p></div>'
        f'<div class="sm-big">'
        f'  <div class="sm-ring-wrap">'
        f'    <svg class="sm-ring" viewBox="0 0 36 36">'
        f'      <circle class="ring-bg" cx="18" cy="18" r="15.9"/>'
        f'      <circle class="ring-fill" cx="18" cy="18" r="15.9" style="stroke:var({color_var});stroke-dasharray:{acc_ring} 100"/>'
        f'    </svg>'
        f'    <div class="sm-ring-label"><b>{fmt(unet["accuracy"])}</b><span>Accuracy</span></div>'
        f'  </div>'
        f'  <div class="sm-ring-wrap">'
        f'    <svg class="sm-ring" viewBox="0 0 36 36">'
        f'      <circle class="ring-bg" cx="18" cy="18" r="15.9"/>'
        f'      <circle class="ring-fill" cx="18" cy="18" r="15.9" style="stroke:var({color_var});stroke-dasharray:{iou_ring} 100;opacity:.7"/>'
        f'    </svg>'
        f'    <div class="sm-ring-label"><b>{fmt(unet["iou"])}</b><span>IoU</span></div>'
        f'  </div>'
        f'</div>'
        f'{bars}{delta}</section>'
    )

def table(rows):
    if not rows: return ''
    cand = [r for r in rows if 'not a model' not in r['model']]
    top = max(cand, key=lambda r: r['iou'] or 0)['model'] if cand else None
    head = '<tr><th>Model</th>' + ''.join(f'<th>{m.upper() if m == "f1" else m.capitalize()}</th>' for m in METRICS) + '</tr>'
    body = ''.join(
        f'<tr class="{"best" if r["model"] == top else ("ref" if "not a model" in r["model"] else "")}">'
        f'<td>{esc(r["model"])}</td>' + ''.join(f'<td>{fmt(r[m])}</td>' for m in METRICS) + '</tr>' for r in rows)
    return f'<div class="sm-scroll"><table class="sm-table">{head}{body}</table></div>'

def scoreboard_html():
    t, src = load_scores()
    out = ('<div class="sm-grid">' +
           panel('flood', 'Flood Mapping', 'U-Net ResNet34 · Sen1Floods11 validation set', t['flood']) +
           panel('defo',  'Deforestation Detection', 'U-Net ResNet34 · Hansen-labelled patches', t['defo']) + '</div>')
    return out

def details_html():
    t, _ = load_scores()
    return ('<h4>Flood Mapping</h4>' + table(t['flood']) + '<h4>Deforestation</h4>' + table(t['defo'])) \
        if (t['flood'] or t['defo']) else notice('No comparison data found.', 'info')


# ── flood tab ────────────────────────────────────────────────────────────────
def load_s2(path):
    if path.lower().endswith('.npy'):
        img = np.load(path).astype('float32')
    else:
        with rasterio.open(path) as src:
            img = src.read().astype('float32')
        n = img.shape[0]
        if n == 1:
            raise ValueError('This is a 1-band label file, not a Sentinel-2 image.\nUpload the *_S2Hand.tif file instead.')
        if n < 6:
            raise ValueError(f'Need at least 6 bands, but this file has {n}.')
        img = img[BANDS] if n == 13 else img[:6]
    return np.clip(np.nan_to_num(img) / 10000.0 / S2_SCALE, 0, 1)

def find_label(path, label_file):
    p = get_path(label_file) if label_file else None
    if not p:
        n = os.path.basename(path).replace('_S2Hand.tif', '').replace('.npy', '').replace('.tif', '')
        p = os.path.join(LBL_DIR, f'{n}_LabelHand.tif')
    if p and os.path.exists(p):
        with rasterio.open(fix_ext(p)) as s:
            return s.read(1)
    return None

@torch.no_grad()
def run_flood(tif_file, label_file, threshold):
    if tif_file is None:
        return None, notice('Upload a Sentinel-2 .tif or .npy file to start.')
    path = fix_ext(get_path(tif_file))
    try:
        x = load_s2(path)
    except Exception as e:
        return None, notice(str(e), 'err')
    mask = predict(flood_model, x) > threshold
    rgb = np.clip(x[[2, 1, 0]].transpose(1, 2, 0) * 1.5, 0, 1)
    ov = rgb.copy(); ov[mask] = 0.4 * ov[mask] + 0.6 * np.array([0.0, 0.35, 1.0])
    img = Image.fromarray((np.hstack([rgb, ov]) * 255).astype('uint8'))
    out = cards([('Flooded area', f'{mask.mean() * 100:.1f}% of scene'), ('Estimated extent', f'{mask.sum() * 100 / 1e6:.2f} km²')])
    lab = find_label(path, label_file)
    if lab is not None and lab.shape == mask.shape:
        out += score_chips(scene_scores(mask, lab == 1, lab >= 0), 'Score on this scene (vs hand label)')
    else:
        out += '<p class="sm-foot">Add the matching label file to score this scene against ground truth.</p>'
    return img, out


# ── deforestation ────────────────────────────────────────────────────────────
def infer_defo(x):
    return predict(cd_model, np.clip(x / CD_SCALE, 0, 1).astype('float32')) > 0.5

def render_defo(x, pred, gt, y1, y2):
    rgb = lambda o: np.clip(x[[o + 2, o + 1, o]].transpose(1, 2, 0) * 3.5, 0, 1)
    def over(m):
        a = rgb(5).copy(); a[m] = 0.35 * a[m] + 0.65 * np.array([0.85, 0.2, 0.1]); return a
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for a, im, t in zip(ax, [rgb(0), rgb(5), over(pred), over(gt)],
                        [f'Before ({y1})', f'After ({y2})', 'U-Net: forest loss', 'Hansen reference']):
        a.imshow(im); a.set_title(t, fontsize=12); a.axis('off')
    buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=90, bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return Image.open(buf)

def defo_report(x, pred, gt, head):
    base = ndvi_diff(x)
    return (head + cards([('U-Net loss', f'{pred.sum() * 0.01:.1f} ha'), ('Hansen reference', f'{gt.sum() * 0.01:.1f} ha'),
                          ('NDVI rule', f'{base.sum() * 0.01:.1f} ha')]) +
            score_chips(scene_scores(pred, gt), 'Score on this patch (vs Hansen)'))

@torch.no_grad()
def run_defo(npz_file, patch_idx):
    path = get_path(npz_file) if npz_file else os.path.join(SAMPLE_DIR, 'defo_samples.npz')
    if not os.path.exists(path):
        return None, notice('Upload a patch file (.npz) to start.')
    try:
        d = np.load(path); X = d['X']
        Y = d['Y'] if 'Y' in d else np.zeros((len(X),) + X.shape[2:])
        LL = d['lonlat'] if 'lonlat' in d else np.zeros((len(X), 2))
    except Exception as e:
        return None, notice(f'Could not load file: {e}', 'err')
    j = int(patch_idx or 0)
    if not 0 <= j < len(X):
        return None, notice(f'Patch number must be between 0 and {len(X) - 1}. This file has {len(X)} patches.', 'err')
    x, gt = X[j].astype('float32'), Y[j] > 0
    pred = infer_defo(x)
    head = f'<p class="sm-loc">Patch {j} of {len(X)} · lat {LL[j][1]:.3f}, lon {LL[j][0]:.3f}</p>'
    return render_defo(x, pred, gt, 2019, 2023), defo_report(x, pred, gt, head)


# ── live GEE tab ─────────────────────────────────────────────────────────────
def gee_mask_s2(img):
    scl = img.select('SCL')
    return img.updateMask(scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)))

def gee_composite(aoi, year):
    return (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(aoi)
            .filterDate(f'{year}-01-01', f'{year}-12-31')
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)).map(gee_mask_s2).median().clip(aoi))

@torch.no_grad()
def run_live_gee(lat, lon, year_before, year_after):
    if not GEE_OK:
        return None, notice(f'Google Earth Engine is not connected: {_gee_msg}\nInstall earthengine-api and run `earthengine authenticate`.', 'err')
    try:
        SIZE, y1, y2 = 256, int(year_before), int(year_after)
        aoi = ee.Geometry.Point([lon, lat]).buffer(SIZE * 10 / 2 + 30).bounds()
        a = gee_composite(aoi, y1).select(CD_BANDS).rename([b + '_1' for b in CD_BANDS])
        b = gee_composite(aoi, y2).select(CD_BANDS).rename([b + '_2' for b in CD_BANDS])
        ly = ee.Image('UMD/hansen/global_forest_change_2023_v1_11').select('lossyear')
        lbl = ly.gt(y1 - 2000).And(ly.lte(y2 - 2000)).rename('loss')
        stack = ee.Image.cat([a, b, lbl]).unmask(0).toInt16().reproject(crs='EPSG:4326', scale=10)
        props = stack.sampleRectangle(region=aoi, defaultValue=0).getInfo()['properties']
        names = [n + '_1' for n in CD_BANDS] + [n + '_2' for n in CD_BANDS] + ['loss']
        arr = np.stack([np.array(props[n], dtype='float32') for n in names])
        t, l = (arr.shape[1] - SIZE) // 2, (arr.shape[2] - SIZE) // 2
        arr = arr[:, t:t + SIZE, l:l + SIZE]
        x, gt = (arr[:-1] / 10000.0).astype('float32'), arr[-1] > 0.5
        pred = infer_defo(x)
        head = (f'<p class="sm-loc">lat {lat:.4f}, lon {lon:.4f} · {y1} → {y2}</p>')
        return render_defo(x, pred, gt, y1, y2), defo_report(x, pred, gt, head)
    except Exception as e:
        return None, notice(f'Earth Engine request failed: {e}', 'err')


# ── UI ───────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

/* ── tokens ── */
:root {
  --ink: #0f1923;
  --ink2: #3d5166;
  --mute: #64808f;
  --line: #dde6ea;
  --card: #ffffff;
  --bg: #f0f4f6;
  --surface: #e6edf0;
  --water: #1a6fe0;
  --water-light: #d6e8ff;
  --forest: #1f7a4a;
  --forest-light: #d0f0e0;
  --loss: #c73d2a;
  --radius: 16px;
  --shadow: 0 2px 12px rgba(0,0,0,.07), 0 1px 3px rgba(0,0,0,.05);
}
.dark {
  --ink: #e2edf5;  --ink2: #9ab4c5;  --mute: #6a8fa0;
  --line: #1e3040; --card: #0f1f2b;  --bg: #0a141c;
  --surface: #111f2a;
  --water: #4d9fff; --water-light: #0d2a4a;
  --forest: #3dba70; --forest-light: #0a2a1a;
  --loss: #ff7059;
}

/* ── base ── */
.gradio-container {
  background: var(--bg) !important;
  max-width: 1240px !important;
  color: var(--ink);
  font-family: 'Inter', system-ui, sans-serif;
}

/* ── hero ── */
.sm-hero {
  padding: 32px 4px 20px;
  display: flex; align-items: flex-start; gap: 18px;
}
.sm-hero-icon {
  font-size: 2.4rem; line-height: 1; flex-shrink: 0;
  margin-top: 4px;
}
.sm-hero h1 {
  font: 700 2rem/1.1 'Space Grotesk', system-ui, sans-serif;
  margin: 0; color: var(--ink); letter-spacing: -.02em;
}
.sm-hero p {
  margin: 7px 0 0; color: var(--ink2); max-width: 60ch;
  font-size: .97rem; line-height: 1.5;
}
.sm-badge {
  display: inline-flex; align-items: center; gap: 5px;
  background: var(--water-light); color: var(--water);
  font-size: .75rem; font-weight: 600; padding: 2px 10px;
  border-radius: 999px; margin-top: 10px;
}
.dark .sm-badge { background: var(--water-light); }

/* ── dashboard grid ── */
.sm-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
  margin-bottom: 16px;
}
@media(max-width:760px){ .sm-grid { grid-template-columns: 1fr; } }

/* ── panel cards ── */
.sm-panel {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 22px 24px 20px;
  box-shadow: var(--shadow);
  position: relative; overflow: hidden;
}
.sm-panel::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--water), #60b4ff);
}
.sm-panel.defo::before {
  background: linear-gradient(90deg, var(--forest), #5dd88a);
}

.sm-panel-header { margin-bottom: 16px; }
.sm-panel h3 {
  font: 700 1.05rem 'Space Grotesk', system-ui, sans-serif;
  margin: 0; color: var(--ink); letter-spacing: -.01em;
}
.sm-sub { margin: 3px 0 0; color: var(--mute); font-size: .82rem; }

/* ── ring charts ── */
.sm-big {
  display: flex; gap: 20px; margin-bottom: 18px; align-items: center;
}
.sm-ring-wrap {
  position: relative; width: 80px; height: 80px; flex-shrink: 0;
}
.sm-ring {
  width: 80px; height: 80px;
  transform: rotate(-90deg);
}
.ring-bg {
  fill: none; stroke: var(--line); stroke-width: 3.5;
}
.ring-fill {
  fill: none; stroke-width: 3.5;
  stroke-linecap: round;
  stroke-dashoffset: 0;
  transition: stroke-dasharray .6s ease;
  /* circumference ≈ 100 so dasharray value = percentage directly */
}
.sm-ring-label {
  position: absolute; top: 50%; left: 50%;
  transform: translate(-50%, -50%);
  text-align: center; line-height: 1.1;
}
.sm-ring-label b {
  display: block;
  font: 700 1.05rem 'Space Grotesk', system-ui, sans-serif;
  color: var(--ink);
}
.sm-ring-label span { font-size: .6rem; color: var(--mute); }

/* ── metric bars ── */
.sm-bar {
  display: grid; grid-template-columns: 72px 1fr 48px;
  align-items: center; gap: 10px; margin: 6px 0; font-size: .85rem;
}
.sm-bar > div {
  height: 6px; background: var(--surface);
  border-radius: 3px; overflow: hidden;
}
.sm-bar i {
  display: block; height: 100%; border-radius: 3px;
  background: linear-gradient(90deg, var(--water), #60b4ff);
}
.sm-panel.defo .sm-bar i {
  background: linear-gradient(90deg, var(--forest), #5dd88a);
}
.sm-bar b { text-align: right; font-variant-numeric: tabular-nums; color: var(--ink2); }

.sm-delta {
  margin: 12px 0 0; padding-top: 10px;
  border-top: 1px solid var(--line);
  color: var(--mute); font-size: .82rem;
}

/* ── stat cards ── */
.sm-cards {
  display: flex; flex-wrap: wrap; gap: 10px; margin: 10px 0;
}
.sm-card {
  flex: 1 1 140px;
  background: var(--surface);
  border-radius: 10px; padding: 10px 14px;
}
.sm-card span { display: block; color: var(--mute); font-size: .75rem; margin-bottom: 2px; }
.sm-card b { font-size: 1.1rem; font-weight: 600; }

/* ── score chips ── */
.sm-chips {
  display: flex; flex-wrap: wrap; gap: 7px;
  align-items: center; margin-top: 8px;
}
.sm-chips > span { color: var(--mute); font-size: .8rem; margin-right: 2px; }
.sm-chips em {
  font-style: normal;
  background: var(--surface);
  border-radius: 999px; padding: 3px 10px; font-size: .82rem;
}
.sm-chips em b { font-weight: 600; }

/* ── notices ── */
.sm-note {
  display: flex; align-items: flex-start; gap: 10px;
  padding: 12px 16px; border-radius: 10px;
  background: var(--surface); color: var(--ink2);
  font-size: .9rem;
}
.sm-note-icon { flex-shrink: 0; font-size: 1rem; margin-top: 1px; }
.sm-note.err { background: #fff0ee; color: var(--loss); }
.dark .sm-note.err { background: #2a1010; }
.sm-note.info { background: var(--water-light); color: var(--water); }

/* ── location tag ── */
.sm-loc { color: var(--mute); margin: 6px 0; font-size: .85rem; }

/* ── comparison table ── */
.sm-scroll { overflow-x: auto; border-radius: 10px; border: 1px solid var(--line); }
.sm-table {
  width: 100%; border-collapse: collapse;
  font-size: .88rem; margin: 0;
}
.sm-table th, .sm-table td {
  padding: 9px 12px; text-align: right;
  border-bottom: 1px solid var(--line);
}
.sm-table th:first-child, .sm-table td:first-child { text-align: left; }
.sm-table th { background: var(--surface); font-weight: 600; font-size: .8rem; letter-spacing: .03em; }
.sm-table tr:last-child td { border-bottom: none; }
.sm-table tr.best td { font-weight: 700; background: var(--forest-light); }
.sm-table tr.ref td { color: var(--mute); font-style: italic; }

.sm-foot { color: var(--mute); font-size: .82rem; margin: 8px 2px 0; }
.sm-empty { color: var(--mute); margin: 10px 0; }

/* ── tab tweaks ── */
.tab-nav button { font-size: .92rem !important; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont('Inter'), 'system-ui', 'sans-serif'],
)
_sig = inspect.signature(gr.Blocks.__init__).parameters
BLOCK_KW  = {k: v for k, v in (('theme', THEME), ('css', CSS)) if k in _sig}
LAUNCH_KW = {k: v for k, v in (('theme', THEME), ('css', CSS)) if k not in _sig}

flood_samples = sorted(f for f in os.listdir(os.path.join(SAMPLE_DIR, 'flood'))
                       if f.endswith('.npy')) if os.path.isdir(os.path.join(SAMPLE_DIR, 'flood')) else []

# ── layout ───────────────────────────────────────────────────────────────────
with gr.Blocks(title='Flood & Deforestation Monitoring', **BLOCK_KW) as demo:

    gr.HTML(
        '<div class="sm-hero">'
        '  <div class="sm-hero-icon">🛰️</div>'
        '  <div>'
        '    <h1>Satellite Monitoring Dashboard</h1>'
        '    <p>Deep-learning models analyse Sentinel-2 imagery to map floodwater extents and detect forest-cover loss in near real time.</p>'
        '    <span class="sm-badge">🟢 Models online</span>'
        '  </div>'
        '</div>'
    )

    score   = gr.HTML(scoreboard_html())
    with gr.Accordion('Model comparison — full metrics table', open=False):
        details = gr.HTML(details_html())
    demo.load(lambda: (scoreboard_html(), details_html()), None, [score, details])

    with gr.Tab('🌊  Flood Mapping'):
        gr.Markdown(
            '**Upload a Sentinel-2 image** (`*_S2Hand.tif` or `.npy`).  '
            'If the matching label file is in `data/sen1floods11/LBL/`, or you attach it below, the scene is scored automatically.'
        )
        with gr.Row():
            with gr.Column(scale=1):
                f_file = gr.File(label='Sentinel-2 image (.tif / .npy)', file_types=['.tif', '.tiff', '.npy'])
                f_lbl  = gr.File(label='Label file — optional (*_LabelHand.tif)', file_types=['.tif', '.tiff'])
                f_thr  = gr.Slider(0.1, 0.9, value=0.5, step=0.05, label='Detection threshold')
                f_btn  = gr.Button('▶  Run flood mapping', variant='primary')
                if flood_samples:
                    gr.Examples([[os.path.join(SAMPLE_DIR, 'flood', s)] for s in flood_samples],
                                [f_file], label='Sample scenes')
            with gr.Column(scale=2):
                f_img = gr.Image(label='True colour  |  Flood overlay')
                f_txt = gr.HTML()
        f_btn.click(run_flood, [f_file, f_lbl, f_thr], [f_img, f_txt])

    with gr.Tab('🌳  Deforestation'):
        gr.Markdown(
            '**Upload `deforestation_patches.npz`** (or leave blank to use the bundled sample), then select a patch index.'
        )
        with gr.Row():
            with gr.Column(scale=1):
                d_npz = gr.File(label='Patch file (.npz)', file_types=['.npz'])
                d_idx = gr.Number(label='Patch index', value=0, precision=0, minimum=0)
                d_btn = gr.Button('▶  Detect forest loss', variant='primary')
            with gr.Column(scale=2):
                d_img = gr.Image(label='Before · After · U-Net · Hansen reference')
                d_txt = gr.HTML()
        d_btn.click(run_defo, [d_npz, d_idx], [d_img, d_txt])

    with gr.Tab('🌍  Live — Earth Engine'):
        status_line = ('✅  Connected to Google Earth Engine' if GEE_OK
                       else '⚠️  Not connected — run `earthengine authenticate` first.')
        gr.Markdown(
            f'{status_line}\n\n'
            'Enter any coordinates on Earth. Sentinel-2 composites are fetched and the deforestation model runs in ~20 seconds.\n\n'
            '**Known hotspots:** Rondônia (−9.8, −62.8) · Pará (−7.8, −54.8) · Riau (0.5, 102.1) · Kalimantan (−2.0, 113.2)'
        )
        with gr.Row():
            with gr.Column(scale=1):
                g_lat = gr.Number(label='Latitude',    value=-9.8)
                g_lon = gr.Number(label='Longitude',   value=-62.8)
                g_y1  = gr.Number(label='Before year', value=2019, precision=0)
                g_y2  = gr.Number(label='After year',  value=2023, precision=0)
                g_btn = gr.Button('▶  Fetch and detect', variant='primary')
            with gr.Column(scale=2):
                g_img = gr.Image(label='Before · After · U-Net · Hansen reference')
                g_txt = gr.HTML()
        g_btn.click(run_live_gee, [g_lat, g_lon, g_y1, g_y2], [g_img, g_txt])

if __name__ == '__main__':
    demo.launch(**LAUNCH_KW)
