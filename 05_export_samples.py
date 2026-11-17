"""Create the small samples/ folder used by app.py (and by Hugging Face): top-flooded validation chips + held-out deforestation patches."""
import rasterio
from common import *

if __name__ == '__main__':
    out_flood = os.path.join(SAMPLE_DIR, 'flood'); os.makedirs(out_flood, exist_ok=True)
    _, va_ids = split_flood_ids()
    fr = []
    for n in va_ids:
        with rasterio.open(os.path.join(FLOOD_DIR, 'LBL', f'{n}_LabelHand.tif')) as src:
            lab = src.read(1)
        fr.append(((lab == 1).sum() / max((lab >= 0).sum(), 1), n))
    fr.sort(reverse=True)
    for frac, n in fr[:5]:
        with rasterio.open(os.path.join(FLOOD_DIR, 'S2', f'{n}_S2Hand.tif')) as src:
            arr = src.read().astype('float32')[BANDS]
        np.save(os.path.join(out_flood, f'{n}.npy'), arr.astype('float16'))
        print(f'flood sample {n}: {frac:.0%} water')

    d = np.load(NPZ)
    _, va_idx = split_defo_idx(len(d['X']))
    val = np.sort(va_idx[:8])
    np.savez_compressed(os.path.join(SAMPLE_DIR, 'defo_samples.npz'), X=d['X'][val], Y=d['Y'][val], lonlat=d['lonlat'][val])
    print('Samples written to', SAMPLE_DIR)
