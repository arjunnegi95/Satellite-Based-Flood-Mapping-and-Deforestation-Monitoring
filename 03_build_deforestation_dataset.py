"""Build the deforestation dataset ONCE from Google Earth Engine (Sentinel-2 before/after + Hansen loss labels).
Needs: `pip install earthengine-api`, a registered Earth Engine project (set GEE_PROJECT or edit common.py).
First run opens a browser window to log in.

Improvements over v1:
  - More patches   : N_POS 120→400, N_NEG 80→150
  - More regions   : added Congo Basin, Mato Grosso (Brazil), West Africa, Myanmar, Bolivia
  - Multi-year pairs: added 2016→2019 pair so each location can yield 2 training samples
  - Stricter cloud filter: CLOUDY_PIXEL_PERCENTAGE < 25 (was 40) for cleaner composites
"""
import os, time, random
import concurrent.futures as cf
import numpy as np
import ee
from common import NPZ, SEED, GEE_PROJECT, CD_BANDS

try:
    ee.Initialize(project=GEE_PROJECT)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT)

HANSEN = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')


def mask_s2(img):
    scl = img.select('SCL')
    ok = scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
    return img.updateMask(ok)


def sr_composite(aoi, year):
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(aoi)
           .filterDate(f'{year}-01-01', f'{year}-12-31')
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 25))   # stricter (was 40)
           .map(mask_s2))
    return col.median().clip(aoi)


def fetch_patch(lon, lat, y1=2019, y2=2023, size=256):
    aoi = ee.Geometry.Point([lon, lat]).buffer(size * 10 / 2 + 30).bounds()
    a = sr_composite(aoi, y1).select(CD_BANDS).rename([b + '_1' for b in CD_BANDS])
    b = sr_composite(aoi, y2).select(CD_BANDS).rename([b + '_2' for b in CD_BANDS])
    ly = HANSEN.select('lossyear')
    lbl = ly.gt(y1 - 2000).And(ly.lte(y2 - 2000)).rename('loss')
    stack = ee.Image.cat([a, b, lbl]).unmask(0).toInt16().reproject(crs='EPSG:4326', scale=10)
    props = stack.sampleRectangle(region=aoi, defaultValue=0).getInfo()['properties']
    names = [n + '_1' for n in CD_BANDS] + [n + '_2' for n in CD_BANDS] + ['loss']
    arr = np.stack([np.array(props[n], dtype='float32') for n in names])
    if arr.shape[1] < size or arr.shape[2] < size:
        return None
    t, l = (arr.shape[1] - size) // 2, (arr.shape[2] - size) // 2
    arr = arr[:, t:t + size, l:l + size]
    if (arr[3] > 0).mean() < 0.7 or (arr[8] > 0).mean() < 0.7:        # too many cloud-masked pixels
        return None
    return (arr[:-1] / 10000.0).astype('float16'), (arr[-1] > 0.5).astype('uint8')


# ---- expanded deforestation hotspots ----
# Format: [west, south, east, north]
REGIONS = {
    # Original 4 regions
    'Rondonia_BR':          [-63.5, -10.5, -62.0,  -9.0],
    'Para_BR':              [-55.5,  -8.5, -54.0,  -7.0],
    'Riau_ID':              [101.5,   0.0, 102.8,   1.0],
    'CentralKalimantan':    [112.5,  -2.5, 114.0,  -1.5],
    # New regions
    'MatoGrosso_BR':        [-55.0, -13.0, -53.0, -11.5],   # Brazil cerrado/Amazon edge
    'Amazonas_BR':          [-67.0,  -6.0, -65.5,  -4.5],   # western Amazon
    'Bolivia':              [-67.0, -17.0, -65.0, -15.0],   # Chiquitano dry forest
    'CongoDRC':             [ 24.0,   0.5,  26.0,   2.0],   # Congo Basin
    'GabonCongo':           [ 11.0,  -1.5,  13.0,   0.5],   # Central African forest
    'WestAfrica':           [ -6.0,   5.0,  -4.0,   7.0],   # Côte d'Ivoire / Ghana
    'SumatraID':            [104.0,  -3.5, 105.5,  -2.0],   # Sumatra peatlands
    'PapuaNG':              [141.0,  -6.5, 143.0,  -5.0],   # Papua New Guinea
    'Myanmar':              [ 96.0,  18.0,  98.0,  20.0],   # Myanmar uplands
    'CambodiaLaos':         [104.0,  13.0, 106.0,  15.0],   # Mekong region
}

# ---- multi-year pairs: more samples per location ----
YEAR_PAIRS = [
    (2016, 2019),   # pre-2019 clearing
    (2019, 2023),   # original pair
]


def try_fetch(args):
    lon, lat, y1, y2 = args
    for _ in range(2):
        try:
            return (lon, lat, y1, y2), fetch_patch(lon, lat, y1, y2)
        except Exception:
            time.sleep(2)
    return (lon, lat, y1, y2), None


if __name__ == '__main__':
    if os.path.exists(NPZ):
        raise SystemExit(f'Dataset already exists: {NPZ} (delete it to rebuild)')
    random.seed(SEED)

    # Increased from 120/80 to 400/150
    N_POS, N_NEG = 400, 150

    # Build point list: each region × each year pair × 200 random points
    pts = []
    for y1, y2 in YEAR_PAIRS:
        for b in REGIONS.values():
            pts += [(random.uniform(b[0], b[2]), random.uniform(b[1], b[3]), y1, y2)
                    for _ in range(200)]
    random.shuffle(pts)

    Xl, Yl, LL = [], [], []
    npos = nneg = 0

    with cf.ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(try_fetch, p) for p in pts]
        for k, f in enumerate(cf.as_completed(futs), 1):
            pt, out = f.result()
            if out is not None:
                x, y = out
                frac = y.mean()
                if frac >= 0.02 and npos < N_POS:
                    Xl.append(x); Yl.append(y); LL.append(pt); npos += 1
                elif frac < 0.002 and nneg < N_NEG:
                    Xl.append(x); Yl.append(y); LL.append(pt); nneg += 1
            if k % 20 == 0:
                print(f'{k} tried | kept {npos} loss-patches + {nneg} no-loss patches', flush=True)
            if npos >= N_POS and nneg >= N_NEG:
                for g in futs:
                    g.cancel()
                break

    print('Final:', npos, 'positive,', nneg, 'negative patches')
    np.savez_compressed(NPZ, X=np.stack(Xl), Y=np.stack(Yl),
                        lonlat=np.array([[p[0], p[1]] for p in LL]),
                        years=np.array([[p[2], p[3]] for p in LL]))
    print('Saved ->', NPZ)
