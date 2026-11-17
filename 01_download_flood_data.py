"""Download the Sen1Floods11 hand-labeled chips from the public Google Cloud bucket (no login needed)."""
import os, requests
import concurrent.futures as cf
from common import FLOOD_DIR

BUCKET = 'sen1floods11'
PREFIXES = {'S2': 'v1.1/data/flood_events/HandLabeled/S2Hand/',
            'LBL': 'v1.1/data/flood_events/HandLabeled/LabelHand/'}


def list_objects(prefix):
    names, token = [], None
    while True:
        params = {'prefix': prefix, 'maxResults': 1000}
        if token:
            params['pageToken'] = token
        r = requests.get(f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o', params=params, timeout=60)
        r.raise_for_status()
        j = r.json()
        names += [it['name'] for it in j.get('items', []) if not it['name'].endswith('/')]
        token = j.get('nextPageToken')
        if not token:
            return names


def download(job):
    name, dest = job
    if os.path.exists(dest):
        return
    r = requests.get(f'https://storage.googleapis.com/{BUCKET}/{name}', stream=True, timeout=120)
    r.raise_for_status()
    with open(dest + '.part', 'wb') as f:
        for chunk in r.iter_content(1 << 20):
            f.write(chunk)
    os.replace(dest + '.part', dest)


if __name__ == '__main__':
    jobs = []
    for sub, prefix in PREFIXES.items():
        os.makedirs(os.path.join(FLOOD_DIR, sub), exist_ok=True)
        for name in list_objects(prefix):
            jobs.append((name, os.path.join(FLOOD_DIR, sub, os.path.basename(name))))
    print(len(jobs), 'files to fetch (already-downloaded files are skipped)')
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        for i, _ in enumerate(ex.map(download, jobs), 1):
            if i % 50 == 0:
                print(f'{i}/{len(jobs)}', flush=True)
    print('Done ->', FLOOD_DIR)
