from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import urllib.request
import zipfile
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

REVISION = '93e82ae507c36b3f2c8227eade8b792fcbefcca6'
URL = f'https://codeload.github.com/zae-bayern/elpv-dataset/zip/{REVISION}'
PREFIX = f'elpv-dataset-{REVISION}/src/elpv_dataset/data/'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(folder, archive=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'manifest.json').exists():
        raise ValueError('Data already prepared; use a fresh directory')
    archive = Path(archive) if archive else folder / 'upstream.zip'
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.startswith(PREFIX) or name.endswith('/'):
                continue
            rel = PurePosixPath(name[len(PREFIX):])
            if '..' in rel.parts or rel.is_absolute():
                raise ValueError('Unsafe archive path')
            target = folder / str(rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
        license_text = z.read(f'elpv-dataset-{REVISION}/LICENSE').decode()
    (folder / 'DATA_LICENSE.txt').write_text(license_text, encoding='utf-8')
    records = read_labels(folder)
    checksums = {r['path']: digest(folder / r['path']) for r in records}
    manifest = {'upstream': URL, 'revision': REVISION, 'archive_sha256': digest(archive),
                'labels_sha256': digest(folder / 'labels.csv'), 'images': checksums,
                'license': 'CC BY-NC-SA 4.0', 'count': len(records)}
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest

def read_labels(folder):
    records = []
    seen = set()
    for line in (Path(folder) / 'labels.csv').read_text().splitlines():
        path, probability, kind = line.split()
        rel = PurePosixPath(path)
        p = float(probability)
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] != 'images':
            raise ValueError('Invalid image path')
        if path in seen or not np.isfinite(p) or not 0 <= p <= 1 or kind not in ('mono', 'poly'):
            raise ValueError('Invalid or duplicate annotation')
        seen.add(path)
        records.append({'path': path, 'label': p, 'type': kind})
    if not records:
        raise ValueError('No annotations')
    return records

def load(folder):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    if digest(folder / 'labels.csv') != manifest['labels_sha256']:
        raise ValueError('Labels checksum mismatch')
    records = read_labels(folder)
    if len(records) != manifest['count']:
        raise ValueError('Record count mismatch')
    images = []
    pixels_seen = set()
    duplicates = []
    kept = []
    for r in records:
        if digest(folder / r['path']) != manifest['images'][r['path']]:
            raise ValueError('Image checksum mismatch')
        with Image.open(folder / r['path']) as im:
            if im.mode != 'L' or im.size != (300, 300):
                raise ValueError('Expected 300 x 300 grayscale ELPV image')
            arr = np.asarray(im).copy()
        pixel_hash = hashlib.sha256(arr.tobytes()).hexdigest()
        if pixel_hash in pixels_seen:
            duplicates.append(r['path'])
            continue
        pixels_seen.add(pixel_hash)
        kept.append({**r, 'pixel_sha256': pixel_hash})
        images.append(arr)
    return kept, images, {k: v for k, v in manifest.items() if k != 'images'}, duplicates

def split(records, seed=42):
    certain = np.array([i for i, r in enumerate(records) if r['label'] in (0.0, 1.0)])
    ambiguous = np.array([i for i, r in enumerate(records) if r['label'] not in (0.0, 1.0)], dtype=int)
    strata = np.array([f"{r['type']}-{r['label']}" for r in records])
    train, other = train_test_split(certain, test_size=.4, random_state=seed, stratify=strata[certain])
    validation, test = train_test_split(other, test_size=.5, random_state=seed, stratify=strata[other])
    return {'train': train, 'validation': validation, 'test': test, 'ambiguous': ambiguous}
