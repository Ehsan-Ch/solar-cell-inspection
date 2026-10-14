from pathlib import Path
from collections import Counter
from io import BytesIO
import base64
import csv
import json
import platform
import importlib.metadata
import numpy as np
from PIL import Image
from .data import load, split
from .features import batch, occlusion
from .model import fit, predict, metrics, bootstrap
from .review import ReviewQueue

def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def thumbnail(image):
    buffer = BytesIO()
    Image.fromarray(image).resize((180, 180), Image.Resampling.BILINEAR).save(buffer, format='JPEG', quality=80)
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode()

def run(folder, output):
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    records, images, source, duplicates = load(folder)
    partition = split(records)
    print('Extracting image features...', flush=True)
    X = batch(images)
    y = np.array([r['label'] for r in records])
    print('Selecting model on validation images...', flush=True)
    model, selection = fit(X, y, partition)
    scores = predict(model, X)
    test = partition['test']; threshold = model['threshold']
    result = metrics(y[test], scores[test], threshold)
    intervals = bootstrap(y[test], scores[test], threshold)
    subgroups = {}
    for kind in ('mono', 'poly'):
        idx = [i for i in test if records[i]['type'] == kind]
        subgroups[kind] = metrics(y[idx], scores[idx], threshold)
    # Stress tests transform the fixed test set; neither model nor threshold is retuned.
    stress = {}
    for name, transform in [('darker_20pct', lambda a: np.round(a*.8).astype(np.uint8)),
                            ('lighter_20pct', lambda a: np.clip(np.round(a*1.2),0,255).astype(np.uint8))]:
        stress[name] = metrics(y[test], predict(model, batch([transform(images[i]) for i in test])), threshold)
    roles = {int(i): role for role, ids in partition.items() for i in ids}
    predictions = [{**r, 'split': roles[i], 'score': float(scores[i]),
                    'flagged': bool(scores[i] >= threshold)} for i, r in enumerate(records)]
    selected = []
    for name, mask in [('true_positive', (y[test] == 1) & (scores[test] >= threshold)),
                       ('false_negative', (y[test] == 1) & (scores[test] < threshold)),
                       ('false_positive', (y[test] == 0) & (scores[test] >= threshold)),
                       ('true_negative', (y[test] == 0) & (scores[test] < threshold))]:
        candidates = sorted(test[mask], key=lambda i: abs(scores[i]-threshold))[:4]
        for i in candidates:
            selected.append({**predictions[i], 'outcome': name, 'image': thumbnail(images[i]),
                             'sensitivity': occlusion(images[i], lambda f: predict(model, f)).tolist()})
    ambiguous = partition['ambiguous']
    summary = {'source': source, 'duplicates_removed': duplicates, 'seed': 42,
               'counts': {k: len(v) for k, v in partition.items()},
               'labels': dict(Counter(str(r['label']) for r in records)),
               'features': X.shape[1], 'model': {k: model[k] for k in ('kind','C','width','threshold')},
               'selection': selection, 'test': result, 'bootstrap95': intervals,
               'subgroups': subgroups, 'stress': stress,
               'ambiguous': {'n': len(ambiguous), 'flagged': int(np.sum(scores[ambiguous] >= threshold)),
                             'mean_score': float(scores[ambiguous].mean())},
               'environment': {'python': platform.python_version(), **{p: importlib.metadata.version(p)
                    for p in ['numpy','pillow','scikit-learn','scikit-image']}},
               'limitations': ['Image-level split; module IDs unavailable, so module leakage is possible.',
                   'Only clear 0/1 annotations used for training and metrics; intermediate labels are excluded.',
                   'Scores are logistic outputs, not calibrated probabilities.',
                   'Brightness perturbations are synthetic stress tests, not external validation.']}
    (out/'model.json').write_text(json.dumps(model, indent=2, allow_nan=False), encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    write_csv(out/'predictions.csv', predictions)
    write_csv(out/'splits.csv', [{'path': r['path'], 'split': roles[i], 'pixel_sha256': r['pixel_sha256']}
                                for i, r in enumerate(records)])
    queue = ReviewQueue(out/'inspection.sqlite')
    try:
        for i in test:
            queue.add(records[i]['path'], float(scores[i]), threshold)
        write_csv(out/'review_queue.csv', queue.cases())
    finally:
        queue.close()
    from .report import render
    report_data = {'summary': summary, 'examples': selected,
                   'test': [predictions[i] for i in test]}
    (out/'report.html').write_text(render(report_data), encoding='utf-8')
    return summary
