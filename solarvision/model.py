import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def select_threshold(y, scores, target=.9):
    y, scores = np.asarray(y), np.asarray(scores)
    if not 0 < target <= 1 or set(np.unique(y)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Need finite scores, both classes and a valid recall target')
    candidates = []
    for t in np.unique(scores):
        pred = scores >= t
        tp = np.sum(pred & (y == 1))
        recall = tp / np.sum(y == 1)
        precision = tp / np.sum(pred)
        if recall >= target:
            candidates.append((precision, float(t), recall))
    precision, threshold, recall = max(candidates)
    return threshold, {'precision': float(precision), 'recall': float(recall), 'target': target}

def metrics(y, score, threshold):
    y, score = np.asarray(y), np.asarray(score)
    tn, fp, fn, tp = confusion_matrix(y, score >= threshold, labels=[0, 1]).ravel()
    return {'n': len(y), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'precision': float(tp / max(tp+fp, 1)), 'recall': float(tp / max(tp+fn, 1)),
            'specificity': float(tn / max(tn+fp, 1)), 'accuracy': float((tp+tn)/len(y)),
            'f1': float(2*tp/max(2*tp+fp+fn, 1)), 'review_fraction': float((tp+fp)/len(y)),
            'ap': float(average_precision_score(y, score)),
            'auc': float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None}

def fit(X, y, partition):
    train, val = partition['train'], partition['validation']
    candidates = []
    for kind, width in [('intensity', 42), ('texture', X.shape[1])]:
        for c in [.01, .1, 1]:
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000, random_state=42))
            pipe.fit(X[train, :width], y[train])
            score = pipe.predict_proba(X[val, :width])[:, 1]
            candidates.append((float(average_precision_score(y[val], score)), kind, c, width, pipe, score))
    best = max(candidates, key=lambda r: r[0])
    ap, kind, c, width, pipe, scores = best
    scaler, logistic = pipe.steps[0][1], pipe.steps[1][1]
    threshold, target = select_threshold(y[val], scores)
    saved = {'feature_version': 1, 'kind': kind, 'C': c, 'width': width,
             'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
             'coef': logistic.coef_[0].tolist(), 'intercept': float(logistic.intercept_[0]),
             'threshold': threshold}
    selection = [{'kind': r[1], 'C': r[2], 'validation_ap': r[0]} for r in candidates]
    return saved, {'candidates': selection, 'validation_target': target, 'selected_validation_ap': ap}

def predict(model, X):
    X = np.asarray(X)
    if model['feature_version'] != 1 or X.ndim != 2 or X.shape[1] < model['width'] or not np.isfinite(X).all():
        raise ValueError('Invalid model version or feature matrix')
    z = ((X[:, :model['width']] - model['mean']) / model['scale']) @ np.array(model['coef']) + model['intercept']
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))

def bootstrap(y, scores, threshold, repetitions=1000):
    rng = np.random.default_rng(84)
    values = {'recall': [], 'precision': [], 'ap': []}
    y, scores = np.asarray(y), np.asarray(scores)
    groups = [np.flatnonzero(y == k) for k in [0, 1]]
    for _ in range(repetitions):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        m = metrics(y[idx], scores[idx], threshold)
        for key in values:
            values[key].append(m[key])
    return {k: np.quantile(v, [.025, .975]).tolist() for k, v in values.items()}
