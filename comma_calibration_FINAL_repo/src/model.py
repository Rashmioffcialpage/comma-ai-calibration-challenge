import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import Ridge

SMOOTH_SIGMAS = (2, 5, 10, 20, 40, 80)
RIDGE_ALPHA = 0.03

def interpolate_features(a):
    a = np.asarray(a, dtype=float).copy()
    x = np.arange(len(a))
    for j in range(a.shape[1]):
        ok = np.isfinite(a[:, j])
        if ok.sum() > 1:
            a[~ok, j] = np.interp(x[~ok], x[ok], a[ok, j])
        else:
            a[~ok, j] = 0.0
    return a

def make_features(raw):
    raw = interpolate_features(raw)
    scales = [raw]
    for sigma in SMOOTH_SIGMAS:
        scales.append(gaussian_filter1d(raw, sigma=sigma, axis=0))
    # Normalized time helps model slow camera-motion drift within a clip.
    return np.c_[*scales, np.linspace(0.0, 1.0, len(raw))]

def fit(X, y):
    models = []
    for target in range(2):
        m = Ridge(alpha=RIDGE_ALPHA)
        m.fit(X, y[:, target])
        models.append(m)
    return models

def predict(models, X):
    return np.column_stack([m.predict(X) for m in models])
