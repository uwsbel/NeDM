"""Hand-built terrain+speed features of one route corridor (same tensor the risk model sees).

Written by the 2026-09-14 transfer audit as the fairest simple baseline; kept verbatim so the hand-rule arm
is exactly the rule that audit evaluated. X channels: 0 height rel. to start, 1 along-path grade, 2 cross-slope,
3 commanded speed, 4 valid mask; 96 stations x 32 lateral samples (centre columns 15-16).
"""
import numpy as np

FEATURES = ['maxgrade', 'maxcross', 'max_rise', 'max_dip', 'relief', 'climb10', 'drop10', 'upfrac0.25', 'absfrac0.25',
            'upslow0.25_4.5', 'upslow0.15_4.5', 'up_over_v', 'max_up', 'max_down', 'max_absgc', 'deficit15',
            'crest_acc', 'dip_acc', 'cross_at_speed', 'grade_at_speed', 'mean_absgrade', 'mean_absgc']


def _terrain(X, L):
    c = slice(15, 17)
    h = X[0, :, c].mean(1).astype(float); ga = X[1, :, c].mean(1).astype(float); gc = X[2, :, c].mean(1).astype(float)
    v = X[3, :, 16].astype(float); n = len(h); ds = L / (n - 1)
    out = dict(max_rise=float(h.max()), max_dip=float(-h.min()), relief=float(h.max() - h.min()))
    w = max(1, int(round(10.0 / ds)))
    climb = 0.0; drop = 0.0
    for i in range(n):
        j = min(n, i + w + 1)
        climb = max(climb, float(h[i:j].max() - h[i])); drop = max(drop, float(h[i] - h[i:j].min()))
    out['climb10'] = climb; out['drop10'] = drop
    for th in (0.15, 0.25, 0.35):
        out[f'upfrac{th}'] = float((ga > th).mean())
        out[f'absfrac{th}'] = float((np.abs(ga) > th).mean())
        for vth in (3.0, 4.5):
            out[f'upslow{th}_{vth}'] = float(((ga > th) & (v < vth)).mean())
    out['up_over_v'] = float((np.maximum(ga, 0) / np.maximum(v, 0.5)).max())
    out['max_up'] = float(ga.max()); out['max_down'] = float(-ga.min())
    out['max_absgc'] = float(np.abs(gc).max())
    w15 = max(1, int(round(15.0 / ds))); deficit = -1e9
    for i in range(n):
        j = min(n, i + w15 + 1)
        deficit = max(deficit, float(h[i:j].max() - h[i]) - v[i] ** 2 / (2 * 9.81))
    out['deficit15'] = deficit
    d2 = np.gradient(np.gradient(h, ds), ds)
    out['crest_acc'] = float((-d2 * v ** 2).max()); out['dip_acc'] = float((d2 * v ** 2).max())
    out['cross_at_speed'] = float((np.abs(gc) * v).max()); out['grade_at_speed'] = float((np.abs(ga) * v).max())
    out['mean_absgrade'] = float(np.abs(ga).mean()); out['mean_absgc'] = float(np.abs(gc).mean())
    return out


def route_features(X, L):
    f = _terrain(X, L)
    f['maxgrade'] = float(np.abs(X[1, :, 14:18]).max()); f['maxcross'] = float(np.abs(X[2, :, 14:18]).max())
    return np.array([f[c] for c in FEATURES], float)
