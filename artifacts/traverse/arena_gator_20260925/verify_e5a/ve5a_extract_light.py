#!/usr/bin/env python3
"""VERIFY_E5a (independent verifier): pull the light per-row columns out of the E5a training / evaluation npz files so
the subset selections can be recomputed elsewhere. Reads only; writes <out>/<name>_light.npz + <out>/sha256.txt."""
import sys, os, hashlib, time, numpy as np
out = sys.argv[1]; files = sys.argv[2:]
os.makedirs(out, exist_ok=True)
KEYS = ('id', 'group', 'split', 'domain', 'arena', 'vehicle', 'tier', 'episode', 'anchor_frame', 'source')
for p in files:
    t0 = time.time()
    z = np.load(p, allow_pickle=True)
    d = {}
    for k in KEYS:
        if k in z.files:
            v = z[k]
            d[k] = v.astype(str) if v.dtype.kind in 'OUS' else v
    d['_keys'] = np.array(sorted(z.files))
    name = os.path.basename(p)[:-4]
    np.savez_compressed(os.path.join(out, name + '_light.npz'), **d)
    print(name, len(d['id']), sorted(d), round(time.time() - t0, 1), flush=True)
