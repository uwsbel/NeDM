#!/usr/bin/env python3
"""VERIFY_S1 (independent verifier of soil stage 1): pull the light per-row columns out of the soil stage-1 npz files
(per-arena files, subsets, evaluation files) so selections, tiers, vehicles and suite ids can be checked elsewhere.
Reads only; writes <out>/<name>_light.npz."""
import sys, os, time, numpy as np
out = sys.argv[1]; files = sys.argv[2:]
os.makedirs(out, exist_ok=True)
KEYS = ('id', 'group', 'split', 'domain', 'arena', 'vehicle', 'tier', 'episode', 'anchor_frame', 'source', 'status',
        'profile', 'fail', 'unsafe', 'route_len')
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
    print(name, len(d['id']), round(time.time() - t0, 1), flush=True)
