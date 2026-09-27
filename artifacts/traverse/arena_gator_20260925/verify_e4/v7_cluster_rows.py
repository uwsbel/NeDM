#!/usr/bin/env python3
"""VERIFY E4 (cluster login node): rows of the builder's g203 soil test file and Gator pilot test file recomputed
per episode with the unchanged tools (n2_reanchor_dataset.one, ga_build_mixed.cut_episode) from the raw runs, for a
random sample of episodes; own anchor count via the same rule is in v2 (local)."""
import json, os, sys
import numpy as np
G3 = '/work1/dannegrut/harry/experiments/arena_gator_20260925'
sys.path.insert(0, G3 + '/source/scripts')
import n2_reanchor_dataset as RA, ga_build_mixed as GBM
rng = np.random.default_rng(11)
out = {}
for name, path, mroot, roots in [
        ('g203_soil', G3 + '/e4/tests/test_partial_g203/ci_g203_hmmwv_crm.npz', G3 + '/map_roots/g203', dict(crm=[G3 + '/soil_v1/runs'])),
        ('gator_pilot', G3 + '/e4/tests/test_pilot_gator/ci_f104_gator_both.npz', G3 + '/e4/map_roots/f104',
         dict(rigid=[G3 + '/pilot_gator/rigid/runs'], crm=[G3 + '/pilot_gator/soil/runs']))]:
    RA.init(mroot, dict(anchors=4, min_rem=12.0))
    z = np.load(path, allow_pickle=True)
    ep = z['episode'].astype(str); dom = z['domain'].astype(int); af = z['anchor_frame'].astype(int); ids = z['id'].astype(str)
    X = z['X']; C = z['ctx']; H = z['hist']; M = z['hmask']; P = z['privileged']; E = z['E']; T = z['T']
    for w, code in (('rigid', 0), ('crm', 1)):
        if w not in roots: continue
        eps = sorted(set(ep[dom == code])); pick = rng.choice(eps, min(25, len(eps)), replace=False)
        ok = 0; bad = []
        for e in pick:
            d = os.path.join(roots[w][0], e)
            rows = np.flatnonzero((dom == code) & (ep == e)); rr = {r['anchor_frame']: r for r in RA.one(d)}
            _, (h, m, p) = GBM.cut_episode((e, [int(af[i]) for i in rows], roots[w], w == 'crm'))
            good = len(rr) == len(rows) and np.array_equal(h, H[rows]) and np.array_equal(m, M[rows]) and np.array_equal(p, P[rows])
            for i in rows:
                r = rr.get(int(af[i]))
                good = good and r is not None and np.array_equal(r['X'], X[i]) and np.array_equal(r['ctx'], C[i]) and np.array_equal(r['E'], E[i], equal_nan=True) \
                    and np.array_equal(r['T'], T[i], equal_nan=True) and r['fail'] == int(z['fail'][i]) and r['unsafe'] == int(z['unsafe'][i]) \
                    and r['event_idx'] == int(z['event_idx'][i]) and f"{r['id']}@{w}" == ids[i]
            ok += bool(good)
            if not good: bad.append(e)
        out[f'{name}|{w}'] = dict(sampled=len(pick), identical=ok, bad=bad[:5])
print(json.dumps(out, indent=1))
