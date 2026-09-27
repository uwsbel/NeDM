#!/usr/bin/env python3
"""VERIFY E4: every array of the three subset files equals the source rows of the f104 file (matched by id)."""
import json
import numpy as np

K3 = '/home/harry/NeDM-traverse_mppi/artifacts/traverse/arena_gator_20260925'
BIG = np.load(K3 + '/e4/f104_hmmwv/ci_f104_hmmwv_both.npz', allow_pickle=True)
bid = BIG['id'].astype(str); pos = {s: i for i, s in enumerate(bid)}
subs = {n: np.load(K3 + f'/e4/subsets/{n}_f104_hmmwv_rigid.npz', allow_pickle=True) for n in ('M1', 'LC545', 'LC272')}
ix = {n: np.array([pos[s] for s in S['id'].astype(str)]) for n, S in subs.items()}
out = {n: {} for n in subs}
for k in BIG.files:
    b = BIG[k]
    for n, S in subs.items():
        a = S[k]
        if k in ('hist_cols', 'priv_names'):
            out[n][k] = bool(np.array_equal(a, b)); continue
        bb = b[ix[n]]
        out[n][k] = bool(np.array_equal(a.astype(str), bb.astype(str))) if a.dtype == object else bool(np.array_equal(a, bb, equal_nan=True))
        del a, bb
    del b
res = {n: dict(all_equal=all(v.values()), not_equal=[k for k, x in v.items() if not x], n_keys=len(v)) for n, v in out.items()}
print(json.dumps(res, indent=1))
json.dump(res, open(K3 + '/verify_e4/v5_subset_heavy.json', 'w'), indent=1)
