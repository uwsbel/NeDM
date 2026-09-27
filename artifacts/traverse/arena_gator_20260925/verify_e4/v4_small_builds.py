#!/usr/bin/env python3
"""VERIFY E4: small builds made by me with scripts/ag_build_ds.py from 40 random rigid + 30 random soil f104 runs
(one-world rigid, one-world soil, two-world) must equal the big f104 file's rows of the same episodes, every array.
Tests that rows depend only on their own episode, and the one-world path against the two-world (ga_build_mixed) one."""
import json, sys
import numpy as np

K3 = '/home/harry/NeDM-traverse_mppi/artifacts/traverse/arena_gator_20260925'
BIG = np.load(K3 + '/e4/f104_hmmwv/ci_f104_hmmwv_both.npz', allow_pickle=True)
bid = BIG['id'].astype(str); pos = {s: i for i, s in enumerate(bid)}
cache = {}
out = {}
for name in ('pos_rigid/ci_f104_hmmwv_rigid', 'pos_crm/ci_f104_hmmwv_crm', 'pos_both/ci_f104_hmmwv_both'):
    S = np.load(f'/tmp/ve4/{name}.npz', allow_pickle=True)
    sid = S['id'].astype(str)
    missing = [s for s in sid if s not in pos]
    ix = np.array([pos[s] for s in sid if s in pos])
    # the big file must have no other rows on these episodes
    eps = set(zip(S['domain'].astype(int).tolist(), S['episode'].astype(str)))
    extra = int(sum(1 for d, e in zip(BIG['domain'].astype(int), BIG['episode'].astype(str)) if (int(d), e) in eps)) - len(sid) if name.endswith('both') or True else None
    eq = {}
    for k in sorted(set(S.files) & set(BIG.files)):
        a = S[k]
        if k not in cache:
            cache[k] = BIG[k]
        b = cache[k]
        if a.ndim == 0 or (k in ('hist_cols', 'priv_names')):
            eq[k] = bool(np.array_equal(a, b)); continue
        b = b[ix]
        eq[k] = bool(np.array_equal(a.astype(str), b.astype(str))) if a.dtype == object or b.dtype == object else bool(np.array_equal(a, b, equal_nan=True))
    out[name] = dict(rows=len(sid), missing_in_big=len(missing), big_rows_on_these_episodes_minus_rows=extra,
                     keys_only_small=sorted(set(S.files) - set(BIG.files)), keys_only_big=sorted(set(BIG.files) - set(S.files)),
                     all_equal=all(eq.values()), not_equal=[k for k, v in eq.items() if not v])
print(json.dumps(out, indent=1))
json.dump(out, open(K3 + '/verify_e4/v4_small_builds.json', 'w'), indent=1)
