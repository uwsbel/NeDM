#!/usr/bin/env python3
"""arena_gator_20260925 soil stage 1: light-column sanity check of every soil ci / subset / evaluation file (tier range,
world, vehicle, arenas, one standing-start row per episode, unique ids, split counts). numpy only; cluster paths."""
import numpy as np, glob, os, hashlib, sys
from collections import Counter
for p in sorted(glob.glob('/work1/dannegrut/harry/experiments/arena_gator_20260925/e4/soil_s1/subsets/*.npz') + glob.glob('/work1/dannegrut/harry/experiments/arena_gator_20260925/e5/eval_soil/*.npz') + glob.glob('/work1/dannegrut/harry/experiments/arena_gator_20260925/e4/soil_s1/*/ci_*.npz')):
    z = np.load(p, allow_pickle=True)
    t = z['tier'].astype(int); dom = z['domain'].astype(int); ar = z['arena'].astype(str); veh = z['vehicle'].astype(str); sp = z['split'].astype(str)
    ids = z['id'].astype(str); ep = z['episode'].astype(str); af = z['anchor_frame'].astype(int)
    bad = [i for i in ids if not i.endswith('@crm')]
    st = af == 0
    print(os.path.basename(p), 'rows', len(ids), 'tiers', t.min(), t.max(), 'domain', sorted(set(dom)), 'vehicle', sorted(set(veh)),
          'arenas', dict(Counter(ar[st])), 'episodes', len(set(ep)), 'startrows==episodes', st.sum() == len(set(ep)), 'dup', len(ids) - len(set(ids)), 'non-crm ids', len(bad),
          'split starts', dict(Counter(sp[st])))
