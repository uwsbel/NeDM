#!/usr/bin/env python3
"""VERIFY_gator_full: body-in-soil flag of the task B stage-2 Gator evaluation drives, own code (as VERIFY_S1's rule):
lowest hull point more than 0.05 m under the undisturbed surface for more than 1 s in a row (vehicle_extra.npz
belly_clearance_min_m; frame spacing from outcome.json elapsed_s / (frames - 1)). Runs = the drives my own route-content
matching chose (outcomes.json 'chosen_runs'), read from the local copies in e6/runs_soil (hash-checked against the cluster).
  python scripts/ag_vbf_belly.py <K3> <outcomes.json> <out json>"""
import sys, json
from pathlib import Path
import numpy as np
K3 = Path(sys.argv[1]); ch = json.load(open(sys.argv[2]))['chosen_runs']; out = {}
for arm, gm in ch.items():
    if not arm.endswith('_gator'):
        continue
    flags = []
    for g, rid in gm.items():
        d = K3 / 'e6/runs_soil' / rid
        o = json.load(open(d / 'outcome.json'))
        b = np.load(d / 'vehicle_extra.npz')['belly_clearance_min_m'].astype(float)
        dt = o['elapsed_s'] / max(o['frames'] - 1, 1)
        best = cur = 0
        for v in b < -0.05:
            cur = cur + 1 if v else 0; best = max(best, cur)
        flags.append(best * dt > 1.0)
    out[arm] = dict(n=len(flags), flagged=int(sum(flags)), share_pct=100 * float(np.mean(flags)))
    print(arm, out[arm])
json.dump(out, open(sys.argv[3], 'w'), indent=1)
