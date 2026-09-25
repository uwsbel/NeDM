#!/usr/bin/env python3
"""Drift check (REVIEW_R1 amendment 7b) for arena_gator_20260925: tonight's HMMWV runs of old ids vs the old runs.

  soil : drift__<id> runs (G3/soil_v1/runs) vs crm_f104_v1/collect_v1/runs/<id>. Required: every npz array identical
         (trajectory, command_reference, anchor_state, crm_extra) and the same outcome status / elapsed time
         (soil runs are bit-identical across AMD GPU types).
  rigid: drift__<id> runs (G3/rigid_v1/runs) vs fdm_f104_50h_20260909/production_v3/runs/<id> (collect_traverse_f104.py,
         night 2). Required: >= 95 % the same outcome status (rigid Chrono is deterministic per node only); elapsed-time
         and trajectory agreement reported.
Run locally on synced copies:
  python scripts/ag_drift_check.py --mode soil --runs <dir with drift__*> [--ref DIR] [--out json]
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REF = {'soil': ROOT / 'artifacts/traverse/crm_f104_v1/collect_v1/runs',
       'rigid': ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/production_v3/runs'}
NPZ = {'soil': ('trajectory.npz', 'command_reference.npz', 'anchor_state.npz', 'crm_extra.npz'),
       'rigid': ('trajectory.npz', 'command_reference.npz', 'anchor_state.npz')}


def npz_equal(a, b):
    if not (a.exists() and b.exists()):
        return None
    x, y = np.load(a), np.load(b)
    if sorted(x.files) != sorted(y.files):
        return False
    def eq(u, v):
        if u.shape != v.shape or u.dtype != v.dtype:
            return False
        return np.array_equal(u, v, equal_nan=u.dtype.kind in 'fc')
    return all(eq(x[k], y[k]) for k in x.files)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['soil', 'rigid'], required=True)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--ref')
    ap.add_argument('--out')
    a = ap.parse_args(argv)
    ref = Path(a.ref) if a.ref else REF[a.mode]
    rows = []
    for d in sorted(Path(a.runs).glob('drift__*')):
        if not (d / 'outcome.json').exists():
            continue
        rid = d.name[len('drift__'):]
        o, r = json.load(open(d / 'outcome.json')), json.load(open(ref / rid / 'outcome.json'))
        row = dict(id=rid, status=o['status'], ref_status=r['status'], elapsed_s=o['elapsed_s'], ref_elapsed_s=r['elapsed_s'],
                   same_status=o['status'] == r['status'], same_elapsed=abs(o['elapsed_s'] - r['elapsed_s']) < 1e-9)
        row['npz_identical'] = {f: npz_equal(d / f, ref / rid / f) for f in NPZ[a.mode]}
        rows.append(row)
    n = len(rows)
    same = sum(r['same_status'] for r in rows)
    all_npz = sum(all(v is not False for v in r['npz_identical'].values()) and any(v for v in r['npz_identical'].values()) for r in rows)
    res = dict(mode=a.mode, n=n, same_status=same, same_elapsed=sum(r['same_elapsed'] for r in rows), npz_identical_runs=all_npz)
    if a.mode == 'soil':
        res['pass'] = n > 0 and all_npz == n and same == n
        res['rule'] = 'every npz array identical and same status (soil is bit-identical across GPU types)'
    else:
        res['pass'] = n > 0 and same >= 0.95 * n
        res['rule'] = '>= 95 % same outcome status (rigid deterministic per node only)'
    res['mismatches'] = [r for r in rows if not r['same_status'] or (a.mode == 'soil' and not all(r['npz_identical'].values()))]
    res['rows'] = rows
    print(json.dumps({k: v for k, v in res.items() if k != 'rows'}, indent=1))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
