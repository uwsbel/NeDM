#!/usr/bin/env python3
"""Per-drive extras for the E6b rigid evaluation index (arena_gator_20260925): simulation host, chassis ground contact,
belly clearance; and the one-node-per-group check (VERIFY_E6a point 2: Chrono rigid is deterministic per node only, so
every drive of a group must come from one host).

  PYTHONPATH=src:scripts python scripts/ag_e6b_extras.py --index $K3/e6/index/rigid_eval_v1.json --out $K3/e6/index/rigid_eval_v1_extras.json
Per driven row: host (simulation_provenance.json), chassis_contact_n (max chassis contact force in rich_intervals.npz;
None if the run kept no rich intervals), belly_min_m / belly_frames_below (outcome vehicle.belly, Gator runs).
Summary: groups with drives from more than one host (listed; they are flagged, not dropped), per (vehicle, arm, set)
share of drives with chassis contact.
"""
import argparse, json, os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def host_of(d):
    try:
        j = json.load(open(os.path.join(d, 'simulation_provenance.json')))
    except (OSError, ValueError):
        return None

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k.lower() in ('host', 'hostname', 'node') and isinstance(v, str):
                    return v
                r = walk(v)
                if r:
                    return r
        return None
    h = walk(j)
    return h.split('.')[0] if h else None


def one(d):
    out = dict(host=host_of(d), chassis_contact_n=None, belly_min_m=None, belly_frames_below=None)
    p = os.path.join(d, 'rich_intervals.npz')
    if os.path.exists(p):
        r = np.load(p)
        ks = [k for k in r.files if 'chassis_contact' in k and 'max' in k]
        out['chassis_contact_n'] = float(r[ks[0]].max()) if ks else None
    try:
        o = json.load(open(os.path.join(d, 'outcome.json')))
        b = (o.get('vehicle') or {}).get('belly') if isinstance(o.get('vehicle'), dict) else None
        if b:
            out['belly_min_m'] = b.get('min_clearance_m'); out['belly_frames_below'] = b.get('frames_below_surface')
    except (OSError, ValueError):
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    rows = [r for r in json.load(open(a.index))['rows'] if not r.get('missing')]
    dirs = sorted({r['run_dir'] for r in rows})
    with ThreadPoolExecutor(a.workers) as ex:
        ex_ = dict(zip(dirs, ex.map(one, dirs)))
    hosts = defaultdict(set)
    for r in rows:
        if r['world'] == 'rigid':
            hosts[r['group']].add(ex_[r['run_dir']]['host'])
    multi = {g: sorted(str(h) for h in v) for g, v in hosts.items() if len(v) > 1}
    nohost = sum(1 for d in dirs if ex_[d]['host'] is None)
    contact = defaultdict(list)
    for r in rows:
        c = ex_[r['run_dir']]['chassis_contact_n']
        if c is not None:
            contact[f"{r['vehicle']}|{r['arm']}|{r['set']}"].append(c > 0)
    summ = dict(driven_rows=len(rows), distinct_runs=len(dirs), runs_without_host=nohost, groups=len(hosts), groups_multi_host=len(multi),
                multi_host_examples=dict(list(multi.items())[:20]), hosts=dict(Counter(ex_[d]['host'] for d in dirs)),
                chassis_contact_share={k: dict(n=len(v), share=100 * float(np.mean(v))) for k, v in sorted(contact.items())})
    json.dump(dict(summary=summ, runs=ex_, multi_host_groups=multi), open(a.out, 'w'), indent=None, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != 'chassis_contact_share'}, indent=1))


if __name__ == '__main__':
    main()
