#!/usr/bin/env python3
"""Per-run host and chassis-contact record of the rigid evaluation drives (arena_gator_20260925, module E6b final
analysis). Runs on the AMD login node with the system python3 (numpy only), read-only on the run folders, so that
simulation_provenance.json and rich_intervals.npz do not have to be copied to the workstation.

  python3 ag_e6b_host_extract.py --runs $G3/rigid_eval/runs --out $G3/tools/e6b/final/run_hosts.json

Per run id: host (simulation_provenance.json, first key named host/hostname/node, short name, the rule of
scripts/ag_eval_pool.py run_host and scripts/ag_e6b_extras.py host_of), chassis_contact_n (max of
max_chassis_contact_resultant_n in rich_intervals.npz, None without that file), belly_min_m / belly_frames_below
(outcome.json vehicle.belly, Gator runs), complete (episode_complete.json present).
"""
import argparse, json, os
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
    out = dict(host=host_of(d), chassis_contact_n=None, belly_min_m=None, belly_frames_below=None,
               complete=os.path.exists(os.path.join(d, 'episode_complete.json')))
    p = os.path.join(d, 'rich_intervals.npz')
    if os.path.exists(p):
        try:
            r = np.load(p)
            ks = [k for k in r.files if 'chassis_contact' in k and 'max' in k]
            out['chassis_contact_n'] = float(r[ks[0]].max()) if ks and len(r[ks[0]]) else None
        except Exception as e:  # noqa: BLE001
            out['rich_error'] = str(e)
    try:
        o = json.load(open(os.path.join(d, 'outcome.json')))
        v = o.get('vehicle')
        b = v.get('belly') if isinstance(v, dict) else None
        if b:
            out['belly_min_m'] = b.get('min_clearance_m'); out['belly_frames_below'] = b.get('frames_below_surface')
    except (OSError, ValueError):
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', required=True); ap.add_argument('--out', required=True); ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    ids = sorted(os.listdir(a.runs))
    with ThreadPoolExecutor(a.workers) as ex:
        res = dict(zip(ids, ex.map(lambda i: one(os.path.join(a.runs, i)), ids)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(runs_dir=a.runs, n=len(ids), runs=res), open(a.out, 'w'))
    hosts = {}
    for v in res.values():
        hosts[v['host']] = hosts.get(v['host'], 0) + 1
    print(json.dumps(dict(n=len(ids), complete=sum(v['complete'] for v in res.values()), no_host=hosts.get(None, 0),
                          hosts=len(hosts), with_contact_record=sum(v['chassis_contact_n'] is not None for v in res.values()))))


if __name__ == '__main__':
    main()
