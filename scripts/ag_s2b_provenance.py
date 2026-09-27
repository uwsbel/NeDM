#!/usr/bin/env python3
"""Provenance check of the soil evaluation drives (arena_gator_20260925, S2 finish). Login node, python3 standard library.

Input: a json list of {run_id, route (G3 path of the task row's route file), route_file_sha256 (sha256 of the local
locked pick file's bytes), vehicle, group} (one per run the five soil evaluation mappings point to).
Per run folder (G3/soil_v1/runs/<run_id>): collection_request.json route path and route_sha256 (file bytes) equal the
task row's route and the locked local pick file (the driven route is the locked pick), case.json id equals the group, the soil config name / step / spacing, the collector sha256, the
vehicle block (Gator: name gator, calibrated wheel radii, spawn 0.35 m, wrapper sha; HMMWV: no vehicle block), the host.
  python3 ag_s2b_provenance.py --list runs.json --runs $G3/soil_v1/runs --out prov.json
"""
import argparse, json, os
from collections import Counter

CAL = {'front': 0.19575, 'rear': 0.2275}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', required=True); ap.add_argument('--runs', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    items = json.load(open(a.list))
    bad, cnt = [], Counter()
    hosts = Counter()
    for it in items:
        d = os.path.join(a.runs, it['run_id'])
        try:
            c = json.load(open(os.path.join(d, 'collection_request.json')))
            case = json.load(open(os.path.join(d, 'case.json')))
        except Exception as e:  # noqa: BLE001
            bad.append(dict(run_id=it['run_id'], why=f'unreadable {type(e).__name__}')); continue
        why = []
        if c.get('route') != it['route']:
            why.append('route path')
        if c.get('route_sha256') != it['route_file_sha256']:
            why.append('route file sha256')
        if case.get('id') != it['group']:
            why.append('case id')
        cc = c.get('crm_config') or {}
        cfg = (cc.get('name'), cc.get('step_s'), cc.get('spacing_m'))
        cnt[f'config {cfg}'] += 1
        cnt[f"collector {str(c.get('collector_sha256'))[:8]}"] += 1
        v = c.get('vehicle')
        vn = v.get('name') if isinstance(v, dict) else v
        if it['vehicle'] == 'gator':
            if vn != 'gator':
                why.append('vehicle')
            else:
                g = v.get('soil_wheel_geometry') or {}
                rr = {k: (g.get(k) or {}).get('radius_m') for k in CAL}
                if rr != CAL:
                    why.append(f'wheel {rr}')
                if v.get('spawn_dz_m') != 0.35:
                    why.append('spawn')
                cnt[f"gator wrapper {str(v.get('wrapper_sha256'))[:8]} switch {str(v.get('ag_vehicle_sha256'))[:8]}"] += 1
        elif vn not in (None, 'hmmwv'):
            why.append(f'vehicle {vn}')
        cnt[f"vehicle_block {vn}"] += 1
        hosts[str(c.get('host')).split('.')[0][:4]] += 1
        if why:
            bad.append(dict(run_id=it['run_id'], why=why))
    res = dict(runs=len(items), bad=len(bad), bad_runs=bad[:200], counts=dict(cnt), host_prefix=dict(hosts))
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'bad_runs'}, indent=1))


if __name__ == '__main__':
    main()
