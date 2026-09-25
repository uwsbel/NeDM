#!/usr/bin/env python3
"""First-episode checks of the E3b2 launches (arena_gator_20260925), run locally on synced copies.

Soil (runs made by the new soil_v2 jobs, i.e. not in the soil_v1 launch and not imported from the pilot):
  every run: completion marker, crm_qa.check validity, launch check passed, finite arrays, physics step 0.001 s and
  particle spacing 0.08 m in collection_request.json;
  Gator rows (gator__*): a 'vehicle' block with name gator, the frozen ag_vehicle.py / ag_crm_collect.py hashes and the
  calibrated soil cylinders (front 0.19575 m, rear 0.2275 m) in outcome.json and collection_request.json, and
  vehicle_extra.npz present; HMMWV rows (bitid__, spread headroom, g203/g228): no 'vehicle' block anywhere and no
  vehicle_extra.npz;
  bitid__<id>: every npz array identical to the soil_v1 run <id> (made by crm_collect.py directly), same status and time.
Rigid (runs made by the pool steps): completion marker, launch and native-height checks, finite state/action/pose,
  Gator rows with the Gator vehicle block + the Gator runtime fingerprint, HMMWV spread rows without a vehicle block,
  with the HMMWV fingerprint and the r2 source root.
  python scripts/ag_e3b2_verify.py --soil-runs /tmp/ag_e3b2/soil_runs --rigid-runs /tmp/ag_e3b2/rigid_runs --out <json>
"""
import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
import crm_qa

AG_VEHICLE_SHA = '072716ee715469950409c334482bf990853dd4f70b3c85790eec99f04b331f9d'
WRAPPER_SOIL_SHA = 'b52e1fa69afe5745839bc9c32c5425b7a0328acbed61abdf0e4963644e5e2346'
WRAPPER_RIGID_SHA = '11c2891680c88c2fc2ca10cbb1f13712ae89814090e2be3ecf0085ebe5ae80e5'
GATOR_FP_SHA = '599c514e84a28800e567d3b6260402ebce5d7ed6efeb1561168e2d81cc12ed77'
SOIL_R = {'front': 0.19575, 'rear': 0.2275}
NPZ_SOIL = ('trajectory.npz', 'command_reference.npz', 'anchor_state.npz', 'crm_extra.npz')


def npz_equal(a, b):
    x, y = np.load(a), np.load(b)
    if sorted(x.files) != sorted(y.files):
        return False
    return all(x[k].shape == y[k].shape and x[k].dtype == y[k].dtype
               and np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind in 'fc') for k in x.files)


def finite(z):
    return bool(all(np.isfinite(z[k]).all() for k in ('state', 'action', 'pose') if k in z.files))


def soil(runs, refs):
    out, bit = [], []
    for d in sorted(Path(runs).iterdir()):
        if not (d / 'episode_complete.json').exists():
            continue
        rid = d.name
        o = json.load(open(d / 'outcome.json')); rq = json.load(open(d / 'collection_request.json'))
        launch = json.load(open(d / 'initial_state_validation.json'))
        q = crm_qa.check(str(d))
        z = np.load(d / 'trajectory.npz')
        cfg = rq.get('crm_config', {})
        gator = rid.startswith('gator__')
        r = dict(id=rid, vehicle='gator' if gator else 'hmmwv', status=o['status'], elapsed_s=o['elapsed_s'],
                 qa_ok=bool(q['ok']), qa_flag=q.get('flag'), launch_passed=bool(launch.get('passed')), finite=finite(z),
                 step_s=cfg.get('step_s'), spacing_m=cfg.get('spacing_m', cfg.get('spacing')), host=rq.get('host'),
                 rtf=o['crm']['rtf_sim_over_wall'], vehicle_block_outcome='vehicle' in o, vehicle_block_request='vehicle' in rq,
                 vehicle_extra=(d / 'vehicle_extra.npz').exists())
        if gator:
            v = o.get('vehicle', {})
            geom = v.get('soil_wheel_geometry', {})
            r.update(v_name=v.get('name'), v_sha_ok=v.get('ag_vehicle_sha256') == AG_VEHICLE_SHA,
                     wrapper_sha_ok=v.get('wrapper_sha256') == WRAPPER_SOIL_SHA,
                     wheel_ok=all(abs(geom.get(k, {}).get('radius_m', -1) - SOIL_R[k]) < 1e-9 for k in SOIL_R))
            r['vehicle_ok'] = bool(r['vehicle_block_outcome'] and r['vehicle_block_request'] and r['vehicle_extra']
                                   and r['v_name'] == 'gator' and r['v_sha_ok'] and r['wrapper_sha_ok'] and r['wheel_ok'])
        else:
            r['vehicle_ok'] = not (r['vehicle_block_outcome'] or r['vehicle_block_request'] or r['vehicle_extra'])
        out.append(r)
        if rid.startswith('bitid__'):
            ref = Path(refs) / rid[len('bitid__'):]
            ro = json.load(open(ref / 'outcome.json'))
            eq = {f: npz_equal(d / f, ref / f) for f in NPZ_SOIL}
            bit.append(dict(id=rid, status=o['status'], ref_status=ro['status'], elapsed_s=o['elapsed_s'],
                            ref_elapsed_s=ro['elapsed_s'], npz_identical=eq, identical=all(eq.values())
                            and o['status'] == ro['status'] and o['elapsed_s'] == ro['elapsed_s'],
                            ref_host=json.load(open(ref / 'collection_request.json')).get('host'), host=rq.get('host')))
    return out, bit


def rigid(runs):
    out = []
    for d in sorted(Path(runs).iterdir()):
        if not (d / 'episode_complete.json').exists():
            continue
        rid = d.name
        o = json.load(open(d / 'outcome.json')); rq = json.load(open(d / 'collection_request.json'))
        launch = json.load(open(d / 'initial_state_validation.json')); nh = json.load(open(d / 'native_height_check.json'))
        z = np.load(d / 'trajectory.npz')
        gator = rid.startswith('gator__')
        r = dict(id=rid, vehicle='gator' if gator else 'hmmwv', status=o['status'], elapsed_s=o['elapsed_s'],
                 launch_passed=bool(launch.get('passed')), native_height_passed=bool(nh.get('passed')), finite=finite(z),
                 vehicle_block='vehicle' in o, fingerprint_sha=rq.get('runtime_fingerprint_sha256'),
                 source_root=rq.get('source_root'), gates=rq.get('ext', {}).get('gates'),
                 rich_telemetry_left=(d / 'rich_telemetry.npz').exists(), rich_intervals=(d / 'rich_intervals.npz').exists())
        if gator:
            v = o.get('vehicle', {})
            r['vehicle_ok'] = bool(v.get('name') == 'gator' and v.get('ag_vehicle_sha256') == AG_VEHICLE_SHA
                                   and v.get('wrapper_sha256') == WRAPPER_RIGID_SHA and r['fingerprint_sha'] == GATOR_FP_SHA
                                   and (d / 'vehicle_extra.npz').exists())
        else:
            r['vehicle_ok'] = bool(not r['vehicle_block'] and r['fingerprint_sha'] != GATOR_FP_SHA
                                   and not (d / 'vehicle_extra.npz').exists() and str(r['source_root']).endswith('/r2/source'))
        out.append(r)
    return out


def summarise(rows, keys):
    res = {}
    for veh in sorted({r['vehicle'] for r in rows}):
        rs = [r for r in rows if r['vehicle'] == veh]
        res[veh] = dict(n=len(rs), status=dict(Counter(r['status'] for r in rs)),
                        fail_share=sum(r['status'] != 'goal_reached' for r in rs) / len(rs),
                        **{f'not_{k}': sum(not r[k] for r in rs) for k in keys})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--soil-runs')
    ap.add_argument('--soil-refs', help='local copy of the soil_v1 runs the bitid rows re-drive')
    ap.add_argument('--rigid-runs')
    ap.add_argument('--out')
    a = ap.parse_args()
    res = {}
    if a.soil_runs:
        s, b = soil(a.soil_runs, a.soil_refs)
        res['soil'] = summarise(s, ('qa_ok', 'launch_passed', 'finite', 'vehicle_ok'))
        res['soil']['step_spacing'] = dict(Counter(f"{r['step_s']}/{r['spacing_m']}" for r in s))
        res['soil']['bitid'] = b
        res['soil_rows'] = s
    if a.rigid_runs:
        r = rigid(a.rigid_runs)
        res['rigid'] = summarise(r, ('launch_passed', 'native_height_passed', 'finite', 'vehicle_ok'))
        res['rigid']['rich_telemetry_left'] = sum(x['rich_telemetry_left'] for x in r)
        res['rigid']['gates'] = dict(Counter(json.dumps(x['gates'], sort_keys=True) for x in r))
        res['rigid_rows'] = r
    print(json.dumps({k: v for k, v in res.items() if not k.endswith('_rows')}, indent=1))
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
