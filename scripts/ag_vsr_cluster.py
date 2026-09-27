#!/usr/bin/env python3
"""arena_gator_20260925, skeptical check of the soil results: login-node read-outs (numpy + json only, read only).

Own code; imports nothing from the project. Three parts, each written as one json into --out:
  eval    : the 12,310 soil evaluation run folders (ids from --eval-ids): collection request (route / case path and
            file sha256, soil config, time), sha256 of outcome.json / episode_complete.json / reference.json, the soil
            physics QA re-implemented (array shapes, finite states, launch check, explosion = speed > 15 m/s or chassis
            height above the BMP ground > 2.5 m, breakthrough without a stall), the Gator belly flag, the drive's mtimes.
  gator   : the 15,235 Gator training rows of soil_v3.json (kind designed / on_policy) against the HMMWV collect_v1
            drives of the same ids: complete, own QA, launch, status, simulated s, belly flag (clearance < -0.05 m for
            more than 1 s in a row, i.e. > 20 consecutive 0.05 s frames), designed speed profile read from the route
            file's meta (speed_profile_id), split, tier.
  pilot   : the pilot soil rows (calibrated wheels 'gator__' vs radius + 0.08 m 'gatorR8__') against collect_v1.

  python3 ag_vsr_cluster.py --part eval --eval-ids ids.txt --out DIR
"""
import argparse, hashlib, json, os, sys, time
from collections import Counter, defaultdict
import numpy as np

G3 = '/work1/dannegrut/harry/experiments/arena_gator_20260925'
REF = '/work1/dannegrut/harry/experiments/crm_f104_20260916/collect_v1/runs'
DT = 0.05


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def jl(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def longest(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def qa(d):
    """Own re-implementation of the soil validation rules; returns (ok, flag, info)."""
    o = jl(d + '/outcome.json'); iv = jl(d + '/initial_state_validation.json')
    if o is None or iv is None or not os.path.exists(d + '/trajectory.npz') or not os.path.exists(d + '/crm_extra.npz'):
        return False, 'unreadable', {}
    try:
        z = np.load(d + '/trajectory.npz'); e = np.load(d + '/crm_extra.npz')
        st, ac, po = z['state'], z['action'], z['pose']
    except Exception as ex:  # noqa: BLE001
        return False, 'unreadable', dict(err=type(ex).__name__)
    n = len(st)
    info = dict(frames=n, elapsed=float(o['elapsed_s']), status=o['status'])
    if not (n > 0 and st.shape == (n, 17) and ac.shape == (n, 3) and po.shape == (n, 3) and abs(n * DT - float(o['elapsed_s'])) < 1e-6):
        return False, 'shape', info
    if not (np.isfinite(st).all() and np.isfinite(ac).all() and np.isfinite(po).all()):
        return False, 'nonfinite', info
    if not bool(iv.get('passed')):
        return False, 'launch', info
    v = np.sqrt(st[:, 0] ** 2 + st[:, 1] ** 2); hgt = e['pos_z_m'] - e['bmp_ground_z_m']
    info.update(vmax=float(v.max()), hmax=float(hgt.max()))
    if v.max() > 15.0 or hgt.max() > 2.5:
        return False, 'explosion', info
    if o['status'] in ('soil_breakthrough_terminated', 'rollover'):
        tail = slice(max(0, n - 120), n)
        stall = longest((np.abs(st[tail, 0]) < 1.0) & (ac[tail, 1] > 0.3))
        info['stall_frames'] = int(stall)
        if stall < 20 and o['status'] == 'soil_breakthrough_terminated':
            return False, 'unstalled_break', info
    return True, None, info


def belly(d):
    p = d + '/vehicle_extra.npz'
    if not os.path.exists(p):
        return None
    z = np.load(p)
    b = z['belly_clearance_min_m'].astype(float); fr = z['frame']
    deep = b < -0.05
    run = longest(deep)
    return dict(min=float(np.nanmin(b)), run_frames=int(run), flag=bool(run * DT > 1.0), total_s=float(deep.sum() * DT),
                frames_contiguous=bool((np.diff(fr) == 1).all()) if len(fr) > 1 else True, n=int(len(b)))


def part_eval(a):
    ids = [x.strip() for x in open(a.eval_ids) if x.strip()]
    recs = []
    for rid in ids:
        d = f'{G3}/soil_v1/runs/{rid}'
        cr = jl(d + '/collection_request.json') or {}
        ok, flag, info = qa(d)
        r = dict(id=rid, ok=ok, flag=flag, status=info.get('status'), elapsed=info.get('elapsed'),
                 route=cr.get('route'), route_sha256=cr.get('route_sha256'), case=cr.get('case'), case_sha256=cr.get('case_sha256'),
                 config=(cr.get('crm_config') or {}).get('name'), step=(cr.get('crm_config') or {}).get('step_s'),
                 spacing=(cr.get('crm_config') or {}).get('spacing_m'), host=cr.get('host'),
                 req_mtime=os.path.getmtime(d + '/collection_request.json') if os.path.exists(d + '/collection_request.json') else None,
                 out_mtime=os.path.getmtime(d + '/outcome.json') if os.path.exists(d + '/outcome.json') else None,
                 outcome_sha=sha(d + '/outcome.json'), ec_sha=sha(d + '/episode_complete.json'),
                 reference_sha=sha(d + '/reference.json') if os.path.exists(d + '/reference.json') else None,
                 route_file_now_sha=sha(cr['route']) if cr.get('route') and os.path.exists(cr['route']) else None)
        o = jl(d + '/outcome.json') or {}
        vb = o.get('vehicle')
        r['vehicle'] = vb.get('name') if isinstance(vb, dict) else vb
        if isinstance(vb, dict):
            r['wheel'] = vb.get('soil_wheel_geometry', {}).get('rear', {}).get('radius_m'), vb.get('soil_wheel_geometry', {}).get('front', {}).get('radius_m')
            r['spawn_dz'] = vb.get('spawn_dz_m'); r['wrapper_sha'] = vb.get('wrapper_sha256'); r['switch_sha'] = vb.get('ag_vehicle_sha256')
            r['belly'] = belly(d)
        recs.append(r)
    json.dump(dict(part='eval', n=len(recs), created=time.strftime('%F %T'), rows=recs), open(os.path.join(a.out, 'eval.json'), 'w'))
    print(dict(n=len(recs), bad=sum(not r['ok'] for r in recs), flags=dict(Counter(r['flag'] for r in recs if not r['ok']))))


def profile_of(route_path):
    m = (jl(route_path) or {}).get('meta') or {}
    return m.get('speed_profile_id'), m.get('route_index')


def part_gator(a):
    rows = [r for r in json.load(open(f'{G3}/tasks/soil_v3.json')) if r['id'].startswith('gator__') and r.get('kind') in ('designed', 'on_policy')]
    recs = []
    for r in rows:
        d = f"{G3}/soil_v1/runs/{r['id']}"; pid = r['id'][len('gator__'):]
        x = dict(id=r['id'], tier=r['tier'], kind=r['kind'], split=r.get('split'), wheel=r.get('wheel'), extra=r.get('extra'))
        x['complete'] = os.path.exists(d + '/episode_complete.json')
        prof, ri = profile_of(r['route']) if r['kind'] == 'designed' else ('planner_proposal', None)
        x['profile'] = prof; x['route_index'] = ri
        if x['complete']:
            ok, flag, info = qa(d); x.update(ok=ok, flag=flag, status=info.get('status'), sim_s=info.get('elapsed'))
            o = jl(d + '/outcome.json') or {}; vb = o.get('vehicle')
            x['vehicle'] = vb.get('name') if isinstance(vb, dict) else vb
            x['rear_radius'] = (vb or {}).get('soil_wheel_geometry', {}).get('rear', {}).get('radius_m') if isinstance(vb, dict) else None
            x['belly'] = belly(d)
        ho = jl(f'{REF}/{pid}/outcome.json')
        x['h_status'] = ho.get('status') if ho else None; x['h_sim_s'] = float(ho['elapsed_s']) if ho else None
        recs.append(x)
    json.dump(dict(part='gator', n=len(recs), created=time.strftime('%F %T'), rows=recs), open(os.path.join(a.out, 'gator.json'), 'w'))
    print(dict(n=len(recs), complete=sum(x['complete'] for x in recs)))


def part_pilot(a):
    root = f'{G3}/pilot_gator/soil/runs'
    recs = []
    for rid in sorted(os.listdir(root)):
        d = f'{root}/{rid}'
        if not os.path.exists(d + '/episode_complete.json'):
            recs.append(dict(id=rid, complete=False)); continue
        pre, pid = rid.split('__', 1)
        ok, flag, info = qa(d)
        o = jl(d + '/outcome.json') or {}; vb = o.get('vehicle') or {}
        ho = jl(f'{REF}/{pid}/outcome.json')
        recs.append(dict(id=rid, prefix=pre, pid=pid, complete=True, ok=ok, flag=flag, status=info.get('status'),
                         rear_radius=vb.get('soil_wheel_geometry', {}).get('rear', {}).get('radius_m') if isinstance(vb, dict) else None,
                         h_status=ho.get('status') if ho else None, belly=belly(d)))
    json.dump(dict(part='pilot', n=len(recs), rows=recs), open(os.path.join(a.out, 'pilot.json'), 'w'))
    print(dict(n=len(recs), by_prefix=dict(Counter(r.get('prefix') for r in recs))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--part', required=True, choices=['eval', 'gator', 'pilot'])
    ap.add_argument('--eval-ids'); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dict(eval=part_eval, gator=part_gator, pilot=part_pilot)[a.part](a)


if __name__ == '__main__':
    main()
