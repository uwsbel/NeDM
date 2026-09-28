#!/usr/bin/env python3
"""Read-only per-route extraction for the offroad_vehicles_20260927 results docs (runs on the AMD login node).
Writes JSON lines to stdout; reads only. One line per (row) with the Polaris/arm run and the Gator / HMMWV twins."""
import json, os, sys
from multiprocessing import Pool
import numpy as np

G4 = '/work1/dannegrut/harry/experiments/offroad_vehicles_20260927'
G3RUNS = '/work1/dannegrut/harry/experiments/arena_gator_20260925/soil_v1/runs'
HRUNS = '/work1/dannegrut/harry/experiments/crm_f104_20260916/collect_v1/runs'
HFLAG = '/work1/dannegrut/harry/experiments/crm_f104_20260916/collect_v1/runs_flagged'
sys.path.insert(0, G4 + '/source/scripts')
import crm_qa  # noqa: E402

DT = 0.05


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def longest(m):
    best = cur = 0
    for v in m:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def run_info(d, qa=True, belly=True):
    x = dict(dir=d, exists=os.path.isdir(d))
    if not x['exists']:
        return x
    x['complete'] = os.path.isfile(os.path.join(d, 'episode_complete.json'))
    o = load(os.path.join(d, 'outcome.json')) or {}
    x['status'] = o.get('status')
    x['sim_s'] = o.get('elapsed_s')
    x['wall_s'] = o.get('wall_s')
    v = o.get('vehicle')
    x['vehicle'] = v.get('name') if isinstance(v, dict) else v
    iv = load(os.path.join(d, 'initial_state_validation.json')) or {}
    x['launch_ok'] = bool(iv.get('passed', False)) if iv else None
    cr = load(os.path.join(d, 'collection_request.json')) or {}
    x['host'] = cr.get('host')
    cf = load(os.path.join(d, 'collection_failure.json'))
    x['collection_failure'] = (str(cf.get('error', ''))[:200] if cf else None)
    if qa and x['complete']:
        q = crm_qa.check(d)
        x['qa_ok'] = bool(q['ok'])
        x['qa_flag'] = q.get('flag')
    if belly and x['complete']:
        try:
            b = np.load(os.path.join(d, 'vehicle_extra.npz'))['belly_clearance_min_m'].astype(float)
            deep = b < -0.05
            x['belly_min_m'] = float(np.nanmin(b))
            x['belly_run_s'] = longest(deep) * DT
            x['belly_flag'] = bool(x['belly_run_s'] > 1.0)
        except Exception as e:  # noqa: BLE001
            x['belly_error'] = type(e).__name__
    return x


def route_info(p):
    r = load(p)
    if not r:
        return dict(route_missing=True)
    m = r.get('meta') or {}
    sp = np.asarray(r.get('speeds') or [], float)
    inner = sp[1:-1] if len(sp) > 2 else sp
    return dict(candidate=m.get('candidate'), speed_profile_id=m.get('speed_profile_id'), cruise=m.get('cruise_speed_mps'),
                lateral_offset=m.get('lateral_offset_m'), route_index=m.get('route_index'),
                v_min=float(inner.min()) if len(inner) else None, v_max=float(inner.max()) if len(inner) else None,
                v_mean=float(inner.mean()) if len(inner) else None, length_m=float(r['stations'][-1]) if r.get('stations') else None)


def work(r):
    tid = r['id']
    pre, pid = tid.split('__', 1)
    out = dict(id=tid, pre=pre, pair_id=r.get('pair_id', pid), arm=r.get('arm') or pre, vehicle=r.get('vehicle'), sample=r.get('sample'),
               kind=r.get('kind'), tier=r.get('tier'), collect_tier=r.get('collect_tier'), group=r.get('group'), stratum=r.get('stratum'),
               split=r.get('split'), src=r.get('_src'))
    out['failed_file'] = os.path.isfile(os.path.join(G4, 'soil_v1/failed', tid + '.json'))
    out['run'] = run_info(os.path.join(G4, 'soil_v1/runs', tid))
    if r.get('kind') in ('designed', 'on_policy'):
        out['route'] = route_info(r['route'])
        if r.get('_twins'):
            out['gator'] = run_info(os.path.join(G3RUNS, 'gator__' + out['pair_id']), qa=True, belly=True)
            hd = os.path.join(HRUNS, out['pair_id'])
            if not os.path.isdir(hd):
                hd = os.path.join(HFLAG, out['pair_id'])
            out['hmmwv'] = run_info(hd, qa=True, belly=False)
    return out


def main():
    rows = []
    seen = set()
    v5 = json.load(open(G4 + '/tasks/soil_v5_polaris.json'))
    for r in v5:
        if r.get('vehicle') == 'polaris' and r.get('kind') in ('designed', 'on_policy') and int(r.get('tier', -99)) >= 0:
            r = dict(r, _src='collection', _twins=True)
            rows.append(r); seen.add(r['id'])
    for f in ('smoke_v2_polaris.json', 'analysis_m113_combined.json'):
        for r in json.load(open(G4 + '/tasks/' + f)):
            if r['id'] in seen:
                continue
            if r.get('sample') in ('A', 'B', 'bitid'):
                rows.append(dict(r, _src=f, _twins=False)); seen.add(r['id'])
    sys.stderr.write(f'{len(rows)} rows\n')
    with Pool(int(os.environ.get('NP', '12'))) as p:
        for i, o in enumerate(p.imap(work, rows, chunksize=16)):
            sys.stdout.write(json.dumps(o) + '\n')
            if i % 2000 == 0:
                sys.stderr.write(f'{i}\n')


if __name__ == '__main__':
    main()
