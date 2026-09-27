#!/usr/bin/env python3
"""Soil throughput on a node before / after soil_v3 overlap workers were added (arena_gator_20260925, soil track step 2).
Login node, python3, standard library only.

For every finished soil run in <out>/runs whose collection_request.json names one of the hosts: end = outcome.json mtime,
start = end - wall_s, simulated length = episode_complete.json actual_elapsed_s, loop cost = 1 / crm.rtf_sim_over_wall.
Throughput of a host in a window [a, b] = simulated seconds of the runs that ENDED in the window / (b - a), per GPU, and
the same with each run's simulated seconds pro-rated to the part of its wall interval inside the window (less biased for
short windows). Reported for the test host before (t0 - window .. t0) and after (t0 + settle .. now), split by worker
kind (evaluation rows = soil_v3 workers, training rows = soil_v2 workers), and for control hosts over the same windows.
  python3 ag_s2_overlap_eval.py --out $G3/soil_v1 --host k004-002 --control k004-003 --t0 14:41:32 [--gpus 8] [--json f]
"""
import argparse, json, os, statistics, time
from pathlib import Path


def t_of(s):
    if ':' in s:
        p = [int(x) for x in s.split(':')]
        lt = time.localtime()
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, p[0], p[1], p[2] if len(p) > 2 else 0, 0, 0, -1))
    return float(s)


def load(p):
    try:
        return json.load(open(p))
    except (OSError, ValueError):
        return None


def runs_on(out, hosts, since):
    res = []
    rd = Path(out) / 'runs'
    for e in os.scandir(rd):
        o = Path(e.path) / 'outcome.json'
        try:
            m = o.stat().st_mtime
        except OSError:
            continue
        if m < since:
            continue
        req = load(Path(e.path) / 'collection_request.json') or {}
        host = str(req.get('host') or req.get('hostname') or (req.get('provenance') or {}).get('host') or '')
        if not host:
            txt = json.dumps(req)
            host = next((h for h in hosts if h in txt), '')
        h = next((x for x in hosts if host.startswith(x)), None)
        if h is None:
            continue
        oc = load(o) or {}
        ec = load(Path(e.path) / 'episode_complete.json') or {}
        wall = float(oc.get('wall_s') or ec.get('wall_s') or 0.0)
        sim = float(ec.get('actual_elapsed_s') or oc.get('elapsed_s') or 0.0)
        crm = oc.get('crm') if isinstance(oc.get('crm'), dict) else {}
        rtf = crm.get('rtf_sim_over_wall')
        res.append(dict(id=e.name, host=h, end=m, start=m - wall, wall=wall, sim=sim, loop=(1.0 / rtf) if rtf else None,
                        kind='eval' if ('__' in e.name and not e.name.startswith('gator__f104_v2_group_')) else 'train'))
    return res


def window(rs, a, b):
    ended = [r for r in rs if a <= r['end'] <= b]
    prorated = 0.0
    for r in rs:
        lo, hi = max(a, r['start']), min(b, r['end'])
        if hi > lo and r['wall'] > 0:
            prorated += r['sim'] * (hi - lo) / r['wall']
    loops = [r['loop'] for r in ended if r['loop']]
    return dict(n=len(ended), sim_s=sum(r['sim'] for r in ended), rate_ended=sum(r['sim'] for r in ended) / (b - a),
                rate_prorated=prorated / (b - a), loop_median=statistics.median(loops) if loops else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--host', required=True)
    ap.add_argument('--control', nargs='*', default=[])
    ap.add_argument('--t0', required=True)
    ap.add_argument('--window', type=float, default=3600)
    ap.add_argument('--settle', type=float, default=300, help='skip the first s after t0 (episodes in flight)')
    ap.add_argument('--gpus', type=int, default=8)
    ap.add_argument('--json')
    a = ap.parse_args()
    t0 = t_of(a.t0); now = time.time()
    hosts = [a.host] + a.control
    rs = runs_on(a.out, hosts, t0 - a.window - 3600)
    res = dict(t0=time.strftime('%H:%M:%S', time.localtime(t0)), now=time.strftime('%H:%M:%S', time.localtime(now)), gpus=a.gpus, hosts={})
    for h in hosts:
        r_h = [r for r in rs if r['host'] == h]
        before = window(r_h, t0 - a.window, t0)
        after = window(r_h, t0 + a.settle, now)
        after_k = {k: window([r for r in r_h if r['kind'] == k], t0 + a.settle, now) for k in ('train', 'eval')}
        res['hosts'][h] = dict(role='test' if h == a.host else 'control', before=before, after=after, after_by_kind=after_k,
                               ratio_prorated=(after['rate_prorated'] / before['rate_prorated']) if before['rate_prorated'] else None,
                               ratio_loop=(after['loop_median'] / before['loop_median']) if before['loop_median'] and after['loop_median'] else None)
    for h, v in res['hosts'].items():
        print(f"{h} ({v['role']}): before {v['before']['rate_prorated'] / a.gpus:.3f} sim-s/s per GPU ({v['before']['n']} runs, loop {v['before']['loop_median']}); "
              f"after {v['after']['rate_prorated'] / a.gpus:.3f} ({v['after']['n']} runs, loop {v['after']['loop_median']}); ratio {v['ratio_prorated']}; "
              f"after by kind: " + ', '.join(f"{k} {x['rate_prorated'] / a.gpus:.3f} ({x['n']} runs, loop {x['loop_median']})" for k, x in v['after_by_kind'].items()))
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
