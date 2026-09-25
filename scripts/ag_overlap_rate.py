#!/usr/bin/env python3
"""Soil episode rate on chosen nodes before / after rigid pool steps started there (arena_gator_20260925, E3b2;
PLAN 7.8: rigid work may share a soil node only if soil slows by < 10 %).

For every finished soil run in <out>/runs (outcome.json present; host from collection_request.json), the episode's
loop cost = 1 / crm.rtf_sim_over_wall (wall s per simulated s of the simulation loop) and its whole cost = wall_s /
elapsed_s; the run's end = outcome.json mtime, its start = end - wall_s. Runs are split per host at --t0 (epoch s or
HH:MM today): 'before' = ended before t0 (the last --window s), 'after' = started after t0. Reports the per-host
medians, the ratio after/before, and the same for the control hosts (all other hosts with runs in both windows), so a
change of the episode mix over time is visible. Python 3.9, standard library only (login node).
  python3 ag_overlap_rate.py --out $G3/soil_v1 --hosts k006-004-v4 --t0 05:02 [--window 3600] [--json f]
"""
import argparse, json, os, statistics, time
from pathlib import Path


def t_of(s):
    if ':' in s:
        h, m = s.split(':')[:2]
        lt = time.localtime()
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, int(h), int(m), 0, 0, 0, -1))
    return float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--hosts', nargs='+', required=True, help='short host names with rigid steps (test nodes)')
    ap.add_argument('--t0', required=True)
    ap.add_argument('--window', type=float, default=3600)
    ap.add_argument('--prefix', default='', help='only run ids with this prefix')
    ap.add_argument('--json')
    a = ap.parse_args()
    t0 = t_of(a.t0)
    runs = Path(a.out) / 'runs'
    per = {}
    for d in os.scandir(runs):
        if not d.name.startswith(a.prefix):
            continue
        o = Path(d.path) / 'outcome.json'
        try:
            end = o.stat().st_mtime
        except FileNotFoundError:
            continue
        if end < t0 - a.window - 3600:
            continue
        try:
            oj = json.load(open(o)); rq = json.load(open(Path(d.path) / 'collection_request.json'))
            loop = 1.0 / float(oj['crm']['rtf_sim_over_wall']); whole = float(oj['wall_s']) / float(oj['elapsed_s'])
        except Exception:
            continue
        host = str(rq.get('host', '?')).split('.')[0]
        start = end - float(oj['wall_s'])
        side = 'before' if (end <= t0 and end >= t0 - a.window) else ('after' if start >= t0 else None)
        if side:
            per.setdefault(host, {'before': [], 'after': []})[side].append((loop, whole, d.name))
    res = {'t0': time.strftime('%H:%M:%S', time.localtime(t0)), 'window_s': a.window, 'hosts': {}}

    def summ(v):
        return dict(n=len(v), loop_med=statistics.median([x[0] for x in v]) if v else None,
                    whole_med=statistics.median([x[1] for x in v]) if v else None)
    ctrl_ratios = []
    for h, v in sorted(per.items()):
        b, af = summ(v['before']), summ(v['after'])
        ratio = af['loop_med'] / b['loop_med'] if (b['n'] and af['n']) else None
        res['hosts'][h] = dict(test=h in a.hosts, before=b, after=af, loop_ratio=ratio)
        if h not in a.hosts and ratio is not None:
            ctrl_ratios.append(ratio)
    res['control_loop_ratio_median'] = statistics.median(ctrl_ratios) if ctrl_ratios else None
    for h in a.hosts:
        r = res['hosts'].get(h, {}).get('loop_ratio')
        c = res['control_loop_ratio_median']
        res['hosts'].setdefault(h, {})['slowdown_vs_control'] = (r / c - 1) if (r and c) else None
    for h, v in res['hosts'].items():
        if not v.get('before'):
            continue
        print(f"{'TEST ' if v['test'] else 'ctrl '}{h:14s} before n={v['before']['n']:3d} loop {v['before']['loop_med']}  "
              f"after n={v['after']['n']:3d} loop {v['after']['loop_med']}  ratio {v['loop_ratio']}"
              + (f"  slowdown vs control {v['slowdown_vs_control']}" if v['test'] else ''))
    print('control median ratio', res['control_loop_ratio_median'])
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
