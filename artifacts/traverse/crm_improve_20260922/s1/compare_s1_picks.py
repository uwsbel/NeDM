#!/usr/bin/env python3
"""S1 pick sets vs the K1 A5 picks of the same models: speed step at the branch, start heading, predicted risk.

speed step = picked route's first speed - vehicle forward speed at frame 60 (history window's last valid vx, equal to
pass1_state.json vx); start heading = |wrap(route headings[0] - vehicle yaw at frame 60)|. Pick-time quantities only
(no driving). Writes s1/compare_s1_picks.json and prints a table.

  PYTHONPATH=src:scripts $PY artifacts/traverse/crm_improve_20260922/s1/compare_s1_picks.py
"""
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
K1 = ROOT / 'artifacts/traverse/generalist_20260921/A_adapt'
S1 = Path(__file__).resolve().parent
POSES = json.load(open(K1 / 'a5/poses_crm.json'))
P1 = json.load(open(K1 / 'a5/pass1_state.json'))['worlds']['crm']['groups']
SETS = {'K1 H free': K1 / 'a5/picks_crm_H', 'S1 H cont': S1 / 'picks_crm_H_cont', 'S1 H cont_head': S1 / 'picks_crm_H_conthead',
        'K1 Spcrm free': K1 / 'a5/picks_crm_Spcrm', 'S1 Spcrm cont_head': S1 / 'picks_crm_Spcrm_conthead'}
PAIRS = [('S1 H cont', 'K1 H free'), ('S1 H cont_head', 'K1 H free'), ('S1 H cont_head', 'S1 H cont'), ('S1 Spcrm cont_head', 'K1 Spcrm free')]
STEP_BINS = [(-np.inf, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, np.inf)]
HEAD_BINS = [(0, 5), (5, 15), (15, 30), (30, np.inf)]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def load(d):
    rows = {}
    for g in sorted(POSES):
        pk = json.load(open(d / 'picks' / f'{g}.json'))['arms']['B']
        r = json.load(open(d / 'routes' / f"{pk['route_id']}.json"))
        v = np.asarray(r['speeds'], float); st = np.asarray(r['stations'], float)
        acc = np.diff(v ** 2) / (2 * np.maximum(np.diff(st), 1e-8))
        vx = float(np.load(POSES[g]['history'])['hist'][-1, 0]); yaw = float(POSES[g]['pose'][2])
        assert abs(vx - P1[g]['vx']) == 0.0
        rows[g] = dict(v0=vx, start_speed=float(v[0]), step=float(v[0] - vx), head=abs(math.degrees(wrap(float(r['headings'][0]) - yaw))),
                       z=pk['z_mean'], P=pk['P'], T=pk['T'], mean_speed=pk['mean_speed'], max_lat=pk['max_lateral_m'], sha=pk['route_sha256'],
                       kind=pk['kind'], acc_max=float(acc.max()), acc_min=float(acc.min()), n_eval=pk['n_evaluated'], tries=pk['tries'])
    return rows


def binned(x, bins):
    x = np.asarray(x, float); n = len(x)
    return {f'{lo:g}..{hi:g}': [int(((x >= lo) & (x < hi)).sum()), round(100.0 * ((x >= lo) & (x < hi)).mean(), 1)] for lo, hi in bins}


def summary(rows):
    c = lambda k: np.array([r[k] for r in rows.values()], float)
    step, head = c('step'), c('head')
    return dict(n=len(rows), speed_step_mean=float(step.mean()), speed_step_median=float(np.median(step)),
                speed_step_abs_p95=float(np.quantile(np.abs(step), 0.95)), speed_step_abs_max=float(np.abs(step).max()),
                speed_step_bins=binned(step, STEP_BINS), start_heading_mean=float(head.mean()), start_heading_median=float(np.median(head)),
                start_heading_p95=float(np.quantile(head, 0.95)), start_heading_max=float(head.max()), start_heading_bins=binned(head, HEAD_BINS),
                z_mean=float(c('z').mean()), z_median=float(np.median(c('z'))), P_mean=float(c('P').mean()), T_mean=float(c('T').mean()),
                mean_speed=float(c('mean_speed').mean()), max_lateral_mean=float(c('max_lat').mean()), accel_max=float(c('acc_max').max()),
                accel_min=float(c('acc_min').min()), tries_mean=float(c('tries').mean()),
                pick_kind={k: int(sum(1 for r in rows.values() if r['kind'] == k)) for k in ('anchor', 'sample', 'mean')})


def main():
    data = {k: load(d) for k, d in SETS.items()}
    out = dict(sets={k: str(d.relative_to(ROOT)) for k, d in SETS.items()}, summary={k: summary(v) for k, v in data.items()}, pairs={})
    for a, b in PAIRS:
        A, B = data[a], data[b]; gs = sorted(A)
        dz = np.array([A[g]['z'] - B[g]['z'] for g in gs]); dT = np.array([A[g]['T'] - B[g]['T'] for g in gs])
        out['pairs'][f'{a} vs {b}'] = dict(same_route=int(sum(A[g]['sha'] == B[g]['sha'] for g in gs)), dz_mean=float(dz.mean()),
                                           dz_median=float(np.median(dz)), frac_higher_risk=float((dz > 1e-9).mean()),
                                           dP_mean=float(np.mean([A[g]['P'] - B[g]['P'] for g in gs])), dT_mean=float(dT.mean()))
    json.dump(out, open(S1 / 'compare_s1_picks.json', 'w'), indent=1)
    print(f"{'set':22s} {'step mean':>9s} {'|step| p95':>10s} {'|step| max':>10s} {'head med':>8s} {'head p95':>8s} {'head max':>8s} {'z mean':>7s} {'P mean':>8s} {'T mean':>6s} {'v mean':>6s}")
    for k, s in out['summary'].items():
        print(f"{k:22s} {s['speed_step_mean']:+9.3f} {s['speed_step_abs_p95']:10.3f} {s['speed_step_abs_max']:10.3f} {s['start_heading_median']:8.1f} "
              f"{s['start_heading_p95']:8.1f} {s['start_heading_max']:8.1f} {s['z_mean']:7.2f} {s['P_mean']:8.5f} {s['T_mean']:6.1f} {s['mean_speed']:6.2f}")
    for k, s in out['summary'].items():
        print(f"{k:22s} step bins {s['speed_step_bins']}\n{'':22s} heading bins {s['start_heading_bins']}  kinds {s['pick_kind']}")
    for k, p in out['pairs'].items():
        print(k, p)


if __name__ == '__main__':
    main()
