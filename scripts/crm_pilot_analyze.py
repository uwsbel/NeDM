"""CRM calibration pilot: outcome mix per configuration, paired against each other and against the rigid outcome of
the SAME episode id (night-2 routes were all driven on rigid ground; labels from station_ds_all).

  python scripts/crm_pilot_analyze.py <pilot dir with A/ B/ C/ subdirs of runs/>
"""
import glob, json, os, sys
from collections import Counter
import numpy as np

sys.path.insert(0, 'scripts')
from f104_n2_analyze import labels  # the single label function of every closed-loop test

PILOT = sys.argv[1]
RIGID = np.load('artifacts/traverse/crm_f104_v1/rigid_labels_station_ds_all.npz', allow_pickle=True)
rigid = {i: (int(u), int(f)) for i, u, f in zip(RIGID['id'], RIGID['unsafe'], RIGID['fail'])}


def kind(rid):
    if '_op_' in rid:
        return 'on_policy'
    return ('constant_2', 'constant_4', 'constant_6', 'smooth_2_6_2')[int(rid.split('_route_')[1]) % 4]


res = {}
for cfg in sorted(os.listdir(PILOT)):
    runs = sorted(glob.glob(f'{PILOT}/{cfg}/runs/*/episode_complete.json'))
    if not runs:
        continue
    lab = {}
    for f in runs:
        d = os.path.dirname(f)
        try:
            l = labels(d)
        except ValueError:
            continue
        o = json.load(open(d + '/outcome.json'))
        l.update(rtf=o['crm']['rtf_sim_over_wall'], sink=o['crm']['mean_spindle_height_above_bmp_m'],
                 slip95=o['crm']['max_abs_slip_ratio_p95'])
        lab[os.path.basename(d)] = l
    res[cfg] = lab
    n = len(lab)
    print(f'\n== {cfg}: {n} episodes, {sum(l["elapsed"] for l in lab.values()) / 3600:.3f} sim-h, '
          f'rtf {np.mean([l["rtf"] for l in lab.values()]):.3f}')
    print('  status', dict(Counter(l['status'] for l in lab.values())))
    print(f'  fail {np.mean([l["fail"] for l in lab.values()]):.3f}  unsafe {np.mean([l["unsafe"] for l in lab.values()]):.3f}'
          f'  tilt>30 {np.mean([l["max_tilt"] > 30 for l in lab.values()]):.3f}'
          f'  mean elapsed {np.mean([l["elapsed"] for l in lab.values()]):.1f} s')
    both = [i for i in lab if i in rigid]
    print(f'  rigid on the same {len(both)} ids: fail {np.mean([rigid[i][1] for i in both]):.3f} unsafe {np.mean([rigid[i][0] for i in both]):.3f}')
    for k in ('constant_2', 'constant_4', 'constant_6', 'smooth_2_6_2', 'on_policy'):
        ids = [i for i in lab if kind(i) == k]
        if ids:
            print(f'    {k:13s} n={len(ids):3d}  CRM fail {np.mean([lab[i]["fail"] for i in ids]):.2f} unsafe {np.mean([lab[i]["unsafe"] for i in ids]):.2f}'
                  f' | rigid fail {np.mean([rigid[i][1] for i in ids if i in rigid]):.2f} unsafe {np.mean([rigid[i][0] for i in ids if i in rigid]):.2f}'
                  f' | time CRM ok {np.median([lab[i]["elapsed"] for i in ids if not lab[i]["fail"]] or [np.nan]):.1f} s')

cfgs = list(res)
for a in range(len(cfgs)):
    for b in range(a + 1, len(cfgs)):
        A, B = res[cfgs[a]], res[cfgs[b]]
        ids = sorted(set(A) & set(B))
        if not ids:
            continue
        same_status = np.mean([A[i]['status'] == B[i]['status'] for i in ids])
        same_unsafe = np.mean([A[i]['unsafe'] == B[i]['unsafe'] for i in ids])
        ok = [i for i in ids if not A[i]['fail'] and not B[i]['fail']]
        dt = [B[i]['elapsed'] - A[i]['elapsed'] for i in ok]
        print(f'\n{cfgs[a]} vs {cfgs[b]}: {len(ids)} shared; same status {same_status:.3f}; same unsafe {same_unsafe:.3f}; '
              f'fail {np.mean([A[i]["fail"] for i in ids]):.3f} vs {np.mean([B[i]["fail"] for i in ids]):.3f}; '
              f'unsafe {np.mean([A[i]["unsafe"] for i in ids]):.3f} vs {np.mean([B[i]["unsafe"] for i in ids]):.3f}; '
              f'time diff on joint successes median {np.median(dt) if dt else float("nan"):+.2f} s (n={len(ok)})')
        for i in ids:
            if A[i]['status'] != B[i]['status']:
                print(f'   {i}: {A[i]["status"]} ({A[i]["elapsed"]:.1f} s) vs {B[i]["status"]} ({B[i]["elapsed"]:.1f} s)')
json.dump(res, open(f'{PILOT}/pilot_labels.json', 'w'), indent=1)
