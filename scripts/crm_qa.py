"""Validate collected CRM episodes and count validated driving hours.

An episode counts iff it has the atomic completion marker, a passed launch check, consistent finite arrays, and no
invalid-physics flag. Flags (episode moved aside with --quarantine, never labelled):
  explosion        body speed > 15 m/s or chassis more than 2.5 m above the BMP surface (SPH blow-up / launch)
  unstalled_break  soil_breakthrough NOT preceded by >= 1 s of effortful near-stop or crawling (< 1 m/s at throttle
                   > 0.3) in the last 6 s - i.e. not the known "stalled, dug in" mechanism but a wheel punched through
                   the soil floor at speed. Rollovers at speed are kept (they can be real) and only counted.
numpy only; runs on the cluster login/compute nodes.

  python crm_qa.py <collect dir> [--quarantine] [--workers 16]
"""
import argparse, glob, json, os, sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np

DT = 0.05


def check(d):
    rid = os.path.basename(d)
    try:
        o = json.load(open(d + '/outcome.json')); z = np.load(d + '/trajectory.npz'); e = np.load(d + '/crm_extra.npz')
        c = json.load(open(d + '/case.json')); launch = json.load(open(d + '/initial_state_validation.json'))
        st, ac, po = z['state'], z['action'], z['pose']
    except Exception as exc:
        return dict(id=rid, ok=False, flag='unreadable', detail=str(exc))
    n = len(st)
    row = dict(id=rid, group=c['id'], split=c['split'], status=o['status'], elapsed_s=n * DT, frames=n,
               kind='on_policy' if '_op_' in rid else 'designed', rtf=o['crm']['rtf_sim_over_wall'],
               host=json.load(open(d + '/collection_request.json')).get('host'))
    if not (st.shape == (n, 17) and ac.shape == (n, 3) and po.shape == (n, 3) and n > 0 and abs(n * DT - o['elapsed_s']) < 1e-6):
        return dict(row, ok=False, flag='shape')
    if not (np.isfinite(st).all() and np.isfinite(ac).all() and np.isfinite(po).all()):
        return dict(row, ok=False, flag='nonfinite')
    if not launch.get('passed', False):
        return dict(row, ok=False, flag='launch')
    speed = np.hypot(st[:, 0], st[:, 1]); height = e['pos_z_m'] - e['bmp_ground_z_m']
    if speed.max() > 15. or height.max() > 2.5:
        return dict(row, ok=False, flag='explosion', detail=f'vmax {speed.max():.1f} hmax {height.max():.2f}')
    if o['status'] in ('soil_breakthrough_terminated', 'rollover'):
        tail = slice(max(0, n - 120), n)
        slow = (np.abs(st[tail, 0]) < 1.0) & (ac[tail, 1] > 0.3)
        run = best = 0
        for s in slow:
            run = run + 1 if s else 0; best = max(best, run)
        row['stalled_before'] = bool(best >= 20)
        # A breakthrough without a preceding stall is a wheel punched through the floor at speed (no rigid support
        # under the soil layer): invalid physics. A rollover at speed is kept: it can be real, and it is only reported.
        if best < 20 and o['status'] == 'soil_breakthrough_terminated':
            return dict(row, ok=False, flag='unstalled_break')
    return dict(row, ok=True, flag=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('--quarantine', action='store_true'); ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    runs = sorted(os.path.dirname(f) for f in glob.glob(a.root + '/runs/*/episode_complete.json'))
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(check, runs, chunksize=32))
    ok = [r for r in rows if r['ok']]; bad = [r for r in rows if not r['ok']]
    hours = lambda rs: sum(r['elapsed_s'] for r in rs) / 3600.
    out = dict(collect_dir=a.root, episodes_with_marker=len(rows), validated_episodes=len(ok), validated_hours=hours(ok),
               flagged=dict(Counter(r['flag'] for r in bad)), flagged_hours=hours([r for r in bad if 'elapsed_s' in r]),
               by_split={s: dict(episodes=sum(r['split'] == s for r in ok), hours=hours([r for r in ok if r['split'] == s]),
                                 groups=len({r['group'] for r in ok if r['split'] == s})) for s in ('train', 'val', 'test')},
               status=dict(Counter(r['status'] for r in ok)),
               fail_rate=float(np.mean([r['status'] != 'goal_reached' for r in ok])) if ok else None,
               by_kind={k: dict(n=sum(r['kind'] == k for r in ok),
                                fail=float(np.mean([r['status'] != 'goal_reached' for r in ok if r['kind'] == k] or [np.nan])))
                        for k in ('designed', 'on_policy')},
               routes_per_group=dict(Counter(Counter(r['group'] for r in ok).values())),
               stalled_before_break=dict(Counter(str(r.get('stalled_before')) for r in ok if r['status'] in ('soil_breakthrough_terminated', 'rollover'))),
               mean_rtf=float(np.mean([r['rtf'] for r in ok])) if ok else None,
               hosts=len({r.get('host') for r in ok}), flagged_ids=[dict(id=r['id'], flag=r['flag'], detail=r.get('detail')) for r in bad])
    json.dump(out, open(a.root + '/qa.json', 'w'), indent=1)
    if a.quarantine and bad:
        os.makedirs(a.root + '/runs_flagged', exist_ok=True)
        os.makedirs(a.root + '/failed', exist_ok=True)
        for r in bad:
            src = f"{a.root}/runs/{r['id']}"; dst = f"{a.root}/runs_flagged/{r['id']}"
            if os.path.isdir(src) and not os.path.exists(dst):
                os.rename(src, dst)
                # retire the task so a late worker never re-collects it
                json.dump({'attempts': 99, 'rc': 'qa_' + str(r['flag'])}, open(f"{a.root}/failed/{r['id']}.json", 'w'))
    print(json.dumps({k: v for k, v in out.items() if k != 'flagged_ids'}, indent=1))
    print('flagged examples', out['flagged_ids'][:8])


if __name__ == '__main__':
    main()
