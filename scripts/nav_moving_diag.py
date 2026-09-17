"""Planning from rest vs replanning while moving, measured on the nav_v1 runs themselves.

The risk model reads geometry and the commanded speed profile; it reads NO vehicle state. A route therefore gets
the same predicted risk whether the vehicle is parked or doing 6 m/s, while the real risk plainly differs. The
labels behind the model also all come from episodes that started at rest. This script asks whether that matters
in practice, using only what the runs already recorded:

  * is the predicted risk still discriminative for decisions made at speed?  (AUC of predicted route logit
    against "a backward slide starts before the next decision + a 2 s tail", split by the speed at the decision)
  * how large is the speed command's jump at a replan, and does a big jump precede trouble?
  * how much does the chosen route change from one decision to the next (route chatter)?
  * how much of the corridor the mask invalidates, as a function of distance to the waypoint.
"""
import argparse, glob, json, os
import numpy as np

DT = 0.05


def auc(scores, labels):
    s = np.asarray(scores, float); y = np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float('nan')
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    for v in np.unique(s):
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--arm', default='R1')
    ap.add_argument('--tail-s', type=float, default=2.0)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    rows, chatter, invalid_by_dist = [], [], []
    runs = sorted(glob.glob(os.path.join(a.root, 'runs', f'*__{a.arm}')))
    for d in runs:
        if not os.path.exists(d + '/mission_outcome.json'):
            continue
        dec = json.load(open(d + '/decisions.json'))
        routes = {r['route_id']: np.asarray(r['waypoints'], float) for r in json.load(open(d + '/routes.json'))}
        z = np.load(d + '/trajectory.npz')
        vx = z['state'][:, 0].astype(float); thr = z['action'][:, 1].astype(float)
        slide = (vx < -0.10) & (thr > 0.3) | (vx < -0.30)
        slide[:20] = False
        n = len(vx)
        dec = [x for x in dec if 'plan_s' in x]
        for i, x in enumerate(dec):
            f0 = x['frame']
            f1 = dec[i + 1]['frame'] if i + 1 < len(dec) else n
            f1 = min(n, f1 + int(round(a.tail_s / DT)))
            if f0 >= n:
                continue
            rows.append(dict(mission=os.path.basename(d).split('__')[0], v_now=x['v_now'],
                             logit=x['logit'], risk=x['risk'], dist=x['dist_to_goal_m'],
                             invalid=x['invalid_fraction_picked'], trigger=x['trigger'],
                             first=bool(i == 0), label=int(slide[f0:f1].any())))
            invalid_by_dist.append((x['dist_to_goal_m'], x['invalid_fraction_picked']))
            if i and x.get('route_id') is not None and dec[i - 1].get('route_id') is not None:
                p, q = routes.get(dec[i - 1]['route_id']), routes.get(x['route_id'])
                if p is not None and q is not None:
                    m = min(len(p), len(q), 60)          # first ~30 m of each route, 0.5 m steps
                    chatter.append(float(np.linalg.norm(p[:m] - q[:m], axis=1).max()))
    if not rows:
        raise SystemExit(f'no {a.arm} runs under {a.root}')
    v = np.array([r['v_now'] for r in rows]); y = np.array([r['label'] for r in rows])
    s = np.array([r['logit'] for r in rows])
    trig = np.array([r['trigger'] for r in rows])
    first = np.array([r['first'] for r in rows])
    rest, moving = v < 0.5, v >= 2.0
    groups = {
        'mission_start': first,
        'waypoint_start': (trig == 'leg') & ~first,
        'periodic_moving': (trig == 'periodic') & (v >= 2.0),
        'periodic_slow': (trig == 'periodic') & (v < 2.0),
    }
    inv = np.array(invalid_by_dist)
    out = {
        'arm': a.arm, 'n_runs': len(runs), 'n_decisions': len(rows), 'tail_s': a.tail_s,
        'slide_follows_rate': float(y.mean()),
        'overall': {'n': len(y), 'events': int(y.sum()), 'auc': auc(s, y)},
        'at_rest': {'n': int(rest.sum()), 'events': int(y[rest].sum()), 'auc': auc(s[rest], y[rest]),
                    'event_rate': float(y[rest].mean()) if rest.any() else float('nan')},
        'moving': {'n': int(moving.sum()), 'events': int(y[moving].sum()), 'auc': auc(s[moving], y[moving]),
                   'event_rate': float(y[moving].mean()) if moving.any() else float('nan')},
        'by_group': {k: {'n': int(m.sum()), 'events': int(y[m].sum()),
                         'event_rate': float(y[m].mean()) if m.any() else float('nan'),
                         'auc': auc(s[m], y[m]) if m.any() else float('nan'),
                         'mean_speed': float(v[m].mean()) if m.any() else float('nan')}
                     for k, m in groups.items()},
        'speed_at_decision': {'mean': float(v.mean()), 'p50': float(np.percentile(v, 50)),
                              'p90': float(np.percentile(v, 90)), 'frac_above_2': float((v >= 2).mean())},
        'route_change_m': ({'mean': float(np.mean(chatter)), 'p50': float(np.percentile(chatter, 50)),
                            'p90': float(np.percentile(chatter, 90)), 'n': len(chatter)} if chatter else {}),
        'invalid_fraction_by_distance': {
            f'{lo}-{hi} m': float(inv[(inv[:, 0] >= lo) & (inv[:, 0] < hi), 1].mean())
            for lo, hi in ((0, 10), (10, 15), (15, 20), (20, 25), (25, 100))
            if ((inv[:, 0] >= lo) & (inv[:, 0] < hi)).any()},
    }
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
