"""nav_v1 read-out: completion, unsafe events, travel time, and the sensing-to-planning latency budget."""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

DEV = {'f104', 'g228', 'g203', 'g217', 'g216', 'g231'}


def arena_tag(mission_id):
    return mission_id.split('_nav_')[0]


def load(root):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, 'runs', '*', 'mission_outcome.json'))):
        o = json.load(open(p))
        d = os.path.dirname(p)
        arm = os.path.basename(d).split('__')[1]
        mid = os.path.basename(d).split('__')[0]
        dec = json.load(open(d + '/decisions.json')) if os.path.exists(d + '/decisions.json') else []
        o.update(arm=arm, mission_id=mid, arena=arena_tag(mid), dir=d,
                 unseen=arena_tag(mid) not in DEV, decisions_raw=dec)
        rows.append(o)
    return rows


def boot_paired(a, b, n=20000, seed=0):
    """Bootstrap CI for mean(a - b) over paired missions."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    if len(d) == 0:
        return float('nan'), (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(1)
    return float(d.mean()), (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def summarise(rows, key=lambda r: True):
    sel = [r for r in rows if key(r)]
    if not sel:
        return {}
    return {
        'n': len(sel),
        'complete': sum(r['success'] for r in sel),
        'complete_rate': float(np.mean([r['success'] for r in sel])),
        'waypoints_reached': float(np.mean([r['goals_reached'] for r in sel])),
        'waypoint_rate': float(np.sum([r['goals_reached'] for r in sel]) / np.sum([r['n_goals'] for r in sel])),
        'unsafe': sum(bool(r['any_slide']) or not r['success'] for r in sel),
        'any_slide': sum(bool(r['any_slide']) for r in sel),
        'median_time_s': float(np.median([r['total_time_s'] for r in sel])),
        'median_distance_m': float(np.median([r['distance_m'] for r in sel])),
        'max_tilt_deg_p90': float(np.percentile([r['max_tilt_deg'] for r in sel], 90)),
        'decisions_mean': float(np.mean([r['n_decisions'] for r in sel])),
        'lat_p50_s': float(np.median([r['sense_to_plan_s']['p50'] for r in sel])),
        'statuses': dict(sorted(defaultdict(int, {s: sum(1 for r in sel if r['status'] == s)
                                                  for s in {r['status'] for r in sel}}).items())),
        # leg level: 30 missions is a thin sample, ~200 legs is less thin
        'legs': {
            'n': sum(len(r['legs']) for r in sel),
            'reached': sum(1 for r in sel for L in r['legs'] if L.get('status') == 'goal_reached'),
            'with_slide': sum(1 for r in sel for L in r['legs'] if L.get('slid')),
            'median_time_s': float(np.median([L['elapsed_s'] for r in sel for L in r['legs']
                                              if 'elapsed_s' in L])) if sel else float('nan'),
            'backward_s_total': float(sum(L.get('back_s', 0.) for r in sel for L in r['legs'])),
            'tilt_over_30': sum(1 for r in sel for L in r['legs'] if L.get('max_tilt_deg', 0) > 30),
        },
    }


def latency_budget(rows):
    stages = ['render_s', 'backproject_s', 'propose_s', 'speed_clip_s', 'mask_s', 'corridor_s', 'score_s']
    acc = defaultdict(list); tot = []
    for r in rows:
        for d in r['decisions_raw']:
            if 'plan_s' not in d:
                continue
            acc['render_s'].append(d['render_s']); acc['backproject_s'].append(d['backproject_s'])
            for k, v in d.get('timing', {}).items():
                acc[k].append(v)
            tot.append(d['sense_to_plan_s'])
    out = {k: {'mean': float(np.mean(v)), 'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95))}
           for k, v in acc.items() if v}
    if tot:
        out['sense_to_plan_s'] = {'mean': float(np.mean(tot)), 'p50': float(np.percentile(tot, 50)),
                                  'p95': float(np.percentile(tot, 95)), 'max': float(np.max(tot)),
                                  'n': len(tot), 'implied_hz_p50': float(1.0 / np.percentile(tot, 50))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--reference', default='W')
    a = ap.parse_args()
    rows = load(a.root)
    arms = sorted({r['arm'] for r in rows})
    res = {'n_runs': len(rows), 'arms': arms, 'by_arm': {}, 'by_arm_dev': {}, 'by_arm_unseen': {},
           'latency': {}, 'paired_vs_reference': {}}
    for arm in arms:
        sel = [r for r in rows if r['arm'] == arm]
        res['by_arm'][arm] = summarise(sel)
        res['by_arm_dev'][arm] = summarise(sel, lambda r: not r['unseen'])
        res['by_arm_unseen'][arm] = summarise(sel, lambda r: r['unseen'])
        res['latency'][arm] = latency_budget(sel)
    ref = {r['mission_id']: r for r in rows if r['arm'] == a.reference}
    for arm in arms:
        if arm == a.reference:
            continue
        cur = {r['mission_id']: r for r in rows if r['arm'] == arm}
        common = sorted(set(ref) & set(cur))
        if not common:
            continue
        t_m, t_ci = boot_paired([cur[m]['total_time_s'] for m in common], [ref[m]['total_time_s'] for m in common])
        g_m, g_ci = boot_paired([cur[m]['goals_reached'] for m in common], [ref[m]['goals_reached'] for m in common])
        both = [(bool(cur[m]['success']), bool(ref[m]['success'])) for m in common]
        res['paired_vs_reference'][arm] = {
            'n_paired': len(common),
            'time_mean_diff_s': t_m, 'time_ci': t_ci,
            'waypoints_mean_diff': g_m, 'waypoints_ci': g_ci,
            'complete_arm_only': sum(1 for x, y in both if x and not y),
            'complete_ref_only': sum(1 for x, y in both if y and not x),
            'unsafe_arm': sum(1 for m in common if cur[m]['any_slide'] or not cur[m]['success']),
            'unsafe_ref': sum(1 for m in common if ref[m]['any_slide'] or not ref[m]['success']),
        }
    # missions where the arms actually disagree: the candidates for a video and for the narrative
    div = []
    for arm in arms:
        if arm == a.reference:
            continue
        cur = {r['mission_id']: r for r in rows if r['arm'] == arm}
        for m in sorted(set(ref) & set(cur)):
            x, y = cur[m], ref[m]
            if x['success'] != y['success'] or abs(x['total_time_s'] - y['total_time_s']) > 15 \
                    or x['any_slide'] != y['any_slide']:
                div.append({'mission': m, 'arm': arm, 'arena': x['arena'], 'unseen': x['unseen'],
                            'arm_status': x['status'], 'ref_status': y['status'],
                            'arm_goals': x['goals_reached'], 'ref_goals': y['goals_reached'],
                            'arm_time_s': x['total_time_s'], 'ref_time_s': y['total_time_s'],
                            'arm_slide': bool(x['any_slide']), 'ref_slide': bool(y['any_slide'])})
    res['divergent_missions'] = div
    json.dump(res, open(a.out, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('n_runs', 'arms')}, indent=1))
    for arm in arms:
        s = res['by_arm'][arm]
        print(f"{arm:5s} n={s['n']:3d} complete {s['complete']:3d}/{s['n']:<3d} wp {s['waypoint_rate']*100:5.1f}% "
              f"unsafe {s['unsafe']:3d} t50 {s['median_time_s']:6.1f}s dec {s['decisions_mean']:5.1f} "
              f"lat50 {s['lat_p50_s']:.2f}s")
    print('wrote', a.out)


if __name__ == '__main__':
    main()
