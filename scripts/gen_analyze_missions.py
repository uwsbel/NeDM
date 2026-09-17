"""gen_v1 section B analysis: five-goal missions, three arms (n2, rule, straight6)."""
import json, sys
from math import comb
from pathlib import Path
import numpy as np

G = 'artifacts/traverse/fdm_f104_50h_20260909/gen_v1'
ARMS = ['n2', 'rule', 'straight6']


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    runs = Path(sys.argv[1] if len(sys.argv) > 1 else G + '/missions_run/runs')
    by = {}
    for p in sorted(runs.glob('*/mission_outcome.json')):
        o = json.load(open(p)); by.setdefault(o['mission'], {})[o['arm']] = o
    arena = lambda m: m.split('_mission_')[0]
    full = {m: v for m, v in by.items() if all(a in v for a in ARMS)}
    out = {'missions_with_all_arms': {a: sum(arena(m) == a for m in full) for a in sorted({arena(m) for m in by})}}
    def summ(ms, arm):
        v = [full[m][arm] for m in ms]
        succ = [x for x in v if x['success']]
        return dict(n=len(v), success=float(np.mean([x['success'] for x in v])), goals_reached=float(np.mean([x['goals_reached'] for x in v])),
                    any_slide=float(np.mean([x['any_slide'] for x in v])), tilt_over30=float(np.mean([x['max_tilt_deg'] > 30 for x in v])),
                    median_max_tilt=float(np.median([x['max_tilt_deg'] for x in v])),
                    median_time_successful_s=float(np.median([x['total_time_s'] for x in succ])) if succ else None,
                    no_route=int(sum(x['status'] == 'no_route' for x in v)),
                    failure_modes={s: int(sum(x['status'] == s for x in v)) for s in sorted({x['status'] for x in v}) if s != 'mission_complete'})
    def pair(ms, a, b, key):
        x = np.array([full[m][a][key] for m in ms], int); y = np.array([full[m][b][key] for m in ms], int)
        bb = int(((x == 1) & (y == 0)).sum()); cc = int(((x == 0) & (y == 1)).sum())
        return dict(n=len(ms), a_better=bb, b_better=cc, p=mcnemar(bb, cc))
    sets = {'f104': [m for m in full if arena(m) == 'f104'], 'new_pooled': [m for m in full if arena(m) != 'f104']}
    for a in sorted({arena(m) for m in full} - {'f104'}):
        sets[a] = [m for m in full if arena(m) == a]
    out['summary'] = {s: {arm: summ(ms, arm) for arm in ARMS} for s, ms in sets.items() if ms}
    out['M1_f104_success_n2_vs_straight6'] = pair(sets['f104'], 'n2', 'straight6', 'success')
    out['M2_f104_success_n2_vs_rule'] = pair(sets['f104'], 'n2', 'rule', 'success')
    out['new_success_n2_vs_straight6'] = pair(sets['new_pooled'], 'n2', 'straight6', 'success')
    out['new_success_n2_vs_rule'] = pair(sets['new_pooled'], 'n2', 'rule', 'success')
    for s in ('f104', 'new_pooled'):
        for a, b in (('n2', 'straight6'), ('n2', 'rule')):
            x = np.array([not full[m][a]['any_slide'] for m in sets[s]], int); y = np.array([not full[m][b]['any_slide'] for m in sets[s]], int)
            out[f'{s}_no_slide_{a}_vs_{b}'] = dict(a_better=int(((x == 1) & (y == 0)).sum()), b_better=int(((x == 0) & (y == 1)).sum()),
                                                   p=mcnemar(int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())))
    json.dump(out, open(G + '/mission_results.json', 'w'), indent=1)
    print(json.dumps(out['missions_with_all_arms']))
    for s in ('f104', 'new_pooled'):
        print(f'\n{s}')
        for arm, d in out['summary'][s].items():
            print(f"  {arm:10s} success {100*d['success']:5.1f}%  goals {d['goals_reached']:.2f}/5  slide {100*d['any_slide']:5.1f}%  "
                  f"tilt>30 {100*d['tilt_over30']:5.1f}%  median time {d['median_time_successful_s'] or 0:.0f} s  failures {d['failure_modes']}")
    for k in ('M1_f104_success_n2_vs_straight6', 'M2_f104_success_n2_vs_rule', 'new_success_n2_vs_straight6', 'new_success_n2_vs_rule',
              'f104_no_slide_n2_vs_straight6', 'f104_no_slide_n2_vs_rule', 'new_pooled_no_slide_n2_vs_straight6', 'new_pooled_no_slide_n2_vs_rule'):
        print(k, out[k])


if __name__ == '__main__':
    main()
