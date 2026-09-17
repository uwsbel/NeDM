"""Pick the 5 scenarios to film, from the screened trios.

A scenario qualifies when the three routes tell three different stories in Chrono:
  optimal     reaches the goal, never slides, modest body tilt
  suboptimal  reaches the goal but is clearly worse -- slower, or leaning harder
  risky       stalls or slides backwards
Ranked by how legible the contrast is (does the risky one really stop, how much slower the middle one is),
with a spread penalty so the five are not all in the same corner of the arena.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, 'scripts')
from f104_n2_analyze import labels

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
OUT = ROOT + '/night2_v1/demo_v1'
ROLES = ('optimal', 'suboptimal', 'risky')


def main():
    runs = sys.argv[1] if len(sys.argv) > 1 else OUT + '/runs'
    picks = {p['group']: p for p in json.load(open(OUT + '/picks.json'))}
    by = {}
    for g, p in picks.items():
        v = {}
        for r in ROLES:
            d = f'{runs}/{g}__{r}'
            if os.path.exists(d + '/outcome.json') and os.path.exists(d + '/trajectory.npz'):
                v[r] = dict(labels(d), **p['roles'][r])
        if len(v) == 3:
            by[g] = v
    print(f'{len(by)}/{len(picks)} groups with all three routes driven\n')
    rows = []
    for g, v in by.items():
        o, s, b = v['optimal'], v['suboptimal'], v['risky']
        ok = (o['unsafe'] == 0 and b['unsafe'] == 1 and s['fail'] == 0)
        worse = max(s['elapsed'] - o['elapsed'], (s['max_tilt'] - o['max_tilt']) * 1.5)
        score = (3.0 * b['fail'] + 1.5 * min(b['back_s'], 4.0) + min(worse, 12.0) / 3.0
                 + 0.5 * min(s['unsafe'], 1))
        rows.append(dict(group=g, ok=bool(ok), score=float(score), worse=float(worse),
                         **{r: {k: v[r][k] for k in ('status', 'elapsed', 'max_tilt', 'back_s', 'min_vx',
                                                     'risk', 'unsafe', 'fail', 'detour_m', 'mean_speed')}
                            for r in ROLES}))
    rows.sort(key=lambda r: (-r['ok'], -r['score']))
    for r in rows:
        o, s, b = r['optimal'], r['suboptimal'], r['risky']
        print(f"{'OK ' if r['ok'] else '   '}{r['group']}  score {r['score']:5.2f} | "
              f"opt {o['status'][:12]:12s} {o['elapsed']:5.1f}s tilt {o['max_tilt']:4.1f} | "
              f"sub {s['status'][:12]:12s} {s['elapsed']:5.1f}s tilt {s['max_tilt']:4.1f} | "
              f"bad {b['status'][:12]:12s} {b['elapsed']:5.1f}s back {b['back_s']:4.1f}s")
    # spread: keep at most one scenario per 12 m of start position
    cases = {g: json.load(open(f"{ROOT}/cases_haz_final/{g}.json")) for g in by}
    chosen, used = [], []
    for r in rows:
        if not r['ok']:
            continue
        p = np.asarray(cases[r['group']]['layout']['start_xy'], float)
        if any(np.linalg.norm(p - q) < 12.0 for q in used):
            continue
        used.append(p); chosen.append(r['group'])
        if len(chosen) == 5:
            break
    json.dump(dict(chosen=chosen, rows=rows), open(OUT + '/selection.json', 'w'), indent=1)
    print('\nchosen:', chosen)


if __name__ == '__main__':
    main()
