#!/usr/bin/env python3
"""Declared group subsets of the arena study (arena_gator_20260925, E1b; PLAN section 7 items 1, 2 and 4).

Every subset is a pure function of the group ids: groups are ordered by md5(group id) as a hex string, lowest first,
and the first N are taken. Nothing about outcomes, picks or geometry is used (the f104 subset filters on the case's
evaluation_stratum first, as PLAN 7.4 asks for hill/crater groups).

  soil_unseen_subset.json   the 125 lowest-md5 groups of each of the 8 unseen test arenas (4 near + 4 spread)
  f104_indist_200.json      the 200 lowest-md5 hill/crater groups of f104_pair_group_* (the fresh 600 of the
                            800-group f104 suite, generalist_20260921/cases/pair_v1/cases)
  spread_headroom_groups.json  the 75 lowest-md5 groups of each spread arena's suite (the soil headroom scan)

  PYTHONPATH=src:scripts python scripts/ag_declared_subsets.py
"""
from __future__ import annotations

import argparse, hashlib, json, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
NEAR = ['g260', 'g271', 'g251', 'g247']
SPREAD = ['g258', 'g268', 'g263', 'g241']
PAIR = ROOT / 'artifacts/traverse/generalist_20260921/cases/pair_v1/cases'
SUITE800 = ROOT / 'artifacts/traverse/generalist_20260921/A_adapt/suite/cases'
FEATURE_STRATA = ('hill_cross_slope', 'hill_entry_cross_exit', 'crater_cross_slope', 'crater_entry_cross_exit')


def md5(s):
    return hashlib.md5(s.encode()).hexdigest()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def lowest(ids, n):
    out = sorted(ids, key=md5)[:n]
    assert len(out) == n, (len(ids), n)
    return out


def suite_records(arena):
    d = K3 / f'cases/test_{arena}/cases'
    return d, {r['scene_id']: r for r in json.loads((d / 'cases.json').read_text())['records']}


def feature_index(case_dir, g):
    return json.loads((case_dir / 'routes' / g / 'route_00.json').read_text())['meta'].get('feature_index')


def arena_block(arena, n):
    d, recs = suite_records(arena)
    lock = (K3 / f'suites/test_{arena}.SUITE_LOCKED.sha256').read_text().split()[0]
    groups = lowest(list(recs), n)
    feats = Counter(feature_index(d, g) for g in groups)
    return dict(arena=arena, cases_dir=rel(d), suite_lock_sha256=lock, n_suite=len(recs), n=len(groups),
                strata=dict(Counter(recs[g]['evaluation_stratum'] for g in groups)),
                features_covered=len(feats), groups_per_feature={str(k): v for k, v in sorted(feats.items())},
                groups=sorted(groups), md5_cut=md5(sorted(groups, key=md5)[-1]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-suites', type=Path, default=K3 / 'suites')
    ap.add_argument('--out-headroom', type=Path, default=K3 / 'e3/spread_headroom')
    a = ap.parse_args()
    stamp = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), script='scripts/ag_declared_subsets.py', script_sha256=sha(__file__),
                 order='md5(group id) as a lowercase hex string, lowest first')

    soil = dict(schema='ag_soil_unseen_subset_v1', **stamp,
                rule='PLAN 7.1: the 125 lowest-md5 groups of each of the 8 unseen test arenas (near g260 g271 g251 g247; '
                     'spread g258 g268 g263 g241); soil drives these 1,000 groups, rigid drives all 2,000',
                arenas={})
    for arena in NEAR + SPREAD:
        soil['arenas'][arena] = dict(arena_block(arena, 125), family='near' if arena in NEAR else 'spread')
    soil['n_groups'] = sum(v['n'] for v in soil['arenas'].values())
    soil['groups'] = sorted(g for v in soil['arenas'].values() for g in v['groups'])
    assert len(set(soil['groups'])) == soil['n_groups'] == 1000

    recs = {r['scene_id']: r for r in json.loads((PAIR / 'cases.json').read_text())['records']}
    assert len(recs) == 600 and all(g.startswith('f104_pair_group_') for g in recs)
    feat = [g for g, r in recs.items() if r['evaluation_stratum'] in FEATURE_STRATA]
    pick = lowest(feat, 200)
    same = all(sha(PAIR / f'{g}.json') == sha(SUITE800 / f'{g}.json') for g in pick)
    same_routes = all(sha(PAIR / 'routes' / g / 'route_00.json') == sha(SUITE800 / 'routes' / g / 'route_00.json')
                      for g in pick)   # the 800-suite copy holds route_00 only
    f104 = dict(schema='ag_f104_indist_200_v1', **stamp,
                rule='PLAN 7.4: the 200 lowest-md5 hill/crater groups (evaluation_stratum in ' + ', '.join(FEATURE_STRATA)
                     + ') of the 600 f104_pair_group_* groups; not the f104_crm_eval groups CEM was tuned on',
                source_cases=rel(PAIR), source_cases_json_sha256=sha(PAIR / 'cases.json'),
                suite800_cases=rel(SUITE800), case_files_identical_in_suite800=same, route00_identical_in_suite800=same_routes,
                n_pool=len(feat), pool_strata=dict(Counter(recs[g]['evaluation_stratum'] for g in feat)), n=len(pick),
                strata=dict(Counter(recs[g]['evaluation_stratum'] for g in pick)),
                groups_per_feature={str(k): v for k, v in sorted(Counter(feature_index(PAIR, g) for g in pick).items())},
                groups=sorted(pick))

    head = dict(schema='ag_spread_headroom_groups_v1', **stamp,
                rule='PLAN 7.2: the 75 lowest-md5 groups of each spread arena suite; frozen f104-only soil specialist '
                     '(CEM 4x64, ga_planner --world crm --arms B) and straight 6 m/s, as the g217 dev check',
                arenas={arena: arena_block(arena, 75) for arena in SPREAD})
    head['n_groups'] = sum(v['n'] for v in head['arenas'].values())

    # nesting (a consequence of the same order, asserted): the 75 headroom groups are inside the 125 soil groups
    for arena in SPREAD:
        assert set(head['arenas'][arena]['groups']) <= set(soil['arenas'][arena]['groups'])
    a.out_suites.mkdir(parents=True, exist_ok=True); a.out_headroom.mkdir(parents=True, exist_ok=True)
    for obj, p in ((soil, a.out_suites / 'soil_unseen_subset.json'), (f104, a.out_suites / 'f104_indist_200.json'),
                   (head, a.out_headroom / 'spread_headroom_groups.json')):
        p.write_text(json.dumps(obj, indent=1) + '\n')
        print('wrote', rel(p), sha(p)[:16])
    for arena, v in soil['arenas'].items():
        print(f"soil subset {arena} ({v['family']}): {v['n']} groups, strata {v['strata']}, features {v['features_covered']}")
    print('f104 in-distribution:', f104['n'], f104['strata'], 'identical in suite800:', same, same_routes)
    for arena, v in head['arenas'].items():
        print(f"headroom {arena}: {v['n']} groups, strata {v['strata']}, features {v['features_covered']}")


if __name__ == '__main__':
    main()
