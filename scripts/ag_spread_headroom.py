#!/usr/bin/env python3
"""Spread-arena soil headroom scan, pick side (arena_gator_20260925, E1b; PLAN section 7 item 2).

The same two arms as the g217 dev check (scripts/ag_dev_headroom.py), on the 75 lowest-md5 groups of each spread
arena's test suite (e3/spread_headroom/spread_headroom_groups.json):
  Scrm:B     frozen K1 soil specialist (crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt, hashes in
             e3/dev_headroom/models.sha256), CEM 4 x 64 from the case pose at rest: ga_planner.py --world crm --arms B
             --ref-picks /nonexistent_no_reference, run separately per arena with --groups @groups_<a>.txt (per-group
             rng = seed(group, tag), so a subset gives the same picks as the full suite); its output is --picks-root/
             picks_crm_Scrm_<a>
  straight6  the designed anchor with lateral offset 0 at 6 m/s from the night-2 proposal pool, built with exactly
             the code of ag_dev_headroom.py (gen_planner.proposal_pool, rng md5(group + 'crm_proposal')[:8],
             anchor_index(0, 6)); --selfcheck-dev rebuilds the g217 dev straight6 routes and asserts they are
             byte-identical to e3/dev_headroom/straight6/routes before anything is written
Identical routes are driven once (row 'arms' lists both). Rows: soil_v1.json format (as tasks_dev.json), tier -1,
kind 'spread_headroom', vehicle 'hmmwv', ids <g>__Scrm_B / <g>__straight6, episode_seed md5(id)[:8], paths relative to
G3. Ids and episode seeds are asserted unique against --against (the current soil task file). Lock
SPREAD_PICKS_LOCKED.sha256 = one sha256 over name + content of every route file of both arms (sorted by name), then
one line per file (paths relative to K3), written before any drive.
"""
import argparse, hashlib, json, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_map_check
import ag_tasklib as L


def straight6(P, IT, case_path):
    """(pose, route or None, index) exactly as ag_dev_headroom.py builds the straight 6 m/s anchor."""
    case = json.load(open(case_path)); g = case['id']; lay = case['layout']
    pose = [lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']]
    base = {k: np.asarray(v, float) for k, v in json.load(open(Path(case_path).parent / 'routes' / g / 'route_00.json')).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    base['meta'] = {}
    c1, tries = P.proposal_pool(base, pose, np.random.default_rng(int(hashlib.md5((g + 'crm_proposal').encode()).hexdigest()[:8], 16)))
    i6 = P.anchor_index(c1, 0.0, 6.0)
    return g, (None if i6 is None else c1[i6]), i6


def route_dump(r, g, i6):
    return {'waypoints': np.asarray(r['waypoints']).tolist(), 'speeds': np.asarray(r['speeds']).tolist(),
            'stations': np.asarray(r['stations']).tolist(), 'headings': np.asarray(r['headings']).tolist(),
            'meta': {'candidate': 'crm_eval_straight6', 'scene_id': g, 'pool': 'proposal', 'cand_index': i6}}


def selfcheck_dev(P, IT):
    """Rebuild the g217 dev straight6 routes; every one must equal E3a's file byte for byte."""
    dev = L.K3 / 'e3/dev_headroom/straight6/routes'
    P.DS.init_map(str(L.K3 / 'map_roots/g217'))
    n = 0
    with tempfile.TemporaryDirectory() as td:
        for cp in sorted(p for p in (L.K3 / 'cases/dev_g217/cases').glob('*.json') if p.name != 'cases.json'):
            g, r, i6 = straight6(P, IT, cp)
            ref = dev / f'{g}__straight6.json'
            if r is None:
                assert not ref.exists(), g
                continue
            tmp = Path(td) / ref.name
            json.dump(route_dump(r, g, i6), open(tmp, 'w'))
            assert tmp.read_bytes() == ref.read_bytes(), f'straight6 rebuild differs from E3a for {g}'
            n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--arenas', nargs='+', default=['g258', 'g268', 'g263', 'g241'])
    ap.add_argument('--groups', default=str(L.K3 / 'e3/spread_headroom/spread_headroom_groups.json'))
    ap.add_argument('--picks-root', default=str(L.K3 / 'e3/spread_headroom'))
    ap.add_argument('--out', default=str(L.K3 / 'e3/spread_headroom'))
    ap.add_argument('--rows', default=str(L.K3 / 'e3/tasks/spread_headroom_rows.json'))
    ap.add_argument('--against', nargs='*', default=[str(L.K3 / 'e3/tasks/soil_v1.json')])
    ap.add_argument('--tier', type=int, default=-1)
    ap.add_argument('--model-tag', default='Scrm')
    ap.add_argument('--selfcheck-dev', action='store_true')
    a = ap.parse_args(argv)
    import gen_planner as P
    import f104_n2_iter as IT
    n_dev = selfcheck_dev(P, IT) if a.selfcheck_dev else None
    if n_dev is not None:
        print(f'selfcheck: {n_dev} g217 dev straight6 routes rebuilt byte-identical to E3a', flush=True)
    sel = json.load(open(a.groups))
    out = Path(a.out); (out / 'straight6/routes').mkdir(parents=True, exist_ok=True)
    k3_rel = lambda p: str(Path(p).resolve().relative_to(L.K3.resolve()))
    rows, s6, checks, pick_files = [], {}, {}, []
    for arena in a.arenas:
        case_dir = L.K3 / f'cases/test_{arena}/cases'
        map_root = L.K3 / f'map_roots/{arena}'
        chk = ag_map_check.check(map_root, [case_dir], L.ROOT)
        assert chk['ok'], chk['problems']
        checks[arena] = chk
        P.DS.init_map(str(map_root))
        picks = Path(a.picks_root) / f'picks_crm_{a.model_tag}_{arena}'
        ps = json.load(open(picks / 'summary.json'))
        assert ps['arms'] == ['B'] and ps['tags']['B'] == 'n2iter_cem4x64' and not ps['fixed2'] and ps['world'] == 'crm'
        assert ps['models'] == 'artifacts/traverse/crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt', ps['models']
        assert Path(ps['map_root']).resolve() == map_root.resolve(), ps['map_root']
        want = sel['arenas'][arena]['groups']
        got = sorted(p.stem for p in (picks / 'picks').glob('*.json'))
        assert got == sorted(want), f'{arena}: picks for {len(got)} groups, declared {len(want)}'
        pick_files += list((picks / 'routes').glob('*.json'))
        for g in sorted(want):
            cp = case_dir / f'{g}.json'
            case_rel = f'cases/test_{arena}/cases/{g}.json'
            pb = json.load(open(picks / 'picks' / f'{g}.json'))['arms']['B']
            bid = pb['route_id']; run_id = f'{g}__{a.model_tag}_B'
            rows.append(dict(id=run_id, group=g, arena=arena, case=case_rel, route=k3_rel(picks / 'routes' / f'{bid}.json'),
                             tier=a.tier, episode_seed=L.md5_int(run_id), run=True, kind='spread_headroom',
                             arms=[f'{a.model_tag}:B'], sha256=pb['route_sha256'], vehicle='hmmwv'))
            g2, r, i6 = straight6(P, IT, cp)
            assert g2 == g
            if r is None:
                s6[g] = None; continue
            h = IT.route_sha256(r)
            if h == pb['route_sha256']:
                rows[-1]['arms'].append('straight6'); s6[g] = dict(route_id=run_id, index=i6, route_sha256=h, same_as=f'{a.model_tag}:B'); continue
            rid = f'{g}__straight6'
            p6 = out / 'straight6/routes' / f'{rid}.json'
            json.dump(route_dump(r, g, i6), open(p6, 'w'))
            s6[g] = dict(route_id=rid, index=i6, route_sha256=h, mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()),
                         length_m=float(np.asarray(r['stations'])[-1]))
            rows.append(dict(id=rid, group=g, arena=arena, case=case_rel, route=k3_rel(p6), tier=a.tier,
                             episode_seed=L.md5_int(rid), run=True, kind='spread_headroom', arms=['straight6'], sha256=h,
                             vehicle='hmmwv'))
    ids = [r['id'] for r in rows]; seeds = [r['episode_seed'] for r in rows]
    assert len(set(ids)) == len(ids) and len(set(seeds)) == len(seeds)
    for f in a.against:
        old = json.load(open(f))
        assert not set(ids) & {r['id'] for r in old}, f'id clash with {f}'
        assert not set(seeds) & {r['episode_seed'] for r in old}, f'episode_seed clash with {f}'
    missing = [(r['id'], p) for r in rows for p in (r['case'], r['route']) if not L.local_path(p).exists()]
    assert not missing, missing[:3]
    json.dump(dict(arenas=a.arenas, map_checks=checks, picks=s6, selfcheck_dev_straight6=n_dev),
              open(out / 'straight6/picks.json', 'w'), indent=1)
    Path(a.rows).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(a.rows, 'w'), indent=1)
    files = sorted(pick_files + list((out / 'straight6/routes').glob('*.json')), key=lambda p: p.name)
    lock = hashlib.sha256(); lines = []
    for p in files:
        lock.update(p.name.encode()); lock.update(hashlib.sha256(p.read_bytes()).digest())
        lines.append(f'{L.sha256_file(p)}  {k3_rel(p)}')
    (out / 'SPREAD_PICKS_LOCKED.sha256').write_text(lock.hexdigest() + '  (name + content of every route file of both arms, '
                                                    'sorted by name)\n' + '\n'.join(lines) + '\n')
    n_same = sum(1 for v in s6.values() if v and v.get('same_as'))
    print(json.dumps(dict(groups=len(s6), rows=len(rows), per_arena={x: sum(r['arena'] == x for r in rows) for x in a.arenas},
                          straight6_missing=sum(v is None for v in s6.values()), straight6_same_as_B=n_same,
                          lock=lock.hexdigest(), rows_sha256=L.sha256_file(a.rows)), indent=1))


if __name__ == '__main__':
    main()
