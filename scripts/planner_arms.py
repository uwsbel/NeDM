"""Study 2 (planner) pick pass: one-shot vs iterated resampling arms on identical base routes, hashed before any drive.

Per group (start/goal case + frozen base route routes/<g>/route_00.json), on the world's map and ensemble:
  A  one-shot 256   deployed planner (gen_planner.proposal_pool / fixed2_pool with the deployed rng tag, so it
                    reproduces crm_f104_v1/eval_v1 resp. gen_v1 picks bit-for-bit; checked against --ref-picks)
  B  CEM 4x64       iterated resampling, equal budget (f104_n2_iter.plan_iter)
  C  CEM 8x64       iterated, double budget
  D  one-shot 512   budget control, same seed family (deployed tag + '_512')
  E  = B, pessimistic objective (max over ensemble members)
  F  = B, expected cost  P * c_fail + T  (c_fail 60 s, T = commanded route time)
rng per arm = md5(group + tag). Identical routes (sha256 of the route content) across arms share one route file and
one task row; the first arm in --arms order names it (<group>__<arm>).

Writes <out>/picks/<group>.json, <out>/routes/<route_id>.json (collector format), <out>/tasks.json
({id, group, case, route, run, tier, episode_seed, arms[, ref_id][, arena, shard]}; case/route relative to --task-root,
default two levels above --cases i.e. the crm_f104_v1 / gen_v1 root), <out>/tasks_new_only.json (rows whose route
equals an already-driven reference pick get run=false), <out>/summary.json, <out>/PICKS_LOCKED.sha256.
"""
import argparse, glob, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f104_n2_iter as IT
GP, DS = IT.GP, IT.DS

ROOT = Path(__file__).resolve().parents[1]


def arm_specs(deployed_tag, suffix):
    return {
        'A': dict(kind='oneshot', n=256, tag=deployed_tag, objective='mean', label='oneshot256'),
        'B': dict(kind='iter', rounds=4, n=64, tag='n2iter_cem4x64' + suffix, objective='mean', label='cem4x64'),
        'C': dict(kind='iter', rounds=8, n=64, tag='n2iter_cem8x64' + suffix, objective='mean', label='cem8x64'),
        'D': dict(kind='oneshot', n=512, tag=deployed_tag + '_512', objective='mean', label='oneshot512'),
        'E': dict(kind='iter', rounds=4, n=64, tag='n2iter_cem4x64_pess' + suffix, objective='pess', label='cem4x64_pess'),
        'F': dict(kind='iter', rounds=4, n=64, tag='n2iter_cem4x64_ecost' + suffix, objective='expected_cost', c_fail=60.0,
                  label='cem4x64_ecost60'),
    }


def load_case(case_path):
    case = json.load(open(case_path)); g = case['id']; lay = case['layout']
    pose = np.array([lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']], float)
    goal = np.asarray(case['goal_xy'], float)
    rp = Path(case_path).parent / 'routes' / g / 'route_00.json'
    base = {k: np.asarray(v, float) for k, v in json.load(open(rp)).items() if k in ('waypoints', 'speeds', 'stations', 'headings')}
    base['meta'] = {}
    return case, g, lay, pose, goal, base


def route_json(r, rid, g, arm, arms_label, world):
    return {'waypoints': np.asarray(r['waypoints']).tolist(), 'speeds': np.asarray(r['speeds']).tolist(),
            'stations': np.asarray(r['stations']).tolist(), 'headings': np.asarray(r['headings']).tolist(),
            'meta': {'candidate': f'planner_{arm}_{arms_label}', 'scene_id': g, 'arm': arm, 'world': world,
                     'kind': r['meta'].get('kind'), 'round': r['meta'].get('round'), 'rank': r['meta'].get('rank'),
                     'theta': r['meta'].get('theta'), 'max_lateral_m': r['meta'].get('max_lateral_m'),
                     'mean_speed_mps': r['meta'].get('mean_speed_mps')}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', required=True, help='dir with <group>.json and routes/<group>/route_00.json')
    ap.add_argument('--map-root', help='CRM world: root with static_map_v1 (f104_n2_dataset.init_map)')
    ap.add_argument('--rigid-arena', help='rigid world: arena dir (gen_planner.set_map heightmap, as gen_v1)')
    ap.add_argument('--models', required=True, help='glob of N2-format ensemble checkpoints')
    ap.add_argument('--out', required=True)
    ap.add_argument('--world', choices=['crm', 'rigid'], required=True)
    ap.add_argument('--arms', default='A,B,C,D,E,F')
    ap.add_argument('--fixed2', action='store_true', help='geometry-only family at 2 m/s (theta in R^3)')
    ap.add_argument('--ref-picks', help='reference picks dir (default: eval_v1/picks for crm, gen_v1/test_<tag>/picks for rigid)')
    ap.add_argument('--ref-arm', help="reference arm name in those picks (default crm/crm_fixed2 or n2/n2_fixed2)")
    ap.add_argument('--task-root', help='paths in tasks.json are relative to this (default: two levels above --cases)')
    ap.add_argument('--arena-tag', help='rigid tasks: arena tag (default from the arena dir name)')
    ap.add_argument('--shards', type=int, default=24)
    ap.add_argument('--verify', type=int, default=3, help='groups on which the one-shot pool is checked route by route')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no-float16', action='store_true')
    a = ap.parse_args()

    fixed_speed = 2.0 if a.fixed2 else None
    suffix = '_fixed2' if a.fixed2 else ''
    if a.world == 'crm':
        assert a.map_root, '--map-root required for crm'
        DS.init_map(a.map_root)
        deployed_tag = 'crm_fixed2' if a.fixed2 else 'crm_proposal'
        ref_arm_default = 'crm_fixed2' if a.fixed2 else 'crm'
        ref_dir_default = ROOT / 'artifacts/traverse/crm_f104_v1/eval_v1/picks'
        arena_tag = None
    else:
        assert a.rigid_arena, '--rigid-arena required for rigid'
        GP.set_map(a.rigid_arena)
        deployed_tag = 'gen_fixed2' if a.fixed2 else 'gen_night2'
        ref_arm_default = 'n2_fixed2' if a.fixed2 else 'n2'
        cases_tag = Path(a.cases).resolve().parent.name.replace('cases_test_', '')
        ref_dir_default = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/gen_v1' / f'test_{cases_tag}' / 'picks'
        arena_tag = a.arena_tag or ('f104' if 'f104' in Path(a.rigid_arena).name else Path(a.rigid_arena).name.replace('arena_', ''))
    ref_dir = Path(a.ref_picks) if a.ref_picks else ref_dir_default
    ref_arm = a.ref_arm or ref_arm_default
    ref_ok = ref_dir.is_dir()
    print(f'world {a.world} fixed2={a.fixed2} deployed tag {deployed_tag!r}; reference picks {ref_dir} arm {ref_arm!r} ({"found" if ref_ok else "MISSING"})', flush=True)

    model = GP.RiskModel(a.models)
    print(f'{len(model.members)} ensemble members from {a.models} on {model.dev}', flush=True)
    specs = arm_specs(deployed_tag, suffix)
    arms = [s.strip() for s in a.arms.split(',') if s.strip()]
    for arm in arms:
        assert arm in specs, arm
    out = Path(a.out); (out / 'routes').mkdir(parents=True, exist_ok=True); (out / 'picks').mkdir(exist_ok=True)
    cases = sorted(str(p) for p in Path(a.cases).glob('*.json') if p.name != 'cases.json')
    if a.limit:
        cases = cases[:a.limit]
    task_root = Path(a.task_root).resolve() if a.task_root else Path(a.cases).resolve().parents[1]

    tasks, rows, wall = [], [], {arm: 0.0 for arm in arms}
    ref_match = {'checked': 0, 'match': 0, 'mismatch': []}
    verified = []
    t_start = time.time()
    for gi, cp in enumerate(cases):
        case, g, lay, pose, goal, base = load_case(cp)
        scorer = IT.Scorer(model, lay['start_xy'], case['goal_xy'], lay['start_yaw'], float16=not a.no_float16)
        if gi < a.verify:
            nref = IT.selftest(base, pose, IT.seed(g, deployed_tag), n=256, fixed_speed=fixed_speed)
            verified.append((g, nref))
        ref = None
        if ref_ok and (ref_dir / f'{g}.json').exists():
            ref = json.load(open(ref_dir / f'{g}.json'))['arms'].get(ref_arm)
        picks, seen = {}, {}
        summ = dict(group=g, world=a.world, fixed2=a.fixed2, stratum=case.get('evaluation_stratum'),
                    base_length_m=float(base['stations'][-1]), corridors_batched=None, arms={})
        zA = None
        for arm in arms:
            sp = specs[arm]; rng = np.random.default_rng(IT.seed(g, sp['tag']))
            t0 = time.perf_counter()
            if sp['kind'] == 'oneshot':
                res = IT.oneshot(base, pose, goal, scorer, rng, n=sp['n'], fixed_speed=fixed_speed, objective=sp['objective'])
            else:
                res = IT.plan_iter(base, pose, goal, scorer, rounds=sp['rounds'], n=sp['n'], objective=sp['objective'],
                                   c_fail=sp.get('c_fail', 60.0), rng=rng, anchors=True, fixed_speed=fixed_speed)
            wall[arm] += time.perf_counter() - t0
            if res is None:
                summ['arms'][arm] = None; continue
            r = res['route']; h = IT.route_sha256(r)
            first = h not in seen; seen.setdefault(h, arm); rid = f'{g}__{seen[h]}'
            if first:
                json.dump(route_json(r, rid, g, arm, sp['label'], a.world), open(out / 'routes' / f'{rid}.json', 'w'))
                row = dict(id=rid, group=g, case=os.path.relpath(Path(cp).resolve(), task_root),
                           route=os.path.relpath((out / 'routes' / f'{rid}.json').resolve(), task_root), run=True, tier=gi,
                           episode_seed=int(hashlib.md5(rid.encode()).hexdigest()[:8], 16), arms=[arm], sha256=h)
                if arena_tag:
                    row.update(arena=arena_tag, shard=int(hashlib.md5(g.encode()).hexdigest(), 16) % a.shards)
                tasks.append(row)
            else:
                next(t for t in tasks if t['id'] == rid)['arms'].append(arm)
            entry = dict(route_id=rid, arm=arm, label=sp['label'], tag=sp['tag'], objective=sp['objective'], index=res['index'],
                         kind=res['kind'], round=res['round'], theta=res['theta'], z_mean=res['z_mean'], z_pess=res['z_pess'],
                         P=res['P'], T=res['T'], J=res['J'], mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()),
                         max_lateral_m=float(r['meta'].get('max_lateral_m', 0.0)), length_m=float(np.asarray(r['stations'])[-1]),
                         route_sha256=h, n_evaluated=res['n_evaluated'], tries=res['tries'],
                         n_below_1pct=int(((1 - np.exp(-np.exp(res['Z_mean']))) < 0.01).sum()),
                         z_mean_min=float(res['Z_mean'].min()), z_pess_min=float(res['Z_pess'].min()), wall_s=None)
            if sp['kind'] == 'iter':
                entry.update(log=res['log'], mu_final=res['mu'], sd_final=res['sd'],
                             kinds_count={k: int(sum(1 for q in res['kinds'] if q == k)) for k in ('anchor', 'sample', 'mean')})
            if arm == 'A':
                zA = res['z_mean']
                if ref is not None:
                    ok = int(ref['index']) == res['index']
                    ref_match['checked'] += 1; ref_match['match'] += int(ok)
                    if not ok:
                        ref_match['mismatch'].append(dict(group=g, ref_index=ref['index'], index=res['index'],
                                                          ref_logit=ref.get('logit', ref.get(f'logit_{ref_arm}')), z_mean=res['z_mean']))
                    entry.update(ref_index=int(ref['index']), ref_match=ok, ref_route_id=ref['route_id'])
                    if ok:
                        next(t for t in tasks if t['id'] == rid)['ref_id'] = ref['route_id']
            elif zA is not None:
                entry.update(delta_z_vs_A=res['z_mean'] - zA, frac_evaluated_below_A=float((res['Z_mean'] < zA).mean()),
                             same_route_as_A=(picks['A']['route_sha256'] == h) if 'A' in picks else None)
            picks[arm] = entry
        summ['arms'] = picks; summ['corridors_batched'] = IT._CORR.get('identical')
        json.dump(summ, open(out / 'picks' / f'{g}.json', 'w'), indent=1)
        rows.append(summ)
        if (gi + 1) % 10 == 0 or gi == len(cases) - 1:
            el = time.time() - t_start
            print(f'  {gi + 1}/{len(cases)}  {el:.0f}s  ' + ' '.join(f'{arm}={picks[arm]["z_mean"]:.2f}' for arm in arms if picks.get(arm))
                  + f'  ref {ref_match["match"]}/{ref_match["checked"]}', flush=True)

    json.dump(tasks, open(out / 'tasks.json', 'w'), indent=1)
    new_only = [dict(t, run=('ref_id' not in t)) for t in tasks]
    json.dump(new_only, open(out / 'tasks_new_only.json', 'w'), indent=1)
    # lock: sha256 over the sorted route files
    lock = hashlib.sha256()
    for p in sorted((out / 'routes').glob('*.json')):
        lock.update(p.name.encode()); lock.update(hashlib.sha256(p.read_bytes()).digest())
    (out / 'PICKS_LOCKED.sha256').write_text(lock.hexdigest() + '  routes/*.json (name + content, sorted)\n')

    # summary
    def col(arm, key):
        return np.array([r['arms'][arm][key] for r in rows if r['arms'].get(arm) is not None], float)
    summary = dict(world=a.world, fixed2=a.fixed2, n_groups=len(rows), arms=arms, tags={arm: specs[arm]['tag'] for arm in arms},
                   deployed_tag=deployed_tag, models=a.models, map_root=a.map_root, rigid_arena=a.rigid_arena,
                   n_distinct_routes=len(tasks), n_new_drives=int(sum(t['run'] for t in new_only)),
                   ref=dict(dir=str(ref_dir), arm=ref_arm, checked=ref_match['checked'], match=ref_match['match'],
                            mismatches=ref_match['mismatch']),
                   selftest_groups=verified, corridors_batched=IT._CORR.get('identical'),
                   float16_scoring=not a.no_float16, wall_s_total=time.time() - t_start, per_arm={})
    zA = col('A', 'z_mean') if 'A' in arms else None
    for arm in arms:
        z, zp, P, T = col(arm, 'z_mean'), col(arm, 'z_pess'), col(arm, 'P'), col(arm, 'T')
        d = dict(n=int(len(z)), z_mean_mean=float(z.mean()), z_mean_median=float(np.median(z)), z_pess_mean=float(zp.mean()),
                 z_pess_median=float(np.median(zp)), P_mean=float(P.mean()), P_median=float(np.median(P)),
                 T_mean=float(T.mean()), T_median=float(np.median(T)), mean_speed=float(col(arm, 'mean_speed').mean()),
                 max_lateral_mean=float(col(arm, 'max_lateral_m').mean()), n_evaluated_mean=float(col(arm, 'n_evaluated').mean()),
                 wall_s=wall[arm], wall_s_per_group=wall[arm] / max(len(rows), 1),
                 pick_kind={k: int(sum(1 for r in rows if r['arms'].get(arm) and r['arms'][arm]['kind'] == k)) for k in ('anchor', 'sample', 'mean')},
                 pick_round={str(k): int(sum(1 for r in rows if r['arms'].get(arm) and r['arms'][arm]['round'] == k))
                             for k in sorted({r['arms'][arm]['round'] for r in rows if r['arms'].get(arm)})})
        if zA is not None and arm != 'A' and len(z) == len(zA):
            dz = z - zA
            d.update(delta_z_vs_A_mean=float(dz.mean()), delta_z_vs_A_median=float(np.median(dz)),
                     improved_vs_A=int((dz < -1e-9).sum()), worse_vs_A=int((dz > 1e-9).sum()),
                     frac_improved_vs_A=float((dz < -1e-9).mean()),
                     same_route_as_A=int(sum(1 for r in rows if r['arms'].get(arm) and r['arms'][arm].get('same_route_as_A'))),
                     P_delta_vs_A_mean=float((P - col('A', 'P')).mean()), T_delta_vs_A_mean=float((T - col('A', 'T')).mean()))
            if specs[arm]['kind'] == 'iter':
                d.update(frac_evaluated_below_A_mean=float(col(arm, 'frac_evaluated_below_A').mean()),
                         acceptance_by_round=[float(np.mean([e['acceptance'] for r in rows if r['arms'].get(arm)
                                                             for e in r['arms'][arm]['log'] if e.get('round') == k and e.get('n', 0) > 0 and 'kind' not in e]))
                                              for k in range(specs[arm]['rounds'])])
        summary['per_arm'][arm] = d
    # pairwise agreement (identical route)
    agree = {}
    for i, x in enumerate(arms):
        for y in arms[i + 1:]:
            agree[f'{x}={y}'] = int(sum(1 for r in rows if r['arms'].get(x) and r['arms'].get(y)
                                        and r['arms'][x]['route_sha256'] == r['arms'][y]['route_sha256']))
    summary['agreement'] = agree
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('per_arm', 'ref')}, indent=None))
    for arm in arms:
        d = summary['per_arm'][arm]
        print(f"{arm} {specs[arm]['label']:16s} z_mean {d['z_mean_mean']:.3f} (med {d['z_mean_median']:.3f}) z_pess {d['z_pess_mean']:.3f} "
              f"P {d['P_mean']:.4f} T {d['T_mean']:.1f}s v {d['mean_speed']:.2f}" +
              (f"  dz vs A {d['delta_z_vs_A_mean']:+.3f} (med {d['delta_z_vs_A_median']:+.3f}) improved {d['improved_vs_A']}/{d['n']} same {d['same_route_as_A']}" if 'delta_z_vs_A_mean' in d else '')
              + f"  wall {d['wall_s_per_group']:.2f}s/group")
    print(f"reference check: {ref_match['match']}/{ref_match['checked']} A picks equal {ref_arm!r} in {ref_dir}; "
          f"{len(tasks)} distinct routes, {summary['n_new_drives']} new drives; agreement {agree}")


if __name__ == '__main__':
    main()
