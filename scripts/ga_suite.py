"""Frozen evaluation suites for the shared rigid/CRM risk-model study (PLAN A0 and B0).

--suite planner   (A0) one paired case set driven in both worlds + the reference-arm picks of both specialists.
  assemble : 600 fresh f104_pair_group cases + 200 reused f104_crm_eval_group cases are copied (case json +
             routes/<g>/route_00.json, byte-for-byte) into <out>/cases/, with <out>/suite.json (strata fresh / reused,
             evaluation_stratum, split, sha256 of every file).
  picks    : for every (world, model) the reference arms of scripts/planner_arms.py (A = deployed one-shot 256 with the
             world's rng tag, B = CEM 4x64) are run with the SAME loop and formats as planner_arms.main, but the
             corridors come from the OptiX static depth map (f104_n2_dataset.init_map) in BOTH worlds - the heightmap
             path of planner_arms --rigid-arena is not used (PLAN: one geometry source per purpose). Output
             <out>/picks_<world>_<model>/{picks,routes,tasks.json,summary.json,PICKS_LOCKED.sha256}. Route ids are
             <group>__<model>_<first arm>. On the reused groups the picks are checked against the night-2 picks
             (crm_night2_v1/planner/iter_crm) by index and route sha256.
  merge    : per world ONE tasks file with cluster paths, deduplicated by route sha256 across models
             (<out>/tasks_<world>.json, <out>/routes_<world>/<id>.json, <out>/run_index_<world>.json for the analysis).
             CRM rows are relative to CRM_ROOT (crm_f104_20260916) under generalist/suite/..., rigid rows relative to
             the gen_v1 root under generalist_suite/... with arena f104 and shard = md5(group) % 6. CRM rows whose
             route was already driven by night 2 (eval_iter_crm, same collector config) get run=false + ref paths.
             Afterwards (or alone with --stage lock) <out>/SUITE_LOCKED.sha256 is written: one hash over suite.json,
             tasks_crm.json and tasks_rigid.json (name + content, sorted) plus one `sha256  name` line per file; it
             only reads those files (check later with `tail -n +2 SUITE_LOCKED.sha256 | sha256sum -c`).
--suite tracking  (B0) every designed route (route_00..11) of the 55 twin test-split groups that has a recorded PID
  drive in both worlds (CRM collect_v1, rigid production_v3), stratum feasible (goal reached in both) / infeasible,
  with the per-route sha256 of the route file and the recorded outcomes -> tracking_suite.json.

Run from the repo root with PYTHONPATH=src:scripts and the nedm python. Nothing is submitted to the cluster; the rsync
commands are in the module note.
"""
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
K = ROOT / 'artifacts/traverse/generalist_20260921'

FRESH_CASES = K / 'cases/pair_v1/cases'
REUSED_CASES = ROOT / 'artifacts/traverse/crm_f104_v1/cases_eval/cases'
MAP_ROOT = ROOT / 'artifacts/traverse/crm_f104_v1/map_root'
MODELS = {'Scrm': str(ROOT / 'artifacts/traverse/crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt'),
          # byte-identical to /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/gen_v1/models/N2_s*.pt (sha256 checked 09-21)
          'Srigid': str(ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/night2_v1/final/N2_s*.pt')}
DEPLOYED_TAG = {'crm': 'crm_proposal', 'rigid': 'gen_night2'}     # planner_arms.main, speed-free
N2 = ROOT / 'artifacts/traverse/crm_night2_v1/planner'
N2_PICKS, N2_TASKS, N2_RUNS = N2 / 'iter_crm/picks', N2 / 'iter_crm/tasks.json', N2 / 'eval_iter_crm/runs'
CRM_CONFIG = ROOT / 'artifacts/traverse/crm_f104_v1/configs/crm_main.json'
CRM_ROOT = '/work1/dannegrut/harry/experiments/crm_f104_20260916'
GEN_ROOT = '/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/gen_v1'
CRM_PREFIX, RIGID_PREFIX, N2_REF_PREFIX = 'generalist/suite', 'generalist_suite', 'night2/eval_iter_crm/runs'
# tracking suite sources
TWIN = ROOT / 'artifacts/traverse/crm_night2_v1/datasets/twin_crm.npz'
NIGHT2_CASES = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases'
CRM_COLLECT = ROOT / 'artifacts/traverse/crm_f104_v1/collect_v1/runs'
RIGID_PROD = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/production_v3/runs'
REQ_FILES = ('trajectory.npz', 'command_reference.npz', 'outcome.json')


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    try:
        return os.path.relpath(Path(p).resolve(), ROOT)
    except ValueError:
        return str(p)


def load_json(p):
    return json.load(open(p))


# ----------------------------------------------------------------------------------------------------------------------
# A0 assemble
# ----------------------------------------------------------------------------------------------------------------------
def assemble(out, fresh_dir, reused_dir, n_fresh=600, n_reused=200):
    out = Path(out); cdir = out / 'cases'; (cdir / 'routes').mkdir(parents=True, exist_ok=True)
    groups = []
    for stratum, src, n in (('fresh', Path(fresh_dir), n_fresh), ('reused', Path(reused_dir), n_reused)):
        files = sorted(p for p in src.glob('*.json') if p.name != 'cases.json')
        assert len(files) >= n, (stratum, len(files), n)
        for cp in files[:n]:
            g = cp.stem
            rp = src / 'routes' / g / 'route_00.json'
            assert rp.exists(), rp
            shutil.copyfile(cp, cdir / cp.name)
            (cdir / 'routes' / g).mkdir(exist_ok=True)
            shutil.copyfile(rp, cdir / 'routes' / g / 'route_00.json')
            case = load_json(cp)
            assert case['id'] == g, (case['id'], g)
            groups.append(dict(group=g, stratum=stratum, evaluation_stratum=case.get('evaluation_stratum'),
                               split=case.get('split'), source_case=rel(cp), source_route00=rel(rp),
                               case_sha256=sha256_file(cp), route00_sha256=sha256_file(rp),
                               start_xy=case['layout']['start_xy'], start_yaw=case['layout']['start_yaw'],
                               goal_xy=case['goal_xy'], arena=case['arena']))
    # separation between the two strata in the 4-D start+goal space (the generator's --avoid margin is 2 m)
    P = {s: np.array([r['start_xy'] + r['goal_xy'] for r in groups if r['stratum'] == s], float) for s in ('fresh', 'reused')}
    d = np.linalg.norm(P['fresh'][:, None, :] - P['reused'][None, :, :], axis=-1)
    manifest = dict(schema=1, created=time.strftime('%Y-%m-%d %H:%M:%S'), n_groups=len(groups),
                    counts={s: int(sum(r['stratum'] == s for r in groups)) for s in ('fresh', 'reused')},
                    sources=dict(fresh=rel(fresh_dir), reused=rel(reused_dir),
                                 fresh_manifest_sha256=sha256_file(Path(fresh_dir) / 'cases.json') if (Path(fresh_dir) / 'cases.json').exists() else None),
                    min_4d_distance_fresh_reused_m=float(d.min()), exact_pair_matches=int((d < 1e-9).sum()),
                    evaluation_strata={str(k): int(v) for k, v in zip(*np.unique([r['evaluation_stratum'] for r in groups], return_counts=True))},
                    case_split_note='split fields are copied from the source case files; the plan blacklists every suite group by id in all builders',
                    groups=groups)
    json.dump(manifest, open(out / 'suite.json', 'w'), indent=1)
    print(f"suite: {len(groups)} groups ({manifest['counts']}), min 4-D fresh-reused distance {d.min():.2f} m, strata {manifest['evaluation_strata']}")
    return manifest


# ----------------------------------------------------------------------------------------------------------------------
# A0 reference picks: the planner_arms loop on the depth map, both worlds
# ----------------------------------------------------------------------------------------------------------------------
def pick_pass(suite_dir, world, mtag, model_glob, arms, map_root, verify=3, limit=0, n2_picks=N2_PICKS, float16=True):
    import planner_arms as PA
    IT, GP, DS = PA.IT, PA.GP, PA.DS
    suite_dir = Path(suite_dir); manifest = load_json(suite_dir / 'suite.json')
    DS.init_map(str(map_root))                              # OptiX static depth map in BOTH worlds
    deployed_tag = DEPLOYED_TAG[world]
    specs = PA.arm_specs(deployed_tag, '')
    for arm in arms:
        assert arm in specs, arm
    model = GP.RiskModel(model_glob)
    out = suite_dir / f'picks_{world}_{mtag}'; (out / 'routes').mkdir(parents=True, exist_ok=True); (out / 'picks').mkdir(exist_ok=True)
    rows_in = manifest['groups']
    if limit:
        rows_in = [r for r in rows_in if r['stratum'] == 'fresh'][:limit] + [r for r in rows_in if r['stratum'] == 'reused'][:limit]
    print(f'[{world}/{mtag}] {len(model.members)} members from {model_glob} on {model.dev}; deployed tag {deployed_tag!r}; '
          f'{len(rows_in)} groups; map {map_root}', flush=True)
    n2_ok = Path(n2_picks).is_dir()
    tasks, rows, wall = [], [], {arm: 0.0 for arm in arms}
    n2_match = {arm: dict(checked=0, index_match=0, route_match=0, mismatch=[]) for arm in arms}
    verified = []; t_start = time.time()
    for gi, rec in enumerate(rows_in):
        g = rec['group']; cp = suite_dir / 'cases' / f'{g}.json'
        case, g2, lay, pose, goal, base = PA.load_case(str(cp)); assert g2 == g
        scorer = IT.Scorer(model, lay['start_xy'], case['goal_xy'], lay['start_yaw'], float16=float16)
        if gi < verify:
            verified.append((g, IT.selftest(base, pose, IT.seed(g, deployed_tag), n=256)))
        ref = load_json(Path(n2_picks) / f'{g}.json')['arms'] if n2_ok and (Path(n2_picks) / f'{g}.json').exists() else None
        picks, seen = {}, {}
        summ = dict(group=g, world=world, model=mtag, models=model_glob, fixed2=False, stratum=rec['stratum'],
                    evaluation_stratum=rec['evaluation_stratum'], base_length_m=float(base['stations'][-1]), corridors_batched=None, arms={})
        zA = None
        for arm in arms:
            sp = specs[arm]; rng = np.random.default_rng(IT.seed(g, sp['tag']))
            t0 = time.perf_counter()
            if sp['kind'] == 'oneshot':
                res = IT.oneshot(base, pose, goal, scorer, rng, n=sp['n'], objective=sp['objective'])
            else:
                res = IT.plan_iter(base, pose, goal, scorer, rounds=sp['rounds'], n=sp['n'], objective=sp['objective'],
                                   c_fail=sp.get('c_fail', 60.0), rng=rng, anchors=True)
            wall[arm] += time.perf_counter() - t0
            if res is None:
                summ['arms'][arm] = None; continue
            r = res['route']; h = IT.route_sha256(r)
            first = h not in seen; seen.setdefault(h, arm); rid = f'{g}__{mtag}_{seen[h]}'
            if first:
                json.dump(PA.route_json(r, rid, g, arm, sp['label'], world), open(out / 'routes' / f'{rid}.json', 'w'))
                tasks.append(dict(id=rid, group=g, case=f'cases/{g}.json', route=f'picks_{world}_{mtag}/routes/{rid}.json', run=True,
                                  tier=gi, episode_seed=int(hashlib.md5(rid.encode()).hexdigest()[:8], 16), arms=[arm], sha256=h,
                                  world=world, model=mtag, stratum=rec['stratum']))
            else:
                next(t for t in tasks if t['id'] == rid)['arms'].append(arm)
            entry = dict(route_id=rid, arm=arm, model=mtag, label=sp['label'], tag=sp['tag'], objective=sp['objective'], index=res['index'],
                         kind=res['kind'], round=res['round'], theta=res['theta'], z_mean=res['z_mean'], z_pess=res['z_pess'],
                         P=res['P'], T=res['T'], J=res['J'], mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()),
                         max_lateral_m=float(r['meta'].get('max_lateral_m', 0.0)), length_m=float(np.asarray(r['stations'])[-1]),
                         route_sha256=h, n_evaluated=res['n_evaluated'], tries=res['tries'],
                         n_below_1pct=int(((1 - np.exp(-np.exp(res['Z_mean']))) < 0.01).sum()),
                         z_mean_min=float(res['Z_mean'].min()), z_pess_min=float(res['Z_pess'].min()))
            if sp['kind'] == 'iter':
                entry.update(log=res['log'], mu_final=res['mu'], sd_final=res['sd'],
                             kinds_count={k: int(sum(1 for q in res['kinds'] if q == k)) for k in ('anchor', 'sample', 'mean')})
            if ref is not None and ref.get(arm):
                m = n2_match[arm]; m['checked'] += 1
                oi = int(ref[arm]['index']) == res['index']; orr = ref[arm]['route_sha256'] == h
                m['index_match'] += int(oi); m['route_match'] += int(orr)
                if not orr:
                    m['mismatch'].append(dict(group=g, n2_index=ref[arm]['index'], index=res['index'], n2_z=ref[arm].get('z_mean'), z_mean=res['z_mean']))
                entry.update(n2_index=int(ref[arm]['index']), n2_index_match=oi, n2_route_match=orr, n2_route_id=ref[arm]['route_id'])
            if arm == 'A':
                zA = res['z_mean']
            elif zA is not None:
                entry.update(delta_z_vs_A=res['z_mean'] - zA, frac_evaluated_below_A=float((res['Z_mean'] < zA).mean()),
                             same_route_as_A=(picks['A']['route_sha256'] == h) if 'A' in picks else None)
            picks[arm] = entry
        summ['arms'] = picks; summ['corridors_batched'] = IT._CORR.get('identical')
        json.dump(summ, open(out / 'picks' / f'{g}.json', 'w'), indent=1)
        rows.append(summ)
        if (gi + 1) % 25 == 0 or gi == len(rows_in) - 1:
            el = time.time() - t_start
            print(f'  [{world}/{mtag}] {gi + 1}/{len(rows_in)}  {el:.0f}s  ' + ' '.join(f'{arm}={picks[arm]["z_mean"]:.2f}' for arm in arms if picks.get(arm))
                  + '  n2 ' + ' '.join(f"{arm}:{n2_match[arm]['route_match']}/{n2_match[arm]['checked']}" for arm in arms), flush=True)
    json.dump(tasks, open(out / 'tasks.json', 'w'), indent=1)
    lock = hashlib.sha256()
    for p in sorted((out / 'routes').glob('*.json')):
        lock.update(p.name.encode()); lock.update(hashlib.sha256(p.read_bytes()).digest())
    (out / 'PICKS_LOCKED.sha256').write_text(lock.hexdigest() + '  routes/*.json (name + content, sorted)\n')

    def col(arm, key):
        return np.array([r['arms'][arm][key] for r in rows if r['arms'].get(arm) is not None], float)
    summary = dict(world=world, model=mtag, models=model_glob, map_root=str(map_root), geometry='optix_static_depth_map', n_groups=len(rows),
                   arms=arms, tags={arm: specs[arm]['tag'] for arm in arms}, deployed_tag=deployed_tag, n_distinct_routes=len(tasks),
                   n2_reference=dict(dir=str(n2_picks), **{arm: {k: v for k, v in n2_match[arm].items()} for arm in arms}),
                   selftest_groups=verified, corridors_batched=IT._CORR.get('identical'), float16_scoring=float16,
                   wall_s_total=time.time() - t_start, per_arm={}, by_stratum={})
    for arm in arms:
        z, P, T = col(arm, 'z_mean'), col(arm, 'P'), col(arm, 'T')
        summary['per_arm'][arm] = dict(n=int(len(z)), z_mean_mean=float(z.mean()), z_mean_median=float(np.median(z)), P_mean=float(P.mean()),
                                       T_mean=float(T.mean()), T_median=float(np.median(T)), mean_speed=float(col(arm, 'mean_speed').mean()),
                                       max_lateral_mean=float(col(arm, 'max_lateral_m').mean()), wall_s=wall[arm], wall_s_per_group=wall[arm] / max(len(rows), 1),
                                       pick_kind={k: int(sum(1 for r in rows if r['arms'].get(arm) and r['arms'][arm]['kind'] == k)) for k in ('anchor', 'sample', 'mean')})
    for s in ('fresh', 'reused'):
        sub = [r for r in rows if r['stratum'] == s]
        summary['by_stratum'][s] = {arm: dict(n=len(sub), z_mean_mean=float(np.mean([r['arms'][arm]['z_mean'] for r in sub])) if sub else None,
                                              P_mean=float(np.mean([r['arms'][arm]['P'] for r in sub])) if sub else None) for arm in arms}
    summary['agreement'] = {f'{x}={y}': int(sum(1 for r in rows if r['arms'].get(x) and r['arms'].get(y) and r['arms'][x]['route_sha256'] == r['arms'][y]['route_sha256']))
                            for i, x in enumerate(arms) for y in arms[i + 1:]}
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
    print(f"[{world}/{mtag}] done: {len(rows)} groups, {len(tasks)} distinct routes, wall {summary['wall_s_total']:.0f} s; "
          + ' '.join(f"{arm} z {summary['per_arm'][arm]['z_mean_mean']:.3f} P {summary['per_arm'][arm]['P_mean']:.4f} T {summary['per_arm'][arm]['T_mean']:.1f}s" for arm in arms)
          + f"; agreement {summary['agreement']}; night-2 reproduction " + ' '.join(f"{arm} idx {n2_match[arm]['index_match']}/{n2_match[arm]['checked']} route {n2_match[arm]['route_match']}/{n2_match[arm]['checked']}" for arm in arms), flush=True)
    return summary


def n2_driven_routes(ref_arms=('A', 'B')):
    """sha256 -> night-2 run id for every eval_iter_crm route that was driven with the same collector config (local copy
    present) and that was the pick of one of `ref_arms` (default A/B = the S_crm reference arms of the 200 reused groups;
    the same route file may also have been the pick of C-F, which does not matter). Also counts the routes any arm drove."""
    if not Path(N2_TASKS).exists():
        return {}, None
    cfg_name = load_json(CRM_CONFIG).get('name') if Path(CRM_CONFIG).exists() else None
    out, out_any = {}, {}; skipped = 0
    for t in load_json(N2_TASKS):
        d = Path(N2_RUNS) / t['id']
        if not (d / 'episode_complete.json').exists() or not (d / 'outcome.json').exists():
            continue
        o = load_json(d / 'outcome.json'); crm = o.get('crm', {})
        if cfg_name and (crm.get('config', {}).get('name') != cfg_name or crm.get('physics_dt_s') != 0.001):
            skipped += 1; continue
        out_any[t['sha256']] = t['id']
        if set(t.get('arms', [])) & set(ref_arms):
            out[t['sha256']] = t['id']
    return out, dict(config=cfg_name, ref_arms=list(ref_arms), usable=len(out), usable_any_arm=len(out_any),
                     skipped_config_mismatch=skipped, runs=str(N2_RUNS))


def merge(suite_dir, world, mtags, arms, shards=6, ref_arms=('A', 'B')):
    suite_dir = Path(suite_dir); manifest = load_json(suite_dir / 'suite.json')
    strat = {r['group']: r for r in manifest['groups']}
    rdir = suite_dir / f'routes_{world}'; rdir.mkdir(exist_ok=True)
    by_sha, order = {}, []
    for mtag in mtags:
        pdir = suite_dir / f'picks_{world}_{mtag}'
        for t in load_json(pdir / 'tasks.json'):
            key = t['sha256']
            if key not in by_sha:
                by_sha[key] = dict(id=t['id'], group=t['group'], tier=t['tier'], sha256=key, arms=[], aliases=[], src=pdir / 'routes' / f"{t['id']}.json")
                order.append(key)
            else:
                assert by_sha[key]['group'] == t['group'], (key, by_sha[key]['group'], t['group'])
                by_sha[key]['aliases'].append(t['id'])
            by_sha[key]['arms'] += [f'{mtag}:{a}' for a in t['arms']]
    ref_map, ref_info = (n2_driven_routes(ref_arms) if world == 'crm' else ({}, None))
    ref_any = n2_driven_routes(('A', 'B', 'C', 'D', 'E', 'F'))[0] if world == 'crm' else {}
    tasks, index = [], {}
    for key in order:
        e = by_sha[key]; g = e['group']; rid = e['id']
        shutil.copyfile(e['src'], rdir / f'{rid}.json')
        assert sha256_file(rdir / f'{rid}.json') == sha256_file(e['src'])
        row = dict(id=rid, group=g, run=True, tier=e['tier'], arms=e['arms'], sha256=key, aliases=e['aliases'],
                   stratum=strat[g]['stratum'], evaluation_stratum=strat[g]['evaluation_stratum'])
        if world == 'crm':
            row.update(case=f'{CRM_PREFIX}/cases/{g}.json', route=f'{CRM_PREFIX}/routes_crm/{rid}.json',
                       episode_seed=int(hashlib.md5(rid.encode()).hexdigest()[:8], 16), extra=[])
            if key in ref_map:
                row.update(run=False, ref_run=f'{N2_REF_PREFIX}/{ref_map[key]}', ref_run_local=rel(Path(N2_RUNS) / ref_map[key]))
            elif key in ref_any:
                row['n2_route_driven_by_other_arm'] = ref_any[key]     # identical route driven by night-2 C-F only: driven again (info)
        else:
            row.update(arena='f104', case=f'{RIGID_PREFIX}/cases/{g}.json', route=f'{RIGID_PREFIX}/routes/{rid}.json',
                       shard=int(hashlib.md5(g.encode()).hexdigest(), 16) % shards)
        tasks.append(row)
        for alias in [rid] + e['aliases']:
            index[alias] = dict(driven_as=rid, group=g, sha256=key, run_rel=f'runs/{rid}', ref_run_local=row.get('ref_run_local'), ref_run_cluster=row.get('ref_run'))
    json.dump(tasks, open(suite_dir / f'tasks_{world}.json', 'w'), indent=1)
    json.dump(index, open(suite_dir / f'run_index_{world}.json', 'w'), indent=1)
    n_new = sum(t['run'] for t in tasks); n_ref = len(tasks) - n_new
    per_model = {m: sum(1 for t in tasks if any(a.startswith(m + ':') for a in t['arms'])) for m in mtags}
    n_arm_routes = sum(len(t['arms']) for t in tasks)
    n_per_pass = sum(len(load_json(suite_dir / f'picks_{world}_{m}' / 'tasks.json')) for m in mtags)
    ref_by_model = {m: sum(1 for t in tasks if not t['run'] and any(a.startswith(m + ':') for a in t['arms'])) for m in mtags}
    rep = dict(world=world, models=mtags, arms=arms, n_groups=len({t['group'] for t in tasks}), n_arm_routes=n_arm_routes,
               n_distinct_routes=len(tasks), n_drives=n_new, n_reused_reference_drives=n_ref,
               n_reused_touching_model=ref_by_model, n_routes_driven_by_night2_other_arms_only=sum(1 for t in tasks if 'n2_route_driven_by_other_arm' in t),
               n_distinct_per_pass_sum=n_per_pass, n_dedup_across_models=n_per_pass - len(tasks),
               by_stratum={s: dict(distinct=sum(1 for t in tasks if t['stratum'] == s), drives=sum(1 for t in tasks if t['stratum'] == s and t['run']))
                           for s in ('fresh', 'reused')},
               routes_touching_model=per_model, n2_reference=ref_info,
               paths=dict(tasks=rel(suite_dir / f'tasks_{world}.json'), routes=rel(rdir), run_index=rel(suite_dir / f'run_index_{world}.json'),
                          cluster_root=CRM_ROOT if world == 'crm' else GEN_ROOT, cluster_prefix=CRM_PREFIX if world == 'crm' else RIGID_PREFIX),
               shards=(shards if world == 'rigid' else None))
    if world == 'rigid':
        rep['rows_per_shard'] = {str(s): int(sum(1 for t in tasks if t['shard'] == s)) for s in range(shards)}
    json.dump(rep, open(suite_dir / f'merge_{world}.json', 'w'), indent=1)
    print(f"[{world}] merged: {rep['n_groups']} groups, {n_arm_routes} (model, arm) picks -> {len(tasks)} distinct routes, {n_new} drives, "
          f"{n_ref} reused night-2 drives; by stratum {rep['by_stratum']}")
    return rep


SUITE_LOCK_FILES = ('suite.json', 'tasks_crm.json', 'tasks_rigid.json')


def write_suite_lock(suite_dir, names=SUITE_LOCK_FILES):
    """<suite_dir>/SUITE_LOCKED.sha256: line 1 = one sha256 over (file name + sha256 digest of the content) of the suite manifest and
    the merged tasks files, sorted by name (same scheme as PICKS_LOCKED.sha256), then one `<sha256>  <name>` line per file so
    `tail -n +2 SUITE_LOCKED.sha256 | sha256sum -c` re-checks them. Reads the files only; nothing else is written."""
    suite_dir = Path(suite_dir); present = sorted(n for n in names if (suite_dir / n).exists())
    lock = hashlib.sha256(); lines = []
    for n in present:
        h = sha256_file(suite_dir / n); lock.update(n.encode()); lock.update(bytes.fromhex(h)); lines.append(f'{h}  {n}')
    (suite_dir / 'SUITE_LOCKED.sha256').write_text(lock.hexdigest() + '  ' + ' + '.join(present) + ' (name + content, sorted)\n' + '\n'.join(lines) + '\n')
    print(f'SUITE_LOCKED.sha256 {lock.hexdigest()} over {present}')
    return lock.hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
# B0 tracking suite
# ----------------------------------------------------------------------------------------------------------------------
def tracking(out, twin=TWIN, cases_dir=NIGHT2_CASES, crm_runs=CRM_COLLECT, rigid_runs=RIGID_PROD, split='test'):
    z = np.load(twin, allow_pickle=True)
    groups = sorted(set(str(g) for g in z['group'][z['split'] == split]))
    routes, drop = [], dict(missing_crm=0, missing_rigid=0, missing_both=0, reference_mismatch=0)
    worlds = {'crm': Path(crm_runs), 'rigid': Path(rigid_runs)}
    for g in groups:
        cp = Path(cases_dir) / f'{g}.json'; case = load_json(cp)
        assert case['split'] == split, (g, case['split'])
        for k in range(12):
            rid = f'{g}_route_{k:02d}'; rp = Path(cases_dir) / 'routes' / g / f'route_{k:02d}.json'
            have = {w: all((d / rid / f).exists() for f in REQ_FILES) for w, d in worlds.items()}
            if not all(have.values()):
                drop['missing_both' if not any(have.values()) else ('missing_crm' if not have['crm'] else 'missing_rigid')] += 1
                continue
            route = load_json(rp); wp = np.asarray(route['waypoints'], float)
            rec = {}
            for w, d in worlds.items():
                o = load_json(d / rid / 'outcome.json'); c = np.load(d / rid / 'command_reference.npz')
                mism = float(np.abs(c['reference_waypoints'] - wp).max()) if c['reference_waypoints'].shape == wp.shape else float('inf')
                rec[w] = dict(run_dir=rel(d / rid), status=o['status'], goal_reached=bool(o['status'] == 'goal_reached'), elapsed_s=float(o['elapsed_s']),
                              frames=int(o['frames']), positive_work_kj=o.get('positive_work_kj'), reference_waypoint_mismatch_m=mism,
                              trajectory_sha256=sha256_file(d / rid / 'trajectory.npz'))
            if any(rec[w]['reference_waypoint_mismatch_m'] > 1e-6 for w in worlds):
                drop['reference_mismatch'] += 1; continue
            meta = route.get('meta', {})
            routes.append(dict(id=rid, group=g, route_index=k, route_file=rel(rp), route_sha256=sha256_file(rp), case_file=rel(cp), case_sha256=sha256_file(cp),
                               speed_profile_id=meta.get('speed_profile_id'), lateral_offset_m=meta.get('lateral_offset_m'), cruise_speed_mps=meta.get('cruise_speed_mps'),
                               evaluation_stratum=case.get('evaluation_stratum'), length_m=float(route['stations'][-1]), n_waypoints=int(len(wp)),
                               stratum='feasible' if all(rec[w]['goal_reached'] for w in worlds) else 'infeasible', recorded=rec))
    counts = {s: sum(r['stratum'] == s for r in routes) for s in ('feasible', 'infeasible')}
    from collections import Counter
    suite = dict(schema=1, created=time.strftime('%Y-%m-%d %H:%M:%S'), split=split, twin=rel(twin), n_groups=len(groups), groups=groups,
                 n_routes=len(routes), counts=counts, dropped=drop,
                 sources=dict(cases=rel(cases_dir), crm_runs=rel(crm_runs), rigid_runs=rel(rigid_runs), required_files=list(REQ_FILES)),
                 cluster_paths=dict(crm_cases=f'{CRM_ROOT}/cases/night2/<g>.json', crm_routes=f'{CRM_ROOT}/cases/night2/routes/<g>/route_NN.json',
                                    note='the CRM copy was listed on the cluster (read-only); no rigid copy of cases_night2 was found under gen_v1. '
                                         'The B0 evaluation drives run through the new-mode collectors from G (PLAN conventions), so the case and '
                                         'route files named in route_file / case_file must be shipped to G by the B5 task builder.'),
                 status_pairs={f'{a}|{b}': n for (a, b), n in Counter((r['recorded']['crm']['status'], r['recorded']['rigid']['status']) for r in routes).items()},
                 by_profile={p: dict(n=sum(1 for r in routes if r['speed_profile_id'] == p), feasible=sum(1 for r in routes if r['speed_profile_id'] == p and r['stratum'] == 'feasible'))
                             for p in sorted({r['speed_profile_id'] for r in routes}, key=str)},
                 decision_rule='PLAN Milestone B: primary = one-sided 95th percentile of the paired bootstrap of the policy/native-PID ratio of the mean '
                               'station-based cross-track (Winsorised 5 %, unreached stations capped at 6 m) on the feasible stratum < 0.90; '
                               'completion difference > -3 pts (both strata), unsafe-rate difference <= +1 pt, speed-error ratio < 1.10',
                 routes=routes)
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(suite, open(out, 'w'), indent=1)
    print(f"tracking suite: {len(groups)} {split} groups, {len(routes)} routes with recordings in both worlds ({counts}), dropped {drop} -> {out}")
    return suite


# ----------------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--suite', choices=['planner', 'tracking'], required=True)
    ap.add_argument('--out', help='planner: suite dir (default K/A_adapt/suite); tracking: json path (default K/B_tracker/suite/tracking_suite.json)')
    ap.add_argument('--stage', default='all', choices=['all', 'assemble', 'picks', 'merge', 'lock'], help='lock: only (re)write SUITE_LOCKED.sha256')
    ap.add_argument('--pass', dest='pass_', help='picks stage: run ONE pass world:model (e.g. crm:Scrm)')
    ap.add_argument('--jobs', type=int, default=1, help='picks stage: run the passes as this many parallel subprocesses')
    ap.add_argument('--worlds', default='crm,rigid'); ap.add_argument('--models', default='Scrm,Srigid'); ap.add_argument('--arms', default='A,B')
    ap.add_argument('--fresh-cases', default=str(FRESH_CASES)); ap.add_argument('--reused-cases', default=str(REUSED_CASES))
    ap.add_argument('--n-fresh', type=int, default=600); ap.add_argument('--n-reused', type=int, default=200)
    ap.add_argument('--map-root', default=str(MAP_ROOT)); ap.add_argument('--model-glob', action='append', default=[], help='TAG=GLOB overrides')
    ap.add_argument('--n2-picks', default=str(N2_PICKS)); ap.add_argument('--verify', type=int, default=3); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--shards', type=int, default=6); ap.add_argument('--no-float16', action='store_true')
    ap.add_argument('--ref-arms', default='A,B', help='merge: night-2 eval_iter_crm arms whose drives count as the already-driven reference')
    # tracking
    ap.add_argument('--twin', default=str(TWIN)); ap.add_argument('--night2-cases', default=str(NIGHT2_CASES))
    ap.add_argument('--crm-runs', default=str(CRM_COLLECT)); ap.add_argument('--rigid-runs', default=str(RIGID_PROD)); ap.add_argument('--split', default='test')
    a = ap.parse_args()
    if a.suite == 'tracking':
        tracking(a.out or K / 'B_tracker/suite/tracking_suite.json', a.twin, a.night2_cases, a.crm_runs, a.rigid_runs, a.split)
        return
    suite_dir = Path(a.out or K / 'A_adapt/suite'); suite_dir.mkdir(parents=True, exist_ok=True)
    worlds = a.worlds.split(','); mtags = a.models.split(','); arms = a.arms.split(',')
    models = dict(MODELS); models.update(dict(s.split('=', 1) for s in a.model_glob))
    if a.stage in ('all', 'assemble'):
        assemble(suite_dir, a.fresh_cases, a.reused_cases, a.n_fresh, a.n_reused)
    if a.stage in ('all', 'picks'):
        passes = [tuple(a.pass_.split(':'))] if a.pass_ else [(w, m) for w in worlds for m in mtags]
        if a.jobs > 1 and len(passes) > 1:
            (suite_dir / 'logs').mkdir(exist_ok=True); procs = []
            for w, m in passes:
                cmd = [sys.executable, __file__, '--suite', 'planner', '--stage', 'picks', '--pass', f'{w}:{m}', '--out', str(suite_dir), '--arms', a.arms,
                       '--map-root', a.map_root, '--n2-picks', a.n2_picks, '--verify', str(a.verify), '--limit', str(a.limit)] + \
                      [f'--model-glob={k}={v}' for k, v in models.items()] + (['--no-float16'] if a.no_float16 else [])
                lg = open(suite_dir / 'logs' / f'picks_{w}_{m}.log', 'w')
                procs.append((w, m, subprocess.Popen(cmd, stdout=lg, stderr=subprocess.STDOUT, cwd=ROOT), lg))
                while sum(p.poll() is None for _, _, p, _ in procs) >= a.jobs:
                    time.sleep(5)
            for w, m, p, lg in procs:
                rc = p.wait(); lg.close()
                print(f'pass {w}:{m} rc={rc}', flush=True)
                assert rc == 0, (w, m, rc)
        else:
            for w, m in passes:
                pick_pass(suite_dir, w, m, models[m], arms, a.map_root, a.verify, a.limit, a.n2_picks, float16=not a.no_float16)
    if a.stage in ('all', 'merge'):
        for w in worlds:
            merge(suite_dir, w, mtags, arms, a.shards, tuple(s for s in a.ref_arms.split(',') if s))
    if a.stage in ('all', 'merge', 'lock'):
        write_suite_lock(suite_dir)


if __name__ == '__main__':
    main()
