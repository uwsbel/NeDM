#!/usr/bin/env python3
"""Gradient-refined picks from a standing start for one vehicle's ensemble on one suite (offroad_vehicles_20260927,
module M4; PLAN 1.7, 4.2; scout S4 sections 2.4 and 3).

What it runs: scripts/ci_grad.py (unchanged) WITHOUT --poses, i.e. from the case layout pose at rest with an all-masked
history (the standing start; ga_planner.decision_for), with ci_grad's defaults (CEM 4 x 64 = arm B, then 17 starts x 60
Adam steps, pessimistic keep, abstain margin 0.3 logit = arm G). ci_grad writes picks/<g>.json (arms B and G),
routes/<g>__G.json (always) and routes/<g>__B.json (when B differs), tasks.json, summary.json, PICKS_LOCKED.sha256.

What this tool adds (the row builder scripts/ov_eval_tasks.py, like ag_eval_tasks.py, reads only ag_picks.json):
  - the declared group list (ag_picks.declared_groups: 'all', @file, json list, comma list; --first N = lowest md5),
    the map / arena check first (ag_map_check), the checkpoints' provenance (ag_picks.models_info: ci_train deploy
    checkpoints trained with --domain-filter = --world);
  - checks after planning: every group planned, arm G route file present, its content hash equal to the pick, starts
    within 0.25 m of the case pose and ends within 0.25 m of the goal (ag_picks.check_picks), ci_grad's own checks
    (starts reproduced, keep-best, G never worse than B, route contract) all true, PICKS_LOCKED.sha256 equal to the
    route files (ag_picks.lock_routes);
  - --ref-b-picks DIR (a recorded ag_picks.py --mode free folder of the SAME ensemble): ci_grad's arm B must equal the
    recorded CEM pick (route sha256) on every group checked (the S4 self-check: the CEM stage reproduces G_full's picks);
  - --rerun-check N: re-plan the first N groups (md5 order) into a scratch folder; route files (bytes) and the G / B
    pick records (every field except timings) must be identical;
  - ag_picks.json (schema ag_picks_v1, mode 'grad', arm key 'G') with models + sha256, map check, groups sha256, the
    planner command, per-group rows (route file, file sha256, route sha256, P, z_mean, speeds, length, stratum,
    abstained, gain, B route sha256), lock, rerun and reference results; and OV_GRAD_LOCK.json (manifest sha256 +
    route lock) written last. The output folder must not exist (--force replaces it).

  PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts; unset NEDM_VEHICLE
  K3=artifacts/traverse/arena_gator_20260925; SUITE=artifacts/traverse/generalist_20260921/A_adapt/suite/cases
  $PY scripts/ov_grad_picks.py --arena f104 --world crm --cases $SUITE --groups all \
     --models "$K3/e5/deploy/G_full_soil/G_full_soil_deploy_s*.pt" --model-tag G_full \
     --ref-b-picks $K3/e6/picks/crm_bfull/f104/G_full_free --rerun-check 20 \
     --out artifacts/traverse/offroad_vehicles_20260927/e6/picks/f104/G_full_grad
The planner does not know the vehicle: the same folder serves every vehicle the ensemble is driven on (the row builder
names the vehicle per arm).
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ag_map_check                        # noqa: E402
import ag_picks as AP                      # noqa: E402

PY = sys.executable
TIME_KEYS = ('wall_s', 'seconds', 'secs', 'elapsed')


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def strip_times(e):
    if isinstance(e, dict):
        return {k: strip_times(v) for k, v in e.items() if k not in TIME_KEYS}
    if isinstance(e, list):
        return [strip_times(v) for v in e]
    return e


def grad_cmd(a, out, group_file, ref_dir):
    models = a.models if os.path.isabs(a.models) else str(Path.cwd() / a.models)
    cmd = [PY, str(HERE / 'ci_grad.py'), '--cases', str(Path(a.cases).resolve()), '--map-root', str(Path(a.map_root).resolve()),
           '--models', models, '--world', a.world, '--domain', a.world, '--groups', f'@{Path(group_file).resolve()}',
           '--task-root', str(ROOT), '--out', str(Path(out).resolve()), '--arena-tag', a.arena,
           '--steps', str(a.steps), '--starts', str(a.starts), '--keep', a.keep, '--abstain-logit', str(a.abstain_logit)]
    if ref_dir:
        cmd += ['--ref-b-picks', str((Path(ref_dir) / 'picks').resolve())]
    if a.device:
        cmd += ['--device', a.device]
    return cmd


def run_grad(a, out, groups, ref_dir):
    out.mkdir(parents=True, exist_ok=True)
    gf = out / 'groups.txt'
    gf.write_text('\n'.join(groups) + '\n')
    cmd = grad_cmd(a, out, gf, ref_dir)
    env = dict(os.environ, PYTHONPATH=f'{ROOT / "src"}:{HERE}', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '6'))
    env.pop('NEDM_VEHICLE', None)
    t0 = time.time()
    with open(out / 'planner.log', 'w') as fh:
        fh.write(' '.join(cmd) + '\n'); fh.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f'ci_grad failed ({rc}); see {out / "planner.log"}')
    return cmd, time.time() - t0


def pick_records(out, groups):
    recs = {}
    for g in groups:
        p = json.load(open(out / 'picks' / f'{g}.json'))
        recs[g] = {arm: strip_times(p['arms'].get(arm)) for arm in ('B', 'G')}
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arena', required=True); ap.add_argument('--world', choices=['crm', 'rigid'], default='crm')
    ap.add_argument('--cases', required=True); ap.add_argument('--map-root', default=None)
    ap.add_argument('--groups', default='all'); ap.add_argument('--first', type=int, default=None)
    ap.add_argument('--models', required=True); ap.add_argument('--model-tag', required=True)
    ap.add_argument('--ref-b-picks', default=None, help='recorded ag_picks.py --mode free folder of the same ensemble')
    ap.add_argument('--require-ref-all', action='store_true', help='every planned group must have a recorded CEM pick to check')
    ap.add_argument('--rerun-check', type=int, default=0, metavar='N', help='re-plan the first N groups and require identical files')
    ap.add_argument('--steps', type=int, default=60); ap.add_argument('--starts', type=int, default=17)
    ap.add_argument('--keep', choices=['pessimistic', 'mean'], default='pessimistic'); ap.add_argument('--abstain-logit', type=float, default=0.3)
    ap.add_argument('--allow-domain-mismatch', action='store_true')
    ap.add_argument('--out', required=True); ap.add_argument('--force', action='store_true'); ap.add_argument('--device', default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    assert not os.environ.get('NEDM_VEHICLE'), 'NEDM_VEHICLE is set: unset it (the planner is vehicle-free, but the rule is global)'
    a.map_root = Path(a.map_root) if a.map_root else AP.default_map_root(a.arena)
    a.cases = Path(a.cases)
    out = Path(a.out)
    if out.exists():
        if not a.force:
            raise SystemExit(f'{out} exists (use --force to replace it)')
        shutil.rmtree(out)
    cases = AP.case_files(a.cases)
    groups, how = AP.declared_groups(a.groups, a.arena, a.cases, a.first)
    chk = ag_map_check.check(a.map_root, [str(cases[g]) for g in groups], ROOT)
    if not chk['ok']:
        raise SystemExit(f'map/arena check failed: {chk["problems"]}')
    arena_dir = next(iter(chk['arenas']))
    assert os.path.basename(arena_dir)[len('arena_'):].split('_')[0] == a.arena, f'cases name arena {arena_dir}, not {a.arena}'
    minfo = AP.models_info(a.models, a.world, a.allow_domain_mismatch)
    ref_dir = Path(a.ref_b_picks) if a.ref_b_picks else None
    ref_man = None
    if ref_dir is not None:
        ref_man = json.load(open(ref_dir / 'ag_picks.json'))
        assert ref_man['mode'] == 'free' and ref_man['world'] == a.world, (ref_man['mode'], ref_man['world'])
        ref_models = sorted(m['sha256'] for m in ref_man['models'])
        assert ref_models == sorted(m['sha256'] for m in minfo), f'--ref-b-picks {ref_dir} was planned with other checkpoints'
    cmd, wall = run_grad(a, out, groups, ref_dir)
    summary = json.load(open(out / 'summary.json'))
    problems = []
    if summary['n_groups'] != len(groups):
        problems.append(f"planned {summary['n_groups']} of {len(groups)} groups")
    if summary['model_kinds'] != ['ci_train']:
        problems.append(f"model kinds {summary['model_kinds']}")
    for k in ('starts_reproduced_all', 'keep_best_ok_all', 'G_not_worse_than_B'):
        if not summary['checks'].get(k):
            problems.append(f'ci_grad check {k} false')
    if not summary['per_arm']['G'].get('contract_all_valid'):
        problems.append('a G route breaks the route contract')
    rows, prob2 = AP.check_picks(out, groups, 'G', cases)
    problems += prob2
    lock = (out / 'PICKS_LOCKED.sha256').read_text().split()[0]
    if lock != AP.lock_routes(out):
        problems.append('PICKS_LOCKED.sha256 does not match the route files')
    rb = summary.get('ref_b', {})
    ref = None
    if ref_dir is not None:
        ref = dict(dir=str(ref_dir), checked=rb.get('checked'), match=rb.get('match'), z_equal=rb.get('z_equal'), mismatch=rb.get('mismatch', [])[:20],
                   ref_lock=ref_man['picks_locked_sha256'])
        if rb.get('match') != rb.get('checked'):
            problems.append(f"CEM stage differs from the recorded picks on {rb.get('checked', 0) - rb.get('match', 0)} of {rb.get('checked')} groups")
        if a.require_ref_all and rb.get('checked') != len(groups):
            problems.append(f"only {rb.get('checked')} of {len(groups)} groups had a recorded CEM pick")
    assert not problems, problems[:10]
    # per-group rows: ag_picks fields + the refinement outcome
    for g in groups:
        p = json.load(open(out / 'picks' / f'{g}.json'))
        eG, eB = p['arms']['G'], p['arms']['B']
        rows[g].update(abstained=bool(eG['abstained']), gain_logit=eG['gain'], same_route_as_B=bool(eG['same_route_as_B']), B_route_sha256=eB['route_sha256'],
                       B_P=eB['P'], B_z_mean=eB['z_mean'], z_pess=eG['z_pess'], flags=eG.get('flags'), n_valid_finals=eG['n_valid_finals'],
                       ref_b_match=eB.get('ref_match'))
    rerun = None
    if a.rerun_check:
        sub = groups[:a.rerun_check]
        with tempfile.TemporaryDirectory(prefix='ov_grad_rerun_') as td:
            o2 = Path(td) / 'rerun'
            run_grad(a, o2, sub, None)
            names = [f'{g}__{k}.json' for g in sub for k in ('G', 'B')]
            f1 = {n: (out / 'routes' / n).read_bytes() for n in names if (out / 'routes' / n).exists()}
            f2 = {n: (o2 / 'routes' / n).read_bytes() for n in names if (o2 / 'routes' / n).exists()}
            dfiles = sorted(set(f1) ^ set(f2)) + sorted(n for n in set(f1) & set(f2) if f1[n] != f2[n])
            r1, r2 = pick_records(out, sub), pick_records(o2, sub)
            # the first run carried --ref-b-picks (ref_match / ref_z_mean fields in B); drop them for the comparison
            for r in (r1, r2):
                for g in sub:
                    for k in ('ref_match', 'ref_z_mean'):
                        (r[g]['B'] or {}).pop(k, None)
            drec = sorted(g for g in sub if json.dumps(r1[g], sort_keys=True, default=str) != json.dumps(r2[g], sort_keys=True, default=str))
            rerun = dict(groups=len(sub), route_files=len(f1), identical=bool(not dfiles and not drec), differing_route_files=dfiles[:20],
                         differing_pick_records=drec[:20])
        assert rerun['identical'], f'rerun differs: {rerun}'
    pa = summary['per_arm']
    man = dict(schema='ag_picks_v1', tool='scripts/ov_grad_picks.py', tool_sha256=sha256_file(__file__), ci_grad_sha256=sha256_file(HERE / 'ci_grad.py'),
               argv=sys.argv[1:] if argv is None else argv, created=time.strftime('%Y-%m-%d %H:%M:%S'), host=os.uname().nodename,
               arena=a.arena, arena_dir=arena_dir, world=a.world, mode='grad', arm_key='G', model_tag=a.model_tag,
               planner=dict(command=cmd, family='free', arms='B then G', tag=f'cem4x64+grad{a.steps}_{a.keep}',
                            config=summary['config'], start='case layout pose at rest (standing start, all-masked history; ci_grad without --poses)'),
               straight=None, models=minfo, models_glob=a.models, cases_dir=os.path.relpath(a.cases.resolve(), ROOT),
               map_root=os.path.relpath(Path(a.map_root).resolve(), ROOT), map_check=chk,
               groups_spec=a.groups, groups_rule=how, n_groups=len(groups), groups_sha256=hashlib.sha256('\n'.join(groups).encode()).hexdigest(),
               picks_locked_sha256=lock, n_route_files=len(list((out / 'routes').glob('*.json'))),
               wall_s=round(time.time() - t0, 1), planner_wall_s=round(wall, 1), rerun_check=rerun, ref_b_check=ref,
               grad_summary=dict(changed=pa['G']['changed'], abstained=pa['G']['abstained'], n=pa['G']['n'], B_P_mean=pa['B']['P_mean'], G_P_mean=pa['G']['P_mean'],
                                 B_mean_speed=pa['B']['mean_speed'], G_mean_speed=pa['G']['mean_speed'], flags=summary.get('flags'),
                                 seconds_per_group=summary['seconds_per_group']['total'], gpu_peak_gb=summary.get('gpu_peak_gb')),
               numerics=AP.numerics(), picks=rows)
    json.dump(man, open(out / 'ag_picks.json', 'w'), indent=1, default=str)
    lk = dict(schema='ov_grad_lock_v1', dir=str(out), manifest_sha256=sha256_file(out / 'ag_picks.json'), picks_locked_sha256=lock,
              n_route_files=man['n_route_files'], n_groups=len(groups), groups_sha256=man['groups_sha256'],
              models=[dict(path=m['path'], sha256=m['sha256']) for m in minfo], created=time.strftime('%Y-%m-%d %H:%M:%S'))
    json.dump(lk, open(out / 'OV_GRAD_LOCK.json', 'w'), indent=1)
    sp = np.array([r['mean_speed'] for r in rows.values() if r])
    print(json.dumps(dict(out=str(out), model_tag=a.model_tag, groups=len(groups), changed=pa['G']['changed'], abstained=pa['G']['abstained'],
                          B_P_mean=round(pa['B']['P_mean'], 4), G_P_mean=round(pa['G']['P_mean'], 4), G_mean_speed=round(float(sp.mean()), 3),
                          ref_b=None if ref is None else f"{ref['match']}/{ref['checked']}", rerun_identical=None if rerun is None else rerun['identical'],
                          lock=lock[:16], manifest=lk['manifest_sha256'][:16], wall_s=man['wall_s'])), flush=True)
    return man


if __name__ == '__main__':
    main()
