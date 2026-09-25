#!/usr/bin/env python3
"""Evaluation picks for one arena, one world, one model ensemble and one mode (arena_gator_20260925, module E6a;
PLAN 2.2-2.3, 3, 7.1-7.4; REVIEW_R2 amendment 7).

Modes (every mode plans from the case layout pose at rest, i.e. a standing start):
  free       scripts/ci_planner.py --family free --world <world> --arms B      CEM 4 x 64, speed free
  fixed2     scripts/ci_planner.py --family free --fixed2 --world rigid --arms B  CEM 4 x 64, geometry only at 2 m/s
             (ga_planner.py --fixed2 cannot load ci_train checkpoints, REVIEW_R2 1.4)
  straight6  the designed anchor with lateral offset 0 at 6 m/s of the night-2 proposal pool, built with exactly the
             code of scripts/ag_dev_headroom.py (gen_planner.proposal_pool from route_00 and the layout pose, rng
             md5(group + 'crm_proposal')[:8], anchor_index(0, 6)); the route file is the same dump, so on the spread
             arenas it is byte-identical to the headroom scan's straight 6 m/s route (checked by --selfcheck-straight)
  straight2  the same anchor at 2 m/s (anchor_index(0, 2); gen_planner.plan mode 'straight2')
Before anything is planned, scripts/ag_map_check.py must pass (one arena; its BMP sha256 equals the map root's). For
planner modes every checkpoint must be model_kind 'ci_train' trained on this world (--domain-filter), unless
--allow-domain-mismatch.

Groups (--groups): 'all' (every case in --cases), '@file' (one id per line), a json list, or a declared-subset json
(suites/soil_unseen_subset.json: arenas.<arena>.groups; suites/f104_indist_200.json: groups); '--first N' keeps the N
lowest md5(group id) of that list (the declared order of every subset of this study). Every requested group must have
a case file.

Output directory (--out, must not exist unless --force):
  picks/<g>.json, routes/<route id>.json, summary.json, tasks.json, PICKS_LOCKED.sha256   (ga_planner / ci_planner files;
                 straight modes write the same file kinds, arm key 'S')
  groups.txt     the declared groups, one per line
  ag_picks.json  manifest: arena, world, mode, arm key, model tag, checkpoints + sha256, map root + observation sha256,
                 map check, groups sha256, planner command, per-group pick summary, lock, rerun check
--rerun-check plans everything a second time into a scratch folder and requires identical route files (bytes) and
identical pick records (every field except wall times); the result is stored in the manifest.

  PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
  $PY scripts/ag_picks.py --arena g260 --world rigid --mode fixed2 --cases $K3/cases/test_g260/cases \
      --groups $K3/suites/soil_unseen_subset.json --models "$K3/e5/train/M1_rigid/M1a_rigid_deploy_s*.pt" --model-tag M1a \
      --out $K3/e6/picks/rigid/g260/M1a_fixed2 --rerun-check
"""
import argparse, glob, hashlib, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ag_map_check                       # noqa: E402
import ag_tasklib as L                    # noqa: E402

MODES = ('free', 'fixed2', 'straight6', 'straight2')
F104_MAP = ROOT / 'artifacts/traverse/crm_f104_v1/map_root'
PY = sys.executable


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def md5hex(s):
    return hashlib.md5(s.encode()).hexdigest()


def default_map_root(arena):
    return F104_MAP if arena == 'f104' else L.K3 / 'map_roots' / arena


def case_files(cases_dir):
    return {p.stem: p for p in Path(cases_dir).glob('*.json') if p.name != 'cases.json'}


def declared_groups(spec, arena, cases_dir, first=None):
    """(groups in md5 order, description) from the --groups spec."""
    have = case_files(cases_dir)
    if spec == 'all':
        gs, how = sorted(have), 'all cases'
    elif spec.startswith('@'):
        gs, how = [s.strip() for s in open(spec[1:]) if s.strip()], f'list {spec[1:]}'
    elif spec.endswith('.json'):
        j = json.load(open(spec))
        if isinstance(j, list):
            gs, how = [str(x) for x in j], f'json list {spec}'
        elif isinstance(j, dict) and 'arenas' in j and arena in j['arenas']:
            gs, how = list(j['arenas'][arena]['groups']), f'declared subset {spec} arenas.{arena}'
        elif isinstance(j, dict) and 'groups' in j and isinstance(j['groups'], list):
            gs, how = [str(x) for x in j['groups']], f'declared subset {spec} groups'
        else:
            raise SystemExit(f'{spec}: no group list for arena {arena}')
    else:
        gs, how = [s.strip() for s in spec.split(',') if s.strip()], 'comma list'
    assert len(set(gs)) == len(gs), 'duplicate groups in the declared list'
    missing = [g for g in gs if g not in have]
    assert not missing, f'{len(missing)} declared groups have no case file in {cases_dir}: {missing[:5]}'
    gs = sorted(gs, key=md5hex)
    if first:
        gs = gs[:first]; how += f', first {first} by md5(group id)'
    return gs, how


# ------------------------------------------------------------------------------------------------------------------
# straight anchors (exactly ag_dev_headroom.py / ag_spread_headroom.py, cruise speed as a parameter)
# ------------------------------------------------------------------------------------------------------------------
def straight_route(P, case_path, cruise):
    """(group, pose, goal, route or None, index): the offset-0 anchor at `cruise` m/s of the night-2 proposal pool."""
    case = json.load(open(case_path)); g = case['id']; lay = case['layout']
    pose = [lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']]
    base = {k: np.asarray(v, float) for k, v in json.load(open(Path(case_path).parent / 'routes' / g / 'route_00.json')).items()
            if k in ('waypoints', 'speeds', 'stations', 'headings')}
    base['meta'] = {}
    c1, tries = P.proposal_pool(base, pose, np.random.default_rng(int(hashlib.md5((g + 'crm_proposal').encode()).hexdigest()[:8], 16)))
    i = P.anchor_index(c1, 0.0, float(cruise))
    return g, pose, [float(v) for v in case['goal_xy']], (None if i is None else c1[i]), i, case


def straight_dump(r, g, i, cruise):
    return {'waypoints': np.asarray(r['waypoints']).tolist(), 'speeds': np.asarray(r['speeds']).tolist(),
            'stations': np.asarray(r['stations']).tolist(), 'headings': np.asarray(r['headings']).tolist(),
            'meta': {'candidate': f'crm_eval_straight{int(cruise)}', 'scene_id': g, 'pool': 'proposal', 'cand_index': i}}


def lock_routes(out):
    lock = hashlib.sha256()
    for p in sorted((Path(out) / 'routes').glob('*.json')):
        lock.update(p.name.encode()); lock.update(hashlib.sha256(p.read_bytes()).digest())
    return lock.hexdigest()


def write_straight(out, cases, groups, cruise, map_root, world, task_root):
    import gen_planner as P
    import f104_n2_iter as IT
    P.DS.init_map(str(map_root))
    name = f'straight{int(cruise)}'
    (out / 'routes').mkdir(parents=True, exist_ok=True); (out / 'picks').mkdir(exist_ok=True)
    tasks, rows, t0 = [], [], time.time()
    for gi, g in enumerate(groups):
        cp = cases[g]
        g2, pose, goal, r, i, case = straight_route(P, cp, cruise)
        assert g2 == g
        summ = dict(group=g, world=world, mode=name, stratum=case.get('evaluation_stratum'), pose=[float(v) for v in pose], goal=goal,
                    source=dict(pose='layout', goal='case', base='route_00', history='all_masked'), arms={})
        if r is None:
            summ['arms']['S'] = None
        else:
            rid = f'{g}__{name}'; h = IT.route_sha256(r)
            p = out / 'routes' / f'{rid}.json'
            json.dump(straight_dump(r, g, i, cruise), open(p, 'w'))
            summ['arms']['S'] = dict(route_id=rid, arm='S', label=name, tag=f'anchor_offset0_{int(cruise)}mps', index=int(i), kind='anchor',
                                     route_sha256=h, mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()),
                                     length_m=float(np.asarray(r['stations'])[-1]), max_lateral_m=float(r['meta'].get('max_lateral_m', 0.0)),
                                     P=None, z_mean=None, T=float(IT.route_time(r)),
                                     start_dist_to_pose_m=float(np.linalg.norm(np.asarray(r['waypoints'])[0] - np.asarray(pose[:2]))))
            tasks.append(dict(id=rid, group=g, case=os.path.relpath(Path(cp).resolve(), task_root), route=os.path.relpath(p.resolve(), task_root),
                              run=True, tier=gi, episode_seed=L.md5_int(rid), arms=['S'], sha256=h))
        json.dump(summ, open(out / 'picks' / f'{g}.json', 'w'), indent=1)
        rows.append(summ)
    json.dump(tasks, open(out / 'tasks.json', 'w'), indent=1)
    (out / 'PICKS_LOCKED.sha256').write_text(lock_routes(out) + '  routes/*.json (name + content, sorted)\n')
    summary = dict(world=world, mode=name, n_groups=len(rows), arms=['S'], tags={'S': f'anchor_offset0_{int(cruise)}mps'}, map_root=str(map_root),
                   n_distinct_routes=len(tasks), n_missing=int(sum(1 for r in rows if r['arms']['S'] is None)), wall_s_total=time.time() - t0,
                   construction='gen_planner.proposal_pool(route_00, layout pose, rng md5(group+crm_proposal)[:8]); anchor_index(0, %g)' % cruise)
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
    return summary


def selfcheck_straight():
    """Rebuild the E1b spread-headroom and E3a dev straight 6 m/s routes with write_straight's code; byte-identical."""
    import gen_planner as P
    res = {}
    for arena, cdir, ref in [('g217', L.K3 / 'cases/dev_g217/cases', L.K3 / 'e3/dev_headroom/straight6/routes')] + \
            [(a, L.K3 / f'cases/test_{a}/cases', L.K3 / 'e3/spread_headroom/straight6/routes') for a in ('g258', 'g268', 'g263', 'g241')]:
        P.DS.init_map(str(default_map_root(arena)))
        refs = sorted(ref.glob(f'{arena}_*__straight6.json'))
        n = 0
        for rp in refs:
            g = rp.name.split('__')[0]
            g2, pose, goal, r, i, case = straight_route(P, Path(cdir) / f'{g}.json', 6.0)
            assert r is not None and json.dumps(straight_dump(r, g, i, 6.0)).encode() == rp.read_bytes(), f'straight6 rebuild differs for {g}'
            n += 1
        res[arena] = n
    return res


# ------------------------------------------------------------------------------------------------------------------
def models_info(pattern, world, allow_mismatch):
    import torch
    paths = sorted(glob.glob(pattern))
    assert paths, f'no checkpoint matches {pattern}'
    info = []
    for p in paths:
        ck = torch.load(p, map_location='cpu', weights_only=False)
        assert ck.get('model_kind') == 'ci_train', f'{p}: model_kind {ck.get("model_kind")!r} (ci_train deploy checkpoints only)'
        d = dict(path=os.path.relpath(p, ROOT) if str(Path(p).resolve()).startswith(str(ROOT)) else p, sha256=sha256_file(p),
                 tag=ck.get('tag'), seed=ck.get('seed'), mode=ck.get('mode'), domain_filter=ck.get('domain_filter'), arch=ck.get('arch'),
                 cond=ck.get('cond'), ctx_mode=ck.get('ctx_mode'), train_rows=ck.get('train_rows'), ds=ck.get('ds'))
        if d['domain_filter'] != world and not allow_mismatch:
            raise SystemExit(f'{p}: trained with --domain-filter {d["domain_filter"]}, planning world {world} (--allow-domain-mismatch to override)')
        if d['mode'] != 'deploy':
            print(f'WARNING {p}: mode {d["mode"]!r}, not a deploy checkpoint', flush=True)
        info.append(d)
    return info


def numerics():
    import torch
    return dict(device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', torch=torch.__version__,
                note='planner process runs with torch defaults: cudnn.allow_tf32 = %s (TF32 convolutions on this GPU), matmul.allow_tf32 = %s; '
                     'as every earlier local pick of this study; picks are deterministic on this machine (--rerun-check)'
                     % (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32))


def planner_cmd(a, out, group_file):
    models = a.models if os.path.isabs(a.models) else str(Path.cwd() / a.models)      # the glob stays a glob
    cmd = [PY, str(HERE / 'ci_planner.py'), '--family', 'free', '--cases', str(Path(a.cases).resolve()), '--map-root', str(Path(a.map_root).resolve()),
           '--models', models, '--world', a.world, '--arms', 'B', '--groups', f'@{Path(group_file).resolve()}', '--ref-picks', '/nonexistent_no_reference',
           '--task-root', str(L.K3), '--out', str(Path(out).resolve())]
    if a.mode == 'fixed2':
        cmd.append('--fixed2')
    if a.device:
        cmd += ['--device', a.device]
    return cmd


def run_planner(a, out, group_file):
    cmd = planner_cmd(a, out, group_file)
    env = dict(os.environ, PYTHONPATH=f'{ROOT / "src"}:{HERE}', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '6'))
    env.pop('NEDM_VEHICLE', None)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'planner.log', 'w') as fh:
        fh.write(' '.join(cmd) + '\n'); fh.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f'planner failed ({rc}); see {out / "planner.log"}')
    return cmd


def plan_into(a, out, groups, cases):
    out.mkdir(parents=True, exist_ok=True)
    gf = out / 'groups.txt'
    gf.write_text('\n'.join(groups) + '\n')
    if a.mode in ('free', 'fixed2'):
        cmd = run_planner(a, out, gf)
        summary = json.load(open(out / 'summary.json'))
        want_tag = 'n2iter_cem4x64' + ('_fixed2' if a.mode == 'fixed2' else '')
        assert summary['arms'] == ['B'] and summary['tags']['B'] == want_tag, (summary['arms'], summary['tags'])
        assert bool(summary['fixed2']) == (a.mode == 'fixed2') and summary['world'] == a.world
        assert summary['model_kinds'] == ['ci_train'], summary['model_kinds']
        assert summary['n_groups'] == len(groups), (summary['n_groups'], len(groups))
        return cmd, summary, 'B'
    cruise = 6.0 if a.mode == 'straight6' else 2.0
    return None, write_straight(out, cases, groups, cruise, a.map_root, a.world, L.K3), 'S'


def pick_records(out, groups, arm):
    recs = {}
    for g in groups:
        p = json.load(open(out / 'picks' / f'{g}.json'))
        e = p['arms'].get(arm)
        recs[g] = None if e is None else {k: v for k, v in e.items() if k != 'wall_s'}
    return recs


def check_picks(out, groups, arm, cases):
    """Every pick has its route file, the file's content hash equals the recorded one, starts at the case pose."""
    import f104_n2_iter as IT
    rows, problems = {}, []
    for g in groups:
        p = json.load(open(out / 'picks' / f'{g}.json'))
        e = p['arms'].get(arm)
        if e is None:
            problems.append(f'{g}: no pick'); rows[g] = None; continue
        rf = out / 'routes' / f"{e['route_id']}.json"
        r = json.load(open(rf))
        h = IT.route_sha256(r)
        if h != e['route_sha256']:
            problems.append(f'{g}: route file hash {h[:12]} != pick {e["route_sha256"][:12]}')
        case = json.load(open(cases[g])); lay = case['layout']
        d0 = float(np.linalg.norm(np.asarray(r['waypoints'])[0] - np.asarray(lay['start_xy'])))
        d1 = float(np.linalg.norm(np.asarray(r['waypoints'])[-1] - np.asarray(case['goal_xy'])))
        if d0 > 0.25 or d1 > 0.25:
            problems.append(f'{g}: route starts {d0:.2f} m from the pose / ends {d1:.2f} m from the goal')
        v = np.asarray(r['speeds'], float)
        rows[g] = dict(route_id=e['route_id'], route_sha256=h, route_file=str(rf.relative_to(out)), file_sha256=sha256_file(rf),
                       P=e.get('P'), z_mean=e.get('z_mean'), mean_speed=float(v[1:-1].mean()), max_speed=float(v.max()),
                       length_m=float(np.asarray(r['stations'])[-1]), stratum=case.get('evaluation_stratum'))
    return rows, problems


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arena', required=True, help='short arena name (f104, g203, g260, ...)')
    ap.add_argument('--world', choices=['crm', 'rigid'], required=True)
    ap.add_argument('--mode', choices=MODES, required=True)
    ap.add_argument('--cases', required=True, help='suite case dir (<g>.json + routes/<g>/route_00.json)')
    ap.add_argument('--map-root', default=None, help='default: K3/map_roots/<arena> (f104: crm_f104_v1/map_root)')
    ap.add_argument('--groups', default='all', help="'all' | @file | json list | declared subset json")
    ap.add_argument('--first', type=int, default=None, help='keep the N lowest md5(group id) of the declared list')
    ap.add_argument('--models', default=None, help='glob of ci_train deploy checkpoints (planner modes)')
    ap.add_argument('--model-tag', default=None, help='model name used in run ids (M1a, M3b, H, G, ...); straight modes: none')
    ap.add_argument('--allow-domain-mismatch', action='store_true')
    ap.add_argument('--out', required=True)
    ap.add_argument('--force', action='store_true', help='replace an existing --out')
    ap.add_argument('--rerun-check', action='store_true')
    ap.add_argument('--device', default=None)
    ap.add_argument('--selfcheck-straight', action='store_true', help='first rebuild the headroom straight 6 m/s routes byte for byte')
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.mode == 'fixed2':
        assert a.world == 'rigid', 'fixed2 is the rigid read-out (ci_planner --fixed2 --world rigid, REVIEW_R2 amendment 7)'
    planner = a.mode in ('free', 'fixed2')
    if planner:
        assert a.models and a.model_tag, '--models and --model-tag are required for planner modes'
    else:
        assert not a.models, 'straight modes take no models'
    a.map_root = Path(a.map_root) if a.map_root else default_map_root(a.arena)
    a.cases = Path(a.cases)
    out = Path(a.out)
    if out.exists():
        if not a.force:
            raise SystemExit(f'{out} exists (use --force to replace it)')
        shutil.rmtree(out)
    cases = case_files(a.cases)
    groups, how = declared_groups(a.groups, a.arena, a.cases, a.first)
    chk = ag_map_check.check(a.map_root, [str(cases[g]) for g in groups], ROOT)
    if not chk['ok']:
        raise SystemExit(f'map/arena check failed: {chk["problems"]}')
    arena_dir = next(iter(chk['arenas']))
    short = os.path.basename(arena_dir)[len('arena_'):].split('_')[0]
    assert short == a.arena, f'cases name arena {arena_dir}, not {a.arena}'
    sc = selfcheck_straight() if a.selfcheck_straight else None
    minfo = models_info(a.models, a.world, a.allow_domain_mismatch) if planner else None
    cmd, summary, arm = plan_into(a, out, groups, cases)
    rows, problems = check_picks(out, groups, arm, cases)
    assert not problems, problems[:5]
    lock = (out / 'PICKS_LOCKED.sha256').read_text().split()[0]
    assert lock == lock_routes(out), 'PICKS_LOCKED.sha256 does not match the route files'
    rerun = None
    if a.rerun_check:
        with tempfile.TemporaryDirectory(prefix='ag_picks_rerun_') as td:
            o2 = Path(td) / 'rerun'
            plan_into(a, o2, groups, cases)
            files1 = {p.name: p.read_bytes() for p in (out / 'routes').glob('*.json')}
            files2 = {p.name: p.read_bytes() for p in (o2 / 'routes').glob('*.json')}
            rec1, rec2 = pick_records(out, groups, arm), pick_records(o2, groups, arm)
            diff_files = sorted(set(files1) ^ set(files2)) + sorted(k for k in set(files1) & set(files2) if files1[k] != files2[k])
            diff_recs = sorted(g for g in groups if json.dumps(rec1[g], sort_keys=True, default=str) != json.dumps(rec2[g], sort_keys=True, default=str))
            lock2 = (o2 / 'PICKS_LOCKED.sha256').read_text().split()[0]
            rerun = dict(identical=bool(not diff_files and not diff_recs and lock2 == lock), lock_rerun=lock2, route_files=len(files1),
                         differing_route_files=diff_files[:20], differing_pick_records=diff_recs[:20])
        assert rerun['identical'], f'rerun differs: {rerun}'
    man = dict(schema='ag_picks_v1', tool='scripts/ag_picks.py', tool_sha256=sha256_file(__file__),
               argv=sys.argv[1:] if argv is None else argv, created=time.strftime('%Y-%m-%d %H:%M:%S'), host=os.uname().nodename,
               arena=a.arena, arena_dir=arena_dir, world=a.world, mode=a.mode, arm_key=arm, model_tag=a.model_tag,
               planner=dict(command=cmd, family='free', arms='B', tag=summary.get('tags', {}).get(arm), fixed2=a.mode == 'fixed2',
                            start='case layout pose at rest (standing start, all-masked history)') if planner else None,
               straight=dict(cruise_mps=6.0 if a.mode == 'straight6' else 2.0, construction=summary.get('construction')) if not planner else None,
               models=minfo, models_glob=a.models, cases_dir=os.path.relpath(a.cases.resolve(), ROOT),
               map_root=os.path.relpath(Path(a.map_root).resolve(), ROOT), map_check=chk,
               groups_spec=a.groups, groups_rule=how, n_groups=len(groups), groups_sha256=hashlib.sha256('\n'.join(groups).encode()).hexdigest(),
               picks_locked_sha256=lock, n_route_files=len(list((out / 'routes').glob('*.json'))),
               wall_s=round(time.time() - t0, 1), planner_wall_s=summary.get('wall_s_total'), rerun_check=rerun, selfcheck_straight=sc,
               numerics=numerics() if planner else None,
               picks=rows)
    json.dump(man, open(out / 'ag_picks.json', 'w'), indent=1, default=str)
    sp = np.array([r['mean_speed'] for r in rows.values() if r])
    Pv = [r['P'] for r in rows.values() if r and r['P'] is not None]
    print(json.dumps(dict(out=str(out), arena=a.arena, world=a.world, mode=a.mode, model_tag=a.model_tag, groups=len(groups),
                          route_files=man['n_route_files'], lock=lock[:16], mean_speed=float(sp.mean()) if len(sp) else None,
                          mean_pred_P=float(np.mean(Pv)) if Pv else None, rerun_identical=None if rerun is None else rerun['identical'],
                          selfcheck_straight=sc, wall_s=man['wall_s'])), flush=True)
    return man


if __name__ == '__main__':
    main()
