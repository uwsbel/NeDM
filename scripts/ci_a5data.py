#!/usr/bin/env python
"""Task files for the short-approach closed loop (PLAN S2) and the on-distribution continuation data (PLAN S3) of
crm_improve_20260922.  Generalises generalist_20260921's ``ga_approach.py`` (3 s approach, frame 60) to any approach
length L in {0.5, 1, 3} s (decision frame F = round(L / 0.05) = 10, 20, 60) and to the 1,200 twin training groups.

Stages (``--stage``); nothing here submits cluster jobs or runs Chrono:

  approach-routes  ``--cases DIR`` (the 800 suite groups, ``generalist_20260921/A_adapt/suite/cases``, or the 1,200
              twin groups, ``fdm_f104_50h_20260909/cases_night2/cases``): per group the same two candidates and the
              same rule as ga_approach (straight start-goal line at a constant 3 m/s, waypoints every 0.5 m, versus
              route_00 at 3 m/s; lower mean |grade| over the first 12 m wins, ties -> straight, route_00 if the straight
              line fails validation), plus the geometry check that route_00 IS the straight line (largest lateral
              distance of a route_00 waypoint from the line < 0.01 m and start yaw within 0.01 deg of the line
              heading); groups where they differ are listed.  Suite routes are compared with K1's
              ``A_adapt/a5/approach/<g>.json`` by route content hash.  Twin groups: split from the twin file
              (``crm_night2_v1/datasets/twin_crm.npz``) asserted equal to the case file's split; no suite group.
              Writes ``<out>/approach_<set>/<g>.json`` and ``<out>/approach_<set>_index.json``.

  pass1-tasks  CRM rows ``{id: <g>__p1_<len>, group, case, route, run, tier, episode_seed, extra: ['--horizon-s', L]}``
              with paths relative to CRM_ROOT (cases: the existing copies ``generalist/suite/cases`` and
              ``cases/night2``; approach routes under ``crm_improve/pass1_<set>/approach/``) and rigid rows with
              absolute paths (cases: the existing copies under generalist_20260921/a5/cases and
              fdm_f104_50h_20260909/cases_night2_v1; approach routes under G2/pass1_<set>/approach/), arena f104,
              shard md5(group) % 6, mode native, extra ['--horizon-s', L].  Builds: suite at L = 0.5 and 1 s, twin at
              L = 0.5, 1 and 3 s, both worlds; one file per (set, world, L) and one merged file per (set, world).
              Writes the rsync commands (``SHIP_pass1.txt``); they are NOT run.

  analyze     ``--runs DIR [DIR ...] --worlds crm rigid --length L``: from every pass-1 run dir whose name ends in the
              run suffix (default ``__p1_<len>``) the decision state at F = round(L / 0.05): pose, full 17-column
              state, vx, the last applied action, progress along the approach route, the CRM sinkage increase
              (ga_approach's per-wheel formula, frame min(F, last extra row) minus frame 0), the moving flag
              (vx > --vx-min and sinkage increase < 0.1 m; --vx-min defaults to 1.0 m/s for F >= 60 and 0.3 m/s below,
              because at 0.5 s the soil vehicle is still accelerating: K1 pass-1 median vx at frame 10 is 0.89 m/s),
              and the history window(s) (T = --hist-T, default 40) cut at F in the ga_planner convention.  A
              recording with more than F rows uses row F; one with exactly F rows uses the collector's
              terminal_state / terminal_pose (the state after F intervals).  Writes ``decision_<world>.json``,
              ``hist_<world>_T<T>/<g>.npz``, ``poses_<world>.json`` (analysis set = moving in every world, the K1
              convention), ``poses_<world>_all.json`` (every group with a decision state) in the ``ga_planner.py
              --poses`` format, and ``pass1_state.json``.

  continuations  ``--decision DIR --world crm|rigid``: for every twin-group decision state K = 6 continuation routes from
              the decision pose to the goal: 2 from ci_planner.candidates(family='free'), 2 from family='cont_head',
              1 CEM pick of the K1 H ensemble with family free, 1 with family cont_head; deduplicated by route content
              sha256 (a duplicate slot keeps its row with run=false and ref_id); rows <g>__p1_<len>__c<slot> for
              crm_collect_ext.py --mode branch (CRM: ``extra ['--mode', 'branch', '--branch-frame', F, '--branch-route',
              <absolute cluster path>, '--horizon-s', '120']``) and gen_collect_ext.py (rigid: mode 'branch', extra
              ['--branch-frame', F, '--branch-route', <absolute path>, '--horizon-s', '120'], arena, shard); the prefix
              is the approach route of pass 1 (checked against the run's command_reference.npz) and an anchors json in
              the ga_branch_dataset.py format (episode = <g>__p1_<len>, F, cls a5_<len>, group, split from the twin
              file, pose_F, vx_F, remaining_m).  Asserts: no suite group anywhere; twin split respected; unique ids and
              seeds.  Default input = the analysis set of the analyze stage (--all-groups: every decision state).

  merge       ``--inputs f1 f2 ... --out-file X``: one task (or anchors) file per world from several lengths (queue-cap
              convention); asserts unique ids / seeds, one world, no suite group (unless --allow-suite-rows, for the
              pass-1 evaluation drives), twin split.

  selftest-analyze / selftest-continuations  see NOTES_ci_a5data.md.

Run from the repo root: ``PYTHONPATH=src:scripts OMP_NUM_THREADS=6 python scripts/ci_a5data.py --stage ...``.
"""
from __future__ import annotations

import argparse, fnmatch, glob, hashlib, json, math, os, shutil, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(HERE))
import ga_approach as GA5            # straight_route, route00_at_speed, validate_route, grade_profile, history_window, wheel_sinkage

K1 = ROOT / 'artifacts/traverse/generalist_20260921'
K2 = ROOT / 'artifacts/traverse/crm_improve_20260922'
OUT = K2 / 'a5data'
SUITE_CASES = K1 / 'A_adapt/suite/cases'
TWIN_CASES = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases'
TWIN_NPZ = ROOT / 'artifacts/traverse/crm_night2_v1/datasets/twin_crm.npz'
K1_APPROACH = K1 / 'A_adapt/a5/approach'
K1_PASS1 = {'crm': K1 / 'A_adapt/a5/pass1_out_crm/runs', 'rigid': K1 / 'A_adapt/a5/pass1_out_rigid/runs'}
K1_H_MODELS = str(K1 / 'A_adapt/train/deploy_v1/H_deploy_s*.pt')
MAP_ROOT = ROOT / 'artifacts/traverse/crm_f104_v1/map_root'
GRID = GA5.GRID
ARENA = GA5.ARENA

CRM_ROOT = '/work1/dannegrut/harry/experiments/crm_f104_20260916'
G2 = '/work1/dannegrut/harry/experiments/crm_improve_20260922'
CRM_SUB = 'crm_improve'
# existing cluster copies of the case files (sha256 of all 2,000 files == the local copies, checked 2026-09-22)
CLUSTER_CASES = {'suite': dict(crm='generalist/suite/cases', rigid='/work1/dannegrut/harry/experiments/generalist_20260921/a5/cases'),
                 'twin': dict(crm='cases/night2', rigid='/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/cases_night2_v1')}
LOCAL_CASES = {'suite': SUITE_CASES, 'twin': TWIN_CASES}
BUILDS = {'suite': (0.5, 1.0), 'twin': (0.5, 1.0, 3.0)}      # PLAN S2 (suite) and S3 (twin) approach lengths
SUITE_PATTERNS = ('f104_crm_eval_group_*', 'f104_g1_test_group_*', 'f104_pair_group_*')
SUITE_PATTERNS += tuple(['g260_test_group_*', 'g271_test_group_*', 'g251_test_group_*', 'g247_test_group_*', 'g203_heldout_group_*', 'g228_heldout_group_*', 'g217_dev_group_*'])   # arena_gator_20260925 E1 suites (additive)
TWIN_PATTERN = 'f104_v2_group_*'
DT = 0.05
SHARDS = 6
ARENA_TAG = 'f104'
SINK_INC_MAX_M = GA5.SINK_INC_MAX_M
HIST_T = 40
K_CONT = 6
BRANCH_START_TOL_M = 1.0          # crm_collect_ext / gen_collect_ext --branch-start-tol-m default
BRANCH_GOAL_TOL_M = 0.5           # crm_collect_ext --branch-goal-tol-m default
BRANCH_HORIZON = '120'
SLOTS = (('free', 'sample'), ('free', 'sample'), ('cont_head', 'sample'), ('cont_head', 'sample'), ('free', 'cem'), ('cont_head', 'cem'))


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


jdump = GA5.jdump
sha256_file = GA5.sha256_file


def seed_of(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def shard_of(g):
    return int(hashlib.md5(g.encode()).hexdigest(), 16) % SHARDS


def ltag(L):
    """0.5 -> '0p5', 1 -> '1', 3 -> '3' (the length tag in ids, file names and classes)."""
    return f'{float(L):g}'.replace('.', 'p')


def frame_of(L):
    F = int(round(float(L) / DT))
    assert abs(float(L) / DT - F) < 1e-9 and F >= 1, f'approach length {L} s is not a positive multiple of 50 ms'
    return F


def is_suite(s):
    return any(fnmatch.fnmatch(s, p) for p in SUITE_PATTERNS)


def check_blacklist_source():
    """Our suite patterns must equal the builders' blacklist (ga_build_mixed.BLACKLIST)."""
    import ga_build_mixed as GBM
    assert sorted(GBM.BLACKLIST) == sorted(SUITE_PATTERNS), (GBM.BLACKLIST, SUITE_PATTERNS)
    return GBM.blacklisted


def list_groups(cases_dir):
    return sorted(p.stem for p in Path(cases_dir).glob('*.json') if p.name != 'cases.json')


def detect_set(groups):
    if all(fnmatch.fnmatch(g, TWIN_PATTERN) for g in groups):
        return 'twin'
    if all(is_suite(g) for g in groups):
        return 'suite'
    raise SystemExit(f'cannot tell the case set from the group names (mixed or unknown): {groups[:3]}')


_TWIN = {}


def twin_split():
    """{group: split} from the twin file (the only split: 1,089 train / 56 val / 55 test)."""
    if not _TWIN:
        z = np.load(TWIN_NPZ, allow_pickle=True)
        for g, s in zip(z['group'].astype(str).tolist(), z['split'].astype(str).tolist()):
            assert _TWIN.setdefault(g, s) == s, g
        c = {s: sum(v == s for v in _TWIN.values()) for s in ('train', 'val', 'test')}
        assert len(_TWIN) == 1200 and c == {'train': 1089, 'val': 56, 'test': 55}, c
        assert not any(is_suite(g) for g in _TWIN)
    return _TWIN


def group_order(cases_dir, set_name):
    """[(group, info)] in tier order: suite.json order for the suite, sorted names for the twin groups."""
    groups = list_groups(cases_dir)
    sj = Path(cases_dir).parent / 'suite.json'
    if set_name == 'suite' and sj.exists():
        suite = json.load(open(sj))
        order = [(e['group'], dict(stratum=e['stratum'], evaluation_stratum=e['evaluation_stratum'])) for e in suite['groups']]
        assert sorted(g for g, _ in order) == groups, 'suite.json and the cases dir disagree'
        return order
    if set_name == 'twin':
        sp = twin_split()
        assert set(groups) <= set(sp), sorted(set(groups) - set(sp))[:5]
        return [(g, dict(split=sp[g])) for g in groups]
    return [(g, {}) for g in groups]


def route_content_sha(r):
    import gc_control as GC
    return GC.route_sha256(r)


# ============================================================================================================ (a)
def stage_approach_routes(a):
    import gb_crop, gc_control as GC
    import traverse_fdm_rgbd_diverse_chrono as TF
    reader_rejects = []
    cases_dir = Path(a.cases)
    groups = list_groups(cases_dir)
    set_name = a.set or detect_set(groups)
    order = group_order(cases_dir, set_name)
    grid = gb_crop.load_grid(str(a.grid))
    out = Path(a.out); rdir = out / f'approach_{set_name}'
    if rdir.exists():
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True)
    idx, n_gc_disagree, differs, k1_cmp = {}, 0, [], dict(compared=0, equal=0, differ=[])
    t0 = time.time()
    for tier, (g, info0) in enumerate(order):
        cp = cases_dir / f'{g}.json'
        case = json.load(open(cp)); assert case['id'] == g, (cp, case['id'])
        if set_name == 'twin':
            assert not is_suite(g), g
            assert case.get('split') == info0['split'], f'{g}: case split {case.get("split")} != twin split {info0["split"]}'
        lay = case['layout']
        pose = np.array([lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']], float)
        goal = np.asarray(case['goal_xy'], float)
        rp = cases_dir / 'routes' / g / 'route_00.json'
        r00 = json.load(open(rp))
        cands = {'straight': GA5.straight_route(pose[:2], goal), 'route00': GA5.route00_at_speed(r00)}
        info = {}
        for name, r in cands.items():
            reasons = GA5.validate_route(r, pose, goal)
            if not GA5.gc_agrees(r, pose, reasons):
                n_gc_disagree += 1
            gp = GA5.grade_profile(grid, r)
            info[name] = dict(valid=not reasons, reasons=reasons, n_waypoints=int(len(r['waypoints'])),
                              length_m=float(r['stations'][-1]), **{k: v for k, v in gp.items() if k != 'grades_deg'})
        # geometry: is route_00 the straight start-goal line?
        u = (goal - pose[:2]) / np.linalg.norm(goal - pose[:2]); nrm = np.array([-u[1], u[0]])
        w00 = np.asarray(r00['waypoints'], float)
        lat = float(np.abs((w00 - pose[:2]) @ nrm).max())
        yaw_line = math.atan2(goal[1] - pose[1], goal[0] - pose[0])
        dyaw = math.degrees(math.atan2(math.sin(yaw_line - pose[2]), math.cos(yaw_line - pose[2])))
        hd00 = np.asarray(r00['headings'], float)
        dhead = float(np.degrees(np.abs(np.arctan2(np.sin(hd00 - yaw_line), np.cos(hd00 - yaw_line)))).max())
        end00 = [float(np.linalg.norm(w00[0] - pose[:2])), float(np.linalg.norm(w00[-1] - goal))]
        same = lat < 0.01 and abs(dyaw) < 0.01 and max(end00) < 0.01
        if not same:
            differs.append(dict(group=g, route00_max_lateral_m=lat, start_yaw_minus_line_deg=dyaw, route00_max_heading_dev_deg=dhead,
                                route00_start_goal_offsets_m=end00))
        if info['straight']['valid'] and (not info['route00']['valid']
                                          or info['straight']['mean_abs_deg'] <= info['route00']['mean_abs_deg'] + 1e-9):
            choice, why = 'straight', ('route00_invalid' if not info['route00']['valid'] else
                                       'tie' if abs(info['straight']['mean_abs_deg'] - info['route00']['mean_abs_deg']) <= 1e-9 else 'lower_grade')
        elif info['straight']['valid']:
            choice, why = 'route00', 'lower_grade'
        else:
            choice, why = 'route00', ('straight_invalid' if info['route00']['valid'] else 'both_invalid_fallback')
        r = cands[choice]
        r['meta'] = dict(candidate='ci_approach', family='ci_a5data_v1', approach=choice, choice_reason=why,
                         mean_grade_12m_deg=info[choice]['mean_abs_deg'], max_grade_12m_deg=info[choice]['max_abs_deg'],
                         mean_signed_grade_12m_deg=info[choice]['mean_signed_deg'], grade_length_m=GA5.GRADE_LEN_M,
                         cruise_speed_mps=GA5.APPROACH_SPEED_MPS, scene_id=g, group=g, case_set=set_name,
                         straight_valid=info['straight']['valid'], route00_valid=info['route00']['valid'],
                         route00_file_sha256=sha256_file(rp), route00_is_straight_line=bool(same),
                         route00_max_lateral_from_line_m=lat, start_yaw_minus_line_heading_deg=dyaw,
                         grid=os.path.relpath(grid['path'], ROOT), approach_lengths_s=list(BUILDS[set_name]),
                         note='driven with --horizon-s L; only the first L seconds matter', **info0)
        rj = GC.route_to_json(r)
        rj['meta']['route_sha256'] = GC.route_sha256(r)
        jdump(rj, rdir / f'{g}.json')
        try:
            TF.read_route(rdir / f'{g}.json')      # the frozen collector's own reader
        except ValueError as e:
            reader_rejects.append((g, str(e)))
        if set_name == 'suite' and (K1_APPROACH / f'{g}.json').exists():
            k1_cmp['compared'] += 1
            if GC.route_sha256(json.load(open(K1_APPROACH / f'{g}.json'))) == rj['meta']['route_sha256']:
                k1_cmp['equal'] += 1
            else:
                k1_cmp['differ'].append(g)
        idx[g] = dict(tier=tier, approach=choice, choice_reason=why, route_sha256=rj['meta']['route_sha256'],
                      case_sha256=sha256_file(cp), mean_grade_12m_deg=info[choice]['mean_abs_deg'],
                      max_grade_12m_deg=info[choice]['max_abs_deg'], mean_signed_grade_12m_deg=info[choice]['mean_signed_deg'],
                      route00_is_straight_line=bool(same), route00_max_lateral_from_line_m=lat, start_yaw_minus_line_heading_deg=dyaw,
                      length_m=info[choice]['length_m'], straight=info['straight'], route00=info['route00'], **info0)
    mx = np.array([v['max_grade_12m_deg'] for v in idx.values()])
    summ = dict(n_groups=len(idx), choice_counts={c: sum(v['approach'] == c for v in idx.values()) for c in ('straight', 'route00')},
                choice_reasons={r: sum(v['choice_reason'] == r for v in idx.values()) for r in sorted({v['choice_reason'] for v in idx.values()})},
                straight_invalid=sum(not v['straight']['valid'] for v in idx.values()),
                route00_invalid=sum(not v['route00']['valid'] for v in idx.values()),
                gc_control_verdict_disagreements=n_gc_disagree, collector_reader_rejects=reader_rejects,
                route00_differs_from_straight_line=len(differs), route00_differs=differs,
                max_route00_lateral_m=float(max(v['route00_max_lateral_from_line_m'] for v in idx.values())),
                max_abs_start_yaw_minus_line_deg=float(max(abs(v['start_yaw_minus_line_heading_deg']) for v in idx.values())),
                max_grade_12m_gt_12deg=int((mx > 12).sum()), max_grade_12m_gt_17deg=int((mx > 17).sum()),
                max_grade_12m_p50_deg=float(np.median(mx)), length_m_min_p50_max=[float(np.min([v['length_m'] for v in idx.values()])),
                                                                                 float(np.median([v['length_m'] for v in idx.values()])),
                                                                                 float(np.max([v['length_m'] for v in idx.values()]))],
                wall_s=round(time.time() - t0, 1))
    if set_name == 'suite':
        summ['k1_approach_routes_same_content'] = k1_cmp
    if set_name == 'twin':
        summ['split_counts'] = {s: sum(v['split'] == s for v in idx.values()) for s in ('train', 'val', 'test')}
    index = dict(schema=1, created=now(), stage='approach-routes', set=set_name, cases=os.path.relpath(cases_dir.resolve(), ROOT),
                 grid=os.path.relpath(grid['path'], ROOT), grid_sha256=sha256_file(grid['path']),
                 approach_speed_mps=GA5.APPROACH_SPEED_MPS, step_m=GA5.STEP_M, grade_length_m=GA5.GRADE_LEN_M,
                 rule='ga_approach: straight if valid and mean|grade| over the first 12 m <= route_00 at 3 m/s (ties -> straight); '
                      'else route_00 at 3 m/s', summary=summ, groups=idx)
    jdump(index, out / f'approach_{set_name}_index.json')
    print(json.dumps({k: v for k, v in summ.items() if k != 'route00_differs'}, indent=1))
    print(f'wrote {len(idx)} approach routes to {rdir} and {out / f"approach_{set_name}_index.json"}')
    return index


# ============================================================================================================ (b)
def pass1_rows(set_name, index, L, world):
    F = frame_of(L); tag = ltag(L); rows = []
    for g, e in sorted(index['groups'].items(), key=lambda kv: kv[1]['tier']):
        if set_name == 'twin':
            assert not is_suite(g) and twin_split()[g] == e['split'], g
        rid = f'{g}__p1_{tag}'
        prov = dict(sha256=e['route_sha256'], case_sha256=e['case_sha256'], case_set=set_name, length_s=float(L), frame=F,
                    world=world, approach=e['approach'], source='ci_a5data_pass1')
        prov.update({k: e[k] for k in ('split', 'stratum', 'evaluation_stratum') if k in e})
        if world == 'crm':
            row = dict(id=rid, group=g, case=f'{CLUSTER_CASES[set_name]["crm"]}/{g}.json',
                       route=f'{CRM_SUB}/pass1_{set_name}/approach/{g}.json', run=True, tier=int(e['tier']),
                       episode_seed=seed_of(rid), extra=['--horizon-s', f'{float(L):g}'], **prov)
        else:
            row = dict(id=rid, group=g, case=f'{CLUSTER_CASES[set_name]["rigid"]}/{g}.json',
                       route=f'{G2}/pass1_{set_name}/approach/{g}.json', run=True, tier=int(e['tier']),
                       arena=ARENA_TAG, shard=shard_of(g), mode='native', extra=['--horizon-s', f'{float(L):g}'],
                       episode_seed=seed_of(rid), **prov)
        rows.append(row)
    return rows


def assert_rows(rows, what):
    ids = [r['id'] for r in rows]; assert len(set(ids)) == len(ids), f'{what}: ids not unique'
    seeds = [r['episode_seed'] for r in rows]; assert len(set(seeds)) == len(seeds), f'{what}: seeds not unique'


def stage_pass1_tasks(a):
    out = Path(a.out); tdir = out / 'tasks'; tdir.mkdir(parents=True, exist_ok=True)
    sets = a.sets or ['suite', 'twin']; worlds = a.worlds or ['crm', 'rigid']
    summary = dict(created=now(), stage='pass1-tasks', files={})
    for set_name in sets:
        index = json.load(open(out / f'approach_{set_name}_index.json'))
        assert index['set'] == set_name
        lengths = [float(x) for x in a.lengths] if a.lengths else list(BUILDS[set_name])
        for L in lengths:
            assert L in (0.5, 1.0, 3.0), f'approach length {L} not in {{0.5, 1, 3}}'
        for g, e in index['groups'].items():   # every approach route file present and matching the index
            rj = json.load(open(out / f'approach_{set_name}' / f'{g}.json'))
            assert route_content_sha(rj) == e['route_sha256'] == rj['meta']['route_sha256'], g
            assert sha256_file(LOCAL_CASES[set_name] / f'{g}.json') == e['case_sha256'], g
        for world in worlds:
            merged = []
            for L in lengths:
                rows = pass1_rows(set_name, index, L, world); assert_rows(rows, f'{set_name}/{world}/{L}')
                p = tdir / f'tasks_pass1_{set_name}_{world}_L{ltag(L)}.json'; jdump(rows, p)
                summary['files'][p.name] = dict(rows=len(rows), sha256=sha256_file(p), length_s=L, frame=frame_of(L))
                merged += rows
            merged.sort(key=lambda r: (r['tier'], r['length_s']))
            assert_rows(merged, f'{set_name}/{world}')
            p = tdir / f'tasks_pass1_{set_name}_{world}.json'; jdump(merged, p)
            extra = {}
            if world == 'rigid':
                extra['shard_counts'] = {s: sum(r['shard'] == s for r in merged) for s in range(SHARDS)}
            summary['files'][p.name] = dict(rows=len(merged), sha256=sha256_file(p), lengths_s=lengths, merged=True, **extra)
    # seeds: identical per id across the worlds (the B3 convention), unique within every file (asserted above)
    jdump(summary, tdir / 'pass1_tasks_summary.json')
    ship = ship_pass1_text(out, sets)
    (out / 'SHIP_pass1.txt').write_text(ship)
    print(json.dumps(summary, indent=1)); print(ship)


def verify_snippet(task_files, root_var):
    """python3 heredoc (for the cluster; NOT run here): every row's case / route exists; case file sha256 == row;
    route content hash (gc_control.route_sha256 expression) == row; ids unique."""
    files = ' '.join(task_files)
    return f'''ssh amd "cd {root_var} && python3 - {files}" <<'PY'
import hashlib, json, os, sys
bad = 0
for tf in sys.argv[1:]:
    rows = json.load(open(tf)); ids = [r['id'] for r in rows]; assert len(set(ids)) == len(ids), tf
    def rsha(f):
        j = json.load(open(f))
        return hashlib.sha256(json.dumps({{k: [[float(v) for v in p] if isinstance(p, list) else float(p) for p in j[k]] for k in ('waypoints', 'speeds', 'stations', 'headings')}}).encode()).hexdigest()
    for r in rows:
        if not r.get('run', True): continue
        ex = r.get('extra', []); br = ex[ex.index('--branch-route') + 1] if '--branch-route' in ex else None
        for key, f in (('case', r['case']), ('route', r['route']), ('branch', br)):
            if f is not None and not os.path.isfile(f): bad += 1; print('MISSING', r['id'], key, f)
        if os.path.isfile(r['case']) and hashlib.sha256(open(r['case'], 'rb').read()).hexdigest() != r['case_sha256']: bad += 1; print('CASE SHA', r['id'])
        if os.path.isfile(r['route']) and rsha(r['route']) != (r['prefix_sha256'] if br else r['sha256']): bad += 1; print('ROUTE SHA', r['id'])
        if br and os.path.isfile(br) and rsha(br) != r['sha256']: bad += 1; print('BRANCH SHA', r['id'])
    print(tf, len(rows), 'rows checked')
print('verify: bad =', bad)
PY'''


def ship_pass1_text(out, sets):
    L = out.resolve(); C = CRM_ROOT
    lines = ['# Pass-1 approach drives (ci_a5data.py --stage pass1-tasks). NOT run by ci_a5data.py.',
             f'C={C}; G2={G2}; L={L}',
             '# 1. ship the approach routes (both roots) and the task files']
    mk = ' '.join([f'$C/{CRM_SUB}/pass1_{s}/approach' for s in sets] + [f'$G2/pass1_{s}/approach' for s in sets] + ['$G2/tasks'])
    lines.append(f'ssh amd "mkdir -p {mk}"')
    for s in sets:
        lines += [f'rsync -az $L/approach_{s}/ amd:$C/{CRM_SUB}/pass1_{s}/approach/',
                  f'rsync -az $L/approach_{s}/ amd:$G2/pass1_{s}/approach/',
                  f'rsync -az $L/tasks/tasks_pass1_{s}_crm*.json $L/approach_{s}_index.json amd:$C/{CRM_SUB}/pass1_{s}/',
                  f'rsync -az $L/tasks/tasks_pass1_{s}_rigid*.json amd:$G2/tasks/']
    lines += ['# 2. verify on the cluster (paths, case sha256, route content hash); expect "verify: bad = 0"']
    crm_files = [f'{CRM_SUB}/pass1_{s}/tasks_pass1_{s}_crm.json' for s in sets]
    rig_files = [f'$G2/tasks/tasks_pass1_{s}_rigid.json' for s in sets]
    lines += [verify_snippet(crm_files, '$C'), verify_snippet(rig_files, '$G2')]
    lines += ['# 3. launch templates (NOT run; the budget and the queue cap are the orchestrator\'s call).',
              '#    CRM pass 1 = native mode -> the unmodified collector (crm_collect.py, CRM_COLLECTOR unset), MI350X partitions',
              '#    only (K1 determinism check: prefix replays are byte-identical across MI350X nodes); one out dir per set',
              '#    (ids carry the length tag).  Rigid pass 1 through gen_array_g.sbatch with GEN_ROOT=$G2 (G2/source must hold',
              '#    gen_collect_ext.py + source_manifest.json, K1 LAUNCH_rigid_a4_b3.md recipe).']
    for s in sets:
        lines += [f'# S=$C/source/scripts/crm_collect.sbatch; X="--export=ALL,CRM_TASKS=$C/{CRM_SUB}/pass1_{s}/tasks_pass1_{s}_crm.json,'
                  f'CRM_OUT=$C/{CRM_SUB}/pass1_{s}/out,CRM_CONFIG=configs/crm_main.json,CRM_BUDGET_S=13400"',
                  f'# mkdir -p $C/{CRM_SUB}/pass1_{s}/out/logs; sbatch --parsable -p mi3501x -c 24 -t 04:00:00 --array=0-5 -J ci_p1_{s} $X -o $C/{CRM_SUB}/pass1_{s}/out/logs/%x_%A_%a.out $S',
                  f'# mkdir -p $G2/rigid_pass1_{s}/logs; sbatch --parsable -p mi2104x -c 128 -t 03:00:00 --array=0-5 -J ci_rp1_{s} '
                  f'--export=ALL,GEN_ROOT=$G2,GEN_TASKS=$G2/tasks/tasks_pass1_{s}_rigid.json,GEN_OUT=$G2/rigid_pass1_{s},'
                  f'GEN_COLLECTOR=$G2/source/scripts/gen_collect_ext.py -o $G2/rigid_pass1_{s}/logs/%x_%A_%a.out $G2/source/scripts/gen_array_g.sbatch']
    lines += ['# 4. sync back (allowed direction)']
    for s in sets:
        lines += [f'# rsync -az --exclude claims --exclude stale amd:$C/{CRM_SUB}/pass1_{s}/out/runs/ $L/pass1_{s}_crm/runs/',
                  f'# rsync -az amd:$G2/rigid_pass1_{s}/runs/ $L/pass1_{s}_rigid/runs/']
    return '\n'.join(lines) + '\n'


# ============================================================================================================ (c)
def load_cases_goal(groups, cases_dir=None):
    out = {}
    for g in groups:
        d = Path(cases_dir) if cases_dir else LOCAL_CASES['twin' if fnmatch.fnmatch(g, TWIN_PATTERN) else 'suite']
        c = json.load(open(d / f'{g}.json'))
        out[g] = dict(goal=np.asarray(c['goal_xy'], float), start=np.asarray(c['layout']['start_xy'], float),
                      start_yaw=float(c['layout']['start_yaw']), goal_radius_m=float(c['goal_radius_m']), case_split=c.get('split'))
    return out


def analyze_one(run_dir, F, tm, hist_Ts, approach_route=None):
    """Decision state at frame F of one pass-1 recording -> (record, {T: (hist, hmask)}) or (record, None)."""
    run_dir = Path(run_dir)
    rec = dict(run=str(run_dir.resolve()), frame=int(F), moving=False, reasons=[])
    tp = run_dir / 'trajectory.npz'
    traj = np.load(tp, allow_pickle=True)
    outc = json.load(open(run_dir / 'outcome.json')) if (run_dir / 'outcome.json').exists() else {}
    st = np.asarray(traj['state'], np.float32); ac = np.asarray(traj['action'], np.float32); po = np.asarray(traj['pose'], np.float64)
    n = len(po)
    rec.update(n_frames=int(n), status=outc.get('status'), elapsed_s=outc.get('elapsed_s'))
    if n > F:
        s_k, p_k, src, stx = st[F], po[F], 'row', st
    elif n == F and 'terminal_state' in traj.files and 'terminal_pose' in traj.files:
        s_k, p_k, src = np.asarray(traj['terminal_state'], np.float32), np.asarray(traj['terminal_pose'], np.float64), 'terminal'
        stx = np.vstack([st, s_k[None]])
    else:
        rec['frame_source'] = 'absent'; rec['reasons'].append(f'short_recording_{n}_rows')
        return rec, None
    hists = {T: GA5.history_window(stx, ac, F, T) for T in hist_Ts}
    rec.update(frame_source=src, pose=[float(v) for v in p_k], state=[float(v) for v in s_k], vx=float(s_k[0]), vy=float(s_k[1]),
               roll=float(s_k[2]), pitch=float(s_k[3]), yaw_rate=float(s_k[6]),
               last_action=[float(v) for v in ac[F - 1]] if F - 1 < len(ac) else None,
               hist_valid_steps={str(T): int(h[1].sum()) for T, h in hists.items()},
               displacement_from_start_m=float(np.linalg.norm(p_k[:2] - po[0, :2])))
    if approach_route is not None:
        import f104_n2_dataset as DS
        s, dev, Lr = DS.project(np.asarray(p_k[None, :2], float), np.asarray(approach_route['waypoints'], float))
        rec.update(station_F_m=float(s[0]), lateral_dev_F_m=float(dev[0]), approach_len_m=float(Lr), remaining_m=float(Lr - s[0]))
    ep = run_dir / 'crm_extra.npz'
    if ep.exists():
        extra = np.load(ep, allow_pickle=True)
        ke = min(F, len(extra['quat']) - 1)
        sk = GA5.wheel_sinkage(extra, po, tm)
        rec.update(sink_frame=int(ke), sink_inc_max_m=float((sk[ke] - sk[0]).max()), sink_inc_mean_m=float((sk[ke] - sk[0]).mean()),
                   sink_inc_per_wheel_m=[float(v) for v in sk[ke] - sk[0]], sink_frame0_per_wheel_m=[float(v) for v in sk[0]],
                   sink_source='crm_extra')
    else:
        rec.update(sink_source=None)
    return rec, hists


def stage_analyze(a, quiet=False):
    from nedm.traverse.terrain import TerrainMap
    tm = TerrainMap.from_dir(Path(a.arena))
    L = float(a.length); F = frame_of(L); tag = ltag(L)
    suffix = a.run_suffix if a.run_suffix is not None else f'__p1_{tag}'
    vx_min = a.vx_min if a.vx_min is not None else (1.0 if F >= 60 else 0.3)
    hist_Ts = [int(t) for t in (a.hist_T or [HIST_T])]
    worlds = a.worlds or ['crm', 'rigid'][:len(a.runs)]
    assert len(worlds) == len(a.runs), (worlds, a.runs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    approach_dirs = [Path(p) for p in (a.approach_dir or [])]
    state = dict(schema=1, created=now(), stage='analyze', length_s=L, frame=F, run_suffix=suffix, hist_T=hist_Ts,
                 arena=os.path.relpath(Path(a.arena).resolve(), ROOT),
                 criterion=dict(vx_gt_mps=vx_min, sinkage_increase_lt_m=SINK_INC_MAX_M,
                                sinkage='max over wheels of [tyre radius - (spindle_z - TerrainMap height at the spindle xy)] at frame '
                                        'min(F, last crm_extra row) minus frame 0 (ga_approach formula); CRM only'),
                 worlds={}, analysis_set=[], excluded={})
    per_world_moving, per_world_groups = [], []
    for world, rdir in zip(worlds, a.runs):
        rdir = Path(rdir)
        dirs = sorted(p for p in rdir.iterdir() if p.is_dir() and p.name.endswith(suffix) and (p / 'trajectory.npz').exists()) if rdir.is_dir() else []
        if a.groups:
            want = set(a.groups)
            dirs = [d for d in dirs if d.name[:-len(suffix)] in want]
        if a.limit:
            dirs = dirs[:a.limit]
        goals = load_cases_goal([d.name[:-len(suffix)] for d in dirs], a.cases)
        recs, poses, poses_all, masked_all, dup, bad_name = {}, {}, {}, {}, [], []
        hdirs = {T: out / f'hist_{world}_T{T}' for T in hist_Ts}
        for d in dirs:
            g = d.name[:-len(suffix)]
            outc = json.load(open(d / 'outcome.json')) if (d / 'outcome.json').exists() else {}
            if outc.get('case_id') not in (None, g):
                bad_name.append(str(d)); continue
            if g in recs:
                dup.append(str(d)); continue
            ar = None
            for ad in approach_dirs:
                if (ad / f'{g}.json').exists():
                    ar = json.load(open(ad / f'{g}.json')); break
            if ar is None and (d / 'command_reference.npz').exists():
                cr = np.load(d / 'command_reference.npz')
                ar = dict(waypoints=cr['reference_waypoints'])
            rec, hists = analyze_one(d, F, tm, hist_Ts, ar)
            rec.update(group=g, world=world, goal_xy=goals[g]['goal'].tolist(), case_split=goals[g]['case_split'])
            if hists is not None:
                rec['goal_dist_F_m'] = float(np.linalg.norm(goals[g]['goal'] - np.asarray(rec['pose'][:2])))
                slow = rec['vx'] <= vx_min
                sinking = rec.get('sink_inc_max_m') is not None and rec['sink_inc_max_m'] >= SINK_INC_MAX_M
                if slow:
                    rec['reasons'].append(f'vx_{rec["vx"]:.2f}_le_{vx_min:g}')
                if sinking:
                    rec['reasons'].append(f'sinkage_increase_{rec["sink_inc_max_m"]:.3f}_ge_{SINK_INC_MAX_M:g}')
                rec['moving'] = not slow and not sinking
                rec['hist_files'] = {}
                for T, (h, m) in hists.items():
                    hdirs[T].mkdir(parents=True, exist_ok=True)
                    hp = hdirs[T] / f'{g}.npz'
                    np.savez_compressed(hp, hist=h, hmask=m, group=np.asarray(g), frame=np.int32(F), T=np.int32(T),
                                        source=np.asarray('rows+terminal' if rec['frame_source'] == 'terminal' else 'rows'),
                                        pose=np.asarray(rec['pose']), state=np.asarray(rec['state'], np.float32), world=np.asarray(world),
                                        run=np.asarray(rec['run']))
                    rec['hist_files'][str(T)] = str(hp.resolve())
                T0 = hist_Ts[0]
                if rec['frame_source'] == 'row':
                    ent = dict(run=rec['run'], frame=F, vx=rec['vx'], pass1_source='row')
                else:  # an F-row recording: ga_planner indexes pose[F], so give the pose and the window explicitly
                    ent = dict(pose=rec['pose'], history=rec['hist_files'][str(T0)], frame=F, vx=rec['vx'], pass1_run=rec['run'], pass1_source='terminal')
                poses_all[g] = ent
                masked_all[g] = dict(pose=rec['pose'], frame=F, vx=rec['vx'], pass1_run=rec['run'], pass1_source=rec['frame_source'])
                if rec['moving']:
                    poses[g] = ent
            recs[g] = rec
        n_mov = sum(r['moving'] for r in recs.values())
        reasons = {}
        for r in recs.values():
            for x in r['reasons']:
                key = x.split('_')[0] + ('_' + x.split('_')[1] if x.startswith(('sinkage', 'short')) else '')
                reasons[key] = reasons.get(key, 0) + 1
        vxs = [r['vx'] for r in recs.values() if 'vx' in r]
        sinks = [r['sink_inc_max_m'] for r in recs.values() if r.get('sink_inc_max_m') is not None]
        wsum = dict(runs_dir=str(rdir.resolve()), n_runs=len(dirs), n_groups=len(recs), n_moving=n_mov, n_excluded=len(recs) - n_mov,
                    exclusion_reasons=reasons, duplicate_group_runs=dup, case_id_mismatch=bad_name,
                    frame_source_counts={s: sum(1 for r in recs.values() if r.get('frame_source') == s) for s in ('row', 'terminal', 'absent')},
                    statuses={s: sum(1 for r in recs.values() if r.get('status') == s) for s in sorted({str(r.get('status')) for r in recs.values()})},
                    vx_p10_50_90=np.percentile(vxs, [10, 50, 90]).round(3).tolist() if vxs else None,
                    sink_inc_max_p50_p99_max=[round(float(np.median(sinks)), 4), round(float(np.percentile(sinks, 99)), 4), round(float(max(sinks)), 4)] if sinks else None)
        jdump(dict(schema=1, created=now(), world=world, length_s=L, frame=F, run_suffix=suffix, hist_T=hist_Ts, criterion=state['criterion'],
                   summary=wsum, groups=recs), out / f'decision_{world}.json')
        jdump(poses_all, out / f'poses_{world}_all.json')
        for T in hist_Ts[1:]:       # the same entries with the window of another length (models with hist_T = T)
            jdump({g: dict(e, history=recs[g]['hist_files'][str(T)]) if 'history' in e else e for g, e in poses_all.items()},
                  out / f'poses_{world}_all_T{T}.json')
            state['_posesT_' + world + f'_{T}'] = {g: dict(e, history=recs[g]['hist_files'][str(T)]) if 'history' in e else e for g, e in poses.items()}
        state['worlds'][world] = dict(wsum, decision_file=str(out / f'decision_{world}.json'))
        per_world_moving.append({g for g, r in recs.items() if r['moving']}); per_world_groups.append(set(recs))
        state['_poses_' + world] = poses; state['_masked_' + world] = masked_all
    common = set.intersection(*per_world_moving) if per_world_moving else set()
    allg = set.union(*per_world_groups) if per_world_groups else set()
    for w in worlds:
        pw = state.pop('_poses_' + w); mw = state.pop('_masked_' + w)
        jdump({g: pw[g] for g in sorted(common) if g in pw}, out / f'poses_{w}.json')
        jdump({g: mw[g] for g in sorted(common) if g in mw}, out / f'poses_{w}_masked.json')     # no history (all-masked arm)
        for T in hist_Ts[1:]:
            pt = state.pop('_posesT_' + w + f'_{T}')
            jdump({g: pt[g] for g in sorted(common) if g in pt}, out / f'poses_{w}_T{T}.json')
        state['worlds'][w]['poses_file'] = str(out / f'poses_{w}.json')
        state['worlds'][w]['poses_all_file'] = str(out / f'poses_{w}_all.json')
    state['analysis_set'] = sorted(common); state['n_analysis_set'] = len(common)
    state['excluded'] = dict(n=len(allg - common), groups=sorted(allg - common))
    jdump(state, out / 'pass1_state.json')
    if not quiet:
        print(json.dumps({k: v for k, v in state.items() if k not in ('analysis_set', 'excluded')}, indent=1))
        print(f'wrote {out}/pass1_state.json, decision_<world>.json, poses_<world>[_all].json ({len(common)} groups in the analysis set, '
              f'{len(allg - common)} excluded)')
    return state


# ============================================================================================================ (d)
def load_planner():
    """ci_planner.py (PLAN S1 module of this effort): candidates() and plan_decision() are the only entry points used."""
    import ci_planner as CP
    for f in ('candidates', 'plan_decision', 'CIEnsemble'):
        assert hasattr(CP, f), f'ci_planner has no {f}()'
    return CP


def validate_branch(route, pose, goal):
    """Reasons the branch collectors / the planner validator would reject the route (empty = valid)."""
    import gc_control as GC
    reasons = []
    try:
        GC.check_reference_contract(route)
    except ValueError as e:
        reasons.append(f'contract: {e}')
    if not GC.planner_validator(route, np.asarray(pose, float)):
        reasons.append('validator')
    wp = np.asarray(route['waypoints'], float)
    d0 = float(np.linalg.norm(wp[0] - np.asarray(pose[:2], float))); d1 = float(np.linalg.norm(wp[-1] - np.asarray(goal, float)))
    if d0 > BRANCH_START_TOL_M:
        reasons.append(f'start {d0:.3f} m from the pose (> {BRANCH_START_TOL_M})')
    if d1 > BRANCH_GOAL_TOL_M:
        reasons.append(f'end {d1:.3f} m from the goal (> {BRANCH_GOAL_TOL_M})')
    return reasons


def hist_for(rec, T):
    """(hist, hmask, source) of window length T at the decision frame: the analyze npz if it has this T, else the same
    window recomputed from the pass-1 recording (rows + terminal state)."""
    p = (rec.get('hist_files') or {}).get(str(T))
    if p and Path(p).exists():
        z = np.load(p)
        return np.asarray(z['hist'], np.float32), np.asarray(z['hmask'], bool), 'analyze_npz'
    traj = np.load(Path(rec['run']) / 'trajectory.npz'); F = int(rec['frame'])
    st = np.asarray(traj['state'], np.float32); ac = np.asarray(traj['action'], np.float32)
    stx = st if len(st) > F else np.vstack([st, np.asarray(traj['terminal_state'], np.float32)[None]])
    h, m = GA5.history_window(stx, ac, F, T)
    return h, m, 'recomputed_from_run'


def cont_paths(set_name, tag, world, rid):
    if world == 'crm':
        return f'{CRM_ROOT}/{CRM_SUB}/cont_{set_name}_L{tag}/routes_crm/{rid}.json'
    return f'{G2}/cont_{set_name}_L{tag}/routes_rigid/{rid}.json'


def build_continuations(decision_dir, world, out_root, models=K1_H_MODELS, map_root=MAP_ROOT, groups=None, limit=0,
                        all_groups=False, allow_suite=False, device=None, length=None, quiet=False, selftest=False):
    """K = 6 continuations per decision state (slots c0-c5, SLOTS order) -> routes, rows, anchors, report."""
    import gc_control as GC
    CP = load_planner()
    blacklisted = check_blacklist_source()
    decision_dir = Path(decision_dir)
    D = json.load(open(decision_dir / f'decision_{world}.json'))
    ps = json.load(open(decision_dir / 'pass1_state.json'))
    L, F = float(D['length_s']), int(D['frame']); tag = ltag(L)
    assert F == frame_of(L) and (length is None or float(length) == L), (L, F, length)
    recs = D['groups']
    have = sorted(g for g, r in recs.items() if r.get('frame_source') in ('row', 'terminal'))
    todo = have if all_groups else [g for g in have if g in set(ps['analysis_set'])]
    if groups:
        todo = [g for g in todo if g in set(groups)]
    if limit:
        todo = todo[:limit]
    hits = [g for g in todo if blacklisted(g) or is_suite(g)]
    if hits and not allow_suite:
        raise AssertionError(f'suite (evaluation) groups in the continuation input: {hits[:5]} ({len(hits)}); never train on them')
    set_name = 'twin'
    if hits:
        assert len(hits) == len(todo), 'mixed suite / twin decisions'
        set_name = 'suite'                                   # self-test only (allow_suite)
    split_of = twin_split() if set_name == 'twin' else {g: 'selftest' for g in todo}
    aroot = next((r for r in (Path(out_root), OUT) if (r / f'approach_{set_name}_index.json').exists()), None)
    approach_index = json.load(open(aroot / f'approach_{set_name}_index.json'))['groups'] if aroot else {}
    assert approach_index or selftest, f'approach_{set_name}_index.json not found under {out_root} or {OUT} (run --stage approach-routes first)'
    out = Path(out_root) / f'cont_{set_name}_L{tag}'; rdir = out / f'routes_{world}'
    if rdir.exists():               # a rebuild replaces this world's routes (no stale files get shipped)
        shutil.rmtree(rdir)
    rdir.mkdir(parents=True)
    CP.DS.init_map(str(map_root))
    ens = CP.CIEnsemble(models, device)
    T = int(ens.hist_T)
    rows, anchors, fails, invalid, prefix_mismatch = [], [], [], [], []
    t0 = time.time(); cem_wall = 0.0
    for tier, g in enumerate(todo):
        rec = recs[g]
        sp = split_of[g]
        if set_name == 'twin':
            assert rec.get('case_split') in (None, sp), f'{g}: case split {rec.get("case_split")} != twin split {sp}'
        pose = np.asarray(rec['pose'], float); goal = np.asarray(rec['goal_xy'], float); v0 = float(rec['vx'])
        hist, hmask, hsrc = hist_for(rec, T)
        ep = f'{g}__p1_{tag}'
        # the prefix the branch drive replays (the approach route) must be the route pass 1 drove
        prefix_ok = None
        cr = Path(rec['run']) / 'command_reference.npz'
        if aroot and cr.exists():
            z = np.load(cr); ap = json.load(open(aroot / f'approach_{set_name}' / f'{g}.json'))
            prefix_ok = bool(np.array_equal(np.asarray(z['reference_waypoints'], float), np.asarray(ap['waypoints'], float))
                             and np.array_equal(np.asarray(z['reference_speeds'], float), np.asarray(ap['speeds'], float)))
            if not prefix_ok:
                prefix_mismatch.append(g)
        slots = [None] * len(SLOTS)
        for fam in ('free', 'cont_head'):
            seed = seed_of(f'{ep}|{world}|{fam}|samples')
            idx = [j for j, s in enumerate(SLOTS) if s == (fam, 'sample')]
            try:
                cs = CP.candidates(pose, goal, v0, len(idx), seed, family=fam)
            except RuntimeError as e:
                cs = []; fails.append(dict(group=g, family=fam, source='sample', error=str(e)[:200]))
            for j, r in zip(idx, cs):
                slots[j] = dict(route=r, family=fam, source='sample', seed=seed, z_mean=None)
        for fam in ('free', 'cont_head'):
            j = SLOTS.index((fam, 'cem'))
            tc = time.perf_counter()
            res = CP.plan_decision(ens, pose, goal, v0, hist, hmask, group=g, arm='B', family=fam, world=world, domain=world)
            cem_wall += time.perf_counter() - tc
            if res is None:
                fails.append(dict(group=g, family=fam, source='cem', error='no pick')); continue
            slots[j] = dict(route=res['route'], family=fam, source='cem', seed=None, z_mean=float(res['z_mean']), P=float(res['P']),
                            cem_route_sha256=res['route_sha256'])
        seen, conts = {}, []
        for j, s in enumerate(slots):
            if s is None:
                continue
            r = s['route']; rid = f'{ep}__c{j}'
            reasons = validate_branch(r, pose, goal)
            if reasons:
                invalid.append(dict(id=rid, reasons=reasons)); continue
            sha = GC.route_sha256(r)
            if s['source'] == 'cem':
                assert sha == s['cem_route_sha256'], rid
            head = float(abs(math.degrees(math.atan2(math.sin(float(r['headings'][0]) - pose[2]), math.cos(float(r['headings'][0]) - pose[2])))))
            row = dict(id=rid, group=g, split=sp, cls=f'a5_{tag}', episode=ep, anchor_id=f'{ep}@{F}@{world}', F=F, length_s=L,
                       slot=j, family=s['family'], source=s['source'], run=True, tier=tier, episode_seed=seed_of(rid), sha256=sha,
                       world=world, case_set=set_name, prefix_sha256=(approach_index.get(g) or {}).get('route_sha256'),
                       case_sha256=(approach_index.get(g) or {}).get('case_sha256'),
                       start_speed_mps=float(r['speeds'][0]), speed_step_mps=float(r['speeds'][0]) - v0, start_heading_err_deg=head,
                       z_mean=s['z_mean'], sampler_seed=s['seed'], source_module='ci_a5data_continuations')
            bpath = cont_paths(set_name, tag, world, rid)
            if world == 'crm':
                row.update(case=f'{CLUSTER_CASES[set_name]["crm"]}/{g}.json', route=f'{CRM_SUB}/pass1_{set_name}/approach/{g}.json',
                           extra=['--mode', 'branch', '--branch-frame', str(F), '--branch-route', bpath, '--horizon-s', BRANCH_HORIZON])
            else:
                row.update(case=f'{CLUSTER_CASES[set_name]["rigid"]}/{g}.json', route=f'{G2}/pass1_{set_name}/approach/{g}.json',
                           arena=ARENA_TAG, shard=shard_of(g), mode='branch',
                           extra=['--branch-frame', str(F), '--branch-route', bpath, '--horizon-s', BRANCH_HORIZON])
            if sha in seen:
                row['run'] = False; row['ref_id'] = seen[sha]
            else:
                seen[sha] = rid
                rj = GC.route_to_json(r)
                rj['meta'] = dict(rj.get('meta', {}), ci_slot=j, ci_family=s['family'], ci_source=s['source'], ci_group=g, ci_world=world,
                                  ci_length_s=L, ci_frame=F, ci_branch_pose=pose.tolist(), ci_v0_mps=v0, route_sha256=sha,
                                  ci_models=models if s['source'] == 'cem' else None)
                jdump(rj, rdir / f'{rid}.json')
                assert GC.route_sha256(json.load(open(rdir / f'{rid}.json'))) == sha, rid
            rows.append(row)
            conts.append(dict(id=rid, slot=j, family=s['family'], source=s['source'], sha256=sha, run=row['run'], ref_id=row.get('ref_id')))
        anchors.append(dict(anchor_id=f'{ep}@{F}@{world}', episode=ep, world=world, group=g, split=sp, cls=f'a5_{tag}', F=F,
                            t_F_s=round(F * DT, 3), length_s=L, pose_F=rec['pose'], vx_F=rec['vx'], state_F=rec['state'],
                            last_action_F=rec.get('last_action'), goal_xy=rec['goal_xy'], goal_dist_F_m=rec.get('goal_dist_F_m'),
                            remaining_m=rec.get('remaining_m'), station_F_m=rec.get('station_F_m'), lateral_dev_F_m=rec.get('lateral_dev_F_m'),
                            moving=rec['moving'], sink_inc_max_m=rec.get('sink_inc_max_m'), pass1_run=rec['run'], pass1_status=rec.get('status'),
                            frame_source=rec['frame_source'], hist_T=T, hist_source=hsrc, hist_valid_steps=int(hmask.sum()),
                            prefix_route=(f'{CRM_SUB}/pass1_{set_name}/approach/{g}.json' if world == 'crm' else f'{G2}/pass1_{set_name}/approach/{g}.json'),
                            prefix_sha256=(approach_index.get(g) or {}).get('route_sha256'), prefix_equals_pass1_reference=prefix_ok,
                            n_slots=sum(s is not None for s in slots), n_unique=len(seen), conts=conts, source='ci_a5data_continuations'))
        if not quiet and (tier + 1) % 50 == 0:
            print(f'  {tier + 1}/{len(todo)} decisions {time.time() - t0:.0f}s rows {len(rows)}', flush=True)
    # ---- assertions
    if not selftest:
        assert not prefix_mismatch, f'pass-1 runs drove another route than the approach route: {prefix_mismatch[:5]} ({len(prefix_mismatch)})'
        unchecked = [a_['group'] for a_ in anchors if a_['prefix_equals_pass1_reference'] is None]
        assert not unchecked, f'no command_reference.npz in the pass-1 runs of {unchecked[:5]} ({len(unchecked)}): sync it (the prefix check needs it)'
        assert all(a_['prefix_sha256'] for a_ in anchors), 'approach route hash missing'
    ids = [r['id'] for r in rows]; assert len(set(ids)) == len(ids), 'ids not unique'
    seeds = [r['episode_seed'] for r in rows]; assert len(set(seeds)) == len(seeds), 'seeds not unique'
    assert len({a_['episode'] for a_ in anchors}) == len(anchors), 'anchor episodes not unique'
    if not allow_suite:
        bad = [s for r in rows for s in (r['id'], r['group'], r['episode']) if blacklisted(s) or is_suite(s)]
        bad += [s for a_ in anchors for s in (a_['episode'], a_['group']) if blacklisted(s) or is_suite(s)]
        assert not bad, f'suite groups present: {bad[:5]}'
        tw = twin_split()
        assert all(r['split'] == tw[r['group']] for r in rows) and all(a_['split'] == tw[a_['group']] for a_ in anchors), 'twin split violated'
    for r in rows:
        assert r['run'] or r['ref_id'] in set(ids)
        if r['run']:
            assert (rdir / f'{r["id"]}.json').exists()
    jdump(rows, out / f'tasks_cont_{set_name}_{world}_L{tag}.json')
    jdump(anchors, out / f'anchors_{world}_L{tag}.json')
    run_rows = [r for r in rows if r['run']]
    fam_stats = {}
    for fam in ('free', 'cont_head'):
        for src in ('sample', 'cem'):
            v = [r for r in rows if r['family'] == fam and r['source'] == src]
            if v:
                st = np.array([r['speed_step_mps'] for r in v]); hd = np.array([r['start_heading_err_deg'] for r in v])
                fam_stats[f'{fam}/{src}'] = dict(n=len(v), speed_step_p10_50_90=np.percentile(st, [10, 50, 90]).round(3).tolist(),
                                                 speed_step_abs_max=float(np.abs(st).max()), start_heading_p50_max_deg=[round(float(np.median(hd)), 2), round(float(hd.max()), 2)],
                                                 speed_step_bins={b: int(((st >= lo) & (st < hi)).sum()) for b, (lo, hi) in
                                                                  {'<-1.5': (-9e9, -1.5), '-1.5..-0.5': (-1.5, -0.5), '-0.5..0.5': (-0.5, 0.5), '0.5..1.5': (0.5, 1.5), '>1.5': (1.5, 9e9)}.items()})
    rep = dict(created=now(), stage='continuations', world=world, case_set=set_name, length_s=L, frame=F, decision_dir=str(decision_dir.resolve()),
               models=models, hist_T=T, ensemble=ens.describe(), n_decisions=len(todo), n_decision_states=len(have),
               n_analysis_set=len(ps['analysis_set']), all_groups=all_groups, rows=len(rows), run_rows=len(run_rows),
               duplicates=len(rows) - len(run_rows), prefix_checked=sum(a_['prefix_equals_pass1_reference'] is not None for a_ in anchors),
               prefix_mismatch=prefix_mismatch[:50], n_prefix_mismatch=len(prefix_mismatch), slots_missing=int(len(todo) * len(SLOTS) - len(rows) - len(invalid)), invalid=invalid[:50],
               n_invalid=len(invalid), failures=fails[:50], n_failures=len(fails),
               unique_per_decision={k: sum(1 for a_ in anchors if a_['n_unique'] == k) for k in range(len(SLOTS) + 1)},
               split_counts={s: sum(1 for a_ in anchors if a_['split'] == s) for s in sorted({a_['split'] for a_ in anchors})},
               shard_counts={s: sum(1 for r in run_rows if r.get('shard') == s) for s in range(SHARDS)} if world == 'rigid' else None,
               family_stats=fam_stats, cem_wall_s=round(cem_wall, 1), wall_s=round(time.time() - t0, 1),
               tasks=str(out / f'tasks_cont_{set_name}_{world}_L{tag}.json'), anchors=str(out / f'anchors_{world}_L{tag}.json'),
               tasks_sha256=sha256_file(out / f'tasks_cont_{set_name}_{world}_L{tag}.json'), routes_dir=str(rdir),
               selftest_suite_groups=bool(allow_suite and set_name == 'suite'))
    jdump(rep, out / f'report_{world}_L{tag}.json')
    if selftest:      # self-test outputs are never shipped: no SHIP file, a marker instead
        (out / 'SELFTEST_OUTPUT_DO_NOT_SHIP.txt').write_text('written by ci_a5data.py self-tests; not training data, never ship\n')
    else:
        (out / f'SHIP_cont_{world}_L{tag}.txt').write_text(ship_cont_text(out, set_name, tag, world))
    if not quiet:
        print(json.dumps({k: v for k, v in rep.items() if k not in ('invalid', 'failures', 'ensemble')}, indent=1))
    return rep, rows, anchors


def ship_cont_text(out, set_name, tag, world):
    Lp = out.resolve(); C = CRM_ROOT
    d = f'cont_{set_name}_L{tag}'
    if world == 'crm':
        return '\n'.join([
            f'# Continuation (branch) drives, CRM, approach {tag} (ci_a5data.py --stage continuations). NOT run by ci_a5data.py.',
            f'C={C}; G2={G2}; L={Lp}',
            f'ssh amd "mkdir -p $C/{CRM_SUB}/{d}/routes_crm $C/{CRM_SUB}/{d}/out/logs"',
            f'rsync -az $L/routes_crm/ amd:$C/{CRM_SUB}/{d}/routes_crm/',
            f'rsync -az $L/tasks_cont_{set_name}_crm_L{tag}.json $L/anchors_crm_L{tag}.json amd:$C/{CRM_SUB}/{d}/',
            f'# prerequisite: the approach routes at $C/{CRM_SUB}/pass1_{set_name}/approach/ (SHIP_pass1.txt) and crm_collect_ext.py in $G2/source/scripts',
            f'# check: ssh amd "ls $C/{CRM_SUB}/{d}/routes_crm | wc -l"  (= run rows in the report); then (expect "verify: bad = 0"):',
            verify_snippet([f'{CRM_SUB}/{d}/tasks_cont_{set_name}_crm_L{tag}.json'], '$C'),
            f'# launch template (not run): S=$C/source/scripts/crm_collect.sbatch; X="--export=ALL,CRM_TASKS=$C/{CRM_SUB}/{d}/tasks_cont_{set_name}_crm_L{tag}.json,'
            f'CRM_OUT=$C/{CRM_SUB}/{d}/out,CRM_CONFIG=configs/crm_main.json,CRM_COLLECTOR=$G2/source/scripts/crm_collect_ext.py,CRM_BUDGET_S=13400"',
            f'#   sbatch --parsable -p mi3501x -c 24 -t 04:00:00 --array=0-5 -J ci_cont_{tag} $X -o $C/{CRM_SUB}/{d}/out/logs/%x_%A_%a.out $S   (MI350X only)',
            f'# sync back: rsync -az --exclude claims --exclude stale amd:$C/{CRM_SUB}/{d}/out/runs/ $L/runs_crm/',
            f'# label:     PYTHONPATH=src:scripts python scripts/ga_branch_dataset.py --runs $L/runs_crm --anchors $L/anchors_crm_L{tag}.json --world crm --out <npz>']) + '\n'
    return '\n'.join([
        f'# Continuation (branch) drives, rigid, approach {tag} (ci_a5data.py --stage continuations). NOT run by ci_a5data.py.',
        f'G2={G2}; L={Lp}',
        f'ssh amd "mkdir -p $G2/{d}/routes_rigid $G2/rigid_{d}/logs $G2/tasks"',
        f'rsync -az $L/routes_rigid/ amd:$G2/{d}/routes_rigid/',
        f'rsync -az $L/tasks_cont_{set_name}_rigid_L{tag}.json $L/anchors_rigid_L{tag}.json amd:$G2/tasks/',
        f'# prerequisite: the approach routes at $G2/pass1_{set_name}/approach/ (SHIP_pass1.txt) and $G2/source (gen_collect_ext.py + source_manifest.json)',
        '# check (expect "verify: bad = 0"):',
        verify_snippet([f'$G2/tasks/tasks_cont_{set_name}_rigid_L{tag}.json'], '$G2'),
        f'# launch template (not run): sbatch --parsable -p mi2104x -c 128 -t 06:00:00 --array=0-5 -J ci_rcont_{tag} '
        f'--export=ALL,GEN_ROOT=$G2,GEN_TASKS=$G2/tasks/tasks_cont_{set_name}_rigid_L{tag}.json,GEN_OUT=$G2/rigid_{d},'
        f'GEN_COLLECTOR=$G2/source/scripts/gen_collect_ext.py -o $G2/rigid_{d}/logs/%x_%A_%a.out $G2/source/scripts/gen_array_g.sbatch',
        f'# sync back: rsync -az amd:$G2/rigid_{d}/runs/ $L/runs_rigid/',
        f'# label:     PYTHONPATH=src:scripts python scripts/ga_branch_dataset.py --runs $L/runs_rigid --anchors $L/anchors_rigid_L{tag}.json --world rigid --out <npz>']) + '\n'


def stage_continuations(a):
    assert a.decision and a.world, '--decision and --world are required'
    build_continuations(a.decision, a.world, a.out, a.models, a.map_root, a.groups, a.limit, a.all_groups, False, a.device, a.length)


def stage_merge(a):
    """Concatenate task (or anchor) files of several lengths / sets into one file per world (queue-cap convention)."""
    blacklisted = check_blacklist_source()
    rows = [r for f in a.inputs for r in json.load(open(f))]
    key = 'id' if all('id' in r for r in rows) else 'episode'
    ids = [r[key] for r in rows]; assert len(set(ids)) == len(ids), f'{key}s not unique across the inputs'
    if key == 'id':
        seeds = [r['episode_seed'] for r in rows]; assert len(set(seeds)) == len(seeds), 'seeds not unique across the inputs'
        assert len({r.get('world') for r in rows}) == 1, 'inputs from several worlds'
    if not a.allow_suite_rows:
        bad = [r[key] for r in rows if blacklisted(r['group']) or is_suite(r['group'])]
        assert not bad, f'suite groups: {bad[:5]}'
        tw = twin_split()
        assert all(r.get('split') == tw[r['group']] for r in rows), 'twin split violated'
    jdump(rows, a.out_file)
    print(f'merged {len(a.inputs)} files -> {a.out_file}: {len(rows)} rows ({sum(r.get("run", True) for r in rows)} to run), sha256 {sha256_file(a.out_file)}')


# ============================================================================================================ self-tests
def pick_standins(n=5):
    runs = K1_PASS1['crm']
    ds = sorted(d for d in runs.iterdir() if (d / 'trajectory.npz').exists() and (d / 'crm_extra.npz').exists())
    step = max(1, len(ds) // n)
    return [d.name for d in ds[::step][:n]]


def stage_selftest_analyze(a):
    import ga_planner as GP
    sdir = Path(a.out) / 'selftest' / 'analyze'
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    names = pick_standins(5)
    res = dict(created=now(), standins=names, checks={}, notes=[])
    # ---- A: the K1 3 s pass-1 recordings (60 rows + terminal state), both worlds, against K1's own analysis
    runs = {w: sdir / f'runs_{w}' for w in ('crm', 'rigid')}
    for w, rd in runs.items():
        rd.mkdir()
        for nm in names:
            os.symlink(K1_PASS1[w] / nm, rd / nm)
    ns = argparse.Namespace(runs=[str(runs['crm']), str(runs['rigid'])], worlds=['crm', 'rigid'], length='3', run_suffix='__pass1', vx_min=None,
                            hist_T=[40, 20], out=str(sdir / 'out_L3'), arena=a.arena, limit=0, groups=None, cases=str(SUITE_CASES),
                            approach_dir=[str(K1_APPROACH)])
    s3 = stage_analyze(ns, quiet=True)
    k1 = json.load(open(K1 / 'A_adapt/a5/pass1_state.json'))
    dA = []
    for w in ('crm', 'rigid'):
        mine = json.load(open(sdir / 'out_L3' / f'decision_{w}.json'))['groups']
        ref = k1['worlds'][w]['groups']
        for g, r in mine.items():
            q = ref[g]
            dd = dict(world=w, group=g, pose=float(np.abs(np.asarray(r['pose']) - np.asarray(q['pose'])).max()), vx=abs(r['vx'] - q['vx']),
                      moving_equal=r['moving'] == q['moving'], frame_source=(r['frame_source'], q['frame_source']))
            if w == 'crm':
                dd['sink_inc_max'] = abs(r['sink_inc_max_m'] - q['sink_inc_max_m'])
            # history window vs K1's hist npz (written by ga_approach for 60-row recordings)
            kh = K1 / 'A_adapt/a5' / f'hist_{w}' / f'{g}.npz'
            if kh.exists():
                z = np.load(kh); h = np.load(r['hist_files']['40'])
                dd['hist_vs_k1'] = float(np.abs(z['hist'] - h['hist']).max()); dd['hmask_vs_k1_equal'] = bool((z['hmask'] == h['hmask']).all())
            # ga_planner reads the entry; load_history gives the same window
            hh, mm, how = GP.load_history(r['hist_files']['40'], g, 60, 40)
            dd['ga_planner_load_history'] = how
            dd['remaining_plus_station_eq_len'] = abs(r['remaining_m'] + r['station_F_m'] - r['approach_len_m']) < 1e-9
            dA.append(dd)
    res['checks']['A_k1_3s_pose_max'] = max(x['pose'] for x in dA)
    res['checks']['A_k1_3s_vx_max'] = max(x['vx'] for x in dA)
    res['checks']['A_k1_3s_sink_max'] = max(x.get('sink_inc_max', 0.0) for x in dA)
    res['checks']['A_k1_3s_ok'] = all(x['pose'] == 0.0 and x['vx'] == 0.0 and x['moving_equal'] and x.get('sink_inc_max', 0.0) < 1e-12
                                      and x.get('hist_vs_k1', 0.0) == 0.0 and x.get('hmask_vs_k1_equal', True)
                                      and x['frame_source'][0] == x['frame_source'][1] == 'terminal' and x['remaining_plus_station_eq_len'] for x in dA)
    res['checks']['A_hist_vs_k1_compared'] = sum('hist_vs_k1' in x for x in dA)
    res['checks']['A_n_records'] = len(dA)
    res['A_records'] = dA
    pc = json.load(open(sdir / 'out_L3' / 'poses_crm.json'))
    res['checks']['A_poses_explicit_form'] = all({'pose', 'history', 'frame'} <= set(v) and 'run' not in v and v['frame'] == 60 for v in pc.values()) and len(pc) == 5
    # ---- B: shorter approaches.  Truncated copies cut at F (F rows + terminal = row F of the 60-row recording) must give
    #      the same decision state as the full recording read at row F, and the window must equal
    #      ga_planner.history_from_trajectory (full, F) and ga_branch_dataset's prefix_hist convention.
    import ga_branch_dataset as GBD
    for L in (0.5, 1.0):
        F = frame_of(L); tag = ltag(L)
        full, cut = sdir / f'runs_full_L{tag}', sdir / f'runs_cut_L{tag}'
        full.mkdir(); cut.mkdir()
        for nm in names:
            g = nm[:-len('__pass1')]
            src = K1_PASS1['crm'] / nm
            os.symlink(src, full / f'{g}__p1_{tag}')
            c = cut / f'{g}__p1_{tag}'; c.mkdir()
            shutil.copy(src / 'outcome.json', c / 'outcome.json'); shutil.copy(src / 'command_reference.npz', c / 'command_reference.npz')
            z = np.load(src / 'trajectory.npz', allow_pickle=True); e = np.load(src / 'crm_extra.npz', allow_pickle=True)
            n = len(z['pose'])
            tz = {k: (z[k][:F] if z[k].ndim and len(z[k]) == n else z[k]) for k in z.files}
            tz.update(terminal_state=z['state'][F], terminal_pose=z['pose'][F])
            np.savez_compressed(c / 'trajectory.npz', **tz)
            np.savez_compressed(c / 'crm_extra.npz', **{k: (e[k][:F] if e[k].ndim and len(e[k]) == n else e[k]) for k in e.files})
        o = {}
        for nm_, rd in (('full', full), ('cut', cut)):
            ns = argparse.Namespace(runs=[str(rd)], worlds=['crm'], length=str(L), run_suffix=None, vx_min=None, hist_T=[40, F],
                                    out=str(sdir / f'out_{nm_}_L{tag}'), arena=a.arena, limit=0, groups=None, cases=str(SUITE_CASES),
                                    approach_dir=[str(K1_APPROACH)])
            stage_analyze(ns, quiet=True)
            o[nm_] = json.load(open(sdir / f'out_{nm_}_L{tag}' / 'decision_crm.json'))['groups']
        dB = []
        for g in o['full']:
            rf, rc = o['full'][g], o['cut'][g]
            traj = np.load(K1_PASS1['crm'] / f'{g}__pass1' / 'trajectory.npz')
            href, mref = GP.history_from_trajectory(traj, F, 40)
            hb, okb = GBD.prefix_hist(traj['state'].astype(np.float32), traj['action'].astype(float), F)
            hc = np.load(rc['hist_files']['40']); hcF = np.load(rc['hist_files'][str(F)])
            hrF, mrF = GP.history_from_trajectory(traj, F, F)
            dB.append(dict(group=g, L=L, F=F, source=(rf['frame_source'], rc['frame_source']),
                           pose=float(np.abs(np.asarray(rf['pose']) - np.asarray(rc['pose'])).max()), vx=abs(rf['vx'] - rc['vx']),
                           pose_vs_row=float(np.abs(np.asarray(rc['pose']) - traj['pose'][F]).max()),
                           hist_vs_ga_planner=float(np.abs(hc['hist'] - href).max()), hmask_eq=bool((hc['hmask'] == mref).all()),
                           hist_vs_branch_dataset=float(np.abs(hc['hist'] - hb).max()), hmask_eq_branch=bool((hc['hmask'] == okb).all()),
                           histF_vs_ga_planner=float(np.abs(hcF['hist'] - hrF).max()), valid_steps_T40=int(hc['hmask'].sum()),
                           valid_steps_TF=int(hcF['hmask'].sum()), sink_full_cut=(rf.get('sink_inc_max_m'), rc.get('sink_inc_max_m')),
                           sink_frames=(rf.get('sink_frame'), rc.get('sink_frame')), vx_F=rc['vx'], moving=rc['moving']))
        res['checks'][f'B_L{tag}_ok'] = all(x['source'] == ('row', 'terminal') and x['pose'] == 0.0 and x['vx'] == 0.0 and x['pose_vs_row'] == 0.0
                                             and x['hist_vs_ga_planner'] == 0.0 and x['hmask_eq'] and x['hist_vs_branch_dataset'] == 0.0
                                             and x['hmask_eq_branch'] and x['histF_vs_ga_planner'] == 0.0 and x['valid_steps_T40'] == F
                                             and x['valid_steps_TF'] == F for x in dB)
        res['checks'][f'B_L{tag}_sink_full_minus_cut_max_m'] = float(max(abs(x['sink_full_cut'][0] - x['sink_full_cut'][1]) for x in dB))
        res[f'B_L{tag}_records'] = dB
        pcut = json.load(open(sdir / f'out_cut_L{tag}' / 'poses_crm_all.json'))
        res['checks'][f'B_L{tag}_poses_all_count'] = len(pcut)
    # ---- C: the pass-1 task builder writes rows that the workers accept (crm_worker keys; gen_runner_g keys)
    res['all_ok'] = all(v for k, v in res['checks'].items() if isinstance(v, bool))
    jdump(res, sdir / 'RESULTS.json')
    print(json.dumps(res['checks'], indent=1))
    print('SELFTEST analyze', 'OK' if res['all_ok'] else 'FAILED', '->', sdir / 'RESULTS.json')
    return res['all_ok']


def stage_selftest_continuations(a):
    """Continuation stage on 5 K1 pass-1 decision states (suite groups, 3 s approach, both worlds), real ci_planner and the
    K1 H ensemble.  Suite groups are allowed here ONLY (outputs under selftest/, never shipped); the guard is tested."""
    import ga_branch_dataset as GBD, gc_control as GC
    import traverse_fdm_rgbd_diverse_chrono as TF
    sdir = Path(a.out) / 'selftest' / 'continuations'
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    dec = Path(a.out) / 'selftest' / 'analyze' / 'out_L3'
    if not (dec / 'decision_rigid.json').exists():
        assert stage_selftest_analyze(a), 'selftest-analyze failed'
    res = dict(created=now(), decision_dir=str(dec), checks={}, worlds={})
    try:
        build_continuations(dec, 'crm', sdir / 'guard', quiet=True, selftest=True)
        res['checks']['suite_guard_fires'] = False
    except AssertionError as e:
        res['checks']['suite_guard_fires'] = 'suite' in str(e); res['guard_message'] = str(e)[:300]
    K1A5 = K1 / 'A_adapt/a5'
    for world in ('crm', 'rigid'):
        rep, rows, anchors = build_continuations(dec, world, sdir, allow_suite=True, quiet=True, device=a.device, selftest=True)
        W = dict(report={k: rep[k] for k in ('n_decisions', 'rows', 'run_rows', 'duplicates', 'slots_missing', 'n_invalid', 'n_failures',
                                              'unique_per_decision', 'family_stats', 'cem_wall_s', 'wall_s')})
        recs = json.load(open(dec / f'decision_{world}.json'))['groups']
        rdir = sdir / 'cont_suite_L3' / f'routes_{world}'
        # routes re-read from disk: collector reader, contract/validator/tolerances, family properties
        per = []
        for r in rows:
            f = rdir / f'{(r["id"] if r["run"] else r["ref_id"])}.json'
            rj = json.load(open(f)); rec = recs[r['group']]; pose = np.asarray(rec['pose']); v0 = rec['vx']
            TF.read_route(f)
            reasons = validate_branch(rj, pose, rec['goal_xy'])
            sp = np.asarray(rj['speeds'], float)
            e = dict(id=r['id'], family=r['family'], source=r['source'], valid=not reasons, sha_ok=GC.route_sha256(rj) == r['sha256'],
                     start_speed=float(sp[0]), v0=v0, head=r['start_heading_err_deg'],
                     start_err_m=float(np.linalg.norm(np.asarray(rj['waypoints'][0]) - pose[:2])))
            if r['family'] == 'cont_head':
                e['cont_ok'] = abs(float(sp[0]) - min(max(v0, 0.0), 6.0)) < 1e-9 and r['start_heading_err_deg'] <= 20.0 + 1e-9
            per.append(e)
        W['routes_all_valid'] = all(e['valid'] and e['sha_ok'] for e in per)
        W['cont_head_start_at_v0_and_heading_le_20'] = all(e.get('cont_ok', True) for e in per)
        W['free_speed_step_abs_max'] = max(abs(e['start_speed'] - e['v0']) for e in per if e['family'] == 'free')
        W['start_err_max_m'] = max(e['start_err_m'] for e in per)
        # the CEM pick with family free from the K1 decision state == K1's A5 H pick (the same planner, same state)
        k1 = []
        for r in rows:
            if r['family'] == 'free' and r['source'] == 'cem':
                kf = K1A5 / f'picks_{world}_H' / 'routes' / f'{r["group"]}__B.json'
                k1.append(dict(group=r['group'], equal=kf.exists() and GC.route_sha256(json.load(open(kf))) == r['sha256']))
        W['cem_free_equals_k1_H_pick'] = k1
        # row format for the workers
        ok_fmt = []
        for r in rows:
            ex = r['extra']
            if world == 'crm':
                ok_fmt.append(ex[:4] == ['--mode', 'branch', '--branch-frame', '60'] and ex[4] == '--branch-route' and ex[5].startswith(CRM_ROOT + '/')
                              and ex[6:] == ['--horizon-s', '120'] and not os.path.isabs(r['case']) and not os.path.isabs(r['route']))
            else:
                ok_fmt.append(r['mode'] == 'branch' and r['arena'] == 'f104' and r['shard'] == shard_of(r['group']) and ex[:2] == ['--branch-frame', '60']
                              and ex[2] == '--branch-route' and ex[3].startswith(G2 + '/') and ex[4:] == ['--horizon-s', '120']
                              and r['case'].startswith('/work1/') and r['route'].startswith(G2 + '/'))
        W['row_format_ok'] = all(ok_fmt)
        # anchors vs K1's pass-2 branch drives from the same recorded state (pose, vx, the collector's own prefix window)
        av = []
        for an in anchors:
            kd = K1A5 / f'{world}_pass2_runs' / f'{an["group"]}__H_B'
            z = np.load(kd / 'trajectory.npz')
            h = np.load(recs[an['group']]['hist_files']['40'])
            av.append(dict(group=an['group'], branch_frame=int(z['branch_frame']), pose_vs_branch_pose_m=float(np.linalg.norm(np.asarray(an['pose_F'][:2]) - z['branch_pose'][:2])),
                           vx_vs_pass2_row=abs(float(z['state'][60, 0]) - an['vx_F']),
                           hist_vs_branch_hist=float(np.abs(h['hist'] - z['branch_hist']).max()), hmask_equal=bool((h['hmask'] == z['branch_hmask']).all())))
        W['anchors_vs_k1_pass2'] = av
        # the labeller reads these anchors: a K1 pass-2 drive re-named as continuation c4 of our anchor
        lab = sdir / 'labeller' / world; lab.mkdir(parents=True)
        amap = {x['episode']: x for x in anchors}
        GBD.init(str(MAP_ROOT), amap, world)
        lb = []
        for an in anchors:
            g = an['group']; kd = K1A5 / f'{world}_pass2_runs' / f'{g}__H_B'
            d = lab / f'{an["episode"]}__c4'; d.mkdir()
            for f in ('trajectory.npz', 'outcome.json'):
                os.symlink(kd / f, d / f)
            shutil.copy(SUITE_CASES / f'{g}.json', d / 'case.json')
            br = json.load(open(K1A5 / f'routes_pass2_{world}' / f'{g}__H_B.json'))
            np.savez(d / 'command_reference.npz', **{f'branch_reference_{k}': np.asarray(br[k], float) for k in ('waypoints', 'speeds', 'stations', 'headings')})
            n = len(np.load(kd / 'trajectory.npz')['state'])
            if world == 'crm':      # K1's local copies of the pass-2 drives have no crm_extra.npz: zeros (format only)
                np.savez(d / 'crm_extra.npz', slip_ratio=np.zeros((n, 4), np.float32), spindle_z_m=np.zeros((n, 4), np.float32),
                         bmp_ground_z_m=np.zeros(n, np.float32))
            o = GBD.one(str(d))
            if 'drop' in o:
                lb.append(dict(group=g, drop=o['drop'])); continue
            dg, row = o['diag'], o['row']
            lb.append(dict(group=g, episode=row['episode'], cls=row['cls'], split=row['split'], anchor_frame=int(row['anchor_frame']),
                           pose_F_vs_recorded=dg['pose_F_vs_recorded'], vx_F_minus_recorded=dg['vx_F'] - dg['vx_F_recorded'],
                           hash_checked=dg['hash_checked'], status=dg['status'], fail=dg['fail'], route_len_minus_remaining_m=dg['route_len_vs_anchor_rem'],
                           hmask_valid=int(row['hmask'].sum())))
        W['labeller'] = lb
        W['labeller_ok'] = all('drop' not in x and x['cls'] == 'a5_3' and x['anchor_frame'] == 60 and x['hash_checked'] for x in lb)
        res['worlds'][world] = W
        c = res['checks']
        c[f'{world}_six_rows_per_decision'] = rep['rows'] == 6 * rep['n_decisions'] and rep['n_decisions'] == 5
        c[f'{world}_prefix_equals_pass1_reference'] = rep['prefix_checked'] == 5 and rep['n_prefix_mismatch'] == 0
        c[f'{world}_routes_valid'] = W['routes_all_valid']
        c[f'{world}_cont_head_properties'] = W['cont_head_start_at_v0_and_heading_le_20']
        c[f'{world}_cem_free_equals_k1'] = all(x['equal'] for x in k1) and len(k1) == 5
        c[f'{world}_row_format'] = W['row_format_ok']
        c[f'{world}_labeller_accepts_anchors'] = W['labeller_ok']
        c[f'{world}_anchor_pose_vs_k1_pass2_max_m'] = max(x['pose_vs_branch_pose_m'] for x in av)
        c[f'{world}_anchor_hist_vs_k1_pass2_max'] = max(x['hist_vs_branch_hist'] for x in av)
    # the ci_planner CLI (the S2 closed-loop path) reads our poses file and picks what the continuation stage picked (slot 5)
    import subprocess
    cli_out = sdir / 'cli_cont_head'
    cmd = [sys.executable, str(HERE / 'ci_planner.py'), '--family', 'cont_head', '--cases', str(SUITE_CASES), '--map-root', str(MAP_ROOT),
           '--models', K1_H_MODELS, '--world', 'crm', '--domain', 'crm', '--arms', 'B', '--poses', str(dec / 'poses_crm.json'),
           '--verify', '0', '--task-root', str(K2), '--out', str(cli_out)]
    with open(sdir / 'cli_cont_head.log', 'w') as fh:
        rc = subprocess.run(cmd, cwd=ROOT, env=dict(os.environ, PYTHONPATH='src:scripts'), stdout=fh, stderr=subprocess.STDOUT).returncode
    crm_rows = json.load(open(sdir / 'cont_suite_L3' / 'tasks_cont_suite_crm_L3.json'))
    cli = []
    for r in (x for x in crm_rows if x['slot'] == 5):
        pk = json.load(open(cli_out / 'picks' / f'{r["group"]}.json')) if rc == 0 else None
        cli.append(dict(group=r['group'], equal=bool(pk) and pk['arms']['B']['route_sha256'] == r['sha256'],
                        v0_source=pk['family']['v0_source'] if pk else None, v0_crosscheck=(pk['family'].get('v0_check') or {}).get('abs_diff') if pk else None))
    res['cli_cont_head_vs_slot5'] = dict(cmd=' '.join(cmd), rc=rc, groups=cli)
    res['checks']['cli_cont_head_equals_slot5'] = rc == 0 and len(cli) == 5 and all(x['equal'] for x in cli)
    # twin-path smoke (the real, non-self-test path: suite guard active, twin split asserted): K1 A4 CRM prefix replays of
    # twin episodes (designed routes, cut at F = 40, i.e. a 2 s 'approach') renamed <g>__p1_2; 4 train + 1 val + 1 test groups
    a4 = json.load(open(K1 / 'A_adapt/a4/anchors/anchors_crm.json'))
    want, pick, seen_g = {'train': 4, 'val': 1, 'test': 1}, [], set()
    for an in a4:
        if an['cls'] == 'clean_moving' and an['F'] == 40 and want.get(an['split'], 0) > 0 and an['group'] not in seen_g:
            pick.append(an); seen_g.add(an['group']); want[an['split']] -= 1
    tw = sdir / 'twin_smoke'; truns = tw / 'runs'; truns.mkdir(parents=True)
    for an in pick:     # files linked; command_reference.npz rebuilt from the designed route the replay drove (not synced locally)
        src = K1 / 'A_adapt/a4/pass1_out_crm/runs' / f"{an['episode']}__p1"; d = truns / f"{an['group']}__p1_2"; d.mkdir()
        for f in os.listdir(src):
            os.symlink(src / f, d / f)
        dr = json.load(open(ROOT / an['route_file']))
        np.savez(d / 'command_reference.npz', reference_waypoints=np.asarray(dr['waypoints'], float), reference_speeds=np.asarray(dr['speeds'], float),
                 reference_stations=np.asarray(dr['stations'], float), reference_headings=np.asarray(dr['headings'], float))
    stage_analyze(argparse.Namespace(runs=[str(truns)], worlds=['crm'], length='2', run_suffix=None, vx_min=None, hist_T=[40],
                                     out=str(tw / 'decision'), arena=a.arena, limit=0, groups=None, cases=str(TWIN_CASES), approach_dir=None), quiet=True)
    rep_t, rows_t, anchors_t = build_continuations(tw / 'decision', 'crm', tw, quiet=True, device=a.device, selftest=True)
    a4map = {an['group']: an for an in pick}
    res['twin_smoke'] = dict(groups=[an['group'] for an in pick], splits=[an['split'] for an in pick],
                             report={k: rep_t[k] for k in ('n_decisions', 'rows', 'run_rows', 'duplicates', 'n_invalid', 'n_failures', 'split_counts')},
                             pose_vs_a4_anchor_max_m=max(float(np.linalg.norm(np.asarray(x['pose_F'][:2]) - np.asarray(a4map[x['group']]['pose_F'][:2]))) for x in anchors_t),
                             vx_vs_a4_anchor_max=max(abs(x['vx_F'] - a4map[x['group']]['vx_F']) for x in anchors_t))
    res['twin_smoke']['prefix_mismatch_expected'] = rep_t['n_prefix_mismatch']      # these replays drove designed routes: 6 expected
    res['checks']['twin_smoke_prefix_check_fires'] = rep_t['n_prefix_mismatch'] == len(pick)
    try:            # the real (non-self-test) path refuses these decision states because their prefix is not the approach route
        build_continuations(tw / 'decision', 'crm', tw / 'negative', quiet=True, device=a.device, limit=1)
        res['checks']['twin_smoke_real_path_refuses_wrong_prefix'] = False
    except AssertionError as e:
        res['checks']['twin_smoke_real_path_refuses_wrong_prefix'] = 'another route' in str(e)
    shutil.rmtree(tw / 'negative', ignore_errors=True)
    res['checks']['twin_smoke_rows'] = rep_t['rows'] == 6 * len(pick) == 36 and rep_t['n_invalid'] == 0 and rep_t['n_failures'] == 0
    res['checks']['twin_smoke_split_kept'] = all(x['split'] == twin_split()[x['group']] for x in rows_t + anchors_t) and rep_t['split_counts'] == {'test': 1, 'train': 4, 'val': 1}
    res['checks']['twin_smoke_anchor_state_equals_a4_recording'] = res['twin_smoke']['pose_vs_a4_anchor_max_m'] == 0.0 and res['twin_smoke']['vx_vs_a4_anchor_max'] == 0.0
    res['checks']['twin_smoke_case_paths'] = all(r['case'] == f"cases/night2/{r['group']}.json" and r['route'] == f"crm_improve/pass1_twin/approach/{r['group']}.json" for r in rows_t)
    res['checks']['crm_anchor_state_equals_k1_pass2'] = res['checks']['crm_anchor_pose_vs_k1_pass2_max_m'] == 0.0 and res['checks']['crm_anchor_hist_vs_k1_pass2_max'] == 0.0
    res['all_ok'] = all(v for v in res['checks'].values() if isinstance(v, bool))
    jdump(res, sdir / 'RESULTS.json')
    print(json.dumps(res['checks'], indent=1))
    print('SELFTEST continuations', 'OK' if res['all_ok'] else 'FAILED', '->', sdir / 'RESULTS.json')
    return res['all_ok']


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=['approach-routes', 'pass1-tasks', 'analyze', 'continuations', 'merge',
                                                        'selftest-analyze', 'selftest-continuations'])
    ap.add_argument('--out', default=str(OUT), help='output root (approach_<set>/, tasks/, SHIP_*.txt; analyze/continuations: their own dir)')
    ap.add_argument('--cases', help='approach-routes: case dir; analyze/continuations: case dir for goals (default by group prefix)')
    ap.add_argument('--set', choices=['suite', 'twin'], help='approach-routes: case set name (default: from the group names)')
    ap.add_argument('--sets', nargs='*', choices=['suite', 'twin'], help='pass1-tasks: sets to build (default both)')
    ap.add_argument('--worlds', nargs='*', help='pass1-tasks: crm rigid; analyze: world label per --runs dir')
    ap.add_argument('--lengths', nargs='*', help='pass1-tasks: approach lengths in s (default suite 0.5 1, twin 0.5 1 3)')
    ap.add_argument('--grid', default=str(GRID)); ap.add_argument('--arena', default=str(ARENA))
    ap.add_argument('--runs', nargs='+', help='analyze: pass-1 run dir(s), one per world')
    ap.add_argument('--length', help='analyze / continuations: approach length L in s (F = round(L / 0.05))')
    ap.add_argument('--run-suffix', help='analyze: run dir name suffix (default __p1_<len>; K1 runs: __pass1)')
    ap.add_argument('--vx-min', type=float, help='analyze: moving criterion vx threshold (default 1.0 if F >= 60 else 0.3)')
    ap.add_argument('--hist-T', nargs='*', type=int, help='analyze: history window lengths (default 40; the first one goes into poses)')
    ap.add_argument('--approach-dir', nargs='*', help='analyze: dirs with <g>.json approach routes (progress along the route)')
    ap.add_argument('--groups', nargs='*', help='analyze / continuations: only these groups')
    ap.add_argument('--limit', type=int, default=0)
    # continuations
    ap.add_argument('--decision', help='continuations: analyze output dir (decision_<world>.json)')
    ap.add_argument('--world', choices=['crm', 'rigid'], help='continuations: world of the decision states')
    ap.add_argument('--models', default=K1_H_MODELS, help='continuations: CEM ensemble (default the K1 H deploy ensemble)')
    ap.add_argument('--map-root', default=str(MAP_ROOT))
    ap.add_argument('--device', default=None)
    ap.add_argument('--all-groups', action='store_true', help='continuations: every decision state, not only the analysis set')
    ap.add_argument('--inputs', nargs='*', help='merge: task or anchor json files'); ap.add_argument('--out-file', help='merge: output json')
    ap.add_argument('--allow-suite-rows', action='store_true', help='merge: suite rows allowed (pass-1 evaluation drives only)')
    a = ap.parse_args(argv)
    if a.stage == 'approach-routes':
        assert a.cases, '--cases is required'
        stage_approach_routes(a)
    elif a.stage == 'pass1-tasks':
        stage_pass1_tasks(a)
    elif a.stage == 'analyze':
        assert a.runs and a.length, '--runs and --length are required'
        stage_analyze(a)
    elif a.stage == 'continuations':
        stage_continuations(a)
    elif a.stage == 'merge':
        assert a.inputs and a.out_file, '--inputs and --out-file are required'
        stage_merge(a)
    elif a.stage == 'selftest-analyze':
        sys.exit(0 if stage_selftest_analyze(a) else 1)
    elif a.stage == 'selftest-continuations':
        sys.exit(0 if stage_selftest_continuations(a) else 1)


if __name__ == '__main__':
    main()
