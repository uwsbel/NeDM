#!/usr/bin/env python3
"""Offline planner-search probe for the Gator-trained soil planner on f104 (artifacts/traverse/search_probe_20260927).

Question: on the soil start/goal pairs where the Gator-trained planner (model G_full, CEM 4 x 64, free family, standing
start) did not reach the goal when the Gator drove its pick, does a bigger or wider search find routes that the SAME
model rates safe? Offline only: the model's own predicted failure probability; no Chrono, no cluster.

Groups (`groups`): FAIL = every group whose Gfull_free_gator drive failed (index e6/index/soil_eval_bfull.json, vehicle
gator); FAIL_P50 = the FAIL groups whose recorded pick had P > 0.5; CONTROL = the 100 lowest md5(group id) of the
successful groups. All groups are planned in md5(group id) order.

Arms (same G_full ensemble; the case layout pose at rest, route_00 as the base route, all-masked history (the cond
'none' members use no history), objective = ensemble-mean route logit, gen_planner validator unchanged):
  A0   CEM 4 rounds x 64, free family: ci_planner.plan_iter_family exactly as ag_picks.py -> ci_planner.py --arms B
       (rng default_rng(md5(group + 'n2iter_cem4x64')[:8])). Must reproduce the recorded pick (route bytes, P).
  A1   CEM 8 x 256   (rng tag 'sp_cem8x256')
  A2   CEM 16 x 512  (rng tag 'sp_cem16x512'); its best 64 distinct candidates by ensemble-mean logit are stored in
       A2_*/pool/<g>.json as the starts of A3b
  A3   gradient refinement, ci_grad.plan_group called in process with poses=None: that is exactly the standing start
       (ga_planner.decision_for -> layout pose, route_00; ci_planner.decision_v0 -> 'standing_start', v0 0; CIScorer
       with an all-masked window; the cond 'none' members get no history context, DiffEnsemble z=None). 64 starts = the
       A0 pick + the best 63 of A0's round-0 pool; Adam 300 steps, patience 50, lr 0.02 (lateral) / 0.10 (speed),
       gradient clip 10, keep-best and final choice by the ensemble MEAN logit (--keep mean), abstention margin 0 (the
       refined route is taken whenever it is not worse than A0 after the deployed re-score).
  A3b  the same refinement (same code path: ci_grad.Chain / refine / route_contract / deployed re-score) from the A2 pick
       + the next best 63 distinct candidates of A2's pool.
  A4   wider route family: 5 lateral sine modes (amplitude caps by the same formula 0.55 * 0.125 * L^2 / (j pi)^2,
       prior sd 5/j), lateral clip +-20 m, the 4 speed knots unchanged; CEM 16 x 512 (rng tag 'sp_wide5m20_cem16x512').
       Implemented by swapping f104_n2_iter's MODES / PRIOR_SD / from_params / draw_prior / project inside a context
       manager (no file edited); with (3 modes, 10 m) the swapped functions reproduce A0 byte for byte (selftest).

Outputs per arm (<root>/<arm dir>/, the ag_picks.py file set so ag_eval_tasks.py build can read it later):
  picks/<g>.json, routes/<g>__<ARM>.json, groups.txt, tasks.json (ga_planner row format, paths relative to <root>),
  summary.json, PICKS_LOCKED.sha256, ag_picks.json (manifest: models + sha256, map check, command, per-group picks).
Commands: groups | selftest | plan | rerun | finalize | rescore | report

  PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=1
  $PY scripts/sp_search_probe.py groups
  $PY scripts/sp_search_probe.py selftest
  $PY scripts/sp_search_probe.py plan --arms A0                       # then check A0 before anything else
  $PY scripts/sp_search_probe.py plan --arms A1,A2,A3b,A3,A4 --shard 0 --nshards 5   (x5)
  $PY scripts/sp_search_probe.py rerun --n 4 && $PY scripts/sp_search_probe.py finalize
  $PY scripts/sp_search_probe.py rescore && $PY scripts/sp_search_probe.py report
"""
import argparse, contextlib, hashlib, json, math, os, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ci_planner as CP                    # noqa: E402
import ci_grad as CG                       # noqa: E402
import ag_map_check                        # noqa: E402
GA, IT, GP, DS, S, PA = CP.GA, CP.IT, CP.GP, CP.DS, CP.S, CP.PA
torch = CP.torch

K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
REF_DIR = K3 / 'e6/picks/crm_bfull/f104/G_full_free'
INDEX = K3 / 'e6/index/soil_eval_bfull.json'
INDEX_ARM, INDEX_VEHICLE = 'Gfull_free_gator', 'gator'
MAP_ROOT = ROOT / 'artifacts/traverse/crm_f104_v1/map_root'
OUT = ROOT / 'artifacts/traverse/search_probe_20260927'
PY = '/home/harry/miniconda3/envs/nedm/bin/python'

ARMS = {
    'A0': dict(dir='A0_cem4x64_repro', kind='cem', rounds=4, n=64, family='free', tag='n2iter_cem4x64', label='cem4x64',
               words='the original search, re-run: 4 rounds of 64 candidate routes (cross-entropy method)'),
    'A1': dict(dir='A1_cem8x256', kind='cem', rounds=8, n=256, family='free', tag='sp_cem8x256', label='cem8x256',
               words='bigger search: 8 rounds of 256 candidates (8x the routes scored)'),
    'A2': dict(dir='A2_cem16x512', kind='cem', rounds=16, n=512, family='free', tag='sp_cem16x512', label='cem16x512',
               words='much bigger search: 16 rounds of 512 candidates (32x the routes scored)'),
    'A3': dict(dir='A3_grad64x300_fromA0', kind='grad', seed_from='A0', starts=64, steps=300, patience=50, label='grad64x300_mean_fromA0',
               words='gradient refinement of the route shape and speed through the model, from 64 starting routes of the original search, 300 steps'),
    'A3b': dict(dir='A3b_grad64x300_fromA2', kind='grad', seed_from='A2', starts=64, steps=300, patience=50, label='grad64x300_mean_fromA2',
                words='the same gradient refinement, started from the 64 best routes of the much bigger search (A2)'),
    'A4': dict(dir='A4_wide5m20_cem16x512', kind='cem', rounds=16, n=512, family='wide', modes=5, lat_clip=20.0,
               tag='sp_wide5m20_cem16x512', label='wide5m20_cem16x512',
               words='wider route shapes (5 sideways bend terms instead of 3, sideways limit 20 m instead of 10 m) with the much bigger search (16 x 512)'),
    # ablations of A4 on the FAIL P>0.5 pairs only (which half of the wider family matters)
    'A4a': dict(dir='A4a_wide3m20_cem16x512', kind='cem', rounds=16, n=512, family='wide', modes=3, lat_clip=20.0, groups='fail_p50',
                tag='sp_wide3m20_cem16x512', label='wide3m20_cem16x512',
                words='ablation of A4: the original 3 bend terms, sideways limit raised to 20 m, 16 x 512 search'),
    'A4b': dict(dir='A4b_wide5m10_cem16x512', kind='cem', rounds=16, n=512, family='wide', modes=5, lat_clip=10.0, groups='fail_p50',
                tag='sp_wide5m10_cem16x512', label='wide5m10_cem16x512',
                words='ablation of A4: 5 bend terms, the original 10 m sideways limit, 16 x 512 search'),
}
ORDER = ['A0', 'A1', 'A2', 'A3', 'A3b', 'A4']
ABL = ['A4a', 'A4b']
GRAD_OPTS = dict(lr_a=0.02, lr_dv=0.10, keep='mean', abstain_logit=0.0, clip_grad=10.0, frame=60, world='crm')
NS_DECISION = argparse.Namespace(from_run=None, frame=60, pose_along_s=None, pose_override=None, goal=None, history=None)


def md5hex(s):
    return hashlib.md5(s.encode()).hexdigest()


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pfail(z):
    return float(1.0 - np.exp(-np.exp(float(z))))


def ref_manifest():
    return json.load(open(REF_DIR / 'ag_picks.json'))


def cases_dir():
    return ROOT / ref_manifest()['cases_dir']


def models_glob():
    return str(ROOT / ref_manifest()['models_glob'])


def jdump(obj, path, indent=1):
    tmp = Path(str(path) + '.tmp')
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=indent, default=_jdefault)
    os.replace(tmp, path)


def _jdefault(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ------------------------------------------------------------------------------------------------------------------
# groups
# ------------------------------------------------------------------------------------------------------------------
def load_index_rows():
    d = json.load(open(INDEX))
    rows = [r for r in d['rows'] if r['vehicle'] == INDEX_VEHICLE and r['arm'] == INDEX_ARM]
    assert len(rows) == 800 and len({r['group'] for r in rows}) == 800, len(rows)
    return {r['group']: r for r in rows}


def cmd_groups(a):
    rows = load_index_rows()
    man = ref_manifest()
    for g, r in rows.items():          # the index and the pick manifest describe the same routes
        m = man['picks'][g]
        assert m['route_sha256'] == r['route_sha256'] and abs(m['P'] - r['P']) < 1e-12, g
    fail = sorted([g for g, r in rows.items() if r['fail'] == 1], key=md5hex)
    succ = sorted([g for g, r in rows.items() if r['fail'] == 0], key=md5hex)
    fail_p50 = [g for g in fail if rows[g]['P'] > 0.5]
    control = succ[:100]
    allg = sorted(set(fail) | set(control), key=md5hex)
    assert (len(fail), len(fail_p50), len(succ), len(control), len(allg)) == (261, 134, 539, 100, 361), (len(fail), len(fail_p50), len(succ))
    gd = OUT / 'groups'; gd.mkdir(parents=True, exist_ok=True)
    for name, gs in (('fail', fail), ('fail_p50', fail_p50), ('control', control), ('all', allg)):
        (gd / f'{name}.txt').write_text('\n'.join(gs) + '\n')
    meta = dict(index=os.path.relpath(INDEX, ROOT), index_sha256=sha256_file(INDEX), selection=dict(vehicle=INDEX_VEHICLE, arm=INDEX_ARM),
                rules=dict(FAIL="fail == 1 (status prolonged_blockage_terminated or soil_breakthrough_terminated)",
                           FAIL_P50='FAIL and the recorded pick P > 0.5',
                           CONTROL='the 100 lowest md5(group id) of the fail == 0 groups', order='md5(group id) ascending'),
                counts=dict(fail=len(fail), fail_p50=len(fail_p50), success=len(succ), control=len(control), all=len(allg)),
                status_counts={s: sum(1 for g in fail if rows[g]['status'] == s) for s in sorted({rows[g]['status'] for g in fail})},
                groups=dict(fail=fail, fail_p50=fail_p50, control=control, all=allg),
                original={g: dict(P=rows[g]['P'], z=rows[g]['z'], fail=rows[g]['fail'], status=rows[g]['status'], stratum=rows[g]['stratum'],
                                  route_sha256=rows[g]['route_sha256'], run_dir=rows[g]['run_dir'], elapsed=rows[g].get('elapsed'),
                                  max_tilt_deg=rows[g].get('max_tilt_deg')) for g in allg})
    jdump(meta, gd / 'groups.json')
    print(json.dumps(meta['counts']), meta['status_counts'])


def load_groups():
    return json.load(open(OUT / 'groups' / 'groups.json'))


# ------------------------------------------------------------------------------------------------------------------
# the wider family: f104_n2_iter's functions with (modes, lateral clip) as parameters
# ------------------------------------------------------------------------------------------------------------------
def make_family(modes, lat_clip):
    js = np.arange(1, modes + 1)
    prior_sd = np.array([IT.LAT_SIGMA / j for j in range(1, modes + 1)] + [IT.SP_SIGMA] * IT.KNOTS)
    standard = (modes, float(lat_clip)) == (IT.MODES, float(IT.LAT_CLIP))

    def caps(L):
        return IT.BUDGET * IT.KAPPA_MAX * L ** 2 / (js * np.pi) ** 2

    def project(theta, L, fixed_speed=None):
        th = np.asarray(theta, float).copy()
        th[:modes] = np.clip(th[:modes], -caps(L), caps(L))
        if fixed_speed is None:
            th[modes:] = np.clip(th[modes:], -IT.SP_CLIP, IT.SP_CLIP)
        return th

    def draw_prior(rng, L, fixed_speed=None, mu=None, sd=None):
        if mu is None:
            a = rng.normal(0, IT.LAT_SIGMA, modes) / js
            dv = rng.normal(0, 0.0 if fixed_speed is not None else IT.SP_SIGMA, IT.KNOTS)
            th = a if fixed_speed is not None else np.r_[a, dv]
        else:
            th = rng.normal(mu, sd)
        return project(th, L, fixed_speed)

    def from_params(base, theta, fixed_speed=None, **kw):
        assert fixed_speed is None and not kw, 'the probe family is speed-free only'
        xy, station, f, L = IT._base_arrays(base)
        theta = np.asarray(theta, float)
        cap = IT.BUDGET * IT.KAPPA_MAX * L ** 2 / (js * np.pi) ** 2
        a = np.clip(theta[:modes], -cap, cap)
        lat = sum(a[j] * np.sin((j + 1) * np.pi * f) for j in range(modes))
        lat = np.clip(lat, -lat_clip, lat_clip)
        speed = np.asarray(base['speeds'], float)
        dvk = np.clip(theta[modes:modes + IT.KNOTS], -IT.SP_CLIP, IT.SP_CLIP)
        dv = S.speed_knots(f, IT.KNOTS, dvk)
        r = S.shape(xy, station, speed, lat, dv)
        r['meta'] = {'candidate': 'n2_iter' if standard else f'n2_iter_wide{modes}m{lat_clip:g}', 'theta': np.r_[a, dvk].tolist(),
                     'max_lateral_m': float(np.abs(lat).max()), 'mean_speed_mps': float(r['speeds'][1:-1].mean())}
        return r
    return dict(modes=modes, lat_clip=float(lat_clip), prior_sd=prior_sd, caps=caps, project=project, draw_prior=draw_prior,
                from_params=from_params)


@contextlib.contextmanager
def family_patch(modes, lat_clip):
    """f104_n2_iter's module attributes used by ci_planner.plan_iter_family, swapped for the (modes, clip) family."""
    fam = make_family(modes, lat_clip)
    saved = dict(MODES=IT.MODES, PRIOR_SD=IT.PRIOR_SD, from_params=IT.from_params, draw_prior=IT.draw_prior, project=IT.project)
    IT.MODES, IT.PRIOR_SD, IT.from_params, IT.draw_prior, IT.project = modes, fam['prior_sd'], fam['from_params'], fam['draw_prior'], fam['project']
    try:
        yield fam
    finally:
        IT.MODES, IT.PRIOR_SD = saved['MODES'], saved['PRIOR_SD']
        IT.from_params, IT.draw_prior, IT.project = saved['from_params'], saved['draw_prior'], saved['project']


# ------------------------------------------------------------------------------------------------------------------
# planning context
# ------------------------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, need_grad=False, device=None):
        DS.init_map(str(MAP_ROOT))
        self.ens = CP.CIEnsemble(models_glob(), device)
        assert self.ens.kinds == ['ci_train'] and self.ens.conds == ['none'] and not self.ens.needs_hist, (self.ens.kinds, self.ens.conds)
        self.dens = self.tmap = None
        if need_grad:
            self.dens = CG.DiffEnsemble(self.ens); self.tmap = CG.DetMap(DS.G, self.ens.dev, torch.float32)
        self.cases = cases_dir()
        self.specB = PA.arm_specs('crm_proposal', '')['B']
        assert self.specB['tag'] == 'n2iter_cem4x64' and self.specB['rounds'] == 4 and self.specB['n'] == 64

    def decision(self, g):
        cp = str(self.cases / f'{g}.json')
        case, lay, pose, goal, base, hist, hmask, src = GA.decision_for(g, cp, NS_DECISION, None, self.ens.hist_T)
        assert src == dict(pose='layout', goal='case', base='route_00', history='all_masked', frame=None), src
        v0, how, _ = CP.decision_v0({}, hist, hmask, src, None, 60)
        assert (v0, how) == (0.0, 'standing_start'), (v0, how)
        return cp, case, pose, goal, base, hist, hmask, src

    def scorer(self, pose, goal, hist, hmask):
        return CP.CIScorer(self.ens, pose[:2], goal, float(pose[2]), 'crm', hist, hmask, float16=True)


def route_geometry(r, pose, goal, base, lat_clip, modes, theta, kind):
    wp = np.asarray(r['waypoints'], float); v = np.asarray(r['speeds'], float); st = np.asarray(r['stations'], float)
    straight = float(np.linalg.norm(np.asarray(goal, float) - np.asarray(pose, float)[:2]))
    L_base = float(np.asarray(base['stations'])[-1])
    ml = float(r['meta'].get('max_lateral_m', 0.0))
    xy, station, f, L = IT._base_arrays(base)
    cap = IT.BUDGET * IT.KAPPA_MAX * L ** 2 / (np.arange(1, modes + 1) * np.pi) ** 2
    th = None if theta is None else np.asarray(theta, float)
    cap_hits = None if (th is None or kind == 'anchor') else int(np.sum(np.abs(th[:modes]) >= cap - 1e-9))
    dv_hits = None if (th is None or kind == 'anchor') else int(np.sum(np.abs(th[modes:modes + IT.KNOTS]) >= IT.SP_CLIP - 1e-9))
    # lateral distance of the route from the straight base route, measured on the waypoints (independent of meta)
    d_base = float(np.max(np.min(np.linalg.norm(wp[:, None, :] - np.asarray(base['waypoints'], float)[None], axis=2), axis=1)))
    return dict(length_m=float(st[-1]), straight_m=straight, base_length_m=L_base, length_over_straight=float(st[-1] / max(straight, 1e-6)),
                T_s=float(IT.route_time(r)), mean_speed=float(v[1:-1].mean()), max_speed=float(v.max()), max_lateral_m=ml,
                max_dist_from_base_m=d_base, lat_clip_m=float(lat_clip), hits_lat_clip=bool(ml >= lat_clip - 0.05),
                mode1_cap_m=float(cap[0]), cap_hits=cap_hits, dv_clip_hits=dv_hits,
                max_abs_xy_m=float(np.abs(wp).max()), near_edge_34m=bool(np.abs(wp).max() > 34.0))


def cem_entry(res, rid, arm, spec, pose, wall, cpu):
    r = res['route']
    return dict(route_id=rid, arm=arm, label=spec['label'], tag=spec['tag'], objective='mean', index=res['index'], kind=res['kind'],
                round=res['round'], theta=res['theta'], z_mean=res['z_mean'], z_pess=res['z_pess'], P=res['P'], P_pess=pfail(res['z_pess']),
                T=res['T'], J=res['J'], mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()), max_lateral_m=float(r['meta'].get('max_lateral_m', 0.0)),
                length_m=float(np.asarray(r['stations'])[-1]), route_sha256=IT.route_sha256(r), n_evaluated=res['n_evaluated'], tries=res['tries'],
                n_below_1pct=int(((1 - np.exp(-np.exp(res['Z_mean']))) < 0.01).sum()), z_mean_min=float(res['Z_mean'].min()),
                z_pess_min=float(res['Z_pess'].min()), wall_s=wall, cpu_s=cpu, start_xy=[float(v) for v in np.asarray(r['waypoints'])[0]],
                start_dist_to_pose_m=float(np.linalg.norm(np.asarray(r['waypoints'])[0] - pose[:2])),
                log=res['log'], mu_final=res['mu'], sd_final=res['sd'],
                kinds_count={k: int(sum(1 for q in res['kinds'] if q == k)) for k in ('anchor', 'sample', 'mean')},
                family_stats=res.get('family_stats'))


def pool_rows(res, k=64):
    """Best k distinct candidates (route content) of a CEM run by ensemble-mean logit (stable order; row 0 = the pick)."""
    order = np.argsort(res['Z_mean'], kind='stable')
    assert int(order[0]) == res['index'] or res['Z_mean'][int(order[0])] == res['Z_mean'][res['index']]
    rows, seen = [], set()
    for i in [res['index']] + [int(j) for j in order if int(j) != res['index']]:
        r = res['candidates'][i]; h = IT.route_sha256(r)
        if h in seen:
            continue
        seen.add(h)
        kind = res['kinds'][i]; m = r.get('meta', {})
        row = dict(index=int(i), kind=kind, route_sha256=h, z_mean=float(res['Z_mean'][i]), z_pess=float(res['Z_pess'][i]))
        if kind == 'anchor' or m.get('candidate') == 'n2_anchor':
            row.update(kind='anchor', anchor=[float(m['lateral_offset_m']), float(m['cruise_speed_mps'])], theta=None)
        else:
            row.update(theta=[float(x) for x in m['theta']], anchor=None)
        rows.append(row)
        if len(rows) >= k:
            break
    return rows


# ------------------------------------------------------------------------------------------------------------------
# gradient refinement from given starts (the refinement / finals part of ci_grad.plan_group, verbatim in logic)
# ------------------------------------------------------------------------------------------------------------------
def refine_from_starts(ctx, g, base, pose, goal, scorer, starts, opts):
    """starts: list of dict(src, pool_index, kind, route, z_mean, z_pess); row 0 = the reference pick (the abstention
    baseline). Returns (entry pieces, chosen route, finals)."""
    ens, dens, tmap = ctx.ens, ctx.dens, ctx.tmap
    xy, station, f, L = IT._base_arrays(base)
    A0, D0, BSP, LAT0 = [], [], [], []
    for s in starts:
        aa, dd, anc = CG.start_params(s['route'], s['kind'])
        s['a0'], s['dv0'], s['anchor'] = aa, dd, anc
        rr = CG.np_route(base, aa, dd, anc)
        s['start_reproduced'] = IT.route_sha256(rr) == IT.route_sha256(s['route'])
        A0.append(aa); D0.append(dd)
        BSP.append(np.asarray(base['speeds'], float) if anc is None else np.full(len(xy), anc[1]))
        LAT0.append(np.zeros(len(xy)) if anc is None else anc[0] * np.sin(np.pi * f) ** 2)
    dens.set_context(scorer)
    chain = CG.Chain(base, pose, goal, tmap, dens, np.stack(BSP), np.stack(LAT0))
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    R = CG.refine(chain, np.stack(A0), np.stack(D0), steps=opts.steps, lr_a=opts.lr_a, lr_dv=opts.lr_dv, keep=opts.keep,
                  patience=opts.patience, clip_grad=opts.clip_grad)
    torch.cuda.synchronize()
    t_ref = time.perf_counter() - t1
    t2 = time.perf_counter()
    finals, fin_routes = [], []
    for k, s in enumerate(starts):
        refined = int(R['best_step'][k]) > 0
        route = CG.np_route(base, R['a'][k], R['dv'][k], s['anchor']) if refined else s['route']
        ok, info = CG.route_contract(route, pose, goal)
        finals.append(dict(row=k, src=s['src'], pool_index=s['pool_index'], kind=s['kind'], anchor=s['anchor'], start_reproduced=bool(s['start_reproduced']),
                           refined=refined, best_step=int(R['best_step'][k]), a=[float(x) for x in R['a'][k]], dv=[float(x) for x in R['dv'][k]],
                           z0_mean=s['z_mean'], z0_pess=s['z_pess'], J_torch=float(R['J_keep'][k]), z_torch_mean=float(R['Z'][:, k].mean()),
                           z_torch_pess=float(R['Z'][:, k].max()), pen_torch=float(R['pen'][k]), valid=bool(ok), **info, route_sha256=IT.route_sha256(route)))
        fin_routes.append(route)
    vidx = [k for k, fr in enumerate(finals) if fr['valid']]
    ns = len(starts)
    Zf, zmf, zpf = scorer([s['route'] for s in starts] + [fin_routes[k] for k in vidx])
    for k in range(ns):
        finals[k].update(z0_rescored_mean=float(zmf[k]), z0_rescored_pess=float(zpf[k]))
    for j, k in enumerate(vidx, start=ns):
        fr = finals[k]
        fr.update(z_mean=float(zmf[j]), z_pess=float(zpf[j]), z_members=[float(x) for x in Zf[:, j]], J_keep=CG.keep_value(opts.keep, zmf[j], zpf[j]),
                  J0_keep=CG.keep_value(opts.keep, zmf[k], zpf[k]), T=IT.route_time(fin_routes[k]))
        fr['dJ_vs_start'] = fr['J_keep'] - fr['J0_keep']
    J_ref = CG.keep_value(opts.keep, zmf[0], zpf[0])
    best = min(vidx, key=lambda k: finals[k]['J_keep']) if vidx else None
    gain = (J_ref - finals[best]['J_keep']) if best is not None else float('nan')
    abstain = best is None or not (gain >= opts.abstain_logit)
    t_fin = time.perf_counter() - t2
    if abstain:
        route = starts[0]['route']; z_mean, z_pess, z_members = starts[0]['z_mean'], starts[0]['z_pess'], None; fG = None
    else:
        fG = finals[best]; route = fin_routes[best]; z_mean, z_pess, z_members = fG['z_mean'], fG['z_pess'], fG['z_members']
    return dict(R=R, finals=finals, best=best, abstain=abstain, gain=gain, J_ref=J_ref, ref_rescored=dict(z_mean=float(zmf[0]), z_pess=float(zpf[0]),
                z_members=[float(x) for x in Zf[:, 0]]), fG=fG, z_mean=z_mean, z_pess=z_pess, z_members=z_members, t_ref=t_ref, t_fin=t_fin), route


def grad_entry(g, rid, arm, spec, out, route, pose, goal, wall, cpu, extra):
    fG = out['fG']; ab = out['abstain']
    okG, infoG = CG.route_contract(route, pose, goal)
    e = dict(route_id=rid, arm=arm, label=spec['label'], tag=spec['label'], objective='mean', keep=GRAD_OPTS['keep'], abstain_logit=GRAD_OPTS['abstain_logit'],
             abstained=bool(ab), gain=(float(out['gain']) if np.isfinite(out['gain']) else None), gain_unit='ensemble-mean logit, start pick minus refined',
             kind=('grad' if not ab else 'start_pick'), index=None, round=None,
             z_mean=float(out['z_mean']), z_pess=float(out['z_pess']), P=pfail(out['z_mean']), P_pess=pfail(out['z_pess']), z_members=out['z_members'],
             T=IT.route_time(route), J=float(out['z_mean']), mean_speed=float(np.asarray(route['speeds'])[1:-1].mean()),
             max_lateral_m=float(route['meta'].get('max_lateral_m', 0.0)), length_m=float(np.asarray(route['stations'])[-1]),
             route_sha256=IT.route_sha256(route), start_row=(None if ab else int(out['best'])), start_src=(None if ab else fG['src']),
             start_kind=(None if ab else fG['kind']), start_pool_index=(None if ab else fG['pool_index']), best_step=(None if ab else fG['best_step']),
             refined=(False if ab else fG['refined']), theta=(None if ab else dict(a=fG['a'], dv=fG['dv'], anchor=fG['anchor'])),
             n_finals=len(out['finals']), n_valid_finals=int(sum(fr['valid'] for fr in out['finals'])), steps_run=int(out['R']['steps_run']),
             contract=dict(valid=bool(okG), **infoG), start_xy=[float(v) for v in np.asarray(route['waypoints'])[0]],
             start_dist_to_pose_m=infoG['start_dist_m'], end_dist_to_goal_m=infoG['end_dist_m'], ref_rescored=out['ref_rescored'],
             wall_s=wall, cpu_s=cpu, seconds=dict(refine=out['t_ref'], finals=out['t_fin'], **extra.get('seconds', {})))
    e.update({k: v for k, v in extra.items() if k != 'seconds'})
    return e


# ------------------------------------------------------------------------------------------------------------------
# one arm, one group
# ------------------------------------------------------------------------------------------------------------------
def arm_dir(arm, root=None):
    return Path(root or OUT) / ARMS[arm]['dir']


def plan_one(ctx, arm, g, root=None, groups_meta=None):
    spec = ARMS[arm]; d = arm_dir(arm, root)
    (d / 'picks').mkdir(parents=True, exist_ok=True); (d / 'routes').mkdir(exist_ok=True)
    cp, case, pose, goal, base, hist, hmask, src = ctx.decision(g)
    rid = f'{g}__{arm}'
    w0, c0 = time.perf_counter(), time.process_time()
    fam_desc, extra_pick = None, {}
    if spec['kind'] == 'cem':
        scorer = ctx.scorer(pose, goal, hist, hmask)
        fam = CP.Family('free', 0.0, float(pose[2]))
        rng = np.random.default_rng(IT.seed(g, spec['tag']))
        keep = arm == 'A2'
        if spec['family'] == 'wide':
            with family_patch(spec['modes'], spec['lat_clip']):
                res = CP.plan_iter_family(base, pose, goal, scorer, fam, rounds=spec['rounds'], n=spec['n'], objective='mean', rng=rng, anchors=True,
                                          keep_candidates=keep)
            modes, clip = spec['modes'], spec['lat_clip']
        else:
            res = CP.plan_iter_family(base, pose, goal, scorer, fam, rounds=spec['rounds'], n=spec['n'], objective='mean', rng=rng, anchors=True,
                                      keep_candidates=keep)
            modes, clip = IT.MODES, IT.LAT_CLIP
        assert res is not None, f'{g} {arm}: no valid candidate'
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        route = res['route']
        entry = cem_entry(res, rid, arm, spec, pose, wall, cpu)
        fam_desc = dict(family=spec['family'], modes=modes, lat_clip_m=clip, knots=IT.KNOTS, prior_sd=(make_family(modes, clip)['prior_sd'].tolist()))
        if arm == 'A0':
            ref = json.load(open(REF_DIR / 'picks' / f'{g}.json'))['arms']['B']
            ref_bytes = (REF_DIR / 'routes' / f"{ref['route_id']}.json").read_bytes()
            mine_bytes = json.dumps(PA.route_json(route, f'{g}__B', g, 'B', ctx.specB['label'], 'crm')).encode()
            same_rec = all(json.dumps(entry.get(k), sort_keys=True, default=_jdefault) == json.dumps(ref.get(k), sort_keys=True, default=_jdefault)
                           for k in ref if k not in ('route_id', 'arm', 'label', 'tag', 'wall_s'))
            entry['repro'] = dict(route_sha256_equal=entry['route_sha256'] == ref['route_sha256'], P_abs_diff=abs(entry['P'] - ref['P']),
                                  z_mean_abs_diff=abs(entry['z_mean'] - ref['z_mean']), route_bytes_equal=mine_bytes == ref_bytes,
                                  pick_record_equal=bool(same_rec), ref_route_file=os.path.relpath(REF_DIR / 'routes' / f"{ref['route_id']}.json", ROOT))
        if keep:
            jdump(dict(group=g, arm=arm, rows=pool_rows(res, 64)), _pool_path(d, g))
        score_calls = scorer.calls; history = scorer.history
    else:
        opts = argparse.Namespace(starts=spec['starts'], steps=spec['steps'], patience=spec['patience'], **GRAD_OPTS)
        if spec['seed_from'] == 'A0':
            summ, rB, route = CG.plan_group(g, cp, None, ctx.ens, ctx.dens, ctx.tmap, opts, ctx.specB, 'crm')
            wall, cpu = time.perf_counter() - w0, time.process_time() - c0
            eB, eG = summ['arms']['B'], summ['arms']['G']
            ref = json.load(open(REF_DIR / 'picks' / f'{g}.json'))['arms']['B']
            out = dict(fG=None, abstain=eG['abstained'], gain=eG['gain'] if eG['gain'] is not None else float('nan'),
                       z_mean=eG['z_mean'], z_pess=eG['z_pess'], z_members=eG['z_members'], finals=summ['grad']['rows'],
                       R=dict(steps_run=summ['grad']['steps_run']), best=eG['start_row'], t_ref=summ['seconds']['refine'], t_fin=summ['seconds']['finals'],
                       ref_rescored=dict(eG['B_rescored'], z_members=eB.get('z_members_rescored')))
            if not eG['abstained']:
                out['fG'] = dict(src=eG['start_src'], kind=eG['start_kind'], pool_index=eG['start_pool_index'], best_step=eG['best_step'], refined=eG['refined'],
                                 a=eG['theta']['a'], dv=eG['theta']['dv'], anchor=eG['theta']['anchor'])
            entry = grad_entry(g, rid, arm, spec, out, route, pose, goal, wall, cpu,
                               dict(seconds=dict(cem=summ['seconds']['cem'], total=summ['seconds']['total']),
                                    start_pick=dict(arm='A0', route_sha256=eB['route_sha256'], z_mean=eB['z_mean'], z_pess=eB['z_pess'], P=eB['P'],
                                                    equals_recorded_pick=eB['route_sha256'] == ref['route_sha256']),
                                    starts_reproduced=summ['grad']['starts_reproduced'], keep_best_ok=summ['grad']['keep_best_ok'],
                                    flags=eG['flags'], via='ci_grad.plan_group(poses=None)'))
            grad_rows = summ['grad']['rows']; score_calls = summ['score_calls']; history = summ['history']
        else:
            src_arm = spec['seed_from']
            pool = json.load(open(arm_dir(src_arm) / 'pool' / f'{g}.json'))
            scorer = ctx.scorer(pose, goal, hist, hmask)
            starts = []
            for k, p in enumerate(pool['rows'][:spec['starts']]):
                if p['kind'] == 'anchor':
                    rr = CG.np_route(base, np.zeros(IT.MODES), np.zeros(IT.KNOTS), tuple(p['anchor']))
                    rr['meta'].update(candidate='n2_anchor', lateral_offset_m=p['anchor'][0], cruise_speed_mps=p['anchor'][1])
                else:
                    rr = IT.from_params(base, np.asarray(p['theta'], float))
                assert IT.route_sha256(rr) == p['route_sha256'], f'{g}: {src_arm} pool row {k} does not rebuild'
                starts.append(dict(src=f'{src_arm}_pick' if k == 0 else f'{src_arm}_pool', pool_index=p['index'], kind=p['kind'], route=rr,
                                   z_mean=p['z_mean'], z_pess=p['z_pess']))
            src_pick = json.load(open(arm_dir(src_arm) / 'picks' / f'{g}.json'))['arms'][src_arm]
            assert starts[0]['route']['meta'] is not None and pool['rows'][0]['route_sha256'] == src_pick['route_sha256']
            out, route = refine_from_starts(ctx, g, base, pose, goal, scorer, starts, opts)
            wall, cpu = time.perf_counter() - w0, time.process_time() - c0
            entry = grad_entry(g, rid, arm, spec, out, route, pose, goal, wall, cpu,
                               dict(start_pick=dict(arm=src_arm, route_sha256=src_pick['route_sha256'], z_mean=src_pick['z_mean'], z_pess=src_pick['z_pess'], P=src_pick['P']),
                                    starts_reproduced=bool(all(s['start_reproduced'] for s in starts)),
                                    keep_best_ok=bool((out['R']['J_keep'] <= out['R']['J0'] + 1e-6).all()), via='sp_search_probe.refine_from_starts'))
            grad_rows = out['finals']; score_calls = scorer.calls; history = scorer.history
        modes, clip = IT.MODES, IT.LAT_CLIP
        fam_desc = dict(family='free', modes=modes, lat_clip_m=clip, knots=IT.KNOTS)
        extra_pick['grad'] = dict(n_starts=spec['starts'], steps=spec['steps'], patience=spec['patience'], **GRAD_OPTS, rows=grad_rows)
    theta = entry.get('theta')
    th = (None if theta is None else (np.r_[theta['a'], theta['dv']] if isinstance(theta, dict) else np.asarray(theta, float)))
    if isinstance(theta, dict) and theta.get('anchor') is not None:
        th = None
    geo = route_geometry(route, pose, goal, base, clip, modes, th, entry.get('kind'))
    entry['geometry'] = geo
    meta_extra = dict(probe='search_probe_20260927', probe_arm=arm, probe_arm_words=spec['words'])
    rj = PA.route_json(route, rid, g, arm, spec['label'], 'crm')
    rj['meta'].update(meta_extra)
    if spec['kind'] == 'grad':
        rj['meta'].update(kind=entry['kind'], grad_theta=entry['theta'], abstained=entry['abstained'], gain_logit=entry['gain'],
                          start_src=entry['start_src'], best_step=entry['best_step'], z_mean=entry['z_mean'], z_pess=entry['z_pess'])
    rpath = d / 'routes' / f'{rid}.json'
    tmp = Path(str(rpath) + '.tmp'); tmp.write_text(json.dumps(rj)); os.replace(tmp, rpath)
    assert IT.route_sha256(json.load(open(rpath))) == entry['route_sha256']
    gm = groups_meta or load_groups()
    sets = [s for s in ('fail', 'fail_p50', 'control') if g in set(gm['groups'][s])]
    summ = dict(group=g, world='crm', domain='crm', fixed2=False, stratum=case.get('evaluation_stratum'), base_length_m=float(base['stations'][-1]),
                corridors_batched=IT._CORR.get('identical'), arms={arm: entry}, pose=[float(v) for v in pose], goal=[float(v) for v in goal],
                source=src, history=history, score_calls=score_calls,
                family=dict(v0_mps=0.0, v0_source='standing_start', yaw=float(pose[2]), **fam_desc),
                probe=dict(arm=arm, arm_words=spec['words'], sets=sets, original=gm['original'][g]), **extra_pick)
    jdump(summ, d / 'picks' / f'{g}.json')
    return entry


def _pool_path(d, g):
    (Path(d) / 'pool').mkdir(parents=True, exist_ok=True)
    return Path(d) / 'pool' / f'{g}.json'


def cmd_plan(a):
    arms = [s for s in a.arms.split(',') if s]
    for arm in arms:
        assert arm in ARMS, arm
    gm = load_groups()
    groups = gm['groups']['all'] if not a.only else [s for s in a.only.split(',') if s]
    if a.order == 'priority':      # the FAIL P>0.5 pairs first, then the other FAIL pairs, then CONTROL (md5 order inside)
        rank = {g: (0 if g in set(gm['groups']['fail_p50']) else 1 if g in set(gm['groups']['fail']) else 2) for g in groups}
        groups = sorted(groups, key=lambda g: (rank[g], md5hex(g)))
    groups = [g for i, g in enumerate(groups) if i % a.nshards == a.shard]
    need_grad = any(ARMS[x]['kind'] == 'grad' for x in arms)
    torch.set_num_threads(1)
    ctx = Ctx(need_grad=need_grad, device=a.device)
    root = Path(a.root) if a.root else OUT
    t0 = time.time()
    print(f'plan arms {arms} shard {a.shard}/{a.nshards}: {len(groups)} groups -> {root} on {ctx.ens.dev}', flush=True)
    for gi, g in enumerate(groups):
        line = []
        for arm in arms:
            pf = arm_dir(arm, root) / 'picks' / f'{g}.json'
            if g not in set(gm['groups'][ARMS[arm].get('groups', 'all')]):
                continue
            if pf.exists() and not a.force:
                line.append(f'{arm}=done'); continue
            e = plan_one(ctx, arm, g, root, gm)
            torch.cuda.empty_cache()     # hand cached blocks back after every arm (the GPU is shared with other jobs)
            s = f"{arm} P {e['P']:.4f} pess {e['P_pess']:.3f} {e['wall_s']:.1f}s"
            if arm == 'A0':
                s += f" repro {e['repro']['route_bytes_equal']}/{e['repro']['pick_record_equal']}"
            line.append(s)
        print(f'  [{gi + 1}/{len(groups)}] {g} orig P {gm["original"][g]["P"]:.4f} | ' + ' | '.join(line) + f'  ({time.time() - t0:.0f}s)', flush=True)
    print(f'done {len(groups)} groups in {time.time() - t0:.0f}s', flush=True)


# ------------------------------------------------------------------------------------------------------------------
# self-test
# ------------------------------------------------------------------------------------------------------------------
def cmd_selftest(a):
    gm = load_groups()
    ctx = Ctx(need_grad=True, device=a.device)
    groups = gm['groups']['fail_p50'][:2] + gm['groups']['control'][:1]
    res = dict(groups=groups, checks={})
    # T1: the parameterised family with (3 modes, 10 m) equals f104_n2_iter's own functions (rng stream + route bytes)
    fam3 = make_family(IT.MODES, IT.LAT_CLIP)
    ok1 = True
    for g in groups:
        cp, case, pose, goal, base, hist, hmask, src = ctx.decision(g)
        xy, station, f, L = IT._base_arrays(base)
        r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
        for _ in range(50):
            t1, t2 = IT.draw_prior(r1, L), fam3['draw_prior'](r2, L)
            ok1 &= bool(np.array_equal(t1, t2))
            ok1 &= IT.route_sha256(IT.from_params(base, t1)) == IT.route_sha256(fam3['from_params'](base, t2))
            rq = np.random.default_rng(len(g) + _); mu = rq.normal(0, 3, 7); sd = np.abs(rq.normal(1, 0.2, 7))
            ok1 &= bool(np.array_equal(IT.project(mu, L), fam3['project'](mu, L)))
            ok1 &= bool(np.array_equal(IT.draw_prior(np.random.default_rng(3), L, None, mu, sd), fam3['draw_prior'](np.random.default_rng(3), L, None, mu, sd)))
    res['checks']['T1_family3_equals_IT'] = bool(ok1)
    # T2: A0 through the patched path with (3, 10 m) reproduces the recorded pick bytes
    ok2 = []
    for g in groups:
        cp, case, pose, goal, base, hist, hmask, src = ctx.decision(g)
        sc = ctx.scorer(pose, goal, hist, hmask)
        with family_patch(IT.MODES, IT.LAT_CLIP):
            r = CP.plan_iter_family(base, pose, goal, sc, CP.Family('free', 0.0, float(pose[2])), rounds=4, n=64, objective='mean',
                                    rng=np.random.default_rng(IT.seed(g, 'n2iter_cem4x64')), anchors=True)
        ref = json.load(open(REF_DIR / 'picks' / f'{g}.json'))['arms']['B']
        b = json.dumps(PA.route_json(r['route'], f'{g}__B', g, 'B', 'cem4x64', 'crm')).encode() == (REF_DIR / 'routes' / f"{ref['route_id']}.json").read_bytes()
        ok2.append(bool(b and r['P'] == ref['P']))
    res['checks']['T2_patched_A0_reproduces'] = ok2
    res['checks']['T2b_patch_restored'] = bool(IT.MODES == 3 and IT.from_params.__module__ == 'f104_n2_iter' and len(IT.PRIOR_SD) == 7)
    # T3: the wide family (5, 20 m) builds valid routes whose theta has 9 entries and respects the caps
    g = groups[0]; cp, case, pose, goal, base, hist, hmask, src = ctx.decision(g)
    w = make_family(5, 20.0); xy, station, f, L = IT._base_arrays(base); rng = np.random.default_rng(1); nv = 0; mx = 0.0
    for _ in range(400):
        th = w['draw_prior'](rng, L); r = w['from_params'](base, th)
        assert len(r['meta']['theta']) == 9 and np.all(np.abs(np.asarray(r['meta']['theta'][:5])) <= w['caps'](L) + 1e-12)
        if IT.valid(r, pose):
            nv += 1; mx = max(mx, r['meta']['max_lateral_m'])
    res['checks']['T3_wide_prior_acceptance'] = nv / 400; res['checks']['T3_wide_prior_max_lateral_m'] = mx; res['checks']['T3_caps_m'] = w['caps'](L).tolist()
    # T4: refine_from_starts with ci_grad.plan_group's own starts reproduces its G pick (short run: 40 steps)
    opts = argparse.Namespace(starts=64, steps=40, patience=50, **GRAD_OPTS)
    ok4 = []
    for g in groups[:2]:
        cp, case, pose, goal, base, hist, hmask, src = ctx.decision(g)
        summ, rB, rG = CG.plan_group(g, cp, None, ctx.ens, ctx.dens, ctx.tmap, opts, ctx.specB, 'crm')
        sc = ctx.scorer(pose, goal, hist, hmask)
        r = CP.plan_iter_family(base, pose, goal, sc, CP.Family('free', 0.0, float(pose[2])), rounds=4, n=64, objective='mean',
                                rng=np.random.default_rng(IT.seed(g, 'n2iter_cem4x64')), anchors=True, keep_candidates=True)
        n0 = int(r['log'][0]['n'])
        order0 = [int(i) for i in np.argsort(r['Z_mean'][:n0], kind='stable') if int(i) != r['index']][:63]
        starts = [dict(src='B', pool_index=int(r['index']), kind=r['kind'], route=r['route'], z_mean=r['z_mean'], z_pess=r['z_pess'])]
        starts += [dict(src='round0', pool_index=i, kind=r['kinds'][i], route=r['candidates'][i], z_mean=float(r['Z_mean'][i]), z_pess=float(r['Z_pess'][i])) for i in order0]
        out, route = refine_from_starts(ctx, g, base, pose, goal, sc, starts, opts)
        ok4.append(dict(group=g, same_route=IT.route_sha256(route) == IT.route_sha256(rG), z_mine=out['z_mean'], z_ci_grad=summ['arms']['G']['z_mean']))
    res['checks']['T4_refine_from_starts_equals_ci_grad'] = ok4
    res['ok'] = bool(ok1 and all(ok2) and res['checks']['T2b_patch_restored'] and all(x['same_route'] for x in ok4))
    (OUT / 'checks').mkdir(parents=True, exist_ok=True)
    jdump(res, OUT / 'checks' / 'selftest.json')
    print(json.dumps(res, indent=1, default=_jdefault))
    assert res['ok'], 'selftest failed'


# ------------------------------------------------------------------------------------------------------------------
# determinism re-run on a few groups per arm (fresh process per arm, scratch root)
# ------------------------------------------------------------------------------------------------------------------
def cmd_rerun(a):
    gm = load_groups()
    res = {}
    with tempfile.TemporaryDirectory(prefix='sp_rerun_') as td:
        for arm in [x for x in (a.arms.split(',') if a.arms else ORDER)]:
            groups = gm['groups'][ARMS[arm].get('groups', 'all')][:a.n]
            root = Path(td) / arm      # A3b reads the main A2 pool (a deterministic input, checked by the A2 re-run)
            cmd = [PY, str(Path(__file__).resolve()), 'plan', '--arms', arm, '--only', ','.join(groups), '--root', str(root)]
            env = dict(os.environ, PYTHONPATH=f'{ROOT / "src"}:{HERE}', OMP_NUM_THREADS='1')
            p = __import__('subprocess').run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
            assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
            diffs = []
            for g in groups:
                f1 = arm_dir(arm) / 'routes' / f'{g}__{arm}.json'; f2 = arm_dir(arm, root) / 'routes' / f'{g}__{arm}.json'
                e1 = json.load(open(arm_dir(arm) / 'picks' / f'{g}.json'))['arms'][arm]; e2 = json.load(open(arm_dir(arm, root) / 'picks' / f'{g}.json'))['arms'][arm]
                strip = lambda e: {k: v for k, v in e.items() if k not in ('wall_s', 'cpu_s', 'seconds')}
                if f1.read_bytes() != f2.read_bytes() or json.dumps(strip(e1), sort_keys=True, default=_jdefault) != json.dumps(strip(e2), sort_keys=True, default=_jdefault):
                    diffs.append(dict(group=g, bytes_equal=f1.read_bytes() == f2.read_bytes(), P1=e1['P'], P2=e2['P']))
            res[arm] = dict(groups=groups, identical=not diffs, differing=diffs, note='fresh process, only this arm, same machine')
            print(arm, res[arm]['identical'], diffs[:3], flush=True)
    (OUT / 'checks').mkdir(parents=True, exist_ok=True)
    old = json.load(open(OUT / 'checks' / 'rerun.json')) if (OUT / 'checks' / 'rerun.json').exists() else {}
    old.update(res)
    jdump(old, OUT / 'checks' / 'rerun.json')


# ------------------------------------------------------------------------------------------------------------------
# finalize: tasks / summary / lock / manifest per arm
# ------------------------------------------------------------------------------------------------------------------
def lock_routes(d):
    lock = hashlib.sha256()
    for p in sorted((Path(d) / 'routes').glob('*.json')):
        lock.update(p.name.encode()); lock.update(hashlib.sha256(p.read_bytes()).digest())
    return lock.hexdigest()


def numerics():
    return dict(device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu', torch=torch.__version__,
                cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32), matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                note='same process defaults as the original picks (TF32 convolutions, float16-rounded corridors); gradient arms run '
                     'their chain in ci_grad.det_ctx (deterministic cuDNN) and re-score finals with the deployed scorer')


def cmd_finalize(a):
    gm = load_groups(); groups = gm['groups']['all']
    ref = ref_manifest()
    for m in ref['models']:
        assert sha256_file(ROOT / m['path']) == m['sha256'], m['path']
    cdir = cases_dir()
    chk = ag_map_check.check(MAP_ROOT, [str(cdir / f'{g}.json') for g in groups], ROOT)
    assert chk['ok'], chk['problems']
    rerun = json.load(open(OUT / 'checks' / 'rerun.json')) if (OUT / 'checks' / 'rerun.json').exists() else {}
    for arm in (a.arms.split(',') if a.arms else ORDER + [x for x in ABL if arm_dir(x).exists()]):
        spec = ARMS[arm]; d = arm_dir(arm)
        groups = gm['groups'][spec.get('groups', 'all')]
        missing = [g for g in groups if not (d / 'picks' / f'{g}.json').exists()]
        assert not missing, f'{arm}: {len(missing)} groups not planned, e.g. {missing[:3]}'
        (d / 'groups.txt').write_text('\n'.join(groups) + '\n')
        tasks, picks, rows = [], {}, []
        for gi, g in enumerate(groups):
            pk = json.load(open(d / 'picks' / f'{g}.json')); e = pk['arms'][arm]; rows.append(e)
            rf = d / 'routes' / f'{e["route_id"]}.json'
            r = json.load(open(rf)); h = IT.route_sha256(r)
            assert h == e['route_sha256'], (arm, g)
            case = json.load(open(cdir / f'{g}.json')); lay = case['layout']
            d0 = float(np.linalg.norm(np.asarray(r['waypoints'])[0] - np.asarray(lay['start_xy']))); d1 = float(np.linalg.norm(np.asarray(r['waypoints'])[-1] - np.asarray(case['goal_xy'])))
            assert d0 <= 0.25 and d1 <= 0.25, (arm, g, d0, d1)
            ok = GP.safe_validate({k: np.asarray(r[k], float) for k in ('waypoints', 'speeds', 'stations', 'headings')}, [], GP.CFG, np.asarray(pk['pose'], float))['valid']
            assert ok, (arm, g, 'route file fails the validator')
            v = np.asarray(r['speeds'], float)
            picks[g] = dict(route_id=e['route_id'], route_sha256=h, route_file=str(rf.relative_to(d)), file_sha256=sha256_file(rf), P=e['P'], z_mean=e['z_mean'],
                            P_pess=e['P_pess'], z_pess=e['z_pess'], mean_speed=float(v[1:-1].mean()), max_speed=float(v.max()),
                            length_m=float(np.asarray(r['stations'])[-1]), stratum=case.get('evaluation_stratum'), sets=pk['probe']['sets'],
                            original_P=pk['probe']['original']['P'], original_fail=pk['probe']['original']['fail'])
            rid = e['route_id']
            tasks.append(dict(id=rid, group=g, case=os.path.relpath((cdir / f'{g}.json').resolve(), OUT.resolve()),
                              route=os.path.relpath(rf.resolve(), OUT.resolve()), run=True, tier=gi,
                              episode_seed=int(hashlib.md5(rid.encode()).hexdigest()[:8], 16), arms=[arm], sha256=h))
        jdump(tasks, d / 'tasks.json')
        lock = lock_routes(d)
        (d / 'PICKS_LOCKED.sha256').write_text(lock + '  routes/*.json (name + content, sorted)\n')
        wall = np.array([e['wall_s'] for e in rows]); cpu = np.array([e['cpu_s'] for e in rows]); P = np.array([e['P'] for e in rows])
        summary = dict(world='crm', domain='crm', fixed2=False, n_groups=len(groups), arms=[arm], tags={arm: spec.get('tag', spec['label'])}, spec=spec,
                       models=models_glob(), model_kinds=['ci_train'], conds=['none'], map_root=str(MAP_ROOT), n_distinct_routes=len({t['sha256'] for t in tasks}),
                       float16_scoring=True, P_mean=float(P.mean()), P_median=float(np.median(P)), wall_s_total=float(wall.sum()),
                       wall_s_per_group=dict(mean=float(wall.mean()), median=float(np.median(wall)), max=float(wall.max())),
                       cpu_s_per_group=dict(mean=float(cpu.mean()), median=float(np.median(cpu))),
                       task_root=os.path.relpath(OUT, ROOT))
        if arm == 'A0':
            rp = [e['repro'] for e in rows]
            summary['reproduction'] = dict(n=len(rp), route_sha256_equal=int(sum(x['route_sha256_equal'] for x in rp)),
                                           route_bytes_equal=int(sum(x['route_bytes_equal'] for x in rp)), pick_record_equal=int(sum(x['pick_record_equal'] for x in rp)),
                                           P_abs_diff_max=float(max(x['P_abs_diff'] for x in rp)))
        jdump(summary, d / 'summary.json')
        cmdline = {'cem': f"{PY} scripts/sp_search_probe.py plan --arms {arm} [--shard k --nshards K]",
                   'grad': f"{PY} scripts/sp_search_probe.py plan --arms {arm} [--shard k --nshards K]"}[spec['kind']]
        man = dict(schema='ag_picks_v1', tool='scripts/sp_search_probe.py', tool_sha256=sha256_file(__file__), argv=['plan', '--arms', arm],
                   created=time.strftime('%Y-%m-%d %H:%M:%S'), host=os.uname().nodename, arena='f104', arena_dir=ref['arena_dir'], world='crm',
                   mode='free' if spec.get('family', 'free') == 'free' else 'free_wide', arm_key=arm, model_tag='G_full',
                   planner=dict(command=cmdline, description=spec['words'], spec=spec, grad_opts=GRAD_OPTS if spec['kind'] == 'grad' else None,
                                start='case layout pose at rest (standing start, all-masked history)', objective='ensemble-mean route logit',
                                validator='gen_planner.safe_validate anchored at the pose (curvature 0.125 1/m, speed 0-6 m/s, accel 1.5 / decel 2.0 m/s^2, footprint within +-40 m)',
                                seed_rule='rng default_rng(md5(group + tag)[:8])' if spec['kind'] == 'cem' else 'deterministic (ci_grad chain, det_ctx)'),
                   straight=None, models=ref['models'], models_glob=ref['models_glob'], cases_dir=ref['cases_dir'], map_root=os.path.relpath(MAP_ROOT, ROOT),
                   map_check=chk, groups_spec=f"@groups/{spec.get('groups', 'all')}.txt",
                   groups_rule=('FAIL (261) + CONTROL (100)' if spec.get('groups', 'all') == 'all' else 'FAIL with recorded P > 0.5 (134)') + ', md5(group id) order',
                   n_groups=len(groups),
                   groups_sha256=hashlib.sha256('\n'.join(groups).encode()).hexdigest(), picks_locked_sha256=lock,
                   n_route_files=len(list((d / 'routes').glob('*.json'))), wall_s=float(wall.sum()), planner_wall_s=float(wall.sum()),
                   rerun_check=rerun.get(arm), selfcheck_straight=None, numerics=numerics(), task_root=os.path.relpath(OUT, ROOT), picks=picks,
                   not_driven='offline probe: nothing was submitted; see RESULTS.md for how rows would be built (ag_eval_tasks.py build ... @gator)')
        jdump(man, d / 'ag_picks.json')
        print(arm, 'tasks', len(tasks), 'lock', lock[:16], 'P mean', round(float(P.mean()), 4), 'wall/group', round(float(wall.mean()), 2), flush=True)


# ------------------------------------------------------------------------------------------------------------------
# rescore: every arm's pick of a group (and the recorded pick) scored together in ONE batch, per member
# ------------------------------------------------------------------------------------------------------------------
def load_route(p):
    r = json.load(open(p))
    out = {k: np.asarray(r[k], float) for k in ('waypoints', 'speeds', 'stations', 'headings')}
    out['meta'] = r.get('meta', {})
    return out


SECOND = K3 / 'e5/deploy/G_soil/G_soil_deploy_s*.pt'      # stage-1 Gator soil ensemble (tiers 0-6: a subset of G_full's data)


def terrain(r, step=0.5):
    """Terrain along the route centre line from the static depth map (the planner's own elevation source): total climb
    and descent (sum of positive / negative elevation steps every 0.5 m), highest point above the start, steepest 2 m
    grade uphill along the direction of travel (percent)."""
    wp = np.asarray(r['waypoints'], float); st = np.asarray(r['stations'], float)
    n = max(int(st[-1] / step) + 1, 2)
    pts, grid = DS.resample_route(wp, st, n)
    patch, vld = DS.sample_map(pts[:, 0], pts[:, 1])
    z = np.where(vld, patch[3] * DS.G['elev_scale'], np.nan).astype(float)
    if np.isnan(z).any():
        ok = ~np.isnan(z)
        z = np.interp(grid, grid[ok], z[ok]) if ok.any() else np.zeros_like(grid)
    dz = np.diff(z); k = max(int(round(2.0 / step)), 1)
    g2 = (z[k:] - z[:-k]) / (grid[k:] - grid[:-k]) if len(z) > k else np.zeros(1)
    return dict(climb_m=float(np.clip(dz, 0, None).sum()), descent_m=float(-np.clip(dz, None, 0).sum()), max_rise_m=float((z - z[0]).max()),
                max_uphill_grade_pct=float(100 * g2.max()), n_invalid=int((~vld).sum()))


def cmd_rescore(a):
    gm = load_groups(); groups = gm['groups']['all']
    ctx = Ctx(device=a.device)
    ens2 = CP.CIEnsemble(str(SECOND), a.device)
    assert ens2.kinds == ['ci_train'] and ens2.conds == ['none'] and not ens2.needs_hist
    sha2 = {os.path.basename(m['path']): sha256_file(m['path']) for m in ens2.members}
    out = {}
    for gi, g in enumerate(groups):
        cp, case, pose, goal, base, hist, hmask, src = ctx.decision(g)
        sc = ctx.scorer(pose, goal, hist, hmask)
        abl = [x for x in ABL if (arm_dir(x) / 'routes' / f'{g}__{x}.json').exists()]
        names = ['original'] + ORDER + abl
        ref = json.load(open(REF_DIR / 'picks' / f'{g}.json'))['arms']['B']
        paths = [REF_DIR / 'routes' / f"{ref['route_id']}.json"] + [arm_dir(x) / 'routes' / f'{g}__{x}.json' for x in ORDER + abl]
        routes = [load_route(p) for p in paths]
        Z, zm, zp = sc(routes)
        sc2 = CP.CIScorer(ens2, pose[:2], goal, float(pose[2]), 'crm', hist, hmask, float16=True)
        Z2, zm2, zp2 = sc2(routes)
        out[g] = {n: dict(z_members=[float(v) for v in Z[:, i]], z_mean=float(zm[i]), z_pess=float(zp[i]), P=pfail(zm[i]), P_pess=pfail(zp[i]),
                          route_sha256=IT.route_sha256(routes[i]), terrain=terrain(routes[i]),
                          second=dict(z_members=[float(v) for v in Z2[:, i]], z_mean=float(zm2[i]), P=pfail(zm2[i]), P_pess=pfail(zp2[i])))
                  for i, n in enumerate(names)}
        if (gi + 1) % 50 == 0:
            print(f'  rescored {gi + 1}/{len(groups)}', flush=True)
    jdump(dict(note='each group: the recorded pick and every arm pick scored in one batch by the deployed scorer (float16 corridors, G_full '
                    'ensemble, standing start); per-member logits. second = the same routes scored by the stage-1 Gator soil ensemble '
                    '(G_soil, tiers 0-6 = a subset of G_full training data; a correlated second opinion, not an independent one)',
               second_models=os.path.relpath(SECOND, ROOT), second_sha256=sha2, groups=out), OUT / 'rescore.json')
    print('rescore done', len(out))


# ------------------------------------------------------------------------------------------------------------------
# report
# ------------------------------------------------------------------------------------------------------------------
def q(x, qs=(0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0)):
    x = np.asarray(x, float)
    return [float(v) for v in np.quantile(x, qs)] if len(x) else None


def cmd_report(a):
    gm = load_groups()
    sets = dict(FAIL=gm['groups']['fail'], FAIL_P50=gm['groups']['fail_p50'], CONTROL=gm['groups']['control'],
                FAIL_P_LE_50=[g for g in gm['groups']['fail'] if g not in set(gm['groups']['fail_p50'])])
    orig = gm['original']
    rs = json.load(open(OUT / 'rescore.json'))['groups']
    E = {arm: {g: json.load(open(arm_dir(arm) / 'picks' / f'{g}.json'))['arms'][arm] for g in gm['groups']['all']} for arm in ORDER}
    # the recorded pick's own record + geometry (from A0, identical route)
    res = dict(arms={k: dict(words=ARMS[k]['words'], dir=ARMS[k]['dir']) for k in ORDER}, sets={k: len(v) for k, v in sets.items()}, per_arm={}, key={},
               checks={})
    rp = [E['A0'][g]['repro'] for g in gm['groups']['all']]
    res['checks']['A0_reproduction'] = dict(n=len(rp), route_bytes_equal=int(sum(x['route_bytes_equal'] for x in rp)),
                                            route_sha256_equal=int(sum(x['route_sha256_equal'] for x in rp)),
                                            pick_record_equal=int(sum(x['pick_record_equal'] for x in rp)), P_abs_diff_max=float(max(x['P_abs_diff'] for x in rp)))
    dmax = max(abs(rs[g][arm]['z_mean'] - E[arm][g]['z_mean']) for g in gm['groups']['all'] for arm in ORDER)
    res['checks']['rescore_vs_search_max_abs_dlogit'] = float(dmax)
    res['checks']['original_rescore_vs_recorded_max_abs_dP'] = float(max(abs(rs[g]['original']['P'] - orig[g]['P']) for g in gm['groups']['all']))
    for arm in ORDER:
        pa = {}
        for sname, gs in sets.items():
            if not gs:
                pa[sname] = dict(n=0); continue
            P = np.array([E[arm][g]['P'] for g in gs]); Pp = np.array([E[arm][g]['P_pess'] for g in gs])
            Po = np.array([orig[g]['P'] for g in gs])
            geo = [E[arm][g]['geometry'] for g in gs]
            geo0 = [E['A0'][g]['geometry'] for g in gs]
            wall = np.array([E[arm][g]['wall_s'] for g in gs]); cpu = np.array([E[arm][g]['cpu_s'] for g in gs])
            Pm = np.array([rs[g][arm]['P'] for g in gs]); Ppm = np.array([rs[g][arm]['P_pess'] for g in gs])
            P2 = np.array([rs[g][arm]['second']['P'] for g in gs])
            lr = np.array([x['length_over_straight'] for x in geo]); l0 = np.array([x['length_m'] for x in geo0]); ln = np.array([x['length_m'] for x in geo])
            T = np.array([x['T_s'] for x in geo]); T0 = np.array([x['T_s'] for x in geo0])
            pa[sname] = dict(
                n=len(gs), P_deciles=q(P), P_mean=float(P.mean()),
                n_P_lt_0p01=int((P < 0.01).sum()), n_P_lt_0p05=int((P < 0.05).sum()), n_P_lt_0p2=int((P < 0.2).sum()), n_P_gt_0p5=int((P > 0.5).sum()),
                P_pess_deciles=q(Pp), n_Ppess_lt_0p05=int((Pp < 0.05).sum()), n_Ppess_gt_0p5=int((Pp > 0.5).sum()),
                n_disagree_mean_lt_0p05_max_gt_0p5=int(((P < 0.05) & (Pp > 0.5)).sum()),
                n_mean_lt_0p05_max_gt_0p2=int(((P < 0.05) & (Pp > 0.2)).sum()),
                n_lower_than_original=int((P < Po - 1e-12).sum()), n_higher_than_original=int((P > Po + 1e-12).sum()),
                median_P_ratio_to_original=float(np.median(P / np.maximum(Po, 1e-12))),
                rescored_common_batch=dict(n_P_lt_0p05=int((Pm < 0.05).sum()), n_P_gt_0p5=int((Pm > 0.5).sum()), n_Ppess_gt_0p5=int((Ppm > 0.5).sum())),
                second_opinion=dict(P_deciles=q(P2), n_P_lt_0p05=int((P2 < 0.05).sum()), n_P_gt_0p5=int((P2 > 0.5).sum()),
                                    n_safe_both=int(((P < 0.05) & (P2 < 0.05)).sum()), n_safe_first_but_second_gt_0p5=int(((P < 0.05) & (P2 > 0.5)).sum())),
                geometry=dict(length_over_straight=q(lr, (0, .5, .9, 1.0)), length_over_straight_mean=float(lr.mean()),
                              length_vs_original_ratio=q(ln / l0, (0, .5, .9, 1.0)), T_s=q(T, (0, .5, .9, 1.0)), T_vs_original_ratio=q(T / T0, (0, .5, .9, 1.0)),
                              max_lateral_m=q([x['max_lateral_m'] for x in geo], (0, .5, .9, 1.0)),
                              max_dist_from_base_m=q([x['max_dist_from_base_m'] for x in geo], (0, .5, .9, 1.0)),
                              n_hits_lat_clip=int(sum(x['hits_lat_clip'] for x in geo)),
                              n_any_cap_hit=int(sum(1 for x in geo if x['cap_hits'])),
                              n_dv_clip_ge3=int(sum(1 for x in geo if (x['dv_clip_hits'] or 0) >= 3)),
                              n_near_edge_34m=int(sum(x['near_edge_34m'] for x in geo)), max_abs_xy_m=float(max(x['max_abs_xy_m'] for x in geo)),
                              mean_speed=q([x['mean_speed'] for x in geo], (0, .5, .9, 1.0)), n_mean_speed_gt5=int(sum(x['mean_speed'] > 5.0 for x in geo)),
                              n_anchor_pick=int(sum(E[arm][g].get('kind') == 'anchor' for g in gs))),
                runtime=dict(wall_s_mean=float(wall.mean()), wall_s_median=float(np.median(wall)), wall_s_total=float(wall.sum()),
                             cpu_s_mean=float(cpu.mean())))
            tr = [rs[g][arm]['terrain'] for g in gs]; t0 = [rs[g]['original']['terrain'] for g in gs]
            cl = np.array([x['climb_m'] for x in tr]); cl0 = np.array([x['climb_m'] for x in t0])
            pa[sname]['terrain'] = dict(climb_m=q(cl, (0, .5, .9, 1.0)), climb_original_m=q(cl0, (0, .5, .9, 1.0)),
                                        n_less_climb_than_original_by_0p5m=int((cl < cl0 - 0.5).sum()), n_more_climb_by_0p5m=int((cl > cl0 + 0.5).sum()),
                                        max_rise_m=q([x['max_rise_m'] for x in tr], (0, .5, .9, 1.0)),
                                        max_uphill_grade_pct=q([x['max_uphill_grade_pct'] for x in tr], (0, .5, .9, 1.0)),
                                        max_uphill_grade_original_pct=q([x['max_uphill_grade_pct'] for x in t0], (0, .5, .9, 1.0)))
            if ARMS[arm]['kind'] == 'grad':
                pa[sname]['grad'] = dict(abstained=int(sum(E[arm][g]['abstained'] for g in gs)),
                                         start_src={s: int(sum(1 for g in gs if str(E[arm][g]['start_src']) == s)) for s in sorted({str(E[arm][g]['start_src']) for g in gs})},
                                         steps_run=q([E[arm][g]['steps_run'] for g in gs], (0, .5, 1.0)),
                                         valid_finals_frac=float(np.mean([E[arm][g]['n_valid_finals'] / E[arm][g]['n_finals'] for g in gs])))
        res['per_arm'][arm] = pa
    # the original picks (as recorded) in the same format, for the table baseline
    ob = {}
    for sname, gs in sets.items():
        if not gs:
            ob[sname] = dict(n=0); continue
        P = np.array([orig[g]['P'] for g in gs]); Pp = np.array([rs[g]['original']['P_pess'] for g in gs])
        P2 = np.array([rs[g]['original']['second']['P'] for g in gs])
        ob[sname] = dict(n=len(gs), P_deciles=q(P), P_pess_deciles=q(Pp), n_P_lt_0p01=int((P < 0.01).sum()), n_P_lt_0p05=int((P < 0.05).sum()), n_P_lt_0p2=int((P < 0.2).sum()),
                         n_P_gt_0p5=int((P > 0.5).sum()), n_Ppess_gt_0p5=int((Pp > 0.5).sum()), n_disagree_mean_lt_0p05_max_gt_0p5=int(((P < 0.05) & (Pp > 0.5)).sum()),
                         second_opinion=dict(P_deciles=q(P2), n_P_lt_0p05=int((P2 < 0.05).sum()), n_P_gt_0p5=int((P2 > 0.5).sum()),
                                             n_safe_both=int(((P < 0.05) & (P2 < 0.05)).sum()), n_safe_first_but_second_gt_0p5=int(((P < 0.05) & (P2 > 0.5)).sum())))
    res['original'] = ob
    # key sentence(s)
    p50 = sets['FAIL_P50']
    for arm in ORDER:
        safe = [g for g in p50 if E[arm][g]['P'] < 0.05]
        robust = [g for g in safe if E[arm][g]['P_pess'] < 0.05]
        both = [g for g in safe if rs[g][arm]['second']['P'] < 0.05]
        res['key'][arm] = dict(n_safe=len(safe), n_safe_all_members=len(robust), n_safe_and_second_model_safe=len(both),
                               n_lt_0p01=int(sum(E[arm][g]['P'] < 0.01 for g in p50)), groups_safe=safe)
    # ablation of the wider family on the FAIL P>0.5 pairs: A2 (standard family, same budget), A4 (5 terms, 20 m), A4a (3 terms, 20 m), A4b (5 terms, 10 m)
    res['ablation'] = {}
    for arm in ['A2', 'A4'] + [x for x in ABL if all((arm_dir(x) / 'picks' / f'{g}.json').exists() for g in p50)]:
        EE = E[arm] if arm in E else {g: json.load(open(arm_dir(arm) / 'picks' / f'{g}.json'))['arms'][arm] for g in p50}
        P = np.array([EE[g]['P'] for g in p50]); Pp = np.array([EE[g]['P_pess'] for g in p50]); geo = [EE[g]['geometry'] for g in p50]
        tr = [rs[g][arm]['terrain'] for g in p50 if arm in rs[g]]
        res['ablation'][arm] = dict(words=ARMS[arm]['words'], n=len(p50), n_P_lt_0p01=int((P < 0.01).sum()), n_P_lt_0p05=int((P < 0.05).sum()),
                                    n_P_lt_0p2=int((P < 0.2).sum()), n_P_gt_0p5=int((P > 0.5).sum()), n_Ppess_lt_0p05=int((Pp < 0.05).sum()),
                                    n_disagree=int(((P < 0.05) & (Pp > 0.5)).sum()), median_P=float(np.median(P)),
                                    n_lat_gt_10m=int(sum(x['max_lateral_m'] > 10.05 for x in geo)), max_lateral_median_m=float(np.median([x['max_lateral_m'] for x in geo])),
                                    climb_median_m=(float(np.median([x['climb_m'] for x in tr])) if tr else None),
                                    wall_s_mean=float(np.mean([EE[g]['wall_s'] for g in p50])),
                                    acceptance=float(np.mean([sum(l['acceptance'] * l['tries'] for l in EE[g]['log'] if 'kind' not in l) /
                                                              max(sum(l['tries'] for l in EE[g]['log'] if 'kind' not in l), 1) for g in p50])))
    anyb = [g for g in p50 if any(E[arm][g]['P'] < 0.05 for arm in ORDER if arm != 'A0')]
    res['key']['any_bigger_arm'] = dict(n_safe=len(anyb), groups=anyb, note='best of five bigger/wider arms: a union, so it is optimistic')
    cem_only = [g for g in p50 if any(E[arm][g]['P'] < 0.05 for arm in ('A1', 'A2', 'A4'))]
    res['key']['any_sampling_arm'] = dict(n_safe=len(cem_only), note='A1, A2 or A4 (no gradient)')
    # calibration of the ORIGINAL picks on the drives (all 800 pairs): how often did a pick rated in a bin fail?
    rows = load_index_rows(); Pall = np.array([r['P'] for r in rows.values()]); Fall = np.array([r['fail'] for r in rows.values()])
    cal = []
    for lo, hi in [(0, 0.01), (0.01, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 0.9), (0.9, 1.0001)]:
        m = (Pall >= lo) & (Pall < hi)
        cal.append(dict(bin=[lo, min(hi, 1.0)], n=int(m.sum()), failed=int(Fall[m].sum()), fail_rate=float(Fall[m].mean()) if m.any() else None,
                        mean_P=float(Pall[m].mean()) if m.any() else None))
    res['original_calibration_800'] = cal
    # per stratum on FAIL_P50: how many made safe by A2 / A3 / A4
    strata = sorted({orig[g]['stratum'] for g in p50})
    res['fail_p50_by_stratum'] = {s: dict(n=sum(1 for g in p50 if orig[g]['stratum'] == s),
                                          **{arm: int(sum(1 for g in p50 if orig[g]['stratum'] == s and E[arm][g]['P'] < 0.05)) for arm in ORDER})
                                  for s in strata}
    jdump(res, OUT / 'results.json')
    write_tables(res)
    print(json.dumps(res['key'], default=_jdefault)[:2000])


def write_tables(res):
    L = []
    fmt = lambda x: f'{x:.3f}' if x >= 0.001 else (f'{x:.1e}' if x > 0 else '0')
    names = dict(original='recorded pick (driven)', **{k: k for k in ORDER})
    for sname in ('FAIL_P50', 'FAIL', 'CONTROL', 'FAIL_P_LE_50'):
        if not res['sets'][sname]:
            continue
        L.append(f'\n### {sname} (n = {res["sets"][sname]})\n')
        L.append('| search | P<0.01 | P<0.05 | P<0.2 | P>0.5 | median P | 90th pct P | worst-member P>0.5 | mean P<0.05 but worst member P>0.5 |')
        L.append('|---|---|---|---|---|---|---|---|---|')
        o = res['original'][sname]
        L.append(f"| {names['original']} | {o['n_P_lt_0p01']} | {o['n_P_lt_0p05']} | {o['n_P_lt_0p2']} | {o['n_P_gt_0p5']} | {fmt(o['P_deciles'][5])} | {fmt(o['P_deciles'][9])} | {o['n_Ppess_gt_0p5']} | {o['n_disagree_mean_lt_0p05_max_gt_0p5']} |")
        for arm in ORDER:
            d = res['per_arm'][arm][sname]
            L.append(f"| {arm} | {d['n_P_lt_0p01']} | {d['n_P_lt_0p05']} | {d['n_P_lt_0p2']} | {d['n_P_gt_0p5']} | {fmt(d['P_deciles'][5])} | {fmt(d['P_deciles'][9])} | {d['n_Ppess_gt_0p5']} | {d['n_disagree_mean_lt_0p05_max_gt_0p5']} |")
        L.append('\nP deciles (0, 10, ..., 100 %):\n')
        L.append('| search | ' + ' | '.join(f'{i * 10}%' for i in range(11)) + ' |')
        L.append('|---|' + '---|' * 11)
        L.append(f"| {names['original']} | " + ' | '.join(fmt(x) for x in o['P_deciles']) + ' |')
        for arm in ORDER:
            L.append(f'| {arm} | ' + ' | '.join(fmt(x) for x in res['per_arm'][arm][sname]['P_deciles']) + ' |')
        L.append('\nWorst-member P deciles:\n')
        L.append('| search | ' + ' | '.join(f'{i * 10}%' for i in range(11)) + ' |')
        L.append('|---|' + '---|' * 11)
        for arm in ORDER:
            L.append(f'| {arm} | ' + ' | '.join(fmt(x) for x in res['per_arm'][arm][sname]['P_pess_deciles']) + ' |')
        L.append('\nRoute geometry (median [90th pct, max]):\n')
        L.append('| search | length / straight line | length / recorded pick | route time s | time / recorded pick | max sideways offset m | at sideways limit | mean speed m/s | mean speed > 5 | >= 3 speed knots at limit | within 6 m of arena edge |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|')
        for arm in ORDER:
            gg = res['per_arm'][arm][sname]['geometry']
            f3 = lambda v: f'{v[1]:.2f} [{v[2]:.2f}, {v[3]:.2f}]'
            L.append(f"| {arm} | {f3(gg['length_over_straight'])} | {f3(gg['length_vs_original_ratio'])} | {f3(gg['T_s'])} | {f3(gg['T_vs_original_ratio'])} | "
                     f"{f3(gg['max_lateral_m'])} | {gg['n_hits_lat_clip']} | {f3(gg['mean_speed'])} | {gg['n_mean_speed_gt5']} | {gg['n_dv_clip_ge3']} | {gg['n_near_edge_34m']} |")
    L.append('\n### Terrain along the picked routes (median [90th pct, max]; from the static elevation map)\n')
    for sname in ('FAIL_P50', 'CONTROL'):
        if not res['sets'][sname]:
            continue
        L.append(f'\n{sname}:\n')
        L.append('| search | total climb m | recorded pick climb m | less climb than recorded (by > 0.5 m) | more climb | steepest 2 m uphill grade % | recorded pick steepest % |')
        L.append('|---|---|---|---|---|---|---|')
        for arm in ORDER:
            t = res['per_arm'][arm][sname]['terrain']; f3 = lambda v: f'{v[1]:.2f} [{v[2]:.2f}, {v[3]:.2f}]'
            L.append(f"| {arm} | {f3(t['climb_m'])} | {f3(t['climb_original_m'])} | {t['n_less_climb_than_original_by_0p5m']} | {t['n_more_climb_by_0p5m']} | "
                     f"{f3(t['max_uphill_grade_pct'])} | {f3(t['max_uphill_grade_original_pct'])} |")
    if res.get('ablation'):
        L.append('\n### Which part of the wider family matters (FAIL P>0.5 pairs, n = %d, 16 x 512 search each)\n' % res['sets']['FAIL_P50'])
        L.append('| search | bend terms | sideways limit | P<0.01 | P<0.05 | P<0.2 | P>0.5 | worst member P<0.05 | routes > 10 m sideways | median climb m | prior acceptance | s per pair |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for arm, d in res['ablation'].items():
            sp = ARMS[arm]; m = sp.get('modes', 3); c = sp.get('lat_clip', 10.0)
            L.append(f"| {arm} | {m} | {c:g} m | {d['n_P_lt_0p01']} | {d['n_P_lt_0p05']} | {d['n_P_lt_0p2']} | {d['n_P_gt_0p5']} | {d['n_Ppess_lt_0p05']} | {d['n_lat_gt_10m']} | "
                     f"{d['climb_median_m']:.2f} | {d['acceptance']:.2f} | {d['wall_s_mean']:.0f} |")
    L.append('\n### Planning time per start/goal pair (all 361 pairs, this workstation, shared with other jobs)\n')
    L.append('| search | mean s | median s | total min |')
    L.append('|---|---|---|---|')
    for arm in ORDER:
        if not (res['sets']['FAIL'] and res['sets']['CONTROL']):
            break
        rr = [res['per_arm'][arm][s]['runtime'] for s in ('FAIL', 'CONTROL')]
        n = [res['per_arm'][arm][s]['n'] for s in ('FAIL', 'CONTROL')]
        tot = sum(r['wall_s_total'] for r in rr); mean = tot / sum(n)
        L.append(f"| {arm} | {mean:.1f} | {res['per_arm'][arm]['FAIL']['runtime']['wall_s_median']:.1f} (fail set) | {tot / 60:.0f} |")
    (OUT / 'tables.md').write_text('\n'.join(L) + '\n')


# ------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('groups')
    p = sub.add_parser('selftest'); p.add_argument('--device')
    p = sub.add_parser('plan'); p.add_argument('--arms', required=True); p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--only', help='comma list of groups (default: groups/all.txt)'); p.add_argument('--root', help='output root (default the probe root)')
    p.add_argument('--force', action='store_true'); p.add_argument('--device')
    p.add_argument('--order', choices=['md5', 'priority'], default='md5', help='planning order (results do not depend on it)')
    p = sub.add_parser('rerun'); p.add_argument('--n', type=int, default=4); p.add_argument('--arms')
    p = sub.add_parser('finalize'); p.add_argument('--arms')
    p = sub.add_parser('rescore'); p.add_argument('--device')
    sub.add_parser('report')
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    dict(groups=cmd_groups, selftest=cmd_selftest, plan=cmd_plan, rerun=cmd_rerun, finalize=cmd_finalize, rescore=cmd_rescore, report=cmd_report)[a.cmd](a)


if __name__ == '__main__':
    main()
