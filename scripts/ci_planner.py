"""Route planner with a speed- and heading-continuous candidate family at a moving branch (PLAN crm_improve S1).

Wraps scripts/ga_planner.py (not edited): the same loaders, the same Scorer (history z encoded once per decision), the
same --poses decision states, the same arms (A = one-shot 256, B = CEM 4x64, tags and seeds from planner_arms.arm_specs)
and the same output files (picks/<g>.json, routes/<g>__<arm>.json, tasks.json, tasks_new_only.json, summary.json,
PICKS_LOCKED.sha256[, tasks_cluster.json]). What changes is the candidate family (--family):

  free       the current night-2 family, untouched. Reproduces ga_planner.py's picks bit for bit (self-test T1).
  cont       every candidate (the 9 designed anchors, every prior draw, every CEM draw and every CEM mean) gets its
             speed profile replaced by
                 v(s) = min( max(v_sampled(s), sqrt(max(v0^2 - 2 A_DEC s, 0))), sqrt(v0^2 + 2 A_ACC s), V_MAX )
             (s = station from the route start, A_ACC 1.5, A_DEC 2.0, V_MAX 6.0 = f104_n2_sampler), so the route starts
             exactly at the vehicle speed v0 (min(v0, 6)), never asks for more than the 1.5 m/s^2 acceleration ramp from
             v0 and never for less than the 2 m/s^2 deceleration ramp from v0. The element-wise max / min of profiles
             that each respect the discrete accel/decel limits respects them too, and the sampled profile already ends
             in its terminal deceleration cone, so the route still stops at the goal. The transformed candidate is then
             re-validated with the planner's validator (gen_planner.safe_validate anchored at the pose); a candidate
             that fails is dropped and the draw loop continues exactly as f104_n2_iter.plan_iter already does.
  cont_head  cont plus rejection of candidates whose first tangent differs from the vehicle yaw by more than
             --max-head-deg (default 20): |wrap(headings[0] - yaw)| (gc_control.start_heading_err_deg). The anchors'
             lateral offset off * sin^2(pi f) has zero slope at the start, so they always pass this test.

v0 (the vehicle forward speed at the decision, state column 0 = vel_body_x), in this order:
  1. --v0 (single group) or a 'v0' key in the --poses entry                                   -> 'explicit'
  2. the history window given by the entry's / --history npz: its last valid row, column 0     -> 'history_window'
     (for the K1 A5 poses file this is the frame-60 state; equal to pass1_state.json 'vx' on all 800 CRM groups)
  3. a recorded run (entry 'run' / --from-run): state[frame, 0], or terminal_state[0] when frame == len(state)
                                                                                                -> 'trajectory'
  4. the entry's 'pass1_run' (the masked-history poses files carry only this): as 3            -> 'pass1_run'
  5. the case layout pose without any override: 0.0                                           -> 'standing_start'
  anything else (a pose override with no speed) is an error. Negative v0 is clipped to 0. --v0-min (default 0) floors
  the v0 used by the family (see the standing-start caveat below); the recorded v0 is kept separately.

Standing-start caveat: with v0 = 0 the cont routes start at 0 m/s, and the frozen follower commands the speed of the
nearest waypoint (traverse_fdm_rgbd_diverse_chrono.py:218), so a vehicle at rest on waypoint 0 is told 0 m/s and would
not move. The planner runs (self-test T4), but a standing-start drive needs --family free or --v0-min > 0.

Python API (for the data-collection stage; torch is imported by ga_planner at import time, so do not import this
module inside a Chrono collector - build the routes offline and ship them):
  candidates(pose, goal, v0, n, seed, family='cont_head', max_head_deg=20.0) -> list of n validated routes in the
      collector format {waypoints (k,2), speeds, stations, headings: numpy arrays, meta: dict}; prior draws of the
      planner family from base_route(pose, goal) with rng default_rng(seed) (anchors=True puts the valid designed
      anchors first, as the planner's round 0 does). gc_control.route_to_json writes them.
  plan_decision(ens, pose, goal, v0, hist, hmask, group, arm='B', family='cont_head', ...) -> ga_planner-style pick
      (route, z_mean, ...) for one decision state, seeds IT.seed(group, arm tag) as the CLI.

Checkpoints: legacy N2 and model_kind 'ga_train' through ga_planner.Ensemble unchanged; model_kind 'ci_train' through
ci_train.load_ci_model / ci_train.score when scripts/ci_train.py exists (see CIEnsemble / CIScorer below).

  PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
  K1=artifacts/traverse/generalist_20260921/A_adapt
  $PY scripts/ci_planner.py --family cont_head --cases $K1/suite/cases --map-root artifacts/traverse/crm_f104_v1/map_root \
      --models "$K1/train/deploy_v1/H_deploy_s*.pt" --world crm --domain crm --arms B --poses $K1/a5/poses_crm.json \
      --verify 0 --task-root artifacts/traverse/crm_improve_20260922 --out artifacts/traverse/crm_improve_20260922/s1/picks_crm_H_conthead
  $PY scripts/ci_planner.py --selftest artifacts/traverse/crm_improve_20260922/s1/selftest     # T1-T4
"""
import argparse, contextlib, glob, hashlib, importlib, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ga_planner as GA
IT, GP, DS, S, PA = GA.IT, GA.GP, GA.DS, GA.S, GA.PA
torch = GA.torch
ROOT = GA.ROOT

FAMILIES = ('free', 'cont', 'cont_head')
MAX_HEAD_DEG = 20.0
A_ACC, A_DEC, V_MAX = float(S.A_ACC), float(S.A_DEC), float(S.V_MAX)
assert (A_ACC, A_DEC, V_MAX) == (1.5, 2.0, 6.0), (A_ACC, A_DEC, V_MAX)
assert (GP.CFG.max_accel_mps2, GP.CFG.max_decel_mps2, GP.CFG.max_speed_mps) == (A_ACC, A_DEC, V_MAX)
K1 = ROOT / 'artifacts/traverse/generalist_20260921/A_adapt'


# ------------------------------------------------------------------------------------------------------------------
# the family transform
# ------------------------------------------------------------------------------------------------------------------
def wrap_pi(a):
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def start_heading_err_deg(route, yaw):
    """|wrap(headings[0] - yaw)| in degrees (gc_control.start_heading_err_deg)."""
    return abs(math.degrees(wrap_pi(float(np.asarray(route['headings'], float)[0]) - float(yaw))))


def ramps(stations, v0, a_acc=A_ACC, a_dec=A_DEC, v_max=V_MAX):
    """(deceleration ramp, acceleration ramp) from v0 along the route: sqrt(max(v0^2 - 2 a_dec s, 0)), sqrt(v0^2 + 2 a_acc s)."""
    st = np.asarray(stations, float); s = st - st[0]; v0 = max(float(v0), 0.0)
    return np.sqrt(np.maximum(v0 * v0 - 2.0 * a_dec * s, 0.0)), np.sqrt(v0 * v0 + 2.0 * a_acc * s)


def cont_speeds(speeds, stations, v0, a_acc=A_ACC, a_dec=A_DEC, v_max=V_MAX):
    """min(max(sampled, deceleration ramp from v0), acceleration ramp from v0, V_MAX); starts at min(v0, V_MAX)."""
    sp = np.asarray(speeds, float)
    floor, cap = ramps(stations, v0, a_acc, a_dec, v_max)
    return np.minimum(np.minimum(np.maximum(sp, floor), cap), float(v_max))


def accel_profile(route):
    """Along-route acceleration diff(v^2) / (2 ds) (the validator's expression)."""
    v = np.asarray(route['speeds'], float); st = np.asarray(route['stations'], float)
    return np.diff(v ** 2) / (2.0 * np.maximum(np.diff(st), 1e-8))


class Family:
    """The candidate family of one decision: apply() transforms a built candidate, ok() is the acceptance test."""
    def __init__(self, name, v0=0.0, yaw=0.0, max_head_deg=MAX_HEAD_DEG, v0_min=0.0):
        assert name in FAMILIES, name
        self.name, self.yaw, self.max_head_deg = name, float(yaw), float(max_head_deg)
        self.v0_rec = None if v0 is None else float(v0)
        self.v0 = max(float(v0 or 0.0), float(v0_min), 0.0)          # the speed the ramps start from
        self.v0_min = float(v0_min)
        self.reset()

    def reset(self):
        self.stats = dict(built=0, rejected_validator=0, rejected_heading=0, accepted=0, speed_points_raised=0,
                          speed_points_lowered=0)

    def describe(self):
        return dict(family=self.name, v0_mps=self.v0_rec, v0_used_mps=self.v0, v0_min_mps=self.v0_min,
                    yaw=self.yaw, max_head_deg=self.max_head_deg if self.name == 'cont_head' else None)

    def apply(self, r):
        self.stats['built'] += 1
        if self.name == 'free':
            return r
        sp = np.asarray(r['speeds'], float)
        v = cont_speeds(sp, r['stations'], self.v0)
        self.stats['speed_points_raised'] += int((v > sp + 1e-12).sum())
        self.stats['speed_points_lowered'] += int((v < sp - 1e-12).sum())
        out = {k: r[k] for k in ('waypoints', 'stations', 'headings')}
        out['speeds'] = v
        out['meta'] = {**r.get('meta', {}), 'mean_speed_mps': float(v[1:-1].mean()), 'family': self.name,
                       'v0_mps': self.v0_rec, 'v0_used_mps': self.v0, 'start_speed_mps': float(v[0]),
                       'start_heading_err_deg': start_heading_err_deg(r, self.yaw),
                       'max_start_heading_err_deg': self.max_head_deg if self.name == 'cont_head' else None}
        return out

    def ok(self, r, pose):
        if self.name == 'cont_head' and start_heading_err_deg(r, self.yaw) > self.max_head_deg:
            self.stats['rejected_heading'] += 1
            return False
        if not IT.valid(r, pose):
            self.stats['rejected_validator'] += 1
            return False
        self.stats['accepted'] += 1
        return True


# ------------------------------------------------------------------------------------------------------------------
# f104_n2_iter.plan_iter with family hooks (verbatim otherwise; with family 'free' the picks are bit-identical)
# ------------------------------------------------------------------------------------------------------------------
def sample_valid_f(fam, base, pose, rng, need, max_tries, L, fixed_speed, mu=None, sd=None):
    out, th, tries = [], [], 0
    while len(out) < need and tries < max_tries:
        tries += 1
        t = IT.draw_prior(rng, L, fixed_speed, mu, sd)
        r = fam.apply(IT.from_params(base, t, fixed_speed))
        if fam.ok(r, pose):
            out.append(r); th.append(t)
    return out, th, tries


def plan_iter_family(base, pose, goal, scorer, fam=None, rounds=4, n=64, elite_frac=0.15, sd_floor=0.15, weighting='cem',
                     ess_frac=0.25, objective='mean', c_fail=60.0, rng=None, anchors=True, fixed_speed=None, tries_factor=8,
                     mean_in_pop=True, score_mean=None, keep_candidates=False):
    """IT.plan_iter with every built candidate passed through fam.apply and every validity test through fam.ok.
    keep_candidates=True also returns every evaluated candidate (self-tests)."""
    fam = Family('free') if fam is None else fam
    rng = np.random.default_rng() if rng is None else rng
    if score_mean is None:
        score_mean = rounds > 1
    dim = IT.MODES if fixed_speed is not None else IT.MODES + IT.KNOTS
    prior_sd = IT.PRIOR_SD[:dim]
    mu, sd = np.zeros(dim), prior_sd.copy()
    xy, station, f, L = IT._base_arrays(base)
    allc, allth, allz, allzp, allT, kinds, rnd, log = [], [], [], [], [], [], [], []
    tries_total = 0
    anc = []
    if anchors:
        sp = (2.0, 4.0, 6.0) if fixed_speed is None else (float(fixed_speed),)
        anc = [r for r in (fam.apply(r0) for r0 in S.anchors(base, speeds=sp)) if fam.ok(r, pose)]

    def evaluate(cs, ths, ks, k):
        Z, zm, zp = scorer(cs)
        T = np.array([IT.route_time(r) for r in cs])
        allc.extend(cs); allth.extend(ths); kinds.extend(ks); rnd.extend([k] * len(cs))
        allz.append(zm); allzp.append(zp); allT.append(T)
        return zm, zp, T

    for k in range(rounds):
        cs, ths, ks = [], [], []
        if k == 0:
            cs += anc; ths += [None] * len(anc); ks += ['anchor'] * len(anc)
        elif mean_in_pop:
            rm = fam.apply(IT.from_params(base, mu, fixed_speed))
            if fam.ok(rm, pose):
                cs.append(rm); ths.append(np.asarray(rm['meta']['theta'])); ks.append('mean')
        need = n - len(cs)
        drawn, dth, tries = sample_valid_f(fam, base, pose, rng, need, tries_factor * n, L, fixed_speed,
                                           None if k == 0 else mu, None if k == 0 else sd)
        tries_total += tries
        cs += drawn; ths += dth; ks += ['sample'] * len(drawn)
        if not cs:
            log.append(dict(round=k, n=0, tries=tries, acceptance=0.0)); break
        zm, zp, T = evaluate(cs, ths, ks, k)
        Jall = IT.objective_values(np.concatenate(allz), np.concatenate(allzp), np.concatenate(allT), objective, c_fail)
        has = np.array([t is not None for t in allth])
        entry = dict(round=k, n=len(cs), tries=tries, acceptance=len(drawn) / max(tries, 1),
                     zmin_round=float(zm.min()), zmin=float(np.concatenate(allz).min()),
                     jmin=float(Jall.min()), mu=mu.round(3).tolist(), sd=sd.round(3).tolist())
        if k < rounds - 1 or score_mean:
            Th = np.stack([t for t in allth if t is not None]) if has.any() else np.zeros((0, dim))
            Jh = Jall[has]
            if len(Th):
                if weighting == 'cem':
                    ne = min(len(Th), max(4, math.ceil(elite_frac * n)))
                    order = np.argsort(Jh, kind='stable')[:ne]
                    E = Th[order]
                    mu = E.mean(0); sd = np.maximum(E.std(0), sd_floor * prior_sd)
                    entry.update(n_elite=int(ne), ess=float(ne), j_elite_max=float(Jh[order[-1]]))
                elif weighting == 'mppi':
                    w, ess, temp = IT.ess_weights(Jh, ess_frac)
                    mu = (w[:, None] * Th).sum(0)
                    sd = np.maximum(np.sqrt((w[:, None] * (Th - mu) ** 2).sum(0)), sd_floor * prior_sd)
                    entry.update(ess=ess, temperature=temp)
                else:
                    raise ValueError(weighting)
                mu = IT.project(mu, L, fixed_speed)
            entry.update(mu_next=mu.round(3).tolist(), sd_next=sd.round(3).tolist())
        log.append(entry)
    if score_mean and allc:
        rm = fam.apply(IT.from_params(base, mu, fixed_speed))
        if fam.ok(rm, pose):
            evaluate([rm], [np.asarray(rm['meta']['theta'])], ['mean'], rounds)
            log.append(dict(round=rounds, n=1, tries=0, acceptance=1.0, kind='final_mean', zmin_round=float(allz[-1][0])))
        else:
            log.append(dict(round=rounds, n=0, tries=0, acceptance=0.0, kind='final_mean_invalid'))
    if not allc:
        return None
    Zm = np.concatenate(allz); Zp = np.concatenate(allzp); T = np.concatenate(allT)
    J = IT.objective_values(Zm, Zp, T, objective, c_fail)
    i = int(np.argmin(J))          # ties -> first index (anchors first), as np.argmin in the deployed planner
    r = allc[i]
    r = {'waypoints': np.asarray(r['waypoints']), 'speeds': np.asarray(r['speeds']), 'stations': np.asarray(r['stations']),
         'headings': np.asarray(r['headings']),
         'meta': {**r.get('meta', {}), 'candidate': 'n2_iter' if rounds > 1 else r.get('meta', {}).get('candidate', 'n2_wide'),
                  'kind': kinds[i], 'round': int(rnd[i]), 'rank': int(i - sum(1 for q in rnd[:i] if q != rnd[i])),
                  'theta': None if allth[i] is None else np.asarray(allth[i]).tolist()}}
    out = dict(route=r, index=i, kind=kinds[i], round=int(rnd[i]), theta=r['meta']['theta'],
               z_mean=float(Zm[i]), z_pess=float(Zp[i]), P=float(1 - np.exp(-np.exp(Zm[i]))), T=float(T[i]), J=float(J[i]),
               n_evaluated=len(allc), tries=tries_total, log=log, Z_mean=Zm, Z_pess=Zp, T_all=T, kinds=kinds,
               mu=mu.tolist(), sd=sd.tolist(), objective=objective, family=fam.describe(), family_stats=dict(fam.stats))
    if keep_candidates:
        out['candidates'] = allc
    return out


def oneshot_family(base, pose, goal, scorer, rng, fam=None, n=256, fixed_speed=None, objective='mean', c_fail=60.0,
                   keep_candidates=False):
    """IT.oneshot through plan_iter_family (same arguments as the deployed one-shot)."""
    if fixed_speed is None:
        return plan_iter_family(base, pose, goal, scorer, fam, rounds=1, n=n, rng=rng, anchors=True, tries_factor=8,
                                objective=objective, c_fail=c_fail, score_mean=False, keep_candidates=keep_candidates)
    return plan_iter_family(base, pose, goal, scorer, fam, rounds=1, n=n, rng=rng, anchors=False, fixed_speed=fixed_speed,
                            tries_factor=6, objective=objective, c_fail=c_fail, score_mean=False, keep_candidates=keep_candidates)


# ------------------------------------------------------------------------------------------------------------------
# v0 of a decision
# ------------------------------------------------------------------------------------------------------------------
def trajectory_vx(run_dir, frame):
    """Vehicle forward speed at `frame` of a recorded run: state[frame, 0], or terminal_state[0] when frame == len(state)
    (the pass-1 approach drives stop after 60 frames and store the frame-60 state as terminal_state)."""
    t = np.load(Path(run_dir) / 'trajectory.npz')
    st = np.asarray(t['state']); frame = int(frame)
    if frame < len(st):
        return float(st[frame, 0])
    if frame == len(st) and 'terminal_state' in t.files:
        return float(np.asarray(t['terminal_state']).reshape(-1)[0])
    raise ValueError(f'{run_dir}: frame {frame} outside the recording ({len(st)} rows)')


def decision_v0(entry, hist, hmask, src, v0_cli=None, frame_default=60):
    """(v0, source, cross_check) for one decision, rules 1-5 of the module docstring."""
    entry = entry or {}
    check = {}
    if v0_cli is not None:
        return max(float(v0_cli), 0.0), 'explicit', check
    if entry.get('v0') is not None:
        return max(float(entry['v0']), 0.0), 'explicit', check
    frame = int(entry.get('frame', frame_default))
    pass1 = entry.get('pass1_run')
    hsrc = str(src.get('history', ''))
    if ':trajectory@' in hsrc:                  # --history <trajectory.npz> cut at --frame
        path, k = hsrc.rsplit(':trajectory@', 1)
        return max(trajectory_vx(Path(path).parent, int(k)), 0.0), 'trajectory', check
    if hist is not None and hmask is not None and np.asarray(hmask, bool).any() and hsrc.endswith(':npz'):
        m = np.asarray(hmask, bool)
        v0 = float(np.asarray(hist, np.float32)[np.flatnonzero(m)[-1], 0])
        if pass1:
            check['pass1_run_vx'] = trajectory_vx(pass1, frame); check['abs_diff'] = abs(check['pass1_run_vx'] - v0)
        return max(v0, 0.0), 'history_window', check
    run = src.get('run') or entry.get('run')
    if run:
        return max(trajectory_vx(run, src.get('frame', frame)), 0.0), 'trajectory', check
    if pass1:
        return max(trajectory_vx(pass1, frame), 0.0), 'pass1_run', check
    if src.get('pose') == 'layout' and src.get('goal') == 'case':
        return 0.0, 'standing_start', check
    raise ValueError(f"no vehicle speed for a decision with pose source {src.get('pose')!r}: pass --v0 or a 'v0' key in the poses entry")


# ------------------------------------------------------------------------------------------------------------------
# ci_train checkpoints (scripts/ci_train.py, written by another stage)
# ------------------------------------------------------------------------------------------------------------------
def _ci_train_module():
    """scripts/ci_train.py if it exists (another stage writes it), else None -> NotImplementedError at load time."""
    if importlib.util.find_spec('ci_train') is None:
        return None
    return importlib.import_module('ci_train')


def _kind_of(path):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    return (ck.get('model_kind') or 'legacy') if isinstance(ck, dict) else 'legacy'


class CIEnsemble(GA.Ensemble):
    """ga_planner.Ensemble, plus members of model_kind 'ci_train' loaded with ci_train.load_ci_model(path, device) ->
    (model, ck). With no 'ci_train' member this IS ga_planner.Ensemble (the parent constructor runs unchanged); mixing
    ci_train with other kinds is refused. hist_T of the ensemble = the STORED window length (ck hist_T_stored, 40): the
    planner loads 40-frame windows as before and ci_train.score / encode_history cut each member's newest hist_T frames."""
    def __init__(self, pattern, device=None):
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(pattern)
        kinds = [_kind_of(p) for p in paths]
        self.has_ci = 'ci_train' in kinds
        if not self.has_ci:
            super().__init__(pattern, device); return
        CT = _ci_train_module()
        if CT is None or not all(hasattr(CT, f) for f in ('load_ci_model', 'score', 'encode_history', 'vel_from_history')):
            raise NotImplementedError("model_kind 'ci_train' needs scripts/ci_train.py with load_ci_model(), encode_history(), "
                                      "vel_from_history() and score(); the module is absent or incomplete")
        self.CT = CT
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.members = []
        for p, kind in zip(paths, kinds):
            if kind != 'ci_train':
                raise NotImplementedError(f'mixed ensembles (ci_train with {kind}) are not supported: {p}')
            m, ck = CT.load_ci_model(p, device=self.dev)
            if int(ck['cin']) != 6 or list(np.asarray(ck['geom_cols']).ravel()) != [17, 18, 19, 20, 21]:
                raise ValueError(f"{p}: cin {ck['cin']} / geom_cols {ck['geom_cols']} differ from the planner corridor + geom5 contract")
            # zdim 0 in the member dict: ga_planner.main's encode-count assertion only counts its own ga_train members;
            # the ci_train encodes are counted by CIScorer (history['n_encode']) instead.
            self.members.append(dict(path=p, name=os.path.basename(p), kind='ci_train', cond=ck['cond'], model=m, ck=ck, zdim=0,
                                     ci_zdim=int(ck.get('zdim') or 0), use_hist=bool(m.use_hist), ctx_mode=ck.get('ctx_mode', 'geom'),
                                     arch=ck.get('arch'), hist_enc=ck.get('hist_enc'), hist_T_member=int(ck['hist_T']), warnings=[]))
        self.kinds = ['ci_train']; self.conds = sorted({m['cond'] for m in self.members})
        Ts = {int(m['ck'].get('hist_T_stored', m['ck']['hist_T'])) for m in self.members}
        assert len(Ts) == 1, f'members disagree on the stored history window length: {Ts}'
        self.hist_T = max(Ts.pop(), GA.HIST_T)
        self.needs_hist = any(m['use_hist'] or m['ctx_mode'] == 'geom_vel' for m in self.members)
        self.needs_tag = any(m['cond'] == 'tag' for m in self.members)

    def describe(self):
        if not getattr(self, 'has_ci', False):
            return super().describe()
        return [dict(name=m['name'], kind=m['kind'], arch=m['arch'], hist_enc=m['hist_enc'], cond=m['cond'], ctx_mode=m['ctx_mode'],
                     zdim=m['ci_zdim'], hist_T=m['hist_T_member'], nctx=int(m['ck']['nctx']),
                     params=int(sum(p.numel() for p in m['model'].parameters()))) for m in self.members]


class CIScorer(GA.Scorer):
    """ga_planner.Scorer. For a ci_train ensemble: the same corridors (IT.corridors, float16-rounded) and geom5
    (gen_planner.geom_ctx); per member the history context is encoded ONCE here with ci_train.encode_history (None for
    members without history), the velocity context [vx, yaw rate] of the newest valid frame is taken once with
    ci_train.vel_from_history (geom_vel members; zero for an empty window), the tag one-hot from --domain, and every
    candidate batch is scored with ci_train.score(model, ck, X, geom5, z=..., vel=..., domain_onehot=...)."""
    def __init__(self, ens, start_xy, goal_xy, start_yaw, domain=None, hist=None, hmask=None, float16=True):
        if not getattr(ens, 'has_ci', False):
            super().__init__(ens, start_xy, goal_xy, start_yaw, domain, hist, hmask, float16); return
        self.ens, self.start_xy, self.goal_xy, self.yaw, self.f16 = ens, start_xy, goal_xy, start_yaw, float16
        CT = ens.CT; T = ens.hist_T
        if hist is None:
            hist, hmask = np.zeros((T, GA.HIST_DIM), np.float32), np.zeros(T, bool)
        self.hist, self.hmask = np.asarray(hist, np.float32), np.asarray(hmask, bool)
        assert self.hist.shape == (T, GA.HIST_DIM) and self.hmask.shape == (T,), (self.hist.shape, self.hmask.shape, T)
        self.tag = None
        if ens.needs_tag:
            assert domain in GA.DOMAIN_CODE, f'--domain rigid|crm is required for a tag-conditioned member (got {domain!r})'
            self.tag = np.zeros((1, 2), np.float32); self.tag[0, GA.DOMAIN_CODE[domain]] = 1.0
        self.vel = CT.vel_from_history(self.hist, self.hmask)                     # (1, 2) raw [vx, yaw rate]
        self.z, self.history = {}, dict(T=T, n_valid=int(self.hmask.sum()), members={}, vel=self.vel[0].tolist())
        for i, mem in enumerate(ens.members):
            if not mem['use_hist']:
                continue
            z = CT.encode_history(mem['model'], mem['ck'], self.hist, self.hmask, device=ens.dev)
            self.z[i] = z
            d = {}
            if isinstance(z, np.ndarray):
                d['z_norm'] = float(np.linalg.norm(z))
                if mem['cond'] == 'hist_aux' and hasattr(mem['model'], 'dom'):
                    with torch.no_grad():
                        d['p_crm'] = float(torch.sigmoid(mem['model'].dom(torch.as_tensor(z, device=ens.dev)))[0, 0])
            self.history['members'][mem['name']] = d
        self.history['n_encode'] = len(self.z)
        self.calls = 0

    def member_logits(self, X, geom5, bs=256):
        if not getattr(self.ens, 'has_ci', False):
            return super().member_logits(X, geom5, bs)
        CT = self.ens.CT; zs = []
        for i, mem in enumerate(self.ens.members):
            vel = self.vel if mem['ctx_mode'] == 'geom_vel' else None
            zs.append(np.asarray(CT.score(mem['model'], mem['ck'], X, geom5, vel=vel, domain_onehot=self.tag if mem['cond'] == 'tag' else None,
                                          z=self.z.get(i)), np.float64).reshape(-1))
        return np.stack(zs)


# ------------------------------------------------------------------------------------------------------------------
# wrapping ga_planner.main
# ------------------------------------------------------------------------------------------------------------------
CTX = dict(active=False)


def _route_json_family(orig):
    def route_json(r, rid, g, arm, arms_label, world):
        out = orig(r, rid, g, arm, arms_label, world)
        m = r.get('meta', {})
        if 'family' in m:                   # never true for the free family: its route files stay byte-identical
            out['meta'].update({k: m.get(k) for k in ('family', 'v0_mps', 'v0_used_mps', 'start_speed_mps',
                                                      'start_heading_err_deg', 'max_start_heading_err_deg')})
        return out
    return route_json


@contextlib.contextmanager
def wrapped(opts, poses):
    """Patch ga_planner / f104_n2_iter / planner_arms module attributes for one CLI run; restored on exit."""
    saved = dict(decision_for=GA.decision_for, Ensemble=GA.Ensemble, Scorer=GA.Scorer, plan_iter=IT.plan_iter,
                 route_json=PA.route_json)
    orig_decision = GA.decision_for
    CTX.clear(); CTX.update(active=True, groups={}, opts=opts)

    def decision_for(g, cp, a, poses_, T):
        res = orig_decision(g, cp, a, poses_, T)
        case, lay, pose, goal, base, hist, hmask, src = res
        entry = (poses_ or {}).get(g, {})
        if opts.v0 is not None and CTX['groups']:
            raise ValueError('--v0 applies to ONE decision (use --groups with one group, or v0 keys in --poses)')
        v0, how, check = decision_v0(entry, hist, hmask, src, opts.v0, a.frame)
        fam = Family(opts.family, v0, float(pose[2]), opts.max_head_deg, opts.v0_min)
        CTX['current'] = dict(group=g, fam=fam, pose=np.asarray(pose, float), calls=[])
        CTX['groups'][g] = dict(v0_mps=v0, v0_source=how, v0_check=check, yaw=float(pose[2]), family=fam.describe(),
                                base_start_heading_err_deg=start_heading_err_deg(base, pose[2]), calls=CTX['current']['calls'])
        return res

    def plan_iter(base, pose, goal, scorer, **kw):
        fam = CTX['current']['fam']; fam.reset()
        res = plan_iter_family(base, pose, goal, scorer, fam, **kw)
        CTX['current']['calls'].append(dict(stats=dict(fam.stats), rounds=kw.get('rounds', 4), n=kw.get('n', 64)))
        return res

    class _Scorer(CIScorer):
        pass

    GA.decision_for, GA.Ensemble, GA.Scorer, IT.plan_iter = decision_for, CIEnsemble, _Scorer, plan_iter
    PA.route_json = _route_json_family(saved['route_json'])
    try:
        yield CTX
    finally:
        GA.decision_for, GA.Ensemble, GA.Scorer, IT.plan_iter = saved['decision_for'], saved['Ensemble'], saved['Scorer'], saved['plan_iter']
        PA.route_json = saved['route_json']; CTX['active'] = False


def pick_stats(route, v0, yaw):
    v = np.asarray(route['speeds'], float); acc = accel_profile(route)
    return dict(start_speed_mps=float(v[0]), speed_step_mps=float(v[0] - v0), start_heading_err_deg=start_heading_err_deg(route, yaw),
                max_accel_mps2=float(acc.max(initial=0.0)), min_accel_mps2=float(acc.min(initial=0.0)),
                mean_speed_mps=float(v[1:-1].mean()))


STEP_BINS = [(-np.inf, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, np.inf)]
HEAD_BINS = [(0, 5), (5, 15), (15, 30), (30, np.inf)]


def binned(x, bins):
    x = np.asarray(x, float)
    return {f'{lo:g}..{hi:g}': int(((x >= lo) & (x < hi)).sum()) for lo, hi in bins}


def postprocess(out, arms, opts):
    """Add the family block to every picks/<g>.json and to summary.json (after ga_planner.main wrote them)."""
    out = Path(out); rows = {}
    for g, d in CTX['groups'].items():
        p = out / 'picks' / f'{g}.json'
        if not p.exists():
            continue
        pk = json.load(open(p))
        fam = dict(d); calls = fam.pop('calls')
        fam['arms'] = {}
        for arm, call in zip(arms, calls):
            e = pk['arms'].get(arm)
            if e is None:
                fam['arms'][arm] = None; continue
            r = json.load(open(out / 'routes' / f"{e['route_id']}.json"))
            fam['arms'][arm] = dict(pick_stats(r, d['v0_mps'], d['yaw']), stats=call['stats'])
        pk['family'] = fam
        json.dump(pk, open(p, 'w'), indent=1)
        rows[g] = fam
    sp = out / 'summary.json'
    sm = json.load(open(sp))
    agg = dict(family=opts.family, max_head_deg=opts.max_head_deg if opts.family == 'cont_head' else None, v0_min_mps=opts.v0_min,
               v0_sources={s: int(sum(1 for f in rows.values() if f['v0_source'] == s)) for s in sorted({f['v0_source'] for f in rows.values()})},
               v0_crosscheck_max_abs_diff=max([f['v0_check'].get('abs_diff', 0.0) for f in rows.values()] or [0.0]), per_arm={})
    for arm in arms:
        ps = [f['arms'][arm] for f in rows.values() if f['arms'].get(arm)]
        if not ps:
            continue
        step = np.array([q['speed_step_mps'] for q in ps]); head = np.array([q['start_heading_err_deg'] for q in ps])
        st = {k: int(sum(q['stats'][k] for q in ps)) for k in ps[0]['stats']}
        agg['per_arm'][arm] = dict(n=len(ps), speed_step_mean=float(step.mean()), speed_step_abs_max=float(np.abs(step).max()),
                                   speed_step_bins=binned(step, STEP_BINS), start_heading_mean_deg=float(head.mean()),
                                   start_heading_max_deg=float(head.max()), start_heading_bins=binned(head, HEAD_BINS),
                                   max_accel_mps2=float(max(q['max_accel_mps2'] for q in ps)),
                                   min_accel_mps2=float(min(q['min_accel_mps2'] for q in ps)), stats_total=st,
                                   acceptance=st['accepted'] / max(st['built'], 1))
    sm['family'] = agg
    json.dump(sm, open(sp, 'w'), indent=1)
    return agg


def cli(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--family', choices=FAMILIES, default='free')
    ap.add_argument('--max-head-deg', type=float, default=MAX_HEAD_DEG)
    ap.add_argument('--v0', type=float, help='vehicle speed at the decision (single group; overrides the rules)')
    ap.add_argument('--v0-min', type=float, default=0.0, help='floor of the v0 used by the cont ramps (standing starts)')
    ap.add_argument('--selftest', help='run self-tests T1-T5 into this directory and exit')
    ap.add_argument('--selftest-ci', help='run only self-test T5 (ci_train checkpoints) into this directory and exit')
    ap.add_argument('-h', '--help', action='store_true')
    opts, rest = ap.parse_known_args(argv)
    if opts.help:
        print(__doc__); print('ci_planner options: --family free|cont|cont_head, --max-head-deg (20), --v0, --v0-min (0), --selftest DIR\n'
                              'every other option is ga_planner.py\'s:'); sys.argv = [sys.argv[0], '--help']; GA.main(); return
    if opts.selftest:
        return selftest(Path(opts.selftest))
    if opts.selftest_ci:
        return selftest_ci(Path(opts.selftest_ci))
    peek = argparse.ArgumentParser(add_help=False)
    for k in ('--out', '--arms', '--poses'):
        peek.add_argument(k)
    pk, _ = peek.parse_known_args(rest)
    arms = [s.strip() for s in (pk.arms or 'A,B').split(',') if s.strip()]
    poses = json.load(open(pk.poses)) if pk.poses else None
    assert opts.v0 is None or not poses or len(poses) == 1, '--v0 applies to ONE decision; put a v0 key in each --poses entry instead'
    argv0 = sys.argv
    print(f'ci_planner: family {opts.family}' + (f' max start heading {opts.max_head_deg:g} deg' if opts.family == 'cont_head' else '')
          + (f' v0 floor {opts.v0_min:g}' if opts.v0_min else ''), flush=True)
    with wrapped(opts, poses):
        sys.argv = [argv0[0]] + list(rest)
        try:
            GA.main()
        finally:
            sys.argv = argv0
        agg = postprocess(pk.out, arms, opts)
    for arm, d in agg['per_arm'].items():
        print(f"family {opts.family} arm {arm}: n {d['n']} speed step mean {d['speed_step_mean']:+.3f} |max| {d['speed_step_abs_max']:.3f} "
              f"bins {d['speed_step_bins']}; start heading mean {d['start_heading_mean_deg']:.1f} max {d['start_heading_max_deg']:.1f} "
              f"bins {d['start_heading_bins']}; accel [{d['min_accel_mps2']:.3f}, {d['max_accel_mps2']:.3f}]; acceptance {d['acceptance']:.3f}; "
              f"v0 sources {agg['v0_sources']} cross-check {agg['v0_crosscheck_max_abs_diff']:.2e}", flush=True)
    return agg


# ------------------------------------------------------------------------------------------------------------------
# python API
# ------------------------------------------------------------------------------------------------------------------
def candidates(pose, goal, v0, n, seed, family='cont_head', max_head_deg=MAX_HEAD_DEG, *, anchors=False, tries_factor=64,
               base=None, v0_min=0.0, fixed_speed=None):
    """n validated routes of the planner family from `pose` (x, y, yaw) to `goal` (gx, gy), collector format.

    base = gen_planner.base_route(pose, goal) unless given; prior draws theta ~ the planner's sampling Gaussian with
    rng default_rng(seed) (IT.draw_prior + IT.from_params, exactly the planner's round-0 draws), each passed through the
    family transform and acceptance (free: validator only; cont: speed ramps from v0 then validator; cont_head: + start
    heading within max_head_deg of the yaw). anchors=True puts the valid designed anchors first (as the planner's round
    0). Raises RuntimeError if tries_factor * n draws give fewer than n routes. Deterministic given the arguments."""
    pose = np.asarray(pose, float).reshape(3); goal = np.asarray(goal, float).reshape(2)
    base = GP.base_route(pose, goal) if base is None else base
    if not IT.valid(base, pose):
        raise RuntimeError(f'no valid base route from {pose.round(3).tolist()} to {goal.round(3).tolist()}')
    fam = Family(family, v0, float(pose[2]), max_head_deg, v0_min)
    rng = np.random.default_rng(int(seed))
    xy, station, f, L = IT._base_arrays(base)
    out, kinds, draws = [], [], 0
    if anchors:
        sp = (2.0, 4.0, 6.0) if fixed_speed is None else (float(fixed_speed),)
        for r in (fam.apply(r0) for r0 in S.anchors(base, speeds=sp)):
            if len(out) < n and fam.ok(r, pose):
                out.append(r); kinds.append('anchor')
    while len(out) < n and draws < int(tries_factor) * int(n):
        draws += 1
        r = fam.apply(IT.from_params(base, IT.draw_prior(rng, L, fixed_speed), fixed_speed))
        if fam.ok(r, pose):
            out.append(r); kinds.append('sample')
    if len(out) < n:
        raise RuntimeError(f'only {len(out)}/{n} {family} candidates after {draws} draws; stats {fam.stats}')
    res = []
    for i, (r, kind) in enumerate(zip(out, kinds)):
        route = {k: np.asarray(r[k], float) for k in ('waypoints', 'speeds', 'stations', 'headings')}
        d0 = float(np.linalg.norm(route['waypoints'][0] - pose[:2])); d1 = float(np.linalg.norm(route['waypoints'][-1] - goal))
        assert d0 <= 0.25 and d1 <= 0.25, (d0, d1)
        route['meta'] = {**r.get('meta', {}), **fam.describe(), 'kind': kind, 'index': i, 'seed': int(seed), 'draws': draws,
                         'branch_pose': pose.tolist(), 'goal_xy': goal.tolist(), 'base': dict(base.get('meta', {})),
                         'start_speed_mps': float(route['speeds'][0]),
                         'speed_step_mps': float(route['speeds'][0] - (v0 or 0.0)),
                         'start_heading_err_deg': start_heading_err_deg(route, pose[2]), 'start_err_m': d0, 'end_err_m': d1,
                         'route_sha256': IT.route_sha256(route), 'family_stats': dict(fam.stats)}
        res.append(route)
    return res


def plan_decision(ens, pose, goal, v0, hist=None, hmask=None, group='g', arm='B', family='cont_head', max_head_deg=MAX_HEAD_DEG,
                  world='crm', domain=None, map_root=None, v0_min=0.0, float16=True, base=None, fixed2=False):
    """One decision with the CLI's semantics: ens = CIEnsemble / ga_planner.Ensemble or a glob; base_route(pose, goal);
    seed IT.seed(group, arm tag) with the tags of planner_arms.arm_specs (world crm -> 'crm_proposal' for arm A)."""
    if map_root is not None and DS.G.get('rgbd') is None:
        DS.init_map(map_root)
    if isinstance(ens, str):
        ens = CIEnsemble(ens)
    pose = np.asarray(pose, float); goal = np.asarray(goal, float)
    base = GP.base_route(pose, goal) if base is None else base
    deployed_tag = ('crm_fixed2' if fixed2 else 'crm_proposal') if world == 'crm' else ('gen_fixed2' if fixed2 else 'gen_night2')
    sp = PA.arm_specs(deployed_tag, '_fixed2' if fixed2 else '')[arm]
    scorer = CIScorer(ens, pose[:2], goal, float(pose[2]), domain or world, hist, hmask, float16=float16)
    fam = Family(family, v0, float(pose[2]), max_head_deg, v0_min)
    rng = np.random.default_rng(IT.seed(group, sp['tag']))
    fs = 2.0 if fixed2 else None
    if sp['kind'] == 'oneshot':
        res = oneshot_family(base, pose, goal, scorer, rng, fam, n=sp['n'], fixed_speed=fs, objective=sp['objective'])
    else:
        res = plan_iter_family(base, pose, goal, scorer, fam, rounds=sp['rounds'], n=sp['n'], objective=sp['objective'],
                               c_fail=sp.get('c_fail', 60.0), rng=rng, anchors=True, fixed_speed=fs)
    if res is not None:
        res['route_sha256'] = IT.route_sha256(res['route'])
        res.update(pick_stats(res['route'], v0 or 0.0, pose[2]))
    return res


# ------------------------------------------------------------------------------------------------------------------
# self-tests T1-T4
# ------------------------------------------------------------------------------------------------------------------
def _run_cli(args, log):
    import subprocess
    cmd = [sys.executable, str(Path(__file__).resolve())] + [str(x) for x in args]
    env = dict(os.environ, PYTHONPATH='src:scripts', OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS', '6'))
    with open(log, 'w') as fh:
        fh.write(' '.join(cmd) + '\n'); fh.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    assert rc == 0, f'{cmd} failed, see {log}'
    return cmd


def selftest(out):
    out.mkdir(parents=True, exist_ok=True)
    R = dict(started=time.strftime('%Y-%m-%d %H:%M:%S'))
    poses_path = K1 / 'a5/poses_crm.json'; poses = json.load(open(poses_path))
    cases = K1 / 'suite/cases'; mroot = 'artifacts/traverse/crm_f104_v1/map_root'
    H = str(K1 / 'train/deploy_v1/H_deploy_s*.pt'); ref = K1 / 'a5/picks_crm_H_named/picks'; ref_routes = K1 / 'a5/picks_crm_H/routes'
    groups = sorted(poses)
    common = ['--cases', cases, '--map-root', mroot, '--world', 'crm', '--domain', 'crm', '--verify', '0', '--task-root', ROOT / 'artifacts/traverse/crm_improve_20260922']

    # T1: family free on 3 groups == K1 picks (route sha256), route files byte-identical to K1's
    g3 = groups[:3]
    d1 = out / 't1_free'
    R['t1_cmd'] = _run_cli(['--family', 'free', '--models', H, '--arms', 'B', '--poses', poses_path, '--groups', ','.join(g3), '--out', d1] + common, out / 't1_free.log')
    t1 = []
    for g in g3:
        mine = json.load(open(d1 / 'picks' / f'{g}.json'))['arms']['B']; theirs = json.load(open(ref / f'{g}.json'))['arms']['B']
        same_bytes = (d1 / 'routes' / f'{g}__B.json').read_bytes() == (ref_routes / f'{g}__B.json').read_bytes()
        t1.append(dict(group=g, sha_mine=mine['route_sha256'], sha_k1=theirs['route_sha256'], equal=mine['route_sha256'] == theirs['route_sha256'],
                       z_mine=mine['z_mean'], z_k1=theirs['z_mean'], index_mine=mine['index'], index_k1=theirs['index'], route_file_bytes_equal=same_bytes))
    R['T1'] = dict(groups=t1, passed=all(x['equal'] and x['z_mine'] == x['z_k1'] and x['route_file_bytes_equal'] for x in t1))
    print('T1', json.dumps(R['T1'], indent=None), flush=True)

    # in-process setup for T1b-T4
    DS.init_map(str(ROOT / mroot))
    ens = CIEnsemble(H)
    specs = PA.arm_specs('crm_proposal', '')

    def decision(g):
        a = argparse.Namespace(from_run=None, frame=60, pose_along_s=None, pose_override=None, goal=None, history=None)
        case, lay, pose, goal, base, hist, hmask, src = GA.decision_for(g, str(cases / f'{g}.json'), a, poses, ens.hist_T)
        v0, how, check = decision_v0(poses[g], hist, hmask, src)
        return pose, goal, base, hist, hmask, src, v0, how, check

    # T1b: plan_iter_family(free) == f104_n2_iter.plan_iter / oneshot in process (arms A and B), pools identical
    t1b = []
    for g in g3:
        pose, goal, base, hist, hmask, src, v0, how, _ = decision(g)
        sc = GA.Scorer(ens, pose[:2], goal, float(pose[2]), 'crm', hist, hmask)
        for arm in ('A', 'B'):
            sp = specs[arm]
            if sp['kind'] == 'oneshot':
                ra = IT.oneshot(base, pose, goal, sc, np.random.default_rng(IT.seed(g, sp['tag'])), n=sp['n'])
                rb = oneshot_family(base, pose, goal, sc, np.random.default_rng(IT.seed(g, sp['tag'])), Family('free'), n=sp['n'])
            else:
                ra = IT.plan_iter(base, pose, goal, sc, rounds=sp['rounds'], n=sp['n'], rng=np.random.default_rng(IT.seed(g, sp['tag'])))
                rb = plan_iter_family(base, pose, goal, sc, Family('free'), rounds=sp['rounds'], n=sp['n'], rng=np.random.default_rng(IT.seed(g, sp['tag'])))
            t1b.append(dict(group=g, arm=arm, equal_sha=IT.route_sha256(ra['route']) == IT.route_sha256(rb['route']),
                            equal_pool_logits=bool(np.array_equal(ra['Z_mean'], rb['Z_mean'])), n=ra['n_evaluated']))
    R['T1b'] = dict(rows=t1b, passed=all(x['equal_sha'] and x['equal_pool_logits'] for x in t1b))
    print('T1b', R['T1b']['passed'], flush=True)

    def check_pool(cands, pose, v0, head=None):
        e = dict(n=len(cands), start_speed_max_err=0.0, max_accel=-9.0, min_accel=9.0, above_cap=0.0, below_floor=0.0, invalid=0,
                 heading_max_deg=0.0, start_dist_max_m=0.0)
        for r in cands:
            v = np.asarray(r['speeds'], float); acc = accel_profile(r); floor, cap = ramps(r['stations'], v0)
            e['start_speed_max_err'] = max(e['start_speed_max_err'], abs(float(v[0]) - min(v0, V_MAX)))
            e['max_accel'] = max(e['max_accel'], float(acc.max())); e['min_accel'] = min(e['min_accel'], float(acc.min()))
            e['above_cap'] = max(e['above_cap'], float((v - cap).max())); e['below_floor'] = max(e['below_floor'], float((np.minimum(floor, V_MAX) - v).max()))
            e['invalid'] += int(not IT.valid(r, pose))
            e['heading_max_deg'] = max(e['heading_max_deg'], start_heading_err_deg(r, pose[2]))
            e['start_dist_max_m'] = max(e['start_dist_max_m'], float(np.linalg.norm(np.asarray(r['waypoints'])[0] - pose[:2])))
        e['passed'] = bool(e['start_speed_max_err'] <= 1e-6 and e['max_accel'] <= A_ACC + 1e-6 and e['min_accel'] >= -A_DEC - 1e-6
                           and e['above_cap'] <= 1e-9 and e['below_floor'] <= 1e-9 and e['invalid'] == 0
                           and (head is None or e['heading_max_deg'] <= head + 1e-9))
        return e

    # T2 / T3: 20 groups, arms A (one-shot 256) and B (CEM 4x64): every evaluated candidate checked
    g20 = groups[::40][:20]
    T2, T3 = [], []
    for g in g20:
        pose, goal, base, hist, hmask, src, v0, how, check = decision(g)
        sc = GA.Scorer(ens, pose[:2], goal, float(pose[2]), 'crm', hist, hmask)
        for fam_name, rows in (('cont', T2), ('cont_head', T3)):
            for arm in ('A', 'B'):
                sp = specs[arm]; fam = Family(fam_name, v0, float(pose[2]), MAX_HEAD_DEG)
                rng = np.random.default_rng(IT.seed(g, sp['tag']))
                if sp['kind'] == 'oneshot':
                    res = oneshot_family(base, pose, goal, sc, rng, fam, n=sp['n'], keep_candidates=True)
                else:
                    res = plan_iter_family(base, pose, goal, sc, fam, rounds=sp['rounds'], n=sp['n'], rng=rng, keep_candidates=True)
                e = check_pool(res['candidates'], pose, v0, MAX_HEAD_DEG if fam_name == 'cont_head' else None)
                e.update(group=g, arm=arm, v0=v0, v0_source=how, v0_check=check, stats=res['family_stats'],
                         pick_speed_step=float(res['route']['speeds'][0] - v0), pick_heading_deg=start_heading_err_deg(res['route'], pose[2]),
                         z_pick=res['z_mean'], kinds={k: int(sum(1 for q in res['kinds'] if q == k)) for k in ('anchor', 'sample', 'mean')})
                rows.append(e)
    for name, rows in (('T2', T2), ('T3', T3)):
        R[name] = dict(rows=rows, passed=all(x['passed'] for x in rows), n_candidates=int(sum(x['n'] for x in rows)),
                       start_speed_max_err=max(x['start_speed_max_err'] for x in rows), max_accel=max(x['max_accel'] for x in rows),
                       min_accel=min(x['min_accel'] for x in rows), invalid=int(sum(x['invalid'] for x in rows)),
                       heading_max_deg=max(x['heading_max_deg'] for x in rows),
                       acceptance=float(sum(x['stats']['accepted'] for x in rows) / max(sum(x['stats']['built'] for x in rows), 1)),
                       rejected_heading=int(sum(x['stats']['rejected_heading'] for x in rows)),
                       rejected_validator=int(sum(x['stats']['rejected_validator'] for x in rows)),
                       pools_short=[(x['group'], x['arm'], x['n']) for x in rows if x['n'] < (256 if x['arm'] == 'A' else 257)])
        print(name, {k: v for k, v in R[name].items() if k != 'rows'}, flush=True)

    # T3b: the python API candidates() on the same 20 decisions, every family, and determinism
    api = []
    for g in g20[:10]:
        pose, goal, base, hist, hmask, src, v0, how, _ = decision(g)
        for fam_name in FAMILIES:
            c1 = candidates(pose, goal, v0, 6, seed=IT.seed(g, 'api'), family=fam_name)
            c2 = candidates(pose, goal, v0, 6, seed=IT.seed(g, 'api'), family=fam_name)
            e = check_pool(c1, pose, v0, MAX_HEAD_DEG if fam_name == 'cont_head' else None) if fam_name != 'free' else dict(passed=all(IT.valid(r, pose) for r in c1))
            api.append(dict(group=g, family=fam_name, deterministic=[r['meta']['route_sha256'] for r in c1] == [r['meta']['route_sha256'] for r in c2],
                            passed=e['passed'], draws=c1[-1]['meta']['draws']))
    free_equal = []
    for g in g20[:3]:        # free candidates() == the planner's round-0 prior draws of arm A (anchors off -> the draws only)
        pose, goal, base, hist, hmask, src, v0, how, _ = decision(g)
        cand = candidates(pose, goal, v0, 20, seed=IT.seed(g, 'crm_proposal'), family='free', base=base, tries_factor=8)
        xy, st, f, L = IT._base_arrays(base)
        ref_draws, _, _ = IT.sample_valid(base, pose, np.random.default_rng(IT.seed(g, 'crm_proposal')), 20, 160, L, None)
        free_equal.append(all(IT.route_sha256(a) == IT.route_sha256(b) for a, b in zip(cand, ref_draws)))
    R['T3b_api'] = dict(rows=api, passed=all(x['passed'] and x['deterministic'] for x in api) and all(free_equal), free_equals_planner_draws=free_equal)
    print('T3b', R['T3b_api']['passed'], flush=True)

    # T4: standing start (layout pose, no history, v0 = 0) through the CLI, cont_head, arms A,B, 3 groups; and --v0-min 0.5
    d4 = out / 't4_standing'
    R['t4_cmd'] = _run_cli(['--family', 'cont_head', '--models', H, '--arms', 'A,B', '--groups', ','.join(g3), '--out', d4] + common, out / 't4_standing.log')
    d4b = out / 't4_standing_v0min'
    R['t4b_cmd'] = _run_cli(['--family', 'cont_head', '--v0-min', '0.5', '--models', H, '--arms', 'A,B', '--groups', ','.join(g3), '--out', d4b] + common, out / 't4_standing_v0min.log')
    t4 = []
    for d in (d4, d4b):
        for g in g3:
            pk = json.load(open(d / 'picks' / f'{g}.json'))
            for arm in ('A', 'B'):
                fa = pk['family']['arms'][arm]
                t4.append(dict(dir=d.name, group=g, arm=arm, v0=pk['family']['v0_mps'], v0_source=pk['family']['v0_source'],
                               start_speed=fa['start_speed_mps'], heading=fa['start_heading_err_deg'], z=pk['arms'][arm]['z_mean'],
                               history_valid=pk['history']['n_valid'], max_accel=fa['max_accel_mps2']))
    R['T4'] = dict(rows=t4, passed=all(x['v0'] == 0.0 and x['v0_source'] == 'standing_start' and x['heading'] <= MAX_HEAD_DEG + 1e-9
                                       and abs(x['start_speed'] - (0.0 if x['dir'] == 't4_standing' else 0.5)) < 1e-9 for x in t4))
    print('T4', R['T4']['passed'], flush=True)
    R['T5'] = selftest_ci(out)
    R['passed'] = all(R[k]['passed'] for k in ('T1', 'T1b', 'T2', 'T3', 'T3b_api', 'T4', 'T5'))
    R['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(R, open(out / 'RESULTS.json', 'w'), indent=1, default=str)
    print('ALL PASSED' if R['passed'] else 'SOME FAILED', flush=True)
    return R


def selftest_ci(out, ckpt_glob=None):
    """T5: model_kind 'ci_train' members (checkpoints of ci_train.py's own self-test, copied to out/ci_ckpt): the CLI
    plans with them; CIScorer (z once per decision) == ci_train.score with the raw window per call; encode counts."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    ckpt_glob = ckpt_glob or str(out / 'ci_ckpt' / '*.pt')
    R = dict(ckpts=sorted(glob.glob(ckpt_glob)))
    if _ci_train_module() is None or not R['ckpts']:
        R.update(passed=False, skipped='ci_train.py or checkpoints absent'); return R
    poses_path = K1 / 'a5/poses_crm.json'; poses = json.load(open(poses_path)); groups = sorted(poses)[:3]
    cases = K1 / 'suite/cases'; mroot = 'artifacts/traverse/crm_f104_v1/map_root'
    d5 = out / 't5_ci_cli'
    R['cmd'] = _run_cli(['--family', 'cont_head', '--models', ckpt_glob, '--arms', 'A,B', '--poses', poses_path, '--groups', ','.join(groups),
                         '--out', d5, '--cases', cases, '--map-root', mroot, '--world', 'crm', '--domain', 'crm', '--verify', '0',
                         '--task-root', ROOT / 'artifacts/traverse/crm_improve_20260922'], out / 't5_ci_cli.log')
    sm = json.load(open(d5 / 'summary.json'))
    R['cli'] = dict(model_kinds=sm['model_kinds'], members=sm['members'], n_groups=sm['n_groups'],
                    step_abs_max={a: sm['family']['per_arm'][a]['speed_step_abs_max'] for a in ('A', 'B')},
                    heading_max={a: sm['family']['per_arm'][a]['start_heading_max_deg'] for a in ('A', 'B')},
                    n_encode=[json.load(open(d5 / 'picks' / f'{g}.json'))['history']['n_encode'] for g in groups])
    DS.init_map(str(ROOT / mroot))
    ens = CIEnsemble(ckpt_glob); CT = ens.CT
    n_hist = sum(1 for m in ens.members if m['use_hist'])
    counts = {}
    for i, mem in enumerate(ens.members):          # count encoder calls per member
        orig = mem['model'].encode
        def counted(h, m, _o=orig, _i=i):
            counts[_i] = counts.get(_i, 0) + 1
            return _o(h, m)
        mem['model'].encode = counted
    diffs, enc = [], []
    for g in groups:
        a = argparse.Namespace(from_run=None, frame=60, pose_along_s=None, pose_override=None, goal=None, history=None)
        case, lay, pose, goal, base, hist, hmask, src = GA.decision_for(g, str(cases / f'{g}.json'), a, poses, ens.hist_T)
        counts.clear()
        sc = CIScorer(ens, pose[:2], goal, float(pose[2]), 'crm', hist, hmask)
        v0, _, _ = decision_v0(poses[g], hist, hmask, src)
        cands = candidates(pose, goal, v0, 48, seed=IT.seed(g, 't5'), family='cont_head', base=base, anchors=True)
        Z1, _, _ = sc(cands[:24]); Z2, _, _ = sc(cands[24:])
        enc.append(dict(group=g, encodes_after_two_calls=int(sum(counts.values())), expected=n_hist))
        X, L = IT.corridors(cands); X = X.astype(np.float16).astype(np.float32); geom = GP.geom_ctx(pose[:2], goal, float(pose[2]), L)
        for i, mem in enumerate(ens.members):      # reference: ci_train.score with the raw window, encoded inside score
            ref = CT.score(mem['model'], mem['ck'], X, geom, hist=hist, hmask=hmask)
            diffs.append(float(np.abs(np.r_[Z1[i], Z2[i]] - ref).max()))
    R['score_vs_ci_train_max_abs'] = max(diffs); R['encodes'] = enc
    R['passed'] = bool(R['cli']['model_kinds'] == ['ci_train'] and max(R['cli']['step_abs_max'].values()) <= 1e-6
                       and max(R['cli']['heading_max'].values()) <= MAX_HEAD_DEG + 1e-9 and R['score_vs_ci_train_max_abs'] <= 1e-5
                       and all(e['encodes_after_two_calls'] == e['expected'] for e in enc) and all(n == n_hist for n in R['cli']['n_encode']))
    json.dump(R, open(out / 'RESULTS_T5.json', 'w'), indent=1, default=str)
    print('T5', json.dumps({k: v for k, v in R.items() if k not in ('ckpts',)}, default=str), flush=True)
    return R


if __name__ == '__main__':
    cli()
