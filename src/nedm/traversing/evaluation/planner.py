"""Route planning of the traversing evaluation: the decision state, the risk ensemble and its corridor scorer, the
sampling route optimizer and the pick (a planned, straight, given or locked route).

A port of the experiment branch at 901d6c9, bitwise equal to it on the record environment (test_planner.py):
  Decision   standing start (route_00 with its meta cleared; without one, base_route asserted valid), after a recorded
             approach (row F, or the terminal state when pass 1 has exactly F rows; base_route(pose, goal) asserted
             valid) or a released decision state (ga_planner.decision_for, ga_planner.py:288-316; ci_a5data.py:442-460).
  Ensemble   a sorted checkpoint glob of ONE kind in main's RiskModel: ci_train through main's loader and score
             (bs 1024, float32 -> float64); ga_train (cond none | hist_aux) and legacy gen_riskmodel.Net (cnn.* ->
             front.cnn.*) with ga_planner's numpy float32 standardisation, bs 256 and route_logit(haz).double()
             (ga_planner.py:322-387).
  Scorer     corridors on the static map rounded to float16, geom5 at the decision, the history encoded once per member;
             one ensemble call per candidate list (batch composition moves the logits by ~1e-2 on the GPU).
  optimize   f104_n2_iter.plan_iter (objective 'mean'): CEM with a cumulative elite set (every headline), or 'mppi' =
             ESS-tempered exponential weights (the branch's unused weighting='mppi'; off the validated matrix).
             rounds=1 is M1's one-shot pool (nav.py: anchors + 8n tries; its fixed-speed rescue pool has no anchors
             and 6n tries); an arm plans with rounds >= 2.
Planner numerics are asserted (torch defaults, ag_picks.py:199-204), never set; the gradient step (refine.py) scopes
its own deterministic cuDNN flags.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from nedm.traversing.training.risk_model import ZDIM, RiskModel, encode_history, load_risk_model, route_logit
from nedm.traversing.training.risk_model import score as ci_score

from .config import MODEL_KINDS, PICK_ARMS
from .routes import (HIST_DIM, HIST_T, KEYS, MODES, KNOTS, PRIOR_SD, StaticMap, _base_arrays, base_route, draw,
                     family_anchors, from_params, geom5, history_window, load_route, project, route_sha256, route_time,
                     straight, validate)
from .suites import lock_digest

GA_BS = 256
RECORD_ENV = ('NVIDIA GeForce RTX 5090', '2.12.0+cu130')       # where the released picks were planned (luffy)
TORCH_DEFAULTS = dict(cudnn_allow_tf32=True, matmul_allow_tf32=False, cudnn_benchmark=False, cudnn_deterministic=False,
                      deterministic_algorithms=False)


# ------------------------------------------------------------------------------------------------------- decision
def _layout(case):
    return np.array([*case['layout']['start_xy'], case['layout']['start_yaw']], float)


@dataclass(frozen=True, eq=False)
class Decision:
    """Where the planner starts: pose (x, y, yaw), goal (x, y), base route, the 40 x 15 history window (None = all
    masked, a standing start), the decision frame and where the state came from."""
    pose: np.ndarray
    goal: np.ndarray
    base: dict
    hist: np.ndarray | None = None
    hmask: np.ndarray | None = None
    frame: int = 0
    source: str = 'layout'

    def __post_init__(self):
        pose, goal = np.asarray(self.pose, np.float64), np.asarray(self.goal, np.float64)
        if pose.shape != (3,) or goal.shape != (2,) or not (np.isfinite(pose).all() and np.isfinite(goal).all()):
            raise ValueError(f'decision pose {pose} / goal {goal}: need finite (x, y, yaw) and (x, y)')
        if (self.hist is None) != (self.hmask is None):
            raise ValueError('decision history needs both hist and hmask, or neither')
        if self.hist is not None:
            h, m = np.asarray(self.hist), np.asarray(self.hmask)
            if h.shape != (HIST_T, HIST_DIM) or h.dtype != np.float32 or m.shape != (HIST_T,) or m.dtype != bool:
                raise ValueError(f'history {h.shape} {h.dtype} / mask {m.shape} {m.dtype}: need (40, 15) f32 / bool')
        object.__setattr__(self, 'pose', pose)
        object.__setattr__(self, 'goal', goal)

    @classmethod
    def standing(cls, task):
        """The case layout pose at rest; base = route_00 (meta cleared, planner_arms.load_case) or a valid base_route."""
        c = task.read_case()
        pose, goal = _layout(c), np.asarray(c['goal_xy'], float)
        if task.route00 is None:            # a custom task without route_00: asserted valid, as at a moving decision
            return cls._moving(task, pose, goal, None, None, 0, 'layout')
        return cls(pose, goal, task.read_route('route00'))

    @classmethod
    def after_approach(cls, task, pass1, F):
        """Frame F of a pass-1 Record (its arrays: state, action, pose [, terminal_pose, terminal_state])."""
        a = pass1.arrays
        st, ac, po, F = np.asarray(a['state'], np.float32), np.asarray(a['action'], np.float32), \
            np.asarray(a['pose'], np.float64), int(F)
        if len(po) > F:
            pose, term = po[F], None
        elif len(po) == F and 'terminal_pose' in a and 'terminal_state' in a:
            pose, term = np.asarray(a['terminal_pose'], np.float64).reshape(3), a['terminal_state']
        else:
            raise ValueError(f'pass 1 has {len(po)} frames: no decision state at frame {F}')
        hist, hmask = history_window(st, ac, F, terminal_state=term)
        sha = hashlib.sha256(b''.join(np.ascontiguousarray(x).tobytes() for x in (st, ac, po, pose))).hexdigest()
        goal = np.asarray(task.read_case()['goal_xy'], float)
        return cls._moving(task, pose, goal, hist, hmask, F, f'pass1:{sha[:16]}')

    @classmethod
    def from_release(cls, task, poses, env):
        """A released decision state: poses_<world>[_all].json entry -> pose, history npz (its absolute workstation
        path rebased on 'artifacts/' against Env.data and release-checked), goal = the case goal."""
        e = _poses(str(env.file(poses))).get(task.id)
        if e is None or 'history' not in e or 'run' in e:
            raise KeyError(f'{poses}: no pose + history decision state for {task.id}')
        h = e['history']
        if 'artifacts/' not in h:
            raise ValueError(f'{poses}: history path {h} is not under artifacts/')
        with np.load(env.file('data:' + h[h.index('artifacts/'):])) as z:
            hist, hmask = np.asarray(z['hist'], np.float32), np.asarray(z['hmask'], bool)
        goal = np.asarray(e['goal'] if 'goal' in e else task.read_case()['goal_xy'], float)
        return cls._moving(task, np.asarray(e['pose'], float), goal, hist, hmask, int(e['frame']), f'release:{poses}')

    @classmethod
    def _moving(cls, task, pose, goal, hist, hmask, frame, source):
        base = base_route(pose, goal)
        if not validate(base, pose):          # ga_planner.py:315
            raise ValueError(f'{task.id}: no valid base route from {np.round(pose, 3)} to {np.round(goal, 3)}')
        return cls(pose, goal, base, hist, hmask, frame, source)


@lru_cache(maxsize=8)
def _poses(path):
    return json.loads(Path(path).read_text())


# ------------------------------------------------------------------------------------------------------- ensemble
def model_info(ck) -> tuple[str, str | None]:
    """(kind, domain_filter) of a checkpoint (path or loaded dict); config.validate's model_info. kind: 'ci_train'
    (CNN-GRU), 'ga_train', 'legacy' (no model_kind) or the refused 'txjoint' / 'ci_train_<arch>'; domain_filter:
    the ground it was trained on, 'crm' | 'rigid' | 'both' (None: legacy gen_riskmodel.Net records none)."""
    ck = torch.load(ck, map_location='cpu', weights_only=False) if isinstance(ck, (str, Path)) else ck
    if not isinstance(ck, dict) or 'state' not in ck:
        raise ValueError('not a risk-model checkpoint (a dict with a state)')
    kind, arch = ck.get('model_kind') or 'legacy', ck.get('arch')
    kind = kind if kind != 'ci_train' or arch == 'gru' else ('txjoint' if arch == 'txjoint' else f'ci_train_{arch}')
    return kind, ck.get('domain_filter')


def _load(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    kind = model_info(ck)[0]
    if kind == 'ci_train':
        model, ck = load_risk_model(path, device)
        if int(ck['cin']) != 6 or list(np.asarray(ck['geom_cols']).ravel()) != [17, 18, 19, 20, 21] or \
                model.use_hist and int(ck['hist_T']) > HIST_T:      # a longer window would be zero-padded (cut_window)
            raise ValueError(f'{path}: cin {ck["cin"]} / geom_cols {ck["geom_cols"]} / hist_T {ck["hist_T"]}: not the '
                             f'corridor + geom5 model with at most {HIST_T} history frames')
        return model, ck, kind
    if kind not in MODEL_KINDS:
        raise ValueError(f'{path}: model kind {kind!r} is not one of {MODEL_KINDS} (the transformer is context only)')
    st = dict(ck['state'])
    if kind == 'legacy':                  # gen_riskmodel.Net: one GRU layer whatever `layers` says
        if ck.get('arch') != 'gru':
            raise ValueError(f'{path}: legacy arch {ck.get("arch")!r}, only gru')
        st, cond, zdim = {('front.' + k if k.startswith('cnn.') else k): v for k, v in st.items()}, 'none', 0
    else:
        cond, zdim = ck['cond'], int(ck['zdim'])
        if cond not in ('none', 'hist_aux') or int(ck['nctx']) != 5 or (cond == 'hist_aux') != (zdim > 0):
            raise ValueError(f'{path}: ga_train cond {cond!r} / nctx {ck["nctx"]} / zdim {zdim} (none | hist_aux only)')
        if zdim and (int(ck['hist_T']) != HIST_T or len(np.asarray(ck['hist_mu']).ravel()) != HIST_DIM):
            raise ValueError(f'{path}: history {ck["hist_T"]} x {len(ck["hist_mu"])}, need {HIST_T} x {HIST_DIM}')
    model = RiskModel(cond, int(ck['cin']), int(ck['nctx']), zdim=zdim or ZDIM, hist_dh=HIST_DIM,
                      width=int(st['mix.weight_hh_l0'].shape[1]))
    model.load_state_dict(st)             # strict: stacked GRUs, a stray history encoder or domain head are refused
    return model.to(device).eval(), ck, kind


@lru_cache(maxsize=4)
def _ensemble(paths, device):
    return Ensemble(paths, device)


class Ensemble:
    """Members of one kind in sorted path order, in eval mode on `device`."""

    def __init__(self, paths, device):
        self.paths, self.device = tuple(Path(p) for p in paths), device
        if not self.paths:
            raise ValueError('an ensemble needs at least one checkpoint')
        self.sha256 = [hashlib.sha256(p.read_bytes()).hexdigest() for p in self.paths]
        self.members = [_load(p, device) for p in self.paths]
        kinds = sorted({k for _, _, k in self.members})
        if len(kinds) != 1:
            raise ValueError(f'mixed ensemble {kinds}: {self.paths[0].parent}')
        self.kind = kinds[0]

    @classmethod
    def load(cls, paths, device):
        """Cached per process (key: the sorted paths and the device)."""
        return _ensemble(tuple(sorted(str(p) for p in paths)), device)

    def encode(self, hist, hmask):
        """Per-member history code of one decision (None for members without history; None window = all masked)."""
        if hist is None:
            hist, hmask = np.zeros((HIST_T, HIST_DIM), np.float32), np.zeros(HIST_T, bool)
        zs, dev = [], self.device
        for net, ck, kind in self.members:
            if not net.use_hist:
                zs.append(None)
            elif kind == 'ci_train':
                zs.append(encode_history(net, ck, hist, hmask, dev))
            else:                          # ga_planner.standardise_history + GANet.encode (ga_planner.py:125, 337)
                m = np.asarray(hmask, bool)
                mu, sd = (np.asarray(ck[k], np.float32).reshape(1, -1) for k in ('hist_mu', 'hist_sd'))
                h = (np.nan_to_num(np.asarray(hist, np.float32)) - mu) / sd
                h = np.where(m[:, None], h, 0.0).astype(np.float32)
                with torch.no_grad():
                    zs.append(net.encode(torch.tensor(h[None], device=dev), torch.tensor(m[None], device=dev)))
        return zs

    def logits(self, X, g5, zs):
        """(members, n) float64 route logits of raw corridors X (n, cin - 1, 96, 32) f32 at raw contexts g5 (n, 5)."""
        out, dev = [], self.device
        for (net, ck, kind), z in zip(self.members, zs):
            if kind == 'ci_train':
                out.append(np.asarray(ci_score(net, ck, X, g5, z=z), np.float64).reshape(-1))
                continue
            nm, rows = ck['norm'], []
            cont = list(nm.get('cont_index', [0, 1, 2, 3]))
            mu, sd, cmu, csd = (np.asarray(v, np.float32).reshape(-1)
                                for v in (nm['mu'], nm['sd'], ck['ctx_mu'], ck['ctx_sd']))
            for j in range(0, len(X), GA_BS):          # ga_planner.Scorer.member_logits (ga_planner.py:360-387)
                x = X[j:j + GA_BS].astype(np.float32).copy()
                x[:, cont] = (x[:, cont] - mu[None, :, None, None]) / sd[None, :, None, None]
                x = np.concatenate([x, np.ones((len(x), 1, x.shape[2], x.shape[3]), np.float32)], 1)
                c = (g5[j:j + GA_BS] - cmu) / csd
                with torch.no_grad():
                    xt, ct = torch.tensor(x, device=dev), torch.tensor(c, dtype=torch.float32, device=dev)
                    haz = net(xt, ct, z=None if z is None else z.expand(len(x), -1))['haz']
                    rows.append(route_logit(haz).double().cpu().numpy())
            out.append(np.concatenate(rows))
        return np.stack(out)


class Scorer:
    """Candidate routes -> Z (members, n) float64 for one decision on a static planner map (corridors rounded to
    float16 as the deployed pools were, f104_n2_iter.py:170-182); the history is encoded here, once per member."""

    def __init__(self, ens: Ensemble, smap: StaticMap, dec: Decision):
        if any(int(ck['cin']) != 6 for _, ck, _ in ens.members):
            raise ValueError('static-map scoring needs corridor models (cin 6); M1 direct-depth models score in nav.py')
        self.ens, self.smap, self.dec, self.calls = ens, smap, dec, 0
        self.z = ens.encode(dec.hist, dec.hmask)

    def __call__(self, cands):
        X, L = self.smap.corridors(cands)
        Z = self.ens.logits(X.astype(np.float16).astype(np.float32), geom5(self.dec.pose, self.dec.goal, L), self.z)
        self.calls += 1
        return Z


def record_numerics(device='cuda') -> dict:
    """The planner's torch numerics, asserted to be the torch defaults of the record environment (never set here)."""
    b = torch.backends
    flags = dict(cudnn_allow_tf32=b.cudnn.allow_tf32, matmul_allow_tf32=b.cuda.matmul.allow_tf32,
                 cudnn_benchmark=b.cudnn.benchmark, cudnn_deterministic=b.cudnn.deterministic,
                 deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    if flags != TORCH_DEFAULTS:
        raise RuntimeError(f'planner numerics {flags} differ from the torch defaults of record {TORCH_DEFAULTS}')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(f'device {device}: no CUDA GPU here (NEDM_DEVICE=cpu plans comparably, never bitwise)')
    gpu = torch.cuda.get_device_name(torch.device(device)) if str(device).startswith('cuda') else 'cpu'
    return dict(gpu=gpu, torch=torch.__version__, cuda=torch.version.cuda, cudnn=b.cudnn.version(), **flags,
                env='bitwise_env' if (gpu, torch.__version__) == RECORD_ENV else 'comparable')


# ------------------------------------------------------------------------------------------------------ optimizer
@dataclass(frozen=True, eq=False)
class Result:
    """Everything scored, in scoring order (anchors first), and the argmin of the ensemble-mean logit."""
    cands: list
    thetas: list
    kinds: list
    rounds: list
    z_mean: np.ndarray
    z_pess: np.ndarray
    index: int
    log: list
    mu: np.ndarray
    sd: np.ndarray
    tries: int
    iterated: bool

    @property
    def route(self):
        """The pick with plan_iter's meta (f104_n2_iter.py:315-320)."""
        i, r, rd = self.index, self.cands[self.index], self.rounds
        m = r.get('meta', {})
        return {**{k: np.asarray(r[k]) for k in KEYS},
                'meta': {**m, 'candidate': 'n2_iter' if self.iterated else m.get('candidate', 'n2_wide'),
                         'kind': self.kinds[i], 'round': int(rd[i]), 'rank': i - sum(q != rd[i] for q in rd[:i]),
                         'theta': None if self.thetas[i] is None else np.asarray(self.thetas[i]).tolist()}}


def ess_weights(J, ess_frac=0.25):
    """Softmax weights exp(-(J - min J) / T), T bisected (40 steps in [1e-3, 1e3]) to ESS = max(2, ess_frac * n)
    (f104_n2_iter.ess_weights) -> (w, ESS, T)."""
    J = np.asarray(J, float); J = J - J.min()
    target, lo, hi = max(2.0, ess_frac * len(J)), 1e-3, 1e3
    for _ in range(40):
        T = math.sqrt(lo * hi)
        w = np.exp(-J / T); w /= w.sum()
        if 1.0 / np.sum(w ** 2) < target:
            lo = T
        else:
            hi = T
    w = np.exp(-J / math.sqrt(lo * hi)); w /= w.sum()
    return w, float(1.0 / np.sum(w ** 2)), math.sqrt(lo * hi)


def optimize(base, pose, score, rng, *, rounds=4, n=64, update='cem', fixed_speed=None, anchors=True, tries_factor=8,
             valid=validate):
    """f104_n2_iter.plan_iter statement for statement (objective = ensemble-mean logit). theta = (a1..a3 | dv1..dv4),
    R^3 on a fixed-speed family. Round 0: the valid anchors, then prior draws up to n valid (rejected draws consume the
    rng). Later rounds: the refit mean first, then N(mu, sd) draws. After every round of a multi-round search the
    Gaussian is refit on ALL non-anchor thetas scored so far (CEM: the best max(4, ceil(0.15 n)) by stable argsort, sd
    floored at 0.15 prior sd; mppi: ESS-tempered weights), mu projected onto the caps; the final mean is scored alone.
    score(list of routes) -> Z (members, len) float64, one call per list. Returns None when nothing valid was found."""
    if rounds < 1 or n < 1 or tries_factor < 1 or update not in ('cem', 'mppi'):
        raise ValueError(f'optimize: rounds {rounds}, n {n}, tries_factor {tries_factor}, update {update!r}')
    dim = MODES if fixed_speed is not None else MODES + KNOTS
    prior = PRIOR_SD[:dim]
    mu, sd = np.zeros(dim), prior.copy()
    L = _base_arrays(base)[2]
    anc = [r for r in family_anchors(base, (2.0, 4.0, 6.0) if fixed_speed is None else (float(fixed_speed),))
           if valid(r, pose)] if anchors else []
    C, TH, KIND, RND, ZM, ZP, log, tries_total = [], [], [], [], [], [], [], 0

    def evaluate(cs, ths, ks, k):
        Z = np.asarray(score(cs))
        if Z.ndim != 2 or Z.shape[1] != len(cs) or Z.dtype != np.float64:
            raise ValueError(f'score returned {Z.shape} {Z.dtype}, need (members, {len(cs)}) float64')
        C.extend(cs); TH.extend(ths); KIND.extend(ks); RND.extend([k] * len(cs))
        ZM.append(Z.mean(0)); ZP.append(Z.max(0))

    for k in range(rounds):
        if k == 0:
            cs, ths, ks = list(anc), [None] * len(anc), ['anchor'] * len(anc)
        else:
            cs, ths, ks = [], [], []
            rm = from_params(base, mu, fixed_speed)
            if valid(rm, pose):
                cs, ths, ks = [rm], [np.asarray(rm['meta']['theta'])], ['mean']
        tries = drawn = 0
        while len(cs) < n and tries < tries_factor * n:
            tries += 1
            th = draw(rng, L, fixed_speed, None if k == 0 else mu, None if k == 0 else sd)
            r = from_params(base, th, fixed_speed)
            if valid(r, pose):
                cs.append(r); ths.append(th); ks.append('sample'); drawn += 1
        tries_total += tries
        if not cs:
            log.append(dict(round=k, n=0, tries=tries, acceptance=0.0))
            break
        evaluate(cs, ths, ks, k)
        J = np.concatenate(ZM)
        has = np.array([t is not None for t in TH])
        e = dict(round=k, n=len(cs), tries=tries, acceptance=drawn / max(tries, 1), zmin_round=float(ZM[-1].min()),
                 zmin=float(J.min()), jmin=float(J.min()), mu=mu.round(3).tolist(), sd=sd.round(3).tolist())
        if rounds > 1:                     # plan_iter: k < rounds - 1 or score_mean (default rounds > 1)
            if has.any():
                Th, Jh = np.stack([t for t in TH if t is not None]), J[has]
                if update == 'cem':
                    ne = min(len(Th), max(4, math.ceil(0.15 * n)))
                    o = np.argsort(Jh, kind='stable')[:ne]
                    mu, sd = Th[o].mean(0), np.maximum(Th[o].std(0), 0.15 * prior)
                    e.update(n_elite=int(ne), ess=float(ne), j_elite_max=float(Jh[o[-1]]))
                else:                      # f104_n2_iter.py:293-297
                    w, ess, T = ess_weights(Jh, 0.25)
                    mu = (w[:, None] * Th).sum(0)
                    sd = np.maximum(np.sqrt((w[:, None] * (Th - mu) ** 2).sum(0)), 0.15 * prior)
                    e.update(ess=ess, temperature=T)
                mu = project(mu, L, fixed_speed)
            e.update(mu_next=mu.round(3).tolist(), sd_next=sd.round(3).tolist())
        log.append(e)
    if rounds > 1 and C:
        rm = from_params(base, mu, fixed_speed)
        if valid(rm, pose):
            evaluate([rm], [np.asarray(rm['meta']['theta'])], ['mean'], rounds)
            log.append(dict(round=rounds, n=1, tries=0, acceptance=1.0, kind='final_mean', zmin_round=float(ZM[-1][0])))
        else:
            log.append(dict(round=rounds, n=0, tries=0, acceptance=0.0, kind='final_mean_invalid'))
    if not C:
        return None
    zm = np.concatenate(ZM)
    return Result(C, TH, KIND, RND, zm, np.concatenate(ZP), int(np.argmin(zm)), log, mu, sd, tries_total, rounds > 1)


# ----------------------------------------------------------------------------------------------------------- pick
@dataclass(frozen=True, eq=False)
class Pick:
    """The route to drive: arm 'given' | 'straight' | 'B' (sampling search) | 'G' (gradient; B's route if abstained),
    or no route (code U). record: tag, seed, n_evaluated, log, gradient rows, models, map, numerics, ..."""
    route: dict | None
    route_sha256: str | None
    arm: str
    z_mean: float | None = None
    z_pess: float | None = None
    record: dict = field(default_factory=dict)

    @classmethod
    def of(cls, route, arm, z_mean=None, z_pess=None, **record):
        return cls(route, None if route is None else route_sha256(route), arm, z_mean, z_pess, record)

    @classmethod
    def of_result(cls, res: Result, **record):
        """Arm B with ga_planner's pick fields (ga_planner.py:551-562)."""
        i, r = res.index, res.route
        return cls.of(r, 'B', float(res.z_mean[i]), float(res.z_pess[i]), **record, index=i, kind=res.kinds[i],
                      round=int(res.rounds[i]), theta=r['meta']['theta'], P=float(1 - np.exp(-np.exp(res.z_mean[i]))),
                      T=route_time(r), n_evaluated=len(res.cands), tries=res.tries, log=res.log,
                      mu_final=res.mu.tolist(), sd_final=res.sd.tolist(),
                      kinds_count={k: res.kinds.count(k) for k in ('anchor', 'sample', 'mean')})

    @classmethod
    def locked(cls, cfg, task, env):
        """The recorded pick of `task` in the arm's locked pick folder: its tasks.json row -> routes/<row id>.json, whose
        content hash must equal the row sha256 (the release-checked PICKS_LOCKED.sha256 recomputes over the routes).
        The pick must be of the arm's planner (S | B | G) and made at the arm's decision pose (its released state,
        else the case layout)."""
        rows, root = _locked(env, env.expand(cfg.picks.rstrip('/'), task))
        pk = json.loads(env.file(f'{root}/picks/{task.id}.json').read_text())       # the folder covers this task
        letter = PICK_ARMS[cfg.planner][-1]
        arm = 'straight' if letter == 'S' else letter
        pose = _poses(str(env.file(cfg.decisions)))[task.id]['pose'] if cfg.decisions else _layout(task.read_case())
        if not np.array_equal(np.asarray(pk['pose'], float), np.asarray(pose, float)):
            raise ValueError(f'{root}: {task.id} was planned at {pk["pose"]}, not at the decision pose {list(pose)}')
        if task.id not in rows:
            if any(e is not None for e in pk['arms'].values()):
                raise ValueError(f'{root}: {task.id} has a pick but no tasks.json row')
            return cls.of(None, arm, picks=root, reason='no valid route')
        row = rows[task.id]
        if letter not in row['arms'] or not pk['arms'].get(letter):
            raise ValueError(f'{root}: {task.id} has no {letter} pick (row arms {row["arms"]}): planner {cfg.planner}')
        route = load_route(env.path(f'{root}/routes/{row["id"]}.json'))
        if route_sha256(route) != row['sha256']:
            raise ValueError(f'{root}/routes/{row["id"]}.json: content sha256 differs from its tasks.json row')
        e = pk['arms'][letter]
        return cls.of(route, arm, e.get('z_mean'), e.get('z_pess'), picks=root, row=row)

    @classmethod
    def from_dict(cls, d):
        r = None if d['route'] is None else {**{k: np.asarray(d['route'][k], float) for k in KEYS}, 'meta': {}}
        p = cls.of(r, d['arm'], d['z_mean'], d['z_pess'], **d['record'])
        if p.route_sha256 != d['route_sha256']:
            raise ValueError(f'pick route sha256 {p.route_sha256} != recorded {d["route_sha256"]}')
        return p

    def to_dict(self) -> dict:
        r = None if self.route is None else {k: np.asarray(self.route[k], float).tolist() for k in KEYS}
        return dict(route=r, route_sha256=self.route_sha256, arm=self.arm, z_mean=self.z_mean, z_pess=self.z_pess,
                    record=self.record)


@lru_cache(maxsize=64)
def _locked_rows(root, lock_path, tasks_path, routes_dir):
    want = Path(lock_path).read_text().split()[0]
    routes = Path(routes_dir).glob('*.json')                   # ga_planner.py:602-605
    if lock_digest((p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in routes) != want:
        raise ValueError(f'{root}: route files do not recompute PICKS_LOCKED.sha256 {want[:12]}')
    rows = json.loads(Path(tasks_path).read_text())
    by = {r['group']: r for r in rows}
    if len(by) != len(rows):
        raise ValueError(f'{root}: several tasks.json rows for one group')
    return by


def _locked(env, root):
    return _locked_rows(root, str(env.file(root + '/PICKS_LOCKED.sha256')), str(env.file(root + '/tasks.json')),
                        str(env.path(root + '/routes'))), root


# ----------------------------------------------------------------------------------------------------------- plan
def plan(cfg, task, dec: Decision | None, env) -> Pick:
    """One decision of one arm: a locked pick, the given route, the straight anchor or the search (+ refine)."""
    if cfg.picks:
        return Pick.locked(cfg, task, env)
    if cfg.planner == 'given':
        if task.route is None:
            raise ValueError(f'{task.id}: planner given needs a task route')
        return Pick.of(task.read_route('route'), 'given')
    if dec is None:
        raise ValueError(f'{task.id}: planner {cfg.planner} needs a decision state')
    if cfg.planner == 'straight':          # ag_picks.straight_route: rng-free, None if the anchor is invalid
        r, i = straight(dec.base, dec.pose, cfg.speed or 6.0)
        return Pick.of(r, 'straight', index=i, decision=dec.source)
    if cfg.planner not in ('cem', 'cem_grad'):
        raise ValueError(f'planner {cfg.planner!r} does not plan here (live plans inside the drive, nav.py)')
    if task.map is None:
        raise ValueError(f'{task.id}: arena {task.arena} has no static planner map')
    numerics, seed = record_numerics(env.device), cfg.seed(task.id)
    ens = Ensemble.load(env.glob(cfg.models), env.device)
    smap = _map(str(task.map))
    if smap.sha256 != task.sha['map']:
        raise ValueError(f'{task.map}: observation.npz {smap.sha256[:12]}... is not the map the suite checked')
    score = Scorer(ens, smap, dec)
    res = optimize(dec.base, dec.pose, score, np.random.default_rng(seed), rounds=cfg.rounds, n=cfg.samples,
                   update=cfg.update, fixed_speed=cfg.speed)
    rec = dict(tag=cfg.tag, seed=seed, update=cfg.update, decision=dec.source, map_sha256=smap.sha256,
               models=[dict(path=str(p), sha256=s) for p, s in zip(ens.paths, ens.sha256)], numerics=numerics)
    if res is None:
        return Pick.of(None, PICK_ARMS[cfg.planner][-1], reason='no valid candidate', **rec)
    if cfg.planner == 'cem_grad':
        from .refine import refine_pick
        return refine_pick(res, score, cfg.grad or {}, rec)
    return Pick.of_result(res, **rec, score_calls=score.calls)


@lru_cache(maxsize=12)
def _map(path):
    return StaticMap.load(path)
