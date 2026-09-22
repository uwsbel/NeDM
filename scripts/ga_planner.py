"""Route planner for the shared rigid/CRM risk model (plan A3 / A5): the planner_arms pick pass with an ensemble of
legacy N2 checkpoints OR shared-model ('ga_train') checkpoints, both worlds scored from the OptiX static depth map,
and a history context z computed ONCE per decision.

Ensemble (--models glob, sorted): a legacy member (crm_train.py / f104_n2_train.py checkpoint: cnn.* / head.* keys,
nctx 5, no model_kind) is rebuilt with gen_riskmodel.Net exactly as gen_planner.RiskModel does and scored with the
arithmetic of f104_n2_iter.member_logits, so the one-shot and CEM picks equal planner_arms.py bit for bit. A member
with model_kind 'ga_train' (checkpoint contract in PLAN.md 'Conventions and contracts') is rebuilt with GANet below,
a copy of ga_train.GAModel's deployable part (ga_train.py is NOT imported): the n2_arch_train.GRUNet layout
(front.cnn.*, lat, ctx, tconv, mix, head) plus, for the history conditions, a causal GRU over [history | mask]
(henc: 16 -> 32) and z = tanh(hz(final hidden)) (zdim) concatenated to the context before the 32-d context embedding,
and the one-logit domain head dom for hist_aux. Width / zdim / history width are read from the tensor shapes and
checked against the checkpoint fields. Per member the route logit is log sum softplus(hazard); arms optimise the
ensemble mean (E: max). The self-test compares this loader with ga_train.load_ga_model / score on the same inputs.

Decision context per member: ctx = [(geom5 - ctx_mu) / ctx_sd | tag one-hot (rigid, crm) if cond == 'tag'];
geom5 = [goal dx, goal dy, |goal|, start yaw, route length] (gen_planner.geom_ctx). History: hist (T, 15) =
[state cols 0-6, 11-15 | applied action] at 50 ms, hmask (T,) validity, normalised with hist_mu / hist_sd, encoded
once in Scorer.__init__ and broadcast over every candidate; all-masked at startup unless --history / --from-run /
--poses provides a window (history_from_trajectory cuts the PLAN A1 window: row t = [state[j] | action[j-1]],
j = k-T+1+t, valid iff 1 <= j <= k).

Start pose: the case layout pose with base route routes/<g>/route_00.json (planner_arms behaviour), or an override
(--pose-override x y yaw | --pose-along-s t | --from-run dir --frame k | --poses json); with an override (or --goal)
the base route is gen_planner.base_route(pose, goal) and every candidate of the night-2 family starts at the pose
(the lateral basis is zero at both ends, the validator anchors at the pose).

Outputs in the planner_arms formats: picks/<g>.json, routes/<g>__<arm>.json (deduplicated by content sha256),
tasks.json ({id, group, case, route, run, tier, episode_seed, arms, sha256[, ref_id][, arena, shard][, extra]}),
tasks_new_only.json (rows equal to a reference pick get run=false), summary.json, PICKS_LOCKED.sha256; with
--cluster-prefix also tasks_cluster.json (route / case paths rewritten for the cluster root, as n2_grad_ship.py).

  python scripts/ga_planner.py --cases artifacts/traverse/crm_f104_v1/cases_eval/cases \
      --map-root artifacts/traverse/crm_f104_v1/map_root --models 'artifacts/traverse/crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt' \
      --world crm --arms A,B --out <dir>
"""
import argparse, glob, hashlib, json, os, re, shlex, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f104_n2_iter as IT
import planner_arms as PA
from gen_riskmodel import Net as LegacyNet, route_logit
GP, DS, S = IT.GP, IT.DS, IT.S
ROOT = Path(__file__).resolve().parents[1]

HIST_STATE_COLS = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]     # deployable state columns (PLAN conventions)
HIST_ACTION_COLS = [0, 1, 2]                                      # steer, throttle, brake (columns of the action array)
HIST_COLS = HIST_STATE_COLS + HIST_ACTION_COLS                    # ga_train.py's hist_cols field (12 state + 3 action)
HIST_T, HIST_DIM = 40, 15
DOMAIN_CODE = {'rigid': 0, 'crm': 1}                              # tag one-hot index (cache contract: 0 rigid, 1 crm)
GA_CONDS = ('none', 'tag', 'hist', 'hist_aux', 'hist_rma')
DT = 0.05


# ------------------------------------------------------------------------------------------------------------------
# shared-model network (n2_arch_train.GRUNet layout + history encoder); ga_train.py must write this key layout
# ------------------------------------------------------------------------------------------------------------------
class CNNFront(nn.Module):
    """Station-preserving CNN of the deployed net: (B,cin,96,32) -> (B,96,192) [mean,max over the 4 lateral cells]."""
    def __init__(self, cin):
        super().__init__()
        c = [cin, 32, 64, 64, 96]; L = []
        for i in range(4):
            L += [nn.Conv2d(c[i], c[i + 1], 3, stride=(1, 1 if i == 0 else 2), padding=1), nn.BatchNorm2d(c[i + 1]), nn.GELU()]
        self.cnn = nn.Sequential(*L)

    def forward(self, x):
        f = self.cnn(x)
        return torch.cat([f.mean(-1), f.amax(-1)], 1).transpose(1, 2)


class GANet(nn.Module):
    """CNN-GRU risk net with a causal history encoder: the same modules and state-dict keys as ga_train.GAModel
    (front.cnn.*, lat, ctx.0, tconv.0, mix, head; henc + hz for the history conditions; dom for hist_aux).

    forward(x (B,6,96,32) standardised corridor + ones plane, ctx (B,nctx) standardised [geom5 | tag one-hot],
    hist (B,T,15) standardised with masked steps zeroed, hmask (B,T) bool) -> {'haz': (B,96)[, 'z', 'dom': (B,)]}.
    encode: the mask is appended as channel 16, a GRU (16 -> 32) runs over the causal window and z = tanh(hz(final
    hidden)) (zdim). An all-masked window gives the same constant z for every decision (the trainer's startup
    representation, made in-distribution by --hist-drop). z may be passed precomputed (z=...) so the planner
    encodes once per decision. cond none / tag: no history modules, ctx MLP input = nctx.
    """
    def __init__(self, cond, cin, nctx, zdim=16, hist_dh=HIST_DIM, width=64):
        super().__init__()
        self.cond, self.zdim = cond, int(zdim)
        self.use_hist = cond in ('hist', 'hist_aux', 'hist_rma')
        zin = self.zdim if self.use_hist else 0
        self.front = CNNFront(cin); self.lat = nn.Linear(192, 96); self.ctx = nn.Sequential(nn.Linear(nctx + zin, 32), nn.GELU())
        self.tconv = nn.Sequential(nn.Conv1d(129, 96, 5, padding=2), nn.GELU(), nn.Dropout(0.1))
        self.mix = nn.GRU(96, width, num_layers=1, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * width, 1)
        if self.use_hist:
            self.henc = nn.GRU(hist_dh + 1, 32, batch_first=True); self.hz = nn.Linear(32, self.zdim)
        if cond == 'hist_aux':
            self.dom = nn.Linear(self.zdim, 1)
        self.encode_calls = 0

    def encode(self, hist, hmask):
        self.encode_calls += 1
        m = hmask.to(hist.dtype)[..., None]
        _, hn = self.henc(torch.cat([hist * m, m], -1))
        return torch.tanh(self.hz(hn[-1]))

    def backbone(self, x, ctx_full):
        f = F.gelu(self.lat(self.front(x))); B, S_, _ = f.shape
        pos = torch.linspace(0, 1, S_, device=x.device)[None, :, None].expand(B, S_, 1)
        c = self.ctx(ctx_full)[:, None, :].expand(B, S_, 32)
        h = self.tconv(torch.cat([f, c, pos], -1).transpose(1, 2)).transpose(1, 2)
        return self.head(self.mix(h)[0]).squeeze(-1)

    def forward(self, x, ctx, hist=None, hmask=None, z=None):
        out = {}
        if self.use_hist:
            if z is None:
                z = self.encode(hist, hmask)
            ctx = torch.cat([ctx, z], -1); out['z'] = z
        out['haz'] = self.backbone(x, ctx)
        if self.cond == 'hist_aux':
            out['dom'] = self.dom(z).squeeze(-1)
        return out


def standardise_history(hist, hmask, mu, sd):
    """ga_train.prep_hist in numpy: (T,15) raw -> standardised f32 with masked steps zeroed (NaN-safe)."""
    h = np.nan_to_num(np.asarray(hist, np.float32)); m = np.asarray(hmask, bool)
    h = (h - np.asarray(mu, np.float32).reshape(1, -1)) / np.asarray(sd, np.float32).reshape(1, -1)
    return np.where(m[:, None], h, 0.0).astype(np.float32), m


def check_contract(ck, path):
    """PLAN.md checkpoint contract for model_kind 'ga_train'; returns (errors, warnings)."""
    err, warn = [], []
    for k in ('cond', 'state', 'cin', 'nctx', 'zdim', 'norm', 'ctx_mu', 'ctx_sd'):
        if k not in ck:
            err.append(f'missing {k}')
    for k in ('hist_cols', 'hist_T', 'hist_mu', 'hist_sd', 'train_rows', 'split_hash'):
        if k not in ck:
            warn.append(f'missing {k}')
    if err:
        return err, warn
    if ck['cond'] not in GA_CONDS:
        err.append(f"cond {ck['cond']!r} not in {GA_CONDS}")
    if int(ck['cin']) != 6:
        err.append(f"cin {ck['cin']} != 6")
    if len(np.asarray(ck['ctx_mu']).ravel()) != 5 or len(np.asarray(ck['ctx_sd']).ravel()) != 5:
        err.append('ctx_mu / ctx_sd must have 5 entries (geometry only)')
    if int(ck['nctx']) != 5 + (2 if ck['cond'] == 'tag' else 0):
        err.append(f"nctx {ck['nctx']} != 5 (+2 for tag) for cond {ck['cond']!r}")
    zdim = int(ck['zdim'])
    if ck['cond'] in ('hist', 'hist_aux', 'hist_rma') and zdim <= 0:
        err.append(f"cond {ck['cond']!r} with zdim {zdim}")
    if ck['cond'] in ('none', 'tag') and zdim != 0:
        warn.append(f"cond {ck['cond']!r} with zdim {zdim} (history encoder present but the plan has no history here)")
    if zdim:
        for k in ('hist_mu', 'hist_sd'):
            if k in ck and len(np.asarray(ck[k]).ravel()) != HIST_DIM:
                err.append(f'{k} must have {HIST_DIM} entries')
        if 'hist_cols' in ck and list(map(int, np.asarray(ck['hist_cols']).ravel())) != HIST_COLS:
            warn.append(f"hist_cols {list(np.asarray(ck['hist_cols']).ravel())} != planner convention {HIST_COLS}")
        if 'hist_T' in ck and int(ck['hist_T']) != HIST_T:
            warn.append(f"hist_T {ck['hist_T']} != {HIST_T}")
    nm = ck['norm']
    if not (isinstance(nm, dict) and 'mu' in nm and 'sd' in nm):
        err.append('norm must be a dict with mu / sd')
    return err, warn


def build_ga(ck, path='?'):
    """Rebuild a 'ga_train' member from its checkpoint; width / layers / dom head / hist_dim from the state shapes."""
    err, warn = check_contract(ck, path)
    if err:
        raise ValueError(f'{path}: checkpoint contract violated: ' + '; '.join(err))
    st = ck['state']; cond = ck['cond']; zdim = int(ck['zdim']); nctx = int(ck['nctx']); cin = int(ck['cin'])
    if 'mix.weight_hh_l0' not in st or 'ctx.0.weight' not in st or 'head.weight' not in st:
        raise ValueError(f'{path}: state dict does not follow the ga_train.GAModel key layout (mix.*, ctx.0.*, head.*): {sorted(st)[:8]}...')
    if any(re.match(r'mix\.weight_hh_l[1-9]', k) for k in st):
        raise ValueError(f'{path}: stacked mix GRU layers are not part of the contract (one layer)')
    width = int(st['mix.weight_hh_l0'].shape[1])
    use_hist = cond in ('hist', 'hist_aux', 'hist_rma')
    if use_hist:
        if 'henc.weight_ih_l0' not in st or 'hz.weight' not in st:
            raise ValueError(f'{path}: cond {cond!r} but no henc.* / hz.* history encoder in the state dict')
        if int(st['hz.weight'].shape[0]) != zdim:
            raise ValueError(f"{path}: hz maps to {st['hz.weight'].shape[0]} != zdim {zdim}")
        hist_dh = int(st['henc.weight_ih_l0'].shape[1]) - 1
        if hist_dh != HIST_DIM:
            warn.append(f'history input width {hist_dh} != {HIST_DIM}')
        if int(st['ctx.0.weight'].shape[1]) != nctx + zdim:
            raise ValueError(f"{path}: ctx.0.weight expects {st['ctx.0.weight'].shape[1]} inputs, contract nctx + zdim = {nctx + zdim}")
    else:
        hist_dh = HIST_DIM
        if 'henc.weight_ih_l0' in st:
            raise ValueError(f'{path}: cond {cond!r} but a history encoder is present')
        if int(st['ctx.0.weight'].shape[1]) != nctx:
            raise ValueError(f"{path}: ctx.0.weight expects {st['ctx.0.weight'].shape[1]} inputs, contract nctx = {nctx}")
    if (cond == 'hist_aux') != ('dom.weight' in st):
        raise ValueError(f'{path}: dom head present={("dom.weight" in st)} does not match cond {cond!r}')
    m = GANet(cond, cin, nctx, zdim if use_hist else 0, hist_dh, width)
    res = m.load_state_dict(st, strict=False)
    if res.missing_keys:
        raise ValueError(f'{path}: state dict lacks {list(res.missing_keys)}')
    if res.unexpected_keys:
        warn.append(f'unused state entries: {sorted(res.unexpected_keys)}')
    return m, dict(zdim=zdim if use_hist else 0, width=width, layers=1, aux=(cond == 'hist_aux'), hist_dim=hist_dh, warnings=warn)


class Ensemble:
    """Sorted glob of checkpoints of either kind; every member scores every candidate (mean / max over members)."""
    def __init__(self, pattern, device=None):
        self.dev = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(pattern)
        self.members = []
        for p in paths:
            ck = torch.load(p, map_location=self.dev, weights_only=False)
            kind = ck.get('model_kind') or 'legacy'
            if kind == 'ga_train':
                m, info = build_ga(ck, p); cond = ck['cond']
                for w in info['warnings']:
                    print(f'  WARNING {os.path.basename(p)}: {w}', flush=True)
            elif kind == 'legacy':
                m = LegacyNet(ck['cin'], ck['nctx'], arch=ck['arch'], layers=ck.get('layers', 2))
                m.load_state_dict(ck['state']); cond = 'none'; info = dict(zdim=0, warnings=[])
            else:
                raise ValueError(f'{p}: model_kind {kind!r} is not loadable here (legacy N2 or ga_train only)')
            m.to(self.dev).eval()
            self.members.append(dict(path=p, name=os.path.basename(p), kind=kind, cond=cond, model=m, ck=ck, **info))
        self.kinds = sorted({m['kind'] for m in self.members}); self.conds = sorted({m['cond'] for m in self.members})
        Ts = {int(m['ck'].get('hist_T', HIST_T)) for m in self.members if m['zdim']}
        assert len(Ts) <= 1, f'members disagree on hist_T: {Ts}'
        self.hist_T = Ts.pop() if Ts else HIST_T
        self.needs_hist = any(m['zdim'] for m in self.members)
        self.needs_tag = any(m['cond'] == 'tag' for m in self.members)

    def describe(self):
        return [dict(name=m['name'], kind=m['kind'], cond=m['cond'], zdim=m['zdim'], nctx=int(m['ck']['nctx']),
                     params=int(sum(p.numel() for p in m['model'].parameters()))) for m in self.members]


# ------------------------------------------------------------------------------------------------------------------
# history windows and poses
# ------------------------------------------------------------------------------------------------------------------
def history_from_trajectory(traj, k, T=HIST_T):
    """PLAN A1 window at frame k of a recorded episode (trajectory.npz: state (n,17), action (n,3) at 50 ms):
    row t (j = k-T+1+t) = [state[j, 12 deployable cols] | action[j-1]], valid iff 1 <= j <= k (k = 0: all masked)."""
    st = np.asarray(traj['state'], np.float32); ac = np.asarray(traj['action'], np.float32); n = len(st)
    hist = np.zeros((T, HIST_DIM), np.float32); mask = np.zeros(T, bool)
    for t in range(T):
        j = k - T + 1 + t
        if 1 <= j <= min(k, n - 1):
            hist[t, :12] = st[j, HIST_STATE_COLS]; hist[t, 12:] = ac[j - 1]; mask[t] = True
    return hist, mask


def load_history(path, group=None, frame=None, T=HIST_T):
    """npz with hist (T,15) [or (n,T,15) + group] and hmask, or a trajectory.npz cut at --frame."""
    z = np.load(path, allow_pickle=True)
    if 'hist' in z.files:
        hist, mask = np.asarray(z['hist'], np.float32), np.asarray(z['hmask'], bool)
        if hist.ndim == 3:
            if hist.shape[0] == 1:
                hist, mask = hist[0], mask[0]
            else:
                assert group is not None and 'group' in z.files, f'{path}: several windows, need a group key'
                i = np.flatnonzero(z['group'].astype(str) == group)
                assert len(i) == 1, f'{path}: {len(i)} windows for {group}'
                hist, mask = hist[i[0]], mask[i[0]]
        assert hist.shape == (T, HIST_DIM) and mask.shape == (T,), (hist.shape, mask.shape, T)
        return hist, mask, 'npz'
    assert 'state' in z.files and frame is not None, f'{path}: neither hist/hmask nor a trajectory with --frame'
    h, m = history_from_trajectory(z, int(frame), T)
    return h, m, f'trajectory@{int(frame)}'


def pose_along_route(route, t_s):
    """Pose (x, y, yaw) reached after t_s seconds of the commanded profile (route_time arithmetic) and its station."""
    st = np.asarray(route['stations'], float); v = np.asarray(route['speeds'], float)
    wp = np.asarray(route['waypoints'], float); hd = np.unwrap(np.asarray(route['headings'], float))
    vm = np.maximum(0.5 * (v[1:] + v[:-1]), 0.3); t = np.r_[0.0, np.cumsum(np.diff(st) / vm)]
    s = float(np.interp(t_s, t, st))
    yaw = float(np.interp(s, st, hd)); yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    return np.array([np.interp(s, st, wp[:, 0]), np.interp(s, st, wp[:, 1]), yaw]), s


def decision_for(g, cp, a, poses, T):
    """(case, lay, pose, goal, base, hist, hmask, source) for one group: layout pose + route_00 unless overridden."""
    case, _, lay, pose, goal, base = PA.load_case(cp)
    src = dict(pose='layout', goal='case', base='route_00', history='all_masked', frame=None)
    hist = hmask = None
    entry = (poses or {}).get(g, {})
    run_dir, frame = a.from_run or entry.get('run'), int(entry.get('frame', a.frame))
    if run_dir:
        traj = np.load(Path(run_dir) / 'trajectory.npz')
        pose = np.asarray(traj['pose'][frame], float); hist, hmask = history_from_trajectory(traj, frame, T)
        src.update(pose=f'run@{frame}', history=f'run@{frame}', frame=frame, run=str(run_dir))
    if a.pose_along_s is not None:
        pose, s = pose_along_route(base, a.pose_along_s); src.update(pose=f'route_00@{a.pose_along_s:g}s', station_m=float(s))
    if a.pose_override is not None:
        pose = np.asarray(a.pose_override, float); src['pose'] = 'override'
    if entry.get('pose') is not None:
        pose = np.asarray(entry['pose'], float); src['pose'] = 'poses_file'
    if a.goal is not None:
        goal = np.asarray(a.goal, float); src['goal'] = 'override'
    if entry.get('goal') is not None:
        goal = np.asarray(entry['goal'], float); src['goal'] = 'poses_file'
    hp = a.history or entry.get('history')
    if hp:
        hist, hmask, how = load_history(hp, g, entry.get('history_frame', a.frame), T); src['history'] = f'{hp}:{how}'
    if src['pose'] != 'layout' or src['goal'] != 'case':
        base = GP.base_route(pose, goal); src['base'] = 'base_route(pose, goal)'
        src['base_meta'] = {k: base['meta'].get(k) for k in ('start_tangent_scale', 'arc_radius_m')}
        assert IT.valid(base, pose), f'{g}: no valid base route from pose {pose.round(3).tolist()} to goal {goal.round(3).tolist()}'
    return case, lay, pose, goal, base, hist, hmask, src


# ------------------------------------------------------------------------------------------------------------------
# scorer: z once per decision, broadcast over candidates; legacy arithmetic identical to f104_n2_iter.member_logits
# ------------------------------------------------------------------------------------------------------------------
class Scorer:
    """cands -> (Z (members, n), z_mean, z_pess). Corridors from the depth map (DS.init_map), rounded to float16 as
    the deployed pools were; the history is encoded here, once, per member."""
    def __init__(self, ens, start_xy, goal_xy, start_yaw, domain=None, hist=None, hmask=None, float16=True):
        self.ens, self.start_xy, self.goal_xy, self.yaw, self.f16 = ens, start_xy, goal_xy, start_yaw, float16
        self.tag = None
        if ens.needs_tag:
            assert domain in DOMAIN_CODE, f'--domain rigid|crm is required for a tag-conditioned member (got {domain!r})'
            self.tag = np.zeros(2, np.float32); self.tag[DOMAIN_CODE[domain]] = 1.0
        T = ens.hist_T
        if hist is None:
            hist, hmask = np.zeros((T, HIST_DIM), np.float32), np.zeros(T, bool)
        hist = np.asarray(hist, np.float32); hmask = np.asarray(hmask, bool)
        assert hist.shape == (T, HIST_DIM) and hmask.shape == (T,), (hist.shape, hmask.shape, T)
        self.z, self.history = {}, dict(T=T, n_valid=int(hmask.sum()), members={})
        with torch.no_grad():
            for i, mem in enumerate(ens.members):
                if not mem['zdim']:
                    continue
                ck = mem['ck']
                hn, hm = standardise_history(hist, hmask, ck['hist_mu'], ck['hist_sd'])
                z = mem['model'].encode(torch.tensor(hn[None], device=ens.dev), torch.tensor(hm[None], device=ens.dev))
                self.z[i] = z
                d = dict(z_norm=float(z.norm()))
                if mem['cond'] == 'hist_aux':
                    d['p_crm'] = float(torch.sigmoid(mem['model'].dom(z))[0, 0])      # domain head: 1 = crm
                self.history['members'][mem['name']] = d
        self.history['n_encode'] = len(self.z)
        self.calls = 0

    def __call__(self, cands):
        X, L = IT.corridors(cands)
        if self.f16:
            X = X.astype(np.float16).astype(np.float32)
        geom = GP.geom_ctx(self.start_xy, self.goal_xy, self.yaw, L)
        Z = self.member_logits(X, geom); self.calls += 1
        return Z, Z.mean(0), Z.max(0)

    def member_logits(self, X, geom5, bs=256):
        dev = self.ens.dev; zs = []
        for i, mem in enumerate(self.ens.members):
            m, ck = mem['model'], mem['ck']; nm = ck['norm']
            cont = list(nm.get('cont_index', [0, 1, 2, 3])); k = len(cont)
            mu = np.asarray(nm['mu'], np.float32); sd = np.asarray(nm['sd'], np.float32)
            cmu = np.asarray(ck['ctx_mu'], np.float32).reshape(-1); csd = np.asarray(ck['ctx_sd'], np.float32).reshape(-1)
            out = []
            for j in range(0, len(X), bs):
                x = X[j:j + bs].astype(np.float32).copy()
                if cont == list(range(k)):
                    x[:, :k] = (x[:, :k] - mu[None, :, None, None]) / sd[None, :, None, None]
                else:
                    x[:, cont] = (x[:, cont] - mu[None, :, None, None]) / sd[None, :, None, None]
                x = np.concatenate([x, np.ones((len(x), 1, x.shape[2], x.shape[3]), np.float32)], 1)
                c = (geom5[j:j + bs] - cmu) / csd
                if mem['cond'] == 'tag':
                    c = np.concatenate([c, np.broadcast_to(self.tag, (len(c), 2))], 1)
                xt = torch.tensor(x, device=dev); ct = torch.tensor(c, dtype=torch.float32, device=dev)
                with torch.no_grad():
                    if mem['kind'] == 'ga_train':
                        z = self.z[i].expand(len(x), -1) if i in self.z else None
                        haz = m(xt, ct, z=z)['haz']
                    else:
                        haz = m(xt, ct)
                    out.append(route_logit(haz).double().cpu().numpy())
            zs.append(np.concatenate(out))
        return np.stack(zs)


# ------------------------------------------------------------------------------------------------------------------
# test scaffolding: a random-weight checkpoint that follows the ga_train contract
# ------------------------------------------------------------------------------------------------------------------
def random_checkpoint(path, cond='hist_aux', seed=0, zdim=16, width=64, hist_T=HIST_T):
    """Random-weight checkpoint with the fields ga_train.checkpoint_dict writes (self-tests only)."""
    torch.manual_seed(int(seed))
    nctx = 5 + (2 if cond == 'tag' else 0); use_hist = cond in ('hist', 'hist_aux', 'hist_rma')
    m = GANet(cond, 6, nctx, zdim if use_hist else 0, HIST_DIM, width)
    rng = np.random.default_rng(int(seed))
    ck = dict(model_kind='ga_train', cond=cond, state={k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, cin=6, nctx=nctx,
              zdim=int(zdim) if use_hist else 0, width=int(width), hist_cols=list(HIST_COLS), hist_cols_state=list(HIST_STATE_COLS),
              hist_cols_action=list(HIST_ACTION_COLS), hist_T=int(hist_T),
              hist_layout='hist[:, t, :12] = state[k-T+1+t, hist_cols_state]; hist[:, t, 12:] = action[k-T+t, hist_cols_action]; hmask True where the frame exists',
              norm=dict(mu=np.array([0.61, 0.007, 0.004, 3.1], np.float32), sd=np.array([1.01, 0.18, 0.19, 1.74], np.float32),
                        cont_index=[0, 1, 2, 3], channels=['elev_rel', 'grade', 'cross', 'speed', 'valid']),
              ctx_mu=np.array([7.6, 1.9, 44.3, -0.13, 45.5], np.float32), ctx_sd=np.array([32.0, 31.3, 10.2, 1.63, 10.3], np.float32), geom_cols=[17, 18, 19, 20, 21],
              hist_mu=rng.normal(0, 0.5, HIST_DIM).astype(np.float32), hist_sd=(0.5 + rng.random(HIST_DIM)).astype(np.float32),
              train_rows=0, split_hash='random-selftest', split_eval='val', mode='deploy', domain_filter='both', startup_only=False,
              hist_drop=0.2, seed=int(seed), tag=f'random_{cond}', ds='', args={})
    torch.save(ck, path)
    return ck


# ------------------------------------------------------------------------------------------------------------------
def parse_groups(spec):
    if not spec:
        return None
    if spec.startswith('@'):
        return {s.strip() for s in open(spec[1:]) if s.strip()}
    return {s.strip() for s in spec.split(',') if s.strip()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', help='dir with <group>.json and routes/<group>/route_00.json')
    ap.add_argument('--map-root', help='root with static_map_v1 (OptiX depth map, f104_n2_dataset.init_map); BOTH worlds')
    ap.add_argument('--models', help='glob of ensemble checkpoints (legacy N2 or ga_train)')
    ap.add_argument('--out')
    ap.add_argument('--world', choices=['crm', 'rigid'])
    ap.add_argument('--domain', choices=['crm', 'rigid'], help='tag one-hot for cond=tag members (default: --world)')
    ap.add_argument('--arms', default='A,B')
    ap.add_argument('--fixed2', action='store_true', help='geometry-only family at 2 m/s (theta in R^3)')
    ap.add_argument('--ref-picks', help='reference picks dir (planner_arms / eval_v1 format); default crm: eval_v1/picks')
    ap.add_argument('--ref-arms', help="'myarm:refarm,...' e.g. A:crm,B:B (default A:crm for crm, none for rigid)")
    ap.add_argument('--task-root', help='tasks.json paths are relative to this (default: two levels above --cases)')
    ap.add_argument('--arena-tag', default='f104', help='rigid tasks: arena tag')
    ap.add_argument('--shards', type=int, default=6, help='rigid tasks: shard = md5(group) %% shards')
    ap.add_argument('--verify', type=int, default=3, help='groups on which the one-shot pool is checked route by route')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--groups', help='comma list of group ids or @file (one per line)')
    ap.add_argument('--no-float16', action='store_true')
    ap.add_argument('--device')
    # decision context
    ap.add_argument('--history', help='npz with hist (T,15) + hmask (T,) [+ group], or a trajectory.npz cut at --frame')
    ap.add_argument('--frame', type=int, default=60, help='frame for --from-run / trajectory histories')
    ap.add_argument('--from-run', help='run dir with trajectory.npz: pose = recorded pose at --frame, history cut there')
    ap.add_argument('--pose-override', type=float, nargs=3, metavar=('X', 'Y', 'YAW'))
    ap.add_argument('--pose-along-s', type=float, help='pose = point t seconds along route_00 of the case')
    ap.add_argument('--goal', type=float, nargs=2, metavar=('GX', 'GY'), help='goal override (default: the case goal)')
    ap.add_argument('--poses', help='json {group: {pose: [x,y,yaw], goal?: [gx,gy], history?: npz, run?: dir, frame?: k}}; plans those groups only')
    # shipping
    ap.add_argument('--cluster-prefix', help='write tasks_cluster.json with route = <prefix>/<id>.json')
    ap.add_argument('--cluster-case-prefix', help='... and case = <prefix>/<group>.json (default: unchanged)')
    ap.add_argument('--cluster-all', action='store_true', help='tasks_cluster.json from all rows, not only the new drives')
    ap.add_argument('--extra-args', help="extra collector arguments, stored as the row's 'extra' list (shlex split)")
    # test scaffolding
    ap.add_argument('--write-random-ckpt', help='write a random-weight ga_train-contract checkpoint here and exit')
    ap.add_argument('--cond', default='hist_aux'); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--zdim', type=int, default=16)
    a = ap.parse_args()

    if a.write_random_ckpt:
        ck = random_checkpoint(a.write_random_ckpt, a.cond, a.seed, a.zdim)
        print(f"wrote {a.write_random_ckpt}: cond {ck['cond']} nctx {ck['nctx']} zdim {ck['zdim']} {len(ck['state'])} tensors")
        return
    for k in ('cases', 'map_root', 'models', 'out', 'world'):
        assert getattr(a, k), f'--{k.replace("_", "-")} is required'
    t_start = time.time()
    DS.init_map(a.map_root)
    domain = a.domain or a.world
    fixed_speed = 2.0 if a.fixed2 else None
    suffix = '_fixed2' if a.fixed2 else ''
    deployed_tag = ('crm_fixed2' if a.fixed2 else 'crm_proposal') if a.world == 'crm' else ('gen_fixed2' if a.fixed2 else 'gen_night2')
    ref_dir = Path(a.ref_picks) if a.ref_picks else (ROOT / 'artifacts/traverse/crm_f104_v1/eval_v1/picks' if a.world == 'crm' else None)
    ref_ok = bool(ref_dir and ref_dir.is_dir())
    ref_arms = {}
    if a.ref_arms:
        ref_arms = dict(s.split(':') for s in a.ref_arms.split(',') if s.strip())
    elif a.world == 'crm' and not a.ref_picks:
        ref_arms = {'A': 'crm_fixed2' if a.fixed2 else 'crm'}
    print(f'world {a.world} domain {domain} fixed2={a.fixed2} deployed tag {deployed_tag!r}; reference picks {ref_dir} '
          f'{ref_arms} ({"found" if ref_ok else "MISSING"})', flush=True)

    ens = Ensemble(a.models, a.device)
    print(f'{len(ens.members)} members from {a.models} on {ens.dev}: kinds {ens.kinds} conds {ens.conds} hist_T {ens.hist_T}', flush=True)
    for d in ens.describe():
        print('  ', d, flush=True)
    specs = PA.arm_specs(deployed_tag, suffix)
    arms = [s.strip() for s in a.arms.split(',') if s.strip()]
    for arm in arms:
        assert arm in specs, arm
    out = Path(a.out); (out / 'routes').mkdir(parents=True, exist_ok=True); (out / 'picks').mkdir(exist_ok=True)
    cases = sorted(str(p) for p in Path(a.cases).glob('*.json') if p.name != 'cases.json')
    poses = json.load(open(a.poses)) if a.poses else None
    want = parse_groups(a.groups)
    if poses:
        want = set(poses) if want is None else want & set(poses)
    if want is not None:
        cases = [c for c in cases if Path(c).stem in want]
        missing = want - {Path(c).stem for c in cases}
        if missing:
            print(f'  WARNING {len(missing)} requested groups have no case file: {sorted(missing)[:5]}', flush=True)
    if a.limit:
        cases = cases[:a.limit]
    single = a.pose_override is not None or a.pose_along_s is not None or a.from_run or a.history or a.goal is not None
    assert not single or len(cases) == 1, f'--pose-override/--pose-along-s/--from-run/--history/--goal apply to ONE group (got {len(cases)}); use --poses for a batch'
    task_root = Path(a.task_root).resolve() if a.task_root else Path(a.cases).resolve().parents[1]
    extra = shlex.split(a.extra_args) if a.extra_args else None

    tasks, rows, wall = [], [], {arm: 0.0 for arm in arms}
    ref_match = {'checked': 0, 'match': 0, 'mismatch': [], 'per_arm': {arm: [0, 0] for arm in arms}}
    verified = []
    for gi, cp in enumerate(cases):
        g = Path(cp).stem
        case, lay, pose, goal, base, hist, hmask, src = decision_for(g, cp, a, poses, ens.hist_T)
        scorer = Scorer(ens, pose[:2], goal, float(pose[2]), domain, hist, hmask, float16=not a.no_float16)
        if gi < a.verify and src['base'] == 'route_00':
            verified.append((g, IT.selftest(base, pose, IT.seed(g, deployed_tag), n=256, fixed_speed=fixed_speed)))
        refs = {}
        if ref_ok and (ref_dir / f'{g}.json').exists():
            rj = json.load(open(ref_dir / f'{g}.json'))['arms']
            refs = {arm: rj.get(ra) for arm, ra in ref_arms.items() if rj.get(ra)}
        picks, seen = {}, {}
        summ = dict(group=g, world=a.world, domain=domain, fixed2=a.fixed2, stratum=case.get('evaluation_stratum'),
                    base_length_m=float(base['stations'][-1]), corridors_batched=None, arms={},
                    pose=[float(v) for v in pose], goal=[float(v) for v in goal], source=src, history=scorer.history)
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
                json.dump(PA.route_json(r, rid, g, arm, sp['label'], a.world), open(out / 'routes' / f'{rid}.json', 'w'))
                row = dict(id=rid, group=g, case=os.path.relpath(Path(cp).resolve(), task_root),
                           route=os.path.relpath((out / 'routes' / f'{rid}.json').resolve(), task_root), run=True, tier=gi,
                           episode_seed=int(hashlib.md5(rid.encode()).hexdigest()[:8], 16), arms=[arm], sha256=h)
                if a.world == 'rigid':
                    row.update(arena=a.arena_tag, shard=int(hashlib.md5(g.encode()).hexdigest(), 16) % a.shards)
                if extra is not None:
                    row['extra'] = list(extra)
                tasks.append(row)
            else:
                next(t for t in tasks if t['id'] == rid)['arms'].append(arm)
            entry = dict(route_id=rid, arm=arm, label=sp['label'], tag=sp['tag'], objective=sp['objective'], index=res['index'],
                         kind=res['kind'], round=res['round'], theta=res['theta'], z_mean=res['z_mean'], z_pess=res['z_pess'],
                         P=res['P'], T=res['T'], J=res['J'], mean_speed=float(np.asarray(r['speeds'])[1:-1].mean()),
                         max_lateral_m=float(r['meta'].get('max_lateral_m', 0.0)), length_m=float(np.asarray(r['stations'])[-1]),
                         route_sha256=h, n_evaluated=res['n_evaluated'], tries=res['tries'],
                         n_below_1pct=int(((1 - np.exp(-np.exp(res['Z_mean']))) < 0.01).sum()),
                         z_mean_min=float(res['Z_mean'].min()), z_pess_min=float(res['Z_pess'].min()), wall_s=None,
                         start_xy=[float(v) for v in np.asarray(r['waypoints'])[0]],
                         start_dist_to_pose_m=float(np.linalg.norm(np.asarray(r['waypoints'])[0] - pose[:2])))
            if sp['kind'] == 'iter':
                entry.update(log=res['log'], mu_final=res['mu'], sd_final=res['sd'],
                             kinds_count={k: int(sum(1 for q in res['kinds'] if q == k)) for k in ('anchor', 'sample', 'mean')})
            ref = refs.get(arm)
            if ref is not None:
                ok = (ref['route_sha256'] == h) if ref.get('route_sha256') else (int(ref['index']) == res['index'])
                ref_match['checked'] += 1; ref_match['match'] += int(ok)
                ref_match['per_arm'][arm][0] += 1; ref_match['per_arm'][arm][1] += int(ok)
                if not ok:
                    ref_match['mismatch'].append(dict(group=g, arm=arm, ref_index=ref.get('index'), index=res['index'],
                                                      ref_logit=ref.get('logit', ref.get('z_mean', ref.get(f"logit_{ref_arms.get(arm)}"))), z_mean=res['z_mean']))
                entry.update(ref_index=ref.get('index'), ref_match=ok, ref_route_id=ref['route_id'])
                if ok:
                    next(t for t in tasks if t['id'] == rid)['ref_id'] = ref['route_id']
            if arm == 'A':
                zA = res['z_mean']
            elif zA is not None:
                entry.update(delta_z_vs_A=res['z_mean'] - zA, frac_evaluated_below_A=float((res['Z_mean'] < zA).mean()),
                             same_route_as_A=(picks['A']['route_sha256'] == h) if 'A' in picks else None)
            picks[arm] = entry
        summ['arms'] = picks; summ['corridors_batched'] = IT._CORR.get('identical'); summ['score_calls'] = scorer.calls
        json.dump(summ, open(out / 'picks' / f'{g}.json', 'w'), indent=1)
        rows.append(summ)
        if (gi + 1) % 10 == 0 or gi == len(cases) - 1:
            el = time.time() - t_start
            print(f'  {gi + 1}/{len(cases)}  {el:.0f}s  ' + ' '.join(f'{arm}={picks[arm]["z_mean"]:.2f}' for arm in arms if picks.get(arm))
                  + f'  ref {ref_match["match"]}/{ref_match["checked"]}  hist n_valid {scorer.history["n_valid"]} encodes {scorer.history["n_encode"]}', flush=True)

    json.dump(tasks, open(out / 'tasks.json', 'w'), indent=1)
    new_only = [dict(t, run=('ref_id' not in t)) for t in tasks]
    json.dump(new_only, open(out / 'tasks_new_only.json', 'w'), indent=1)
    if a.cluster_prefix:
        ship = []
        for t in (tasks if a.cluster_all else new_only):
            if not t['run']:
                continue
            row = dict(t, route=f"{a.cluster_prefix}/{t['id']}.json",
                       file_sha256=hashlib.sha256((out / 'routes' / f"{t['id']}.json").read_bytes()).hexdigest())   # bytes, as n2_grad_ship
            if a.cluster_case_prefix:
                row['case'] = f"{a.cluster_case_prefix}/{t['group']}.json"
            ship.append(row)
        json.dump(ship, open(out / 'tasks_cluster.json', 'w'), indent=1)
    lock = hashlib.sha256()
    for p in sorted((out / 'routes').glob('*.json')):
        lock.update(p.name.encode()); lock.update(hashlib.sha256(p.read_bytes()).digest())
    (out / 'PICKS_LOCKED.sha256').write_text(lock.hexdigest() + '  routes/*.json (name + content, sorted)\n')

    def col(arm, key):
        return np.array([r['arms'][arm][key] for r in rows if r['arms'].get(arm) is not None], float)
    encode_total = int(sum(m['model'].encode_calls for m in ens.members if m['kind'] == 'ga_train'))
    n_hist_members = int(sum(1 for m in ens.members if m['zdim']))
    summary = dict(world=a.world, domain=domain, fixed2=a.fixed2, n_groups=len(rows), arms=arms, tags={arm: specs[arm]['tag'] for arm in arms},
                   deployed_tag=deployed_tag, models=a.models, members=ens.describe(), model_kinds=ens.kinds, conds=ens.conds,
                   map_root=a.map_root, rigid_arena=None, n_distinct_routes=len(tasks), n_new_drives=int(sum(t['run'] for t in new_only)),
                   ref=dict(dir=str(ref_dir), arms=ref_arms, checked=ref_match['checked'], match=ref_match['match'],
                            per_arm={k: dict(checked=v[0], match=v[1]) for k, v in ref_match['per_arm'].items()}, mismatches=ref_match['mismatch']),
                   selftest_groups=verified, corridors_batched=IT._CORR.get('identical'), float16_scoring=not a.no_float16,
                   history=dict(T=ens.hist_T, members_with_history=n_hist_members, encode_calls_total=encode_total,
                                expected_encode_calls=n_hist_members * len(rows), n_valid_per_group={r['group']: r['history']['n_valid'] for r in rows},
                                score_calls_total=int(sum(r['score_calls'] for r in rows))),
                   pose_sources=sorted({r['source']['pose'] for r in rows}), base_sources=sorted({r['source']['base'] for r in rows}),
                   extra=extra, cluster_prefix=a.cluster_prefix, wall_s_total=time.time() - t_start, per_arm={})
    assert encode_total == n_hist_members * len(rows), f'history encoded {encode_total} times, expected once per member per decision ({n_hist_members * len(rows)})'
    zA = col('A', 'z_mean') if 'A' in arms else None
    for arm in arms:
        z, zp, P, T = col(arm, 'z_mean'), col(arm, 'z_pess'), col(arm, 'P'), col(arm, 'T')
        if not len(z):
            summary['per_arm'][arm] = dict(n=0); continue
        d = dict(n=int(len(z)), z_mean_mean=float(z.mean()), z_mean_median=float(np.median(z)), z_pess_mean=float(zp.mean()),
                 z_pess_median=float(np.median(zp)), P_mean=float(P.mean()), P_median=float(np.median(P)),
                 T_mean=float(T.mean()), T_median=float(np.median(T)), mean_speed=float(col(arm, 'mean_speed').mean()),
                 max_lateral_mean=float(col(arm, 'max_lateral_m').mean()), n_evaluated_mean=float(col(arm, 'n_evaluated').mean()),
                 start_dist_to_pose_max_m=float(col(arm, 'start_dist_to_pose_m').max()),
                 wall_s=wall[arm], wall_s_per_group=wall[arm] / max(len(rows), 1),
                 pick_kind={k: int(sum(1 for r in rows if r['arms'].get(arm) and r['arms'][arm]['kind'] == k)) for k in ('anchor', 'sample', 'mean')},
                 pick_round={str(k): int(sum(1 for r in rows if r['arms'].get(arm) and r['arms'][arm]['round'] == k))
                             for k in sorted({r['arms'][arm]['round'] for r in rows if r['arms'].get(arm)})})
        if zA is not None and arm != 'A' and len(z) == len(zA):
            dz = z - zA
            d.update(delta_z_vs_A_mean=float(dz.mean()), delta_z_vs_A_median=float(np.median(dz)),
                     improved_vs_A=int((dz < -1e-9).sum()), worse_vs_A=int((dz > 1e-9).sum()), frac_improved_vs_A=float((dz < -1e-9).mean()),
                     same_route_as_A=int(sum(1 for r in rows if r['arms'].get(arm) and r['arms'][arm].get('same_route_as_A'))),
                     P_delta_vs_A_mean=float((P - col('A', 'P')).mean()), T_delta_vs_A_mean=float((T - col('A', 'T')).mean()))
        summary['per_arm'][arm] = d
    agree = {}
    for i, x in enumerate(arms):
        for y in arms[i + 1:]:
            agree[f'{x}={y}'] = int(sum(1 for r in rows if r['arms'].get(x) and r['arms'].get(y) and r['arms'][x]['route_sha256'] == r['arms'][y]['route_sha256']))
    summary['agreement'] = agree
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('per_arm', 'ref', 'members', 'history')}, indent=None))
    for arm in arms:
        d = summary['per_arm'][arm]
        if d.get('n'):
            print(f"{arm} {specs[arm]['label']:16s} z_mean {d['z_mean_mean']:.3f} (med {d['z_mean_median']:.3f}) z_pess {d['z_pess_mean']:.3f} "
                  f"P {d['P_mean']:.4f} T {d['T_mean']:.1f}s v {d['mean_speed']:.2f} start-pose dist max {d['start_dist_to_pose_max_m']:.3f} m" +
                  (f"  dz vs A {d['delta_z_vs_A_mean']:+.3f} improved {d['improved_vs_A']}/{d['n']} same {d['same_route_as_A']}" if 'delta_z_vs_A_mean' in d else '')
                  + f"  wall {d['wall_s_per_group']:.2f}s/group")
    print(f"reference check: {ref_match['match']}/{ref_match['checked']} picks equal {ref_arms} in {ref_dir}; {len(tasks)} distinct routes, "
          f"{summary['n_new_drives']} new drives; agreement {agree}; history encodes {encode_total} (= {n_hist_members} members x {len(rows)} decisions)")


if __name__ == '__main__':
    main()
