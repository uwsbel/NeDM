#!/usr/bin/env python
"""Shared, torch-free controller helpers for the external-control collectors (PLAN stages B3 and A4).

Both collectors with new modes (``crm_collect_ext.py`` for deformable soil, ``gen_collect_ext.py`` for rigid
terrain) import this module inside the Chrono process. It therefore imports only numpy and the numpy-only route
code of the repo (``f104_n2_sampler``, ``nedm.traverse.fdm_mppi``); torch is imported in exactly one function
(``export_torch_actor``) and in the self-test.  ``gen_planner`` and ``nedm.traverse.fdm_diverse_planner`` are NOT
imported at module level because both load torch (checked 2026-09-21): the planner's validator config is rebuilt
with the same arguments (``gen_planner.py:30``) and the 20-line reference-contract check is copied
(``fdm_diverse_planner.py:16-37``); the self-test asserts both agree with the originals.

Contents
  hold_clip              one clip per 50 ms frame: steering within +-rate of the previous frame, throttle/brake in
                         [0, 1]; the returned triple is written unchanged at every physics substep of the frame.
                         It REPLACES the frozen loop's per-substep ``previous_steer +- 2*dt`` clamp for that frame:
                         a collector that keeps the frozen clamp after hold_clip would ramp a 0.1 steering step over
                         the 25/50 substeps and break the held-triple contract (substep audit max-min = 0).
  OUPerturb              bounded Ornstein-Uhlenbeck perturbation around the shadow follower's command, with brake
                         taps (default tap rate = the PLAN B3 sizing, ~1.4 taps per 20 s episode); throttle and
                         brake are never both positive; deterministic given the seed.
  NumpyActor             an rsl_rl ActorCritic actor + EmpiricalNormalization exported to npz and evaluated in numpy.
  export_torch_actor     writes that npz from a torch ActorCritic (torch imported inside the function only).
  sample_continuations   k routes of the night-2 route family from a pose to the goal, validated like the planner;
                         with ``v0`` (the vehicle's speed at the branch) the speed profile is floored at a 2 m/s^2
                         deceleration ramp from v0, and a start-heading acceptance (default 15 deg) rejects draws
                         whose first tangent kinks away from the vehicle's heading.
  make_follower          the frozen collector's ChPathFollowerDriver construction, with ``initialize=False`` for the
                         branch swap.  The frozen ``run`` asserts the route start within 0.25 m of the layout start;
                         a branch route starts at the branch pose, so the collector must bypass that check for the
                         branch leg (the prefix route is checked as before).
  PolicyObs              the 38-d WP3 tracking observation plus the past-8-actions and past-8-observable-states
                         blocks, with the frame-0 padding convention; ``history_blocks`` is the offline twin.

Timing convention (PLAN 'Conventions', collector_timing.md section 1): the recorded ``action[k]`` is the triple held
over interval k; ``state[k]`` / ``pose[k]`` are captured at substep 0 of interval k (positions and velocities there
equal the values before ``Synchronize``).  A controller at the top of frame k sees pose[k], state[k] and the last
held triple action[k-1]; its output, after ``hold_clip`` against action[k-1][0], becomes action[k].

Self-test (no Chrono, < 2 min):
  PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python scripts/gc_control.py --selftest \
      --out artifacts/traverse/generalist_20260921/C_collectors/selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (str(ROOT / "src"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CTRL_DT_S = 0.05
STEER_RATE_PER_FRAME = 0.1          # the collectors' 2.0 full-scale/s steering clamp per 50 ms frame
SETTLE_ACTION = (0.0, 0.0, 1.0)     # the settle's command; also fdm_data.build_history's pre-anchor padding
ACTION_LOW = (-1.0, 0.0, 0.0)
ACTION_HIGH = (1.0, 1.0, 1.0)
# Deployable state columns of the 17-column tire_normal_force_omega_pt preset (PLAN 'Conventions'):
# 0-6 vx, vy, roll, pitch, roll rate, pitch rate, yaw rate; 11-14 spindle omegas; 15 engine speed.
OBSERVABLE_COLS = (0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15)
VX_COL, YAW_RATE_COL = 0, 6
DEFAULT_TRACK = {"search_window": 40, "preview_points": 10, "preview_spacing_m": 1.0}  # wp3_tracker_v1's values
GOAL_TOL_M = 0.25


# =============================================================================================== (1) hold_clip
def hold_clip(cmd, prev_steer: float, rate: float = STEER_RATE_PER_FRAME) -> tuple[float, float, float]:
    """Clip a controller command ONCE per frame; the result is the triple written at every substep of the frame.

    ``cmd`` = (steer, throttle, brake) as produced by a policy, the perturbed follower or a schedule.
    ``prev_steer`` = the steering of the triple held over the previous frame (the recorded ``action[k-1][0]``; at
    frame 0 the settle's 0.0).  Steering is clipped to [prev - rate, prev + rate] and then to [-1, 1]; throttle and
    brake to [0, 1].  ``rate`` 0.1 per 50 ms frame equals the collectors' 2.0 full-scale/s per-substep clamp
    (``traverse_fdm_rgbd_diverse_chrono.py:224``, ``crm_collect.py:238``) accumulated over one frame, so a held
    external command can never change steering faster than the native follower could.

    Non-finite inputs raise (a NaN command is a controller bug and must not be driven).  Returns Python floats.
    """
    s, t, b = (float(v) for v in cmd)
    if not (math.isfinite(s) and math.isfinite(t) and math.isfinite(b)):
        raise ValueError(f"non-finite command {cmd!r}")
    p = float(prev_steer)
    if not (math.isfinite(p) and -1.0 <= p <= 1.0):
        raise ValueError(f"previous steering {p!r} outside [-1, 1]")
    if rate < 0:
        raise ValueError("rate must be non-negative")
    s = min(max(s, p - rate), p + rate)
    s = min(max(s, -1.0), 1.0)
    t = min(max(t, 0.0), 1.0)
    b = min(max(b, 0.0), 1.0)
    return (s, t, b)


# =============================================================================================== (2) OUPerturb
def brake_p_for_rate(taps_per_s: float, dt: float = CTRL_DT_S, brake_len_s=(0.5, 1.0)) -> float:
    """The per-frame ``brake_p`` that yields ``taps_per_s`` tap starts per second of driving.

    Taps start only when no tap is active, so tap starts are a renewal process with mean cycle
    ``1 / p + L`` frames (geometric wait + mean tap length ``L`` frames); rate ``r = p / (1 + p L)`` per frame,
    hence ``p = r / (1 - r L)``.  PLAN B3 sizes the perturbed collection at >= 2,000 brake onsets per world from
    1,500 episodes (>= 1.33 per episode): for 20 s episodes that is ``taps_per_s`` 0.067 -> ``brake_p`` 0.0037,
    which is ``OUPerturb``'s default (``DEFAULT_BRAKE_P``).  Scale ``taps_per_s`` to the actual episode length.
    (Cycle ``1/p + L`` instead of ``1/p + L - 1``: +0.2 % bias at L = 15, harmless.)  A tap that starts while the
    follower is already braking is not a brake ONSET in the recorded action (~15 % of taps in the self-test stream),
    so the collectors must count onsets from the recorded action, not taps.
    """
    L = 0.5 * (float(brake_len_s[0]) + float(brake_len_s[1])) / float(dt)
    r = float(taps_per_s) * float(dt)
    if r <= 0:
        return 0.0
    if r * L >= 1.0:
        raise ValueError("requested tap rate exceeds what back-to-back taps can deliver")
    return r / (1.0 - r * L)


DEFAULT_BRAKE_P = brake_p_for_rate(1.4 / 20.0)      # 0.0037 per frame: ~1.4 taps per 20 s episode (PLAN B3)


class OUPerturb:
    """Bounded Ornstein-Uhlenbeck perturbation of a follower command, one call per 50 ms frame.

    Two OU states (steering, throttle) with correlation time ``tau_s`` and stationary standard deviations
    ``steer_sd`` / ``throttle_sd`` are advanced with the exact discretisation
    ``x <- a x + sd sqrt(1 - a^2) N(0, 1)``, ``a = exp(-dt / tau)``, and the applied perturbation is the state
    clipped to ``+-bound_sds * sd`` (default bound_sds = 1: steer within +-0.15, throttle within +-0.25, the PLAN B3
    numbers; the OU state itself is not clipped, so the process keeps its dynamics at the bound).

    Brake taps: at every frame without an active tap a tap starts with probability ``brake_p`` (PER FRAME).  The
    default ``DEFAULT_BRAKE_P`` = ``brake_p_for_rate(1.4 / 20.0)`` = 0.0037 gives ~1.4 tap starts per 20 s episode
    (the PLAN B3 sizing, >= 1.33 brake onsets per episode); 0.05 per frame would mean one tap per second of
    un-braked driving (~12 taps and 46 % braked time per 20 s, self-test numbers) and is NOT the plan's reading.
    A tap's length is uniform in ``brake_len_s`` (rounded to whole frames, at least one) and its level uniform in
    ``brake_level``.

    Exclusivity rule (throttle and brake never both > 0):
      tap active         -> throttle 0, brake = max(follower brake, tap level)
      follower braking   -> throttle 0, brake = follower brake + throttle perturbation (reused as brake noise)
      otherwise          -> throttle = follower throttle + perturbation, brake 0
    Steering = follower steering + perturbation.  Every channel is clipped to its box here; the per-frame
    steering-rate clip is NOT applied here: the collector passes the result through ``hold_clip``.

    Determinism: one ``numpy.random.default_rng(seed)`` stream with a fixed draw order per frame (2 normals, 1
    uniform, and 2 uniforms when a tap starts), so two instances with the same seed and the same follower commands
    produce identical perturbations.  ``step`` returns a float64 (3,) array; ``last`` holds the perturbation
    components for logging.
    """

    def __init__(self, seed: int, dt: float = CTRL_DT_S, tau_s: float = 0.5, steer_sd: float = 0.15,
                 throttle_sd: float = 0.25, brake_p: float = DEFAULT_BRAKE_P, brake_len_s=(0.5, 1.0),
                 brake_level=(0.3, 1.0), bound_sds: float = 1.0):
        if tau_s <= 0 or dt <= 0 or not 0.0 <= brake_p <= 1.0:
            raise ValueError("tau_s and dt must be positive, brake_p in [0, 1]")
        self.seed, self.dt, self.tau = int(seed), float(dt), float(tau_s)
        self.sd = np.array([steer_sd, throttle_sd], float)
        self.bound = float(bound_sds) * self.sd
        self.brake_p = float(brake_p)
        self.brake_len_s = (float(brake_len_s[0]), float(brake_len_s[1]))
        self.brake_level = (float(brake_level[0]), float(brake_level[1]))
        self.reset()

    def reset(self) -> None:
        """Restart the stream from the seed (a new episode with the same seed replays the same perturbation)."""
        self.rng = np.random.default_rng(self.seed)
        self.x = np.zeros(2)
        self.tap_left, self.tap_level = 0, 0.0
        self.frames, self.n_taps, self.braked_frames = 0, 0, 0
        self.last = {"d_steer": 0.0, "d_throttle": 0.0, "tap_active": False, "tap_level": 0.0}

    def step(self, follower_cmd) -> np.ndarray:
        f = np.asarray(follower_cmd, float).reshape(3)
        if not np.isfinite(f).all():
            raise ValueError(f"non-finite follower command {follower_cmd!r}")
        a = math.exp(-self.dt / self.tau)
        self.x = a * self.x + self.sd * math.sqrt(1.0 - a * a) * self.rng.normal(size=2)
        d = np.clip(self.x, -self.bound, self.bound)
        u = self.rng.uniform()                                   # always drawn: keeps the stream aligned
        if self.tap_left == 0 and u < self.brake_p:
            length = self.rng.uniform(*self.brake_len_s)
            self.tap_level = float(self.rng.uniform(*self.brake_level))
            self.tap_left = max(1, int(round(length / self.dt)))
            self.n_taps += 1
        tap = self.tap_left > 0
        steer = min(max(f[0] + d[0], -1.0), 1.0)
        if tap:
            throttle, brake = 0.0, min(max(max(f[2], self.tap_level), 0.0), 1.0)
            self.tap_left -= 1
        elif f[2] > 0.0:
            throttle, brake = 0.0, min(max(f[2] + d[1], 0.0), 1.0)
        else:
            throttle, brake = min(max(f[1] + d[1], 0.0), 1.0), 0.0
        self.frames += 1
        self.braked_frames += int(brake > 0.0)
        self.last = {"d_steer": float(d[0]), "d_throttle": float(d[1]), "tap_active": bool(tap),
                     "tap_level": float(self.tap_level if tap else 0.0)}
        return np.array([steer, throttle, brake], float)

    def summary(self) -> dict:
        return {"seed": self.seed, "frames": self.frames, "taps": self.n_taps, "braked_frames": self.braked_frames,
                "braked_fraction": self.braked_frames / max(self.frames, 1)}

    # kept as a static method for the collectors' existing ``OUPerturb.brake_p_for_rate(...)`` calls
    brake_p_for_rate = staticmethod(brake_p_for_rate)


# =============================================================================================== (3) NumpyActor
_SELU_ALPHA, _SELU_SCALE = 1.6732632423543772848170429916717, 1.0507009873554804934193349852946


def _elu(x):
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


_ACTIVATIONS = {
    "elu": _elu,
    "crelu": _elu,                                              # rsl_rl maps "crelu" to torch.nn.CELU(alpha=1) == ELU
    "relu": lambda x: np.maximum(x, 0.0),
    "tanh": np.tanh,
    "selu": lambda x: _SELU_SCALE * np.where(x > 0.0, x, _SELU_ALPHA * np.expm1(np.minimum(x, 0.0))),
    "lrelu": lambda x: np.where(x > 0.0, x, 0.01 * x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "identity": lambda x: x,
}
ACTOR_FORMAT = "gc_actor_v1"


class NumpyActor:
    """A tracking policy evaluated in numpy: ``act(obs) = clip(center + scale * tanh(mlp((obs - mean) / (sqrt(var) + eps))), low, high)``.

    The normaliser is rsl_rl's ``EmpiricalNormalization`` in eval mode: ``(x - _mean) / (_std + eps)`` with
    ``_std = sqrt(_var)`` and ``eps`` 1e-2 by default (``rsl_rl/modules/normalizer.py``).  The MLP is the
    ``ActorCritic.actor`` Sequential: Linear, activation, ..., Linear (no activation after the last layer; the mean
    action ``act_inference`` returns).  The squash is the tracker env's ``_scale_policy_actions``
    (``tracker_env.py:341-343``) and the WP3 Chrono evaluator's ``PolicyController.act`` (``traverse_wp3_chrono_eval.py:134-141``).

    The steering-rate clamp is NOT applied here (the evaluator applied it after the squash); the collector applies
    ``hold_clip`` to the returned triple.  Weights are float64; ``act`` accepts one observation (num_obs,) or a batch
    (B, num_obs) and returns float64.

    npz keys (``ACTOR_FORMAT``): ``format``, ``num_obs``, ``num_actions``, ``obs_mean``, ``obs_var``, ``obs_eps``,
    ``n_layers``, ``W0, b0, ..., W{n-1}, b{n-1}`` (torch layout, W is (out, in)), ``activation``, ``action_center``,
    ``action_scale``, ``action_low``, ``action_high``, ``meta_json`` (the policy meta, incl. ``obs_layout`` for
    ``PolicyObs.from_meta``).
    """

    def __init__(self, obs_mean, obs_var, layers, activation: str, action_center, action_scale, action_low,
                 action_high, obs_eps: float = 1e-2, meta: dict | None = None):
        self.obs_mean = np.asarray(obs_mean, np.float64).reshape(-1)
        self.obs_var = np.asarray(obs_var, np.float64).reshape(-1)
        self.obs_eps = float(obs_eps)
        self.layers = [(np.asarray(W, np.float64), np.asarray(b, np.float64).reshape(-1)) for W, b in layers]
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; known: {sorted(_ACTIVATIONS)}")
        self.activation = activation
        self.act_fn = _ACTIVATIONS[activation]
        self.center, self.scale, self.low, self.high = (np.asarray(v, np.float64).reshape(-1)
                                                        for v in (action_center, action_scale, action_low, action_high))
        self.meta = dict(meta or {})
        self.num_obs = len(self.obs_mean)
        self.num_actions = self.layers[-1][0].shape[0]
        if self.obs_var.shape != (self.num_obs,) or self.layers[0][0].shape[1] != self.num_obs:
            raise ValueError("normaliser / first layer width mismatch")
        for (W, b), (Wn, _) in zip(self.layers[:-1], self.layers[1:]):
            if W.shape[0] != b.shape[0] or Wn.shape[1] != W.shape[0]:
                raise ValueError("layer shape mismatch")
        for v in (self.center, self.scale, self.low, self.high):
            if v.shape != (self.num_actions,):
                raise ValueError("action affine/bounds must have num_actions entries")
        self._std_eps = np.sqrt(self.obs_var) + self.obs_eps

    @classmethod
    def from_npz(cls, path) -> "NumpyActor":
        with np.load(Path(path), allow_pickle=False) as z:
            fmt = str(z["format"]) if "format" in z.files else ACTOR_FORMAT
            if fmt != ACTOR_FORMAT:
                raise ValueError(f"unexpected actor format {fmt!r}")
            n = int(z["n_layers"])
            layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(n)]
            meta = json.loads(str(z["meta_json"])) if "meta_json" in z.files else {}
            return cls(z["obs_mean"], z["obs_var"], layers, str(z["activation"]), z["action_center"],
                       z["action_scale"], z["action_low"], z["action_high"],
                       float(z["obs_eps"]) if "obs_eps" in z.files else 1e-2, meta)

    def mlp(self, obs) -> np.ndarray:
        """Pre-squash actor output (the torch ``act_inference`` value) for an observation or a batch."""
        o = np.asarray(obs, np.float64)
        if o.shape[-1] != self.num_obs:
            raise ValueError(f"observation width {o.shape[-1]} != {self.num_obs}")
        h = (o - self.obs_mean) / self._std_eps
        last = len(self.layers) - 1
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < last:
                h = self.act_fn(h)
        return h

    def act(self, obs) -> np.ndarray:
        return np.clip(self.center + self.scale * np.tanh(self.mlp(obs)), self.low, self.high)


def _torch_activation_name(module) -> str:
    import torch
    table = [(torch.nn.ELU, "elu"), (torch.nn.SELU, "selu"), (torch.nn.ReLU, "relu"), (torch.nn.CELU, "crelu"),
             (torch.nn.LeakyReLU, "lrelu"), (torch.nn.Tanh, "tanh"), (torch.nn.Sigmoid, "sigmoid"),
             (torch.nn.Identity, "identity")]
    for cls, name in table:
        if isinstance(module, cls):
            if name == "elu" and abs(float(module.alpha) - 1.0) > 0:
                raise ValueError("only ELU(alpha=1) is supported")
            if name == "crelu" and abs(float(module.alpha) - 1.0) > 0:
                raise ValueError("only CELU(alpha=1) is supported")
            if name == "lrelu" and abs(float(module.negative_slope) - 0.01) > 0:
                raise ValueError("only LeakyReLU(0.01) is supported")
            return name
    raise ValueError(f"unsupported activation module {type(module).__name__}")


def export_torch_actor(actor_critic, normalizer, meta: dict, path) -> dict:
    """Write the ``NumpyActor`` npz from an rsl_rl ``ActorCritic`` and its ``EmpiricalNormalization``.

    ``actor_critic.actor`` must be an ``nn.Sequential`` of Linear layers separated by one activation type
    (rsl_rl's construction, ``rsl_rl/modules/actor_critic.py``); ``normalizer`` exposes the ``_mean`` / ``_var`` /
    ``_std`` buffers and ``eps``.  ``meta`` is the run's policy meta: it must contain ``action_scale``,
    ``action_low``, ``action_high`` and ``action_center`` (a 3-list, or the string "dataset_mean" with the mean in
    ``meta["act_mean"]``, the WP3 convention); everything else (e.g. ``obs_layout``, ``num_obs``, ``nrd_hash``) is
    stored verbatim in ``meta_json``.  torch is imported here only.  Returns the dict of arrays written.
    """
    import torch
    actor = actor_critic.actor if hasattr(actor_critic, "actor") else actor_critic
    layers, acts = [], []
    for m in actor:
        if isinstance(m, torch.nn.Linear):
            layers.append((m.weight.detach().cpu().double().numpy(), m.bias.detach().cpu().double().numpy()))
        else:
            acts.append(_torch_activation_name(m))
    if not layers:
        raise ValueError("actor has no Linear layers")
    if len(acts) != len(layers) - 1 or len(set(acts)) > 1:
        raise ValueError(f"actor must be Linear/act/.../Linear with one activation type, got {acts}")
    activation = acts[0] if acts else "identity"
    mean = normalizer._mean.detach().cpu().double().numpy().reshape(-1)
    var = normalizer._var.detach().cpu().double().numpy().reshape(-1)
    std = normalizer._std.detach().cpu().double().numpy().reshape(-1)
    if not np.allclose(std, np.sqrt(var), atol=1e-6, rtol=1e-5):
        raise ValueError("normaliser _std buffer is not sqrt(_var); refusing to guess which one the policy saw")
    eps = float(getattr(normalizer, "eps", 1e-2))
    center = meta["action_center"]
    if isinstance(center, str):
        if center != "dataset_mean":
            raise ValueError(f"unknown action_center {center!r}")
        center = meta["act_mean"]
    arrays = {"format": np.array(ACTOR_FORMAT), "num_obs": np.int64(len(mean)),
              "num_actions": np.int64(layers[-1][0].shape[0]),
              "obs_mean": mean, "obs_var": var, "obs_eps": np.float64(eps), "n_layers": np.int64(len(layers)),
              "activation": np.array(activation),
              "action_center": np.asarray(center, np.float64), "action_scale": np.asarray(meta["action_scale"], np.float64),
              "action_low": np.asarray(meta["action_low"], np.float64), "action_high": np.asarray(meta["action_high"], np.float64),
              "meta_json": np.array(json.dumps(meta, default=_jsonable, sort_keys=True))}
    for i, (W, b) in enumerate(layers):
        arrays[f"W{i}"], arrays[f"b{i}"] = W, b
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return arrays


# =============================================================================================== (4) routes
def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    raise TypeError(f"not JSON serialisable: {type(v).__name__}")


def route_to_json(route: dict) -> dict:
    """The collector route file format (``traverse_fdm_rgbd_diverse_chrono.read_route``): lists, not arrays."""
    out = {k: np.asarray(route[k], float).tolist() for k in ("waypoints", "speeds", "stations", "headings")}
    out["meta"] = json.loads(json.dumps(route.get("meta", {}), default=_jsonable))
    return out


def route_sha256(route: dict) -> str:
    """Content hash of the route as it will be written; the expression of ``f104_n2_iter.route_sha256``."""
    s = json.dumps({k: np.asarray(route[k], float).tolist() for k in ("waypoints", "speeds", "stations", "headings")})
    return hashlib.sha256(s.encode()).hexdigest()


def check_reference_contract(route) -> None:
    """Copy of ``nedm.traverse.fdm_diverse_planner.check_reference_contract`` (lines 16-37; that module imports
    torch).  Rejects inconsistent station encodings, cusps and headings that disagree with the waypoint tangents."""
    xy = np.asarray(route["waypoints"], float)
    station = np.asarray(route["stations"], float)
    heading = np.asarray(route["headings"], float)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 3:
        raise ValueError("Reference needs at least three XY waypoints")
    if station.shape != (len(xy),) or heading.shape != (len(xy),):
        raise ValueError("Reference stations/headings must match waypoints")
    if any(not np.isfinite(a).all() for a in (xy, station, heading)):
        raise ValueError("Nonfinite reference geometry")
    delta = np.diff(xy, axis=0)
    ds = np.linalg.norm(delta, axis=1)
    if (ds <= 1e-8).any() or not np.allclose(station-station[0], np.r_[0., ds.cumsum()], atol=1e-4, rtol=1e-5):
        raise ValueError("Reference station encoding disagrees with waypoint geometry")
    unit = delta/ds[:, None]
    if (np.sum(unit[1:]*unit[:-1], axis=1) < np.cos(np.pi/4.)).any():
        raise ValueError("Reference contains a cusp or sharp segment-heading reversal")
    expected = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))
    error = np.arctan2(np.sin(heading-expected), np.cos(heading-expected))
    if (np.abs(error) > .15).any():
        raise ValueError("Supplied reference headings disagree with waypoint tangents")


def planner_cfg():
    """The deployed planner's validator configuration, ``gen_planner.py:30`` rebuilt without importing gen_planner."""
    from nedm.traverse.fdm_mppi import MPPIConfig
    return MPPIConfig(max_speed_mps=6., min_speed_mps=0.0, max_curvature_inv_m=.125, arena_half_extent_m=40.)


def safe_validate(route, obstacles, cfg, anchor) -> dict:
    """``gen_planner.safe_validate`` (lines 94-99): validate_reference, degenerate routes rejected instead of raising."""
    from nedm.traverse.fdm_mppi import validate_reference
    try:
        return validate_reference(route, obstacles, cfg, anchor)
    except ValueError as e:
        return {'valid': False, 'reasons': [str(e)]}


def planner_validator(route, pose) -> bool:
    """The validator the planner uses for every candidate (``f104_n2_iter.valid``, ``gen_planner.proposal_pool``)."""
    return bool(safe_validate(route, [], planner_cfg(), np.asarray(pose, float))['valid'])


def constant_speed(stations, v):
    """``gen_planner.constant_speed`` (lines 45-47)."""
    stations = np.asarray(stations, float)
    return np.minimum(np.full(len(stations), float(v)), np.sqrt(4 * np.maximum(stations[-1] - stations, 0)))


def _hermite(pose, goal, scale, step_m=.5):
    """``gen_planner._hermite`` (lines 50-58); scale 1.0 is the frozen generator's route_00
    (``fdm_diverse_planner.propose_route_families`` with offset 0, lines 248-278)."""
    pose, goal = np.asarray(pose, float), np.asarray(goal, float)
    delta = goal - pose[:2]; length = float(np.linalg.norm(delta))
    t = np.linspace(0., 1., max(33, int(np.ceil(length / step_m)) + 1))[:, None]
    t0 = scale * length * np.array([np.cos(pose[2]), np.sin(pose[2])])
    xy = (2*t**3 - 3*t**2 + 1) * pose[:2] + (t**3 - 2*t**2 + t) * t0 + (-2*t**3 + 3*t**2) * goal + (t**3 - t**2) * delta
    st = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    hd = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))
    return {'waypoints': xy, 'speeds': constant_speed(st, 2.), 'stations': st, 'headings': hd, 'meta': {'start_tangent_scale': scale}}


def _arc_line(pose, goal, radius, step_m=.5, long_way=False):
    """``gen_planner._arc_line`` (lines 61-91): turn on a circle until facing the goal, then drive straight."""
    p = np.asarray(pose[:2], float); th = float(pose[2]); g = np.asarray(goal, float)
    fwd = np.array([np.cos(th), np.sin(th)]); left = np.array([-fwd[1], fwd[0]])
    d = g - p
    side = 1.0 if fwd[0] * d[1] - fwd[1] * d[0] >= 0 else -1.0     # == np.cross(fwd, g - p) for 2-vectors (deprecated in numpy 2)
    if long_way:
        side = -side
    c = p + side * radius * left
    cg = g - c; dist = float(np.linalg.norm(cg))
    if dist <= radius * 1.05:
        return None
    phi0 = math.atan2(p[1] - c[1], p[0] - c[0])
    base = math.atan2(cg[1], cg[0]); off = math.acos(radius / dist)
    phi_t = base - side * off
    sweep = (phi_t - phi0) * side
    sweep = sweep % (2 * math.pi)
    n_arc = max(2, int(math.ceil(radius * sweep / step_m)) + 1)
    ang = phi0 + side * np.linspace(0, sweep, n_arc)
    arc = c[None] + radius * np.stack([np.cos(ang), np.sin(ang)], 1)
    tp = arc[-1]; L = float(np.linalg.norm(g - tp))
    n_line = max(2, int(math.ceil(L / step_m)) + 1)
    line = tp[None] + (g - tp)[None] * np.linspace(0, 1, n_line)[:, None]
    xy = np.concatenate([arc, line[1:]])
    st = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    if np.any(np.diff(st) <= 1e-6):
        keep = np.r_[True, np.diff(st) > 1e-6]; xy = xy[keep]; st = np.r_[0., np.linalg.norm(np.diff(xy, axis=0), axis=1).cumsum()]
    hd = np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))
    return {'waypoints': xy, 'speeds': constant_speed(st, 2.), 'stations': st, 'headings': hd,
            'meta': {'start_tangent_scale': None, 'arc_radius_m': radius}}


def base_route(pose, goal, validator=None, scales=(1.0, 1.5, 2.0, 0.7, 2.5), radii=(12.0, 10.0, 9.0)) -> dict:
    """``gen_planner.base_route`` (lines 102-121) without the torch-bearing import: the straight 2 m/s route from a
    pose (x, y, yaw) to the goal.  route_00's Hermite first; if the validator rejects it (a moving vehicle facing
    away from the goal), wider/tighter Hermite starts then arc-then-straight shapes in the same fixed order; the
    first accepted shape is used, else route_00 (the caller's continuations will then fail validation and raise)."""
    validator = planner_validator if validator is None else validator
    pose = np.asarray(pose, float)
    first = _hermite(pose, goal, 1.0)
    first['meta'] = {'start_tangent_scale': 1.0}
    if validator(first, pose):
        return first
    options = ([_hermite(pose, goal, sc) for sc in scales[1:]] + [_arc_line(pose, goal, R) for R in radii]
               + [_arc_line(pose, goal, R, long_way=True) for R in radii + (8.5,)])
    for cand in options:
        if cand is not None and validator(cand, pose):
            return cand
    return first


def _wrap_pi(a: float) -> float:
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def start_heading_err_deg(route: dict, yaw: float) -> float:
    """|wrap(headings[0] - yaw)| in degrees: the kink between the vehicle's heading and the route's first tangent."""
    return abs(math.degrees(_wrap_pi(float(np.asarray(route["headings"], float)[0]) - float(yaw))))


def speed_floor_from(v0: float, stations, a_dec: float = 2.0, v_max: float = 6.0) -> np.ndarray:
    """Deceleration ramp from the vehicle's speed: ``min(sqrt(max(v0^2 - 2 a_dec (s - s[0]), 0)), v_max)``."""
    s = np.asarray(stations, float)
    v0 = max(float(v0), 0.0)
    return np.minimum(np.sqrt(np.maximum(v0 * v0 - 2.0 * float(a_dec) * (s - s[0]), 0.0)), float(v_max))


def sample_continuations(pose_xy_yaw, goal_xy, k: int, seed: int, speeds=(2.0, 4.0, 6.0), validator=None,
                         tries_factor: int = 64, v0: float | None = None,
                         max_start_heading_err_deg: float | None = 15.0, v0_accel_cap: bool = False) -> list[dict]:
    """``k`` continuation routes of the night-2 route family from a (moving-prefix) pose to the goal.

    Base = ``base_route(pose, goal)`` (the planner's straight route).  Continuation i is one wide night-2 sample
    (``f104_n2_sampler.sample_one``: sine-basis lateral offset zero at both ends, 4 free-end speed knots +-4 m/s on
    a constant base cruise speed ``speeds[i % len(speeds)]``, speeds in [0.5, 6], terminal deceleration cone and
    accel/decel projection) redrawn until it passes, in this order,
      1. the start-heading acceptance: ``|wrap(headings[0] - pose yaw)| <= max_start_heading_err_deg`` (15 deg by
         default; ``None`` disables it).  The lateral basis vanishes at both ends but its slope does not, so without
         this test a draw's first tangent can kink up to ~40 deg away from the vehicle's heading and the follower
         answers with a full steering lock within 0.5 s of the branch (VERIFY_gc_control.md problem 2);
      2. ``validator(route, pose)`` (default: the planner's validator, ``validate_reference`` with
         ``gen_planner.CFG``'s arguments and no obstacles, anchored at the pose);
      3. the reference contract (``check_reference_contract``).
    ``v0`` (the vehicle's forward speed at the branch, m/s): when given, the sampled speed profile is floored BEFORE
    the tests at the 2 m/s^2 deceleration ramp from v0, ``speeds = min(max(speeds, sqrt(max(v0^2 - 2 A_DEC (s - s0),
    0))), 6.0)`` with ``A_DEC = f104_n2_sampler.A_DEC``, so no continuation starts below the vehicle's speed (a target
    below the current speed makes ``ChPathFollowerDriver`` brake hard, see ``make_follower``).  The element-wise max
    of two profiles that each respect the accel/decel limits respects them too, so the floored route stays
    planner-valid whenever the unfloored one was (checked on 45 real anchors' continuations in the verification).
    ``v0_accel_cap`` (opt-in, off by default; the verifier's second suggestion): additionally cap the profile at the
    1.5 m/s^2 acceleration ramp ``sqrt(v0^2 + 2 A_ACC (s - s0))`` so the route's demand never runs ahead of what the
    vehicle can reach from v0 (without it a slow anchor can get a 6 m/s start target: full throttle from a moving
    state); with both ramps the first speed equals min(v0, 6) exactly.  The element-wise min with the exact
    acceleration ramp also preserves both limits.
    Because the lateral basis vanishes at both ends every route starts at the pose's xy and ends at the goal
    exactly; both are asserted within ``GOAL_TOL_M`` (0.25 m).
    Raises RuntimeError if ``tries_factor * k`` draws do not yield ``k`` valid routes (drop the anchor).

    Returns routes in the collector format {waypoints (n,2), speeds (n,), stations (n,), headings (n,), meta};
    ``meta`` carries ``candidate='n2_wide'``, ``continuation``, ``base_speed_mps``, ``seed``, ``draws``,
    ``branch_pose``, ``goal_xy``, ``route_sha256``, ``base`` (the base route's meta), ``family``, ``v0_mps``,
    ``start_speed_mps``, ``speed_floor_raised_points`` (waypoints whose speed the floor raised), ``v0_accel_cap``,
    ``speed_cap_lowered_points``, ``start_heading_err_deg`` and ``max_start_heading_err_deg``.
    Deterministic given (pose, goal, k, seed, speeds, v0, tolerance).  The seed must be ANCHOR-specific (e.g. a
    hash of the episode id and the cut frame): the lateral and speed draws depend on the seed only, so two anchors
    with similar base routes and the same seed get near-identical continuation shapes.
    """
    import f104_n2_sampler as S
    validator = planner_validator if validator is None else validator
    pose = np.asarray(pose_xy_yaw, float).reshape(3)
    goal = np.asarray(goal_xy, float).reshape(2)
    if k < 1:
        raise ValueError("k must be >= 1")
    if np.linalg.norm(goal - pose[:2]) < 1.0:
        raise ValueError("pose within 1 m of the goal: no continuation to sample")
    if v0 is not None and not np.isfinite(v0):
        raise ValueError(f"non-finite v0 {v0!r}")
    tol = None if max_start_heading_err_deg is None else float(max_start_heading_err_deg)
    base = base_route(pose, goal, validator)
    rng = np.random.default_rng(int(seed))
    out, draws, reasons = [], 0, {}
    max_draws = int(tries_factor) * int(k)
    while len(out) < k and draws < max_draws:
        i = len(out)
        v = float(speeds[i % len(speeds)])
        draws += 1
        r = S.sample_one(base, rng, base_speed=v)
        raised = lowered = 0
        # PLAN A4 speed range 2-6 m/s: floor the profile at 2 m/s outside the terminal deceleration cone (the family's
        # own constant-2 profile, so the element-wise max stays planner-valid). Without this a draw can command 0.5 m/s
        # mid-route and the follower stalls the vehicle on soft soil (CRM collector verification, finding 2).
        st0 = np.asarray(r['stations'], float)
        cone2 = np.minimum(2.0, np.sqrt(np.maximum(2.0 * float(S.A_DEC) * (st0[-1] - st0), 0.0)))
        r['speeds'] = np.maximum(np.asarray(r['speeds'], float), cone2)
        if v0 is not None:
            sampled = np.asarray(r['speeds'], float)
            floor = speed_floor_from(v0, r['stations'], S.A_DEC, S.V_MAX)
            r['speeds'] = np.minimum(np.maximum(sampled, floor), float(S.V_MAX))
            raised = int((r['speeds'] > sampled + 1e-12).sum())
            if v0_accel_cap:
                st = np.asarray(r['stations'], float)
                cap = np.sqrt(min(max(float(v0), 0.0), float(S.V_MAX)) ** 2 + 2.0 * float(S.A_ACC) * (st - st[0]))
                before_cap = r['speeds']
                r['speeds'] = np.minimum(before_cap, cap)
                lowered = int((r['speeds'] < before_cap - 1e-12).sum())
        h_err = start_heading_err_deg(r, pose[2])
        if tol is not None and h_err > tol:
            reasons['start_heading'] = reasons.get('start_heading', 0) + 1
            continue
        if not validator(r, pose):
            reasons['validator'] = reasons.get('validator', 0) + 1
            continue
        try:
            check_reference_contract(r)
        except ValueError as e:
            reasons[str(e)] = reasons.get(str(e), 0) + 1
            continue
        xy = np.asarray(r['waypoints'], float)
        d0, d1 = float(np.linalg.norm(xy[0] - pose[:2])), float(np.linalg.norm(xy[-1] - goal))
        if d0 > GOAL_TOL_M or d1 > GOAL_TOL_M:
            raise AssertionError(f"continuation endpoints off: start {d0:.3f} m, end {d1:.3f} m")
        route = {'waypoints': xy, 'speeds': np.asarray(r['speeds'], float), 'stations': np.asarray(r['stations'], float),
                 'headings': np.asarray(r['headings'], float)}
        route['meta'] = {**r.get('meta', {}), 'family': 'gc_continuation_v1', 'continuation': i, 'base_speed_mps': v,
                         'seed': int(seed), 'draws': draws, 'branch_pose': pose.tolist(), 'goal_xy': goal.tolist(),
                         'base': json.loads(json.dumps(base.get('meta', {}), default=_jsonable)),
                         'route_sha256': route_sha256(route), 'start_err_m': d0, 'end_err_m': d1,
                         'v0_mps': None if v0 is None else float(v0), 'start_speed_mps': float(route['speeds'][0]),
                         'speed_floor_raised_points': raised, 'v0_accel_cap': bool(v0_accel_cap),
                         'speed_cap_lowered_points': lowered, 'start_heading_err_deg': h_err,
                         'max_start_heading_err_deg': tol}
        out.append(route)
    if len(out) < k:
        raise RuntimeError(f"only {len(out)}/{k} valid continuations after {draws} draws; rejections {reasons}; "
                           f"base valid: {validator(base, pose)}")
    return out


# =============================================================================================== (5) make_follower
def follower_points(route: dict, height_fn, z_offset: float = 0.5) -> list[tuple[float, float, float]]:
    """The frozen collector's waypoint decimation (``traverse_fdm_rgbd_diverse_chrono.py:100-106``): keep a waypoint
    when its station is >= 2 m past the last kept one, always keep the last waypoint; z = ground height + z_offset."""
    points = []
    last_station = -10.
    for (x, y), station in zip(route["waypoints"], route["stations"]):
        if station-last_station < 2. and station != route["stations"][-1]:
            continue
        last_station = station
        points.append((float(x), float(y), float(height_fn(x, y))+z_offset))
    return points


def make_follower(veh_module, chrono_module, vehicle, route: dict, height_fn, look_ahead: float = 5.0,
                  steer_gains=(0.8, 0.0, 0.0), speed_gains=(0.6, 0.05, 0.0), z_offset: float = 0.5,
                  initialize: bool = True):
    """The path follower exactly as ``make_driver`` builds it (``traverse_fdm_rgbd_diverse_chrono.py:98-112``,
    reused by ``crm_collect.py:190``): decimated waypoints at ground + 0.5 m, ``ChBezierCurve(points)``,
    ``ChPathFollowerDriver(vehicle, curve, "route", speeds[0])``, look-ahead 5 m, steering gains (0.8, 0, 0), speed
    gains (0.6, 0.05, 0), then ``Initialize()``.  ``height_fn(x, y)`` replaces ``tmap.height``.

    ``initialize=False`` is for the branch swap at ``branch_frame`` (PLAN A4): ``ChClosedLoopDriver::Initialize``
    only adds a fixed body carrying the path's visual asset to the system (``ChPathFollowerDriver.cpp:59-77``); adding
    a body mid-run is avoided.  The controllers are already reset: ``ChPathFollowerDriver``'s constructor calls
    ``Reset()`` (``ChPathFollowerDriver.cpp:131-141``), which zeroes the PID errors and re-seeds the curve tracker
    at the sentinel; at construction the look-ahead distance is still 0 (``ChSteeringController.cpp:49``), so the
    sentinel is the vehicle's own reference point and the tracker's global closest-point search starts there, which
    is the branch route's first waypoint.  The 5 m look-ahead set afterwards is used from the first ``Advance``.
    This is the same order of operations as ``make_driver`` at the settle.  The new driver starts with
    ``m_throttle = 0`` and ``m_steering = 0``; steering continuity across the swap is enforced by the collector's
    per-frame/per-substep steering clamp.

    Speed at the swap (corrected 2026-09-21, VERIFY_gc_control.md problem 1): ``ChClosedLoopDriver::Advance``
    (``ChPathFollowerDriver.cpp:94-108``) keeps BRAKING for as long as ``m_throttle <= 0.2`` and the vehicle is faster
    than the target, and its "reduce throttle" branch itself lands below 0.2 whenever the speed error exceeds
    1.33 m/s.  So a branch route whose first speed is below the vehicle's speed starts with a hard brake burst, not a
    one-substep tap: measured on flat rigid ground (4.0 m/s vehicle, 2.0 m/s target) brake 0.96-1.0 for 0.50 s, 251
    braking substeps in the 3 s after the swap, vx 1.8 m/s after 2 s; carrying the old inputs over changes nothing.
    ``sample_continuations(..., v0=vx_at_branch)`` floors the continuation's speed profile at the 2 m/s^2
    deceleration ramp from the vehicle's speed so the branch never starts below it.

    Notes for the collector authors: (a) in the external modes ``hold_clip`` must REPLACE the frozen per-substep
    ``+-2*dt`` steering clamp, not run before it (with the frozen line kept a held 0.1 steering step is ramped over
    the 25/50 substeps and the substep audit's per-interval max-min = 0 fails); (b) the frozen ``run`` asserts the
    route start within 0.25 m of the layout start (``traverse_fdm_rgbd_diverse_chrono.py:206``,
    ``crm_collect.py:199``); a branch route starts at the branch pose, so that check must be bypassed for the branch
    leg (the prefix route is checked as before).
    """
    points = chrono_module.vector_ChVector3d()
    for x, y, z in follower_points(route, height_fn, z_offset):
        points.append(chrono_module.ChVector3d(x, y, z))
    driver = veh_module.ChPathFollowerDriver(vehicle, chrono_module.ChBezierCurve(points), "route", float(route["speeds"][0]))
    driver.GetSteeringController().SetLookAheadDistance(float(look_ahead))
    driver.GetSteeringController().SetGains(*(float(g) for g in steer_gains))
    driver.GetSpeedController().SetGains(*(float(g) for g in speed_gains))
    if initialize:
        driver.Initialize()
    return driver


# =============================================================================================== (6) PolicyObs
class RouteTracker:
    """numpy route geometry, copied from ``traverse_wp3_chrono_eval.py:66-106`` (itself the mirror of
    ``tracker_env._route_errors`` / ``_preview_body``, lines 365-398).  ``update(first=True)`` searches the whole
    route; later calls search waypoints idx-2 .. idx+search_window-1.  ``preview`` returns 10 x (bx/10, by/10,
    v_ref/5) for points 1 m .. 10 m ahead along the route (index step = round(k * spacing / mean waypoint spacing))."""

    def __init__(self, route: dict, meta: dict | None = None):
        meta = DEFAULT_TRACK if meta is None else meta
        self.xy = np.asarray(route["waypoints"], np.float64)
        self.v = np.asarray(route["speeds"], np.float64)
        self.h = np.asarray(route["headings"], np.float64)
        self.s = np.asarray(route["stations"], np.float64)
        self.n = len(self.xy)
        self.ds = self.s[-1] / max(self.n - 1, 1)
        self.idx = 0
        self.offsets = np.arange(-2, int(meta["search_window"]))
        self.k = np.arange(1, int(meta["preview_points"]) + 1)
        self.spacing = float(meta["preview_spacing_m"])

    def update(self, x: float, y: float, yaw: float, vx: float, first: bool = False) -> dict:
        if first:
            cand = np.arange(self.n)
        else:
            cand = np.clip(self.idx + self.offsets, 0, self.n - 1)
        d = np.hypot(self.xy[cand, 0] - x, self.xy[cand, 1] - y)
        self.idx = int(cand[int(np.argmin(d))])
        wp, h, v_ref = self.xy[self.idx], self.h[self.idx], self.v[self.idx]
        dx, dy = x - wp[0], y - wp[1]
        e_along = dx * math.cos(h) + dy * math.sin(h)
        e_ct = -dx * math.sin(h) + dy * math.cos(h)
        e_h = math.atan2(math.sin(yaw - h), math.cos(yaw - h))
        return {"e_along": e_along, "e_ct": e_ct, "e_h": e_h, "e_v": vx - v_ref, "v_ref": v_ref,
                "route_end": self.idx >= self.n - 2, "station": self.s[self.idx]}

    def preview(self, x: float, y: float, yaw: float) -> np.ndarray:
        step = np.round(self.k * self.spacing / self.ds).astype(int)
        idx = np.minimum(self.idx + step, self.n - 1)
        dx, dy = self.xy[idx, 0] - x, self.xy[idx, 1] - y
        c, s = math.cos(yaw), math.sin(yaw)
        bx, by = c * dx + s * dy, -s * dx + c * dy
        return np.stack([bx / 10.0, by / 10.0, self.v[idx] / 5.0], axis=-1).reshape(-1)


def history_blocks(states, actions, k: int, hist_steps: int = 8, state_cols=OBSERVABLE_COLS,
                   settle_action=SETTLE_ACTION, rest_state=None) -> tuple[np.ndarray, np.ndarray]:
    """Offline twin of the streaming history in ``PolicyObs``: the blocks the policy sees at frame ``k`` of a recording.

    ``states`` (T, 17) and ``actions`` (T, 3) are the recorded arrays.  Returns
      act_block   (H, 3)          = action[k-H .. k-1], oldest first (its last row is the 'last action')
      state_block (H, len(cols))  = state[k-H+1 .. k][:, cols], oldest first (its last row is the current state)
    Indices before frame 0 are padded with ``settle_action`` (0, 0, 1) and the rest state (``rest_state`` if given,
    else ``states[0]``, the settled state at frame 0): the same convention as ``fdm_data.build_history``
    (``fdm_data.py:87-90``).  For a branch at frame F (policy mode after a recorded prefix) pass the prefix arrays
    and k = F: nothing is padded once F >= H.
    """
    states = np.asarray(states, np.float64); actions = np.asarray(actions, np.float64)
    cols = list(state_cols)
    rest = np.asarray(states[0] if rest_state is None else rest_state, np.float64)
    H = int(hist_steps)
    a_idx = np.arange(k - H, k)
    s_idx = np.arange(k - H + 1, k + 1)
    act_block = actions[np.clip(a_idx, 0, None)].copy()
    act_block[a_idx < 0] = np.asarray(settle_action, np.float64)
    state_block = states[np.clip(s_idx, 0, None)][:, cols].copy()
    state_block[s_idx < 0] = rest[cols]
    return act_block, state_block


class PolicyObs:
    """Builds the tracking-policy observation at every frame from pose, state, the route and the last held action.

    Layout (``num_obs = 38 + 3 H + C H`` with H = ``hist_steps`` = 8, C = len(``state_cols``) = 12 -> 158):
      [0:3]     e_along / 10, e_ct / 10, e_h / pi        errors to the nearest waypoint (window idx-2 .. idx+39;
                                                          the first call after reset searches the whole route)
      [3:33]    10 preview points at 1 m spacing, each (bx / 10, by / 10, v_ref / 5) in the body frame
      [33:35]   vx / 10, yaw rate (rad/s, unscaled)       state columns 0 and 6
      [35:38]   last action                               the triple held over the previous frame (action[k-1]);
                                                          (0, 0, 1) at frame 0
      [38:38+3H]         past actions block, ``history_blocks`` act_block flattened row-major (oldest first);
                         its last row equals [35:38]
      [38+3H:38+3H+CH]   past observable states block, ``history_blocks`` state_block flattened row-major (oldest
                         first, the newest row is the CURRENT state); raw physical values unless ``state_mean`` /
                         ``state_std`` (C each) are given, then (s - mean) / std
    [0:38] is exactly the WP3 tracker env observation (``tracker_env.py:478-490``; ``traverse_wp3_chrono_eval.py:
    426-428``).  Non-finite entries are zeroed as the env does (counted in ``n_nonfinite``).

    Frame-0 padding: ``reset(rest_state)`` seeds the buffers with H copies of the rest state (the settled state[0])
    and H copies of the settle action (0, 0, 1).  Per frame k the collector calls
        obs = po.observe(pose[k], state[k], last_action)      # last_action = action[k-1], settle action at k = 0
        cmd = hold_clip(actor.act(obs), last_action[0]); ...  # drive interval k with cmd
        po.push(state[k], cmd)                                # after the frame: (state[k], action[k])
    so ``observe`` at frame k sees states k-H+1..k (buffer + current) and actions k-H..k-1 (buffer).  For a branch
    swap the prefix is loaded with ``seed_history(prefix_states, prefix_actions)`` before the first ``observe`` on
    the branch route.  ``layout()`` documents the slices; ``from_meta(route, meta)`` rebuilds the helper from the
    ``obs_layout`` dict stored in the actor npz's ``meta_json`` (the tracker env writes the same dict).
    """

    def __init__(self, route: dict, preview_points: int = 10, preview_spacing_m: float = 1.0, search_window: int = 40,
                 hist_steps: int = 8, state_cols=OBSERVABLE_COLS, settle_action=SETTLE_ACTION,
                 state_mean=None, state_std=None):
        self.track = RouteTracker(route, {"search_window": int(search_window), "preview_points": int(preview_points),
                                          "preview_spacing_m": float(preview_spacing_m)})
        self.preview_points, self.preview_spacing_m, self.search_window = int(preview_points), float(preview_spacing_m), int(search_window)
        self.H, self.cols = int(hist_steps), tuple(int(c) for c in state_cols)
        self.settle_action = np.asarray(settle_action, np.float64).reshape(3)
        self.state_mean = None if state_mean is None else np.asarray(state_mean, np.float64).reshape(len(self.cols))
        self.state_std = None if state_std is None else np.asarray(state_std, np.float64).reshape(len(self.cols))
        if (self.state_mean is None) != (self.state_std is None):
            raise ValueError("state_mean and state_std must be given together")
        self.base_dim = 3 + 3 * self.preview_points + 2 + 3
        self.num_obs = self.base_dim + 3 * self.H + len(self.cols) * self.H
        self.n_nonfinite = 0
        self._first = True
        self._act_buf = [self.settle_action.copy() for _ in range(self.H)]
        self._state_buf = None

    @classmethod
    def from_meta(cls, route: dict, meta: dict) -> "PolicyObs":
        lay = meta.get("obs_layout", {})
        return cls(route, preview_points=lay.get("preview_points", meta.get("preview_points", 10)),
                   preview_spacing_m=lay.get("preview_spacing_m", meta.get("preview_spacing_m", 1.0)),
                   search_window=lay.get("search_window", meta.get("search_window", 40)),
                   hist_steps=lay.get("hist_steps", 8), state_cols=lay.get("state_cols", OBSERVABLE_COLS),
                   settle_action=lay.get("settle_action", SETTLE_ACTION),
                   state_mean=lay.get("state_mean"), state_std=lay.get("state_std"))

    def layout(self) -> dict:
        H, C = self.H, len(self.cols)
        return {"num_obs": self.num_obs, "errors": [0, 3], "preview": [3, 3 + 3 * self.preview_points],
                "vx_yawrate": [3 + 3 * self.preview_points, self.base_dim - 3], "last_action": [self.base_dim - 3, self.base_dim],
                "past_actions": [self.base_dim, self.base_dim + 3 * H],
                "past_states": [self.base_dim + 3 * H, self.base_dim + 3 * H + C * H],
                "hist_steps": H, "state_cols": list(self.cols), "preview_points": self.preview_points,
                "preview_spacing_m": self.preview_spacing_m, "search_window": self.search_window,
                "settle_action": self.settle_action.tolist(),
                "state_mean": None if self.state_mean is None else self.state_mean.tolist(),
                "state_std": None if self.state_std is None else self.state_std.tolist(),
                "order": "oldest first; past_states newest row = current state; past_actions newest row = last action"}

    def reset(self, rest_state) -> None:
        """Seed the history with the rest state (state[0] at the settle's end) and the settle action; the next
        ``observe`` searches the whole route for the nearest waypoint."""
        rest = np.asarray(rest_state, np.float64)
        self._state_buf = [rest.copy() for _ in range(self.H)]
        self._act_buf = [self.settle_action.copy() for _ in range(self.H)]
        self._first = True

    def seed_history(self, states, actions) -> None:
        """Load the buffers from a recorded prefix (branch swap): the last H rows of ``states`` / ``actions``; a
        prefix shorter than H is padded with its first state and the settle action, like ``history_blocks``."""
        states = np.asarray(states, np.float64); actions = np.asarray(actions, np.float64)
        if len(states) == 0 or len(states) != len(actions):
            raise ValueError("prefix states/actions must be non-empty and equally long")
        m = len(states)
        a_idx = np.arange(m - self.H, m)                 # action[F-H .. F-1] for the first branch frame F = m
        s_idx = np.arange(m - self.H, m)                 # buffer = state[F-H .. F-1]; observe() drops the oldest and appends state[F]
        self._act_buf = [actions[i].copy() if i >= 0 else self.settle_action.copy() for i in a_idx]
        self._state_buf = [states[max(i, 0)].copy() for i in s_idx]
        self._first = True

    def observe(self, pose, state, last_action) -> np.ndarray:
        if self._state_buf is None:
            raise RuntimeError("call reset(rest_state) or seed_history() before observe()")
        x, y, yaw = (float(v) for v in np.asarray(pose, np.float64).reshape(3))
        st = np.asarray(state, np.float64)
        last = np.asarray(last_action, np.float64).reshape(3)
        err = self.track.update(x, y, yaw, float(st[VX_COL]), first=self._first)
        self._first = False
        base = np.concatenate([[err["e_along"] / 10.0, err["e_ct"] / 10.0, err["e_h"] / math.pi],
                               self.track.preview(x, y, yaw), [st[VX_COL] / 10.0, st[YAW_RATE_COL]], last])
        if self.H > 0:
            act_block = np.stack(self._act_buf)
            state_block = np.stack(self._state_buf[1:] + [st])[:, list(self.cols)]
        else:
            act_block, state_block = np.zeros((0, 3)), np.zeros((0, len(self.cols)))
        if self.state_mean is not None:
            state_block = (state_block - self.state_mean) / self.state_std
        obs = np.concatenate([base, act_block.reshape(-1), state_block.reshape(-1)])
        bad = ~np.isfinite(obs)
        if bad.any():
            self.n_nonfinite += int(bad.sum())
            obs = np.where(bad, 0.0, obs)
        self.last_err = err
        return obs

    def push(self, state, action) -> None:
        """Append (state[k], action[k]) at the end of frame k."""
        self._state_buf = self._state_buf[1:] + [np.asarray(state, np.float64).copy()]
        self._act_buf = self._act_buf[1:] + [np.asarray(action, np.float64).reshape(3).copy()]


# =============================================================================================== self-test
def _selftest(out_dir: Path) -> dict:
    t0 = time.time()
    report: dict = {}
    out_dir.mkdir(parents=True, exist_ok=True)

    # (1) hold_clip -------------------------------------------------------------------------------------------
    def _eq(a, b):
        return np.allclose(a, b, atol=1e-12, rtol=0) and all(isinstance(v, float) for v in a)
    assert _eq(hold_clip((0.5, 1.3, -0.2), 0.0), (0.1, 1.0, 0.0))
    assert _eq(hold_clip((-0.5, 0.4, 0.2), -0.35), (-0.45, 0.4, 0.2))
    assert _eq(hold_clip((0.0, 0.0, 1.0), 0.95), (0.85, 0.0, 1.0))
    assert _eq(hold_clip((1.5, 0.5, 0.0), 0.95), (1.0, 0.5, 0.0))          # rate would allow 1.05; box clips to 1
    assert _eq(hold_clip((0.3, 0.2, 0.0), 0.25, rate=0.0), (0.25, 0.2, 0.0))
    for bad in (((float("nan"), 0, 0), 0.0), ((0, 0, 0), 1.5)):
        try:
            hold_clip(*bad); raise AssertionError("expected ValueError")
        except ValueError:
            pass
    # held triple over a frame: steering drift per frame bounded by the rate over a random command stream
    rng = np.random.default_rng(0); prev = 0.0; worst = 0.0
    for _ in range(2000):
        s, t, b = hold_clip(rng.uniform(-2, 2, 3), prev); worst = max(worst, abs(s - prev)); prev = s
        assert -1 <= s <= 1 and 0 <= t <= 1 and 0 <= b <= 1
    assert worst <= 0.1 + 1e-12
    report["hold_clip"] = {"max_steer_step": worst, "ok": True}

    # (2) OUPerturb ---------------------------------------------------------------------------------------------
    rng = np.random.default_rng(1)
    n = 20000
    follower = np.stack([rng.uniform(-0.6, 0.6, n), rng.uniform(0, 1, n), np.zeros(n)], 1)
    brake_rows = rng.uniform(size=n) < 0.15          # the follower brakes in 15 % of frames (throttle 0 then)
    follower[brake_rows, 1] = 0.0; follower[brake_rows, 2] = rng.uniform(0.1, 1.0, brake_rows.sum())
    assert abs(OUPerturb(seed=1).brake_p - DEFAULT_BRAKE_P) == 0.0 and abs(DEFAULT_BRAKE_P - 0.0037) < 2e-4
    assert OUPerturb.brake_p_for_rate(1.4 / 20.0) == DEFAULT_BRAKE_P == brake_p_for_rate(1.4 / 20.0)
    ou_a, ou_b = OUPerturb(seed=7, brake_p=0.05), OUPerturb(seed=7, brake_p=0.05)   # the 0.05 statistics kept for reference
    ou_c = OUPerturb(seed=8, brake_p=0.05)
    seq_a = np.stack([ou_a.step(f) for f in follower]); seq_b = np.stack([ou_b.step(f) for f in follower])
    seq_c = np.stack([ou_c.step(f) for f in follower])
    assert np.array_equal(seq_a, seq_b), "OUPerturb not deterministic for equal seeds"
    assert not np.array_equal(seq_a, seq_c), "different seeds gave identical streams"
    assert ((seq_a[:, 1] > 0) & (seq_a[:, 2] > 0)).sum() == 0, "throttle and brake both positive"
    assert (seq_a[:, 0] >= -1).all() and (seq_a[:, 0] <= 1).all() and (seq_a[:, 1:] >= 0).all() and (seq_a[:, 1:] <= 1).all()
    d_steer = seq_a[:, 0] - follower[:, 0]
    free = (follower[:, 2] == 0) & (seq_a[:, 2] == 0)
    d_thr = seq_a[free, 1] - follower[free, 1]
    assert np.abs(d_steer).max() <= 0.15 + 1e-12 and np.abs(d_thr).max() <= 0.25 + 1e-12
    interior = (np.abs(follower[:, 0]) < 0.8)
    assert np.abs(d_steer[interior]).max() <= 0.15 + 1e-12
    # OU statistics: stationary sd and lag-1 autocorrelation exp(-dt/tau) on the unclipped-ish steering perturbation
    ds = d_steer[interior]
    ac1 = np.corrcoef(ds[:-1], ds[1:])[0, 1]
    # single-frame exact bound of the step: with the exact discretisation |x_{k+1} - a x_k| <= 4 sd sqrt(1-a^2) w.h.p.
    ou_a.reset(); seq_r = np.stack([ou_a.step(f) for f in follower])
    assert np.array_equal(seq_r, seq_a), "reset() did not replay the stream"
    summ = ou_a.summary()
    taps = summ["taps"]
    assert taps > 0 and summ["braked_fraction"] > 0.15
    # every tap zeroes throttle for its whole duration and lasts 10-20 frames when it starts from a free frame
    tap_flags, lefts, cmds = [], [], []
    ou_t = OUPerturb(seed=3, brake_p=0.05)
    for f in np.stack([np.zeros(n), np.full(n, 0.4), np.zeros(n)], 1):
        cmds.append(ou_t.step(f)); tap_flags.append(ou_t.last["tap_active"]); lefts.append(ou_t.tap_left)
    tap_flags, lefts, cmds = np.asarray(tap_flags), np.asarray(lefts), np.stack(cmds)
    # a tap starts at a frame where the remaining-frame counter jumps up; its drawn length is that counter + 1
    starts = np.flatnonzero(np.diff(np.r_[0, lefts]) > 0)
    lengths = lefts[starts] + 1
    assert len(starts) == ou_t.n_taps and lengths.min() >= 10 and lengths.max() <= 20, (lengths.min(), lengths.max())
    assert (cmds[tap_flags, 1] == 0.0).all() and (cmds[tap_flags, 2] >= 0.3).all()      # throttle 0 during every tap
    assert (cmds[~tap_flags, 2] == 0.0).all()                                              # no brake outside taps here
    runs = lengths
    # the collector chain: perturbation -> hold_clip keeps the per-frame steering step <= 0.1 and the channels exclusive
    ou_h = OUPerturb(seed=4); prev = 0.0; worst_h = 0.0
    for f in follower[:5000]:
        s_, t_, b_ = hold_clip(ou_h.step(f), prev); worst_h = max(worst_h, abs(s_ - prev)); prev = s_
        assert not (t_ > 0 and b_ > 0)
    assert worst_h <= 0.1 + 1e-12
    # tap-rate helper: the requested rate is met to within 25 % over 200 x 400-frame episodes at the PLAN B3 sizing
    p_frame = OUPerturb.brake_p_for_rate(1.4 / 20.0)
    got = []
    for ep in range(200):
        ou_r = OUPerturb(seed=100 + ep)                 # the DEFAULT brake_p is that sizing
        assert ou_r.brake_p == p_frame
        for _ in range(400):
            ou_r.step((0.0, 0.4, 0.0))
        got.append(ou_r.n_taps)
    assert abs(np.mean(got) - 1.4) < 0.35, (p_frame, np.mean(got))
    report["ou_perturb"] = {"default_brake_p": DEFAULT_BRAKE_P, "brake_p_for_1p4_taps_per_20s": p_frame, "taps_per_20s_at_default_p": float(np.mean(got)),
                            "reference_stats_at_brake_p": 0.05,
                            "max_steer_step_after_hold_clip": worst_h,"frames": n, "taps": taps, "braked_fraction": summ["braked_fraction"],
                            "steer_perturb_sd": float(ds.std()), "steer_lag1_autocorr": float(ac1),
                            "expected_lag1": math.exp(-0.05 / 0.5), "steer_at_bound_fraction": float((np.abs(ds) >= 0.15 - 1e-12).mean()),
                            "tap_len_frames_min_max": [int(runs.min()), int(runs.max())],
                            "mean_frames_between_tap_starts": float(n / max(taps, 1))}

    # (3) NumpyActor vs torch ------------------------------------------------------------------------------------
    import torch
    torch.manual_seed(0)
    num_obs, hidden = 158, [512, 256, 128]
    try:
        from rsl_rl.modules import ActorCritic, EmpiricalNormalization
        ac = ActorCritic(num_obs, num_obs, 3, hidden, hidden, "elu", 0.7)
        norm = EmpiricalNormalization(shape=[num_obs])
        src = "rsl_rl"
    except ImportError:  # pragma: no cover - the nedm env has rsl_rl
        ac, norm, src = None, None, "fallback"
    if ac is None:
        class _AC(torch.nn.Module):
            def __init__(self):
                super().__init__()
                dims = [num_obs] + hidden
                layers = []
                for a, b in zip(dims[:-1], dims[1:]):
                    layers += [torch.nn.Linear(a, b), torch.nn.ELU()]
                layers.append(torch.nn.Linear(dims[-1], 3))
                self.actor = torch.nn.Sequential(*layers)
            def act_inference(self, o):
                return self.actor(o)
        ac = _AC()
        class _Norm(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.eps = 1e-2
                self.register_buffer("_mean", torch.zeros(1, num_obs)); self.register_buffer("_var", torch.ones(1, num_obs))
                self.register_buffer("_std", torch.ones(1, num_obs))
            def update(self, x):
                self._mean = x.mean(0, keepdim=True); self._var = x.var(0, unbiased=False, keepdim=True); self._std = self._var.sqrt()
            def forward(self, x):
                return (x - self._mean) / (self._std + self.eps)
        norm = _Norm()
    with torch.no_grad():
        for p in ac.actor.parameters():
            p.mul_(3.0)                                   # larger weights: exercise the nonlinearity and the tanh
    fit = torch.randn(4096, num_obs) * torch.linspace(0.05, 3.0, num_obs) + torch.linspace(-1, 1, num_obs)
    norm.train(); norm.update(fit); norm.eval(); ac.eval()
    meta = {"action_center": "dataset_mean", "act_mean": [-0.003, 0.2, 0.02], "action_scale": [1.0, 0.7, 0.5],
            "action_low": [-1.0, 0.0, 0.0], "action_high": [1.0, 1.0, 1.0], "num_obs": num_obs,
            "obs_layout": PolicyObs({"waypoints": [[0, 0], [1, 0], [2, 0]], "speeds": [1, 1, 1], "stations": [0, 1, 2],
                                     "headings": [0, 0, 0]}).layout(), "policy": {"activation": "elu"}}
    npz_path = out_dir / "selftest_actor.npz"
    export_torch_actor(ac, norm, meta, npz_path)
    actor = NumpyActor.from_npz(npz_path)
    obs = torch.randn(512, num_obs) * torch.linspace(0.05, 3.0, num_obs) + torch.linspace(-1, 1, num_obs)
    with torch.no_grad():
        ref = ac.act_inference(norm(obs)).double().numpy()
        ref_a = np.clip(np.asarray(meta["act_mean"]) + np.asarray(meta["action_scale"]) * np.tanh(ref),
                        meta["action_low"], meta["action_high"])
    mine = actor.act(obs.numpy())
    single = np.stack([actor.act(o) for o in obs.numpy()[:16]])
    diff_act = float(np.abs(mine - ref_a).max()); diff_pre = float(np.abs(actor.mlp(obs.numpy()) - ref).max())
    assert diff_act < 1e-5, diff_act
    assert np.allclose(single, mine[:16], atol=1e-12, rtol=0)      # single vs batched: BLAS summation order only
    assert actor.meta["obs_layout"]["num_obs"] == 158 and actor.num_obs == num_obs and actor.activation == "elu"
    # a second activation type and the pre-squash path
    ac2 = torch.nn.Sequential(torch.nn.Linear(num_obs, 64), torch.nn.Tanh(), torch.nn.Linear(64, 3))
    class _Wrap(torch.nn.Module):
        def __init__(self, a):
            super().__init__(); self.actor = a
    p2 = out_dir / "selftest_actor_tanh.npz"
    export_torch_actor(_Wrap(ac2), norm, {**meta, "action_center": [0.0, 0.5, 0.5], "action_scale": [1.0, 0.5, 0.5]}, p2)
    a2 = NumpyActor.from_npz(p2)
    with torch.no_grad():
        ref2 = ac2(norm(obs)).double().numpy()
    diff2 = float(np.abs(a2.mlp(obs.numpy()) - ref2).max())
    assert diff2 < 1e-5, diff2
    assert np.allclose(a2.act(obs.numpy()), np.clip(np.array([0, .5, .5]) + np.array([1, .5, .5]) * np.tanh(ref2), [-1, 0, 0], [1, 1, 1]),
                       atol=1e-5, rtol=0)
    report["numpy_actor"] = {"source": src, "hidden": hidden, "num_obs": num_obs, "n_obs_tested": 512,
                             "max_abs_action_diff": diff_act, "max_abs_presquash_diff": diff_pre,
                             "tanh_net_max_abs_presquash_diff": diff2, "fraction_clipped": float((mine != np.asarray(meta["act_mean"]) + np.asarray(meta["action_scale"]) * np.tanh(actor.mlp(obs.numpy()))).mean()),
                             "npz": str(npz_path)}

    # (4) sample_continuations ------------------------------------------------------------------------------------
    cases = [((-20.0, 5.0, 0.3), (25.0, -10.0)), ((10.0, -30.0, 2.0), (-15.0, 20.0)), ((0.0, 0.0, 3.0), (30.0, 0.0)),
             ((25.0, 25.0, -2.3), (-20.0, -25.0))]
    unsolvable = ((33.0, 33.0, 0.7), (-20.0, -25.0))      # arena corner facing outward: no base route validates
    try:
        sample_continuations(*unsolvable, 3, seed=11, tries_factor=8)
        raise AssertionError("expected RuntimeError for the unsolvable pose")
    except RuntimeError:
        pass
    cont_report = []
    for pose, goal in cases:
        routes = sample_continuations(pose, goal, 3, seed=11)
        routes2 = sample_continuations(pose, goal, 3, seed=11)
        assert [r["meta"]["route_sha256"] for r in routes] == [r["meta"]["route_sha256"] for r in routes2]
        assert len({r["meta"]["route_sha256"] for r in routes}) == 3
        for r in routes:
            assert planner_validator(r, pose)
            check_reference_contract(r)
            xy = r["waypoints"]
            assert np.linalg.norm(xy[0] - np.asarray(pose[:2])) <= GOAL_TOL_M and np.linalg.norm(xy[-1] - np.asarray(goal)) <= GOAL_TOL_M
            assert r["speeds"].min() >= 0.0 and r["speeds"].max() <= 6.0 + 1e-9
            j = route_to_json(r); json.dumps(j)
        cont_report.append({"pose": list(pose), "goal": list(goal), "base": routes[0]["meta"]["base"],
                            "draws": routes[-1]["meta"]["draws"],
                            "mean_speeds": [float(r["meta"]["mean_speed_mps"]) for r in routes],
                            "max_lateral_m": [float(r["meta"]["max_lateral_m"]) for r in routes],
                            "lengths_m": [float(r["stations"][-1]) for r in routes]})
    # a validator that rejects everything must raise
    try:
        sample_continuations(cases[0][0], cases[0][1], 2, seed=1, validator=lambda r, p: False, tries_factor=3)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    # (4b) speed floor from the vehicle's speed (v0) and the start-heading acceptance (verifier problems 1 and 2)
    import f104_n2_sampler as S
    assert float(S.V_MAX) == 6.0 == float(planner_cfg().max_speed_mps) and float(S.A_DEC) == 2.0
    assert np.allclose(speed_floor_from(4.0, [0.0, 1.0, 4.0, 5.0]), [4.0, np.sqrt(12.0), 0.0, 0.0])
    assert np.allclose(speed_floor_from(7.0, [0.0, 2.0, 4.0]), [6.0, 6.0, np.sqrt(33.0)]) and speed_floor_from(-3.0, [0., 1.]).max() == 0.0
    assert abs(start_heading_err_deg({"headings": [3.1]}, -3.1) - math.degrees(2 * math.pi - 6.2)) < 1e-9
    floor_report = []
    for pose, goal in cases:
        for v0 in (0.0, 3.3, 5.0, 7.0):
            routes = sample_continuations(pose, goal, 3, seed=11, v0=v0)
            for r in routes:
                assert planner_validator(r, pose); check_reference_contract(r)
                floor = speed_floor_from(v0, r["stations"], S.A_DEC, S.V_MAX)
                assert (r["speeds"] >= floor - 1e-9).all() and r["speeds"].max() <= 6.0 + 1e-9
                assert r["speeds"][0] >= min(v0, 6.0) - 1e-9 and r["meta"]["start_speed_mps"] == float(r["speeds"][0])
                assert start_heading_err_deg(r, pose[2]) <= 15.0 + 1e-9 and r["meta"]["start_heading_err_deg"] <= 15.0
                assert r["meta"]["v0_mps"] == v0 and r["meta"]["max_start_heading_err_deg"] == 15.0
            capped = sample_continuations(pose, goal, 3, seed=11, v0=v0, v0_accel_cap=True)
            for r in capped:
                assert planner_validator(r, pose); check_reference_contract(r)
                st = r["stations"]; cap = np.sqrt(min(max(v0, 0.0), 6.0) ** 2 + 2 * S.A_ACC * (st - st[0]))
                assert (r["speeds"] <= cap + 1e-9).all() and (r["speeds"] >= speed_floor_from(v0, st, S.A_DEC, S.V_MAX) - 1e-9).all()
                assert abs(float(r["speeds"][0]) - min(v0, 6.0)) < 1e-9 and r["meta"]["v0_accel_cap"] is True
            floor_report.append({"pose": list(pose), "v0": v0, "start_speeds": [float(r["speeds"][0]) for r in routes],
                                 "capped_start_speeds": [float(r["speeds"][0]) for r in capped],
                                 "capped_draws": capped[-1]["meta"]["draws"], "capped_lowered_points": [r["meta"]["speed_cap_lowered_points"] for r in capped],
                                 "raised_points": [r["meta"]["speed_floor_raised_points"] for r in routes],
                                 "heading_err_deg": [round(r["meta"]["start_heading_err_deg"], 2) for r in routes],
                                 "draws": routes[-1]["meta"]["draws"]})
        # the acceptance test can be switched off (pre-fix behaviour: no 'start_heading' rejections, deterministic)
        off1 = sample_continuations(pose, goal, 3, seed=11, max_start_heading_err_deg=None)
        off2 = sample_continuations(pose, goal, 3, seed=11, max_start_heading_err_deg=None)
        assert [r["meta"]["route_sha256"] for r in off1] == [r["meta"]["route_sha256"] for r in off2]
        assert all(r["meta"]["max_start_heading_err_deg"] is None and r["meta"]["v0_mps"] is None for r in off1)
    # a tolerance nothing can meet must raise (not loop forever)
    try:
        sample_continuations(cases[0][0], cases[0][1], 2, seed=1, max_start_heading_err_deg=-1.0, tries_factor=3)
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
    # cross-checks against the torch-bearing originals (self-test only)
    from dataclasses import asdict
    import gen_planner as GP
    from nedm.traverse import fdm_diverse_planner as FDP
    assert asdict(planner_cfg()) == asdict(GP.CFG), "validator config differs from gen_planner.CFG"
    assert not GP.safe_validate(GP.base_route(*unsolvable), [], GP.CFG, np.asarray(unsolvable[0]))["valid"]
    base_same = 0
    for pose, goal in cases:
        a, b = base_route(pose, goal), GP.base_route(pose, goal)
        assert all(np.array_equal(np.asarray(a[k], float), np.asarray(b[k], float)) for k in ("waypoints", "speeds", "stations", "headings"))
        base_same += 1
    contract_same = 0
    for pose, goal in cases:
        for r in sample_continuations(pose, goal, 3, seed=5):
            FDP.check_reference_contract(r); contract_same += 1
            assert GP.safe_validate(r, [], GP.CFG, np.asarray(pose, float))["valid"]
    bad = {"waypoints": np.array([[0, 0], [1, 0], [0.5, 0.1], [2, 0]], float), "stations": np.array([0, 1, 1.5, 2.]),
           "headings": np.zeros(4), "speeds": np.ones(4)}
    for fn in (check_reference_contract, FDP.check_reference_contract):
        try:
            fn(bad); raise AssertionError("expected ValueError")
        except ValueError:
            pass
    # randomised poses/goals over the arena: the copied base_route (incl. the wider-Hermite and arc-then-straight
    # fallbacks) must be bitwise equal to gen_planner.base_route
    rrng = np.random.default_rng(20260921)
    n_rand = n_fallback = 0
    while n_rand < 60:
        pose = np.r_[rrng.uniform(-36, 36, 2), rrng.uniform(-np.pi, np.pi)]; goal = rrng.uniform(-36, 36, 2)
        if np.linalg.norm(goal - pose[:2]) < 3.0:
            continue
        a, b = base_route(pose, goal), GP.base_route(pose, goal)
        assert a["meta"] == b["meta"] and all(np.array_equal(np.asarray(a[k], float), np.asarray(b[k], float))
                                              for k in ("waypoints", "speeds", "stations", "headings")), (pose, goal)
        n_rand += 1; n_fallback += a["meta"] != {"start_tangent_scale": 1.0}
    base_same += n_rand
    # recorded moving-prefix poses of the local CRM demo episodes (read-only, optional): cuts at 2, 4, 6 s
    real_cases = []
    demo = ROOT / "artifacts/traverse/crm_f104_v1/demo_v1"
    for rd in sorted(demo.glob("run_*")) if demo.exists() else []:
        if not (rd / "trajectory.npz").exists() or not (rd / "case.json").exists():
            continue
        with np.load(rd / "trajectory.npz") as z:
            pose_rec = np.asarray(z["pose"], float); vx_rec = np.asarray(z["state"], float)[:, 0]
        goal_rec = np.asarray(json.loads((rd / "case.json").read_text())["goal_xy"], float)
        for cut_s in (2.0, 4.0, 6.0):
            F = int(round(cut_s / CTRL_DT_S))
            if F >= len(pose_rec):
                continue
            seed_a = int(hashlib.md5(f"{rd.name}:{F}".encode()).hexdigest()[:8], 16)
            v0 = float(vx_rec[F])
            before = sample_continuations(pose_rec[F], goal_rec, 3, seed=seed_a, max_start_heading_err_deg=None)   # pre-fix sampler
            rs = sample_continuations(pose_rec[F], goal_rec, 3, seed=seed_a, v0=v0)
            for r in rs:
                assert planner_validator(r, pose_rec[F]); check_reference_contract(r)
                assert r["speeds"][0] >= min(v0, 6.0) - 1e-9 and r["meta"]["start_heading_err_deg"] <= 15.0
            real_cases.append({"run": rd.name, "cut_s": cut_s, "vx": v0, "draws": rs[-1]["meta"]["draws"],
                               "base": rs[0]["meta"]["base"], "lengths_m": [float(r["stations"][-1]) for r in rs],
                               "before_fix": {"draws": before[-1]["meta"]["draws"],
                                              "start_speeds": [float(r["speeds"][0]) for r in before],
                                              "heading_err_deg": [round(start_heading_err_deg(r, pose_rec[F][2]), 2) for r in before],
                                              "n_below_v0": int(sum(float(r["speeds"][0]) < min(v0, 6.0) - 1e-9 for r in before)),
                                              "n_over_15deg": int(sum(start_heading_err_deg(r, pose_rec[F][2]) > 15.0 for r in before))},
                               "after_fix": {"start_speeds": [float(r["speeds"][0]) for r in rs],
                                             "heading_err_deg": [round(r["meta"]["start_heading_err_deg"], 2) for r in rs],
                                             "raised_points": [r["meta"]["speed_floor_raised_points"] for r in rs]}})
    # the A4 anchor list (read-only, optional): a deterministic 15-anchor subset of anchors_crm.json (pose_F, vx_F, goal_xy)
    anchors_file = ROOT / "artifacts/traverse/generalist_20260921/A_adapt/a4/anchors/anchors_crm.json"
    crm_anchor_cases = []
    if anchors_file.exists():
        anchors = json.loads(anchors_file.read_text())
        pick = anchors[::max(1, len(anchors) // 15)][:15]
        for a in pick:
            pose_a, goal_a, v0 = np.asarray(a["pose_F"], float), np.asarray(a["goal_xy"], float), float(a["vx_F"])
            seed_a = int(hashlib.md5(a["anchor_id"].encode()).hexdigest()[:8], 16)
            rec = {"anchor_id": a["anchor_id"], "vx": v0, "goal_dist_m": float(np.linalg.norm(goal_a - pose_a[:2]))}
            try:
                before = sample_continuations(pose_a, goal_a, 3, seed=seed_a, max_start_heading_err_deg=None)
                rs = sample_continuations(pose_a, goal_a, 3, seed=seed_a, v0=v0)
            except (RuntimeError, ValueError) as exc:
                rec["unsolvable"] = str(exc)[:120]; crm_anchor_cases.append(rec); continue
            for r in rs:
                assert planner_validator(r, pose_a); check_reference_contract(r)
                assert r["speeds"][0] >= min(v0, 6.0) - 1e-9 and r["meta"]["start_heading_err_deg"] <= 15.0
            capped = sample_continuations(pose_a, goal_a, 3, seed=seed_a, v0=v0, v0_accel_cap=True)
            assert all(abs(float(r["speeds"][0]) - min(v0, 6.0)) < 1e-9 and planner_validator(r, pose_a) for r in capped)
            rec.update({"draws_before": before[-1]["meta"]["draws"], "draws_after": rs[-1]["meta"]["draws"],
                        "draws_after_with_accel_cap": capped[-1]["meta"]["draws"],
                        "after_start_above_v0_mps": [round(float(r["speeds"][0]) - min(v0, 6.0), 3) for r in rs],
                        "before_n_below_v0": int(sum(float(r["speeds"][0]) < min(v0, 6.0) - 1e-9 for r in before)),
                        "before_n_over_15deg": int(sum(start_heading_err_deg(r, pose_a[2]) > 15.0 for r in before)),
                        "after_start_speeds": [float(r["speeds"][0]) for r in rs],
                        "after_heading_err_deg": [round(r["meta"]["start_heading_err_deg"], 2) for r in rs]})
            crm_anchor_cases.append(rec)
    report["continuations"] = {"cases": cont_report, "base_routes_identical_to_gen_planner": base_same,
                               "random_cases": n_rand, "random_cases_using_fallback_shape": int(n_fallback),
                               "contract_agreements_with_fdm_diverse_planner": contract_same,
                               "v0_floor_and_heading_cases": floor_report,
                               "recorded_crm_demo_poses": real_cases,
                               "recorded_crm_demo_max_draws": max([c["draws"] for c in real_cases], default=None),
                               "recorded_crm_demo_summary": {
                                   "anchors": len(real_cases), "continuations": 3 * len(real_cases),
                                   "before_fix_below_v0": int(sum(c["before_fix"]["n_below_v0"] for c in real_cases)),
                                   "before_fix_over_15deg": int(sum(c["before_fix"]["n_over_15deg"] for c in real_cases)),
                                   "after_fix_at_or_above_v0": int(sum(sum(s >= min(c["vx"], 6.0) - 1e-9 for s in c["after_fix"]["start_speeds"]) for c in real_cases)),
                                   "after_fix_within_15deg": int(sum(sum(h <= 15.0 for h in c["after_fix"]["heading_err_deg"]) for c in real_cases)),
                                   "max_draws_after_fix": max([c["draws"] for c in real_cases], default=None)},
                               "a4_anchor_list_subset": crm_anchor_cases,
                               "a4_anchor_list_summary": {
                                   "anchors": len(crm_anchor_cases), "unsolvable": int(sum("unsolvable" in c for c in crm_anchor_cases)),
                                   "continuations": 3 * sum("unsolvable" not in c for c in crm_anchor_cases),
                                   "before_fix_below_v0": int(sum(c.get("before_n_below_v0", 0) for c in crm_anchor_cases)),
                                   "before_fix_over_15deg": int(sum(c.get("before_n_over_15deg", 0) for c in crm_anchor_cases)),
                                   "after_fix_at_or_above_v0": int(sum(sum(s >= min(c["vx"], 6.0) - 1e-9 for s in c.get("after_start_speeds", [])) for c in crm_anchor_cases)),
                                   "after_fix_within_15deg": int(sum(sum(h <= 15.0 for h in c.get("after_heading_err_deg", [])) for c in crm_anchor_cases)),
                                   "max_draws_after_fix": max([c["draws_after"] for c in crm_anchor_cases if "draws_after" in c], default=None),
                                   "max_draws_after_fix_with_accel_cap": max([c["draws_after_with_accel_cap"] for c in crm_anchor_cases if "draws_after_with_accel_cap" in c], default=None)}}

    # (5) make_follower vs the frozen make_driver, with stub Chrono modules ------------------------------------------
    class _Vec(list):
        pass
    calls: list = []
    class _Steer:
        def SetLookAheadDistance(self, d): calls.append(("look_ahead", float(d)))
        def SetGains(self, *g): calls.append(("steer_gains", tuple(float(v) for v in g)))
    class _Speed:
        def SetGains(self, *g): calls.append(("speed_gains", tuple(float(v) for v in g)))
    class _Driver:
        def __init__(self, vehicle, curve, name, speed):
            calls.append(("ctor", vehicle, tuple(curve), name, float(speed))); self.s, self.v = _Steer(), _Speed()
        def GetSteeringController(self): return self.s
        def GetSpeedController(self): return self.v
        def Initialize(self): calls.append(("Initialize",))
    class _Chrono:
        vector_ChVector3d = staticmethod(lambda: _Vec())
        ChVector3d = staticmethod(lambda x, y, z: (x, y, z))
        ChBezierCurve = staticmethod(lambda pts: tuple(pts))
    class _Veh:
        ChPathFollowerDriver = _Driver
    class _TMap:
        def height(self, x, y): return 0.1 * x - 0.05 * y + math.sin(0.3 * x)
    import traverse_fdm_rgbd_diverse_chrono as frozen
    route = sample_continuations((-20.0, 5.0, 0.3), (25.0, -10.0), 1, seed=2)[0]
    tm = _TMap()
    calls.clear(); frozen.make_driver(_Chrono, _Veh, "veh", route, tm); ref_calls = list(calls)
    calls.clear(); make_follower(_Veh, _Chrono, "veh", route, tm.height); mine_calls = list(calls)
    assert ref_calls == mine_calls, "make_follower differs from the frozen make_driver"
    calls.clear(); make_follower(_Veh, _Chrono, "veh", route, tm.height, initialize=False)
    assert calls == ref_calls[:-1] and ref_calls[-1] == ("Initialize",)
    pts = follower_points(route, tm.height)
    st = np.asarray(route["stations"])
    kept = [i for i, (x, y) in enumerate(route["waypoints"]) if any(abs(x - p[0]) < 1e-12 and abs(y - p[1]) < 1e-12 for p in pts)]
    assert kept[0] == 0 and kept[-1] == len(st) - 1 and (np.diff(st[kept])[:-1] >= 2.0).all()
    report["make_follower"] = {"identical_call_sequence": True, "n_route_points": int(len(st)), "n_follower_points": len(pts),
                               "min_kept_spacing_m": float(np.diff(st[kept]).min()), "calls": [c[0] for c in ref_calls]}

    # (6) PolicyObs ---------------------------------------------------------------------------------------------
    import traverse_wp3_chrono_eval as W
    route = sample_continuations((-20.0, 5.0, 0.3), (25.0, -10.0), 1, seed=3)[0]
    rng = np.random.default_rng(5)
    T = 120
    xy = np.asarray(route["waypoints"]); hd = np.asarray(route["headings"]); n = len(xy)
    idx = np.minimum((np.arange(T) * 0.7).astype(int), n - 1)
    poses = np.stack([xy[idx, 0] + rng.normal(0, 0.3, T), xy[idx, 1] + rng.normal(0, 0.3, T), hd[idx] + rng.normal(0, 0.1, T)], 1)
    states = rng.normal(0, 1, (T, 17)); states[0] = rng.normal(0, 0.01, 17)         # settled rest state at frame 0
    actions = np.clip(rng.normal([0, 0.3, 0.1], 0.2, (T, 3)), [-1, 0, 0], [1, 1, 1])
    po = PolicyObs(route)
    assert po.num_obs == 158 and po.layout()["past_states"] == [62, 158]
    po.reset(states[0])
    rt = W.RouteTracker(route, None)
    max_diff = 0.0
    for k in range(T):
        last = SETTLE_ACTION if k == 0 else actions[k - 1]
        obs = po.observe(poses[k], states[k], last)
        err = rt.update(*poses[k], states[k, 0], first=(k == 0))
        ref = np.concatenate([[err["e_along"] / 10.0, err["e_ct"] / 10.0, err["e_h"] / math.pi], rt.preview(*poses[k]),
                              [states[k, 0] / 10.0, states[k, 6]], last])
        ab, sb = history_blocks(states, actions, k)
        full = np.concatenate([ref, ab.reshape(-1), sb.reshape(-1)])
        max_diff = max(max_diff, float(np.abs(obs - full).max()))
        assert obs.shape == (158,) and np.isfinite(obs).all()
        if k == 0:
            assert np.array_equal(obs[38:62].reshape(8, 3), np.tile(SETTLE_ACTION, (8, 1)))
            assert np.array_equal(obs[62:].reshape(8, 12), np.tile(states[0][list(OBSERVABLE_COLS)], (8, 1)))
            assert np.array_equal(obs[35:38], SETTLE_ACTION)
        if 0 < k < 8:
            assert np.array_equal(obs[62:].reshape(8, 12)[:8 - k - 1], np.tile(states[0][list(OBSERVABLE_COLS)], (8 - k - 1, 1)))
            assert np.array_equal(obs[38:62].reshape(8, 3)[:8 - k], np.tile(SETTLE_ACTION, (8 - k, 1)))
        assert np.array_equal(obs[38:62].reshape(8, 3)[-1], np.asarray(last, float))
        assert np.array_equal(obs[62:].reshape(8, 12)[-1], states[k][list(OBSERVABLE_COLS)])
        po.push(states[k], actions[k])
    assert max_diff == 0.0, max_diff
    # branch swap: seed the history from a prefix and compare with the offline twin at frame F
    F = 40
    po2 = PolicyObs(route); po2.seed_history(states[:F], actions[:F])
    obs2 = po2.observe(poses[F], states[F], actions[F - 1])
    ab, sb = history_blocks(states, actions, F)
    assert np.array_equal(obs2[38:62].reshape(8, 3), ab) and np.array_equal(obs2[62:].reshape(8, 12), sb)
    # short prefix padding
    po3 = PolicyObs(route); po3.seed_history(states[:3], actions[:3])
    obs3 = po3.observe(poses[3], states[3], actions[2])
    ab, sb = history_blocks(states, actions, 3)
    assert np.array_equal(obs3[38:62].reshape(8, 3), ab) and np.array_equal(obs3[62:].reshape(8, 12), sb)
    # from_meta round trip and normalised states
    lay = po.layout(); lay["state_mean"] = list(range(12)); lay["state_std"] = [2.0] * 12
    po4 = PolicyObs.from_meta(route, {"obs_layout": lay}); po4.reset(states[0])
    o4 = po4.observe(poses[0], states[0], SETTLE_ACTION)
    assert np.allclose(o4[62:].reshape(8, 12), (np.tile(states[0][list(OBSERVABLE_COLS)], (8, 1)) - np.arange(12)) / 2.0)
    assert PolicyObs.from_meta(route, actor.meta).num_obs == 158
    report["policy_obs"] = {"frames": T, "num_obs": 158, "max_abs_diff_vs_wp3_tracker_plus_offline_history": max_diff,
                            "layout": po.layout()}

    report["runtime_s"] = time.time() - t0
    report["torch_loaded_by_module_import"] = False
    (out_dir / "gc_control_selftest.json").write_text(json.dumps(report, indent=2, default=_jsonable) + "\n")
    return report


def _torch_free_import_check() -> bool:
    """True if importing this module (and calling its torch-free entry points) leaves torch unloaded."""
    import subprocess
    code = ("import sys; sys.path[:0] = [%r, %r]; import gc_control as G; import numpy as np\n"
            "G.sample_continuations((-20., 5., .3), (25., -10.), 2, 1); G.hold_clip((0, 0, 1), 0.); G.OUPerturb(1).step((0, .2, 0))\n"
            "G.PolicyObs({'waypoints': [[0, 0], [1, 0], [2, 0]], 'speeds': [1, 1, 1], 'stations': [0, 1, 2], 'headings': [0, 0, 0]})\n"
            "print('torch' in sys.modules)" % (str(ROOT / 'src'), str(HERE)))
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return res.stdout.strip().splitlines()[-1] == "False"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "artifacts/traverse/generalist_20260921/C_collectors/selftest"))
    args = ap.parse_args()
    if not args.selftest:
        ap.print_help(); sys.exit(0)
    rep = _selftest(Path(args.out))
    torch_free = _torch_free_import_check()
    rep["torch_free_import"] = torch_free
    (Path(args.out) / "gc_control_selftest.json").write_text(json.dumps(rep, indent=2, default=_jsonable) + "\n")
    assert torch_free, "importing gc_control / calling its collector entry points loaded torch"
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ("layout", "cases", "recorded_crm_demo_poses", "v0_floor_and_heading_cases", "a4_anchor_list_subset")})
                      for k, v in rep.items()}, indent=1, default=_jsonable))
    print("gc_control self-test OK")
