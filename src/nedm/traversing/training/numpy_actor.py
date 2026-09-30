"""The PPO tracking policy exported to NumPy (``actor.npz``): observation normaliser + actor MLP + action squash.

    act(obs) = clip(center + scale * tanh(mlp((obs - mean) / (sqrt(var) + eps))), low, high)

The normaliser is rsl_rl's ``EmpiricalNormalization`` in eval mode (eps 1e-2); the MLP is ``ActorCritic.actor``
(Linear, ELU, ..., Linear; its output is the mean action ``act_inference`` returns); the squash is
``TrackingEnv._scale_policy_actions``. The steering-rate clamp is not applied here (the controller applies it).
Weights are float64. This module imports torch only inside ``export_torch_actor``, so a deployed controller can load
the actor without torch:

    actor = NumpyActor.from_npz("artifacts/traverse/generalist_20260921/B_tracker/ppo_v2/actor.npz")
    steer_throttle_brake = actor.act(obs)          # obs (158,) or (B, 158)

npz keys (format ``gc_actor_v1``): ``format``, ``num_obs``, ``num_actions``, ``obs_mean``, ``obs_var``, ``obs_eps``,
``n_layers``, ``W0, b0, ..., W{n-1}, b{n-1}`` (torch layout, W is (out, in)), ``activation``, ``action_center``,
``action_scale``, ``action_low``, ``action_high``, ``meta_json`` (the policy meta, incl. ``obs_layout``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ACTOR_FORMAT = "gc_actor_v1"


def _elu(x):
    return np.where(x > 0.0, x, np.expm1(np.minimum(x, 0.0)))


def jsonable(v):
    """``json.dumps(default=...)`` for numpy values."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    raise TypeError(f"not JSON serialisable: {type(v).__name__}")


class NumpyActor:
    """The exported tracking policy evaluated in NumPy (float64); ``act`` takes one observation or a batch."""

    def __init__(self, obs_mean, obs_var, layers, activation: str, action_center, action_scale, action_low,
                 action_high, obs_eps: float = 1e-2, meta: dict | None = None):
        if activation != "elu":
            raise ValueError(f"unsupported activation {activation!r} (the tracker uses elu)")
        self.obs_mean = np.asarray(obs_mean, np.float64).reshape(-1)
        self.obs_var = np.asarray(obs_var, np.float64).reshape(-1)
        self.obs_eps = float(obs_eps)
        self.layers = [(np.asarray(W, np.float64), np.asarray(b, np.float64).reshape(-1)) for W, b in layers]
        self.activation = activation
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
            if str(z["format"]) != ACTOR_FORMAT:
                raise ValueError(f"unexpected actor format {str(z['format'])!r}")
            layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(int(z["n_layers"]))]
            return cls(z["obs_mean"], z["obs_var"], layers, str(z["activation"]), z["action_center"], z["action_scale"],
                       z["action_low"], z["action_high"], float(z["obs_eps"]), json.loads(str(z["meta_json"])))

    def mlp(self, obs) -> np.ndarray:
        """Pre-squash actor output (the torch ``act_inference`` value)."""
        o = np.asarray(obs, np.float64)
        if o.shape[-1] != self.num_obs:
            raise ValueError(f"observation width {o.shape[-1]} != {self.num_obs}")
        h = (o - self.obs_mean) / self._std_eps
        last = len(self.layers) - 1
        for i, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if i < last:
                h = _elu(h)
        return h

    def act(self, obs) -> np.ndarray:
        return np.clip(self.center + self.scale * np.tanh(self.mlp(obs)), self.low, self.high)


def export_torch_actor(actor_critic, normalizer, meta: dict, path) -> dict:
    """Write the ``NumpyActor`` npz from an rsl_rl ``ActorCritic`` and its ``EmpiricalNormalization``. ``meta`` must
    hold ``action_center``, ``action_scale``, ``action_low`` and ``action_high``; all of it is stored in ``meta_json``.
    Returns the arrays written."""
    import torch
    layers, acts = [], []
    for m in actor_critic.actor:
        if isinstance(m, torch.nn.Linear):
            layers.append((m.weight.detach().cpu().double().numpy(), m.bias.detach().cpu().double().numpy()))
        elif isinstance(m, torch.nn.ELU) and float(m.alpha) == 1.0:
            acts.append("elu")
        else:
            raise ValueError(f"unsupported actor module {m!r} (expected Linear / ELU(alpha=1))")
    if not layers or len(acts) != len(layers) - 1:
        raise ValueError(f"actor must be Linear, ELU, ..., Linear; got {len(layers)} Linear and {len(acts)} ELU")
    mean = normalizer._mean.detach().cpu().double().numpy().reshape(-1)
    var = normalizer._var.detach().cpu().double().numpy().reshape(-1)
    std = normalizer._std.detach().cpu().double().numpy().reshape(-1)
    if not np.allclose(std, np.sqrt(var), atol=1e-6, rtol=1e-5):
        raise ValueError("normaliser _std buffer is not sqrt(_var); refusing to guess which one the policy saw")
    arrays = {"format": np.array(ACTOR_FORMAT), "num_obs": np.int64(len(mean)),
              "num_actions": np.int64(layers[-1][0].shape[0]),
              "obs_mean": mean, "obs_var": var, "obs_eps": np.float64(float(getattr(normalizer, "eps", 1e-2))),
              "n_layers": np.int64(len(layers)), "activation": np.array("elu"),
              "action_center": np.asarray(meta["action_center"], np.float64), "action_scale": np.asarray(meta["action_scale"], np.float64),
              "action_low": np.asarray(meta["action_low"], np.float64), "action_high": np.asarray(meta["action_high"], np.float64),
              "meta_json": np.array(json.dumps(meta, default=jsonable, sort_keys=True))}
    for i, (W, b) in enumerate(layers):
        arrays[f"W{i}"], arrays[f"b{i}"] = W, b
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return arrays
