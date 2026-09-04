"""WP3 tracker imagination env: route following inside the frozen WP2 NRD (plan §10).

The environment rolls the WP2 map-indexed dynamics model (``nedm.traverse.nrd_model``)
in parallel over thousands of short fragments and rewards the policy for following
a *route* -- waypoints + speed profile, the ``PlanCandidate`` format the oracle and
the planner emit -- with a geometric reward. It is the traverse analogue of
``nedm.rl.hmmwv_tracking_env.HMMWVNeuralTrackingEnv`` and deliberately keeps what
that study showed transfers to Chrono:

* the tracking term is ``exp(-loss)``: bounded, strictly positive, so ending an
  episode early can only forfeit reward (no termination exploit, dpend lesson);
* a hard steering-rate clamp is applied *during training*, not only at eval
  (the dominant Chrono transfer failure was abrupt steering reversals);
* loose failure bounds so the policy learns recovery instead of being reset;
* the model sees only its 16-frame context; pose is integrated outside it.

What differs from that study, per plan §10: references are routes without
timing (cross-track / heading / speed-profile errors instead of pose-vs-time),
the observation is the reduced deployment set (38-D by default), and episodes
are randomized 1-3 s fragments seeded from *real* recorded context windows at
varied progress along the route, so every imagined rollout stays inside the
horizon WP2 validated.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsl_rl.env import VecEnv

from nedm.traverse import nrd_data as D
from nedm.traverse.nrd_model import DT_S, PITCH, ROLL, VX, YAW_RATE, integrate_pose, load_map_model


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def default_env_cfg() -> dict[str, Any]:
    return {
        "num_envs": 2048,
        "device": "cuda",
        "dynamics_checkpoint": "artifacts/traverse/wp2_mapv2_index_amd/ckpt_best.pt",
        "arena": "assets/traverse/arena_v1",
        "cache": "artifacts/traverse/wp2_z2_cache_v6",
        "map_key": "map_v2",
        "routes": "artifacts/traverse/wp3_routes",
        "split": "train",  # train | val | test -- WP2's pinned layout split
        "families": None,  # e.g. ["spline", "oracle"]; None = every routed family
        "max_bank_episodes": 0,  # 0 = all episodes of the split
        "fragment_steps_min": 20,  # 1 s at 20 Hz
        "fragment_steps_max": 60,  # 3 s
        "preview_points": 10,
        "preview_spacing_m": 1.0,
        "search_window": 40,  # waypoints (0.5 m) searched ahead for the nearest point
        "obs_history_steps": 0,  # >0 appends that many normalized z1 frames (obs ablation)
        "action_low": [-1.0, 0.0, 0.0],
        "action_high": [1.0, 1.0, 1.0],
        "action_center": "dataset_mean",
        "action_scale": [1.0, 0.7, 0.5],
        "steering_rate_limit": 0.1,  # per 20 Hz step = the collector's 2.0 full-scale/s
        "reward": {
            "cross_track_sigma_m": 1.0,
            "heading_sigma_rad": 0.35,
            "speed_sigma_mps": 1.0,
            "cross_track_weight": 2.0,
            "heading_weight": 0.8,
            "speed_weight": 0.5,
            "action_rate_weight": 0.2,
            "throttle_brake_weight": 0.05,
        },
        "termination": {
            "max_cross_track_m": 6.0,
            "max_abs_roll_rad": 0.6,
            "max_abs_pitch_rad": 0.4,
        },
        "auto_reset": True,
        "seed": 0,
    }


def merge_env_cfg(cfg: dict[str, Any] | None) -> dict[str, Any]:
    merged = default_env_cfg()
    for key, value in (cfg or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


class EpisodeBank:
    """Recorded context windows + static scene maps + routes for one split."""

    def __init__(self, cfg: dict[str, Any], norm: D.Normalizer, context: int, device: torch.device,
                 entries: list[tuple[str, dict[str, np.ndarray]]] | None = None):
        """``entries`` = explicit (cache key, route dict) pairs -- the planner scorer uses
        this to roll out several candidate routes on the same episode; default is the
        split's recorded routes from ``cfg["routes"]``."""
        cache, routes = Path(cfg["cache"]), Path(cfg["routes"])
        if entries is None:
            keys = D.load_cache_keys(cache)
            train_keys, val_keys, test_keys = D.split_keys(keys)
            keys = {"train": train_keys, "val": val_keys, "test": test_keys}[cfg["split"]]
            manifest = json.loads((routes / "routes_manifest.json").read_text())
            allowed = set()
            for fam, fam_keys in manifest["families"].items():
                if cfg["families"] is None or fam in cfg["families"]:
                    allowed.update(fam_keys)
            keys = [k for k in keys if k in allowed]
            if cfg["max_bank_episodes"]:
                keys = keys[: int(cfg["max_bank_episodes"])]
            entries = []
            for k in keys:
                with np.load(routes / f"{k}.npz") as r:
                    entries.append((k, {n: r[n] for n in ("waypoints", "speeds", "headings", "stations")}))
        if not entries:
            raise ValueError("no routed episodes in the requested split/families")
        t0 = time.time()
        keys = [k for k, _ in entries]
        z1, act, pose, maps, wps, spd, hdg, sta = [], [], [], [], [], [], [], []
        loaded: dict[str, tuple] = {}
        for k, route in entries:
            if k not in loaded:
                with np.load(cache / f"{k}.npz") as d:
                    loaded[k] = (d["z1"], d["act"], d["pose"], d[cfg["map_key"]])
            a, b, c, m = loaded[k]
            z1.append(a); act.append(b); pose.append(c); maps.append(m)
            wps.append(np.asarray(route["waypoints"], np.float32))
            spd.append(np.asarray(route["speeds"], np.float32))
            hdg.append(np.asarray(route["headings"], np.float32))
            sta.append(np.asarray(route["stations"], np.float32))
        self.keys = keys
        self.n_frames = z1[0].shape[0]
        self.context = context
        z1_np, act_np, pose_np = np.stack(z1), np.stack(act), np.stack(pose)
        self.z1 = torch.from_numpy((z1_np - norm.z1_mean) / norm.z1_std).float().to(device)
        self.act = torch.from_numpy((act_np - norm.act_mean) / norm.act_std).float().to(device)
        self.act_raw = torch.from_numpy(act_np).float().to(device)
        self.pose = torch.from_numpy(pose_np).float().to(device)
        self.maps = torch.from_numpy(np.stack(maps)).to(device)  # (E, 64, 64, 64) float16

        lmax = max(len(w) for w in wps)
        n = len(keys)
        route_xy = np.zeros((n, lmax, 2), np.float32)
        route_v = np.zeros((n, lmax), np.float32)
        route_h = np.zeros((n, lmax), np.float32)
        route_s = np.zeros((n, lmax), np.float32)
        route_len = np.zeros(n, np.int64)
        active_end = np.zeros(n, np.int64)
        for i, (w, v, h, s) in enumerate(zip(wps, spd, hdg, sta)):
            L = len(w)
            route_len[i] = L
            route_xy[i, :L] = w; route_xy[i, L:] = w[-1]
            route_v[i, :L] = v  # padded with 0: the reference stops at the route end
            route_h[i, :L] = h; route_h[i, L:] = h[-1]
            route_s[i, :L] = s; route_s[i, L:] = s[-1]
            # last frame at which the recorded vehicle was still >3 m of arc from the
            # route end -- fragments starting after that would sit in the parked tail
            dist = np.linalg.norm(pose_np[i, :, None, :2] - w[None, :, :], axis=-1)
            s_near = s[dist.argmin(axis=1)]
            live = np.nonzero(s_near < s[-1] - 3.0)[0]
            active_end[i] = int(live[-1]) if len(live) else self.n_frames - 1
        to_t = lambda a: torch.from_numpy(a).to(device)
        self.route_xy, self.route_v = to_t(route_xy), to_t(route_v)
        self.route_h, self.route_s = to_t(route_h), to_t(route_s)
        self.route_len, self.active_end = to_t(route_len), to_t(active_end)
        self.route_ds = self.route_s[torch.arange(n), self.route_len - 1] / (self.route_len - 1).clamp(min=1)
        self.n_episodes = n
        self.load_s = time.time() - t0


class TraverseTrackingEnv(VecEnv):
    """RSL-RL vectorized route-tracking env inside the frozen WP2 map-indexed NRD."""

    def __init__(self, cfg: dict[str, Any] | None = None, device: str | torch.device | None = None,
                 entries: list[tuple[str, dict[str, np.ndarray]]] | None = None):
        self.cfg = merge_env_cfg(cfg)
        if device is not None:
            self.cfg["device"] = str(device)
        self.device = torch.device(self.cfg["device"])
        self.num_envs = int(self.cfg["num_envs"])
        self.num_actions = 3
        self.num_privileged_obs = None
        self.preview_points = int(self.cfg["preview_points"])
        self.obs_history_steps = int(self.cfg["obs_history_steps"])
        self.auto_reset = bool(self.cfg.get("auto_reset", True))
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(self.cfg["seed"]))

        self.model, self.norm, payload = load_map_model(
            self.cfg["dynamics_checkpoint"], self.cfg["arena"], self.device)
        if payload.get("map_mode", "index") != "index":
            raise ValueError("tracker env needs an index-mode WP2 checkpoint (z2 re-cropped each step)")
        self.context = int(payload["config"]["block_size"])
        if self.obs_history_steps > self.context:
            raise ValueError("obs_history_steps exceeds the model context")
        f32 = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)
        self.z1_mean, self.z1_std = f32(self.norm.z1_mean), f32(self.norm.z1_std)
        self.act_mean, self.act_std = f32(self.norm.act_mean), f32(self.norm.act_std)
        self.p_mean, self.p_std = float(self.norm.power_mean[0]), float(self.norm.power_std[0])

        self.bank = EpisodeBank(self.cfg, self.norm, self.context, self.device, entries=entries)
        self.max_episode_length = int(self.cfg["fragment_steps_max"])

        self.action_low = f32(self.cfg["action_low"])
        self.action_high = f32(self.cfg["action_high"])
        self.action_scale = f32(self.cfg["action_scale"])
        self.action_center = (self.act_mean.clone() if self.cfg["action_center"] == "dataset_mean"
                              else f32(self.cfg["action_center"]))

        n, c = self.num_envs, self.context
        dev = self.device
        self.z1_hist = torch.zeros(n, c, 15, device=dev)
        self.act_hist = torch.zeros(n, c, 3, device=dev)
        self.token_hist = torch.zeros(n, c, self.model.token_dim, device=dev)
        self.env_maps = torch.zeros(n, *self.bank.maps.shape[1:], device=dev)
        self.pose = torch.zeros(n, 3, device=dev)
        self.z1_phys = torch.zeros(n, 15, device=dev)
        self.env_ep = torch.zeros(n, dtype=torch.long, device=dev)
        self.route_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.fragment_len = torch.full((n,), self.max_episode_length, dtype=torch.long, device=dev)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.actions = torch.zeros(n, 3, device=dev)
        self.last_actions = torch.zeros(n, 3, device=dev)
        self.energy_kj = torch.zeros(n, device=dev)
        self.progress_m = torch.zeros(n, device=dev)
        self.start_station_m = torch.zeros(n, device=dev)
        self.ep_reward_sum = torch.zeros(n, device=dev)
        self.ep_ct_sum = torch.zeros(n, device=dev)
        self.ep_ct_max = torch.zeros(n, device=dev)
        self.ep_speed_err_sum = torch.zeros(n, device=dev)
        self.search_offsets = torch.arange(-2, int(self.cfg["search_window"]), device=dev)
        self.preview_k = torch.arange(1, self.preview_points + 1, device=dev, dtype=torch.float32)
        self.num_obs = 3 + 3 * self.preview_points + 2 + 3 + 15 * self.obs_history_steps
        self.obs_buf = torch.zeros(n, self.num_obs, device=dev)
        self.rew_buf = torch.zeros(n, device=dev)
        self.reset_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.time_out_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.extras: dict[str, Any] = {}
        self.reset()

    # ------------------------------------------------------------------ resets
    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_observations()
        return self.obs_buf, self.extras

    def reset_idx(self, env_ids: torch.Tensor, episode_ids: torch.Tensor | None = None,
                  start_frames: torch.Tensor | None = None,
                  fragment_steps: torch.Tensor | None = None) -> None:
        m = env_ids.numel()
        if m == 0:
            return
        b = self.bank
        if episode_ids is None:
            episode_ids = torch.randint(0, b.n_episodes, (m,), device=self.device, generator=self.gen)
        if fragment_steps is None:
            lo, hi = int(self.cfg["fragment_steps_min"]), int(self.cfg["fragment_steps_max"])
            fragment_steps = torch.randint(lo, hi + 1, (m,), device=self.device, generator=self.gen)
        if start_frames is None:
            lo = torch.full((m,), self.context, device=self.device, dtype=torch.long)
            hi = torch.minimum(b.active_end[episode_ids], torch.full_like(lo, b.n_frames)) - fragment_steps
            hi = torch.maximum(hi, lo)
            u = torch.rand(m, device=self.device, generator=self.gen)
            start_frames = lo + (u * (hi - lo + 1).float()).long().clamp(max=hi - lo)
        win = start_frames[:, None] + torch.arange(-self.context, 0, device=self.device)[None, :]
        ep_col = episode_ids[:, None]
        self.z1_hist[env_ids] = b.z1[ep_col, win]
        self.act_hist[env_ids] = b.act[ep_col, win]
        hist_pose = b.pose[ep_col, win]
        self.env_maps[env_ids] = b.maps[episode_ids].float()
        with torch.no_grad():
            self.token_hist[env_ids] = self.model.cropper(self.env_maps[env_ids], hist_pose)
        self.pose[env_ids] = hist_pose[:, -1]
        self.z1_phys[env_ids] = self.z1_hist[env_ids, -1] * self.z1_std + self.z1_mean
        self.env_ep[env_ids] = episode_ids
        self.fragment_len[env_ids] = fragment_steps
        self.last_actions[env_ids] = b.act_raw[episode_ids, start_frames - 1]
        self.actions[env_ids] = self.last_actions[env_ids]
        # nearest waypoint over the whole route (fragments start on-route, so this is safe)
        d = (b.route_xy[episode_ids] - self.pose[env_ids, None, :2]).norm(dim=-1)
        valid = torch.arange(b.route_xy.shape[1], device=self.device)[None, :] < b.route_len[episode_ids][:, None]
        d = torch.where(valid, d, torch.full_like(d, float("inf")))
        self.route_idx[env_ids] = d.argmin(dim=1)
        self.start_station_m[env_ids] = b.route_s[episode_ids, self.route_idx[env_ids]]
        self.progress_m[env_ids] = 0.0
        for buf in (self.episode_length_buf, self.energy_kj, self.ep_reward_sum, self.ep_ct_sum,
                    self.ep_ct_max, self.ep_speed_err_sum):
            buf[env_ids] = 0

    # ------------------------------------------------------------------ dynamics
    def _scale_policy_actions(self, policy_actions: torch.Tensor) -> torch.Tensor:
        bounded = torch.tanh(policy_actions)
        return torch.clamp(self.action_center + self.action_scale * bounded, self.action_low, self.action_high)

    def physical_to_policy(self, physical: torch.Tensor) -> torch.Tensor:
        """Inverse of the action squash (for scripted controllers)."""
        u = ((physical - self.action_center) / self.action_scale).clamp(-0.999, 0.999)
        return torch.atanh(u)

    @torch.no_grad()
    def _nn_step(self, driver_actions: torch.Tensor) -> None:
        self.act_hist[:, -1] = (driver_actions - self.act_mean) / self.act_std
        delta, power, _ = self.model(self.z1_hist, self.token_hist, self.act_hist)
        z1_next = self.z1_hist[:, -1] + delta[:, -1]
        self.z1_phys = z1_next * self.z1_std + self.z1_mean
        self.pose = integrate_pose(self.pose, self.z1_phys)
        token_next = self.model.cropper(self.env_maps, self.pose.unsqueeze(1))[:, 0]
        self.z1_hist = torch.cat([self.z1_hist[:, 1:], z1_next.unsqueeze(1)], dim=1)
        self.token_hist = torch.cat([self.token_hist[:, 1:], token_next.unsqueeze(1)], dim=1)
        self.act_hist = torch.cat([self.act_hist[:, 1:], self.act_hist[:, -1:]], dim=1)
        self.energy_kj += (power[:, -1, 0] * self.p_std + self.p_mean) * DT_S

    # ------------------------------------------------------------------ route geometry
    def _route_errors(self) -> dict[str, torch.Tensor]:
        b, ep = self.bank, self.env_ep
        last = (b.route_len[ep] - 1)[:, None]
        cand = (self.route_idx[:, None] + self.search_offsets[None, :]).clamp(min=0)
        cand = torch.minimum(cand, last)
        pts = b.route_xy[ep[:, None], cand]
        d = (pts - self.pose[:, None, :2]).norm(dim=-1)
        j = d.argmin(dim=1, keepdim=True)
        idx = cand.gather(1, j)[:, 0]
        self.route_idx = idx
        wp, h = b.route_xy[ep, idx], b.route_h[ep, idx]
        v_ref = b.route_v[ep, idx]
        dx, dy = self.pose[:, 0] - wp[:, 0], self.pose[:, 1] - wp[:, 1]
        cos_h, sin_h = torch.cos(h), torch.sin(h)
        e_along = dx * cos_h + dy * sin_h
        e_ct = -dx * sin_h + dy * cos_h
        e_h = wrap_angle(self.pose[:, 2] - h)
        e_v = self.z1_phys[:, VX] - v_ref
        self.progress_m = b.route_s[ep, idx] - self.start_station_m
        return {"e_along": e_along, "e_ct": e_ct, "e_h": e_h, "e_v": e_v, "v_ref": v_ref,
                "route_end": idx >= (b.route_len[ep] - 2)}

    def _preview_body(self) -> torch.Tensor:
        b, ep = self.bank, self.env_ep
        step = (self.preview_k[None, :] * float(self.cfg["preview_spacing_m"]) / b.route_ds[ep][:, None]).round().long()
        idx = torch.minimum(self.route_idx[:, None] + step, (b.route_len[ep] - 1)[:, None])
        pts = b.route_xy[ep[:, None], idx]
        v = b.route_v[ep[:, None], idx]
        dx = pts[..., 0] - self.pose[:, None, 0]
        dy = pts[..., 1] - self.pose[:, None, 1]
        cos_y, sin_y = torch.cos(self.pose[:, 2])[:, None], torch.sin(self.pose[:, 2])[:, None]
        bx = cos_y * dx + sin_y * dy
        by = -sin_y * dx + cos_y * dy
        return torch.stack([bx / 10.0, by / 10.0, v / 5.0], dim=-1).flatten(1)

    # ------------------------------------------------------------------ rsl_rl interface
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = actions.to(device=self.device, dtype=torch.float32)
        driver = self._scale_policy_actions(actions)
        limit = self.cfg.get("steering_rate_limit")
        if limit is not None:
            driver[:, 0] = torch.clamp(driver[:, 0], self.last_actions[:, 0] - float(limit),
                                       self.last_actions[:, 0] + float(limit))
        self.actions = driver
        self._nn_step(driver)
        self.episode_length_buf += 1

        err = self._route_errors()
        rc = self.cfg["reward"]
        loss = (float(rc["cross_track_weight"]) * (err["e_ct"] / float(rc["cross_track_sigma_m"])) ** 2
                + float(rc["heading_weight"]) * (err["e_h"] / float(rc["heading_sigma_rad"])) ** 2
                + float(rc["speed_weight"]) * (err["e_v"] / float(rc["speed_sigma_mps"])) ** 2)
        track_reward = torch.exp(-loss)
        action_rate = ((driver - self.last_actions) ** 2).sum(dim=-1)
        throttle_brake = driver[:, 1] * driver[:, 2]
        reward = (track_reward - float(rc["action_rate_weight"]) * action_rate
                  - float(rc["throttle_brake_weight"]) * throttle_brake)

        tc = self.cfg["termination"]
        failed = ((err["e_ct"].abs() > float(tc["max_cross_track_m"]))
                  | (self.z1_phys[:, ROLL].abs() > float(tc["max_abs_roll_rad"]))
                  | (self.z1_phys[:, PITCH].abs() > float(tc["max_abs_pitch_rad"]))
                  | ~torch.isfinite(self.z1_phys).all(dim=-1) | ~torch.isfinite(self.pose).all(dim=-1))
        time_outs = self.episode_length_buf >= self.fragment_len
        bootstrap = time_outs | err["route_end"]
        dones = failed | bootstrap
        reward = torch.where(torch.isfinite(reward), reward, torch.zeros_like(reward))

        self.rew_buf = reward
        self.reset_buf = dones.long()
        self.time_out_buf = bootstrap
        self.ep_reward_sum += reward
        self.ep_ct_sum += err["e_ct"].abs()
        self.ep_ct_max = torch.maximum(self.ep_ct_max, err["e_ct"].abs())
        self.ep_speed_err_sum += err["e_v"].abs()
        extras = self._make_extras(err, track_reward, action_rate, throttle_brake, dones, failed, bootstrap)
        self.last_actions = driver.clone()
        if self.auto_reset:
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                self.reset_idx(done_ids)
        self._compute_observations()
        extras["observations"] = {"critic": self.obs_buf}
        self.extras = extras
        return self.obs_buf, self.rew_buf, dones.long(), self.extras

    def _make_extras(self, err, track_reward, action_rate, throttle_brake, dones, failed, bootstrap):
        log = {
            "/tracking/track_reward": track_reward.mean(),
            "/tracking/cross_track_abs_m": err["e_ct"].abs().mean(),
            "/tracking/heading_err_abs_rad": err["e_h"].abs().mean(),
            "/tracking/speed_err_abs_mps": err["e_v"].abs().mean(),
            "/tracking/action_rate": action_rate.mean(),
            "/tracking/throttle_brake": throttle_brake.mean(),
        }
        extras: dict[str, Any] = {"observations": {"critic": self.obs_buf}, "time_outs": bootstrap, "log": log}
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            lengths = self.episode_length_buf[done_ids].float().clamp(min=1.0)
            ep = {
                "/episode/reward": self.ep_reward_sum[done_ids].mean(),
                "/episode/length": lengths.mean(),
                "/episode/mean_cross_track_m": (self.ep_ct_sum[done_ids] / lengths).mean(),
                "/episode/max_cross_track_m": self.ep_ct_max[done_ids].mean(),
                "/episode/mean_speed_err_mps": (self.ep_speed_err_sum[done_ids] / lengths).mean(),
                "/episode/fail_rate": failed[done_ids].float().mean(),
                "/episode/route_end_rate": err["route_end"][done_ids].float().mean(),
                "/episode/progress_m": self.progress_m[done_ids].mean(),
            }
            ep.update(log)
            extras["episode"] = ep
        return extras

    def _compute_observations(self) -> None:
        err = self._route_errors()
        parts = [
            torch.stack([err["e_along"] / 10.0, err["e_ct"] / 10.0, err["e_h"] / math.pi], dim=-1),
            self._preview_body(),
            torch.stack([self.z1_phys[:, VX] / 10.0, self.z1_phys[:, YAW_RATE]], dim=-1),
            self.last_actions,
        ]
        if self.obs_history_steps > 0:
            parts.append(self.z1_hist[:, -self.obs_history_steps:].flatten(1))
        self.obs_buf = torch.cat(parts, dim=-1)
        self.obs_buf = torch.where(torch.isfinite(self.obs_buf), self.obs_buf, torch.zeros_like(self.obs_buf))
        self.extras = {"observations": {"critic": self.obs_buf}}

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        self._compute_observations()
        return self.obs_buf, self.extras

    def get_privileged_observations(self) -> None:
        return None


@torch.no_grad()
def pure_pursuit_actions(env: TraverseTrackingEnv, lookahead_m: float = 5.0,
                         gain: float = 1.5) -> torch.Tensor:
    """Scripted reference controller on the env's own observation (sanity baseline).

    Steering from the pure-pursuit angle to the preview point nearest ``lookahead_m``;
    throttle/brake from the speed error to the first preview point's profile speed.
    Returns pre-squash policy actions so it can be fed to ``env.step``.
    """
    obs = env.obs_buf
    p = env.preview_points
    prev = obs[:, 3:3 + 3 * p].view(-1, p, 3)
    k = max(0, min(p - 1, int(round(lookahead_m / float(env.cfg["preview_spacing_m"]))) - 1))
    bx, by = prev[:, k, 0] * 10.0, prev[:, k, 1] * 10.0
    alpha = torch.atan2(by, bx.clamp(min=0.5))
    steer = (gain * alpha).clamp(-1.0, 1.0)
    v_ref = prev[:, 0, 2] * 5.0
    vx = obs[:, 3 + 3 * p] * 10.0
    e_v = v_ref - vx
    throttle = torch.where(e_v > -0.3, (0.25 + 0.25 * e_v).clamp(0.0, 1.0), torch.zeros_like(e_v))
    brake = torch.where(e_v < -0.3, (-0.3 * e_v).clamp(0.0, 1.0), torch.zeros_like(e_v))
    stop = v_ref < 0.2
    throttle = torch.where(stop, torch.zeros_like(throttle), throttle)
    brake = torch.where(stop, torch.ones_like(brake) * 0.6, brake)
    return env.physical_to_policy(torch.stack([steer, throttle, brake], dim=-1))
