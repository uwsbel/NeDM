"""Imagined-drive environment for the PPO route tracker: recorded route fragments driven inside the frozen NRD model.

Every env replays one fragment of a recorded f104 drive (rigid or soil, train groups of the dynamics cache only):
  * reset: a random bank episode (``domain_frac`` of the resets on soil) and a decision frame s uniform in
    [0, active_end - len], with active_end the last frame at which the recorded vehicle was still more than 3 m of
    arc from the route end and len uniform in ``fragment_steps_min`` .. ``fragment_steps_max`` (1 to 3 s). The NRD
    context is the recorded frames s-15 .. s; frames before 0 are padded with frame 0's state and pose and the
    settle action (0, 0, 1), as a controller's history is at the start of a drive.
  * step: the policy output (pre-tanh) is squashed onto the full box (steer centre 0 scale 1, throttle and brake
    centre 0.5 scale 0.5), the steering is clamped to +-0.1 of the last action (the collectors' 2 full-scale/s),
    the NRD predicts the next state and the pose is dead-reckoned; the terrain crop is re-taken at the new pose.
  * reward: exp(-(2 (e_ct / 1 m)^2 + 0.8 (e_h / 0.35 rad)^2 + w_v (e_v / 1 m/s)^2)) - 0.2 |da|^2
    - 0.05 throttle * brake + 0.5 x the along-track advance in metres (clipped to +-2 m); w_v = ``speed_weight``.
  * end: cross-track error > 6 m, |roll| > 0.6 rad, |pitch| > 0.4 rad or a non-finite state fails; the fragment
    length or the route end (last two waypoints) ends the episode with bootstrapping.
  * observation (158 numbers, the deployed controller's layout, ``obs_layout``): [0:3] along-track, cross-track and
    heading error to the nearest waypoint (/10, /10, /pi); [3:33] 10 route points 1 .. 10 m ahead in the body frame
    (x/10, y/10, reference speed/5); [33:35] vx/10 and yaw rate; [35:38] the last action; [38:62] the last 8 actions
    (oldest first); [62:158] the last 8 observable states (``state.OBSERVABLE_COLS``, oldest first, the newest row
    is the current state), raw physical values.

    env = TrackingEnv(merge_env_cfg({"num_envs": 2048, "reward": {"speed_weight": 1.5}, "seed": 2}), device="cuda")
    obs, rew, dones, extras = env.step(policy_actions)
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
from rsl_rl.env import VecEnv

import dynamics_data as D
from dynamics_model import integrate_pose, load_nrd
from state import DT_S, OBSERVABLE_COLS, PITCH, ROLL, SETTLE_ACTION, VX, YAW_RATE

DEFAULT_NRD = "artifacts/traverse/generalist_20260921/B_tracker/nrd_tag_v3/ckpt_best.pt"


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def default_env_cfg() -> dict[str, Any]:
    return {
        "num_envs": 2048,
        "device": "cuda",
        "nrd_checkpoint": DEFAULT_NRD,
        "grid": D.DEFAULT_GRID,  # the grid the NRD was trained on (sha256-checked against the checkpoint)
        "cache": D.DEFAULT_CACHE,
        "split": "train",  # fragments come from the train groups only
        "max_bank_episodes": 0,  # 0 = every train episode
        "max_frames": 1200,
        "min_frames": 40,  # shorter episodes are skipped
        "domain_frac": 0.5,  # fraction of resets on soil; None = the cache's natural mix
        "twin_split": None,  # twin group split to cross-check the cache split against (None = default path; 'none' skips)
        "fragment_steps_min": 20,  # 1 s at 20 Hz
        "fragment_steps_max": 60,  # 3 s
        "preview_points": 10,
        "preview_spacing_m": 1.0,
        "search_window": 40,
        "hist_steps": 8,
        "state_cols": list(OBSERVABLE_COLS),
        "settle_action": list(SETTLE_ACTION),
        "action_low": [-1.0, 0.0, 0.0],
        "action_high": [1.0, 1.0, 1.0],
        "action_center": [0.0, 0.5, 0.5],
        "action_scale": [1.0, 0.5, 0.5],
        "steering_rate_limit": 0.1,
        "reward": {
            "cross_track_sigma_m": 1.0,
            "heading_sigma_rad": 0.35,
            "speed_sigma_mps": 1.0,
            "cross_track_weight": 2.0,
            "heading_weight": 0.8,
            "speed_weight": 0.5,
            "action_rate_weight": 0.2,
            "throttle_brake_weight": 0.05,
            "progress_weight": 0.5,
            "progress_clip_m": 2.0,
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


def obs_layout(preview_points: int, preview_spacing_m: float, search_window: int, hist_steps: int, state_cols,
               settle_action) -> dict:
    """Slices of the observation; stored in ``policy_meta.json`` and the actor npz (the deployed controller reads it)."""
    H, C, P = int(hist_steps), len(state_cols), int(preview_points)
    base = 3 + 3 * P + 2 + 3
    return {"num_obs": base + 3 * H + C * H, "errors": [0, 3], "preview": [3, 3 + 3 * P],
            "vx_yawrate": [3 + 3 * P, base - 3], "last_action": [base - 3, base],
            "past_actions": [base, base + 3 * H], "past_states": [base + 3 * H, base + 3 * H + C * H],
            "hist_steps": H, "state_cols": [int(c) for c in state_cols], "preview_points": P,
            "preview_spacing_m": float(preview_spacing_m), "search_window": int(search_window),
            "settle_action": [float(v) for v in settle_action], "state_mean": None, "state_std": None,
            "order": "oldest first; past_states newest row = current state; past_actions newest row = last action"}


class FragmentBank:
    """The train episodes of the dynamics cache, flat on the device (``offset[e] + frame``), with their routes."""

    def __init__(self, cfg: dict[str, Any], norm, device: torch.device):
        t0 = time.time()
        cache_dir = D.resolve_path(cfg["cache"])
        man = D.read_manifest(cache_dir)
        D.check_group_split_consistency(man)
        self.twin_check = D.check_split_against_twin(man, cfg.get("twin_split"))
        self.manifest_sha256 = D.file_sha256(cache_dir / "cache_manifest.json")
        held = D.heldout_groups(man)
        keys = D.select_keys(man, splits=(cfg["split"],), max_episodes=int(cfg.get("max_bank_episodes") or 0), seed=int(cfg.get("seed", 0)))
        if not keys:
            raise ValueError(f"no episodes in split {cfg['split']!r} of {cache_dir}")
        bad = {man["group_of"][k] for k in keys} & held
        assert not bad, f"held-out groups in the fragment bank: {sorted(bad)[:5]}"
        min_frames, max_frames = int(cfg.get("min_frames", 0)), int(cfg.get("max_frames") or 0)
        z1, act, pose, lengths, dom, routes = [], [], [], [], [], []
        for k in keys:
            with np.load(cache_dir / f"{k}.npz", allow_pickle=False) as z:
                T = int(z["z1"].shape[0])
                if max_frames and T > max_frames:
                    T = max_frames
                if T < max(min_frames, 2):
                    continue
                z1.append(np.asarray(z["z1"][:T], np.float32)); act.append(np.asarray(z["act"][:T], np.float32))
                pose.append(np.asarray(z["pose"][:T], np.float32))
                routes.append({f: np.asarray(z[f"route_{f}"], np.float32) for f in D.ROUTE_FIELDS})
                d = int(z["domain"]) if "domain" in z.files else int(man["domain_of"][k])
                assert d == int(man["domain_of"][k]), (k, d, man["domain_of"][k])
            lengths.append(T); dom.append(d)
        if not lengths:
            raise ValueError("every episode was filtered out (min_frames)")
        if z1[0].shape[-1] != len(norm.z1_mean):
            raise ValueError(f"cache z1 is {z1[0].shape[-1]}-D, the NRD expects {len(norm.z1_mean)}-D")
        n = self.n_episodes = len(lengths)
        lengths = np.asarray(lengths, np.int64)
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        z1_np, act_np, pose_np = np.concatenate(z1), np.concatenate(act), np.concatenate(pose)
        # routes padded to the longest; speeds padded with 0 (the reference stops at the route end)
        lmax = max(len(r["waypoints"]) for r in routes)
        route_xy = np.zeros((n, lmax, 2), np.float32); route_v = np.zeros((n, lmax), np.float32)
        route_h = np.zeros((n, lmax), np.float32); route_s = np.zeros((n, lmax), np.float32)
        route_len = np.zeros(n, np.int64); active_end = np.zeros(n, np.int64)
        for i, r in enumerate(routes):
            w, v, h, s = r["waypoints"], r["speeds"], r["headings"], r["stations"]
            L = len(w); route_len[i] = L
            route_xy[i, :L] = w; route_xy[i, L:] = w[-1]
            route_v[i, :L] = v
            route_h[i, :L] = h; route_h[i, L:] = h[-1]
            route_s[i, :L] = s; route_s[i, L:] = s[-1]
            # last frame at which the recorded vehicle was still > 3 m of arc from the route end
            p = pose_np[offsets[i]:offsets[i] + lengths[i], :2]
            dist = np.linalg.norm(p[:, None, :] - w[None, :, :], axis=-1)
            s_near = s[dist.argmin(axis=1)]
            live = np.nonzero(s_near < s[-1] - 3.0)[0]
            active_end[i] = int(live[-1]) if len(live) else lengths[i] - 1
        to_t = lambda a, dt=None: torch.from_numpy(np.ascontiguousarray(a)).to(device) if dt is None else torch.from_numpy(np.ascontiguousarray(a)).to(device, dt)
        self.z1 = to_t((z1_np - norm.z1_mean) / norm.z1_std, torch.float32)   # (F, 17) normalised
        self.act = to_t((act_np - norm.act_mean) / norm.act_std, torch.float32)  # (F, 3) normalised
        self.act_raw = to_t(act_np); self.pose = to_t(pose_np)
        self.offset, self.length, self.active_end = to_t(offsets), to_t(lengths), to_t(active_end)
        self.domain = to_t(np.asarray(dom, np.int64))
        self.route_xy, self.route_v, self.route_h, self.route_s = to_t(route_xy), to_t(route_v), to_t(route_h), to_t(route_s)
        self.route_len = to_t(route_len)
        self.route_ds = self.route_s[torch.arange(n, device=device), self.route_len - 1] / (self.route_len - 1).clamp(min=1)
        self.dom_eps = {d: torch.nonzero(self.domain == d).flatten() for d in range(len(D.DOMAIN_VOCAB))}
        self.n_frames_total = int(lengths.sum())
        self.load_s = time.time() - t0

    def counts(self) -> dict:
        return {D.DOMAIN_VOCAB[d]: int(len(e)) for d, e in self.dom_eps.items()}


class TrackingEnv(VecEnv):
    """rsl_rl vectorised route-tracking env inside the frozen NRD model."""

    def __init__(self, cfg: dict[str, Any] | None = None, device: str | torch.device | None = None):
        self.cfg = merge_env_cfg(cfg)
        if device is not None:
            self.cfg["device"] = str(device)
        self.device = torch.device(self.cfg["device"])
        self.num_envs = int(self.cfg["num_envs"])
        self.num_actions = 3
        self.preview_points = int(self.cfg["preview_points"])
        self.H = int(self.cfg["hist_steps"])
        self.cols = [int(c) for c in self.cfg["state_cols"]]
        self.auto_reset = bool(self.cfg.get("auto_reset", True))
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(self.cfg["seed"]))

        self.model, self.norm, payload = load_nrd(self.cfg["nrd_checkpoint"], self.device, grid_path=self.cfg.get("grid"))
        self.nrd_payload = {k: payload[k] for k in ("model_kind", "cond", "crop_k", "crop_half_m", "grid_path", "grid_sha256", "step", "token_dim")}
        self.nrd_sha256 = D.file_sha256(D.resolve_path(self.cfg["nrd_checkpoint"]))
        self.context = int(payload["config"]["block_size"])
        if self.H > self.context:
            raise ValueError("hist_steps exceeds the model context")
        f32 = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)
        self.z1_mean, self.z1_std = f32(self.norm.z1_mean), f32(self.norm.z1_std)
        self.act_mean, self.act_std = f32(self.norm.act_mean), f32(self.norm.act_std)
        self.p_mean, self.p_std = float(self.norm.power_mean[0]), float(self.norm.power_std[0])
        self.z1_dim = len(self.norm.z1_mean)

        self.bank = FragmentBank(self.cfg, self.norm, self.device)
        self.max_episode_length = int(self.cfg["fragment_steps_max"])

        self.action_low, self.action_high = f32(self.cfg["action_low"]), f32(self.cfg["action_high"])
        self.action_center, self.action_scale = f32(self.cfg["action_center"]), f32(self.cfg["action_scale"])
        self.settle = f32(self.cfg["settle_action"])
        self.settle_n = (self.settle - self.act_mean) / self.act_std
        self.domain_frac = self.cfg.get("domain_frac")

        n, c, dev = self.num_envs, self.context, self.device
        self.z1_hist = torch.zeros(n, c, self.z1_dim, device=dev)
        self.act_hist = torch.zeros(n, c, 3, device=dev)
        self.token_hist = torch.zeros(n, c, self.model.token_dim, device=dev)
        self.state_hist8 = torch.zeros(n, self.H, len(self.cols), device=dev)
        self.act_hist8 = torch.zeros(n, self.H, 3, device=dev)
        self.env_domain = torch.zeros(n, dtype=torch.long, device=dev)
        self.pose = torch.zeros(n, 3, device=dev)
        self.z1_phys = torch.zeros(n, self.z1_dim, device=dev)
        self.env_ep = torch.zeros(n, dtype=torch.long, device=dev)
        self.start_frame = torch.zeros(n, dtype=torch.long, device=dev)
        self.route_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.fragment_len = torch.full((n,), self.max_episode_length, dtype=torch.long, device=dev)
        self.episode_length_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.actions = torch.zeros(n, 3, device=dev)
        self.last_actions = torch.zeros(n, 3, device=dev)
        self.last_power = torch.zeros(n, device=dev)
        self.energy_kj = torch.zeros(n, device=dev)
        self.station_cont = torch.zeros(n, device=dev)
        self.progress_m = torch.zeros(n, device=dev)
        self.start_station_m = torch.zeros(n, device=dev)
        self.ep_reward_sum = torch.zeros(n, device=dev)
        self.ep_ct_sum = torch.zeros(n, device=dev)
        self.ep_ct_max = torch.zeros(n, device=dev)
        self.ep_speed_err_sum = torch.zeros(n, device=dev)
        self.search_offsets = torch.arange(-2, int(self.cfg["search_window"]), device=dev)
        self.preview_k = torch.arange(1, self.preview_points + 1, device=dev, dtype=torch.float32)
        self.num_obs = 3 + 3 * self.preview_points + 2 + 3 + 3 * self.H + len(self.cols) * self.H
        self.obs_buf = torch.zeros(n, self.num_obs, device=dev)
        self.rew_buf = torch.zeros(n, device=dev)
        self.reset_buf = torch.zeros(n, dtype=torch.long, device=dev)
        self.time_out_buf = torch.zeros(n, dtype=torch.bool, device=dev)
        self.extras: dict[str, Any] = {}
        self.reset()

    # ------------------------------------------------------------------ metadata
    def obs_layout(self) -> dict:
        lay = obs_layout(self.preview_points, float(self.cfg["preview_spacing_m"]), int(self.cfg["search_window"]),
                         self.H, self.cols, self.cfg["settle_action"])
        assert lay["num_obs"] == self.num_obs, (lay["num_obs"], self.num_obs)
        return lay

    def policy_meta(self) -> dict:
        return {"num_obs": int(self.num_obs), "num_actions": 3, "obs_layout": self.obs_layout(),
                "action_center": [float(v) for v in self.cfg["action_center"]], "action_scale": [float(v) for v in self.cfg["action_scale"]],
                "action_low": [float(v) for v in self.cfg["action_low"]], "action_high": [float(v) for v in self.cfg["action_high"]],
                "steering_rate_limit": self.cfg.get("steering_rate_limit"), "settle_action": [float(v) for v in self.cfg["settle_action"]],
                "nrd_checkpoint": str(self.cfg["nrd_checkpoint"]), "nrd_sha256": self.nrd_sha256, "nrd": self.nrd_payload,
                "cache": str(self.cfg["cache"]), "cache_manifest_sha256": self.bank.manifest_sha256, "bank_split": self.cfg["split"],
                "split_cross_check": self.bank.twin_check,
                "bank_episodes": self.bank.n_episodes, "bank_counts": self.bank.counts(), "dt_s": DT_S,
                "env_cfg": {k: v for k, v in self.cfg.items() if k != "device"}}

    # ------------------------------------------------------------------ resets
    def reset(self) -> tuple[torch.Tensor, dict]:
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_observations()
        return self.obs_buf, self.extras

    def _draw_episodes(self, m: int) -> torch.Tensor:
        b = self.bank
        if self.domain_frac is None or len(b.dom_eps[0]) == 0 or len(b.dom_eps[1]) == 0:
            return torch.randint(0, b.n_episodes, (m,), device=self.device, generator=self.gen)
        crm = torch.rand(m, device=self.device, generator=self.gen) < float(self.domain_frac)
        out = torch.empty(m, dtype=torch.long, device=self.device)
        for d, mask in ((0, ~crm), (1, crm)):
            k = int(mask.sum())
            if k:
                eps = b.dom_eps[d]
                out[mask] = eps[torch.randint(0, len(eps), (k,), device=self.device, generator=self.gen)]
        return out

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        m = env_ids.numel()
        if m == 0:
            return
        b, dev = self.bank, self.device
        episode_ids = self._draw_episodes(m)
        lo, hi = int(self.cfg["fragment_steps_min"]), int(self.cfg["fragment_steps_max"])
        fragment_steps = torch.randint(lo, hi + 1, (m,), device=dev, generator=self.gen)
        # decision frame s in [0, active_end - len]
        lo = torch.zeros(m, device=dev, dtype=torch.long)
        hi = torch.maximum(b.active_end[episode_ids] - fragment_steps, lo)
        u = torch.rand(m, device=dev, generator=self.gen)
        start_frames = lo + (u * (hi - lo + 1).float()).long().clamp(max=hi - lo)
        start_frames = torch.minimum(start_frames, b.length[episode_ids] - 1)
        off = b.offset[episode_ids][:, None]
        # NRD context: frames s-context+1 .. s; frame-0 padding (state/pose of frame 0, settle action)
        win = start_frames[:, None] + torch.arange(-self.context + 1, 1, device=dev)[None, :]
        neg = win < 0
        flat = off + win.clamp(min=0)
        self.z1_hist[env_ids] = b.z1[flat]
        act_ctx = b.act[flat]
        act_ctx[neg] = self.settle_n
        self.act_hist[env_ids] = act_ctx
        hist_pose = b.pose[flat]
        with torch.no_grad():
            self.token_hist[env_ids] = self.model.token(hist_pose)
        self.pose[env_ids] = hist_pose[:, -1]
        self.z1_phys[env_ids] = self.z1_hist[env_ids, -1] * self.z1_std + self.z1_mean
        # observation history: states s-H+1 .. s (frame-0 padded), actions s-H .. s-1 (settle-padded)
        s_idx = start_frames[:, None] + torch.arange(-self.H + 1, 1, device=dev)[None, :]
        a_idx = start_frames[:, None] + torch.arange(-self.H, 0, device=dev)[None, :]
        st = b.z1[off + s_idx.clamp(min=0)] * self.z1_std + self.z1_mean
        self.state_hist8[env_ids] = st[:, :, self.cols]
        a8 = b.act_raw[off + a_idx.clamp(min=0)]
        a8[a_idx < 0] = self.settle
        self.act_hist8[env_ids] = a8
        self.last_actions[env_ids] = a8[:, -1]
        self.actions[env_ids] = a8[:, -1]
        self.env_ep[env_ids] = episode_ids
        self.env_domain[env_ids] = b.domain[episode_ids]
        self.start_frame[env_ids] = start_frames
        self.fragment_len[env_ids] = fragment_steps
        # nearest waypoint over the whole route (fragments start on-route) and the continuous station there
        d = (b.route_xy[episode_ids] - self.pose[env_ids, None, :2]).norm(dim=-1)
        valid = torch.arange(b.route_xy.shape[1], device=dev)[None, :] < b.route_len[episode_ids][:, None]
        d = torch.where(valid, d, torch.full_like(d, float("inf")))
        idx = d.argmin(dim=1)
        self.route_idx[env_ids] = idx
        wp, h = b.route_xy[episode_ids, idx], b.route_h[episode_ids, idx]
        e_along = (self.pose[env_ids, 0] - wp[:, 0]) * torch.cos(h) + (self.pose[env_ids, 1] - wp[:, 1]) * torch.sin(h)
        self.station_cont[env_ids] = b.route_s[episode_ids, idx] + e_along
        self.start_station_m[env_ids] = b.route_s[episode_ids, idx]
        self.progress_m[env_ids] = 0.0
        for buf in (self.episode_length_buf, self.energy_kj, self.ep_reward_sum, self.ep_ct_sum, self.ep_ct_max, self.ep_speed_err_sum):
            buf[env_ids] = 0

    # ------------------------------------------------------------------ dynamics
    def _scale_policy_actions(self, policy_actions: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.action_center + self.action_scale * torch.tanh(policy_actions), self.action_low, self.action_high)

    def physical_to_policy(self, physical: torch.Tensor, clip: float = 0.99) -> torch.Tensor:
        """Inverse of the squash (pre-tanh actions); the box edges are pulled inside by ``clip``."""
        u = ((physical - self.action_center) / self.action_scale).clamp(-clip, clip)
        return torch.atanh(u)

    @torch.no_grad()
    def _nn_step(self, driver_actions: torch.Tensor) -> None:
        self.act_hist[:, -1] = (driver_actions - self.act_mean) / self.act_std
        delta, power = self.model(self.z1_hist, self.token_hist, self.act_hist, self.env_domain)
        z1_next = self.z1_hist[:, -1] + delta[:, -1]
        self.z1_phys = z1_next * self.z1_std + self.z1_mean
        self.pose = integrate_pose(self.pose, self.z1_phys)
        token_next = self.model.token(self.pose.unsqueeze(1))[:, 0]
        self.z1_hist = torch.cat([self.z1_hist[:, 1:], z1_next.unsqueeze(1)], dim=1)
        self.token_hist = torch.cat([self.token_hist[:, 1:], token_next.unsqueeze(1)], dim=1)
        self.act_hist = torch.cat([self.act_hist[:, 1:], self.act_hist[:, -1:]], dim=1)
        self.act_hist8 = torch.cat([self.act_hist8[:, 1:], driver_actions.unsqueeze(1)], dim=1)
        self.state_hist8 = torch.cat([self.state_hist8[:, 1:], self.z1_phys[:, self.cols].unsqueeze(1)], dim=1)
        self.last_power = power[:, -1, 0] * self.p_std + self.p_mean  # kW
        self.energy_kj += self.last_power * DT_S

    # ------------------------------------------------------------------ route geometry
    def _route_errors(self) -> dict[str, torch.Tensor]:
        b, ep = self.bank, self.env_ep
        last = (b.route_len[ep] - 1)[:, None]
        cand = torch.minimum((self.route_idx[:, None] + self.search_offsets[None, :]).clamp(min=0), last)
        pts = b.route_xy[ep[:, None], cand]
        d = (pts - self.pose[:, None, :2]).norm(dim=-1)
        j = d.argmin(dim=1, keepdim=True)
        idx = cand.gather(1, j)[:, 0]
        self.route_idx = idx
        wp, h, v_ref = b.route_xy[ep, idx], b.route_h[ep, idx], b.route_v[ep, idx]
        dx, dy = self.pose[:, 0] - wp[:, 0], self.pose[:, 1] - wp[:, 1]
        cos_h, sin_h = torch.cos(h), torch.sin(h)
        e_along = dx * cos_h + dy * sin_h
        e_ct = -dx * sin_h + dy * cos_h
        e_h = wrap_angle(self.pose[:, 2] - h)
        e_v = self.z1_phys[:, VX] - v_ref
        station = b.route_s[ep, idx]
        self.progress_m = station - self.start_station_m
        return {"e_along": e_along, "e_ct": e_ct, "e_h": e_h, "e_v": e_v, "v_ref": v_ref,
                "station_cont": station + e_along, "route_end": idx >= (b.route_len[ep] - 2)}

    def _preview_body(self) -> torch.Tensor:
        b, ep = self.bank, self.env_ep
        step = (self.preview_k[None, :] * float(self.cfg["preview_spacing_m"]) / b.route_ds[ep][:, None]).round().long()
        idx = torch.minimum(self.route_idx[:, None] + step, (b.route_len[ep] - 1)[:, None])
        pts = b.route_xy[ep[:, None], idx]
        v = b.route_v[ep[:, None], idx]
        dx, dy = pts[..., 0] - self.pose[:, None, 0], pts[..., 1] - self.pose[:, None, 1]
        cos_y, sin_y = torch.cos(self.pose[:, 2])[:, None], torch.sin(self.pose[:, 2])[:, None]
        bx, by = cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy
        return torch.stack([bx / 10.0, by / 10.0, v / 5.0], dim=-1).flatten(1)

    # ------------------------------------------------------------------ rsl_rl interface
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        actions = actions.to(device=self.device, dtype=torch.float32)
        driver = self._scale_policy_actions(actions)
        limit = self.cfg.get("steering_rate_limit")
        if limit is not None:
            driver = driver.clone()
            driver[:, 0] = torch.clamp(driver[:, 0], self.last_actions[:, 0] - float(limit), self.last_actions[:, 0] + float(limit))
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
        clip_m = float(rc.get("progress_clip_m", 2.0))
        advance = (err["station_cont"] - self.station_cont).clamp(-clip_m, clip_m)
        advance = torch.where(torch.isfinite(advance), advance, torch.zeros_like(advance))
        self.station_cont = err["station_cont"]
        reward = (track_reward - float(rc["action_rate_weight"]) * action_rate
                  - float(rc["throttle_brake_weight"]) * throttle_brake + float(rc.get("progress_weight", 0.0)) * advance)

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
        extras = self._make_extras(err, track_reward, action_rate, throttle_brake, advance, dones, failed, bootstrap)
        self.last_actions = driver.clone()
        if self.auto_reset:
            done_ids = dones.nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                self.reset_idx(done_ids)
        self._compute_observations()
        extras["observations"] = {"critic": self.obs_buf}
        self.extras = extras
        return self.obs_buf, self.rew_buf, dones.long(), self.extras

    def _make_extras(self, err, track_reward, action_rate, throttle_brake, advance, dones, failed, bootstrap):
        """Per-step means (``log``) and, when envs finish, per-episode means (``episode``) for the rsl_rl logger."""
        log = {
            "/tracking/track_reward": track_reward.mean(),
            "/tracking/cross_track_abs_m": err["e_ct"].abs().mean(),
            "/tracking/heading_err_abs_rad": err["e_h"].abs().mean(),
            "/tracking/speed_err_abs_mps": err["e_v"].abs().mean(),
            "/tracking/action_rate": action_rate.mean(),
            "/tracking/throttle_brake": throttle_brake.mean(),
            "/tracking/advance_m": advance.mean(),
            "/tracking/vx_mps": self.z1_phys[:, VX].mean(),
        }
        for d, name in enumerate(D.DOMAIN_VOCAB):
            m = self.env_domain == d
            if int(m.sum()):
                log[f"/tracking/{name}/track_reward"] = track_reward[m].mean()
                log[f"/tracking/{name}/cross_track_abs_m"] = err["e_ct"][m].abs().mean()
                log[f"/tracking/{name}/speed_err_abs_mps"] = err["e_v"][m].abs().mean()
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
                "/episode/energy_kj": self.energy_kj[done_ids].mean(),
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
            self.act_hist8.flatten(1),
            self.state_hist8.flatten(1),
        ]
        self.obs_buf = torch.cat(parts, dim=-1)
        self.obs_buf = torch.where(torch.isfinite(self.obs_buf), self.obs_buf, torch.zeros_like(self.obs_buf))
        self.extras = {"observations": {"critic": self.obs_buf}}

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        self._compute_observations()
        return self.obs_buf, self.extras

    # ------------------------------------------------------------------ imitation samples
    @torch.no_grad()
    def sample_imitation(self, n_resets: int) -> dict[str, torch.Tensor]:
        """``n_resets`` x num_envs pairs (observation at a random decision frame s of a random bank episode, the
        recorded PID action act[s]) with their domain; the env is reset again afterwards."""
        obs, tgt, dom = [], [], []
        ids = torch.arange(self.num_envs, device=self.device)
        for _ in range(int(n_resets)):
            self.reset_idx(ids)
            self._compute_observations()
            obs.append(self.obs_buf.clone())
            tgt.append(self.bank.act_raw[self.bank.offset[self.env_ep] + self.start_frame].clone())
            dom.append(self.env_domain.clone())
        self.reset()
        return {"obs": torch.cat(obs), "action": torch.cat(tgt), "domain": torch.cat(dom)}
