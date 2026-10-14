"""Start states from recorded G1 episodes: the NRD history and the HOMIE teacher memory at a control step.

The data are the contact-NRD G1 locomotion data (production v1 by default; the headline NRD trained on it). Only the
chosen rows of the 18 GB unified_data.npz are read (memory map). Splits: 0 training, 1 validation (also used to
select the NRD checkpoint), 2 sealed test (never read here: every loader refuses split 2 rows).

Time grid. Records are 10 ms; a control step is 20 ms = 2 records; control step k starts at record 2k. The teacher
decided step k from the state at record 2k (and the command of record 2k); its raw output is policy_output[2k], the
applied 27 targets are action[2k] (= action[2k + 1]).

A start at control step k holds:
- the NRD history: the states at records 2(k-3), 2(k-2), 2(k-1), 2k with 20 ms step-average rates (exactly
  nedm.contact_nrd.data.step_average_rates, as the window rollouts of the evaluation build them; records 0 and 1
  keep the recorded rates), ok = record >= 0, entries before the episode start zeroed;
- the teacher memory before it decides step k: 6 frames of the steps k-6 .. k-1 (oldest first; zero frames before the
  episode start, as after the deploy reset), each from the state at record 2j with averaged rates, the command of
  record 2j and the previous clipped output clip(policy_output[2(j-1)]) (zero for j = 0); and the last clipped output
  clip(policy_output[2(k-1)]).
A teacher step HomieTeacher.step(command[2k], hist[:, -1]) from this memory gives the teacher's targets of step k.
With instantaneous rates (rates="instantaneous") the same reconstruction reproduces the recorded outputs up to the
float32 rounding of the stored states (check in scripts/g1_footstep_rl/build_start_bank.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from nedm.g1_footstep_rl.homie_policy import ACTION_CLIP, DEFAULT_ANGLES, FRAME, HISTORY, N_LEG
from nedm.contact_nrd.data import step_average_rates
from nedm.g1_footstep_rl.npz_rows import load_rows, npz_members
from nedm.g1_footstep_rl import REPO_ROOT
from nedm.g1_footstep_rl.homie import state_frame

DATA = REPO_ROOT / "artifacts/contact_nrd/g1_locomotion/train"
FAMILIES = ("nominal", "difficult", "probe_drop", "probe_landing")
KINDS = ("stand", "squat", "turn", "straight", "lateral", "walk", "other")
FULL_HEIGHT = 0.74
STRIDE = 2             # records per control (and model) step
NRD_HISTORY = 4


def open_arrays(data_dir=DATA):
    """{name: read-only memmap} of unified_data.npz (nothing is read until indexed)."""
    return npz_members(Path(data_dir) / "unified_data.npz")


def kind_of(c):
    """Segment kind from the exact zeros of a command row (vx, vy, wz, height) (g1_locomotion.sample_commands)."""
    vx, vy, wz, h = (float(v) for v in c[:4])
    if vx == 0 and vy == 0 and wz == 0:
        return "stand" if h >= FULL_HEIGHT - 1e-6 else "squat"
    if vx == 0 and vy == 0:
        return "turn"
    if vy == 0 and vx != 0:
        return "straight"
    if wz == 0 and vy != 0:
        return "lateral"
    if vx != 0 and vy != 0 and wz != 0:
        return "walk"
    return "other"


def kind_codes(command):
    """[..., 4] commands -> [...] int8 indices into KINDS (vectorised kind_of)."""
    c = np.asarray(command, np.float64)
    vx, vy, wz, h = np.moveaxis(c, -1, 0)
    out = np.full(vx.shape, KINDS.index("other"), np.int8)
    zero = (vx == 0) & (vy == 0) & (wz == 0)
    out[zero & (h >= FULL_HEIGHT - 1e-6)] = KINDS.index("stand")
    out[zero & (h < FULL_HEIGHT - 1e-6)] = KINDS.index("squat")
    out[(vx == 0) & (vy == 0) & (wz != 0)] = KINDS.index("turn")
    out[(vy == 0) & (vx != 0)] = KINDS.index("straight")
    out[(wz == 0) & (vy != 0)] = KINDS.index("lateral")
    out[(vx != 0) & (vy != 0) & (wz != 0)] = KINDS.index("walk")
    return out


def load_episodes(rows, data_dir=DATA, arrays=None):
    """Rows of the data -> dict of numpy arrays (E episodes, T records): raw [E, T, 67] (stored, instantaneous rates),
    avg [E, T, 67] (20 ms step-average rates, data.step_average_rates), action [E, T, 27], command [E, T, 4],
    policy_output [E, T, 12], contacts [E, T-1, 2] bool, lengths, splits, family, episode_id and the per-episode
    scalars. Refuses sealed-test rows (split 2)."""
    rows = np.asarray(rows, np.int64)
    arrays = arrays if arrays is not None else open_arrays(data_dir)
    splits = np.asarray(arrays["splits"][rows])
    if (splits == 2).any():
        raise ValueError("sealed test rows (split 2) must not be read")
    d = load_rows(data_dir, rows, "cpu", 0.02)
    raw = d["states"].clone()
    step_average_rates(d, {"articulated"})
    out = dict(rows=rows, raw=raw[:, :, 0].numpy(), avg=d["states"][:, :, 0].numpy(),
               action=d["actions"][:, :, 0].numpy(), contacts=d["contacts"].numpy(),
               lengths=d["lengths"].numpy().astype(np.int64), splits=splits, family=d["family"].numpy(),
               episode_id=np.asarray(d["episode_id"]))
    for key in ("command", "policy_output"):
        out[key] = np.asarray(arrays[key][rows], np.float32)
    for key in ("leg_noise", "upper_motion", "abrupt_commands", "perturbed_start", "terminal_reason", "seed",
                "group_id"):
        out[key] = np.asarray(arrays[key][rows])
    return out


def nrd_history(states, e, k):
    """NRD histories at control steps k of episodes e (index arrays [N]) from states [E, T, 67] -> (hist [N, 4, 67]
    float64, ok [N, 4] bool). Entries before the episode start are zeroed and masked (the window mode of the
    evaluation)."""
    lag = np.arange(1 - NRD_HISTORY, 1)
    rec = (np.asarray(k)[:, None] + lag) * STRIDE
    ok = rec >= 0
    hist = states[np.asarray(e)[:, None], np.maximum(rec, 0)] * ok[..., None]
    return hist.astype(np.float64), ok


def teacher_memory(states, command, policy_output, e, k):
    """Teacher memory before it decides control step k (see the module docstring) -> (history [N, 6, 76] float32,
    last_action [N, 12] float32). states [E, T, 67] with the rates the teacher should see (averaged in the NRD)."""
    e, k = np.asarray(e), np.asarray(k)
    j = k[:, None] + np.arange(-HISTORY, 0)                       # steps k-6 .. k-1, oldest first
    valid = j >= 0
    jr = np.maximum(j, 0) * STRIDE                                 # record of each step
    prev = np.maximum(j - 1, 0) * STRIDE                           # record of the step before
    st = torch.as_tensor(states[e[:, None], jr].reshape(-1, 67))
    cmd = torch.as_tensor(command[e[:, None], jr].reshape(-1, 4))
    last = np.clip(policy_output[e[:, None], prev], -ACTION_CLIP, ACTION_CLIP) * (j >= 1)[..., None]
    f = state_frame(cmd, st, torch.as_tensor(last.reshape(-1, N_LEG), dtype=torch.float32)).numpy()
    hist = f.reshape(len(e), HISTORY, FRAME) * valid[..., None]
    last_k = np.clip(policy_output[e, np.maximum(k - 1, 0) * STRIDE], -ACTION_CLIP, ACTION_CLIP) * (k >= 1)[:, None]
    return hist.astype(np.float32), last_k.astype(np.float32)


def teacher_targets(policy_output):
    """Recorded raw outputs [..., 12] -> the teacher's leg targets [..., 12] (without any added leg noise)."""
    return DEFAULT_ANGLES[:N_LEG] + 0.25 * np.clip(np.asarray(policy_output, np.float64), -ACTION_CLIP, ACTION_CLIP)


def heading_rates_np(states):
    """[..., 67] -> [..., 3] forward and sideways velocity in the heading frame, world yaw rate."""
    q = states[..., 3:7]
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = states[..., 34], states[..., 35]
    return np.stack((c * vx + s * vy, -s * vx + c * vy, states[..., 39]), -1)


def segment_start(command, lengths):
    """[E, T] record index at which the current constant-command segment began."""
    E, T = command.shape[:2]
    change = np.ones((E, T), bool)
    change[:, 1:] = np.any(command[:, 1:] != command[:, :-1], -1)
    idx = np.where(change, np.arange(T)[None], 0)
    return np.maximum.accumulate(idx, axis=1)


def start_labels(ep, e, k):
    """Labels of starts (episode index e, control step k) in loaded episodes ep (load_episodes)."""
    e, k = np.asarray(e), np.asarray(k)
    r = k * STRIDE
    cmd = ep["command"][e, r]
    seg0 = segment_start(ep["command"], ep["lengths"])[e, r]
    vel = heading_rates_np(ep["avg"][e, r])
    # 0.5 s past mean: displacement over the last 0.5 s, rotated into the heading at the start; yaw change / 0.5 s
    back = np.maximum(r - 50, 0)
    have = r >= 50
    st, sb = ep["raw"][e, r], ep["raw"][e, back]
    yaw = lambda s: np.arctan2(2 * (s[..., 3] * s[..., 6] + s[..., 4] * s[..., 5]), 1 - 2 * (s[..., 5] ** 2 + s[..., 6] ** 2))
    y1, y0 = yaw(st), yaw(sb)
    dxy = st[:, :2] - sb[:, :2]
    c, s = np.cos(y1), np.sin(y1)
    mean = np.stack((c * dxy[:, 0] + s * dxy[:, 1], -s * dxy[:, 0] + c * dxy[:, 1],
                     np.angle(np.exp(1j * (y1 - y0)))), -1) / 0.5
    mean[~have] = np.nan
    con = ep["contacts"]
    n_int = con.shape[1]
    before = con[e, np.clip(r - 2, 0, n_int - 1)] | con[e, np.clip(r - 1, 0, n_int - 1)]
    before &= (r >= 1)[:, None]
    after = con[e, np.clip(r, 0, n_int - 1)] | con[e, np.clip(r + 1, 0, n_int - 1)]
    steps_left = (ep["lengths"][e] - 1) // STRIDE - k
    return dict(command=cmd.astype(np.float32), kind=kind_codes(cmd), t_seg=((r - seg0) * 0.01).astype(np.float32),
                vel_avg=vel.astype(np.float32), vel_mean_0p5s=mean.astype(np.float32), contact_before=before,
                contact_after=after, steps_to_end=steps_left.astype(np.int16), time_s=(r * 0.01).astype(np.float32))


def build_starts(ep, e, k, rates="averaged"):
    """All bank fields of starts (e, k) in loaded episodes ep: NRD history (averaged rates), the teacher memory
    (rates "averaged" = what the teacher sees in the NRD, or "instantaneous" = what it saw in MuJoCo), the recorded
    command / raw output / applied action of step k and the labels."""
    e, k = np.asarray(e), np.asarray(k)
    hist, ok = nrd_history(ep["avg"], e, k)
    th, tl = teacher_memory(ep["avg"] if rates == "averaged" else ep["raw"], ep["command"], ep["policy_output"], e, k)
    r = k * STRIDE
    out = dict(hist=hist, ok=ok, teacher_history=th, teacher_last_action=tl,
               policy_output=ep["policy_output"][e, r].astype(np.float32), action=ep["action"][e, r].astype(np.float32),
               row=ep["rows"][e].astype(np.int64), episode_id=ep["episode_id"][e].astype(np.int64),
               k=k.astype(np.int16), family=ep["family"][e].astype(np.int8),
               leg_noise=ep["leg_noise"][e].astype(np.float32), upper_motion=ep["upper_motion"][e].astype(np.int8),
               abrupt_commands=ep["abrupt_commands"][e].astype(np.int8),
               perturbed_start=ep["perturbed_start"][e].astype(np.int8),
               early_end=(ep["terminal_reason"][e] != 0).astype(np.int8))
    out.update(start_labels(ep, e, k))
    return out


class Bank:
    """A saved start bank (bank_v1.npz, written by scripts/g1_footstep_rl/build_start_bank.py), memory-mapped.

    sample(n, rng, family=None) -> dict of torch tensors (hist float64, ok, teacher_history, teacher_last_action,
    command, ...) for n random starts."""

    def __init__(self, path):
        self.path = Path(path)
        self.arrays = npz_members(self.path)
        self.n = len(self.arrays["k"])
        meta = self.path.with_suffix(".json")
        self.summary = json.loads(meta.read_text()) if meta.exists() else {}

    def take(self, idx, device="cpu"):
        idx = np.sort(np.asarray(idx, np.int64))
        out = {}
        for key, arr in self.arrays.items():
            x = np.array(arr[idx])
            out[key] = torch.as_tensor(x, device=device) if x.dtype != np.bool_ else torch.as_tensor(x, device=device).bool()
        return out

    def sample(self, n, rng, family=None, device="cpu"):
        pool = np.arange(self.n) if family is None else np.flatnonzero(np.asarray(self.arrays["family"]) == FAMILIES.index(family))
        return self.take(rng.choice(pool, size=n, replace=len(pool) < n), device)
