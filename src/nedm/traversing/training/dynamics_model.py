"""The NRD vehicle-dynamics model: main's causal transformer over [state, terrain-crop token, action, ground type].

Per 50 ms frame the input token is ``[z1 (17, z-scored), crop token (64), action (3, z-scored), domain one-hot (2)]``;
the heads predict the change of the normalised z1 and the normalised power. The crop token is an MLP on an 8 x 8
ego-aligned elevation crop of the f104 terrain grid (+-6 m, heights relative to the vehicle centre / 2 m, plus a
validity flag), re-taken at the dead-reckoned pose every step. The pose is integrated outside the network
(``integrate_pose``). Backbone: ``nedm.core.training.model_transformer.ContinuousTransformer`` (main, unchanged).

    model, norm, payload = load_nrd("artifacts/traverse/generalist_20260921/B_tracker/nrd_tag_v3/ckpt_best.pt", "cuda")
    delta, power = model(z1_norm, model.token(pose), act_norm, domain)   # (B, L, 17), (B, L, 1)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nedm.traversing.training.dynamics_data import DEFAULT_GRID, DOMAIN_VOCAB, ROOT, Normalizer, file_sha256, resolve_path
from nedm.traversing.training.state import ACT_DIM, DT_S, VX, VY, YAW_RATE, Z1_DIM

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from nedm.core.training.model_transformer import ContinuousTransformer, TransformerConfig  # noqa: E402

MODEL_KIND = "gb_nrd"  # checkpoint tag of the released model
SCALE_M = 2.0  # crop heights are divided by this


def integrate_pose(pose: torch.Tensor, z1_phys: torch.Tensor) -> torch.Tensor:
    """(B, 3) pose, (B, 17) physical-unit state -> (B, 3) pose one frame later (yaw first, then the world velocity
    with the new yaw)."""
    yaw = pose[:, 2] + DT_S * z1_phys[:, YAW_RATE]
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    vx_world = cos_yaw * z1_phys[:, VX] - sin_yaw * z1_phys[:, VY]
    vy_world = sin_yaw * z1_phys[:, VX] + cos_yaw * z1_phys[:, VY]
    return torch.stack([pose[:, 0] + DT_S * vx_world, pose[:, 1] + DT_S * vy_world, yaw], dim=1)


# ------------------------------------------------------------------------------------------ terrain crop
def load_grid(path: str) -> dict:
    """grid.npz -> dict(z (n, n) f32 with row 0 at -y and columns along +x, mpp, half_extent, n). ``grid.json`` next
    to it may set ``half_extent_m`` / ``mpp``; the f104 grid is 512 x 512 over +-40 m either way."""
    if os.path.isdir(path):
        path = os.path.join(path, "grid.npz")
    z = np.asarray(np.load(path)["z"], dtype=np.float32)
    meta_path = os.path.join(os.path.dirname(path), "grid.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    n = z.shape[0]
    half = float(meta.get("half_extent_m", 40.0)); mpp = float(meta.get("mpp", 2 * half / n))
    assert z.shape == (n, n) and abs(mpp * n - 2 * half) < 1e-6, (z.shape, mpp, half)
    assert np.isfinite(z).all(), "grid has empty (NaN) cells"
    return dict(z=z, mpp=mpp, half_extent=half, n=n, path=os.path.abspath(path))


class EgoCrop(nn.Module):
    """k x k heights at ``linspace(-half_m, half_m, k)`` forward (axis 0) x left (axis 1) of a pose, minus the height
    at the pose, / 2 m; bilinear (``grid_sample``, border padding), differentiable in the pose."""

    def __init__(self, grid: dict, device="cuda") -> None:
        super().__init__()
        self.register_buffer("z", torch.tensor(np.asarray(grid["z"], dtype=np.float32))[None, None], persistent=False)
        self.mpp, self.half_extent = float(grid["mpp"]), float(grid["half_extent"])
        self._off: dict = {}
        self.to(device)

    def _apply(self, fn, *a, **kw):  # guard: .half()/.bfloat16() must not quantise the grid (training is float32)
        z = self.z; super()._apply(fn, *a, **kw); self.z = z.to(self.z.device); return self

    def sample_height(self, xy: torch.Tensor) -> torch.Tensor:
        """xy (..., 2) -> (...,) map height; align_corners=False maps x to column (x + half) / mpp - 0.5."""
        g = xy.float() / self.half_extent
        out = F.grid_sample(self.z.float(), g.reshape(1, -1, 1, 2), mode="bilinear", padding_mode="border", align_corners=False)
        return out.reshape(xy.shape[:-1])

    def forward(self, pose: torch.Tensor, k: int = 8, half_m: float = 6.0):
        """pose (..., 3) -> heights (..., k, k), valid (...,) bool (every sample point inside the grid)."""
        key = (int(k), float(half_m))
        if key not in self._off or self._off[key][0].device != self.z.device:
            with torch.inference_mode(False):  # cached offsets must stay usable for backward
                o = np.linspace(-half_m, half_m, k)
                du, dv = np.meshgrid(o, o, indexing="ij")  # du forward, dv left
                self._off[key] = tuple(torch.as_tensor(a.ravel(), dtype=torch.float32, device=self.z.device) for a in (du, dv))
        du, dv = self._off[key]
        pose = pose.float()
        x, y, yaw = pose[..., 0:1], pose[..., 1:2], pose[..., 2:3]
        c, s = torch.cos(yaw), torch.sin(yaw)
        px, py = x + du * c - dv * s, y + du * s + dv * c  # (..., k*k)
        h = self.sample_height(torch.stack([px, py], -1)) - self.sample_height(pose[..., :2])[..., None]
        valid = ((px.abs() < self.half_extent) & (py.abs() < self.half_extent)).all(-1)
        return (h / SCALE_M).reshape(*pose.shape[:-1], k, k), valid


class CropTokenizer(nn.Module):
    """Crop at the pose -> token: MLP(k*k heights + valid flag -> hidden -> token_dim)."""

    def __init__(self, grid: dict, k: int, half_m: float, token_dim: int = 64, hidden: int = 128, device="cpu"):
        super().__init__()
        self.crop = EgoCrop(grid, device)
        self.k, self.half_m, self.token_dim = int(k), float(half_m), int(token_dim)
        self.mlp = nn.Sequential(nn.Linear(self.k * self.k + 1, hidden), nn.GELU(), nn.Linear(hidden, token_dim))

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        h, ok = self.crop(pose, self.k, self.half_m)
        x = torch.cat([h.flatten(-2), ok.to(h.dtype).unsqueeze(-1)], dim=-1)
        return self.mlp(x.to(self.mlp[0].weight.dtype))


# ------------------------------------------------------------------------------------------ model
class NRDModel(nn.Module):
    """Backbone over [z1, crop token, action, domain one-hot]; heads: delta z1 (normalised), power (normalised)."""

    def __init__(self, cfg: dict, grid: dict, device="cpu"):
        super().__init__()
        self.cfg = dict(cfg)
        if cfg["cond"] != "tag":
            raise ValueError(f"only the ground-type-conditioned model (cond='tag') is supported, got {cfg['cond']!r}")
        self.z1_dim, self.act_dim = int(cfg.get("z1_dim", Z1_DIM)), int(cfg.get("act_dim", ACT_DIM))
        self.token_dim = int(cfg.get("token_dim", 64))
        self.n_domains = len(DOMAIN_VOCAB)
        self.tokenizer = CropTokenizer(grid, int(cfg["crop_k"]), float(cfg["crop_half_m"]), self.token_dim,
                                       int(cfg.get("token_hidden", 128)), device)
        self.backbone = ContinuousTransformer(TransformerConfig(
            input_dim=self.z1_dim + self.token_dim + self.act_dim + self.n_domains, block_size=int(cfg["block_size"]),
            n_layer=int(cfg["n_layer"]), n_head=int(cfg["n_head"]), n_embd=int(cfg["n_embd"]),
            dropout=float(cfg["dropout"]), bias=bool(cfg["bias"])))
        hidden, n_embd = int(cfg["head_hidden_dim"]), int(cfg["n_embd"])
        mlp = lambda out: nn.Sequential(nn.Linear(n_embd, hidden), nn.GELU(), nn.Linear(hidden, out))
        self.state_head, self.power_head = mlp(self.z1_dim), mlp(1)
        self.to(device)

    def token(self, pose: torch.Tensor) -> torch.Tensor:
        """pose (..., 3) -> crop token (..., token_dim)."""
        return self.tokenizer(pose)

    def forward(self, z1: torch.Tensor, token: torch.Tensor, act: torch.Tensor, domain: torch.Tensor):
        """z1 (B, L, 17), token (B, L, 64), act (B, L, 3), domain (B,) int -> delta (B, L, 17), power (B, L, 1)."""
        oh = F.one_hot(domain.long(), self.n_domains).to(z1.dtype)
        feat = self.backbone(torch.cat([z1, token, act, oh[:, None, :].expand(-1, z1.shape[1], -1)], dim=-1))
        return self.state_head(feat), self.power_head(feat)


def model_config(*, crop_k: int, crop_half_m: float, block_size: int = 16, n_layer: int = 6, n_head: int = 8,
                 n_embd: int = 256, token_dim: int = 64, token_hidden: int = 128) -> dict:
    return {"block_size": int(block_size), "n_layer": int(n_layer), "n_head": int(n_head), "n_embd": int(n_embd),
            "dropout": 0.0, "bias": False, "head_hidden_dim": int(n_embd), "z1_dim": Z1_DIM,
            "act_dim": ACT_DIM, "token_dim": int(token_dim), "token_hidden": int(token_hidden), "cond": "tag",
            "crop_k": int(crop_k), "crop_half_m": float(crop_half_m)}


# ------------------------------------------------------------------------------------------ checkpoint
def save_nrd(path, model: NRDModel, norm: Normalizer, step: int, metrics: dict, grid_path: str, grid_sha256: str,
             delta_scale=None, extra: dict | None = None, train_state: dict | None = None) -> None:
    """Write the checkpoint atomically; ``train_state`` (optimizer, schedule, RNG) only in ``ckpt_last.pt``."""
    cfg = dict(model.cfg)
    payload = {"model_kind": MODEL_KIND, "model": model.state_dict(), "config": cfg, "normalization": norm.to_dict(),
               "step": int(step), "metrics": metrics, "z1_dim": model.z1_dim, "crop_k": cfg["crop_k"],
               "crop_half_m": cfg["crop_half_m"], "cond": cfg["cond"], "token_dim": cfg["token_dim"],
               "grid_path": str(grid_path), "grid_sha256": grid_sha256, "domain_vocab": list(DOMAIN_VOCAB),
               "delta_scale": None if delta_scale is None else [float(v) for v in delta_scale]}
    if extra:
        payload.update(extra)
    if train_state is not None:
        payload["train_state"] = train_state
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_nrd(path, device, grid_path: str | None = None, frozen: bool = True):
    """Checkpoint -> (model on device, Normalizer, payload). The grid is ``grid_path``, else the checkpoint's stored
    path if it exists, else ``DEFAULT_GRID``; its sha256 must equal the checkpoint's."""
    payload = torch.load(resolve_path(path), map_location="cpu", weights_only=False)
    if payload.get("model_kind") != MODEL_KIND:
        raise ValueError(f"{path}: model_kind {payload.get('model_kind')!r} != {MODEL_KIND!r}")
    gp = grid_path or (payload["grid_path"] if resolve_path(payload["grid_path"]).exists() else DEFAULT_GRID)
    gpath = resolve_path(gp)
    if not gpath.exists():
        raise FileNotFoundError(f"grid {gp} not found (checkpoint grid sha256 {payload['grid_sha256'][:16]})")
    sha = file_sha256(gpath)
    if sha != payload["grid_sha256"]:
        raise ValueError(f"grid {gp} sha256 {sha[:16]} != checkpoint's {payload['grid_sha256'][:16]}")
    model = NRDModel(payload["config"], load_grid(str(gpath)), device)
    model.load_state_dict(payload["model"], strict=True)
    if frozen:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model, Normalizer.from_dict(payload["normalization"]), payload
