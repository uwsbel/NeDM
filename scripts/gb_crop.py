"""Ego-aligned elevation crops from the Chrono-frame v2 grid (generalist plan, dynamics crops for B).

Grid: ``grid.npz['z']`` (512 x 512, metres) with rows increasing with +y (row 0 = -y) and columns increasing with +x,
cell centres at ``-half + (i + 0.5) * mpp`` (same convention as ``sensor_dataset_v2.sample`` and ``TerrainMap``). The
grid is a measurement of Chrono's mesh, so poses need no 511/512 scale. Crop: k x k samples at
``linspace(-half_m, half_m, k)`` forward (axis 0) x left (axis 1), heights minus the map height at the vehicle centre,
divided by 2 m (as ``traverse_wp2_add_terrain.terrain_patch``); ``valid`` = every sample point inside +-half_extent.
Numpy and torch (``grid_sample``, bilinear, border padding, ``align_corners=False``: the normalised coordinate maps to
exactly ``fx = (x + half) / mpp - 0.5``) agree to < 1e-4. Self-test: ``python scripts/gb_crop.py --selftest --out DIR``.
"""
from __future__ import annotations

import argparse, hashlib, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCALE_M = 2.0  # heights are divided by this (arena relief ~ +-2 m), matching the NRD cache terrain field


def load_grid(path: str) -> dict:
    """grid.npz (or its directory) -> dict(z (n,n) f32 row0=-y, mpp, half_extent, n). Requires a fully covered grid."""
    if os.path.isdir(path):
        path = os.path.join(path, "grid.npz")
    z = np.asarray(np.load(path)["z"], dtype=np.float32)
    meta_path = os.path.join(os.path.dirname(path), "grid.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    n = z.shape[0]
    half = float(meta.get("half_extent_m", 40.0)); mpp = float(meta.get("mpp", 2 * half / n))
    assert z.shape == (n, n) and abs(mpp * n - 2 * half) < 1e-6, (z.shape, mpp, half)
    assert np.isfinite(z).all(), "grid has empty (NaN) cells; gb_crop expects full coverage"
    return dict(z=z, mpp=mpp, half_extent=half, n=n, path=os.path.abspath(path))


def _offsets(k: int, half_m: float):
    o = np.linspace(-half_m, half_m, k)
    du, dv = np.meshgrid(o, o, indexing="ij")  # du forward, dv left
    return du.ravel(), dv.ravel()


def _ego_points(x, y, yaw, du, dv):
    c, s = np.cos(yaw), np.sin(yaw)
    return x + du * c - dv * s, y + du * s + dv * c


def sample_height(grid: dict, xy: np.ndarray) -> np.ndarray:
    """Bilinear map height at world xy (..., 2) -> (...,), border-clamped (numpy; torch: EgoCrop.sample_height)."""
    xy = np.asarray(xy, dtype=np.float64)
    z, n = grid["z"], grid["n"]
    fx = (xy[..., 0] + grid["half_extent"]) / grid["mpp"] - 0.5
    fy = (xy[..., 1] + grid["half_extent"]) / grid["mpp"] - 0.5
    i0 = np.clip(np.floor(fy).astype(int), 0, n - 2); j0 = np.clip(np.floor(fx).astype(int), 0, n - 2)
    ty = np.clip(fy - i0, 0, 1); tx = np.clip(fx - j0, 0, 1)
    return (z[i0, j0] * (1 - ty) * (1 - tx) + z[i0, j0 + 1] * (1 - ty) * tx
            + z[i0 + 1, j0] * ty * (1 - tx) + z[i0 + 1, j0 + 1] * ty * tx)


def ego_crop_np(grid: dict, pose: np.ndarray, k: int = 8, half_m: float = 6.0):
    """pose (..., 3) world x, y, yaw -> heights (..., k, k) f32 relative to the centre / 2 m, valid (...,) bool."""
    pose = np.asarray(pose, dtype=np.float64)
    du, dv = _offsets(k, half_m)
    px, py = _ego_points(pose[..., 0:1], pose[..., 1:2], pose[..., 2:3], du, dv)  # (..., k*k)
    h = sample_height(grid, np.stack([px, py], -1)) - sample_height(grid, pose[..., :2])[..., None]
    valid = ((np.abs(px) < grid["half_extent"]) & (np.abs(py) < grid["half_extent"])).all(-1)
    return (h / SCALE_M).astype(np.float32).reshape(*pose.shape[:-1], k, k), valid


class EgoCrop(nn.Module):
    """Torch twin of ego_crop_np on one static grid (non-persistent buffer), differentiable in pose."""

    def __init__(self, grid: dict, device="cuda") -> None:
        super().__init__()
        # torch.tensor copies: a CPU module must not alias the caller's numpy grid (in-place edits of grid['z'] would leak in)
        self.register_buffer("z", torch.tensor(np.asarray(grid["z"], dtype=np.float32))[None, None], persistent=False)
        self.mpp, self.half_extent = float(grid["mpp"]), float(grid["half_extent"])
        self._off: dict = {}
        self.to(device)

    def _apply(self, fn, *a, **kw):  # .half()/.bfloat16() on a parent must not quantise the grid: keep the f32 values
        z = self.z; super()._apply(fn, *a, **kw); self.z = z.to(self.z.device); return self

    def sample_height(self, xy: torch.Tensor) -> torch.Tensor:
        """xy (..., 2) -> (...,) bilinear map height, border-clamped."""
        g = xy.float() / self.half_extent  # normalised [-1, 1], float32; align_corners=False => fx = (x + half)/mpp - 0.5
        out = F.grid_sample(self.z.float(), g.reshape(1, -1, 1, 2), mode="bilinear", padding_mode="border", align_corners=False)
        return out.reshape(xy.shape[:-1])

    def forward(self, pose: torch.Tensor, k: int = 8, half_m: float = 6.0):
        """pose (B, 3) or (B, L, 3) -> heights (..., k, k) relative to the centre / 2 m, valid (...,) bool."""
        key = (int(k), float(half_m))
        if key not in self._off or self._off[key][0].device != self.z.device:  # (re)built after .to(device)
            with torch.inference_mode(False):  # cached offsets must be normal tensors (usable for backward later)
                du, dv = _offsets(k, half_m)
                self._off[key] = tuple(torch.as_tensor(a, dtype=torch.float32, device=self.z.device) for a in (du, dv))
        du, dv = self._off[key]
        pose = pose.float()  # sample points in float32 whatever the caller's dtype (fp16 poses would quantise them)
        x, y, yaw = pose[..., 0:1], pose[..., 1:2], pose[..., 2:3]
        c, s = torch.cos(yaw), torch.sin(yaw)
        px, py = x + du * c - dv * s, y + du * s + dv * c  # (..., k*k)
        h = self.sample_height(torch.stack([px, py], -1)) - self.sample_height(pose[..., :2])[..., None]
        valid = ((px.abs() < self.half_extent) & (py.abs() < self.half_extent)).all(-1)
        return (h / SCALE_M).reshape(*pose.shape[:-1], k, k), valid


def _selftest(grid_path: str, out: str, arena: str) -> None:  # numbers land in out/selftest.json
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20260921)
    grid = load_grid(grid_path); dev = "cuda" if torch.cuda.is_available() else "cpu"
    crop = EgoCrop(grid, dev); rep: dict = dict(grid=grid["path"], device=dev, torch=torch.__version__)
    rep["grid_sha256"] = hashlib.sha256(open(grid["path"], "rb").read()).hexdigest()
    # 0. sign conventions on synthetic ramps: z = x (columns) and z = y (rows); axis 0 forward, axis 1 left
    yy, xx = np.meshgrid(np.arange(grid["n"]), np.arange(grid["n"]), indexing="ij")
    for name, ramp, yaw, want in (("x_ramp_yaw0", xx, 0.0, "axis0_up"), ("y_ramp_yaw0", yy, 0.0, "axis1_up"),
                                  ("x_ramp_yaw90", xx, np.pi / 2, "axis1_down")):
        g2 = dict(grid, z=(-grid["half_extent"] + (ramp + 0.5) * grid["mpp"]).astype(np.float32))
        h = ego_crop_np(g2, np.array([3.0, -5.0, yaw]), 8, 6.0)[0]
        d0, d1 = float(h[-1, 0] - h[0, 0]), float(h[0, -1] - h[0, 0])  # in /2 m units; 12 m span -> 6
        rep[f"convention_{name}"] = dict(expect=want, d_axis0=d0, d_axis1=d1)
    # 1. height sampler vs the existing v2 corridor sampler (sensor_dataset_v2.sample)
    import sensor_dataset_v2 as sd
    sd.init_grid(os.path.dirname(grid["path"]))
    xy = rng.uniform(-42, 42, (20000, 2))
    rep["sample_vs_sensor_dataset_v2_max_abs_m"] = float(np.abs(sample_height(grid, xy) - sd.sample(sd.G["z"], xy[:, 0], xy[:, 1])).max())
    ht = crop.sample_height(torch.as_tensor(xy, dtype=torch.float32, device=dev)).cpu().numpy()
    rep["sample_torch_vs_numpy_max_abs_m"] = float(np.abs(ht - sample_height(grid, xy)).max())
    # 2. numpy vs torch crops on 10,000 poses incl. beyond the edges; (B,3) and (B,L,3) shapes
    pose = np.c_[rng.uniform(-46, 46, (10000, 2)), rng.uniform(-np.pi, np.pi, 10000)]
    for k, hm in ((8, 6.0), (16, 4.0)):
        hn, vn = ego_crop_np(grid, pose, k, hm)
        ht_, vt = crop(torch.as_tensor(pose, dtype=torch.float32, device=dev), k, hm)
        hl, vl = crop(torch.as_tensor(pose.reshape(100, 100, 3), dtype=torch.float32, device=dev), k, hm)
        rep[f"k{k}_half{hm:g}"] = dict(max_abs_diff=float(np.abs(hn - ht_.cpu().numpy()).max()),
                                       valid_frac=float(vn.mean()), valid_agree=bool((vn == vt.cpu().numpy()).all()),
                                       BL_shape_equal=bool(torch.equal(hl.reshape(-1, k, k), ht_) and torch.equal(vl.reshape(-1), vt)),
                                       height_abs_p99=float(np.quantile(np.abs(hn[vn]), 0.99)))
    # 3. differentiability: autograd vs central finite differences of sum(crop) w.r.t. (x, y, yaw), 200 interior poses
    p = torch.as_tensor(pose[np.abs(pose[:, :2]).max(1) < 30][:200], dtype=torch.float32, device=dev).requires_grad_(True)
    crop(p, 8, 6.0)[0].sum().backward(); ga = p.grad.detach().cpu().numpy(); gf = np.zeros_like(ga)
    for j, eps in enumerate((1e-3, 1e-3, 1e-4)):
        e = np.zeros(3); e[j] = eps
        gf[:, j] = ((ego_crop_np(grid, pose[np.abs(pose[:, :2]).max(1) < 30][:200] + e, 8, 6.0)[0].sum((1, 2))
                     - ego_crop_np(grid, pose[np.abs(pose[:, :2]).max(1) < 30][:200] - e, 8, 6.0)[0].sum((1, 2))) / (2 * eps))
    rep["grad_autograd_vs_fd"] = dict(median_abs_diff=[float(v) for v in np.median(np.abs(ga - gf), 0)],
                                      median_abs_grad=[float(v) for v in np.median(np.abs(gf), 0)], finite=bool(np.isfinite(ga).all()))
    # 3b. regressions: forward on cpu then .to(dev); first use of a size under inference_mode then backward; .half() parent
    c2 = EgoCrop(grid, "cpu"); c2(torch.as_tensor(pose[:8], dtype=torch.float32)); c2.to(dev)
    with torch.inference_mode(): c2(torch.as_tensor(pose[:8], dtype=torch.float32, device=dev), 16, 4.0)
    pg = torch.as_tensor(pose[:8], dtype=torch.float32, device=dev).requires_grad_(True); c2(pg, 16, 4.0)[0].sum().backward()
    hh = nn.Sequential(c2).half()[0](torch.as_tensor(pose[:64], dtype=torch.float16, device=dev), 8, 6.0)[0]
    rep["regress_to_device_inference_half"] = dict(grad_finite=bool(torch.isfinite(pg.grad).all()), z_dtype=str(c2.z.dtype),
        half_parent_fp16_pose_max_abs_diff=float(np.abs(hh.float().cpu().numpy() - ego_crop_np(grid, pose[:64].astype(np.float16), 8, 6.0)[0]).max()))
    # 4. reference: traverse_wp2_add_terrain.terrain_patch on TerrainMap (BMP) for 20 poses, k=8, +-6 m
    from nedm.traverse.terrain import TerrainMap
    from traverse_wp2_add_terrain import terrain_patch
    tm = TerrainMap.from_dir(__import__("pathlib").Path(arena)); p20 = pose[np.abs(pose[:, :2]).max(1) < 31][:20]
    ours = ego_crop_np(grid, p20, 8, 6.0)[0].reshape(20, -1); s = 511 / 512
    du, dv = _offsets(8, 6.0); px, py = _ego_points(p20[:, 0:1], p20[:, 1:2], p20[:, 2:3], du, dv)
    scaled = (tm.height(px * s, py * s) - tm.height(p20[:, 0] * s, p20[:, 1] * s)[:, None]) / SCALE_M
    for name, ref in (("raw", terrain_patch(tm, p20)), ("pose_scaled_511_512", terrain_patch(tm, p20 * [s, s, 1.0])),
                      ("points_scaled_511_512", scaled)):
        d = (ours - ref) * SCALE_M
        rep[f"ref_terrain_patch_{name}"] = dict(rmse_m=float(np.sqrt(np.mean(d ** 2))), max_abs_m=float(np.abs(d).max()))
    # 5. throughput on the GPU: B=4096, k=16, forward and forward+backward
    pb = torch.as_tensor(pose[:4096], dtype=torch.float32, device=dev)
    for name, fn in (("forward", lambda: crop(pb, 16, 4.0)), ("forward_backward", lambda: crop(pb.requires_grad_(True), 16, 4.0)[0].sum().backward())):
        for _ in range(10): fn()
        if dev == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter(); n_it = 100
        for _ in range(n_it): fn()
        if dev == "cuda": torch.cuda.synchronize()
        rep[f"throughput_B4096_k16_{name}_crops_per_s"] = float(4096 * n_it / (time.perf_counter() - t0))
    json.dump(rep, open(os.path.join(out, "selftest.json"), "w"), indent=1); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--grid", default="artifacts/traverse/crm_f104_v1/grids/arena_f104_50h_v1/grid.npz")
    ap.add_argument("--arena", default="assets/traverse/arena_f104_50h_v1")
    ap.add_argument("--out", default="artifacts/traverse/generalist_20260921/B_tracker/selftest/crop")
    a = ap.parse_args()
    if a.selftest:
        _selftest(a.grid, a.out, a.arena)
