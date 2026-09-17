"""v2 corridor sampler: the same 96 x 32 corridor as scripts/sensor_dataset.tensor10, read out of the
back-projected metric world grid (scripts/sensor_map_v2.py) instead of the flat-ground image lookup.

Channel layout is identical to sensor_dataset.X10 so the existing checkpoints can be fed unchanged:
  0 elev_rel  1 grade  2 cross  3 speed  4 valid  5 depth_rel  6 ray_sec1  7-9 R,G,B

Differences from v1, all of them consequences of the v2 grid (documented, not hidden):
  * height/range/sec/colour are read at the cell that CONTAINS the back-projected 3D point, so a corridor sample
    at world (x, y) gets the pixel that actually measured (x, y) rather than the pixel whose flat-ground
    projection is (x, y);
  * the grid is 0.15625 m over +-40 m for every channel, so channels 5-9 are no longer read at the 1024-px
    native resolution (v1 read 0-4 at 0.1868 m and 5-9 at 0.0934 m);
  * cells with no contributing pixel are invalid (grid cover == 0); outside +-40 m is invalid.
Everything else (station/lateral geometry, e0 / d0 reference, gradient clipping, fills) is copied from
sensor_dataset.tensor10 verbatim.
"""
import json
from pathlib import Path
import numpy as np

N_STATION, N_LATERAL, HALF_WIDTH_M = 96, 32, 6.0


class V2Map:
    def __init__(self, griddir):
        griddir = Path(griddir)
        m = json.load(open(griddir / 'grid.json'))
        g = np.load(griddir / 'grid.npz')
        self.mpp = float(m['mpp']); self.half = float(m['half_extent_m']); self.n = int(m['n'])
        self.cam_h = float(m['camera_height_m'])
        self.max_depth = float(m['camera']['max_depth_m'])
        self.z = g['z'].astype(np.float32)
        self.range_m = g['range_m'].astype(np.float32)
        self.sec = g['sec'].astype(np.float32)
        self.rgb = g['rgb'].astype(np.float32)          # (3, n, n)
        self.cover = g['cover']
        # grid.npz is indexed [row = y bin, col = x bin]; bin i spans [-half + i*mpp, -half + (i+1)*mpp)
        self.ok = (self.cover > 0) & np.isfinite(self.z) & np.isfinite(self.range_m)

    def _idx(self, x, y):
        col = (np.asarray(x, float) + self.half) / self.mpp - 0.5
        row = (np.asarray(y, float) + self.half) / self.mpp - 0.5
        return row, col

    def sample(self, x, y):
        """Bilinear read of every channel plus a validity mask (all four contributing cells must be covered)."""
        row, col = self._idx(x, y)
        n = self.n
        r0 = np.clip(np.floor(row).astype(int), 0, n - 2); c0 = np.clip(np.floor(col).astype(int), 0, n - 2)
        fr = np.clip(row - r0, 0, 1); fc = np.clip(col - c0, 0, 1)
        inside = (row >= -0.5) & (row <= n - 0.5) & (col >= -0.5) & (col <= n - 0.5)
        okq = (self.ok[r0, c0] & self.ok[r0, c0 + 1] & self.ok[r0 + 1, c0] & self.ok[r0 + 1, c0 + 1]) & inside

        def bl(img):
            p00 = img[..., r0, c0]; p01 = img[..., r0, c0 + 1]
            p10 = img[..., r0 + 1, c0]; p11 = img[..., r0 + 1, c0 + 1]
            f_r = fr if p00.ndim == fr.ndim else fr[None]
            f_c = fc if p00.ndim == fc.ndim else fc[None]
            return p00 * (1 - f_r) * (1 - f_c) + p01 * (1 - f_r) * f_c + p10 * f_r * (1 - f_c) + p11 * f_r * f_c

        return dict(z=bl(self.z), range_m=bl(self.range_m), sec=bl(self.sec), rgb=bl(self.rgb), valid=okq)


def corridor_xy(wp, st):
    """Identical to sensor_dataset.corridor."""
    wp = np.asarray(wp, float)
    s = np.asarray(st, float)
    if s.size != len(wp) or not np.all(np.diff(s) > 0):
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    grid = np.linspace(s[0], s[-1], N_STATION)
    pts = np.stack([np.interp(grid, s, wp[:, 0]), np.interp(grid, s, wp[:, 1])], 1)
    d = np.gradient(pts, axis=0); tn = np.linalg.norm(d, axis=1, keepdims=True); tn[tn < 1e-9] = 1e-9
    tang = d / tn; norm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    off = np.linspace(-HALF_WIDTH_M, HALF_WIDTH_M, N_LATERAL)
    gx = pts[:, 0:1] + norm[:, 0:1] * off[None, :]; gy = pts[:, 1:2] + norm[:, 1:2] * off[None, :]
    return gx, gy, grid


def tensor10_v2(vmap, wp, sp, st):
    wp = np.asarray(wp, float); sp = np.asarray(sp, float)
    gx, gy, grid = corridor_xy(wp, st)
    s = vmap.sample(gx, gy)
    valid = s['valid']
    elev = np.where(valid, s['z'], np.nan)
    e0 = elev[0, N_LATERAL // 2] if np.isfinite(elev[0, N_LATERAL // 2]) else np.nanmean(elev[0])
    if not np.isfinite(e0):
        e0 = 0.0
    fill = np.where(np.isfinite(elev), elev, e0)
    ds = max(float(grid[-1] - grid[0]) / (N_STATION - 1), 1e-3); dl = 2 * HALF_WIDTH_M / (N_LATERAL - 1)
    ga = np.clip(np.gradient(fill, ds, axis=0), -2, 2); gc = np.clip(np.gradient(fill, dl, axis=1), -2, 2)
    s_ref = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    v = np.interp(np.linspace(0, s_ref[-1], N_STATION), s_ref, sp)
    depth = np.where(valid, s['range_m'], np.nan)
    dvalid = valid & np.isfinite(depth) & (np.nan_to_num(depth, nan=0.0) < vmap.max_depth - 1e-3)
    if dvalid[0, N_LATERAL // 2]:
        d0 = depth[0, N_LATERAL // 2]
    elif dvalid[0].any():
        d0 = np.nanmean(np.where(dvalid[0], depth[0], np.nan))
    else:
        d0 = vmap.cam_h
    depth_rel = np.where(dvalid, np.nan_to_num(depth, nan=0.0) - d0, 0.0)
    sec1 = np.where(valid, s['sec'] - 1.0, 0.0)
    rgb = np.where(valid[None], s['rgb'], 0.0)
    X = np.stack([fill - e0, ga, gc, np.repeat(v[:, None], N_LATERAL, 1), valid.astype(float),
                  depth_rel, sec1, rgb[0], rgb[1], rgb[2]])
    return X.astype(np.float32), float(grid[-1] - grid[0])
