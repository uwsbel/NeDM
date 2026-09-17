"""Corridor extraction from a vehicle-INCLUDED overhead frame, with a footprint-based exclusion zone.

Rules (from the pilot brief):
  * the 96 x 32 corridor coordinates and the full route are preserved - samples inside the exclusion zone are marked
    INVALID, never dropped, shortened or rescaled;
  * bilinear interpolation is accounted for: a sample is valid only if all four grid cells it reads are covered and
    outside the exclusion zone;
  * the relative-height and relative-range references come from the first station whose valid samples lie OUTSIDE the
    exclusion zone - never from simulator heights and never from a vehicle-free image;
  * nothing fills the hidden ground.
"""
import numpy as np
import sensor_dataset_v2 as V2

HALF_LENGTH_M, HALF_WIDTH_M_VEH = 2.6, 1.3      # the footprint the route validator already uses (MPPIConfig)
MIN_VALID_PER_STATION = 4


def exclusion_mask(pose, margin_m, n=None, mpp=None, half=None):
    """Grid-cell mask of the vehicle footprint (+margin) at the measured pose, as a rotated rectangle."""
    n = n or V2.G['n']; mpp = mpp or V2.G['mpp']; half = half or V2.G['half']
    xs = -half + (np.arange(n) + 0.5) * mpp
    X, Y = np.meshgrid(xs, xs)                                  # X varies along columns, Y along rows
    dx, dy = X - float(pose[0]), Y - float(pose[1])
    c, s = np.cos(float(pose[2])), np.sin(float(pose[2]))
    along = dx * c + dy * s; across = -dx * s + dy * c
    return (np.abs(along) <= HALF_LENGTH_M + margin_m) & (np.abs(across) <= HALF_WIDTH_M_VEH + margin_m)


def tensor12_excluded(wp, sp, st, exclude=None):
    """V2.tensor12 with an exclusion zone; returns (X, route_len, info)."""
    wp = np.asarray(wp, float); sp = np.asarray(sp, float)
    gx, gy, grid = V2.corridor_points(wp, st)
    cover = V2.G['cover'].astype(np.float32)
    if exclude is not None:
        cover = np.where(exclude, 0.0, cover)                  # excluded cells behave exactly like uncovered cells
    inside = (np.abs(gx) < V2.G['half'] - 1e-6) & (np.abs(gy) < V2.G['half'] - 1e-6)
    cov = V2.sample(cover, gx, gy)
    zs = V2.sample(V2.G['z'], gx, gy)
    valid = inside & (cov > 0.999) & np.isfinite(zs)           # all four contributing cells covered and not excluded
    n_lat = valid.shape[1]; c = n_lat // 2
    ref_station = None
    for i in range(valid.shape[0]):
        if valid[i].sum() >= MIN_VALID_PER_STATION:
            ref_station = i; break
    if ref_station is None:
        z0 = float(np.nanmean(np.where(valid, zs, np.nan))) if valid.any() else 0.0
    else:
        z0 = float(np.nanmean(np.where(valid[ref_station], zs[ref_station], np.nan)))
    zf = np.where(valid, zs, z0)                                # excluded samples carry no terrain information
    ds = max(float(grid[-1] - grid[0]) / (valid.shape[0] - 1), 1e-3); dl = 2 * V2.HALF_WIDTH_M / (n_lat - 1)
    ga = np.clip(np.gradient(zf, ds, axis=0), -2, 2); gc = np.clip(np.gradient(zf, dl, axis=1), -2, 2)
    rng = V2.sample(V2.G['range_m'], gx, gy); sec = V2.sample(V2.G['sec'], gx, gy)
    if ref_station is None:
        r0 = float(np.nanmean(np.where(valid, rng, np.nan))) if valid.any() else V2.G['cam_h']
    else:
        r0 = float(np.nanmean(np.where(valid[ref_station], rng[ref_station], np.nan)))
    rng = np.where(valid, rng, V2.G['cam_h']); sec = np.where(valid, sec, 1.0)
    rgb = V2.sample(V2.G['rgb'], gx, gy)
    s_ref = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(wp, axis=0), axis=1))]
    v = np.interp(np.linspace(0, s_ref[-1], valid.shape[0]), s_ref, sp)
    X = np.stack([zf - z0, ga, gc, np.repeat(v[:, None], n_lat, 1), valid.astype(float),
                  rng, rng - r0, sec - 1.0, rgb[0], rgb[1], rgb[2], np.clip(cov, 0, 8) / 8.0]).astype(np.float32)
    info = dict(invalid_fraction=float(1 - valid.mean()), reference_station=ref_station,
                invalid_stations=int((valid.sum(1) == 0).sum()), z0=z0, r0=r0,
                first_valid_station=ref_station, station_invalid_counts=(n_lat - valid.sum(1)).tolist())
    return X, float(grid[-1] - grid[0]), info
