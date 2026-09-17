"""Independent re-implementation + adversarial check of the v2 depth->world adapter (Task A verification).

Nothing is imported from scripts/sensor_map_v2.py: the back-projection and the rasteriser are written from
scratch here.  TerrainMap (authored BMP heights) is used ONLY as an evaluation reference.

  /home/harry/miniconda3/envs/nedm/bin/python <this file>
"""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("/home/harry/NeDM-traverse_mppi")
sys.path.insert(0, str(ROOT / "src"))
from nedm.traverse.terrain import TerrainMap

MAPS = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v1/maps"
GRIDS = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/grids"
OUT = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/diag/verify_geometry"
ARENAS = ["arena_f104_50h_v1", "arena_g203", "arena_g216", "arena_g217", "arena_g228", "arena_g231"]


def load_capture(arena):
    obs = json.load(open(MAPS / arena / "observation.json"))
    with np.load(MAPS / arena / "observation.npz") as o:
        depth = o["depth_m"].astype(np.float64)
        rgb = o["rgb"].astype(np.float64) / 255.0
        rgbd = o["rgbd"].astype(np.float64)
    return obs, depth, rgb, rgbd


def backproject(depth, cam, f_override=None, cx_override=None, cy_override=None, ray_y_sign=-1.0):
    """My own back-projection.  Returns x, y, z, sec, valid (all HxW, float64)."""
    h, w = depth.shape
    H = float(cam["cam_height_m"])
    f = (w / 2.0) / math.tan(float(cam["hfov_rad"]) / 2.0) if f_override is None else f_override
    cx = (w - 1) / 2.0 if cx_override is None else cx_override
    cy = (h - 1) / 2.0 if cy_override is None else cy_override
    col = np.arange(w, dtype=np.float64)[None, :]
    row = np.arange(h, dtype=np.float64)[:, None]
    rx = (col - cx) / f + 0.0 * row
    ry = ray_y_sign * (row - cy) / f + 0.0 * col
    sec = np.sqrt(1.0 + rx * rx + ry * ry)
    valid = np.isfinite(depth) & (depth > 0) & (depth < float(cam["max_depth_m"]) - 1e-6)
    axial = np.where(valid, depth / sec, np.nan)
    return rx * axial, ry * axial, H - axial, sec, valid


def rasterise(x, y, z, valid, extra=None, half=40.0, n=512, how="mean"):
    """Independent rasteriser (uses np.add.at / sorting rather than bincount-with-weights)."""
    mpp = 2 * half / n
    ins = valid & (np.abs(x) < half) & (np.abs(y) < half)
    xi = np.floor((x[ins] + half) / mpp).astype(np.int64)
    yi = np.floor((y[ins] + half) / mpp).astype(np.int64)
    xi = np.clip(xi, 0, n - 1); yi = np.clip(yi, 0, n - 1)
    idx = yi * n + xi
    cnt = np.zeros(n * n); np.add.at(cnt, idx, 1.0)
    out = {}
    src = {"z": z[ins]}
    if extra:
        src.update({k: v[ins] for k, v in extra.items()})
    for k, vals in src.items():
        if how == "mean":
            s = np.zeros(n * n); np.add.at(s, idx, vals)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[k] = (s / cnt).reshape(n, n)
        elif how in ("min", "max"):
            fill = np.inf if how == "min" else -np.inf
            a = np.full(n * n, fill)
            (np.minimum.at if how == "min" else np.maximum.at)(a, idx, vals)
            a[~np.isfinite(a)] = np.nan
            out[k] = a.reshape(n, n)
        elif how == "last":
            a = np.full(n * n, np.nan); a[idx] = vals
            out[k] = a.reshape(n, n)
    out["cover"] = cnt.reshape(n, n)
    out["n_inside"] = int(ins.sum())
    return out


def stats(e):
    e = np.abs(np.asarray(e, float))
    return dict(mae=float(e.mean()), p95=float(np.quantile(e, .95)), p99=float(np.quantile(e, .99)),
                max=float(e.max()), rmse=float(np.sqrt((e ** 2).mean())), n=int(e.size))


def v1_flat_elev(rgbd, cam, X, Y):
    """v1's flat-ground sampling of the 512-px encoded elevation channel, re-implemented from the formula
    in scripts/sensor_dataset.py:init_map/tensor10 (ground = 2*H*tan(hfov/2); mpp = ground/npx; ctr=(npx-1)/2)."""
    npx = rgbd.shape[1]
    ground = 2 * float(cam["cam_height_m"]) * math.tan(float(cam["hfov_rad"]) / 2)
    mpp = ground / npx
    ctr = (npx - 1) / 2.0
    row = ctr - Y / mpp
    col = ctr + X / mpp
    r0 = np.clip(np.floor(row).astype(int), 0, npx - 2)
    c0 = np.clip(np.floor(col).astype(int), 0, npx - 2)
    fr = np.clip(row - r0, 0, 1); fc = np.clip(col - c0, 0, 1)
    A = rgbd[3]
    p00, p01, p10, p11 = A[r0, c0], A[r0, c0 + 1], A[r0 + 1, c0], A[r0 + 1, c0 + 1]
    v00, v01, v10, v11 = (rgbd[3][r0, c0], rgbd[3][r0, c0 + 1], rgbd[3][r0 + 1, c0], rgbd[3][r0 + 1, c0 + 1])
    valid = np.minimum(np.minimum(v00, v01), np.minimum(v10, v11)) > -1.999
    patch = p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc + p10 * fr * (1 - fc) + p11 * fr * fc
    return patch * float(cam["elevation_scale_m"]), valid


def main():
    res = {}
    for arena in ARENAS:
        obs, depth, rgb, rgbd = load_capture(arena)
        cam = obs["camera"]
        H = float(cam["cam_height_m"])
        x, y, z, sec, valid = backproject(depth, cam)
        mine = rasterise(x, y, z, valid, extra={"range_m": depth, "sec": sec})

        with np.load(GRIDS / arena / "grid.npz") as g:
            sz = g["z"].astype(np.float64); scov = g["cover"].astype(np.float64)
            srng = g["range_m"].astype(np.float64); ssec = g["sec"].astype(np.float64)
        meta = json.load(open(GRIDS / arena / "grid.json"))
        n, half, mpp = meta["n"], meta["half_extent_m"], meta["mpp"]

        # terrain reference at cell centres (row index increases with +y, same as TerrainMap.height_grid)
        c = -half + (np.arange(n) + 0.5) * mpp
        X, Y = np.meshgrid(c, c)
        tm = TerrainMap.from_dir(ROOT / "assets/traverse" / arena)
        ref = tm.height(X, Y)
        slope = tm.slope(X, Y)
        # cell centres coincide with BMP pixel centres?
        ref_direct = tm.height_grid
        centre_match = float(np.abs(ref - ref_direct).max())

        e1, v1valid = v1_flat_elev(rgbd, cam, X, Y)

        err2 = sz - ref
        err1 = e1 - ref
        ok_all = np.isfinite(sz)
        ok_v1 = ok_all & v1valid

        # quantization floor of the 8-bit reference
        am = json.load(open(ROOT / "assets/traverse" / arena / "arena_meta.json"))
        step = float(am["quantization_step_m"])

        r = np.hypot(X, Y)
        def binned(err, key, edges):
            o = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (key >= lo) & (key < hi) & ok_all
                if m.sum() < 50: continue
                o.append([float(lo), float(hi), int(m.sum()), float(np.abs(err[m]).mean())])
            return o

        # orientation variants (adversarial)
        variants = {"identity": sz, "flipud": np.flipud(sz), "fliplr": np.fliplr(sz),
                    "transpose": sz.T, "rot90": np.rot90(sz), "rot180": np.rot90(sz, 2),
                    "rot270": np.rot90(sz, 3)}
        orient = {}
        for k, v in variants.items():
            m = np.isfinite(v)
            orient[k] = dict(mae=float(np.abs(v - ref)[m].mean()),
                             corr=float(np.corrcoef(v[m], ref[m])[0, 1]))
        # how asymmetric is the terrain itself (power of the flip test)
        self_asym = {k: float(np.abs(f(ref) - ref).mean())
                     for k, f in dict(flipud=np.flipud, fliplr=np.fliplr, transpose=lambda a: a.T,
                                      rot90=np.rot90, rot180=lambda a: np.rot90(a, 2)).items()}

        # ray_y sign flipped adapter (row 0 = -y)
        xs, ys, zs, _, vs = backproject(depth, cam, ray_y_sign=+1.0)
        gs = rasterise(xs, ys, zs, vs)["z"]
        ms = np.isfinite(gs)
        flip_adapter = dict(mae=float(np.abs(gs - ref)[ms].mean()), corr=float(np.corrcoef(gs[ms], ref[ms])[0, 1]))

        # intrinsics variants
        w = depth.shape[1]
        f0 = (w / 2) / math.tan(float(cam["hfov_rad"]) / 2)
        intr = {}
        for name, (ff, cc) in {
            "shipped_f_w2_c_wm1_2": (f0, (w - 1) / 2.0),
            "f_wm1_2": (((w - 1) / 2) / math.tan(float(cam["hfov_rad"]) / 2), (w - 1) / 2.0),
            "f_times_0.999": (f0 * 0.999, (w - 1) / 2.0),
            "f_times_1.001": (f0 * 1.001, (w - 1) / 2.0),
            "pp_at_w_over_2": (f0, w / 2.0),
        }.items():
            xx, yy, zz, _, vv = backproject(depth, cam, f_override=ff, cx_override=cc, cy_override=cc)
            gg = rasterise(xx, yy, zz, vv)["z"]
            mm = np.isfinite(gg)
            intr[name] = float(np.abs(gg - ref)[mm].mean())

        # range convention variants
        conv = {}
        hh, ww = depth.shape
        col = np.arange(ww, dtype=float)[None, :]; row = np.arange(hh, dtype=float)[:, None]
        rx = (col - (ww - 1) / 2) / f0 + 0 * row; ry = -(row - (hh - 1) / 2) / f0 + 0 * col
        secf = np.sqrt(1 + rx ** 2 + ry ** 2)
        vld = np.isfinite(depth) & (depth > 0) & (depth < float(cam["max_depth_m"]) - 1e-6)
        ax = np.where(vld, depth / secf, np.nan)
        for name, (px, py, pz) in {
            "xy_from_ray_range": (rx * depth, ry * depth, H - ax),
            "z_from_ray_range": (rx * ax, ry * ax, H - depth),
            "both_ray_range": (rx * depth, ry * depth, H - depth),
        }.items():
            gg = rasterise(px, py, pz, vld)["z"]
            mm = np.isfinite(gg)
            conv[name] = float(np.abs(gg - ref)[mm].mean())

        # aggregation variants
        agg = {}
        for how in ("mean", "min", "max", "last"):
            gg = rasterise(x, y, z, valid, how=how)["z"]
            mm = np.isfinite(gg)
            agg[how] = float(np.abs(gg - ref)[mm].mean())

        # registration sweep (shift the world->grid mapping)
        reg = {}
        for d in (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0):
            gg = rasterise(x + d * mpp, y, z, valid)["z"]
            mm = np.isfinite(gg)
            reg[f"dx={d:+.2f}"] = float(np.abs(gg - ref)[mm].mean())
            gg = rasterise(x, y + d * mpp, z, valid)["z"]
            mm = np.isfinite(gg)
            reg[f"dy={d:+.2f}"] = float(np.abs(gg - ref)[mm].mean())

        # edge rings
        ii, jj = np.meshgrid(np.arange(n), np.arange(n))
        ring = np.minimum(np.minimum(ii, n - 1 - ii), np.minimum(jj, n - 1 - jj))
        edge = {}
        for lo, hi, name in [(0, 1, "0"), (1, 4, "1-3"), (4, 16, "4-15"), (16, 64, "16-63"), (64, n, "64+")]:
            m = (ring >= lo) & (ring < hi) & ok_all
            edge[name] = [int(m.sum()), float(np.abs(err2[m]).mean())]

        # cover histogram, per-cover error
        covh = {str(k): int((scov == k).sum()) for k in range(0, 9)}
        err_by_cover = {str(k): [int(((scov == k) & ok_all).sum()),
                                 float(np.abs(err2[(scov == k) & ok_all]).mean()) if ((scov == k) & ok_all).sum() else None]
                        for k in range(1, 5)}

        # pixel accounting
        n_at_max = int((depth >= float(cam["max_depth_m"]) - 1e-6).sum())
        n_pixels = depth.size

        res[arena] = dict(
            reproduce_shipped=dict(
                max_abs_dz=float(np.nanmax(np.abs(mine["z"] - sz))),
                cover_identical=bool(np.array_equal(mine["cover"], scov)),
                max_abs_drange=float(np.nanmax(np.abs(mine["range_m"] - srng))),
                max_abs_dsec=float(np.nanmax(np.abs(mine["sec"] - ssec))),
            ),
            grid_centre_matches_bmp_pixel_centre_max_abs_m=centre_match,
            cells_total=int(n * n), cells_finite=int(ok_all.sum()), cells_empty=int((scov == 0).sum()),
            cells_v1valid=int(ok_v1.sum()),
            v2_all_cells=stats(err2[ok_all]), v2_bias=float(err2[ok_all].mean()),
            v2_on_v1valid=stats(err2[ok_v1]),
            v1_on_v1valid=stats(err1[ok_v1]), v1_bias=float(err1[ok_v1].mean()),
            v2_corr=float(np.corrcoef(sz[ok_all], ref[ok_all])[0, 1]),
            v1_corr=float(np.corrcoef(e1[ok_v1], ref[ok_v1])[0, 1]),
            terrain_std=float(ref.std()),
            quant_step_m=step, ideal_quant_mae=step / 4.0,
            v2_mae_over_floor=float(stats(err2[ok_all])["mae"] / (step / 4.0)),
            by_slope_tan=binned(err2, slope, [0, .05, .10, .15, .20, .25, .30, .40, 10.]),
            by_slope_tan_v1=binned(err1, slope, [0, .05, .10, .15, .20, .25, .30, .40, 10.]),
            by_radius_m=binned(err2, r, [0, 5, 10, 15, 20, 25, 30, 35, 40, 60]),
            by_radius_m_v1=binned(err1, r, [0, 5, 10, 15, 20, 25, 30, 35, 40, 60]),
            orientation=orient, terrain_self_asymmetry=self_asym, row0_minus_y_adapter=flip_adapter,
            intrinsics=intr, range_convention=conv, aggregation=agg, registration=reg,
            err_by_edge_ring=edge, cover_hist=covh, err_by_cover=err_by_cover,
            pixels_inside_window=mine["n_inside"], pixels_total=n_pixels, pixels_at_max_depth=n_at_max,
            pixels_not_at_max=n_pixels - n_at_max,
            stored_z_vs_range_over_sec=stats((H - srng / ssec - sz)[ok_all]),
        )
        s = res[arena]
        print(f"{arena}: v2 MAE {s['v2_all_cells']['mae']:.5f} (floor {s['ideal_quant_mae']:.5f}, "
              f"x{s['v2_mae_over_floor']:.2f}) | v1 MAE {s['v1_on_v1valid']['mae']:.5f} | "
              f"repro dz {s['reproduce_shipped']['max_abs_dz']:.2e} cover_ok {s['reproduce_shipped']['cover_identical']} | "
              f"empty {s['cells_empty']}", flush=True)

    json.dump(res, open(OUT / "v_taskA.json", "w"), indent=1)
    print("wrote", OUT / "v_taskA.json")


if __name__ == "__main__":
    main()
