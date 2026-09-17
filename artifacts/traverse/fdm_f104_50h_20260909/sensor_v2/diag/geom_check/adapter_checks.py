"""Task A3: defect checks on scripts/sensor_map_v2.py.

Orientation/sign (flip/transpose/shift variants), axial-vs-ray range convention, intrinsics variants,
cell aggregation, edge behaviour, empty cells. Read-only; TerrainMap is an evaluation reference only.
  /home/harry/miniconda3/envs/nedm/bin/python <this file>
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path("/home/harry/NeDM-traverse_mppi")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from nedm.traverse.terrain import TerrainMap

GRIDS = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/grids"
MAPS = ROOT / "artifacts/traverse/fdm_f104_50h_20260909/sensor_v1/maps"
ARENAS = ["arena_f104_50h_v1", "arena_g203", "arena_g216", "arena_g217", "arena_g228", "arena_g231"]
MPP, HALF, N = 80.0 / 512, 40.0, 512


def rasterise(depth, cam, *, use_axial_xy=True, use_axial_z=True, f=None, cx=None, cy=None,
              row0_plus_y=True, agg='mean'):
    """Generalised version of sensor_map_v2.grid_from_capture so conventions can be swapped one at a time."""
    h, w = depth.shape
    H = float(cam['cam_height_m'])
    f = (w / 2) / math.tan(float(cam['hfov_rad']) / 2) if f is None else f
    cx = (w - 1) / 2 if cx is None else cx
    cy = (h - 1) / 2 if cy is None else cy
    v, u = np.mgrid[0:h, 0:w]
    ray_x = (u - cx) / f
    ray_y = (-(v - cy) if row0_plus_y else (v - cy)) / f
    sec = np.sqrt(1 + ray_x ** 2 + ray_y ** 2)
    ok = np.isfinite(depth) & (depth > 0) & (depth < float(cam['max_depth_m']) - 1e-6)
    axial = np.where(ok, depth / sec, np.nan)
    rxy = axial if use_axial_xy else np.where(ok, depth, np.nan)
    x = ray_x * rxy; y = ray_y * rxy
    z = H - (axial if use_axial_z else np.where(ok, depth, np.nan))
    inside = ok & (np.abs(x) < HALF) & (np.abs(y) < HALF)
    xi = np.clip(((x[inside] + HALF) / MPP).astype(int), 0, N - 1)
    yi = np.clip(((y[inside] + HALF) / MPP).astype(int), 0, N - 1)
    flat = yi * N + xi
    cnt = np.bincount(flat, minlength=N * N).astype(np.float64)
    with np.errstate(invalid='ignore', divide='ignore'):
        if agg == 'mean':
            zg = (np.bincount(flat, weights=z[inside], minlength=N * N) / cnt).reshape(N, N)
        elif agg == 'max':
            zg = np.full(N * N, -np.inf); np.maximum.at(zg, flat, z[inside])
            zg = np.where(cnt > 0, zg, np.nan).reshape(N, N)
        elif agg == 'min':
            zg = np.full(N * N, np.inf); np.minimum.at(zg, flat, z[inside])
            zg = np.where(cnt > 0, zg, np.nan).reshape(N, N)
        elif agg == 'first':
            zg = np.full(N * N, np.nan); zg[flat[::-1]] = z[inside][::-1]
            zg = zg.reshape(N, N)
    return zg, cnt.reshape(N, N)


def mae(a, b, m):
    return float(np.abs(a[m] - b[m]).mean())


def main():
    out = {}
    c = -HALF + (np.arange(N) + 0.5) * MPP
    X, Y = np.meshgrid(c, c)
    for arena in ARENAS:
        cam = json.load(open(MAPS / arena / "observation.json"))['camera']
        with np.load(MAPS / arena / "observation.npz") as o:
            depth = o['depth_m'].astype(np.float64)
        with np.load(GRIDS / arena / "grid.npz") as g:
            z = g['z'].astype(np.float64); cover = g['cover'].astype(np.float64)
            rng = g['range_m'].astype(np.float64); sec = g['sec'].astype(np.float64); rgbg = g['rgb']
        terrain = TerrainMap.from_dir(ROOT / "assets/traverse" / arena)
        ref = terrain.height_grid                      # h[iy, ix], row 0 at -y -- same indexing as the v2 grid
        # cell centres coincide with BMP pixel centres to float precision (the 2.6e-5 m residue is the
        # pixels-1.001 clamp inside TerrainMap._fractional_index on the last row/column)
        assert np.abs(ref - terrain.height(X, Y)).max() < 1e-4
        m = (cover > 0) & np.isfinite(z)
        a = {}

        # --- 0. shipped grid reproduces from source, and self-consistency ---
        zr, cr = rasterise(depth, cam)
        a['reproduces_shipped_grid'] = dict(
            max_abs_z_diff=float(np.nanmax(np.abs(zr - z))), cover_identical=bool((cr == cover).all()))

        # --- 1. orientation / sign: shipped grid vs deliberately transformed references ---
        variants = {'identity': ref, 'flipud(y-mirror)': np.flipud(ref), 'fliplr(x-mirror)': np.fliplr(ref),
                    'transpose': ref.T, 'rot90': np.rot90(ref), 'rot180': np.rot90(ref, 2),
                    'rot270': np.rot90(ref, 3), 'flip_both': np.flipud(np.fliplr(ref))}
        a['orientation'] = {k: dict(mae=mae(z, vv, m), corr=float(np.corrcoef(z[m], vv[m])[0, 1]))
                            for k, vv in variants.items()}
        a['terrain_self_asymmetry'] = {k: mae(ref, vv, m) for k, vv in variants.items() if k != 'identity'}
        # sign of ray_y flipped inside the adapter (row 0 = -y)
        zf, _ = rasterise(depth, cam, row0_plus_y=False)
        mf = np.isfinite(zf) & m
        a['row0_minus_y_adapter_variant'] = dict(mae=mae(zf, ref, mf), corr=float(np.corrcoef(zf[mf], ref[mf])[0, 1]))
        # half-cell / one-cell registration
        a['shift_cells'] = {}
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            sm = m[max(dy, 0):N + min(dy, 0), max(dx, 0):N + min(dx, 0)]
            zs = z[max(dy, 0):N + min(dy, 0), max(dx, 0):N + min(dx, 0)]
            rs = ref[max(-dy, 0):N + min(-dy, 0), max(-dx, 0):N + min(-dx, 0)]
            a['shift_cells'][f'dx{dx}_dy{dy}'] = float(np.abs(zs[sm] - rs[sm]).mean())

        # --- 2. axial vs ray range convention ---
        for name, kw in [('xy_from_ray_range_instead_of_axial', dict(use_axial_xy=False)),
                         ('z_from_ray_range_instead_of_axial', dict(use_axial_z=False)),
                         ('both_from_ray_range', dict(use_axial_xy=False, use_axial_z=False))]:
            zz, cc = rasterise(depth, cam, **kw)
            mm = (cc > 0) & np.isfinite(zz)
            a[name] = dict(mae=mae(zz, ref, mm), empty_cells=int((cc == 0).sum()))

        # --- 3. intrinsics variants (is the shipped f / principal point the best fit?) ---
        w = depth.shape[1]; f0 = (w / 2) / math.tan(float(cam['hfov_rad']) / 2)
        a['intrinsics'] = {}
        for name, kw in [('f=(w/2)/tan  [shipped]', {}), ('f=((w-1)/2)/tan', dict(f=((w - 1) / 2) / math.tan(cam['hfov_rad'] / 2))),
                         ('f*1.001', dict(f=f0 * 1.001)), ('f*0.999', dict(f=f0 * 0.999)),
                         ('principal point w/2', dict(cx=w / 2, cy=w / 2))]:
            zz, cc = rasterise(depth, cam, **kw)
            mm = (cc > 0) & np.isfinite(zz)
            a['intrinsics'][name] = dict(mae=mae(zz, ref, mm), empty_cells=int((cc == 0).sum()))

        # --- 4. cell aggregation ---
        a['aggregation'] = {}
        for agg in ('mean', 'max', 'min', 'first'):
            zz, cc = rasterise(depth, cam, agg=agg)
            mm = (cc > 0) & np.isfinite(zz)
            a['aggregation'][agg] = dict(mae=mae(zz, ref, mm), p95=float(np.quantile(np.abs(zz[mm] - ref[mm]), .95)))
        h_, _ = np.histogram(cover, bins=np.arange(-0.5, 9.5))
        a['cover_histogram_0..8'] = [int(v) for v in h_]
        a['empty_cells'] = int((cover == 0).sum())
        # mean-of-ratios vs ratio-of-means: does z = H - range/sec hold for the STORED cell means?
        zrec = float(cam['cam_height_m']) - rng / sec
        a['stored_range_over_sec_recovers_stored_z'] = dict(
            mae=float(np.abs(zrec[m] - z[m]).mean()), max=float(np.abs(zrec[m] - z[m]).max()))
        a['stored_range_over_sec_vs_terrain'] = dict(mae=mae(zrec, ref, m))
        # error vs number of contributing pixels
        a['err_by_cover'] = {int(k): dict(cells=int((cover[m] == k).sum()),
                                          mae=float(np.abs(z[m] - ref[m])[cover[m] == k].mean()))
                             for k in np.unique(cover[m]) if (cover[m] == k).sum() > 50}

        # --- 5. edge behaviour ---
        edge = np.minimum(np.minimum(np.arange(N)[None, :], (N - 1 - np.arange(N))[None, :]),
                          np.minimum(np.arange(N)[:, None], (N - 1 - np.arange(N))[:, None]))
        a['err_by_edge_ring'] = {}
        for lo, hi, lab in [(0, 1, 'outermost row/col'), (1, 4, 'cells 1-3 from edge'),
                            (4, 16, 'cells 4-15'), (16, 64, 'cells 16-63'), (64, 999, 'interior')]:
            mm = m & (edge >= lo) & (edge < hi)
            a['err_by_edge_ring'][lab] = dict(cells=int(mm.sum()), mae=mae(z, ref, mm),
                                              max=float(np.abs(z[mm] - ref[mm]).max()))
        # points dropped by the strict |x|<HALF test / clipped indices
        _, cnt_all = rasterise(depth, cam)
        a['pixels_inside_window'] = int(cnt_all.sum())
        a['pixels_total'] = int(depth.size)
        a['bmp_quantization_step_m'] = float(terrain.meta['quantization_step_m'])
        a['rgb_finite_fraction'] = float(np.isfinite(rgbg).mean())
        out[arena] = a
        print(f"{arena}: identity MAE {a['orientation']['identity']['mae']:.4f} | "
              f"next-best variant { min((k for k in a['orientation'] if k!='identity'), key=lambda k: a['orientation'][k]['mae']) } "
              f"{min(a['orientation'][k]['mae'] for k in a['orientation'] if k!='identity'):.4f} | "
              f"row0=-y {a['row0_minus_y_adapter_variant']['mae']:.4f} | "
              f"ray-range xy {a['xy_from_ray_range_instead_of_axial']['mae']:.4f} "
              f"z {a['z_from_ray_range_instead_of_axial']['mae']:.4f}", flush=True)

    p = GRIDS.parent / "diag/geom_check/adapter_checks.json"
    json.dump(out, open(p, "w"), indent=1)
    print("wrote", p)


if __name__ == "__main__":
    main()
