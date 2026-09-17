"""Is the fixed OptiX depth (local RTX 5090 build) geometrically the same as the AMD lavapipe (Vulkan-RT) depth?

Compares one 1024x1024 overhead depth frame rendered locally by the benchmark harness (frame0 after the 0.8 s settle)
against the stored AMD frame of the same case, pixel for pixel and after back-projection into the planner's metric
grid (scripts/sensor_map_v2.grid_from_arrays), and both against the arena heightmap (TerrainMap, with the 511/512 radial
correction from Chrono's RigidTerrain frame). Also: a fitted image-space scale/shift between the two depth images (the
old OptiX FOV bug would show as a radial scale of ~0.83) and RGB-vs-depth registration in each frame that has RGB.

  PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 compare_depth_vs_amd.py \
      --optix optix5090_depth_1024_frame0.npz [--optix-rgbd optix5090_rgbd_1024_frame0.npz] --out compare.json
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
from scipy import ndimage

REPO = Path('/home/harry/NeDM-traverse_mppi')
sys.path.insert(0, str(REPO / 'src')); sys.path.insert(0, str(REPO / 'scripts'))
import sensor_map_v2 as M  # noqa: E402
from nedm.traverse.terrain import TerrainMap  # noqa: E402

AMD = REPO / 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/vehicle_frames/g216_v2_group_0000'
EXCL_M = 6.0
SCALE_C = 511. / 512.          # Chrono RigidTerrain frame -> TerrainMap frame


def stats(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if a.size == 0:
        return {'n': 0}
    b = np.abs(a)
    return {'n': int(a.size), 'mean_signed': float(a.mean()), 'median_abs': float(np.median(b)),
            'p90_abs': float(np.percentile(b, 90)), 'p99_abs': float(np.percentile(b, 99)),
            'p999_abs': float(np.percentile(b, 99.9)), 'max_abs': float(b.max()),
            'frac_gt_0.05m': float((b > .05).mean()), 'frac_gt_0.5m': float((b > .5).mean())}


class Cam:
    def __init__(self, cam, w, h):
        self.H = float(cam['cam_height_m']); self.maxd = float(cam['max_depth_m'])
        self.f = (w / 2) / math.tan(float(cam['hfov_rad']) / 2)
        v, u = np.mgrid[0:h, 0:w]
        self.u, self.v = u, v
        self.rx = (u - (w - 1) / 2) / self.f
        self.ry = -(v - (h - 1) / 2) / self.f
        self.sec = np.sqrt(1 + self.rx ** 2 + self.ry ** 2)

    def valid(self, d):
        return np.isfinite(d) & (d > 0) & (d < self.maxd - 1e-6)

    def world(self, d):
        ax = np.where(self.valid(d), d / self.sec, np.nan)
        return self.rx * ax, self.ry * ax, self.H - ax

    def project(self, x, y, z):
        ax = self.H - z
        return (self.u.shape[1] - 1) / 2 + x / ax * self.f, (self.u.shape[0] - 1) / 2 - y / ax * self.f


def fit_warp(d_ref, d_mov, mask, iters=15):
    """Gauss-Newton fit of q = c + (sx*(u-cx)+tx, sy*(v-cy)+ty) minimising d_mov(q) - d_ref(p) on mask."""
    h, w = d_ref.shape; cx, cy = (w - 1) / 2, (h - 1) / 2
    v, u = np.nonzero(mask); du, dv = u - cx, v - cy
    ref = d_ref[v, u].astype(float)
    mov = np.where(np.isfinite(d_mov), d_mov, 0).astype(float)
    gy, gx = np.gradient(mov)
    p = np.array([1., 1., 0., 0.])
    for _ in range(iters):
        qu = cx + p[0] * du + p[2]; qv = cy + p[1] * dv + p[3]
        s = ndimage.map_coordinates(mov, [qv, qu], order=1, mode='nearest')
        r = s - ref
        gxi = ndimage.map_coordinates(gx, [qv, qu], order=1); gyi = ndimage.map_coordinates(gy, [qv, qu], order=1)
        keep = np.abs(r) < max(.5, 3 * np.median(np.abs(r)))
        J = np.stack([gxi * du, gyi * dv, gxi, gyi], 1)[keep]
        step = np.linalg.lstsq(J, -r[keep], rcond=None)[0]
        p += step
        if np.abs(step).max() < 1e-7:
            break
    return {'scale_x': float(p[0]), 'scale_y': float(p[1]), 'shift_u_px': float(p[2]), 'shift_v_px': float(p[3]),
            'residual_median_abs_m': float(np.median(np.abs(r[keep]))), 'n': int(keep.sum())}


def scale_scan(d_ref, d_mov, mask, scales):
    h, w = d_ref.shape; cx, cy = (w - 1) / 2, (h - 1) / 2
    v, u = np.nonzero(mask); ref = d_ref[v, u]
    mov = np.where(np.isfinite(d_mov), d_mov, 0.)
    out = []
    for s in scales:
        smp = ndimage.map_coordinates(mov, [cy + s * (v - cy), cx + s * (u - cx)], order=1, mode='constant', cval=np.nan)
        out.append((float(s), float(np.nanmedian(np.abs(smp - ref)))))
    return out


def rgb_registration(depth, rgb, cam, pose, zref, tmap, sky_rgb):
    """RGB vs depth in ONE frame: off-arena (sky) mask edges and the vehicle roof position."""
    rgb = np.asarray(rgb).astype(float)
    sky_rgb_mask = np.abs(rgb - np.asarray(sky_rgb, float)).max(-1) <= 2.
    sky_depth_mask = ~cam.valid(depth)
    rep = {'sky_color_used': [float(c) for c in sky_rgb],
           'rgb_sky_fraction': float(sky_rgb_mask.mean()), 'depth_invalid_fraction': float(sky_depth_mask.mean()),
           'mask_disagreement_pixels': int((sky_rgb_mask != sky_depth_mask).sum())}
    # per row: first/last terrain column in each image (terrain edge positions), per column: first/last row
    def edges(mask_terrain, axis):
        any_ = mask_terrain.any(axis=axis)
        first = np.argmax(mask_terrain, axis=axis).astype(float)
        last = (mask_terrain.shape[axis] - 1 - np.argmax(np.flip(mask_terrain, axis=axis), axis=axis)).astype(float)
        first[~any_] = np.nan; last[~any_] = np.nan
        return first, last
    tr, td = ~sky_rgb_mask, ~sky_depth_mask
    for name, axis in (('left_right_edge_cols', 1), ('top_bottom_edge_rows', 0)):
        f1, l1 = edges(tr, axis); f2, l2 = edges(td, axis)
        diff = np.concatenate([f1 - f2, l1 - l2])
        diff = diff[np.isfinite(diff)]
        # ignore the few rows/cols tangent to a corner where the edge is ill-defined
        rep[name + '_rgb_minus_depth_px'] = {'median': float(np.median(diff)), 'mean': float(diff.mean()),
                                              'p99_abs': float(np.percentile(np.abs(diff), 99)),
                                              'max_abs': float(np.abs(diff).max())}
    # vehicle: blue roof marker in RGB vs the high part of the vehicle in depth vs where the pose says the marker is
    blue = (rgb[..., 2] > rgb[..., 0] + 60) & (rgb[..., 2] > rgb[..., 1] + 40)
    x, y, z = cam.world(depth)
    near = np.hypot(x - pose[0], y - pose[1]) < EXCL_M
    ground = tmap.height(np.nan_to_num(x) * SCALE_C, np.nan_to_num(y) * SCALE_C)
    above = z - ground
    roof = near & (above > 1.55)
    body = near & (above > 0.5)
    c, s = math.cos(pose[2]), math.sin(pose[2])
    mx, my = pose[0] + c * .1, pose[1] + s * .1
    eu, ev = cam.project(mx, my, zref + .95 + .06)
    def centroid(m):
        vv, uu = np.nonzero(m)
        return [float(uu.mean()), float(vv.mean())] if uu.size else None
    rep['vehicle'] = {'expected_marker_px_from_pose': [float(eu), float(ev)],
                      'rgb_blue_marker': {'n_px': int(blue.sum()), 'centroid_uv': centroid(blue)},
                      'depth_roof_above_1.55m': {'n_px': int(roof.sum()), 'centroid_uv': centroid(roof)},
                      'depth_body_above_0.5m': {'n_px': int(body.sum()), 'centroid_uv': centroid(body)}}
    cb, cr = centroid(blue), centroid(roof)
    if cb and cr:
        rep['vehicle']['rgb_marker_minus_depth_roof_px'] = [cb[0] - cr[0], cb[1] - cr[1]]
    if cb:
        rep['vehicle']['rgb_marker_minus_pose_expected_px'] = [cb[0] - eu, cb[1] - ev]
    if cr:
        rep['vehicle']['depth_roof_minus_pose_expected_px'] = [cr[0] - eu, cr[1] - ev]
    # sub-pixel shift between the RGB blue-marker mask and the depth roof mask (cross-correlation peak)
    if blue.sum() and roof.sum():
        cu, cv = int(round(eu)), int(round(ev)); r = 40
        a_ = blue[cv - r:cv + r, cu - r:cu + r].astype(float); b_ = roof[cv - r:cv + r, cu - r:cu + r].astype(float)
        best = None
        for dv in range(-6, 7):
            for du in range(-6, 7):
                sc = float((np.roll(np.roll(b_, dv, 0), du, 1) * a_).sum())
                if best is None or sc > best[0]:
                    best = (sc, du, dv)
        rep['vehicle']['xcorr_shift_depth_roof_to_rgb_marker_px'] = [best[1], best[2]]
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--optix', required=True, help='harness frame0 npz (depth_m, pose[, rgb])')
    ap.add_argument('--optix-rgbd', default=None, help='frame0 npz from the --with-rgb run, for RGB registration')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    obs = json.loads((AMD / 'observation.json').read_text())
    camcfg = obs['camera']
    with np.load(AMD / 'observation.npz') as o:
        d_a = o['depth_m'].astype(np.float64); rgb_a = o['rgb']
    with np.load(a.optix) as o:
        d_o = o['depth_m'].astype(np.float64); pose_o = o['pose'].tolist()
        rgb_o = o['rgb'] if 'rgb' in o.files else None
    pose_a = obs['measured_pose_xy_yaw']
    arena = REPO / obs['arena']
    tmap = TerrainMap.from_dir(arena)
    h, w = d_a.shape
    assert d_o.shape == d_a.shape, (d_o.shape, d_a.shape)
    cam = Cam(camcfg, w, h)
    rep = {'optix_frame': a.optix, 'amd_frame': str(AMD), 'pose_optix_xy_yaw': pose_o, 'pose_amd_xy_yaw': pose_a,
           'pose_xy_difference_m': float(math.hypot(pose_o[0] - pose_a[0], pose_o[1] - pose_a[1])),
           'pose_yaw_difference_rad': float(pose_o[2] - pose_a[2]), 'focal_px': cam.f,
           'old_bug_focal_px': (w / 2) * (math.pi / 2) / float(camcfg['hfov_rad'])}
    rep['old_bug_expected_scale'] = rep['old_bug_focal_px'] / cam.f

    va, vo = cam.valid(d_a), cam.valid(d_o)
    rep['valid_fraction'] = {'optix': float(vo.mean()), 'amd': float(va.mean()),
                             'valid_mask_disagreement_pixels': int((va != vo).sum())}
    xa, ya, za = cam.world(d_a); xo, yo, zo = cam.world(d_o)
    def near_vehicle(x, y):
        return (np.hypot(x - pose_a[0], y - pose_a[1]) < EXCL_M) | (np.hypot(x - pose_o[0], y - pose_o[1]) < EXCL_M)
    excl = near_vehicle(np.nan_to_num(xa, nan=1e9), np.nan_to_num(ya, nan=1e9)) | \
        near_vehicle(np.nan_to_num(xo, nan=1e9), np.nan_to_num(yo, nan=1e9))
    terr = va & vo & ~excl
    rep['terrain_pixels'] = int(terr.sum()); rep['excluded_near_vehicle_pixels'] = int((va & vo & excl).sum())
    diff = d_o - d_a
    rep['depth_optix_minus_amd_terrain_m'] = stats(diff[terr])
    # interior (not on the arena rim, where a 1-px edge shift flips a pixel between terrain and sky/cliff)
    interior = terr & (np.hypot(xa, ya) < 38.) & (np.abs(xa) < 39.) & (np.abs(ya) < 39.)
    rep['depth_optix_minus_amd_interior_m'] = stats(diff[interior])
    # radial profile of the signed difference (a FOV error grows with image radius)
    rad = np.hypot(cam.u - (w - 1) / 2, cam.v - (h - 1) / 2)
    prof = []
    for r0 in range(0, 725, 50):
        m = terr & (rad >= r0) & (rad < r0 + 50)
        if m.sum() > 100:
            prof.append({'radius_px': [r0, r0 + 50], 'n': int(m.sum()), 'median_signed_m': float(np.median(diff[m])),
                         'p99_abs_m': float(np.percentile(np.abs(diff[m]), 99))})
    rep['radial_profile_optix_minus_amd'] = prof
    # where the largest differences are
    big = terr & (np.abs(diff) > .5)
    rep['pixels_abs_diff_gt_0.5m'] = {'n': int(big.sum()),
                                      'median_radius_world_m': float(np.median(np.hypot(xa[big], ya[big]))) if big.any() else None}
    # world-position error equivalent: the XY each backend assigns to the same pixel
    rep['world_xy_optix_minus_amd_terrain_m'] = stats(np.hypot(xo - xa, yo - ya)[terr])

    # image-space warp between the two depth images: identity (scale 1, shift 0) means the same camera model
    fitmask = interior & ~ndimage.binary_dilation(~(va & vo), iterations=3)
    rep['fitted_warp_optix_onto_amd'] = fit_warp(d_a, d_o, fitmask)
    scales = np.round(np.concatenate([np.arange(.80, 1.2001, .01)]), 4)
    scan = scale_scan(d_a, d_o, fitmask, scales)
    rep['scale_scan_median_abs_m'] = {f'{s:.2f}': v for s, v in scan}
    rep['scale_scan_best'] = min(scan, key=lambda t: t[1])
    # control: what the old FOV bug would have produced, simulated by warping the AMD depth with the old focal length
    k = cam.f / rep['old_bug_focal_px']
    cx, cy = (w - 1) / 2, (h - 1) / 2
    fake = ndimage.map_coordinates(np.where(va, d_a, np.nan), [cy + k * (cam.v - cy), cx + k * (cam.u - cx)],
                                   order=1, mode='constant', cval=np.nan)
    fm = terr & np.isfinite(fake)
    rep['control_old_bug_simulated_minus_amd_m'] = stats((fake - d_a)[fm])
    xf = cam.rx * (fake / cam.sec); yf = cam.ry * (fake / cam.sec)
    # true world XY of a pixel under the old bug = ray(old f) * axial; the planner decodes it with the correct f
    rx_old = (cam.u - cx) / rep['old_bug_focal_px']; ry_old = -(cam.v - cy) / rep['old_bug_focal_px']
    ax_old = fake / np.sqrt(1 + rx_old ** 2 + ry_old ** 2)
    rep['control_old_bug_world_xy_error_m'] = stats(np.hypot(rx_old * ax_old - xf, ry_old * ax_old - yf)[fm])

    # per-pixel height against the heightmap
    for name, (x, y, z, vm) in (('optix', (xo, yo, zo, vo)), ('amd', (xa, ya, za, va))):
        m = vm & ~excl & (np.abs(x) < 39.5) & (np.abs(y) < 39.5)
        x = np.nan_to_num(x); y = np.nan_to_num(y)
        rep[f'pixel_z_minus_terrainmap_{name}_m'] = stats((z - tmap.height(x * SCALE_C, y * SCALE_C))[m])
        rep[f'pixel_z_minus_terrainmap_no511_{name}_m'] = stats((z - tmap.height(x, y))[m])

    # back-projection into the planner grid
    rgb_for_o = rgb_o if rgb_o is not None else np.zeros((h, w, 3), np.uint8)
    go = M.grid_from_arrays(d_o, rgb_for_o, camcfg); ga = M.grid_from_arrays(d_a, rgb_a, camcfg)
    n = M.N; cc = -M.HALF + (np.arange(n) + .5) * M.MPP
    gx, gy = np.meshgrid(cc, cc)            # grid[row=y, col=x]
    gex = (np.hypot(gx - pose_a[0], gy - pose_a[1]) < EXCL_M) | (np.hypot(gx - pose_o[0], gy - pose_o[1]) < EXCL_M)
    co, ca = go['cover'] > 0, ga['cover'] > 0
    both = co & ca & ~gex
    rep['grid'] = {'coverage_optix': float(co.mean()), 'coverage_amd': float(ca.mean()),
                   'coverage_disagreement_cells': int((co != ca).sum()),
                   'cells_compared': int(both.sum()),
                   'z_optix_minus_amd_m': stats((go['z'] - ga['z'])[both]),
                   'mean_pixels_per_cell_optix': float(go['cover'][co].mean()),
                   'mean_pixels_per_cell_amd': float(ga['cover'][ca].mean())}
    hmap = tmap.height(gx * SCALE_C, gy * SCALE_C)
    inner = both & (np.abs(gx) < 39.5) & (np.abs(gy) < 39.5)
    rep['grid']['z_optix_minus_terrainmap_m'] = stats((go['z'] - hmap)[inner])
    rep['grid']['z_amd_minus_terrainmap_m'] = stats((ga['z'] - hmap)[inner])
    rep['grid']['z_optix_minus_terrainmap_no511_m'] = stats((go['z'] - tmap.height(gx, gy))[inner])

    # RGB-vs-depth registration
    from nedm.traverse.scene import SKY_RGB
    corner = rgb_a[2, 2].astype(float)
    rep['rgb_registration'] = {'amd_lavapipe': rgb_registration(d_a, rgb_a, cam, pose_a, obs['chassis_ref_z_m'], tmap,
                                                                corner)}
    if a.optix_rgbd:
        with np.load(a.optix_rgbd) as o:
            d_r = o['depth_m'].astype(np.float64); rgb_r = o['rgb']; pose_r = o['pose'].tolist()
        zr = float(tmap.height(pose_r[0] * SCALE_C, pose_r[1] * SCALE_C)) + (obs['chassis_ref_z_m']
                                                                           - float(tmap.height(pose_a[0] * SCALE_C, pose_a[1] * SCALE_C)))
        rep['rgb_registration']['optix_rgbd_run'] = rgb_registration(d_r, rgb_r, cam, pose_r, zr, tmap,
                                                                     rgb_r[2, 2].astype(float))
        rep['rgb_registration']['optix_rgbd_run']['depth_identical_to_depth_only_run'] = bool(np.array_equal(d_r, d_o))
        rep['rgb_registration']['optix_rgbd_run']['depth_max_abs_diff_vs_depth_only_run_m'] = float(
            np.nanmax(np.abs(np.where(cam.valid(d_r) & vo, d_r - d_o, 0))))
        rep['rgb_registration']['optix_rgbd_run']['pose'] = pose_r
        rep['rgb_registration']['sky_rgb_nominal_255'] = [c * 255 for c in SKY_RGB]
        # RGB images themselves: OptiX vs AMD sky masks (same scene, different renderer)
        sa = np.abs(rgb_a.astype(float) - corner).max(-1) <= 2.
        so = np.abs(rgb_r.astype(float) - rgb_r[2, 2].astype(float)).max(-1) <= 2.
        rep['rgb_registration']['sky_mask_optix_vs_amd_disagreement_pixels'] = int((sa != so).sum())
    Path(a.out).write_text(json.dumps(rep, indent=1) + '\n')
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
