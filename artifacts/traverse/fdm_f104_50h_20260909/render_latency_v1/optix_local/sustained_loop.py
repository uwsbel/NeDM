"""Sustained OptiX rendering at 1024x1024: steady-state frame rate, GPU memory, and a check that every frame shows the
vehicle where it is NOW (not a stale frame), on the same scene as render_latency_bench.py.

After the 0.8 s settle the vehicle drives a closed circle (radius 10 m, 4 m/s). Two regimes, 300 frames each:
  moving  N physics steps between frames (default 25 = 0.05 s, vehicle moves ~0.2 m per frame), per-frame freshness check
  tight   1 physics step between frames (vehicle still driving): as close to back-to-back rendering as the manual
          trigger allows
GPU memory / utilisation are sampled with nvidia-smi once per second in a background thread.

  PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 sustained_loop.py --label sustained_depth [--with-rgb]
"""
import argparse, json, math, subprocess, sys, threading, time
from pathlib import Path
import numpy as np

REPO = Path('/home/harry/NeDM-traverse_mppi')
HERE = Path(__file__).resolve().parent
CASE = REPO / 'artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/cases_g216/cases/g216_v2_group_0000.json'


def gpu_sample():
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True).stdout.strip().split(',')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True).stdout.strip()
    return {'t': time.time(), 'mem_used_mib': float(out[0]), 'util_pct': float(out[1]), 'compute_apps': apps}


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.samples = []; self.stop = False; self.tag = 'idle'

    def run(self):
        while not self.stop:
            s = gpu_sample(); s['tag'] = self.tag; self.samples.append(s); time.sleep(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--steps-between', type=int, default=25)
    ap.add_argument('--with-rgb', action='store_true')
    ap.add_argument('--chrono-data', default='/home/harry/chrono/data')
    a = ap.parse_args()
    sys.path.insert(0, str(REPO / 'src'))
    import pychrono as chrono
    import pychrono.vehicle as veh
    from nedm.traverse.layout import EpisodeLayout
    from nedm.traverse.scene import RenderSpec, build_config, build_scene
    from nedm.traverse.terrain import TerrainMap

    sampler = Sampler(); base = gpu_sample(); sampler.start()
    case = json.loads(CASE.read_text())
    arena = (REPO / case['arena']).resolve(); layout = EpisodeLayout.from_json(case['layout'])
    tmap = TerrainMap.from_dir(arena)
    cfg = build_config(arena, (*layout.start_xy, float(tmap.height(*layout.start_xy)) + .75), layout.start_yaw)
    cfg['chrono_data_root'] = str(Path(a.chrono_data)); cfg['vehicle_data_root'] = str(Path(a.chrono_data) / 'vehicle')
    W = a.size; hfov = math.radians(47.)
    spec = RenderSpec(width=W, height=W, cam_height_m=110., hfov_rad=hfov, max_depth_m=180., plan_markers=False,
                      light_elevation_deg=45., with_rgb=a.with_rgb)
    sampler.tag = 'build'
    t0 = time.perf_counter(); scene = build_scene(cfg, layout, tmap, arena, plan=None, render=spec); build_s = time.perf_counter() - t0
    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
    vehicle = hmmwv.GetVehicle(); dt = float(cfg['simulation']['step_size_s'])

    # closed circle of radius R, tangent to the start heading, turning left
    x0, y0 = layout.start_xy; yaw = layout.start_yaw; R = 10.
    cx, cy = x0 - R * math.sin(yaw), y0 + R * math.cos(yaw)
    pts = chrono.vector_ChVector3d()
    for lap in range(8):
        for k in range(36):
            th = yaw - math.pi / 2 + 2 * math.pi * k / 36
            px, py = cx + R * math.cos(th), cy + R * math.sin(th)
            pts.append(chrono.ChVector3d(px, py, float(tmap.height(px, py)) + .5))
    driver = veh.ChPathFollowerDriver(vehicle, chrono.ChBezierCurve(pts), 'circle', 0.)
    driver.GetSteeringController().SetLookAheadDistance(5.); driver.GetSteeringController().SetGains(.8, 0., 0.)
    driver.GetSpeedController().SetGains(.6, .05, 0.); driver.Initialize()

    def step(n, speed):
        driver.SetDesiredSpeed(speed)
        for _ in range(n):
            ts = float(system.GetChTime())
            driver.Synchronize(ts); inputs = driver.GetInputs()
            terrain.Synchronize(ts); hmmwv.Synchronize(ts, inputs, terrain)
            driver.Advance(dt); terrain.Advance(dt); hmmwv.Advance(dt)

    def pose():
        fr = vehicle.GetChassis().GetBody().GetFrameRefToAbs()
        return np.array([fr.GetPos().x, fr.GetPos().y, fr.GetRot().GetCardanAnglesZYX().z, fr.GetPos().z])

    f = (W / 2) / math.tan(hfov / 2); c0 = (W - 1) / 2

    def vehicle_centroid(depth, p):
        """world XY centroid of pixels > 0.5 m above the heightmap within 4.5 m of pose p (window around its projection)"""
        u0 = int(round(c0 + p[0] / (110. - p[3]) * f)); v0 = int(round(c0 - p[1] / (110. - p[3]) * f)); r = int(6 * f / 108)
        vs, us = np.mgrid[max(v0 - r, 0):min(v0 + r, W), max(u0 - r, 0):min(u0 + r, W)]
        d = depth[vs, us].astype(float); rx = (us - c0) / f; ry = -(vs - c0) / f
        ax = d / np.sqrt(1 + rx ** 2 + ry ** 2); x = rx * ax; y = ry * ax; z = 110. - ax
        m = (d < 179.9) & (np.hypot(x - p[0], y - p[1]) < 4.5)
        m &= (z - tmap.height(np.clip(x, -40, 40) * 511 / 512, np.clip(y, -40, 40) * 511 / 512)) > .5
        return (float(x[m].mean()), float(y[m].mean()), int(m.sum())) if m.any() else (math.nan, math.nan, 0)

    def body_offset(cxy, p):
        dx, dy = cxy[0] - p[0], cxy[1] - p[1]; c, s = math.cos(p[2]), math.sin(p[2])
        return c * dx + s * dy, -s * dx + c * dy       # forward, left

    sampler.tag = 'settle'
    t0 = time.perf_counter(); step(int(round(.8 / dt)), 0.); settle_s = time.perf_counter() - t0
    # get up to speed before timing
    step(int(round(2. / dt)), 4.)
    res = {'label': a.label, 'size': W, 'with_rgb': a.with_rgb, 'scene_build_s': build_s, 'settle_physics_s': settle_s,
           'gpu_baseline_before_build': base, 'regimes': {}}
    for regime, nsteps in (('moving', a.steps_between), ('tight', 1)):
        sampler.tag = regime
        ren, upd, tak, phys, fresh = [], [], [], [], []
        prev = pose(); cpu0 = time.process_time(); w0 = time.perf_counter()
        for i in range(a.frames):
            t0 = time.perf_counter(); step(nsteps, 4.); t1 = time.perf_counter()
            p = pose()
            scene.manager.Update(); t2 = time.perf_counter()
            if a.with_rgb:
                scene.rgb_tap.take(timeout_s=60.)
            depth = scene.depth_tap.take(timeout_s=60.); t3 = time.perf_counter()
            phys.append(t1 - t0); upd.append(t2 - t1); tak.append(t3 - t2); ren.append(t3 - t1)
            if regime == 'moving':
                cx_, cy_, n = vehicle_centroid(depth, p)
                if n:
                    fresh.append([*body_offset((cx_, cy_), p), *body_offset((cx_, cy_), prev), n,
                                  float(math.hypot(p[0] - prev[0], p[1] - prev[1]))])
            prev = p
        wall = time.perf_counter() - w0; cpu = time.process_time() - cpu0
        ren = np.asarray(ren)
        rec = {'frames': a.frames, 'physics_steps_between_frames': nsteps,
               'render_s': {'median': float(np.median(ren)), 'p10': float(np.percentile(ren, 10)),
                            'p90': float(np.percentile(ren, 90)), 'p99': float(np.percentile(ren, 99)),
                            'max': float(ren.max()), 'mean': float(ren.mean())},
               'update_median_s': float(np.median(upd)), 'take_median_s': float(np.median(tak)),
               'physics_between_median_s': float(np.median(phys)),
               'render_only_fps_from_median': float(1 / np.median(ren)), 'render_only_fps_from_mean': float(1 / ren.mean()),
               'loop_wall_s': wall, 'loop_fps_including_physics': a.frames / wall,
               'process_cpu_s_per_wall_s': cpu / wall,
               'render_s_first10': ren[:10].tolist(), 'render_s_last10': ren[-10:].tolist()}
        if fresh:
            F = np.asarray(fresh)
            rec['freshness'] = {
                'frames_checked': int(len(F)),
                'vehicle_pixels_median': float(np.median(F[:, 4])),
                'moved_between_frames_median_m': float(np.median(F[:, 5])),
                'centroid_minus_current_pose_forward_m': {'median': float(np.median(F[:, 0])), 'std': float(F[:, 0].std())},
                'centroid_minus_current_pose_left_m': {'median': float(np.median(F[:, 1])), 'std': float(F[:, 1].std())},
                'centroid_minus_previous_frame_pose_forward_m': {'median': float(np.median(F[:, 2])), 'std': float(F[:, 2].std())},
                'note': 'a fresh frame gives a constant body-frame offset to the CURRENT pose (small std); a one-frame-stale '
                        'frame would be constant relative to the PREVIOUS pose instead'}
        res['regimes'][regime] = rec
        print(regime, json.dumps({k: v for k, v in rec.items() if not k.startswith('render_s_')}), flush=True)
    sampler.tag = 'done'; time.sleep(1.2); sampler.stop = True; sampler.join()
    S = sampler.samples
    for tag in ('build', 'settle', 'moving', 'tight'):
        mem = [s['mem_used_mib'] for s in S if s['tag'] == tag]; ut = [s['util_pct'] for s in S if s['tag'] == tag]
        if mem:
            res.setdefault('gpu', {})[tag] = {'samples': len(mem), 'mem_used_mib_max': max(mem), 'mem_used_mib_median': float(np.median(mem)),
                                              'util_pct_median': float(np.median(ut)), 'util_pct_max': max(ut)}
    apps = [s['compute_apps'] for s in S if s['tag'] in ('moving', 'tight') and s['compute_apps']]
    res['gpu_compute_apps_example'] = apps[-1] if apps else None
    res['gpu_samples'] = S
    (HERE / f'{a.label}.json').write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res.get('gpu'), indent=1), res['gpu_compute_apps_example'], 'baseline', base['mem_used_mib'])


if __name__ == '__main__':
    main()
