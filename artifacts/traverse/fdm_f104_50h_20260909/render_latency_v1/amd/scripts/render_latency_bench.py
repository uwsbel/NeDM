"""Where does the time go when Chrono::Sensor renders the overhead depth frame? One harness for every backend.

Builds exactly the scene the navigation runner renders (arena heightmap + HMMWV, overhead camera 110 m / 47 deg /
180 m max range), reproducing a stored vehicle-capture case so the frame can be checked pixel-for-pixel against the
depth the AMD cluster produced with lavapipe. For each image size it measures, per frame and separately:

  update_s   scene.manager.Update()  (scene sync + render launch; may or may not block, backend dependent)
  take_s     waiting for the frame + copying it into numpy
  step_s     physics advanced between frames (NOT part of the render cost, reported for scale)

in three regimes: `static` (vehicle parked, only simulated time advances), `moving` (vehicle driven at 4 m/s
between frames, so the dynamic geometry changes) and the first frame after scene build (pipeline/shader/BVH
warm-up). The same script runs on the AMD Vulkan-RT/lavapipe build, the local OptiX build and a local
Vulkan-RT build; the backend is whatever the imported pychrono was compiled with (and, for Vulkan, whatever ICD
VK_ICD_FILENAMES points at).

  python render_latency_bench.py --case CASE.json --label optix_5090 --sizes 256 512 1024 \
      --chrono-data DATA --out OUT_DIR [--with-rgb] [--frames 8]
"""
import argparse, json, math, os, platform, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def now():
    return time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True, help='a vehicle-capture case json (layout + arena + settle_reference)')
    ap.add_argument('--label', required=True)
    ap.add_argument('--sizes', type=int, nargs='+', default=[256, 512, 1024])
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--with-rgb', action='store_true')
    ap.add_argument('--chrono-data', required=True)
    ap.add_argument('--repo-root', default=str(ROOT), help='tree providing src/nedm and assets (cluster: nav_v1/source)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    repo = Path(a.repo_root).resolve()
    sys.path.insert(0, str(repo / 'src'))
    t_import = now()
    import pychrono as chrono
    import pychrono.vehicle as veh
    from nedm.traverse.layout import EpisodeLayout
    from nedm.traverse.scene import RenderSpec, build_config, build_scene
    from nedm.traverse.terrain import TerrainMap
    t_import = now() - t_import

    case = json.loads(Path(a.case).read_text())
    arena = (repo / case['arena']).resolve()
    layout = EpisodeLayout.from_json(case['layout'])
    tmap = TerrainMap.from_dir(arena)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = {k: os.environ.get(k) for k in ('VK_ICD_FILENAMES', 'LP_NUM_THREADS', 'OMP_NUM_THREADS',
                                           'CUDA_VISIBLE_DEVICES', 'MESA_SHADER_CACHE_DIR')}
    result = {'label': a.label, 'host': platform.node(), 'cpus': os.cpu_count(), 'pychrono': chrono.__file__,
              'env': env, 'case': case['id'], 'with_rgb': a.with_rgb, 'import_s': t_import, 'sizes': {}}

    for size in a.sizes:
        cfg = build_config(arena, (*layout.start_xy, float(tmap.height(*layout.start_xy)) + .75), layout.start_yaw)
        cfg['chrono_data_root'] = str(Path(a.chrono_data).resolve())
        cfg['vehicle_data_root'] = str(Path(a.chrono_data).resolve() / 'vehicle')
        spec = RenderSpec(width=size, height=size, cam_height_m=110., hfov_rad=math.radians(47.), max_depth_m=180.,
                          plan_markers=False, light_elevation_deg=45., **({} if a.with_rgb else {'with_rgb': False}))
        t0 = now(); scene = build_scene(cfg, layout, tmap, arena, plan=None, render=spec); build_s = now() - t0
        hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
        vehicle = hmmwv.GetVehicle()
        dt = float(cfg['simulation']['step_size_s'])

        # a straight 4 m/s path follower along the case's settle reference, used only for the `moving` regime
        pts = chrono.vector_ChVector3d()
        ref = case['settle_reference']; last = -10.
        for (x, y), s in zip(ref['waypoints'], ref['stations']):
            if s - last < 2. and s != ref['stations'][-1]:
                continue
            last = s; pts.append(chrono.ChVector3d(float(x), float(y), float(tmap.height(x, y)) + .5))
        driver = veh.ChPathFollowerDriver(vehicle, chrono.ChBezierCurve(pts), 'bench', 0.)
        driver.GetSteeringController().SetLookAheadDistance(5.); driver.GetSteeringController().SetGains(.8, 0., 0.)
        driver.GetSpeedController().SetGains(.6, .05, 0.); driver.Initialize()

        def step(n, speed):
            driver.SetDesiredSpeed(speed)
            for _ in range(n):
                ts = float(system.GetChTime())
                driver.Synchronize(ts); inputs = driver.GetInputs()
                terrain.Synchronize(ts); hmmwv.Synchronize(ts, inputs, terrain)
                driver.Advance(dt); terrain.Advance(dt); hmmwv.Advance(dt)

        def frame():
            t0 = now(); scene.manager.Update(); t1 = now()
            rgb = scene.rgb_tap.take(timeout_s=600.) if a.with_rgb else None
            depth = scene.depth_tap.take(timeout_s=600.); t2 = now()
            return t1 - t0, t2 - t1, depth, rgb

        # settle exactly like the capture (0.8 s at desired speed 0), then the warm-up frame
        t0 = now(); step(int(round(0.8 / dt)), 0.); settle_s = now() - t0
        up, tk, depth0, rgb0 = frame()
        ref_pose = vehicle.GetChassis().GetBody().GetFrameRefToAbs()
        pose0 = [ref_pose.GetPos().x, ref_pose.GetPos().y, ref_pose.GetRot().GetCardanAnglesZYX().z]
        ok = np.isfinite(depth0) & (depth0 > 0) & (depth0 < 180.)
        np.savez_compressed(out / f'{a.label}_{size}_frame0.npz', depth_m=depth0, pose=np.asarray(pose0),
                            **({'rgb': rgb0} if rgb0 is not None else {}))
        rec = {'scene_build_s': build_s, 'settle_physics_s': settle_s, 'first_frame': {'update_s': up, 'take_s': tk},
               'valid_fraction': float(ok.mean()), 'depth_min': float(depth0[ok].min()),
               'depth_max': float(depth0[ok].max()), 'pose_after_settle': pose0, 'regimes': {}}
        for regime, n_steps, speed in (('static', 1, 0.), ('moving', int(round(0.25 / dt)), 4.)):
            ups, tks, sts = [], [], []
            for _ in range(a.frames):
                t0 = now(); step(n_steps, speed); sts.append(now() - t0)
                u, t, _, _ = frame(); ups.append(u); tks.append(t)
            tot = np.asarray(ups) + np.asarray(tks)
            rec['regimes'][regime] = {
                'n': a.frames, 'update_s': ups, 'take_s': tks, 'physics_between_s': sts,
                'total_median_s': float(np.median(tot)), 'total_min_s': float(tot.min()),
                'total_max_s': float(tot.max()), 'update_median_s': float(np.median(ups)),
                'take_median_s': float(np.median(tks)), 'fps_median': float(1. / np.median(tot))}
        result['sizes'][str(size)] = rec
        print(json.dumps({'label': a.label, 'size': size, 'build_s': round(build_s, 2),
                          'first': round(up + tk, 4),
                          'static_med': round(rec['regimes']['static']['total_median_s'], 4),
                          'moving_med': round(rec['regimes']['moving']['total_median_s'], 4),
                          'valid': round(rec['valid_fraction'], 4)}), flush=True)
        # tear down before the next size
        del driver, scene, hmmwv, system, terrain, vehicle
    (out / f'{a.label}.json').write_text(json.dumps(result, indent=1) + '\n')
    print('wrote', out / f'{a.label}.json')


if __name__ == '__main__':
    main()
