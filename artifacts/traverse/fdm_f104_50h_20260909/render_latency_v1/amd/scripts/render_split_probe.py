"""Split the AMD (Vulkan-RT on lavapipe) overhead-frame cost into 'scene rebuild' and 'ray tracing'.

Companion to render_latency_bench.py (same scene, same case, same camera). Reading the Chrono Vulkan-RT source
(ChVulkanRTScene::SyncFromSystem + ChVulkanRTGpuRenderer::BuildScene/BuildAccelerationStructures) suggests that
whenever ANY body pose or velocity changes, the whole scene (the 522k-triangle heightmap included) is re-staged,
flattened into one world-space triangle list, re-uploaded and one bottom-level BVH is rebuilt from scratch before
the rays are traced. This probe tests that directly with a regime the harness does not have:

  frozen   only simulated time advances (system.SetChTime), no body moves -> scene revision must NOT change,
           so the frame is pure ray tracing + readback
  static   one physics step with the vehicle parked (the harness's `static`)
  moving   0.25 s of driving at 4 m/s between frames (the harness's `moving`)

Per frame it records wall time of manager.Update() and of the frame take, the Vulkan scene revision before/after
(did a rebuild happen?), and per-thread CPU time from /proc/self/task (main thread = serial part; all other
threads = lavapipe/llvmpipe workers etc.). Monotonic timestamps of every Update() are stored so a
`perf record -k CLOCK_MONOTONIC` profile can be cut per regime afterwards (perf_split.py).
"""
import argparse, json, math, os, platform, re, sys, time
from pathlib import Path
import numpy as np

CLK = os.sysconf('SC_CLK_TCK')


def thread_cpu():
    out = {}
    for d in os.listdir('/proc/self/task'):
        try:
            s = open(f'/proc/self/task/{d}/stat').read()
        except OSError:
            continue
        rp = s.rfind(')')
        comm = s[s.find('(') + 1:rp]
        f = s[rp + 2:].split()
        out[int(d)] = (comm, (int(f[11]) + int(f[12])) / CLK, int(f[11]) / CLK, int(f[12]) / CLK)
    return out


def cpu_delta(a, b, main_tid, split=None):
    main = 0.; workers = 0.; by_name = {}
    for tid, (comm, t, ut, st) in b.items():
        prev = a.get(tid, (comm, 0., 0., 0.))
        dt = t - prev[1]
        if split is not None:
            k = 'main' if tid == main_tid else 'workers'
            split[k + '_user_s'] = split.get(k + '_user_s', 0.) + ut - prev[2]
            split[k + '_sys_s'] = split.get(k + '_sys_s', 0.) + st - prev[3]
        if dt <= 0:
            continue
        if tid == main_tid:
            main += dt
        else:
            workers += dt
            key = re.sub(r'\d+', 'N', comm)
            by_name[key] = by_name.get(key, 0.) + dt
    return main, workers, by_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--sizes', type=int, nargs='+', default=[1024])
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--with-rgb', action='store_true')
    ap.add_argument('--chrono-data', required=True)
    ap.add_argument('--repo-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--hide-vehicle', action='store_true', help='remove the HMMWV visual meshes (~273k triangles) before the first render')
    a = ap.parse_args()
    repo = Path(a.repo_root).resolve()
    sys.path.insert(0, str(repo / 'src'))
    import pychrono as chrono
    import pychrono.vehicle as veh
    from nedm.traverse.layout import EpisodeLayout
    from nedm.traverse.scene import RenderSpec, build_config, build_scene
    from nedm.traverse.terrain import TerrainMap

    case = json.loads(Path(a.case).read_text())
    arena = (repo / case['arena']).resolve()
    layout = EpisodeLayout.from_json(case['layout'])
    tmap = TerrainMap.from_dir(arena)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    main_tid = int(os.getpid())
    result = {'label': a.label, 'host': platform.node(), 'pid': main_tid, 'cpus': os.cpu_count(),
              'affinity': len(os.sched_getaffinity(0)), 'case': case['id'], 'with_rgb': a.with_rgb, 'hide_vehicle': a.hide_vehicle,
              'env': {k: os.environ.get(k) for k in ('LP_NUM_THREADS', 'VK_ICD_FILENAMES', 'MESA_SHADER_CACHE_DIR',
                                                      'LP_NATIVE_VECTOR_WIDTH', 'OMP_NUM_THREADS')},
              'clock': 'time.monotonic (CLOCK_MONOTONIC)', 'sizes': {}}

    for size in a.sizes:
        cfg = build_config(arena, (*layout.start_xy, float(tmap.height(*layout.start_xy)) + .75), layout.start_yaw)
        cfg['chrono_data_root'] = str(Path(a.chrono_data).resolve())
        cfg['vehicle_data_root'] = str(Path(a.chrono_data).resolve() / 'vehicle')
        spec = RenderSpec(width=size, height=size, cam_height_m=110., hfov_rad=math.radians(47.), max_depth_m=180.,
                          plan_markers=False, light_elevation_deg=45., **({} if a.with_rgb else {'with_rgb': False}))
        t0 = time.monotonic(); c0 = thread_cpu()
        scene = build_scene(cfg, layout, tmap, arena, plan=None, render=spec)
        build_s = time.monotonic() - t0; bm, bw, _ = cpu_delta(c0, thread_cpu(), main_tid)
        hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
        if a.hide_vehicle:
            hmmwv.SetChassisVisualizationType(chrono.VisualizationType_NONE)
            hmmwv.SetWheelVisualizationType(chrono.VisualizationType_NONE)
            hmmwv.SetTireVisualizationType(chrono.VisualizationType_NONE)
        vehicle = hmmwv.GetVehicle()
        dt = float(cfg['simulation']['step_size_s'])
        vscene = getattr(scene.manager, 'vulkan_scene', None) or scene.manager.scene

        def revision():
            try:
                return int(vscene.GetRevision())
            except Exception:
                return -1

        pts = chrono.vector_ChVector3d()
        ref = case['settle_reference']; last = -10.
        for (x, y), s in zip(ref['waypoints'], ref['stations']):
            if s - last < 2. and s != ref['stations'][-1]:
                continue
            last = s; pts.append(chrono.ChVector3d(float(x), float(y), float(tmap.height(x, y)) + .5))
        driver = veh.ChPathFollowerDriver(vehicle, chrono.ChBezierCurve(pts), 'probe', 0.)
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
            r0 = revision(); c0 = thread_cpu(); p0 = os.times()
            m0 = time.monotonic(); scene.manager.Update(); m1 = time.monotonic()
            if a.with_rgb:
                scene.rgb_tap.take(timeout_s=900.)
            scene.depth_tap.take(timeout_s=900.); m2 = time.monotonic()
            c1 = thread_cpu(); p1 = os.times(); r1 = revision()
            split = {}
            mcpu, wcpu, names = cpu_delta(c0, c1, main_tid, split)
            return {'update_s': m1 - m0, 'take_s': m2 - m1, 'total_s': m2 - m0, 'mono': [m0, m2],
                    'cpu_user_sys_split': split, 'proc_user_s': p1.user - p0.user, 'proc_sys_s': p1.system - p0.system,
                    'rev_before': r0, 'rev_after': r1, 'rebuilt': (r1 != r0),
                    'main_cpu_s': mcpu, 'worker_cpu_s': wcpu, 'worker_cpu_by_name': names,
                    'proc_cpu_s': (p1.user + p1.system) - (p0.user + p0.system)}

        t0 = time.monotonic(); step(int(round(0.8 / dt)), 0.); settle_s = time.monotonic() - t0
        first = frame()
        census = {}
        for tid, (comm, *_) in thread_cpu().items():
            key = re.sub(r'\d+', 'N', comm); census[key] = census.get(key, 0) + 1
        stats = {}
        try:
            st = vscene.GetStats()
            stats = {k: int(getattr(st, k)) for k in ('bodies', 'other_items', 'visible_shapes', 'boxes', 'spheres',
                                                       'cylinders', 'triangle_meshes', 'unsupported_shapes')}
        except Exception as e:
            stats = {'error': repr(e)}
        tri = None
        try:
            prims = vscene.GetPrimitives()
            tri = [int(len(p.triangles)) for p in prims]
        except Exception as e:
            tri = repr(e)
        rec = {'dt': dt, 'scene_build_s': build_s, 'scene_build_main_cpu_s': bm, 'scene_build_worker_cpu_s': bw,
               'settle_physics_s': settle_s, 'first_frame': first, 'thread_census': census,
               'n_threads': sum(census.values()), 'scene_stats': stats, 'primitive_triangles': tri, 'regimes': {}}
        for regime in ('frozen', 'static', 'moving', 'frozen_after'):
            frames = []; phys = []
            for _ in range(a.frames):
                t0 = time.monotonic()
                if regime.startswith('frozen'):
                    system.SetChTime(system.GetChTime() + dt)
                elif regime == 'static':
                    step(1, 0.)
                else:
                    step(int(round(0.25 / dt)), 4.)
                phys.append(time.monotonic() - t0)
                frames.append(frame())
            tot = np.asarray([f['total_s'] for f in frames])
            rec['regimes'][regime] = {
                'frames': frames, 'physics_between_s': phys,
                'total_median_s': float(np.median(tot)), 'total_min_s': float(tot.min()), 'total_max_s': float(tot.max()),
                'main_cpu_median_s': float(np.median([f['main_cpu_s'] for f in frames])),
                'worker_cpu_median_s': float(np.median([f['worker_cpu_s'] for f in frames])),
                'rebuilt_count': int(sum(f['rebuilt'] for f in frames))}
        result['sizes'][str(size)] = rec
        print(json.dumps({'label': a.label, 'size': size, 'build_s': round(build_s, 2),
                          'first': round(first['total_s'], 3), 'threads': rec['n_threads'],
                          **{r: [round(v['total_median_s'], 3), round(v['main_cpu_median_s'], 2),
                                 round(v['worker_cpu_median_s'], 2), v['rebuilt_count']]
                             for r, v in rec['regimes'].items()}}), flush=True)
        (out / f'{a.label}.json').write_text(json.dumps(result, indent=1) + '\n')
        del driver, scene, hmmwv, system, terrain, vehicle, vscene
    (out / f'{a.label}.json').write_text(json.dumps(result, indent=1) + '\n')
    print('wrote', out / f'{a.label}.json')


if __name__ == '__main__':
    main()
