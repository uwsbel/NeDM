"""Where inside one Vulkan-RT frame does the time go? (companion to scripts/render_latency_bench.py, Vulkan-RT builds only)

Builds the same runner scene as the shared harness (arena + HMMWV, overhead camera 110 m / 47 deg / 180 m), settles
0.8 s, then renders frames in three regimes that differ only in what changed since the previous frame:

  time_only   simulated time is advanced (system.SetChTime) but NO body moves -> the scene signature is unchanged, so
              the renderer re-uses its acceleration structure: the frame is pure ray tracing + GPU->host readback.
  one_step    one physics step (vehicle parked) -> bodies move by micrometres, which invalidates the whole scene.
  moving      0.25 s of driving at 4 m/s (the harness `moving` regime).

For one_step/moving the scene staging is triggered explicitly first (manager.scene.SyncFromSystem, timed as `stage_s`:
Chrono walks every visual shape and copies/compares every triangle), and then manager.Update() is timed (`update_s`:
flattening every triangle into one world-space list, uploading it, rebuilding the single bottom-level acceleration
structure from scratch, tracing and reading back). The scene revision before/after tells whether a rebuild happened.
update_s(one_step) - update_s(time_only) is therefore the per-frame rebuild cost.

CPU accounting per call: main-thread CPU (time.thread_time) and all-other-threads CPU (process_time - thread_time),
so serial work in the Python/Chrono thread is separated from lavapipe's worker threads.

  PYTHONPATH=<vulkan build>/bin VK_ICD_FILENAMES=... python render_breakdown.py --case CASE --label L --size 1024 \
      --frames 5 --chrono-data DATA --out OUT_DIR
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np

REPO = Path('/home/harry/NeDM-traverse_mppi')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--case', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--chrono-data', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(REPO / 'src'))
    import pychrono as chrono
    import pychrono.vehicle as veh
    import pychrono.sensor as sens
    from nedm.traverse.layout import EpisodeLayout
    from nedm.traverse import scene as scene_mod
    from nedm.traverse.scene import RenderSpec, build_config, build_scene
    from nedm.traverse.terrain import TerrainMap

    class VerboseManager(sens.ChSensorManager):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.SetVerbose(True)
    scene_mod.sens.ChSensorManager = VerboseManager

    case = json.loads(Path(a.case).read_text())
    arena = (REPO / case['arena']).resolve()
    layout = EpisodeLayout.from_json(case['layout'])
    tmap = TerrainMap.from_dir(arena)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    cfg = build_config(arena, (*layout.start_xy, float(tmap.height(*layout.start_xy)) + .75), layout.start_yaw)
    cfg['chrono_data_root'] = str(Path(a.chrono_data).resolve())
    cfg['vehicle_data_root'] = str(Path(a.chrono_data).resolve() / 'vehicle')
    spec = RenderSpec(width=a.size, height=a.size, cam_height_m=110., hfov_rad=math.radians(47.), max_depth_m=180.,
                      plan_markers=False, light_elevation_deg=45., with_rgb=False)
    t0 = time.perf_counter(); scene = build_scene(cfg, layout, tmap, arena, plan=None, render=spec)
    build_s = time.perf_counter() - t0
    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
    vehicle = hmmwv.GetVehicle(); dt = float(cfg['simulation']['step_size_s'])
    vscene = scene.manager.scene

    pts = chrono.vector_ChVector3d(); ref = case['settle_reference']; last = -10.
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

    tick = os.sysconf('SC_CLK_TCK')

    def thread_cpu():
        # per-thread CPU (utime+stime, clock ticks) keyed by thread id; the main (Python/Chrono) thread has tid == pid
        acc = {}
        for task in Path('/proc/self/task').iterdir():
            try:
                fields = (task / 'stat').read_text().rsplit(')', 1)[1].split()
                acc[int(task.name)] = int(fields[11]) + int(fields[12])
            except (FileNotFoundError, ProcessLookupError, IndexError, ValueError):
                pass
        return acc

    pid = os.getpid()

    def timed(fn):
        c0 = thread_cpu()
        w0, p0, t0_ = time.perf_counter(), time.process_time(), time.thread_time()
        r = fn()
        w, p, t = time.perf_counter() - w0, time.process_time() - p0, time.thread_time() - t0_
        c1 = thread_cpu()
        d = {tid: (v - c0.get(tid, 0)) / tick for tid, v in c1.items()}
        others = sorted((v for tid, v in d.items() if tid != pid and v > 0), reverse=True)
        return r, {'wall_s': w, 'main_thread_cpu_s': t, 'other_threads_cpu_s': p - t,
                   'proc_main_thread_cpu_s': d.get(pid, 0.), 'proc_other_threads_cpu_s': float(sum(others)),
                   'proc_other_threads_active': len(others), 'proc_busiest_other_thread_cpu_s': others[0] if others else 0.,
                   'threads_total': len(c1)}

    step(int(round(0.8 / dt)), 0.)
    rev0 = vscene.GetRevision()
    _, first = timed(lambda: scene.manager.Update())
    _, first_take = timed(lambda: scene.depth_tap.take(timeout_s=600.))
    st = vscene.GetStats()
    rec = {'label': a.label, 'size': a.size, 'env': {k: os.environ.get(k) for k in ('VK_ICD_FILENAMES', 'LP_NUM_THREADS',
                                                                                   'MESA_SHADER_CACHE_DIR')},
           'scene_build_s': build_s, 'first_frame': {'update': first, 'take': first_take,
                                                     'revision_before': rev0, 'revision_after': vscene.GetRevision()},
           'scene_stats': {k: getattr(st, k) for k in ('bodies', 'other_items', 'visible_shapes', 'boxes', 'spheres',
                                                        'cylinders', 'triangle_meshes', 'unsupported_shapes')},
           'regimes': {}}
    for regime in ('time_only', 'one_step', 'moving', 'time_only_again'):
        frames = []
        for _ in range(a.frames):
            f = {}
            if regime.startswith('time_only'):
                system.SetChTime(system.GetChTime() + dt)
            else:
                step(1 if regime == 'one_step' else int(round(0.25 / dt)), 0. if regime == 'one_step' else 4.)
                r0 = vscene.GetRevision()
                _, f['stage'] = timed(lambda: vscene.SyncFromSystem(system))
                f['stage_changed_scene'] = bool(vscene.GetRevision() != r0)
            r1 = vscene.GetRevision()
            _, f['update'] = timed(lambda: scene.manager.Update())
            f['update_changed_revision'] = bool(vscene.GetRevision() != r1)
            _, f['take'] = timed(lambda: scene.depth_tap.take(timeout_s=600.))
            frames.append(f)
        def med(key, sub):
            vals = [fr[key][sub] for fr in frames if key in fr]
            return float(np.median(vals)) if vals else None
        rec['regimes'][regime] = {
            'frames': frames,
            'median': {f'{k}_{s}': med(k, s) for k in ('stage', 'update', 'take')
                       for s in ('wall_s', 'main_thread_cpu_s', 'other_threads_cpu_s')},
            'scene_rebuilt_each_frame': all(fr.get('stage_changed_scene', False) for fr in frames)}
        m = rec['regimes'][regime]['median']
        print(json.dumps({'label': a.label, 'size': a.size, 'regime': regime,
                          'stage_wall': m['stage_wall_s'], 'update_wall': m['update_wall_s'],
                          'update_main_cpu': m['update_main_thread_cpu_s'],
                          'update_other_cpu': m['update_other_threads_cpu_s'], 'take_wall': m['take_wall_s'],
                          'rebuilt': rec['regimes'][regime]['scene_rebuilt_each_frame']}), flush=True)
    (out / f'{a.label}.json').write_text(json.dumps(rec, indent=1) + '\n')
    print('wrote', out / f'{a.label}.json')


if __name__ == '__main__':
    main()
