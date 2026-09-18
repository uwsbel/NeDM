#!/usr/bin/env python3
"""Rigid-ground episode that starts MOVING: the vehicle is spawned on a remaining route at speed v0 (Chrono SetInitFwdVel)
with a 0.1 s free settle instead of the 0.8 s braked settle. Everything else is the frozen rigid runner
(traverse_fdm_rgbd_diverse_chrono.run_chrono) with gen_collect's stop policy hooks; the launch-state gate is relaxed to
allow the moving start. Purpose: fresh ground truth for the velocity-input study (rigid only, CPU).
  rigid_moving_collect.py --source-root SRC --case case.json --route route.json --v0 4.0 --out DIR --chrono-data DATA
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', required=True); p.add_argument('--case', required=True); p.add_argument('--route', required=True)
    p.add_argument('--out', required=True); p.add_argument('--chrono-data', required=True); p.add_argument('--v0', type=float, required=True)
    p.add_argument('--horizon-s', type=float, default=120.); p.add_argument('--settle-s', type=float, default=0.1)
    p.add_argument('--episode-seed', type=int, default=None)   # provenance only (passed by crm_worker.py)
    a = p.parse_args()
    source, out = Path(a.source_root).resolve(), Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source / 'scripts')); sys.path.insert(0, str(source / 'src'))
    import gen_collect
    module = gen_collect.import_runner(source)
    module.SETTLE_S = a.settle_s                     # read at call time by run_chrono: frame = -round(SETTLE_S/DT)
    from nedm.traverse import scene as scene_mod
    orig_build_config = scene_mod.build_config

    def moving_build_config(arena_dir, start_xyz, start_yaw, **kw):
        cfg = orig_build_config(arena_dir, start_xyz, start_yaw, **kw)
        cfg['vehicle']['init']['fwd_vel_mps'] = float(a.v0)      # hmmwv_data.create_hmmwv -> SetInitFwdVel
        return cfg
    module.__dict__['build_config'] = moving_build_config           # run_chrono imports build_config inside the function
    scene_mod.build_config = moving_build_config
    run, adapter = gen_collect.adapted_function(module)
    case = gen_collect.read(a.case)

    class MovingPolicy(gen_collect.StopPolicy):
        def on_anchor(self, scene, state, pose, history):        # no rest gate: the vehicle is meant to be moving
            self.validate_native_height(scene)
            self.initial_state_report = {'passed': True, 'moving_start_v0': a.v0, 'body_horizontal_speed_mps': float(np.linalg.norm(np.asarray(state)[:2])),
                                         'pose': np.asarray(pose).tolist(), 'settle_s': a.settle_s}
            gen_collect.dump(Path(self.args.out) / 'initial_state_validation.json', self.initial_state_report)
    a.minimum_elapsed_s, a.confirm_s, a.recovery_tail_s, a.disable_early_stop = 24., 2., 8., False
    a.f104_policy = MovingPolicy(a, case)
    a.command, a.backend, a.depth_ray_scale = 'collect', 'Vulkan_RT_lavapipe', 1.
    a.record_rgbd_stride, a.path_height_source, a.render_parity, a.rich_telemetry, a.frame_observer = 0, 'truth', False, False, None
    a.case, a.route, a.out = str(Path(a.case).resolve()), str(Path(a.route).resolve()), str(out)
    started = time.time()
    run(a)
    o = gen_collect.read(out / 'outcome.json'); z = np.load(out / 'trajectory.npz')
    gen_collect.dump(out / 'episode_complete.json', {'schema': 'rigid_moving_v1', 'v0': a.v0, 'settle_s': a.settle_s, 'status': o['status'],
                                                     'actual_elapsed_s': int(len(z['state'])) * 0.05, 'vx_first_frame': float(z['state'][0, 0]), 'wall_s': time.time() - started})
    print(json.dumps({'status': o['status'], 'elapsed_s': o['elapsed_s'], 'vx0': float(z['state'][0, 0]), 'v0': a.v0}))


if __name__ == '__main__':
    main()
