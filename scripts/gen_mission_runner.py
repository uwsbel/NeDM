"""Five-goal mission: ONE continuous Chrono simulation, replanning from the vehicle's actual pose at every goal.

Physics, vehicle, terrain, path follower and the per-step loop statements are the frozen collector's
(source_v1 scripts/traverse_fdm_rgbd_diverse_chrono.py run_chrono), reproduced here because the frozen function
has no way to switch routes mid-run. What is different, by design:
  * at each goal (chassis within the goal radius) the next route is planned from the measured pose and a fresh
    ChPathFollowerDriver with the frozen gains is built on it (simulation clock paused while planning; the
    steering-rate limiter state is carried over; PID integrators restart);
  * the collector's stop rules are applied per leg with leg-relative time (stall confirmation not before 24 s of
    leg time, 2 s confirm + 8 s tail), plus rollover > 60 deg, terrain-bounds exit and a 120 s leg horizon;
  * no rich telemetry, no rendering.
Arms: n2 (deployed model), rule (hand rule), straight6 (6 m/s straight route from the measured pose).

  python gen_mission_runner.py --mission m.json --arm n2 --out DIR --source-root SRC --chrono-data DATA
"""
import argparse, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np

DT = .05


def dump(p, v):
    Path(p).write_text(json.dumps(v, indent=1, allow_nan=False) + '\n')


class LegStop:
    """The collector's bounded-blockage rule, with elapsed time counted from the start of the leg."""
    def __init__(self, minimum_s=24., confirm_s=2., tail_s=8., diameter_m=.25):
        self.minimum_s, self.confirm_s, self.tail_s, self.diameter_m = minimum_s, confirm_s, tail_s, diameter_m
        self.first = None; self.confirmed = False

    def check(self, elapsed, poses, actions, parked, terminal_pose):
        if np.max(np.abs(np.asarray(terminal_pose)[:2])) > 40.:
            return 'terrain_bounds_exit'
        if len(actions) < 40:
            return None
        bounded = False
        if not any(parked[-40:]) and np.all(np.asarray(actions[-40:])[:, 1] > .3):
            pts = np.concatenate((np.asarray(poses[-40:])[:, :2], np.asarray(terminal_pose)[None, :2]))
            bounded = bool(np.square(pts[:, None, :] - pts[None, :, :]).sum(-1).max() <= self.diameter_m ** 2)
        if not bounded:
            self.first, self.confirmed = None, False
            return None
        if elapsed + 1e-9 < self.minimum_s:
            return None
        if self.first is None:
            self.first = elapsed
        if elapsed + 1e-9 >= self.first + self.confirm_s:
            self.confirmed = True
        if elapsed + 1e-9 >= self.first + self.confirm_s + self.tail_s:
            return 'prolonged_blockage_terminated'
        return None


def route_json(r):
    return {k: np.asarray(r[k]).tolist() for k in ('waypoints', 'speeds', 'stations', 'headings')} | {'meta': r.get('meta', {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mission', required=True); ap.add_argument('--arm', required=True, choices=['n2', 'rule', 'straight6', 's'])
    ap.add_argument('--sensor-map', default=None, help='captured RGB-D map dir; if given, every arm plans from the image')
    ap.add_argument('--out', required=True); ap.add_argument('--source-root', required=True)
    ap.add_argument('--chrono-data', required=True); ap.add_argument('--leg-horizon-s', type=float, default=120.)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'mission_outcome.json').exists():
        print('exists', out); return
    source = Path(a.source_root).resolve()
    sys.path.insert(0, str(source / 'src'))
    spec = importlib.util.spec_from_file_location('frozen_runner', source / 'scripts/traverse_fdm_rgbd_diverse_chrono.py')
    fr = importlib.util.module_from_spec(spec); sys.modules['frozen_runner'] = fr; spec.loader.exec_module(fr)
    import gen_planner as P
    import torch
    torch.set_num_threads(1)
    import pychrono as chrono
    import pychrono.vehicle as veh
    from nedm.hmmwv_data import WHEEL_SPECS, capture_row
    from nedm.training.constants import STATE_FIELD_PRESETS
    from nedm.traverse.layout import EpisodeLayout
    from nedm.traverse.scene import build_config, build_scene
    from nedm.traverse.terrain import TerrainMap

    m = json.load(open(a.mission))
    arena = (source / m['arena']).resolve()
    tmap = TerrainMap.from_dir(arena)
    if a.sensor_map:
        P.set_sensor_map(a.sensor_map)
    else:
        P.set_map(arena)
    model = P.RiskModel(device='cpu') if a.arm == 'n2' else (P.SensorRiskModel(os.environ['GEN_S_MODELS'], device='cpu') if a.arm == 's' else None)
    rule = P.HandRule() if a.arm == 'rule' else None
    rng = np.random.default_rng(int(hashlib.md5((m['id'] + a.arm).encode()).hexdigest()[:8], 16))
    goals = [np.asarray(g, float) for g in m['goals']]
    radius = float(m.get('goal_radius_m', 2.5))
    layout = EpisodeLayout.from_json(m['layout'])
    config = build_config(arena, (*layout.start_xy, float(tmap.height(*layout.start_xy)) + .75), layout.start_yaw)
    config['chrono_data_root'] = str(Path(a.chrono_data).resolve())
    config['vehicle_data_root'] = str(Path(a.chrono_data).resolve() / 'vehicle')
    scene = build_scene(config, layout, tmap, arena, plan=None, render=None)
    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
    vehicle = hmmwv.GetVehicle()
    engine = vehicle.GetEngine()
    tire_radii = {name: float(vehicle.GetTire(axle, side).GetRadius()) for name, axle, side in WHEEL_SPECS}
    fields = STATE_FIELD_PRESETS['tire_normal_force_omega_pt']
    dt = float(config['simulation']['step_size_s']); substeps = int(round(DT / dt))

    def plan_leg(pose, k):
        t0 = time.time()
        mode = a.arm
        r, info = P.plan(np.asarray(pose, float), goals[k], rng, model=model, rule=rule, mode=mode)
        info['plan_wall_s'] = time.time() - t0
        return r, info

    start_pose = [layout.start_xy[0], layout.start_xy[1], layout.start_yaw]
    route, info = plan_leg(start_pose, 0)
    legs = []
    if route is None:
        dump(out / 'mission_outcome.json', {'mission': m['id'], 'arm': a.arm, 'status': 'no_route_leg0', 'goals_reached': 0,
                                            'legs': [{'goal_index': 0, 'status': 'no_route', 'plan': info}]})
        return
    dump(out / 'route_leg0.json', route_json(route))
    driver = fr.make_driver(chrono, veh, vehicle, route, tmap)
    xy, speed = np.asarray(route['waypoints'], float), np.asarray(route['speeds'], float)
    frame, previous_steer, wp = -int(round(fr.SETTLE_S / DT)), 0., 0
    k = 0; leg_start = 0; stop = LegStop()
    leg_poses, leg_actions, leg_parked = [], [], []
    S_, A_, Pz, Lg = [], [], [], []
    status = None
    cur = {'goal_index': 0, 'plan': info, 'start_frame': 0}
    wall0 = time.time()
    while True:
        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        pos = np.array([ref.GetPos().x, ref.GetPos().y])
        wp = fr.nearest_index(xy, pos, wp)
        at_end = wp >= len(xy) - 2 and np.linalg.norm(pos - xy[-1]) < 3.
        driver.SetDesiredSpeed(0. if frame < 0 or at_end else float(speed[wp]))
        for sub in range(substeps):
            ts = float(system.GetChTime())
            driver.Synchronize(ts)
            inputs = driver.GetInputs()
            previous_steer = 0. if frame < 0 else float(np.clip(inputs.m_steering, previous_steer - 2. * dt, previous_steer + 2. * dt))
            inputs.m_steering = previous_steer
            terrain.Synchronize(ts)
            hmmwv.Synchronize(ts, inputs, terrain)
            if sub == 0 and frame >= 0:
                row = capture_row(hmmwv, terrain, 'fdm_rgbd_eval', 'follower', m['id'], 'development',
                                  frame, ts, inputs, include_tires=True, tire_radii=tire_radii)
                row['engine_motor_speed_radps'] = float(engine.GetMotorSpeed())
                row['engine_motorshaft_torque_nm'] = float(engine.GetOutputMotorshaftTorque())
                state = np.array([float(row[f]) for f in fields], np.float32)
                pose = np.array([row['pos_x_m'], row['pos_y_m'], row['yaw_rad']], np.float64)
                action = np.array([inputs.m_steering, inputs.m_throttle, inputs.m_braking], np.float32)
                S_.append(state); A_.append(action); Pz.append(pose); Lg.append(k)
                leg_poses.append(pose); leg_actions.append(action); leg_parked.append(bool(at_end))
            driver.Advance(dt); terrain.Advance(dt); hmmwv.Advance(dt)
        if frame >= 0:
            ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
            tp = np.array([ref.GetPos().x, ref.GetPos().y, ref.GetRot().GetCardanAnglesZYX().z])
            leg_elapsed = (frame + 1 - leg_start) * DT
            if abs(float(vehicle.GetRoll())) > math.radians(60.) or abs(float(vehicle.GetPitch())) > math.radians(60.):
                status = 'rollover'
            elif np.linalg.norm(tp[:2] - goals[k]) <= radius:
                cur.update(status='goal_reached', elapsed_s=leg_elapsed, end_frame=frame + 1)
                legs.append(cur)
                if k == len(goals) - 1:
                    status = 'mission_complete'
                else:
                    k += 1
                    route, info = plan_leg(tp, k)
                    cur = {'goal_index': k, 'plan': info, 'start_frame': frame + 1, 'start_pose': tp.tolist()}
                    if route is None:
                        cur.update(status='no_route', elapsed_s=0.); legs.append(cur); status = 'no_route'
                    else:
                        dump(out / f'route_leg{k}.json', route_json(route))
                        driver = fr.make_driver(chrono, veh, vehicle, route, tmap)
                        xy, speed = np.asarray(route['waypoints'], float), np.asarray(route['speeds'], float)
                        wp = 0; leg_start = frame + 1; stop = LegStop()
                        leg_poses, leg_actions, leg_parked = [], [], []
            else:
                s = stop.check(leg_elapsed, leg_poses, leg_actions, leg_parked, tp)
                if s is not None:
                    status = s
                elif leg_elapsed + 1e-9 >= a.leg_horizon_s:
                    status = 'timeout'
            if status is not None:
                if status not in ('mission_complete', 'no_route'):
                    cur.update(status=status, elapsed_s=leg_elapsed, end_frame=frame + 1); legs.append(cur)
                break
        frame += 1
    S_, A_, Pz, Lg = np.asarray(S_), np.asarray(A_), np.asarray(Pz), np.asarray(Lg, np.int16)
    np.savez_compressed(out / 'trajectory.npz', state=S_, action=A_, pose=Pz, leg=Lg, fields=np.asarray(fields))
    vx, thr = S_[:, 0], A_[:, 1]
    s0 = 20
    back = (vx < -0.10) & (thr > 0.3)
    slide_frames = np.flatnonzero((back | (vx < -0.30)) & (np.arange(len(vx)) >= s0))
    for L in legs:
        a0, a1 = L.get('start_frame', 0), L.get('end_frame', len(vx))
        seg = np.arange(len(vx))[(np.arange(len(vx)) >= max(a0, s0)) & (np.arange(len(vx)) < a1)]
        L['slid'] = bool(len(np.intersect1d(seg, slide_frames)))
        L['back_s'] = float(back[seg].sum() * DT) if len(seg) else 0.
        L['max_tilt_deg'] = float(np.degrees(np.abs(S_[seg, 2:4])).max()) if len(seg) else 0.
    reached = sum(1 for L in legs if L.get('status') == 'goal_reached')
    dump(out / 'mission_outcome.json', {
        'mission': m['id'], 'arena': m['arena'], 'arm': a.arm, 'status': status, 'goals_reached': reached,
        'success': status == 'mission_complete', 'total_time_s': len(vx) * DT, 'any_slide': bool(len(slide_frames)),
        'back_s': float(back[s0:].sum() * DT), 'max_tilt_deg': float(np.degrees(np.abs(S_[s0:, 2:4])).max()) if len(vx) > s0 else 0.,
        'legs': legs, 'wall_s': time.time() - wall0})
    print(json.dumps({'mission': m['id'], 'arm': a.arm, 'status': status, 'goals_reached': reached,
                      'sim_s': len(vx) * DT, 'wall_s': round(time.time() - wall0)}), flush=True)


if __name__ == '__main__':
    main()
