"""Continuous sensor-driven waypoint navigation: ONE Chrono rollout, replanning from fresh overhead depth.

The vehicle drives a sequence of waypoints without ever being reset. At each planning decision the simulation
renders one overhead RGB-D frame WITH the vehicle in it, back-projects it to the metric grid, masks the vehicle's
own footprint, builds candidate corridors from that single frame, scores them with the frozen ensemble and hands
the winner to the path follower. Modes:

  --mode waypoint          one decision per waypoint (the existing multi-goal behaviour, sensor-driven)
  --mode periodic --period 2.0   a decision every 2 s of simulated time, toward the active waypoint

Physics, vehicle, terrain, path geometry, stop rules and labels are the frozen collector's. What differs, by
design, from scripts/gen_mission_runner.py:
  * the planner input is a live depth frame, not the arena heightmap;
  * throttle/braking come from nav_online.SpeedPI so the speed integrator survives a route change (the Chrono
    path follower is rebuilt for steering only, which is proportional and therefore stateless);
  * the reference path handed to the follower is lifted with SENSED heights (nav_online.path_heights) instead of
    the heightmap, unless --path-heights map is given;
  * planning latency can be charged to the simulation (--latency-s -1 uses each decision's own measured wall
    time; a positive value charges a fixed budget), in which case the vehicle keeps following the previous route
    until the new one activates.

  python nav_runner.py --mission m.json --mode periodic --period 2.0 --out DIR \
      --source-root SRC --chrono-data DATA --models 'matched_Dabs_s*.pt'
"""
import argparse, hashlib, importlib.util, json, math, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DT = .05


def dump(p, v):
    Path(p).write_text(json.dumps(v, indent=1, allow_nan=False) + '\n')


def route_json(r):
    return {k: np.asarray(r[k]).tolist() for k in ('waypoints', 'speeds', 'stations')} | {'meta': r.get('meta', {})}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mission', required=True)
    ap.add_argument('--mode', choices=['waypoint', 'periodic'], default='periodic')
    ap.add_argument('--period', type=float, default=2.0, help='simulated seconds between decisions')
    ap.add_argument('--models', default=None, help='glob of frozen checkpoints (default $NAV_MODELS)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--source-root', required=True)
    ap.add_argument('--chrono-data', required=True)
    ap.add_argument('--latency-s', type=float, default=0.0,
                    help='0 = free planning; -1 = charge the whole measured wall time; -2 = charge everything but '
                         'the render (the software rasteriser is a simulator cost, not an algorithm cost); '
                         '>0 = charge a fixed budget')
    ap.add_argument('--path-heights', choices=['sensed', 'map'], default='sensed')
    ap.add_argument('--no-mask', action='store_true')
    ap.add_argument('--margin-m', type=float, default=1.5)
    ap.add_argument('--sense-radius-m', type=float, default=0.0,
                    help='0 = the whole overhead frame; >0 keeps only terrain within this radius of the vehicle '
                         'and marks the rest unobserved, as a stand-in for a limited-range sensor')
    ap.add_argument('--leg-horizon-s', type=float, default=120.)
    ap.add_argument('--mission-horizon-s', type=float, default=700.)
    ap.add_argument('--min-replan-dist-m', type=float, default=6.0, help='skip replanning inside this radius of the waypoint')
    ap.add_argument('--fallback-speed-mps', type=float, default=2.0,
                    help='commanded speed on fallback shapes (tight rescue arc, surrogate goal, straight ahead); '
                         'the follower cannot hold an 8 m radius at 4 m/s')
    ap.add_argument('--save-frames', action='store_true', help='store each decision grid + rgb (video missions only)')
    ap.add_argument('--rgb', choices=['on', 'off', 'auto'], default='auto',
                    help="render the overhead RGB camera as well as depth; 'auto' = only when frames are saved. "
                         "Refused unless the checkpoints ignore the colour channels.")
    ap.add_argument('--torch-threads', type=int, default=1)
    ap.add_argument('--device', default='cpu', help="risk-model device ('cpu' on the cluster, 'cuda' on a GPU box)")
    ap.add_argument('--video-dir', default=None,
                    help='save Chrono::Sensor RGB camera frames here while driving (forces the RGB camera on)')
    ap.add_argument('--video-every', type=int, default=2, help='simulation frames (0.05 s each) between video frames')
    ap.add_argument('--chase-cam', action='store_true', help='also render a camera following behind the vehicle')
    ap.add_argument('--video-quality', type=int, default=90)
    ap.add_argument('--latency-replay', default=None,
                    help="decisions.json of an earlier run: charge exactly the delays that run charged (by simulation "
                         "frame) instead of measuring this machine's wall clock, so a delay-charged roll reproduces")
    ap.add_argument('--pick', choices=['model', 'random'], default='model',
                    help="'random' keeps the whole loop identical but picks a candidate at random: the control "
                         'for whether the risk model is doing anything inside the continuous loop')
    ap.add_argument('--keep-current', action='store_true',
                    help='score the part of the route the vehicle is already on alongside the fresh candidates')
    ap.add_argument('--switch-margin', type=float, default=0.0,
                    help='with --keep-current, switch away only if the best candidate beats the current route by '
                         'this much in route logit')
    ap.add_argument('--reachable-speed', action='store_true',
                    help='clip each candidate speed command to the acceleration cone from the current speed; '
                         'see nav_online for why this is off by default')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if (out / 'mission_outcome.json').exists():
        print('exists', out); return
    source = Path(a.source_root).resolve()
    sys.path.insert(0, str(source / 'src'))
    spec = importlib.util.spec_from_file_location('frozen_runner', source / 'scripts/traverse_fdm_rgbd_diverse_chrono.py')
    fr = importlib.util.module_from_spec(spec); sys.modules['frozen_runner'] = fr; spec.loader.exec_module(fr)
    import gen_mission_runner as GMR      # LegStop: the frozen per-leg stop policy, unchanged
    import gen_planner as P
    import nav_online as NAV
    import torch
    torch.set_num_threads(a.torch_threads)
    import pychrono as chrono
    import pychrono.vehicle as veh
    from nedm.hmmwv_data import WHEEL_SPECS, capture_row
    from nedm.training.constants import STATE_FIELD_PRESETS
    from nedm.traverse.layout import EpisodeLayout
    from nedm.traverse.scene import RenderSpec, build_config, build_scene
    from nedm.traverse.terrain import TerrainMap

    m = json.load(open(a.mission))
    arena = (source / m['arena']).resolve()
    tmap = TerrainMap.from_dir(arena)
    goals = [np.asarray(g, float) for g in m['goals']]
    radius = float(m.get('goal_radius_m', 2.5))
    layout = EpisodeLayout.from_json(m['layout'])
    config = build_config(arena, (*layout.start_xy, float(tmap.height(*layout.start_xy)) + .75), layout.start_yaw)
    config['chrono_data_root'] = str(Path(a.chrono_data).resolve())
    config['vehicle_data_root'] = str(Path(a.chrono_data).resolve() / 'vehicle')
    nav = NAV.Navigator(a.models or os.environ['NAV_MODELS'], margin_m=a.margin_m,
                        mask=not a.no_mask, reachable=a.reachable_speed, device=a.device,
                        sense_radius_m=a.sense_radius_m)
    want_rgb = (a.save_frames or bool(a.video_dir)) if a.rgb == 'auto' else (a.rgb == 'on')
    if a.video_dir and not want_rgb:
        raise SystemExit('--video-dir needs the RGB camera')
    if not want_rgb and any(c in nav.channels for c in ('R', 'G', 'B')):
        raise SystemExit(f'checkpoints read colour channels {nav.channels}; --rgb off would change their input')
    render = RenderSpec(width=NAV.CAMERA['width'], height=NAV.CAMERA['height'], cam_height_m=NAV.CAMERA['cam_height_m'],
                        hfov_rad=NAV.CAMERA['hfov_rad'], max_depth_m=NAV.CAMERA['max_depth_m'],
                        plan_markers=False, light_elevation_deg=45.0, with_rgb=want_rgb)
    scene = build_scene(config, layout, tmap, arena, plan=None, render=render)
    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
    vehicle = hmmwv.GetVehicle(); engine = vehicle.GetEngine()
    tire_radii = {n: float(vehicle.GetTire(ax, sd).GetRadius()) for n, ax, sd in WHEEL_SPECS}
    fields = STATE_FIELD_PRESETS['tire_normal_force_omega_pt']
    dt = float(config['simulation']['step_size_s']); substeps = int(round(DT / dt))

    replay = None
    if a.latency_replay:
        replay = {int(d['frame']): float(d['latency_charged_s']) for d in json.load(open(a.latency_replay))
                  if 'latency_charged_s' in d}
    replay_misses = []
    rng = np.random.default_rng(int(hashlib.md5((m['id'] + a.mode + str(a.period)).encode()).hexdigest()[:8], 16))
    frames_dir = out / 'frames'
    if a.save_frames:
        frames_dir.mkdir(exist_ok=True)

    blank_rgb = np.zeros((NAV.CAMERA['height'], NAV.CAMERA['width'], 3), np.uint8)

    # Optional chase camera, rigidly attached behind and above the chassis. Sensors add no bodies, so it cannot
    # change the dynamics; it only has to be drained on every manager.Update() like the other taps.
    chase_tap = None
    if a.video_dir and a.chase_cam:
        import pychrono.sensor as sens
        from nedm.traverse.scene import _rgb_tap
        # attached to the static terrain body and re-posed before every render from the vehicle's position and
        # heading only: a camera rigidly fixed to the chassis pitches with it and dips under the ground on crests
        chase = sens.ChCameraSensor(scene.patch_body, 1.0 / dt,
                                    chrono.ChFramed(chrono.ChVector3d(0.0, 0.0, 100.0), chrono.QuatFromAngleY(0.30)),
                                    960, 540, math.radians(62.0))
        chase.SetName('chase_rgb'); chase.SetLag(0.0); chase.SetCollectionWindow(0.0)
        chase.PushFilter(sens.ChFilterRGBA8Access())
        scene.manager.AddSensor(chase)
        chase_tap = _rgb_tap(chase)
    video_dir = Path(a.video_dir) if a.video_dir else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)
    video_index = []
    last_render = {'frame': None}
    chase_state = {'yaw': None}

    def place_chase():
        if chase_tap is None:
            return
        p, cz = pose_now()
        yaw = float(p[2]) if chase_state['yaw'] is None else chase_state['yaw']
        dyaw = (float(p[2]) - yaw + math.pi) % (2 * math.pi) - math.pi
        yaw += 0.35 * dyaw                                   # light smoothing of the heading only
        chase_state['yaw'] = yaw
        back, up = 13.0, 5.5
        cx, cy = p[0] - back * math.cos(yaw), p[1] - back * math.sin(yaw)
        lim = lambda q: float(np.clip(q, -39.9, 39.9))       # the heightmap ends at the arena edge
        ground = max(float(tmap.height(lim(cx), lim(cy))), float(tmap.height(lim(p[0]), lim(p[1]))))
        z = max(cz + up, ground + 4.0)
        pitch = math.atan2(z - (cz + 0.5), back)
        chase.SetOffsetPose(chrono.ChFramed(chrono.ChVector3d(cx, cy, z),
                                            chrono.QuatFromAngleZ(yaw) * chrono.QuatFromAngleY(pitch)))

    def render_all():
        """One manager.Update() per simulated instant, every tap drained. A second Update at the same simulated time
        would not trigger the cameras (they are due only once time has advanced), so a decision and a video frame
        at the same instant share one render."""
        if last_render['frame'] == frame:
            return last_render
        t0 = time.perf_counter()
        place_chase()
        scene.manager.Update()
        rgb = scene.rgb_tap.take() if want_rgb else blank_rgb
        depth = scene.depth_tap.take()
        ch = chase_tap.take() if chase_tap is not None else None
        last_render.update(frame=frame, rgb=rgb, depth=depth, chase=ch, s=time.perf_counter() - t0)
        return last_render

    def sense():
        R = render_all()
        return R['rgb'], R['depth'], R['s']

    def capture_video():
        if video_dir is None or frame % a.video_every:
            return
        from PIL import Image
        R = render_all()
        Image.fromarray(R['rgb']).save(video_dir / f'ov_{frame:06d}.jpg', quality=a.video_quality)
        if R.get('chase') is not None:
            Image.fromarray(R['chase']).save(video_dir / f'ch_{frame:06d}.jpg', quality=a.video_quality)
        p, _ = pose_now()
        video_index.append({'frame': frame, 'pose': p.tolist(), 'goal_index': k,
                            'active_route_id': route.get('_rid') if isinstance(route, dict) else None,
                            'speed': float(vehicle.GetSpeed())})

    def build_follower(route, fallback_z):
        """Chrono path follower on the chosen route; steering only (throttle/brake come from SpeedPI)."""
        wp = np.asarray(route['waypoints'], float); st = np.asarray(route['stations'], float)
        if a.path_heights == 'sensed':
            zz, frac = NAV.path_heights(wp, fallback_z)
        else:
            zz, frac = tmap.height(wp[:, 0], wp[:, 1]), float('nan')
        pts = chrono.vector_ChVector3d(); last = -10.
        for (x, y), s, z in zip(wp, st, zz):
            if s - last < 2. and s != st[-1]:
                continue
            last = s
            pts.append(chrono.ChVector3d(float(x), float(y), float(z) + .5))
        d = veh.ChPathFollowerDriver(vehicle, chrono.ChBezierCurve(pts), 'route', float(route['speeds'][0]))
        d.GetSteeringController().SetLookAheadDistance(5.)
        d.GetSteeringController().SetGains(.8, 0., 0.)
        d.GetSpeedController().SetGains(.6, .05, 0.)
        d.Initialize()
        return d, frac

    def pose_now():
        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        return np.array([ref.GetPos().x, ref.GetPos().y, ref.GetRot().GetCardanAnglesZYX().z]), float(ref.GetPos().z)

    # ---------------- settle (frozen: 0.8 s, desired speed 0, steering straight) ----------------
    seed_route = P.base_route([layout.start_xy[0], layout.start_xy[1], layout.start_yaw], goals[0])
    pts = chrono.vector_ChVector3d(); last = -10.
    for (x, y), s in zip(np.asarray(seed_route['waypoints']), np.asarray(seed_route['stations'])):
        if s - last < 2. and s != seed_route['stations'][-1]:
            continue
        last = s
        pts.append(chrono.ChVector3d(float(x), float(y), float(tmap.height(x, y)) + .5))
    driver = veh.ChPathFollowerDriver(vehicle, chrono.ChBezierCurve(pts), 'settle', 0.)
    driver.GetSteeringController().SetLookAheadDistance(5.)
    driver.GetSteeringController().SetGains(.8, 0., 0.)
    driver.GetSpeedController().SetGains(.6, .05, 0.)
    driver.Initialize()
    for _ in range(int(round(fr.SETTLE_S / DT))):
        driver.SetDesiredSpeed(0.)
        for _ in range(substeps):
            ts = float(system.GetChTime())
            driver.Synchronize(ts); inputs = driver.GetInputs(); inputs.m_steering = 0.
            terrain.Synchronize(ts); hmmwv.Synchronize(ts, inputs, terrain)
            driver.Advance(dt); terrain.Advance(dt); hmmwv.Advance(dt)

    # ---------------- drive ----------------
    pid = NAV.SpeedPI()
    decisions, legs, S_, A_, Pz, Lg, Rt = [], [], [], [], [], [], []
    k = 0; frame = 0; leg_start = 0; stop = GMR.LegStop()
    leg_poses, leg_actions, leg_parked = [], [], []
    route = None; driver = None; prev_steer = 0.; wp_i = 0
    pending = None; next_decision_frame = 0; route_id = -1; last_decision_frame = -1000
    status = None; wall0 = time.time()
    cur = {'goal_index': 0, 'start_frame': 0}
    n_skipped = 0

    def decide(tag):
        nonlocal route_id, n_skipped, last_decision_frame
        pose, chassis_z = pose_now()
        dist = float(np.linalg.norm(pose[:2] - goals[k]))
        if tag == 'periodic' and dist < a.min_replan_dist_m:
            n_skipped += 1
            return None
        rgb, depth, render_s = sense()
        obs = nav.observe(rgb, depth, pose=pose)
        v_now = float(vehicle.GetSpeed())
        aim = goals[k]
        r, info = nav.decide(pose, aim, rng, v_now=v_now, pick=a.pick,
                             current=(route if (a.keep_current and tag == 'periodic') else None),
                             switch_margin=a.switch_margin)
        if r is None:
            fb = NAV.fallback_base(pose, goals[k])
            if fb is not None:
                r, info2 = nav.decide(pose, aim, rng, v_now=v_now, base=fb, pick=a.pick,
                                      fixed_speed=a.fallback_speed_mps)
                info = {**info, **info2, 'used_fallback_base': True}
        if r is None:
            q = NAV.surrogate_goal(pose, goals[k])
            if q is not None:
                aim = q
                r, info2 = nav.decide(pose, aim, rng, v_now=v_now, base=NAV.any_base(pose, q), pick=a.pick,
                                      fixed_speed=a.fallback_speed_mps)
                info2['surrogate_goal'] = [float(q[0]), float(q[1])]
                info2['surrogate_for_goal'] = k
                info = {**info, **info2}
        if r is None:
            q, sr = NAV.straight_ahead(pose)
            if q is not None:
                aim = q
                r, info2 = nav.decide(pose, aim, rng, v_now=v_now, base=sr, pick=a.pick,
                                      fixed_speed=a.fallback_speed_mps)
                info2['straight_ahead'] = [float(q[0]), float(q[1])]
                info = {**info, **info2}
        leak = NAV.mask_leak(pose, a.margin_m, tmap) if not a.no_mask else {}
        rec = {'frame': frame, 'sim_s': frame * DT, 'goal_index': k, 'trigger': tag,
               'aim_xy': [float(aim[0]), float(aim[1])],
               'pose': pose.tolist(), 'v_now': v_now, 'dist_to_goal_m': dist,
               'render_s': render_s, **obs, **{f'leak_{x}': y for x, y in leak.items()},
               **{x: y for x, y in info.items() if x != 'timing'},
               'timing': info.get('timing', {}),
               'sense_to_plan_s': render_s + obs['backproject_s'] + info.get('plan_s', 0.0)}
        if a.save_frames and r is not None:
            np.savez_compressed(frames_dir / f'dec_{len(decisions):04d}.npz',
                                z=nav.grid['z'].astype(np.float16), cover=nav.grid['cover'].astype(np.uint8),
                                rgb=(nav.grid['rgb'] * 255).astype(np.uint8), pose=pose,
                                route=np.asarray(r['waypoints'], np.float32),
                                speeds=np.asarray(r['speeds'], np.float32))
        decisions.append(rec)
        last_decision_frame = frame
        print(f"  dec {len(decisions):4d} t={frame*DT:7.2f}s goal {k} {tag:8s} v={v_now:4.1f} d={dist:5.1f} "
              f"pick={info.get('index', -1):3d} risk={info.get('risk', float('nan')):.4f} "
              f"inval={info.get('invalid_fraction', float('nan')):.3f} lat={rec['sense_to_plan_s']:.2f}s "
              f"(render {render_s:.2f} bp {obs['backproject_s']:.2f} plan {info.get('plan_s', 0):.2f})", flush=True)
        if r is None:
            return 'no_route'
        route_id += 1
        rec['route_id'] = route_id
        r['_rid'] = route_id
        Rt.append(route_json(r) | {'route_id': route_id, 'frame': frame, 'goal_index': k})
        return (r, chassis_z, rec['sense_to_plan_s'])

    first = decide('leg')
    if first is None or first == 'no_route':
        dump(out / 'mission_outcome.json', {'mission': m['id'], 'mode': a.mode, 'period_s': a.period,
                                            'status': 'no_route_leg0', 'goals_reached': 0, 'success': False,
                                            'decisions': decisions})
        return
    route, chassis_z, _ = first
    driver, zfrac = build_follower(route, chassis_z - .75)
    xy, spd = np.asarray(route['waypoints'], float), np.asarray(route['speeds'], float)
    cur['route_id'] = route_id
    next_decision_frame = frame + int(round(a.period / DT))
    capture_video()

    while True:
        pose, chassis_z = pose_now()
        pos = pose[:2]
        wp_i = fr.nearest_index(xy, pos, wp_i)
        at_end = wp_i >= len(xy) - 2 and np.linalg.norm(pos - xy[-1]) < 3.
        target = 0. if at_end else float(spd[wp_i])
        for sub in range(substeps):
            ts = float(system.GetChTime())
            driver.Synchronize(ts); inputs = driver.GetInputs()
            prev_steer = float(np.clip(inputs.m_steering, prev_steer - 2. * dt, prev_steer + 2. * dt))
            inputs.m_steering = prev_steer
            inputs.m_throttle, inputs.m_braking = pid.throttle, pid.braking
            terrain.Synchronize(ts); hmmwv.Synchronize(ts, inputs, terrain)
            if sub == 0:
                row = capture_row(hmmwv, terrain, 'nav_v1', 'follower', m['id'], 'development', frame, ts,
                                  inputs, include_tires=True, tire_radii=tire_radii)
                row['engine_motor_speed_radps'] = float(engine.GetMotorSpeed())
                row['engine_motorshaft_torque_nm'] = float(engine.GetOutputMotorshaftTorque())
                st_ = np.array([float(row[f]) for f in fields], np.float32)
                ps_ = np.array([row['pos_x_m'], row['pos_y_m'], row['yaw_rad']], np.float64)
                ac_ = np.array([inputs.m_steering, inputs.m_throttle, inputs.m_braking], np.float32)
                S_.append(st_); A_.append(ac_); Pz.append(ps_); Lg.append(k)
                leg_poses.append(ps_); leg_actions.append(ac_); leg_parked.append(bool(at_end))
            pid.advance(float(vehicle.GetSpeed()), target, dt)
            driver.Advance(dt); terrain.Advance(dt); hmmwv.Advance(dt)
        frame += 1
        pose, chassis_z = pose_now()
        leg_elapsed = (frame - leg_start) * DT

        # ---- activate a pending (latency-charged) plan ----
        if pending is not None and frame >= pending['activate_frame']:
            route = pending['route']
            driver, zfrac = build_follower(route, chassis_z - .75)
            xy, spd = np.asarray(route['waypoints'], float), np.asarray(route['speeds'], float)
            wp_i = 0; pending = None

        if abs(float(vehicle.GetRoll())) > math.radians(60.) or abs(float(vehicle.GetPitch())) > math.radians(60.):
            status = 'rollover'
        elif np.linalg.norm(pose[:2] - goals[k]) <= radius:
            cur.update(status='goal_reached', elapsed_s=leg_elapsed, end_frame=frame)
            legs.append(cur)
            if k == len(goals) - 1:
                status = 'mission_complete'
            else:
                k += 1
                d = decide('leg')
                cur = {'goal_index': k, 'start_frame': frame, 'start_pose': pose.tolist()}
                if d is None or d == 'no_route':
                    cur.update(status='no_route', elapsed_s=0.); legs.append(cur); status = 'no_route'
                else:
                    route, cz, _ = d
                    driver, zfrac = build_follower(route, cz - .75)
                    xy, spd = np.asarray(route['waypoints'], float), np.asarray(route['speeds'], float)
                    wp_i = 0; leg_start = frame; stop = GMR.LegStop(); pending = None
                    cur['route_id'] = route_id
                    leg_poses, leg_actions, leg_parked = [], [], []
                    next_decision_frame = frame + int(round(a.period / DT))
        else:
            s = stop.check(leg_elapsed, leg_poses, leg_actions, leg_parked, pose)
            if s is not None:
                status = s
            elif leg_elapsed + 1e-9 >= a.leg_horizon_s:
                status = 'timeout'
            elif frame * DT >= a.mission_horizon_s:
                status = 'mission_timeout'
            elif (wp_i >= len(xy) - 2 and np.linalg.norm(pose[:2] - xy[-1]) < 3.
                  and frame - last_decision_frame > 20 and pending is None):
                # the active route ended short of the waypoint (it aimed at a surrogate goal, or the follower
                # drifted): plan again instead of parking there. The stop policy and the horizons above still
                # apply, so this cannot loop forever.
                d = decide('reroute')
                if isinstance(d, tuple):
                    route, cz, _ = d
                    driver, zfrac = build_follower(route, cz - .75)
                    xy, spd = np.asarray(route['waypoints'], float), np.asarray(route['speeds'], float)
                    wp_i = 0
                else:
                    status = 'no_route_reroute'
        if status is not None:
            if status not in ('mission_complete', 'no_route'):
                cur.update(status=status, elapsed_s=leg_elapsed, end_frame=frame); legs.append(cur)
            break

        # ---- periodic replanning ----
        if a.mode == 'periodic' and pending is None and frame >= next_decision_frame:
            next_decision_frame = frame + int(round(a.period / DT))
            d = decide('periodic')
            if isinstance(d, tuple):
                r, cz, wall = d
                if replay is not None and a.latency_s != 0:
                    if frame in replay:
                        lat = replay[frame]
                    else:
                        replay_misses.append(frame)
                        lat = wall - decisions[-1]['render_s']
                elif a.latency_s == 0:
                    lat = 0.0
                elif a.latency_s > 0:
                    lat = a.latency_s
                elif a.latency_s == -1:
                    lat = wall
                else:
                    lat = wall - decisions[-1]['render_s']
                nlat = int(math.ceil(lat / DT - 1e-9))
                if nlat <= 0:
                    route = r
                    driver, zfrac = build_follower(route, cz - .75)
                    xy, spd = np.asarray(route['waypoints'], float), np.asarray(route['speeds'], float)
                    wp_i = 0
                else:
                    pending = {'route': r, 'activate_frame': frame + nlat}
                decisions[-1]['latency_charged_s'] = float(nlat * DT)
        capture_video()

    S_, A_, Pz, Lg = np.asarray(S_), np.asarray(A_), np.asarray(Pz), np.asarray(Lg, np.int16)
    np.savez_compressed(out / 'trajectory.npz', state=S_, action=A_, pose=Pz, leg=Lg, fields=np.asarray(fields))
    dump(out / 'decisions.json', decisions)
    dump(out / 'routes.json', Rt)
    import platform
    import pychrono.sensor as _sens
    render_backend = 'OptiX' if hasattr(_sens, 'ChOptixSensor') else 'Vulkan RT'
    if video_dir is not None:
        dump(video_dir / 'index.json', {'every_frames': a.video_every, 'dt': DT,
                                        'camera': dict(NAV.CAMERA, backend=render_backend + ' (' + platform.node() + ')'),
                                        'chase_cam': chase_tap is not None, 'frames': video_index})
    vx, thr = S_[:, 0], A_[:, 1]
    s0 = 20
    back = (vx < -0.10) & (thr > 0.3)
    idx = np.arange(len(vx))
    slide = (back | (vx < -0.30)) & (idx >= s0)
    for L in legs:
        a0, a1 = L.get('start_frame', 0), L.get('end_frame', len(vx))
        seg = (idx >= max(a0, s0)) & (idx < a1)
        L['slid'] = bool(slide[seg].any())
        L['back_s'] = float(back[seg].sum() * DT)
        L['max_tilt_deg'] = float(np.degrees(np.abs(S_[seg, 2:4])).max()) if seg.any() else 0.
    reached = sum(1 for L in legs if L.get('status') == 'goal_reached')
    dec_ok = [d for d in decisions if 'plan_s' in d]
    lat = np.array([d['sense_to_plan_s'] for d in dec_ok]) if dec_ok else np.zeros(1)
    dump(out / 'mission_outcome.json', {
        'mission': m['id'], 'arena': m['arena'], 'mode': a.mode, 'period_s': a.period,
        'latency_s_setting': a.latency_s, 'path_heights': a.path_heights, 'mask': not a.no_mask,
        'rgb_rendered': want_rgb, 'torch_threads': a.torch_threads, 'device': a.device,
        'video_dir': a.video_dir, 'chase_cam': chase_tap is not None,
        'render_backend': render_backend, 'host': platform.node(),
        'latency_replay': a.latency_replay, 'latency_replay_misses': replay_misses,
        'route_reversal_limit_deg': NAV.MAX_STEP_TURN_DEG,
        'reachable_speed_clip': a.reachable_speed, 'sense_radius_m': a.sense_radius_m, 'pick': a.pick,
        'fallback_speed_mps': a.fallback_speed_mps,
        'keep_current': a.keep_current,
        'switch_margin': a.switch_margin,
        'status': status, 'goals_reached': reached, 'n_goals': len(goals),
        'success': status == 'mission_complete', 'total_time_s': len(vx) * DT,
        'distance_m': float(np.linalg.norm(np.diff(Pz[:, :2], axis=0), axis=1).sum()),
        'any_slide': bool(slide.any()), 'back_s': float(back[s0:].sum() * DT),
        'max_tilt_deg': float(np.degrees(np.abs(S_[s0:, 2:4])).max()) if len(vx) > s0 else 0.,
        'n_decisions': len(decisions), 'n_decisions_skipped_near_goal': n_skipped,
        'sense_to_plan_s': {'mean': float(lat.mean()), 'p50': float(np.percentile(lat, 50)),
                            'p95': float(np.percentile(lat, 95)), 'max': float(lat.max())},
        'legs': legs, 'wall_s': time.time() - wall0})
    print(json.dumps({'mission': m['id'], 'mode': a.mode, 'period': a.period, 'status': status,
                      'goals': f'{reached}/{len(goals)}', 'sim_s': round(len(vx) * DT, 1),
                      'dec': len(decisions), 'lat_p50': round(float(np.percentile(lat, 50)), 2),
                      'wall_s': round(time.time() - wall0)}), flush=True)


if __name__ == '__main__':
    main()
