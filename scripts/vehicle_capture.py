"""One overhead RGB-D frame per planning decision, WITH the HMMWV present, after the frozen 0.8 s settle.

Same scene builder, physics config, driver and settle as the frozen collector's pre-drive phase
(scripts/traverse_fdm_rgbd_diverse_chrono.py: frames -16..-1 hold desired speed 0), and the same overhead camera as
the vehicle-free arena capture (1024x1024, 110 m, 47 deg, max range 180 m, light elevation 45 deg). Saves the raw
sensor frame plus the MEASURED vehicle pose at capture time - that pose is what the exclusion mask uses.

  python vehicle_capture.py --source-root SRC --case CASE.json --out DIR --chrono-data DATA
"""
import argparse, importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np

CAMERA = {'width': 1024, 'height': 1024, 'hfov_rad': math.radians(47.), 'cam_height_m': 110.,
          'depth_ray_scale': 1., 'max_depth_m': 180., 'backend': 'Vulkan_RT_lavapipe',
          'depth_measurement': 'Euclidean_ray_range_m', 'model_image_size': 512, 'elevation_scale_m': 10.,
          'observation_mode': 'One overhead RGB-D frame per planning decision, vehicle present, after the 0.8 s settle'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', required=True); ap.add_argument('--case', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--chrono-data', required=True)
    a = ap.parse_args()
    src = Path(a.source_root).resolve()
    sys.path.insert(0, str(src / 'src')); sys.path.insert(0, str(src / 'scripts'))
    spec = importlib.util.spec_from_file_location('frozen_runner', src / 'scripts/traverse_fdm_rgbd_diverse_chrono.py')
    fr = importlib.util.module_from_spec(spec); sys.modules['frozen_runner'] = fr; spec.loader.exec_module(fr)
    import pychrono as chrono
    import pychrono.vehicle as veh
    from nedm.hmmwv_data import WHEEL_SPECS, capture_row
    from nedm.training.constants import STATE_FIELD_PRESETS
    from nedm.traverse.layout import EpisodeLayout
    from nedm.traverse.scene import RenderSpec, build_config, build_scene
    from nedm.traverse.terrain import TerrainMap

    case = json.loads(Path(a.case).read_text())
    arena = (src / case['arena']).resolve()
    layout = EpisodeLayout.from_json(case['layout'])
    tmap = TerrainMap.from_dir(arena)
    config = build_config(arena, (*layout.start_xy, float(tmap.height(*layout.start_xy)) + .75), layout.start_yaw)
    config['chrono_data_root'] = str(Path(a.chrono_data).resolve())
    config['vehicle_data_root'] = str(Path(a.chrono_data).resolve() / 'vehicle')
    render = RenderSpec(width=CAMERA['width'], height=CAMERA['height'], cam_height_m=CAMERA['cam_height_m'],
                        hfov_rad=CAMERA['hfov_rad'], max_depth_m=CAMERA['max_depth_m'], plan_markers=False,
                        light_elevation_deg=45.0)
    scene = build_scene(config, layout, tmap, arena, plan=None, render=render)
    hmmwv, system, terrain = scene.hmmwv, scene.system, scene.terrain
    vehicle = hmmwv.GetVehicle(); engine = vehicle.GetEngine()
    tire_radii = {n: float(vehicle.GetTire(ax, sd).GetRadius()) for n, ax, sd in WHEEL_SPECS}
    fields = STATE_FIELD_PRESETS['tire_normal_force_omega_pt']
    route = case['settle_reference']
    driver = fr.make_driver(chrono, veh, vehicle, route, tmap)
    dt = float(config['simulation']['step_size_s']); substeps = int(round(fr.DT / dt))
    t0 = time.time()
    for frame in range(-int(round(fr.SETTLE_S / fr.DT)), 0):     # frozen settle: desired speed 0, steering straight
        driver.SetDesiredSpeed(0.)
        for _ in range(substeps):
            ts = float(system.GetChTime())
            driver.Synchronize(ts); inputs = driver.GetInputs(); inputs.m_steering = 0.
            terrain.Synchronize(ts); hmmwv.Synchronize(ts, inputs, terrain)
            driver.Advance(dt); terrain.Advance(dt); hmmwv.Advance(dt)
    ts = float(system.GetChTime())
    driver.Synchronize(ts); inputs = driver.GetInputs(); inputs.m_steering = 0.
    terrain.Synchronize(ts); hmmwv.Synchronize(ts, inputs, terrain)
    row = capture_row(hmmwv, terrain, 'vehicle_capture', 'follower', case['id'], 'development', 0, ts, inputs,
                      include_tires=True, tire_radii=tire_radii)
    row['engine_motor_speed_radps'] = float(engine.GetMotorSpeed())
    row['engine_motorshaft_torque_nm'] = float(engine.GetOutputMotorshaftTorque())
    state = np.array([float(row[f]) for f in fields], np.float32)
    pose = np.array([row['pos_x_m'], row['pos_y_m'], row['yaw_rad']], np.float64)
    ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
    chassis_z = float(ref.GetPos().z)
    scene.manager.Update()
    rgb, depth = scene.rgb_tap.take(), scene.depth_tap.take()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'observation.npz', rgb=rgb, depth_m=depth, pose=pose, state=state)
    from PIL import Image
    Image.fromarray(rgb).save(out / 'rgb.png')
    json.dump({'schema': 'vehicle_included_overhead_rgbd_v1', 'case_id': case['id'], 'arena': case['arena'],
               'camera': CAMERA, 'measured_pose_xy_yaw': pose.tolist(), 'chassis_ref_z_m': chassis_z,
               'settle_s': fr.SETTLE_S, 'vehicle_present': True,
               'valid_depth_fraction': float(np.mean(np.isfinite(depth) & (depth > 0) & (depth < CAMERA['max_depth_m']))),
               'wall_s': time.time() - t0}, open(out / 'observation.json', 'w'), indent=1)
    print(json.dumps({'case': case['id'], 'pose': pose.round(3).tolist(), 'wall_s': round(time.time() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
