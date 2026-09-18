#!/usr/bin/env python3
"""Local OptiX twin of scripts/sensor_capture_map.py: one static, vehicle-free overhead RGB-D capture of an arena.
Same camera, lights, rigid-mesh terrain and encoding; cluster-only provenance (SLURM id, source manifest, runtime
fingerprint) replaced by local provenance. The rigid mesh is a render-only proxy of the UNDEFORMED surface."""
import argparse, hashlib, json, math, platform, subprocess, sys
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, default=Path('/home/harry/NeDM-traverse_mppi'))
    ap.add_argument('--arena', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--grass-texture', action='store_true',
                    help='default is the flat colour (.42,.5,.32), which is what every AMD capture rendered')
    a = ap.parse_args()
    repo = a.repo.resolve(); sys.path.insert(0, str(repo / 'src')); sys.path.insert(0, str(repo / 'scripts'))
    import pychrono as chrono, pychrono.vehicle as veh, pychrono.sensor as sens
    from nedm.traverse.scene import _rgb_tap, _depth_tap, overhead_camera_pose, SKY_RGB
    from nedm.traverse.terrain import TerrainMap
    from nedm.traverse.camera import CameraModel
    from nedm.traverse.fdm_diverse_data import encode_global_rgbd
    from PIL import Image
    assert hasattr(sens, 'ChOptixSensor'), 'this pychrono is not an OptiX build'
    arena = a.arena.resolve(); meta = json.loads((arena / 'arena_meta.json').read_text())
    assert meta['size_m'] == 80.
    hmin, hmax = float(meta['height_min_m']), float(meta['height_max_m'])
    if a.out.exists(): raise ValueError('Use a new map output directory')
    a.out.mkdir(parents=True)
    system = chrono.ChSystemSMC(); system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    terrain = veh.RigidTerrain(system); mat = chrono.ChContactMaterialSMC()
    mat.SetFriction(.9); mat.SetRestitution(.01); mat.SetYoungModulus(2.e7)
    patch = terrain.AddPatch(mat, chrono.CSYSNORM, str(arena / meta['bmp']), 80., 80., hmin, hmax)
    texture = repo / 'chrono/data/sensor/textures/grass_texture.jpg'
    if a.grass_texture and texture.is_file(): patch.SetTexture(str(texture), 40., 40.)
    else: patch.SetColor(chrono.ChColor(.42, .5, .32))
    terrain.Initialize(); system.GetCollisionSystem().BindAll()
    tmap = TerrainMap.from_dir(arena)
    rng = np.random.default_rng(104); xy = rng.uniform(-38., 38., (4096, 2))
    native = np.asarray([terrain.GetHeight(chrono.ChVector3d(float(x), float(y), 20.)) for x, y in xy])
    interp = tmap.height(xy[:, 0], xy[:, 1]); err = native - interp
    p95 = float(np.quantile(np.abs(err), .95)); assert p95 < .05, 'BMP/world orientation or height range mismatch'
    camera = {'width': 1024, 'height': 1024, 'hfov_rad': math.radians(47.), 'cam_height_m': 110., 'depth_ray_scale': 1.,
              'max_depth_m': 180., 'backend': 'OptiX_luffy_RTX5090_fork_fovfix_9e1a0448b',
              'depth_measurement': 'Euclidean_ray_range_m', 'model_image_size': 512, 'elevation_scale_m': 10.,
              'observation_mode': 'One static vehicle-free global terrain RGB-D map of the UNDEFORMED surface (rigid-mesh proxy)'}
    manager = sens.ChSensorManager(system); manager.scene.SetAmbientLight(chrono.ChVector3f(.35, .35, .38))
    manager.scene.AddDirectionalLight(chrono.ChColor(1., .95, .85), math.radians(45.), math.radians(120.))
    bg = sens.Background(); bg.mode = sens.BackgroundMode_SOLID_COLOR
    bg.color_zenith = chrono.ChVector3f(*SKY_RGB); manager.scene.SetBackground(bg)
    pose = overhead_camera_pose(camera['cam_height_m']); body = patch.GetGroundBody()
    rgb_cam = sens.ChCameraSensor(body, 500., pose, 1024, 1024, camera['hfov_rad'])
    rgb_cam.SetLag(0.); rgb_cam.SetCollectionWindow(0.); rgb_cam.PushFilter(sens.ChFilterRGBA8Access()); manager.AddSensor(rgb_cam)
    dep_cam = sens.ChDepthCamera(body, 500., pose, 1024, 1024, camera['hfov_rad'], camera['max_depth_m'])
    dep_cam.SetLag(0.); dep_cam.SetCollectionWindow(0.); manager.AddSensor(dep_cam)
    rgb_tap, dep_tap = _rgb_tap(rgb_cam), _depth_tap(dep_cam)
    manager.Update(); rgb = rgb_tap.take(timeout_s=120.); depth = dep_tap.take(timeout_s=120.)
    rgbd = encode_global_rgbd(rgb, depth, camera); assert np.isfinite(rgbd).all()
    np.savez_compressed(a.out / 'observation.npz', rgb=rgb, depth_m=depth, rgbd=rgbd)
    Image.fromarray(rgb).save(a.out / 'rgb.png')
    np.savez_compressed(a.out / 'native_height_audit.npz', xy=xy, native_height_m=native, bmp_bilinear_height_m=interp)
    cm = CameraModel(width=1024, height=1024, hfov_rad=camera['hfov_rad'], cam_height_m=110.)
    corners = [cm.world_to_pixel(x, y, z) for x in (-40., 40.) for y in (-40., 40.) for z in (hmin, hmax)]
    assert all(0 <= u < 1024 and 0 <= v < 1024 for u, v in corners), 'Arena not fully visible'
    git = lambda d: subprocess.run(['git', '-C', str(d), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    (a.out / 'observation.json').write_text(json.dumps({
        'schema': 'f104_family_static_rgbd_v1', 'arena': arena.name, 'camera': camera,
        'arena_bmp_sha256': sha(arena / meta['bmp']), 'arena_meta_sha256': sha(arena / 'arena_meta.json'),
        'observation_sha256': sha(a.out / 'observation.npz'),
        'native_geometry': {'length_m': 80., 'width_m': 80., 'height_min_m': hmin, 'height_max_m': hmax,
                            'sample_count': 4096, 'p95_abs_error_m': p95, 'rmse_m': float(np.sqrt(np.mean(err ** 2)))},
        'vehicle_or_goal_markers_rendered': False, 'rgbd_input_contains_authored_height_samples': False,
        'all_corners_visible': True, 'native_height_samples_are_audit_only': True,
        'capture_script_sha256': sha(__file__), 'host': platform.node(), 'pychrono': chrono.__file__,
        'chrono_git': git('/home/harry/chrono'), 'repo_git': git(repo),
        'terrain_colour': 'grass_texture' if (a.grass_texture and texture.is_file()) else 'flat_0.42_0.5_0.32'},
        indent=2, allow_nan=False) + '\n')
    print(json.dumps({'map': str(a.out), 'native_height_p95_m': p95,
                      'valid_depth_fraction': float(np.mean(depth < camera['max_depth_m'] - 1e-6))}))

if __name__ == '__main__':
    main()
