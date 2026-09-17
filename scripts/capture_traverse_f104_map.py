#!/usr/bin/env python3
"""One static, vehicle-free RGB-D map and native height audit for exact f104.

Sensor pixels are retained unchanged. Terrain truth is saved separately for
geometry verification and never substituted for measured depth in model input.
"""
import argparse,hashlib,json,math,os,sys
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def run(a):
    assert os.environ.get('SLURM_JOB_ID'),'AMD allocation required'
    source=a.source_root.resolve();sys.path.insert(0,str(source/'src'));sys.path.insert(0,str(source/'scripts'))
    import pychrono as chrono
    import pychrono.vehicle as veh
    import pychrono.sensor as sens
    from nedm.traverse.scene import _rgb_tap,_depth_tap,overhead_camera_pose,SKY_RGB
    from nedm.traverse.terrain import TerrainMap
    from nedm.traverse.camera import CameraModel
    from nedm.traverse.fdm_diverse_data import encode_global_rgbd
    from traverse_fdm_rgbd_diverse_batch import runtime_fingerprint
    from PIL import Image
    arena=a.arena.resolve();meta=json.loads((arena/'arena_meta.json').read_text())
    assert (meta['size_m'],meta['height_min_m'],meta['height_max_m'])==(80.,-1.8,3.9)
    assert sha(arena/meta['bmp'])==a.bmp_sha256
    if a.out.exists():raise ValueError('Use a new map output directory')
    a.out.mkdir(parents=True)
    system=chrono.ChSystemSMC();system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    terrain=veh.RigidTerrain(system);mat=chrono.ChContactMaterialSMC()
    mat.SetFriction(.9);mat.SetRestitution(.01);mat.SetYoungModulus(2.e7)
    patch=terrain.AddPatch(mat,chrono.CSYSNORM,str(arena/meta['bmp']),80.,80.,-1.8,3.9)
    texture=source/'chrono/data/sensor/textures/grass_texture.jpg'
    if texture.is_file():patch.SetTexture(str(texture),40.,40.)
    else:patch.SetColor(chrono.ChColor(.42,.5,.32))
    terrain.Initialize();system.GetCollisionSystem().BindAll()
    tmap=TerrainMap.from_dir(arena)
    rng=np.random.default_rng(104);xy=rng.uniform(-38.,38.,(4096,2))
    native=np.asarray([terrain.GetHeight(chrono.ChVector3d(float(x),float(y),20.)) for x,y in xy])
    interpolated=tmap.height(xy[:,0],xy[:,1]);error=native-interpolated
    p95=float(np.quantile(np.abs(error),.95));assert p95<.05,'BMP/world orientation or height range mismatch'
    camera={'width':1024,'height':1024,'hfov_rad':math.radians(47.),'cam_height_m':110.,'depth_ray_scale':1.,
      'max_depth_m':180.,'backend':'Vulkan_RT_lavapipe','depth_measurement':'Euclidean_ray_range_m',
      'model_image_size':512,'elevation_scale_m':10.,'observation_mode':'One static vehicle-free global terrain RGB-D map; same physical terrain shared by all episodes'}
    manager=sens.ChSensorManager(system);manager.scene.SetAmbientLight(chrono.ChVector3f(.35,.35,.38))
    manager.scene.AddDirectionalLight(chrono.ChColor(1.,.95,.85),math.radians(45.),math.radians(120.))
    bg=sens.Background();bg.mode=sens.BackgroundMode_SOLID_COLOR;bg.color_zenith=chrono.ChVector3f(*SKY_RGB);manager.scene.SetBackground(bg)
    pose=overhead_camera_pose(camera['cam_height_m']);body=patch.GetGroundBody()
    rgb_camera=sens.ChCameraSensor(body,500.,pose,1024,1024,camera['hfov_rad']);rgb_camera.SetLag(0.);rgb_camera.SetCollectionWindow(0.);rgb_camera.PushFilter(sens.ChFilterRGBA8Access());manager.AddSensor(rgb_camera)
    depth_camera=sens.ChDepthCamera(body,500.,pose,1024,1024,camera['hfov_rad'],camera['max_depth_m']);depth_camera.SetLag(0.);depth_camera.SetCollectionWindow(0.);manager.AddSensor(depth_camera)
    rgb_tap,depth_tap=_rgb_tap(rgb_camera),_depth_tap(depth_camera)
    manager.Update();rgb=rgb_tap.take();depth=depth_tap.take()
    rgbd=encode_global_rgbd(rgb,depth,camera)
    assert np.isfinite(rgbd).all()
    np.savez_compressed(a.out/'observation.npz',rgb=rgb,depth_m=depth,rgbd=rgbd)
    Image.fromarray(rgb).save(a.out/'rgb.png')
    np.savez_compressed(a.out/'native_height_audit.npz',xy=xy,native_height_m=native,bmp_bilinear_height_m=interpolated)
    cm=CameraModel(width=1024,height=1024,hfov_rad=camera['hfov_rad'],cam_height_m=110.)
    corners=[cm.world_to_pixel(x,y,z) for x in(-40.,40.) for y in(-40.,40.) for z in(-1.8,3.9)]
    assert all(0<=u<1024 and 0<=v<1024 for u,v in corners),'Arena not fully visible'
    dump(a.out/'observation.json',{'schema':'fdm_f104_static_rgbd_v1','camera':camera,'arena_bmp_sha256':a.bmp_sha256,
      'arena_meta_sha256':sha(arena/'arena_meta.json'),'observation_sha256':sha(a.out/'observation.npz'),
      'native_geometry':{'length_m':80.,'width_m':80.,'height_min_m':-1.8,'height_max_m':3.9,'sample_count':4096,'p95_abs_error_m':p95,'rmse_m':float(np.sqrt(np.mean(error**2)))},
      'vehicle_or_goal_markers_rendered':False,'rgbd_input_contains_authored_height_samples':False,
      'all_corners_visible':True,'native_height_samples_are_audit_only':True,'capture_script_sha256':sha(__file__),
      'source_root':str(source),'source_manifest_sha256':sha(source/'source_manifest.json'),
      'runtime':runtime_fingerprint(a.chrono_data),'slurm_job_id':os.environ['SLURM_JOB_ID'],
      'join_contract':'Join this static map by exact BMP/meta/material/camera hashes. Each episode supplies its own measured pose/state/history and reference; do not copy this map capture into an episode anchor.'})
    print(json.dumps({'map':str(a.out),'native_height_p95_m':p95,'all_corners_visible':True}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);p.add_argument('--arena',type=Path,required=True);p.add_argument('--bmp-sha256',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--chrono-data',required=True);run(p.parse_args())
