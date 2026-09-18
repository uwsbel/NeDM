"""Re-drive one route on CRM soil locally (luffy, CUDA soil solver) with an OptiX chase camera -> frames for a video.

Physics = scripts/crm_collect.py unchanged (its optional hooks are used: sensors and visual shapes only, attached to
existing bodies). SPH particles are invisible to the sensor module from Python, so the camera sees the vehicle over a
visual-only mesh of the UNDEFORMED heightmap: ruts are not drawn; wheels sinking into the soil show as wheels cutting
below that surface.

  PYTHONPATH=/home/harry/chrono/build/bin /usr/bin/python3.12 scripts/crm_demo_render.py --case C --route R --out DIR \
      --title "..." --risk 0.004
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / 'src'))
import crm_collect as CC

ap = argparse.ArgumentParser()
ap.add_argument('--case', required=True); ap.add_argument('--route', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--title', default=''); ap.add_argument('--risk', type=float, default=None)
ap.add_argument('--config', default=str(ROOT / 'artifacts/traverse/crm_f104_v1/configs/crm_main.json'))
ap.add_argument('--no-render', action='store_true')
a = ap.parse_args()
out = Path(a.out); frames_dir = out / 'frames'; frames_dir.mkdir(parents=True, exist_ok=True)
S = {}
W, H = 960, 540


def font(size):
    for f in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf'):
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def terrain_obj(tmap, path):
    """Visual mesh in Chrono's frame: BMP samples on the patch edges (80/511 m pitch), image row 0 = +y."""
    h = np.flipud(tmap.height_grid); n = h.shape[0]; size = tmap.size_m
    xs = -size / 2 + np.arange(n) * size / (n - 1); ys = size / 2 - np.arange(n) * size / (n - 1)
    X, Y = np.meshgrid(xs, ys)
    v = np.column_stack([X.ravel(), Y.ravel(), h.ravel()])
    idx = np.arange(n * n).reshape(n, n) + 1
    a_, b_, c_, d_ = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.column_stack([a_, c_, b_]), np.column_stack([b_, c_, d_])])
    with open(path, 'w') as f:
        np.savetxt(f, v, fmt='v %.4f %.4f %.4f'); np.savetxt(f, faces, fmt='f %d %d %d')


def scene_hook(chrono, veh, hmmwv, system, terrain, tmap, route, case):
    import pychrono.sensor as sens
    from nedm.traverse.scene import _rgb_tap, SKY_RGB
    S.update(chrono=chrono, tmap=tmap, hmmwv=hmmwv, route=route, case=case)
    hmmwv.SetChassisVisualizationType(chrono.VisualizationType_MESH)
    hmmwv.SetWheelVisualizationType(chrono.VisualizationType_MESH)
    hmmwv.SetTireVisualizationType(chrono.VisualizationType_MESH)
    ground = terrain.GetGroundBody()
    obj = ROOT / 'artifacts/traverse/crm_f104_v1/demo_v1/terrain_visual.obj'
    if not obj.exists():
        terrain_obj(tmap, obj)
    mesh = chrono.ChTriangleMeshConnected.CreateFromWavefrontFile(str(obj), True, True)
    shape = chrono.ChVisualShapeTriangleMesh(); shape.SetMesh(mesh); shape.SetMutable(False)
    mat = chrono.ChVisualMaterial(); mat.SetDiffuseColor(chrono.ChColor(0.55, 0.45, 0.32)); mat.SetRoughness(0.9)
    shape.SetMaterial(0, mat) if shape.GetNumMaterials() else shape.AddMaterial(mat)
    ground.AddVisualShape(shape, chrono.ChFramed(chrono.VNULL, chrono.QUNIT))

    def marker(x, y, r, rgb, dz=0.25):
        sph = chrono.ChVisualShapeSphere(r); m = chrono.ChVisualMaterial()
        m.SetDiffuseColor(chrono.ChColor(*rgb)); m.SetEmissiveColor(chrono.ChColor(*[0.6 * c for c in rgb])); sph.SetMaterial(0, m)
        ground.AddVisualShape(sph, chrono.ChFramed(chrono.ChVector3d(float(x), float(y), float(tmap.height(x, y)) + dz), chrono.QUNIT))
    last = -9.
    for (x, y), s in zip(route['waypoints'], route['stations']):
        if s - last >= 1.5:
            marker(x, y, 0.13, (0.05, 0.85, 0.95)); last = s
    gx, gy = case['goal_xy']
    for k in range(24):
        ang = 2 * math.pi * k / 24
        marker(gx + 2.5 * math.cos(ang), gy + 2.5 * math.sin(ang), 0.2, (0.95, 0.15, 0.8), 0.35)
    if a.no_render:
        return
    manager = sens.ChSensorManager(system)
    manager.scene.SetAmbientLight(chrono.ChVector3f(0.35, 0.35, 0.38))
    manager.scene.AddDirectionalLight(chrono.ChColor(1.0, 0.95, 0.85), math.radians(40.0), math.radians(120.0))
    bg = sens.Background(); bg.mode = sens.BackgroundMode_SOLID_COLOR; bg.color_zenith = chrono.ChVector3f(*SKY_RGB)
    manager.scene.SetBackground(bg)
    cam = sens.ChCameraSensor(ground, 1000.0, chrono.ChFramed(chrono.ChVector3d(0, 0, 100.), chrono.QuatFromAngleY(0.3)), W, H, math.radians(62.0))
    cam.SetName('chase'); cam.SetLag(0.0); cam.SetCollectionWindow(0.0); cam.PushFilter(sens.ChFilterRGBA8Access())
    manager.AddSensor(cam)
    S.update(manager=manager, cam=cam, tap=_rgb_tap(cam), yaw=None)
    # minimap: terrain thumbnail + route
    h = np.flipud(tmap.height_grid); lo, hi = h.min(), h.max()
    g = ((h - lo) / (hi - lo) * 255).astype(np.uint8)
    thumb = Image.fromarray(g).resize((200, 200)).convert('RGB')
    thumb = Image.blend(thumb, Image.new('RGB', (200, 200), (150, 120, 80)), 0.35)
    d = ImageDraw.Draw(thumb)
    px = lambda x, y: (100 + x * 2.5, 100 - y * 2.5)
    d.line([px(x, y) for x, y in route['waypoints']], fill=(20, 220, 240), width=2)
    d.ellipse([px(gx, gy)[0] - 6, px(gx, gy)[1] - 6, px(gx, gy)[0] + 6, px(gx, gy)[1] + 6], outline=(240, 40, 200), width=2)
    S.update(thumb=thumb, px=px, track=[])


def frame_hook(frame, row, state, pose, action, desired_speed):
    if a.no_render:
        return
    chrono, tmap = S['chrono'], S['tmap']
    yaw = float(pose[2]) if S['yaw'] is None else S['yaw']
    yaw += 0.35 * ((float(pose[2]) - yaw + math.pi) % (2 * math.pi) - math.pi); S['yaw'] = yaw
    back, up = 12.0, 5.0
    cx, cy = pose[0] - back * math.cos(yaw), pose[1] - back * math.sin(yaw)
    lim = lambda q: float(np.clip(q, -39.9, 39.9))
    cz = float(row['pos_z_m'])
    z = max(cz + up, max(float(tmap.height(lim(cx), lim(cy))), float(tmap.height(lim(pose[0]), lim(pose[1])))) + 3.5)
    pitch = math.atan2(z - (cz + 0.5), back)
    S['cam'].SetOffsetPose(chrono.ChFramed(chrono.ChVector3d(cx, cy, z), chrono.QuatFromAngleZ(yaw) * chrono.QuatFromAngleY(pitch)))
    S['manager'].Update()
    rgb = S['tap'].take(timeout_s=120.0)
    img = Image.fromarray(rgb); d = ImageDraw.Draw(img, 'RGBA')
    d.rectangle([0, 0, W, 64], fill=(0, 0, 0, 150))
    d.text((12, 6), a.title, fill=(255, 255, 255), font=font(20))
    risk = '' if a.risk is None else f'planner risk {100 * a.risk:.1f} %    '
    slip = max(abs(float(row[f'{n}_slip_ratio'])) for n in ('tire_fl', 'tire_fr', 'tire_rl', 'tire_rr'))
    d.text((12, 34), f'{risk}t = {frame * 0.05:5.1f} s    speed {float(state[0]):4.1f} m/s (commanded {desired_speed:3.1f})    throttle {float(action[1]):.2f}'
                     f'    worst wheel slip ratio {slip:5.1f}', fill=(255, 235, 170), font=font(16))
    S['track'].append(S['px'](pose[0], pose[1]))
    mini = S['thumb'].copy(); md = ImageDraw.Draw(mini)
    if len(S['track']) > 1:
        md.line(S['track'], fill=(255, 255, 255), width=2)
    x, y = S['track'][-1]; md.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 60, 60))
    img.paste(mini, (W - 212, H - 212))
    img.save(frames_dir / f'f_{frame:05d}.jpg', quality=88)


if __name__ == '__main__':
    for stale in out.glob('*'):
        if stale.is_file(): stale.unlink()
    for stale in frames_dir.glob('*.jpg'): stale.unlink()
    CC.main(['--source-root', str(ROOT), '--case', a.case, '--route', a.route, '--out', str(out), '--chrono-data', str(ROOT / 'chrono/data'),
             '--crm-config', a.config, '--horizon-s', '120'], scene_hook=scene_hook, frame_hook=frame_hook)
