"""Re-drive one route on CRM soil with the VSG run-time visualiser so the SOIL PARTICLES are visible (ruts, dig-in).

Runs under the conda `nedm` Python (pychrono 10.0.0: has VSG + the SPH visualisation plug-in; the source build used for
the collection has VSG switched off). Same collector loop / controller / soil settings via scripts/crm_collect.py, but a
different Chrono version than the collection, so trajectories are close, not identical. Needs a desktop display.

  DISPLAY=:1 /home/harry/miniconda3/envs/nedm/bin/python scripts/crm_demo_vsg.py --case C --route R --out DIR --title T --risk 0.004
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / 'src'))
import crm_collect as CC

ap = argparse.ArgumentParser()
ap.add_argument('--case', required=True); ap.add_argument('--route', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--title', default=''); ap.add_argument('--risk', type=float, default=None)
ap.add_argument('--config', default=str(ROOT / 'artifacts/traverse/crm_f104_v1/configs/crm_main.json'))
ap.add_argument('--color', default='height', choices=['height', 'velocity'])
ap.add_argument('--chase', type=float, nargs=2, default=[9.0, 3.0], metavar=('DIST', 'HEIGHT'))
ap.add_argument('--max-frames', type=int, default=0)
a = ap.parse_args()
out = Path(a.out); frames_dir = out / 'frames'; frames_dir.mkdir(parents=True, exist_ok=True)
S = {'rows': []}


def scene_hook(chrono, veh, hmmwv, system, terrain, tmap, route, case):
    import math
    import pychrono.fsi as fsi
    S.update(chrono=chrono, veh=veh, route=route, case=case)
    hmmwv.SetChassisVisualizationType(chrono.VisualizationType_MESH)
    hmmwv.SetWheelVisualizationType(chrono.VisualizationType_MESH)
    hmmwv.SetTireVisualizationType(chrono.VisualizationType_MESH)
    ground = terrain.GetGroundBody()

    def marker(x, y, r, rgb, dz):
        sph = chrono.ChVisualShapeSphere(r); sph.SetColor(chrono.ChColor(*rgb))
        ground.AddVisualShape(sph, chrono.ChFramed(chrono.ChVector3d(float(x), float(y), float(tmap.height(x, y)) + dz), chrono.QUNIT))
    last = -9.
    for (x, y), s in zip(route['waypoints'], route['stations']):
        if s - last >= 2.0:
            marker(x, y, 0.12, (0.05, 0.85, 0.95), 0.45); last = s
    gx, gy = case['goal_xy']
    for k in range(24):
        ang = 2 * math.pi * k / 24
        marker(gx + 2.5 * math.cos(ang), gy + 2.5 * math.sin(ang), 0.2, (0.95, 0.15, 0.8), 0.5)

    vis_fsi = fsi.ChSphVisualizationVSG(terrain.GetFsiSystemSPH())
    vis_fsi.EnableFluidMarkers(True); vis_fsi.EnableBoundaryMarkers(False); vis_fsi.EnableRigidBodyMarkers(False)
    if a.color == 'height':
        S['col'] = fsi.ParticleHeightColorCallback(-2.0, 4.0); vis_fsi.SetSPHColorCallback(S['col'], chrono.ChColormap.Type_BROWN)
    else:
        S['col'] = fsi.ParticleVelocityColorCallback(0.0, 3.0); vis_fsi.SetSPHColorCallback(S['col'], chrono.ChColormap.Type_KINDLMANN)
    vis = veh.ChWheeledVehicleVisualSystemVSG()
    vis.AttachVehicle(hmmwv.GetVehicle()); vis.AttachPlugin(vis_fsi)
    vis.SetWindowTitle('HMMWV on CRM soil'); vis.SetWindowSize(1280, 720); vis.SetWindowPosition(40, 40)
    vis.EnableSkyTexture() if hasattr(vis, 'EnableSkyTexture') else None
    vis.SetLightIntensity(1.0); vis.SetLightDirection(1.5 * chrono.CH_PI_2, chrono.CH_PI_4); vis.SetCameraAngleDeg(45)
    vis.SetChaseCamera(chrono.ChVector3d(0, 0, 1.2), a.chase[0], a.chase[1])
    vis_fsi.SetImageOutputDirectory(str(frames_dir)); vis_fsi.SetImageOutput(True)   # vis.WriteImageToFile segfaults in this build
    for name, arg in (('SetGuiVisibility', False), ('SetLogoVisible', False), ('SetBaseGuiVisibility', False)):
        if hasattr(vis, name): getattr(vis, name)(arg)
    vis.Initialize()
    S.update(vis=vis, vis_fsi=vis_fsi)


def frame_hook(frame, row, state, pose, action, desired_speed):
    veh, vis = S['veh'], S['vis']
    di = veh.DriverInputs(); di.m_steering, di.m_throttle, di.m_braking = float(action[0]), float(action[1]), float(action[2])
    t = frame * 0.05
    vis.Synchronize(t, di); vis.Advance(0.05)
    if not vis.Run():
        raise SystemExit('visualiser window closed')
    vis.Render()   # the plug-in writes frames/img_NNNNN.png on every render
    slip = max(abs(float(row[f'{n}_slip_ratio'])) for n in ('tire_fl', 'tire_fr', 'tire_rl', 'tire_rr'))
    S['rows'].append(dict(frame=frame, t=t, vx=float(state[0]), cmd=float(desired_speed), thr=float(action[1]), slip=slip,
                          x=float(pose[0]), y=float(pose[1])))
    if len(S['rows']) % 100 == 0:
        json.dump(S['rows'], open(out / 'overlay.json', 'w'))
    if a.max_frames and len(S['rows']) >= a.max_frames:
        json.dump(S['rows'], open(out / 'overlay.json', 'w'))
        raise SystemExit('max frames reached')


if __name__ == '__main__':
    for stale in out.glob('*'):
        if stale.is_file(): stale.unlink()
    for stale in frames_dir.glob('*'): stale.unlink()
    json.dump(dict(title=a.title, risk=a.risk), open(out / 'title.json', 'w'))
    try:
        CC.main(['--source-root', str(ROOT), '--case', a.case, '--route', a.route, '--out', str(out), '--chrono-data', '/home/harry/miniconda3/envs/nedm/share/chrono/data',
                 '--crm-config', a.config, '--horizon-s', '120'], scene_hook=scene_hook, frame_hook=frame_hook)
    finally:
        json.dump(S['rows'], open(out / 'overlay.json', 'w'))
