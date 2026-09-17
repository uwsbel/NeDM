import json, sys, math, time
from pathlib import Path
REPO = Path('/home/harry/NeDM-traverse_mppi'); sys.path.insert(0, str(REPO/'src'))
import pychrono as chrono
from nedm.traverse.layout import EpisodeLayout
from nedm.traverse.scene import build_config, build_scene
from nedm.traverse.terrain import TerrainMap
case = json.loads(Path('/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/cases_g216/cases/g216_v2_group_0000.json').read_text())
arena = (REPO / case['arena']).resolve(); layout = EpisodeLayout.from_json(case['layout']); tmap = TerrainMap.from_dir(arena)
cfg = build_config(arena, (*layout.start_xy, float(tmap.height(*layout.start_xy)) + .75), layout.start_yaw)
cfg['chrono_data_root'] = '/home/harry/chrono/data'; cfg['vehicle_data_root'] = '/home/harry/chrono/data/vehicle'
t0=time.time(); sc = build_scene(cfg, layout, tmap, arena, plan=None, render=None); print('build', time.time()-t0)
tot=0
for i, b in enumerate(sc.system.GetBodies()):
    vm = b.GetVisualModel()
    if vm is None: continue
    for inst in vm.GetShapeInstances():
        sh = inst.shape
        n = None
        tm = chrono.CastToChVisualShapeTriangleMesh(sh) if hasattr(chrono,'CastToChVisualShapeTriangleMesh') else None
        name = type(sh).__name__
        try:
            if tm is not None:
                n = tm.GetMesh().GetNumTriangles()
        except Exception as e:
            n = None
        print(i, b.GetName(), b.IsFixed(), name, n, sh.IsVisible())
        if n: tot += n
print('other items', len(sc.system.GetOtherPhysicsItems()), 'tri total', tot)
