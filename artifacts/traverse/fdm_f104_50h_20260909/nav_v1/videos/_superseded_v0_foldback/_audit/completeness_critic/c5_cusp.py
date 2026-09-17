import json, sys, numpy as np
sys.path.insert(0,'/home/harry/NeDM-traverse_mppi/src')
from nedm.traverse import fdm_mppi as F
B='/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909/nav_v1/local_luffy/video_runs/runs/'
for run, rids in [('g204_nav_003__R2',[36,37]),('g204_nav_003__R1L',[60,61,62]),('g203_nav_001__R1L',[76,77,78,79])]:
    R={r['route_id']:r for r in json.load(open(B+run+'/routes.json'))}
    D={d.get('route_id'):d for d in json.load(open(B+run+'/decisions.json'))}
    for rid in rids:
        w=np.array(R[rid]['waypoints']); pose=np.array(D[rid]['pose'])
        h=np.degrees(np.arctan2(np.diff(w[:,1]),np.diff(w[:,0]))); dh=np.abs((np.diff(h)+180)%360-180); cusp=int(dh.argmax())+1
        anchor=int(np.argmin(np.linalg.norm(w-pose[:2],axis=1)))
        full=F._curvature_max(w); sliced=F._curvature_max(w[min(anchor,len(w)-2):])
        print(run, 'route', rid, 'n pts', len(w), 'cusp at point', cusp, 'anchor-nearest point index', anchor, 'curv full %.3f sliced %.3f'%(full, sliced), 'limit 0.125')
