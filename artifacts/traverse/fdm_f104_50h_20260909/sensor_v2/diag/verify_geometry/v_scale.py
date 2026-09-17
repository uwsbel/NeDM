"""Fine search for a residual radial-scale term between the back-projected world frame and TerrainMap."""
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/harry/NeDM-traverse_mppi"); sys.path.insert(0, str(ROOT/"src"))
from nedm.traverse.terrain import TerrainMap
MAPS = ROOT/"artifacts/traverse/fdm_f104_50h_20260909/sensor_v1/maps"
OUT = Path("/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/diag/verify_geometry")
ARENAS=["arena_f104_50h_v1","arena_g203","arena_g216","arena_g217","arena_g228","arena_g231"]
res={}
for arena in ARENAS:
    cam = json.load(open(MAPS/arena/"observation.json"))["camera"]
    with np.load(MAPS/arena/"observation.npz") as o: depth=o["depth_m"].astype(np.float64)
    h,w=depth.shape; H=float(cam["cam_height_m"]); f0=(w/2)/math.tan(float(cam["hfov_rad"])/2)
    col=np.arange(w,dtype=float)[None,:]; row=np.arange(h,dtype=float)[:,None]
    rx=(col-(w-1)/2)/f0+0*row; ry=-(row-(h-1)/2)/f0+0*col
    sec=np.sqrt(1+rx**2+ry**2); ok=np.isfinite(depth)&(depth>0)&(depth<float(cam["max_depth_m"])-1e-6)
    ax=np.where(ok,depth/sec,np.nan); x0=rx*ax; y0=ry*ax; z=H-ax
    tm=TerrainMap.from_dir(ROOT/"assets/traverse"/arena)
    m=ok&(np.abs(x0)<38)&(np.abs(y0)<38)
    X0,Y0,Z=x0[m],y0[m],z[m]
    def rmse(s, dH=0.0):
        e=Z+dH-tm.height(X0*s,Y0*s); return float(np.sqrt((e**2).mean())), float(np.abs(e).mean()), float(e.mean())
    ss=np.arange(0.9975,1.0011,0.00025)
    curve=[(float(s),)+rmse(s) for s in ss]
    best=min(curve,key=lambda t:t[1])
    # joint with a height offset
    bs=best[0]; dHs=np.arange(-0.004,0.0041,0.001)
    curve2=[(float(d),)+rmse(bs,d) for d in dHs]
    best2=min(curve2,key=lambda t:t[1])
    res[arena]=dict(rmse_s1=rmse(1.0), best_s=best[0], rmse_best_s=best[1], mae_best_s=best[2],
                    curve=[[round(a,5),round(b,6),round(c,6),round(d,6)] for a,b,c,d in curve],
                    best_dH_at_best_s=best2[0], rmse_best_sdH=best2[1], mae_best_sdH=best2[2],
                    displacement_at_r40_m=float(40*abs(1-best[0])))
    print(f"{arena}: s=1 rmse {res[arena]['rmse_s1'][0]:.5f} mae {res[arena]['rmse_s1'][1]:.5f} | "
          f"best s={best[0]:.5f} rmse {best[1]:.5f} mae {best[2]:.5f} | +dH={best2[0]:+.3f} rmse {best2[1]:.5f} mae {best2[2]:.5f}", flush=True)
json.dump(res, open(OUT/"v_scale.json","w"), indent=1)
