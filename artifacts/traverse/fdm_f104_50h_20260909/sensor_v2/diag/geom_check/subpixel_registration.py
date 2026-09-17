import json, sys
from pathlib import Path
import numpy as np
ROOT = Path("/home/harry/NeDM-traverse_mppi"); sys.path.insert(0, str(ROOT/"src"))
from nedm.traverse.terrain import TerrainMap
G = ROOT/"artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/grids"
MPP, HALF, N = 80/512, 40.0, 512
c = -HALF + (np.arange(N)+0.5)*MPP
X, Y = np.meshgrid(c, c)
out={}
for arena in ["arena_f104_50h_v1","arena_g231"]:
    with np.load(G/arena/"grid.npz") as g: z=g['z'].astype(np.float64); cov=g['cover']
    t=TerrainMap.from_dir(ROOT/"assets/traverse"/arena)
    m=(cov>0)&np.isfinite(z)
    best=None; tab={}
    for dx in np.arange(-1.0,1.01,0.25):
        for dy in np.arange(-1.0,1.01,0.25):
            e=float(np.abs(z[m]-t.height(X+dx*MPP, Y+dy*MPP)[m]).mean())
            tab[f"{dx:+.2f},{dy:+.2f}"]=round(e,5)
            if best is None or e<best[0]: best=(e,dx,dy)
    out[arena]=dict(best_mae=round(best[0],5), best_shift_cells=[best[1],best[2]],
                    mae_at_zero=tab["+0.00,+0.00"],
                    row=[ (k,v) for k,v in tab.items() if k.endswith(",+0.00")],
                    col=[ (k,v) for k,v in tab.items() if k.startswith("+0.00,")])
    print(arena, 'best', best, 'at zero', tab["+0.00,+0.00"])
    print('  dx sweep (dy=0):', [(k.split(',')[0],v) for k,v in tab.items() if k.endswith(",+0.00")])
    print('  dy sweep (dx=0):', [(k.split(',')[1],v) for k,v in tab.items() if k.startswith("+0.00,")])
json.dump(out, open(ROOT/"artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/diag/geom_check/subpixel_registration.json","w"), indent=1)
