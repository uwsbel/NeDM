import json
from pathlib import Path
import numpy as np
HERE = Path('/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/diag/task_c')
ARENAS = ['f104','g203','g216','g217','g228','g231']; MODELS=['n2','e0','d']; POOLS=['proposal','fixed2']
def spearman(a,b):
    ra=np.argsort(np.argsort(a,1),1).astype(float); rb=np.argsort(np.argsort(b,1),1).astype(float)
    ra-=ra.mean(1,keepdims=True); rb-=rb.mean(1,keepdims=True)
    return (ra*rb).sum(1)/np.sqrt((ra**2).sum(1)*(rb**2).sum(1))
D={a:dict(np.load(HERE/f'scores/{a}_ctl.npz')) for a in ARENAS}
cat=lambda k: np.concatenate([D[a][k] for a in ARENAS],0)
out={}
print('control: v1 = deployed image lookup, fg = flat placement rasterised on the v2 grid, v2 = back-projected grid')
print(f'{"pool":9s} {"model":5s} | {"flip v1>fg":>10s} {"flip fg>v2":>10s} {"flip v1>v2":>10s} | '
      f'{"rho v1,fg":>9s} {"rho fg,v2":>9s} {"rho v1,v2":>9s} | {"p50|dzc| v1>fg":>14s} {"fg>v2":>8s}')
for p in POOLS:
    for m in MODELS:
        z1=cat(f'{p}_{m}_v1'); zf=cat(f'{p}_{m}_fg'); z2=cat(f'{p}_{m}_v2')
        f1=(z1.argmin(1)!=zf.argmin(1)).mean(); f2=(zf.argmin(1)!=z2.argmin(1)).mean(); f3=(z1.argmin(1)!=z2.argmin(1)).mean()
        r1=spearman(z1,zf).mean(); r2=spearman(zf,z2).mean(); r3=spearman(z1,z2).mean()
        c=lambda d: np.median(np.abs(d-d.mean(1,keepdims=True)))
        d1=c(zf-z1); d2=c(z2-zf); d3=c(z2-z1)
        out[f'{p}/{m}']=dict(n_groups=len(z1),flip_v1_fg=float(f1),flip_fg_v2=float(f2),flip_v1_v2=float(f3),
                             rho_v1_fg=float(r1),rho_fg_v2=float(r2),rho_v1_v2=float(r3),
                             p50_abs_dz_centred_v1_fg=float(d1),p50_abs_dz_centred_fg_v2=float(d2),p50_abs_dz_centred_v1_v2=float(d3))
        print(f'{p:9s} {m:5s} | {100*f1:9.1f}% {100*f2:9.1f}% {100*f3:9.1f}% | {r1:9.4f} {r2:9.4f} {r3:9.4f} | '
              f'{d1:14.4f} {d2:8.4f}')
json.dump(out, open(HERE/'control_analysis.json','w'), indent=1)
print('\nwrote', HERE/'control_analysis.json', f'({len(cat("proposal_n2_v1"))} groups, stride 4)')
