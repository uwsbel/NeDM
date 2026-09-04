import csv, glob, json, sys, collections
import numpy as np
ROOT=sys.argv[1]; TERR=sys.argv[2]
cmd={"cmd_vx_mps":[], "cmd_vy_mps":[], "cmd_wz_radps":[]}
ach_vx=[]; ach_vy=[]; ach_wz=[]
n=0
for f in sorted(glob.glob(f"{ROOT}/{TERR}_*/episodes/*.csv")):
    try:
        rows=list(csv.DictReader(open(f)))
    except FileNotFoundError:
        continue          # collection still running; --overwrite recreates dirs
    if not rows: continue
    n+=1
    for k in cmd: cmd[k].extend(float(r[k]) for r in rows)
    t=np.array([float(r["time_s"]) for r in rows])
    x=np.array([float(r["pos_x_m"]) for r in rows]); y=np.array([float(r["pos_y_m"]) for r in rows])
    yaw=np.unwrap(np.array([float(r["yaw_rad"]) for r in rows]))
    vx=np.gradient(x,t); vy=np.gradient(y,t)
    cy,sy=np.cos(yaw),np.sin(yaw)
    ach_vx.extend(cy*vx+sy*vy); ach_vy.extend(-sy*vx+cy*vy); ach_wz.extend(np.gradient(yaw,t))
print(f"{TERR}: {n} episodes")
def hist(name, a, lo, hi, bins=10):
    a=np.array(a); a=a[np.isfinite(a)]
    h,e=np.histogram(a,bins=bins,range=(lo,hi))
    print(f"  {name:12s} min {a.min():+.3f} max {a.max():+.3f} mean {a.mean():+.3f}")
    for c,l,r in zip(h,e[:-1],e[1:]):
        print(f"     [{l:+.2f},{r:+.2f}) {'#'*int(50*c/max(h.max(),1)):50s} {100*c/len(a):5.1f}%")
print(" COMMANDED:")
hist("vx",cmd["cmd_vx_mps"],-0.5,0.5); hist("vy",cmd["cmd_vy_mps"],-0.5,0.5); hist("wz",cmd["cmd_wz_radps"],-1,1)
print(" ACHIEVED (body frame):")
hist("vx",ach_vx,-0.5,0.5); hist("vy",ach_vy,-0.5,0.5); hist("wz",ach_wz,-1,1)
