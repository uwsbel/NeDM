"""Command -> achieved transfer, per axis, using only the family that drives it."""
import csv, glob, sys
import numpy as np
ROOT, TERR = sys.argv[1], sys.argv[2]
AXES = [("constant","cmd_vx_mps","vx"), ("lateral","cmd_vy_mps","vy"), ("pivot","cmd_wz_radps","wz")]
for fam, cmdcol, axis in AXES:
    pts=[]
    for f in sorted(glob.glob(f"{ROOT}/{TERR}_{fam}_*/episodes/*.csv")):
        try: rows=list(csv.DictReader(open(f)))
        except FileNotFoundError: continue
        if not rows: continue
        t=np.array([float(r["time_s"]) for r in rows])
        x=np.array([float(r["pos_x_m"]) for r in rows]); y=np.array([float(r["pos_y_m"]) for r in rows])
        yaw=np.unwrap(np.array([float(r["yaw_rad"]) for r in rows]))
        m=t>t[0]+1.0
        vx=np.gradient(x,t); vy=np.gradient(y,t); cy,sy=np.cos(yaw),np.sin(yaw)
        got={"vx":(cy*vx+sy*vy)[m].mean(), "vy":(-sy*vx+cy*vy)[m].mean(),
             "wz":np.gradient(yaw,t)[m].mean()}[axis]
        pts.append((float(rows[0][cmdcol]), got))
    pts.sort()
    c=np.array([p[0] for p in pts]); a=np.array([p[1] for p in pts])
    print(f"\n{TERR} / {fam}: commanded {axis} -> achieved {axis}   (n={len(pts)})")
    for lo,hi in [(-1.01,-0.6),(-0.6,-0.35),(-0.35,-0.15),(-0.15,0.15),(0.15,0.35),(0.35,0.6),(0.6,1.01)]:
        s=(c>=lo)&(c<hi)
        if s.sum(): print(f"   cmd [{lo:+.2f},{hi:+.2f}) n={s.sum():3d}  achieved mean {a[s].mean():+.3f}  "
                          f"ratio {a[s].mean()/((lo+hi)/2):.2f}")
    print(f"   achieved range [{a.min():+.3f}, {a.max():+.3f}];  "
          f"{100*np.mean(np.abs(a)<0.05):.0f}% of episodes below 0.05 in magnitude")
