"""Spot check of two further Task B claims: the one-constant local cross-slope formula, and the
cross-arena two-sample separability of each depth-side representation."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path('/home/harry/NeDM-traverse_mppi'); sys.path.insert(0, str(ROOT/'src')); sys.path.insert(0, str(ROOT/'scripts'))
from nedm.traverse.terrain import TerrainMap
import sensor_dataset as SD
BASE = ROOT/'artifacts/traverse/fdm_f104_50h_20260909/sensor_v1'
OUT = Path(__file__).parent
ARENAS = {'f104':'arena_f104_50h_v1','g203':'arena_g203','g216':'arena_g216','g217':'arena_g217','g228':'arena_g228','g231':'arena_g231'}
NR = 250

def bil(img,row,col):
    n0,n1=img.shape[-2],img.shape[-1]
    r0=np.clip(np.floor(row).astype(int),0,n0-2); c0=np.clip(np.floor(col).astype(int),0,n1-2)
    fr=np.clip(row-r0,0,1); fc=np.clip(col-c0,0,1)
    return (img[...,r0,c0]*(1-fr)*(1-fc)+img[...,r0,c0+1]*(1-fr)*fc+img[...,r0+1,c0]*fr*(1-fc)+img[...,r0+1,c0+1]*fr*fc)

def extract(tag,arena):
    SD.init_map(str(BASE/f'maps/{arena}'))
    H=SD.G['cam_h']; f=SD.G['f']; ctrW=SD.G['ctrW']; mppW=SD.G['mppW']; dep=SD.G['depth'].astype(np.float64)
    D=[]; SEC=[]; S0=[]; D0=[]; Z=[]; RX=[]; RY=[]
    files=sorted((BASE/f'test2_{tag}/routes').glob('*.json'))[:NR]
    for fp in files:
        r=json.load(open(fp)); wp=np.asarray(r['waypoints'],float); st=np.asarray(r['stations'],float)
        gx,gy,_=SD.corridor(wp,st)
        rw=ctrW-gy/mppW; cw=ctrW+gx/mppW
        d=bil(dep,rw,cw); rx=(cw-ctrW)/f; ry=-(rw-ctrW)/f
        sec=np.sqrt(1+rx**2+ry**2)
        ok=(d<SD.G['max_depth']-1e-3)&(np.abs(gx)<39.5)&(np.abs(gy)<39.5)
        if not ok[0,16]: continue
        if not ok.all(): continue          # keep whole stations clean for the lateral-slope fit
        D.append(d); SEC.append(sec); Z.append(H-d/sec); RX.append(rx); RY.append(ry)
        D0.append(np.full(d.shape,d[0,16])); S0.append(np.full(d.shape,sec[0,16]))
    return [np.stack(a) for a in (D,SEC,S0,D0,Z,RX,RY)]

def lstsq_auc(A,B,seed=0):
    """Two-sample separability: logistic-free proxy = ridge classifier AUC on held-out halves."""
    rng=np.random.default_rng(seed)
    X=np.vstack([A,B]); y=np.r_[np.zeros(len(A)),np.ones(len(B))]
    mu=X.mean(0); sd=X.std(0)+1e-9; X=(X-mu)/sd
    idx=rng.permutation(len(X)); tr=idx[:len(idx)//2]; te=idx[len(idx)//2:]
    Xt=np.c_[X[tr],np.ones(len(tr))]
    w=np.linalg.solve(Xt.T@Xt+1e-3*np.eye(Xt.shape[1]), Xt.T@y[tr])
    s=np.c_[X[te],np.ones(len(te))]@w; yy=y[te]
    r=np.argsort(np.argsort(s))+1.0; n1=yy.sum(); n0=len(yy)-n1
    return float((r[yy==1].sum()-n1*(n1+1)/2)/(n0*n1))

P={}
for tag,arena in ARENAS.items():
    P[tag]=extract(tag,arena)
    print(tag,'routes',P[tag][0].shape[0],flush=True)

# --- one-constant local cross-slope formula, constant fitted on f104 ---
res={}
def cross(arr,dl): return np.gradient(arr,dl,axis=2)
dl=12.0/31
Dc,SECc,S0c,D0c,Zc,_,_=P['f104']
dzdl=cross(Zc,dl); ddl=cross(Dc-D0c,dl); dsl=cross(SECc,dl)
# fit constant Dconst minimising ||dzdl + (1/s)ddl - (Dconst+dr)/s^2 dsl||
a=dsl/SECc**2; b=dzdl+ddl/SECc-((Dc-D0c)/SECc**2)*dsl
Dconst=float((a*b).sum()/(a*a).sum())
for tag in ARENAS:
    Dp,SECp,S0p,D0p,Zp,_,_=P[tag]
    t=cross(Zp,dl); dd=cross(Dp-D0p,dl); ds=cross(SECp,dl)
    pred=-(1/SECp)*dd+((Dconst+(Dp-D0p))/SECp**2)*ds
    pred_nosec=-(1/SECp)*dd
    res[tag]=dict(Dconst=Dconst, rmse=float(np.sqrt(((pred-t)**2).mean())),
                  r2=float(1-((pred-t)**2).mean()/t.var()),
                  rmse_nosec=float(np.sqrt(((pred_nosec-t)**2).mean())),
                  r2_nosec=float(1-((pred_nosec-t)**2).mean()/t.var()),
                  true_slope_std=float(t.std()))
    print(tag,'cross-slope rmse %.4f r2 %.4f | no-sec rmse %.4f r2 %.3f'%(res[tag]['rmse'],res[tag]['r2'],res[tag]['rmse_nosec'],res[tag]['r2_nosec']),flush=True)

# --- cross-arena separability of each representation ---
def feats(tag):
    Dp,SECp,S0p,D0p,Zp,RX,RY=P[tag]
    z0=(110.0-D0p/S0p)
    return {'height_z':np.stack([Zp.ravel()],1),
            'rel_height':np.stack([(Zp-z0).ravel()],1),
            'v1_depthrel_sec':np.stack([(Dp-D0p).ravel(),(SECp-1).ravel()],1),
            'abs_range_sec':np.stack([Dp.ravel(),(SECp-1).ravel()],1),
            'v1_depthrel_sec_sec0':np.stack([(Dp-D0p).ravel(),(SECp-1).ravel(),(S0p-1).ravel()],1),
            'abs_range_raydir':np.stack([Dp.ravel(),RX.ravel(),RY.ravel()],1)}
F={t:feats(t) for t in ARENAS}
auc={}
rng=np.random.default_rng(0)
for k in F['f104']:
    vals=[]
    A=F['f104'][k]; A=A[rng.choice(len(A),min(200000,len(A)),replace=False)]
    for t in ARENAS:
        if t=='f104': continue
        B=F[t][k]; B=B[rng.choice(len(B),min(200000,len(B)),replace=False)]
        vals.append(lstsq_auc(A,B))
    auc[k]=[float(np.mean(vals)),[round(v,3) for v in vals]]
    print(k,auc[k],flush=True)
json.dump(dict(cross_slope=res,separability_auc=auc),open(OUT/'v_taskB2.json','w'),indent=1)
