"""Independent reproduction of TASK B's central claim: is the omitted d0 worth millimetres or metres?

Built from scratch (does not read task_b's points/*.npz): samples the SAME v1 corridor tensor the D arm was
trained on (scripts/sensor_dataset.tensor10) for a sample of saved test-2 routes on two arenas, then

  1. checks the identity  z - z0 = (H - z0)(1 - s0/s) - dd/s  on real numbers;
  2. fits the single constant Qbar on f104 and evaluates the closed form z - z0 = Qbar(1 - s0/s) - dd/s on the
     other arena (the "d0 costs ~4 mm" claim);
  3. measures how much of the remaining error is attributable to d0 (i.e. to z0 varying), by substituting the
     true per-route (H - z0) instead of Qbar;
  4. reports the perspective/terrain split of depth_rel and the linear-readout weight ratio.

Simulator heights are used only as an evaluation reference (channel 0 of the same tensor is the pipeline's own
height, so the comparison is internal).
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path('/home/harry/NeDM-traverse_mppi')
EXP = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'src'))
import sensor_dataset as SD

ARENA_DIR = {'f104': 'arena_f104_50h_v1', 'g203': 'arena_g203', 'g216': 'arena_g216',
             'g217': 'arena_g217', 'g228': 'arena_g228', 'g231': 'arena_g231'}
H = 110.0
NROUTE = 250


def collect(tag, nroute=NROUTE):
    SD.G.clear()
    SD.init_map(str(EXP / 'sensor_v1/maps' / ARENA_DIR[tag]))
    files = sorted((EXP / f'sensor_v1/test2_{tag}/routes').glob('*.json'))[:nroute]
    rows = []
    for fp in files:
        r = json.load(open(fp))
        wp = np.asarray(r['waypoints'], float); sp = np.asarray(r['speeds'], float)
        st = np.asarray(r['stations'], float)
        X, _ = SD.tensor10(wp, sp, st)
        elev_rel, valid, dd, sec1 = X[0], X[4] > 0.5, X[5], X[6]
        sec = sec1 + 1.0
        sec0 = sec[0, 16] + 0.0
        m = valid & np.isfinite(dd)
        m[0, 16] = False                        # the reference cell itself is trivially 0
        # keep interior points only (same rule as task B)
        rows.append(dict(dd=dd[m], sec=sec[m], sec0=np.full(m.sum(), sec0), zrel=elev_rel[m],
                         z0=np.full(m.sum(), np.nan)))
        # z0 = camera height - d0/sec0 ; d0 is the depth at the start cell
        # recover d0 from the tensor: depth_rel is 0 at [0,16] by construction, so use the height channel
        rows[-1]['z0'][:] = 0.0                 # elev_rel is relative, absolute z0 filled below
    return rows


def absolute_start_height(tag, nroute=NROUTE):
    """z0 from the capture itself (channel 0 is relative), via z = H - d/sec at the start cell."""
    SD.G.clear(); SD.init_map(str(EXP / 'sensor_v1/maps' / ARENA_DIR[tag]))
    files = sorted((EXP / f'sensor_v1/test2_{tag}/routes').glob('*.json'))[:nroute]
    out = []
    for fp in files:
        r = json.load(open(fp))
        wp = np.asarray(r['waypoints'], float); st = np.asarray(r['stations'], float)
        gx, gy, _ = SD.corridor(wp, st)
        rowW = SD.G['ctrW'] - gy / SD.G['mppW']; colW = SD.G['ctrW'] + gx / SD.G['mppW']
        depth = SD.bilinear(SD.G['depth'], rowW, colW)
        sec = np.sqrt(1 + ((colW - SD.G['ctrW']) / SD.G['f']) ** 2 + ((rowW - SD.G['ctrW']) / SD.G['f']) ** 2)
        out.append((H - depth[0, 16] / sec[0, 16], depth[0, 16], sec[0, 16]))
    return out


def build(tag):
    rows = collect(tag)
    z0d = absolute_start_height(tag)
    dd = np.concatenate([r['dd'] for r in rows])
    sec = np.concatenate([r['sec'] for r in rows])
    sec0 = np.concatenate([r['sec0'] for r in rows])
    zrel = np.concatenate([r['zrel'] for r in rows])
    z0 = np.concatenate([np.full(len(r['dd']), z0d[i][0]) for i, r in enumerate(rows)])
    d0 = np.concatenate([np.full(len(r['dd']), z0d[i][1]) for i, r in enumerate(rows)])
    return dict(dd=dd, sec=sec, sec0=sec0, zrel=zrel, z0=z0, d0=d0)


out = {}
data = {t: build(t) for t in ('f104', 'g231', 'g217')}

for t, D in data.items():
    exact = (H - D['z0']) * (1 - D['sec0'] / D['sec']) - D['dd'] / D['sec']
    out[f'{t}/identity_vs_height_channel'] = dict(
        n=int(len(D['dd'])),
        mae=float(np.abs(exact - D['zrel']).mean()),
        max=float(np.abs(exact - D['zrel']).max()),
        zrel_std=float(D['zrel'].std()))
    print(f"{t}: identity (H-z0)(1-s0/s)-dd/s vs the tensor's own height channel: "
          f"MAE {np.abs(exact-D['zrel']).mean():.6f} m, max {np.abs(exact-D['zrel']).max():.6f} m "
          f"(n={len(D['dd'])}, target sd {D['zrel'].std():.3f} m)")

# fit the single constant on f104, evaluate everywhere
F = data['f104']
A = (1 - F['sec0'] / F['sec'])
Q = float(((F['zrel'] + F['dd'] / F['sec']) * A).sum() / (A * A).sum())
out['Qbar_fitted_on_f104'] = Q
print(f"\nQbar fitted on f104 = {Q:.3f} m  (H - mean z0 = {H - F['z0'].mean():.3f})")
for t, D in data.items():
    pred = Q * (1 - D['sec0'] / D['sec']) - D['dd'] / D['sec']
    err = pred - D['zrel']
    predt = (H - D['z0']) * (1 - D['sec0'] / D['sec']) - D['dd'] / D['sec']   # true per-route constant
    out[f'{t}/closed_form'] = dict(rmse_m=float(np.sqrt((err ** 2).mean())),
                                   p99_abs=float(np.percentile(np.abs(err), 99)),
                                   max_abs=float(np.abs(err).max()),
                                   rmse_with_true_z0=float(np.sqrt(((predt - D['zrel']) ** 2).mean())),
                                   drop_sec_terms_rmse=float(np.sqrt(((-D['dd'] / D['sec'] - D['zrel']) ** 2).mean())),
                                   z0_std=float(np.unique(D['z0']).std()))
    r = out[f'{t}/closed_form']
    print(f"  {t}: closed form RMSE {r['rmse_m']:.4f} m (p99 {r['p99_abs']:.4f}, max {r['max_abs']:.4f}); "
          f"with the TRUE per-route (H-z0): {r['rmse_with_true_z0']:.6f} m; "
          f"without the sec terms: {r['drop_sec_terms_rmse']:.3f} m; z0 sd {r['z0_std']:.3f} m")

# perspective / terrain split and the linear readout
for t, D in data.items():
    persp = (H - D['z0']) * (D['sec'] - D['sec0'])
    terr = -D['zrel'] * D['sec']
    M = np.stack([D['dd'], D['sec'] - 1, D['sec0'] - 1, np.ones_like(D['dd'])], 1)
    w, *_ = np.linalg.lstsq(M, D['zrel'], rcond=None)
    res = M @ w - D['zrel']
    out[f'{t}/structure'] = dict(depth_rel_std=float(D['dd'].std()), persp_std=float(persp.std()),
                                 terr_std=float(terr.std()), ratio=float(persp.std() / terr.std()),
                                 identity_max_err=float(np.abs(persp + terr - D['dd']).max()),
                                 lstsq_w=[float(x) for x in w], lstsq_rmse=float(np.sqrt((res ** 2).mean())),
                                 sec1_std=float((D['sec'] - 1).std()),
                                 weight_ratio=float(abs(w[1] / w[0])),
                                 scale_ratio=float(D['dd'].std() / (D['sec'] - 1).std()))
    r = out[f'{t}/structure']
    print(f"  {t}: depth_rel sd {r['depth_rel_std']:.3f} = perspective {r['persp_std']:.3f} + terrain "
          f"{r['terr_std']:.3f} (ratio {r['ratio']:.2f}, identity max err {r['identity_max_err']:.2e}); "
          f"lstsq w={np.round(w,3)} RMSE {r['lstsq_rmse']:.4f}; weight ratio {r['weight_ratio']:.1f}, "
          f"channel scale ratio {r['scale_ratio']:.1f}")

json.dump(out, open(HERE / 'verify_taskb_algebra.json', 'w'), indent=1)
print('\nwrote', HERE / 'verify_taskb_algebra.json')
