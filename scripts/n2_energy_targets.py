"""Per-station energy and time targets from recorded episodes (numpy only).

For every run dir: progress along the planned route (f104_n2_dataset.project), its running maximum (high-water mark),
cumulative POSITIVE motorshaft work W+ (kJ, from positive_work_kj_per_interval; identical to engine speed x torque
integrated where positive) and elapsed time. On the 96-station grid of the route: E[j] = W+ spent when the vehicle first
reached station j, T[j] = time to reach it; NaN beyond the furthest station reached (censored: failed / stalled routes).
  python scripts/n2_energy_targets.py --runs 'glob' ['glob' ...] --out energy_ds.npz --workers 14
"""
import argparse, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f104_n2_dataset import project, N_STATION, DT


def one(d):
    try:
        z = np.load(d + '/trajectory.npz'); o = json.load(open(d + '/outcome.json')); c = np.load(d + '/command_reference.npz')
    except Exception:
        return None
    wp = np.asarray(c['reference_waypoints'], float); pose = z['pose'][:, :2].astype(float); n = len(pose)
    s, dev, L = project(pose, wp)
    hwm = np.maximum.accumulate(s)
    w = np.cumsum(np.asarray(z['positive_work_kj_per_interval'], float))          # W+ up to the END of interval i
    t = (np.arange(n) + 1) * DT
    grid = np.linspace(0.0, L, N_STATION)
    E = np.full(N_STATION, np.nan, np.float32); T = np.full(N_STATION, np.nan, np.float32)
    reach = float(hwm[-1])
    for j, sj in enumerate(grid):
        if sj > reach + 1e-9: break
        k = int(np.searchsorted(hwm, sj, side='left'))      # first frame whose high-water mark reaches station j
        k = min(k, n - 1)
        E[j] = w[k]; T[j] = t[k]
    return dict(id=os.path.basename(d), E=E, T=T, total_wplus=float(w[-1]), elapsed=float(n * DT), route_len=float(L),
                reach_frac=float(reach / max(L, 1e-6)), fail=int(o['status'] != 'goal_reached'),
                mean_speed_cmd=float(np.asarray(c['reference_speeds'], float)[1:-1].mean()), max_dev=float(dev.max()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=14)
    a = ap.parse_args()
    ds = sorted(set(sum((glob.glob(p) for p in a.runs), [])))
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for r in ex.map(one, ds, chunksize=64) if r is not None]
    A = lambda k, dt=None: np.array([r[k] for r in rows], dtype=dt)
    np.savez_compressed(a.out, id=A('id', object), E=np.stack([r['E'] for r in rows]), T=np.stack([r['T'] for r in rows]),
                        total_wplus=A('total_wplus', np.float32), elapsed=A('elapsed', np.float32), route_len=A('route_len', np.float32),
                        reach_frac=A('reach_frac', np.float32), fail=A('fail', np.int8), mean_speed_cmd=A('mean_speed_cmd', np.float32),
                        max_dev=A('max_dev', np.float32))
    ok = A('fail') == 0
    print(f'{len(rows)}/{len(ds)} episodes -> {a.out}; goal-reached {ok.sum()}: W+ to goal median {np.median(A("total_wplus")[ok]):.0f} kJ, '
          f'time median {np.median(A("elapsed")[ok]):.1f} s; stations with a target (mean over all rows) {np.isfinite(np.stack([r["E"] for r in rows])).mean():.2f}')


if __name__ == '__main__':
    main()
