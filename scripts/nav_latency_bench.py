"""Where the per-decision time actually goes, and what a GPU would change.

The nav_v1 campaign runs entirely on CPU nodes, where one decision costs ~4.4 s wall of which ~2.7 s is Chrono's
software depth rasteriser. That number says little about what the planner would cost on real hardware, so this
measures the two parts that are actually algorithm — corridor extraction and the frozen ensemble — on whatever
device it is given, using a real stored frame and a real candidate pool.

  python nav_latency_bench.py --models 'matched_Dabs_s*.pt' --frame DIR --device cuda --out bench.json
"""
import argparse, json, time
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def timeit(fn, n=10, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return {'mean': float(np.mean(ts)), 'p50': float(np.percentile(ts, 50)), 'min': float(np.min(ts)), 'n': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True); ap.add_argument('--frame', required=True)
    ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    import nav_online as NAV
    import gen_planner as P
    import vehicle_corridor as VC
    import sensor_map_v2 as M

    obs = json.load(open(Path(a.frame) / 'observation.json'))
    z = np.load(Path(a.frame) / 'observation.npz')
    pose = np.asarray(obs['measured_pose_xy_yaw'], float)
    goal = pose[:2] + 30 * np.array([np.cos(pose[2]), np.sin(pose[2])])
    nav = NAV.Navigator(a.models, device=a.device)
    out = {'device': a.device, 'threads': a.threads, 'torch': torch.__version__,
           'n_members': len(nav.model.members), 'channels': nav.channels}
    if a.device != 'cpu':
        out['gpu'] = torch.cuda.get_device_name(0)
    out['backproject_s'] = timeit(lambda: M.grid_from_arrays(z['depth_m'], z['rgb'], nav.camera))
    nav.observe(z['rgb'], z['depth_m'], pose=pose)
    base = P.base_route(pose, goal)
    cands, _ = P.proposal_pool(base, pose, np.random.default_rng(0))
    out['n_candidates'] = len(cands)
    out['propose_s'] = timeit(lambda: P.proposal_pool(base, pose, np.random.default_rng(0)), n=5)
    ex = VC.exclusion_mask(pose, 1.5)
    out['corridor_s'] = timeit(lambda: NAV.corridors12_batch(cands, ex))
    X, L, _ = NAV.corridors12_batch(cands, ex)
    ctx = P.geom_ctx(pose[:2], goal, pose[2], L)
    out['score_s'] = timeit(lambda: nav.model.score(X, ctx))
    out['algorithmic_total_p50_s'] = (out['backproject_s']['p50'] + out['propose_s']['p50']
                                      + out['corridor_s']['p50'] + out['score_s']['p50'])
    out['implied_rate_hz'] = 1.0 / out['algorithmic_total_p50_s']
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
