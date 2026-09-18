"""Terrain-feature-clustered interval for a paired planner contrast (per_group from scripts/n2_planner_analyze.py results.json).
Groups are clustered by the terrain feature nearest to the start-goal midpoint; the bootstrap resamples clusters, plus a cluster-level sign test.
  python scripts/n2_cluster_ci.py --results <results.json> --cases <case dir> --arena assets/traverse/arena_f104_50h_v1 --pairs B:A,G:A
"""
import argparse, json, sys
from math import comb
from pathlib import Path
import numpy as np
sys.path.insert(0, 'src')
from nedm.traverse.terrain import TerrainMap

ap = argparse.ArgumentParser(); ap.add_argument('--results', required=True); ap.add_argument('--cases', required=True)
ap.add_argument('--arena', default='assets/traverse/arena_f104_50h_v1'); ap.add_argument('--pairs', default='B:A'); ap.add_argument('--label', default='fail')
a = ap.parse_args()
R = json.load(open(a.results))['per_group']; G = sorted(R)
tm = TerrainMap.from_dir(Path(a.arena)); feats = np.array([[f['x_m'], f['y_m']] for f in tm.features])
mid = []
for g in G:
    c = json.load(open(f'{a.cases}/{g}.json')); s = np.array(c['layout']['start_xy']); e = np.array(c['goal_xy']); mid.append((s + e) / 2)
cl = np.argmin(np.linalg.norm(np.array(mid)[:, None, :] - feats[None], axis=-1), axis=1); ids = np.unique(cl)
rng = np.random.default_rng(0); out = {}
for pair in a.pairs.split(','):
    x, y = pair.split(':')
    xa = np.array([R[g][y][a.label] for g in G], float); xb = np.array([R[g][x][a.label] for g in G], float)
    boots = []
    for _ in range(4000):
        pick = rng.choice(ids, len(ids)); sel = np.concatenate([np.where(cl == c)[0] for c in pick]); boots.append(100 * (xb[sel].mean() - xa[sel].mean()))
    wins = int(sum(xb[cl == c].mean() < xa[cl == c].mean() for c in ids)); losses = int(sum(xb[cl == c].mean() > xa[cl == c].mean() for c in ids))
    n = wins + losses; p = min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n) if n else 1.0
    out[pair] = dict(diff_pts=100 * (xb.mean() - xa.mean()), ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], clusters=int(len(ids)),
                     cluster_wins=wins, cluster_losses=losses, sign_p=p)
    print(f"{pair} {a.label}: diff {out[pair]['diff_pts']:+.1f} pts, feature-cluster CI [{out[pair]['ci95'][0]:+.1f}, {out[pair]['ci95'][1]:+.1f}] ({len(ids)} clusters), cluster wins/losses {wins}/{losses}, sign p = {p:.4f}")
json.dump(out, open(Path(a.results).parent / f'cluster_ci_{a.label}.json', 'w'), indent=1)
