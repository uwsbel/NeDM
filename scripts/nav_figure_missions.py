"""The 30 nav_v1 missions on their 10 arenas: development arenas on the top row, unseen arenas on the bottom."""
import argparse, glob, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nedm.traverse.terrain import TerrainMap

DEV = ['arena_f104_50h_v1', 'arena_g228', 'arena_g203', 'arena_g217', 'arena_g216', 'arena_g231']
NEW = ['arena_g213', 'arena_g204', 'arena_g234', 'arena_g223']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--missions', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    ms = [json.load(open(p)) for p in sorted(glob.glob(a.missions + '/*.json'))]
    by = {}
    for m in ms:
        by.setdefault(Path(m['arena']).name, []).append(m)
    order = DEV + NEW
    fig, axes = plt.subplots(2, 6, figsize=(17.5, 6.4), dpi=110)
    for ax in axes.ravel():
        ax.axis('off')
    for i, name in enumerate(order):
        ax = axes[0 if i < 6 else 1, i % 6]
        ax.axis('on'); ax.set_xticks([]); ax.set_yticks([])
        tm = TerrainMap.from_dir(ROOT / 'assets/traverse' / name)
        h = np.flipud(tm.height_grid)
        ax.imshow(h, origin='lower', extent=[-40, 40, -40, 40], cmap='terrain')
        for m in by.get(name, []):
            g = np.asarray(m['goals']); s = np.asarray(m['layout']['start_xy'])
            p = np.vstack([s, g])
            ax.plot(p[:, 0], p[:, 1], '-', color='#d21f1f', lw=1.4, alpha=.9)
            ax.scatter(g[:, 0], g[:, 1], s=22, marker='*', c='w', edgecolors='k', lw=.4, zorder=5)
            ax.scatter(*s, s=16, marker='s', c='k', zorder=5)
        n = len(by.get(name, []))
        tag = 'development' if i < 6 else 'UNSEEN'
        ax.set_title(f"{name.replace('arena_', '')}  ({tag}, {n} missions)", fontsize=9)
    axes[1, 4].axis('off'); axes[1, 5].axis('off')
    fig.suptitle('nav_v1 missions: 30 missions, 5-8 waypoints each, 150-228 m of chained legs', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .94], h_pad=2.4)
    fig.savefig(a.out, bbox_inches='tight')
    print('wrote', a.out)


if __name__ == '__main__':
    main()
