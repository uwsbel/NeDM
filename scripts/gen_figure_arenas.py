"""f104 and its five sibling arenas on one colour scale (terrain relative to each arena's median ground)."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
sys.path.insert(0, 'src')
from nedm.traverse.terrain import TerrainMap

G = 'artifacts/traverse/fdm_f104_50h_20260909/gen_v1'
ARENAS = [('f104 (training arena)', 'arena_f104_50h_v1'), ('g228', 'arena_g228'), ('g203', 'arena_g203'),
          ('g217', 'arena_g217'), ('g216', 'arena_g216'), ('g231', 'arena_g231')]
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
sim = {r['arena']: r for r in json.load(open(G + '/arena_similarity.json'))['ranking']}
fig, axes = plt.subplots(1, 6, figsize=(21, 4.6), facecolor=SURF)
norm = TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=4.0)
for ax, (title, name) in zip(axes, ARENAS):
    tm = TerrainMap.from_dir(Path('assets/traverse') / name)
    h = np.flipud(tm.height_grid).astype(float); h = h - np.median(h); half = tm.size_m / 2
    im = ax.imshow(np.flipud(h), extent=[-half, half, -half, half], origin='lower', cmap='BrBG_r', norm=norm)
    meta = json.load(open(Path('assets/traverse') / name / 'arena_meta.json')); f = meta['family']
    sub = f"{f['n_hills']} hills, {f['n_craters']} craters, slope cap {f['slope_cap_deg']:.0f} deg"
    if name in sim: sub += f"\nsimilarity distance {sim[name]['distance']:.2f}"
    ax.set_title(title, fontsize=12, color=INK); ax.set_xlabel(sub, fontsize=9, color=INK2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color('#d9d8d4')
cb = fig.colorbar(im, ax=axes, fraction=0.012, pad=0.01); cb.set_label('height above median ground (m)', color=INK2, fontsize=9)
fig.suptitle('The training arena and five new arenas from the same terrain generator (5 closest of 40 seeds)', color=INK, fontsize=14, x=0.45)
fig.savefig(G + '/arenas.png', dpi=120, facecolor=SURF, bbox_inches='tight')
print('wrote', G + '/arenas.png')
