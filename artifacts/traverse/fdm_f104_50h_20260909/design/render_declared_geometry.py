"""Render frozen case/reference geometry only; never open physical recordings."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from nedm.traverse.terrain import TerrainMap

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources = [BASE / "cases/cases.json", BASE / "cases_reserve_v1/cases.json"]
records = []
for source in sources:
    manifest = json.loads(source.read_text())
    records.extend((source.parent, record) for record in manifest["records"])
assert len(records) == 1500
assert len({r["scene_id"] for _, r in records}) == 1500
tmap = TerrainMap.from_dir(ROOT / "assets/traverse/arena_f104_50h_v1")
positions, group_ids = [], []
routes = {}
geometry_hashes = set()
cells = np.zeros((40, 40), int)
split_counts = {k: 0 for k in ("train", "val", "test")}
groups_by_feature = {}
for folder, rec in records:
    case_path = folder / rec["case"]
    assert sha(case_path) == rec["case_sha256"]
    case = json.loads(case_path.read_text())
    sid = rec["scene_id"]
    positions.append(case["layout"]["start_xy"])
    group_ids.append(sid)
    split_counts[rec["split"]] += 1
    groups_by_feature.setdefault(str(rec["feature_index"]), []).append(sid)
    group_routes = []
    assert len(rec["routes"]) == 12
    for ri in (0, 4, 8):
        route_path = folder / rec["routes"][ri]
        assert sha(route_path) == rec["route_sha256"][ri]
        route = json.loads(route_path.read_text())
        xy = np.asarray(route["waypoints"], float)
        geometry_hashes.add(hashlib.sha256(xy.tobytes()).hexdigest())
        group_routes.append(xy)
        ij = np.floor((xy + 40) / 2).astype(int)
        assert ((ij >= 0) & (ij < 40)).all()
        flat = np.unique(ij[:, 1] * 40 + ij[:, 0])
        cells.flat[flat] += 1
    routes[sid] = group_routes
assert len(geometry_hashes) == 4500
positions = np.asarray(positions)

# Fixed stratification by authored feature target, including general traversals.
# No predictions, outcomes, completed-hour counters or physical paths are read.
representatives = []
for feature, ids in sorted(groups_by_feature.items()):
    ids = sorted(ids)
    for index in np.unique(np.rint(np.array([.25, .75]) * (len(ids) - 1)).astype(int)):
        representatives.append(ids[index])
representatives = sorted(set(representatives))

plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig = plt.figure(figsize=(16.8, 6.4))
grid = fig.add_gridspec(2, 3, height_ratios=[1, .065], left=.055, right=.985,
                      bottom=.16, top=.82, wspace=.22, hspace=.25)
axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
color_axes = [fig.add_subplot(grid[1, i]) for i in range(3)]
color_axes[1].axis("off")
extent = [-40, 40, -40, 40]
norm = Normalize(-1.8, 3.9)
coords = -40 + (np.arange(tmap.pixels) + .5) * tmap.res
for ax in axes:
    ax.set(xlim=(-40, 40), ylim=(-40, 40), xlabel="World X (m)", ylabel="World Y (m)")
    ax.set_aspect("equal")
    ax.set_xticks([-40, -20, 0, 20, 40]); ax.set_yticks([-40, -20, 0, 20, 40])

height = axes[0].imshow(tmap.height_grid, extent=extent, origin="lower", cmap="terrain", norm=norm)
axes[0].scatter(positions[:, 0], positions[:, 1], s=4.5, c="#102d48", alpha=.65, linewidths=0,
                label="All 1,500 group starts", rasterized=True)
axes[0].set_title("A  Exact height map + all group starts", loc="left", fontsize=12)
axes[0].legend(loc="lower left", framealpha=.95, markerscale=2, fontsize=9)
fig.colorbar(height, cax=color_axes[0], orientation="horizontal", label="Terrain elevation (m)")

axes[1].imshow(tmap.height_grid, extent=extent, origin="lower", cmap="Greys", vmin=-1.8, vmax=4.5, alpha=.28)
path_colors = ["#174b73", "#087f8c", "#dc7732"]
for offset_i, color in enumerate(path_colors):
    lines = [routes[sid][offset_i] for sid in representatives]
    axes[1].add_collection(LineCollection(lines, colors=color, linewidths=.75, alpha=.32, rasterized=True))
starts = np.asarray([routes[sid][0][0] for sid in representatives])
axes[1].scatter(starts[:, 0], starts[:, 1], color="#14242e", s=10, zorder=4)
axes[1].set_title(f"B  {3*len(representatives)} representative spatial references", loc="left", fontsize=12)
color_axes[1].legend(handles=[Line2D([0], [0], color=c, lw=2, label=f"{offset:+d} m offset")
                             for c, offset in zip(path_colors, (0, -4, 4))],
                     loc="center", ncol=3, frameon=False, fontsize=9)

masked = np.ma.masked_where(cells == 0, cells)
axes[2].imshow(tmap.height_grid, extent=extent, origin="lower", cmap="Greys", alpha=.14)
density = axes[2].imshow(masked, extent=extent, origin="lower", cmap="magma", vmin=1,
                         vmax=int(cells.max()), interpolation="nearest")
axes[2].contour(coords, coords, tmap.height_grid, levels=[0, 1, 2, 3], colors="white", linewidths=.35, alpha=.35)
axes[2].set_title("C  All 4,500 spatial references", loc="left", fontsize=12)
fig.colorbar(density, cax=color_axes[2], orientation="horizontal", label="References sampled in each 2 m cell")

fig.suptitle("F104 declared collection design", fontsize=19, fontweight="bold", x=.055, ha="left", y=.97)
fig.text(.055, .90, "1,500 start/goal groups  ·  4,500 distinct spatial references  ·  18,000 PID episode variants  ·  80 × 80 m unchanged terrain", fontsize=12)
fig.text(.055, .065, "COMMAND GEOMETRY ONLY — no realized trajectories, safety outcomes or collected-hour totals are shown.", fontsize=11, fontweight="bold")
fig.text(.055, .025, "Four speed profiles share each spatial reference. Panel B uses a fixed feature-stratified subset; panel C counts centerline samples once per reference, not vehicle footprints.", fontsize=10)
fig.savefig(OUT / "declared_geometry_overview.png", dpi=180)
fig.savefig(OUT / "declared_geometry_overview.pdf")
plt.close(fig)

proof = {"schema": "f104_declared_geometry_figure_v1", "scope": "Frozen commanded geometry only; no physical output files opened",
         "group_count": len(records), "unique_group_ids": len(set(group_ids)), "unique_spatial_references": len(geometry_hashes),
         "episode_variants": 18000, "split_group_counts": split_counts,
         "representative_group_count": len(representatives), "representative_group_ids": representatives,
         "representative_selection": "Within each feature-index stratum and general-route stratum, choose nearest group index to25th/75thpercentile; deduplicate",
         "occupancy_method": "2m grid; one count per spatial reference/cell containing at least one existing<=0.5mreference waypoint; speed siblings collapsed; no footprint inflation",
         "maximum_spatial_references_per_cell": int(cells.max()), "native_simulation_or_model_inference_performed": False,
         "source_sha256": {str(x.relative_to(ROOT)): sha(x) for x in sources},
         "bmp_sha256": sha(ROOT / "assets/traverse/arena_f104_50h_v1/arena_000.bmp"),
         "script_sha256": sha(__file__), "output_sha256": {name: sha(OUT / name) for name in ("declared_geometry_overview.png", "declared_geometry_overview.pdf")}}
(OUT / "declared_geometry_overview.json").write_text(json.dumps(proof, indent=2) + "\n")
print(json.dumps({k: proof[k] for k in ("group_count", "unique_spatial_references", "episode_variants", "representative_group_count")}))
