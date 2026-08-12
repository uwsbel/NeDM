#!/usr/bin/env python3
"""Export the Chrono ``RigidTerrain`` heightmap patch of a bumpy reference as a Wavefront OBJ.

The bumpy terrain is per-episode: ``assign_height_map_index`` hashes the episode id to one
of the 100 ``bumpy_field_*.bmp`` heightmaps, and ``create_rigid_terrain`` hands that BMP to
``RigidTerrain.AddPatch``. This script rebuilds exactly that patch and calls Chrono's own
``RigidTerrain.ExportMeshWavefront``, so the OBJ is the collision/visual mesh the vehicle
actually drove on -- not a re-derivation of it from the BMP.

The patch is a ``length_m x width_m`` field carrying one quad per heightmap pixel, so a
256 px BMP over a 500 m patch gives ~1.96 m triangles. That coarseness is the terrain, not
a rendering artefact.

Usage:
    python scripts/rendering/export_bumpy_terrain_mesh.py \
        --reference b10_s003_multi_steer_00013 --out-dir artifacts/terrain_meshes
    # or, when you only know the row in the eval summary:
    python scripts/rendering/export_bumpy_terrain_mesh.py --reference-index 9 ...
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import pychrono as chrono  # noqa: E402
import numpy as np  # noqa: E402

from nedm.hmmwv_data import (  # noqa: E402
    assign_height_map_index,
    create_rigid_terrain,
    resolve_height_map,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--reference", help="Episode id, e.g. b10_s003_multi_steer_00013.")
    src.add_argument("--reference-index", type=int, help="Row index into --summary's rollouts.")
    parser.add_argument(
        "--summary",
        type=Path,
        help="Eval summary.json used to turn --reference-index into an episode id.",
    )
    parser.add_argument("--chrono-config", type=Path, default=REPO_ROOT / "configs/hmmwv_bumpy_eval.json")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts/terrain_meshes")
    parser.add_argument("--name", help="Basename for the OBJ (default: derived from the episode id).")
    return parser.parse_args()


def episode_id_from_summary(summary: Path, index: int) -> str:
    rollouts = json.loads(summary.read_text())["rollouts"]
    # The summary records "<maneuver>/<episode_id>"; the terrain hash uses the bare id.
    return rollouts[index]["reference"].split("/")[-1]


def main() -> None:
    args = parse_args()
    if args.reference:
        episode_id = args.reference
    else:
        if args.summary is None:
            raise SystemExit("--reference-index needs --summary")
        episode_id = episode_id_from_summary(args.summary, args.reference_index)

    config = json.loads(args.chrono_config.read_text())
    terrain_cfg = config["terrain"]
    height_map = resolve_height_map(config, episode_id)
    if height_map is None:
        raise SystemExit(f"{args.chrono_config} is not a rigid_heightmap config")
    index, bmp_path = height_map
    assert index == assign_height_map_index(episode_id, int(terrain_cfg["height_map_count"]))

    system = chrono.ChSystemSMC()
    terrain = create_rigid_terrain(system, config, height_map_path=bmp_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or f"{episode_id}_bumpy_field_{index:03d}"
    # ExportMeshWavefront writes one .obj per mesh patch, named after the patch's mesh, into
    # the directory it is given. Export into a scratch dir so the rename below is unambiguous.
    scratch = args.out_dir / f".export_{stem}"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    terrain.ExportMeshWavefront(str(scratch))

    produced = sorted(scratch.glob("*.obj"))
    if len(produced) != 1:
        raise SystemExit(f"expected exactly one OBJ from ExportMeshWavefront, got {produced}")
    obj_path = args.out_dir / f"{stem}.obj"
    shutil.move(str(produced[0]), obj_path)
    shutil.rmtree(scratch)

    verts = np.array(
        [[float(v) for v in line.split()[1:4]] for line in obj_path.read_text().splitlines() if line.startswith("v ")]
    )
    faces = sum(1 for line in obj_path.read_text().splitlines() if line.startswith("f "))
    side = int(round(float(np.sqrt(len(verts)))))
    spacing = float(terrain_cfg["length_m"]) / (side - 1) if side > 1 else float("nan")
    meta = {
        "episode_id": episode_id,
        "height_map": bmp_path.name,
        "height_map_index": index,
        "obj": str(obj_path.relative_to(REPO_ROOT)) if obj_path.is_relative_to(REPO_ROOT) else str(obj_path),
        "patch_length_m": terrain_cfg["length_m"],
        "patch_width_m": terrain_cfg["width_m"],
        "height_min_m": terrain_cfg["height_min_m"],
        "height_max_m": terrain_cfg["height_max_m"],
        "vertices": int(len(verts)),
        "faces": faces,
        "grid": f"{side}x{side}",
        "vertex_spacing_m": round(spacing, 4),
        "z_min_m": round(float(verts[:, 2].min()), 4),
        "z_max_m": round(float(verts[:, 2].max()), 4),
        "source": "pychrono RigidTerrain.ExportMeshWavefront",
    }
    (args.out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
