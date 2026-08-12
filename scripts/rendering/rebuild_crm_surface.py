"""Re-reconstruct CRM soil surfaces from kept SPH particle dumps.

The particles are the expensive product -- they take a Chrono FSI rollout to make, and
CRM is non-deterministic, so a re-run gives a *different* rollout rather than the same one
at higher quality. Running the eval with ``--crm-surface-keep-particles`` leaves the
``surface_NNNNN.xyz`` dumps next to the meshes, and this script turns them back into
meshes at whatever resolution / smoothing / iso-threshold you want, in seconds per frame
instead of a re-simulation.

    python scripts/rendering/rebuild_crm_surface.py <crm_surface_dir> \
        --cube-size 1.5 --smoothing-length 2.0 --surface-threshold 0.8 \
        --smoothing-iters 25 --normals-smoothing-iters 10

Why the threshold matters: the tyres throw individual particles up to half a metre above
the bed. At Chrono's default threshold of 0.6 each of those reconstructs as its own little
iso-surface, which renders as a field of angular shards -- fine at 30 m, obviously wrong in
a chase shot. Raising it to ~0.8 drops the isolated flyers (their reconstructed density is
far below rest density) while keeping the churned band the wheels leave behind. Note that
``--mesh-smoothing-weights=on`` deliberately *preserves* isolated particles, so smoothing
alone will not remove them.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nedm.crm_surface import run_splashsurf  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("surface_dir", type=Path, help="crm_surface/ directory holding surface_NNNNN.xyz dumps.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Where to write meshes (default: in place).")
    parser.add_argument("--cube-size", type=float, default=1.5, help="Marching-cubes cell size (x particle radius).")
    parser.add_argument("--smoothing-length", type=float, default=2.0, help="SPH kernel smoothing length (x radius).")
    parser.add_argument("--surface-threshold", type=float, default=0.8, help="Iso-surface density threshold.")
    parser.add_argument("--smoothing-iters", type=int, default=25, help="Mesh smoothing iterations.")
    parser.add_argument("--normals-smoothing-iters", type=int, default=10, help="Normal-field smoothing iterations.")
    parser.add_argument("--particle-radius", type=float, default=None, help="Override (default: manifest spacing / 2).")
    parser.add_argument("--format", choices=("ply", "obj"), default=None, help="Override the manifest's mesh format.")
    parser.add_argument("--limit", type=int, default=None, help="Only rebuild the first N frames (for a test).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    surface_dir = args.surface_dir.expanduser().resolve()
    manifest_path = surface_dir / "crm_surface_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"no crm_surface_manifest.json in {surface_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    particles = sorted(surface_dir.glob("surface_*.xyz"))
    if not particles:
        raise SystemExit(
            f"no particle dumps in {surface_dir} -- the eval must be run with --crm-surface-keep-particles"
        )
    if args.limit is not None:
        particles = particles[: int(args.limit)]

    radius = args.particle_radius
    if radius is None:
        radius = float(manifest["initial_spacing_m"]) / 2.0
    fmt = args.format or manifest.get("format", "ply")
    out_dir = (args.output_dir or surface_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"rebuilding {len(particles)} surfaces -> {out_dir}\n"
        f"  r={radius} l={args.smoothing_length} c={args.cube_size} t={args.surface_threshold} "
        f"smooth={args.smoothing_iters} normals_smooth={args.normals_smoothing_iters}"
    )
    started = time.time()
    for index, particle_path in enumerate(particles):
        mesh_path = out_dir / f"{particle_path.stem}.{fmt}"
        run_splashsurf(
            particle_path,
            mesh_path,
            particle_radius=radius,
            smoothing_length=args.smoothing_length,
            cube_size=args.cube_size,
            surface_threshold=args.surface_threshold,
            normals=True,
            mesh_cleanup=True,
            mesh_smoothing_iters=args.smoothing_iters,
            mesh_smoothing_weights=True,
            normals_smoothing_iters=args.normals_smoothing_iters,
            decimate_barnacles=True,
        )
        if index % 10 == 0 or index == len(particles) - 1:
            elapsed = time.time() - started
            print(f"  {index + 1}/{len(particles)}  {elapsed:.0f}s elapsed", flush=True)

    manifest.update(
        {
            "cube_size": args.cube_size,
            "smoothing_length": args.smoothing_length,
            "surface_threshold": args.surface_threshold,
            "smoothing_iters": args.smoothing_iters,
            "normals_smoothing_iters": args.normals_smoothing_iters,
            "rebuilt": True,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"REBUILT {len(particles)} meshes in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
