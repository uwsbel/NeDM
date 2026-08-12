"""CRM (SPH) particle field -> renderable surface mesh, via the ``splashsurf`` tool.

Chrono ships this exact pipeline (``ChFsiSplashsurfSPH`` /
``ChFsiProblemSPH::WriteReconstructedSurface``), but the conda pychrono build in the
``nedm`` env is compiled with ``CHRONO_HAS_SPLASHSURF`` undefined (see
``$CONDA_PREFIX/include/chrono_fsi/sph/ChFsiConfigSPH.h``), so
``CRMTerrain.WriteReconstructedSurface`` only prints
"Warning: splashsurf not available; no mesh was generated" and writes nothing.

We therefore reproduce Chrono's two steps in Python -- dump SPH fluid-marker
positions, then run ``splashsurf reconstruct`` -- keeping Chrono's parameter
conventions (particle radius = ``initial_spacing / 2``; the smoothing-length /
cube-size / surface-threshold defaults from ``ChFsiSplashsurfSPH``'s constructor;
``--subdomain-grid=on``). Doing it here also buys two things the built-in path
does not offer:

* the reconstruction runs **synchronously**. Chrono's Unix branch appends ``&`` to
  the ``system()`` command ("launch and forget"), so the mesh does not exist when
  the call returns -- unusable for a frame-by-frame render pipeline.
* we can **crop** to a render window. The eval bed (150 x 150 m at 0.08 m spacing)
  is ~11 M particles; a 20 x 14 m window around the vehicle is ~170 k, which
  reconstructs in well under a second instead of minutes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ChFsiSplashsurfSPH's constructor defaults (radius is set separately, from the spacing).
DEFAULT_SMOOTHING_LENGTH = 1.5
DEFAULT_CUBE_SIZE = 0.5
DEFAULT_SURFACE_THRESHOLD = 0.6


@dataclass(frozen=True)
class SurfaceReconstruction:
    """Result of one particle dump + reconstruction."""

    mesh_path: Path
    particle_path: Path
    num_particles: int
    num_particles_total: int
    particle_radius_m: float
    crop_min: tuple[float, float, float] | None
    crop_max: tuple[float, float, float] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mesh_path": str(self.mesh_path),
            "particle_path": str(self.particle_path),
            "num_particles": int(self.num_particles),
            "num_particles_total": int(self.num_particles_total),
            "particle_radius_m": float(self.particle_radius_m),
            "crop_min": list(self.crop_min) if self.crop_min is not None else None,
            "crop_max": list(self.crop_max) if self.crop_max is not None else None,
        }


def find_splashsurf(explicit: str | Path | None = None) -> Path:
    """Locate the ``splashsurf`` executable.

    Order: explicit argument, ``$SPLASHSURF_BIN``, ``$PATH``, then the usual install
    dirs. The last group matters because a non-interactive ssh session does not source
    ``.profile``, so ``~/.local/bin`` is missing from ``$PATH`` even where the binary
    is installed and works fine from a login shell.
    """
    candidates: list[str | Path] = []
    if explicit:
        candidates.append(explicit)
    env_bin = os.environ.get("SPLASHSURF_BIN")
    if env_bin:
        candidates.append(env_bin)
    which = shutil.which("splashsurf")
    if which:
        candidates.append(which)
    candidates.append(Path.home() / ".cargo" / "bin" / "splashsurf")
    candidates.append(Path.home() / ".local" / "bin" / "splashsurf")
    candidates.append(Path("/usr/local/bin/splashsurf"))

    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "splashsurf executable not found. Install it with `cargo install splashsurf` "
        "or point $SPLASHSURF_BIN at the binary."
    )


def fluid_particle_positions(terrain: Any) -> np.ndarray:
    """SPH *fluid* (soil) marker positions from a ``CRMTerrain``, as an (N, 3) array.

    ``GetParticlePositionsNumpy`` returns every marker -- soil particles first, then
    the boundary (container wall) and rigid-body (tire) BCE markers. Only the leading
    ``GetNumFluidMarkers()`` rows are soil; the BCE markers must be dropped or the
    reconstructed surface grows a box around the whole bed and shells around the tires.
    This mirrors Chrono's own ``writeParticleFileJSON``, which writes only the fluid
    block of ``referenceArray``.
    """
    fluid_system = terrain.GetFluidSystemSPH()
    positions = np.asarray(fluid_system.GetParticlePositionsNumpy(), dtype=np.float64)
    num_fluid = int(fluid_system.GetNumFluidMarkers())
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"unexpected particle position shape: {positions.shape}")
    if num_fluid > positions.shape[0]:
        raise ValueError(f"fluid marker count {num_fluid} exceeds particle array {positions.shape[0]}")
    return positions[:num_fluid]


def crop_positions(
    positions: np.ndarray,
    center_xy: tuple[float, float] | None,
    half_extent_xy: tuple[float, float] | None,
) -> tuple[np.ndarray, tuple[float, float, float] | None, tuple[float, float, float] | None]:
    """Keep only particles inside an axis-aligned XY window around ``center_xy``.

    The window is applied in XY only (the bed is thin in Z, and clipping it would
    open the bottom of the reconstructed volume).
    """
    if center_xy is None or half_extent_xy is None:
        return positions, None, None
    cx, cy = float(center_xy[0]), float(center_xy[1])
    hx, hy = abs(float(half_extent_xy[0])), abs(float(half_extent_xy[1]))
    keep = (
        (positions[:, 0] >= cx - hx)
        & (positions[:, 0] <= cx + hx)
        & (positions[:, 1] >= cy - hy)
        & (positions[:, 1] <= cy + hy)
    )
    kept = positions[keep]
    if kept.size == 0:
        raise ValueError(
            f"crop window x=[{cx - hx}, {cx + hx}] y=[{cy - hy}, {cy + hy}] contains no SPH particles"
        )
    lo = (cx - hx, cy - hy, float(kept[:, 2].min()))
    hi = (cx + hx, cy + hy, float(kept[:, 2].max()))
    return kept, lo, hi


def write_particles_xyz(positions: np.ndarray, path: str | Path) -> Path:
    """Write positions as splashsurf's "binary f32 XYZ" format (raw little-endian f32 triples)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(positions, dtype="<f4").tofile(path)
    return path


def run_splashsurf(
    particle_path: str | Path,
    mesh_path: str | Path,
    *,
    particle_radius: float,
    smoothing_length: float = DEFAULT_SMOOTHING_LENGTH,
    cube_size: float = DEFAULT_CUBE_SIZE,
    surface_threshold: float = DEFAULT_SURFACE_THRESHOLD,
    normals: bool = True,
    mesh_cleanup: bool = False,
    mesh_smoothing_iters: int = 0,
    mesh_smoothing_weights: bool = True,
    normals_smoothing_iters: int = 0,
    decimate_barnacles: bool = False,
    num_threads: int | None = None,
    extra_args: list[str] | None = None,
    splashsurf_bin: str | Path | None = None,
    quiet: bool = True,
) -> Path:
    """Run ``splashsurf reconstruct`` synchronously and return the written mesh path.

    The smoothing options exist for close-up renders. Raw marching cubes at a cube size
    near the particle spacing turns each surface particle into its own bump, which reads
    as a field of spikes rather than as soil once the camera is a few metres away instead
    of thirty. ``mesh_smoothing_weights`` makes the smoothing feature-preserving, so the
    rut walls survive iteration counts that would otherwise flatten them, and
    ``decimate_barnacles`` removes the bad triangle configurations that Laplacian
    smoothing turns into spikes.
    """
    if particle_radius <= 0:
        raise ValueError(f"particle_radius must be positive, got {particle_radius}")
    executable = find_splashsurf(splashsurf_bin)
    particle_path = Path(particle_path)
    mesh_path = Path(mesh_path)
    mesh_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(executable),
        "reconstruct",
        str(particle_path),
        "-r",
        f"{float(particle_radius)}",
        "-l",
        f"{float(smoothing_length)}",
        "-c",
        f"{float(cube_size)}",
        "-t",
        f"{float(surface_threshold)}",
        "--subdomain-grid=on",
        f"--normals={'on' if normals else 'off'}",
        f"--mesh-cleanup={'on' if mesh_cleanup else 'off'}",
        f"--decimate-barnacles={'on' if decimate_barnacles else 'off'}",
    ]
    if int(mesh_smoothing_iters) > 0:
        command += [
            "--mesh-smoothing-iters",
            str(int(mesh_smoothing_iters)),
            f"--mesh-smoothing-weights={'on' if mesh_smoothing_weights else 'off'}",
        ]
    if normals and int(normals_smoothing_iters) > 0:
        command += ["--normals-smoothing-iters", str(int(normals_smoothing_iters))]
    if num_threads is not None:
        command += ["-n", str(int(num_threads))]
    if quiet:
        command.append("-q")
    if extra_args:
        command += list(extra_args)
    command += ["-o", str(mesh_path)]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"splashsurf failed (exit {result.returncode}):\n"
            f"  command: {' '.join(command)}\n"
            f"  stdout: {result.stdout[-2000:]}\n"
            f"  stderr: {result.stderr[-2000:]}"
        )
    if not mesh_path.is_file():
        raise RuntimeError(f"splashsurf reported success but {mesh_path} was not written")
    return mesh_path


def export_crm_surface(
    terrain: Any,
    mesh_path: str | Path,
    *,
    initial_spacing_m: float,
    center_xy: tuple[float, float] | None = None,
    half_extent_xy: tuple[float, float] | None = None,
    particle_path: str | Path | None = None,
    keep_particle_file: bool = False,
    smoothing_length: float = DEFAULT_SMOOTHING_LENGTH,
    cube_size: float = DEFAULT_CUBE_SIZE,
    surface_threshold: float = DEFAULT_SURFACE_THRESHOLD,
    normals: bool = True,
    mesh_cleanup: bool = False,
    mesh_smoothing_iters: int = 0,
    mesh_smoothing_weights: bool = True,
    normals_smoothing_iters: int = 0,
    decimate_barnacles: bool = False,
    num_threads: int | None = None,
    splashsurf_bin: str | Path | None = None,
    quiet: bool = True,
) -> SurfaceReconstruction:
    """Dump the current soil particles and reconstruct their surface as a mesh.

    ``mesh_path`` extension picks the format (``.obj``, ``.ply``, ``.vtk``); use
    ``.obj`` for the Blender import path. ``initial_spacing_m`` is the CRM config's
    ``terrain.initial_spacing_m`` -- the particle radius follows Chrono's convention
    of half the spacing.
    """
    mesh_path = Path(mesh_path)
    positions = fluid_particle_positions(terrain)
    total = int(positions.shape[0])
    cropped, crop_min, crop_max = crop_positions(positions, center_xy, half_extent_xy)

    if particle_path is None:
        particle_path = mesh_path.with_suffix(".xyz")
    particle_path = Path(particle_path)
    write_particles_xyz(cropped, particle_path)
    try:
        run_splashsurf(
            particle_path,
            mesh_path,
            particle_radius=float(initial_spacing_m) / 2.0,
            smoothing_length=smoothing_length,
            cube_size=cube_size,
            surface_threshold=surface_threshold,
            normals=normals,
            mesh_cleanup=mesh_cleanup,
            mesh_smoothing_iters=mesh_smoothing_iters,
            mesh_smoothing_weights=mesh_smoothing_weights,
            normals_smoothing_iters=normals_smoothing_iters,
            decimate_barnacles=decimate_barnacles,
            num_threads=num_threads,
            splashsurf_bin=splashsurf_bin,
            quiet=quiet,
        )
    finally:
        if not keep_particle_file and particle_path.is_file():
            particle_path.unlink()

    return SurfaceReconstruction(
        mesh_path=mesh_path,
        particle_path=particle_path,
        num_particles=int(cropped.shape[0]),
        num_particles_total=total,
        particle_radius_m=float(initial_spacing_m) / 2.0,
        crop_min=crop_min,
        crop_max=crop_max,
    )
