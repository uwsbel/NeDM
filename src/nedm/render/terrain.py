"""What the ground looks like, chosen separately from the model.

A terrain is appearance only. The NRD model never sees it, nothing is simulated on it and it
does not deform: a robot's feet go where the reduced state puts them, whatever is drawn
underneath. Changing the terrain changes the pictures and nothing else.

    scene.add_ground(terrain="soil")                                    # by name
    scene.add_ground(terrain=terrain.soil(roughness=0.01, seed=3))      # with options
    scene.add_ground(terrain=terrain.heightmap(z, spacing=0.05))        # any relief you have

Every terrain is a :class:`~nedm.render.scene.Ground`: a plane, and optionally one static
mesh lying on it. That is all a backend draws, so a new terrain needs no backend code.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from nedm.render.scene import Color, Ground, MeshBody

SOIL: Color = (0.62, 0.50, 0.36)
CLIP = 2.5                      # soil relief is cut off at this many times its RMS height


def checker(**options) -> Ground:
    """A plane with a checker of tiles: the default, and the one that shows motion best."""
    return Ground(name="checker", **options)


def plain(color: Color = Ground.color) -> Ground:
    """A bare plane in one color."""
    return Ground(name="plain", color=color, tile_size=None)


def heightmap(heights: np.ndarray, spacing: float, origin: tuple[float, float] | None = None,
              color: Color = SOIL, name: str = "heightmap") -> Ground:
    """Any static relief: ``heights[i, j]`` in metres above the ground height, at
    ``x = origin[0] + i * spacing`` and ``y = origin[1] + j * spacing``.

    Without ``origin`` the map is centered on the world origin. Outside the map the ground is
    a plane just below its lowest point, in the same color.
    """
    z = np.asarray(heights, dtype=np.float32)
    nx, ny = z.shape
    x0, y0 = origin if origin is not None else (-0.5 * (nx - 1) * spacing, -0.5 * (ny - 1) * spacing)
    x, y = np.meshgrid(x0 + spacing * np.arange(nx), y0 + spacing * np.arange(ny), indexing="ij")
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float32)
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    v00 = (i * ny + j).reshape(-1)
    v10, v01, v11 = v00 + ny, v00 + 1, v00 + ny + 1
    faces = np.concatenate([np.stack([v00, v10, v11], axis=1), np.stack([v00, v11, v01], axis=1)])     # both face up
    relief = MeshBody(name, vertices=vertices, faces=faces.astype(np.int32), color=color)
    return Ground(name=name, color=color, tile_size=None, relief=relief)


def soil(extent: tuple[float, float, float, float] = (-9.0, 9.0, -9.0, 9.0), spacing: float = 0.04,
         roughness: float = 0.004, lump_size: float = 0.10, crumbs: float = 0.25, seed: int = 0,
         color: Color = SOIL, skirt: float = 200.0) -> Ground:
    """Loose soil as a static relief: lumps about ``lump_size`` across, ``roughness`` metres RMS.

    ``crumbs`` is how much finer detail, at a third of the lump size, is added on top (0 for none).
    ``spacing`` is the grid step: 0.04 m is enough for small images, and close-ups want 0.02 m,
    which is four times the triangles. The relief fades to flat over the outer fifth of
    ``extent``, and flat ground of the same mesh carries on ``skirt`` metres beyond it, so a
    robot that walks off the relief sees no edge and no change of shade.

    The relief has zero mean, so the surface sits at the ground height on average and a foot
    resting at that height is drawn pressed a little into the lumps. Keep ``roughness`` well
    below the size of what stands on it: nothing lifts a foot over a lump.
    """
    x0, x1, y0, y1 = extent
    nx, ny = int(round((x1 - x0) / spacing)) + 1, int(round((y1 - y0) / spacing)) + 1
    white = np.random.default_rng(seed).standard_normal((nx, ny))
    frequency = np.hypot(np.fft.fftfreq(nx, d=spacing)[:, None], np.fft.rfftfreq(ny, d=spacing)[None, :])
    spectrum = sum(weight * np.exp(-0.5 * ((frequency - 1.0 / size) / (0.5 / size)) ** 2)
                   for size, weight in ((lump_size, 1.0), (lump_size / 3.0, crumbs)))
    z = np.fft.irfft2(np.fft.rfft2(white) * spectrum, s=(nx, ny))
    z -= z.mean()
    z *= roughness / z.std()
    z = np.clip(z, -CLIP * roughness, CLIP * roughness)
    z *= _edge_fade(nx)[:, None] * _edge_fade(ny)[None, :]
    ground = heightmap(z, spacing, origin=(x0, y0), color=color, name="soil")
    if skirt > 0:
        relief = ground.relief
        relief.vertices, relief.faces = _with_skirt(relief.vertices, relief.faces, (x0, x1, y0, y1), skirt)
    return ground


def _with_skirt(vertices: np.ndarray, faces: np.ndarray, extent, reach: float) -> tuple[np.ndarray, np.ndarray]:
    """Add a flat frame at height 0 around ``extent``, ``reach`` metres wide: 8 upward triangles."""
    x0, x1, y0, y1 = extent
    inner = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]                   # counter-clockwise seen from above
    outer = [(x0 - reach, y0 - reach), (x1 + reach, y0 - reach), (x1 + reach, y1 + reach), (x0 - reach, y1 + reach)]
    first = len(vertices)
    corners = np.array([(x, y, 0.0) for x, y in outer + inner], dtype=np.float32)
    frame = []
    for k in range(4):
        a, b = first + k, first + (k + 1) % 4                          # outer edge
        c, d = first + 4 + (k + 1) % 4, first + 4 + k                  # the inner edge facing it
        frame += [(a, b, c), (a, c, d)]
    return np.concatenate([vertices, corners]), np.concatenate([faces, np.array(frame, dtype=faces.dtype)])


def _edge_fade(n: int, fraction: float = 0.2) -> np.ndarray:
    """1 in the middle of ``n`` samples, falling smoothly to 0 at both ends over ``fraction`` of them."""
    ramp = np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) / max(fraction * (n - 1), 1.0), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * ramp)


TERRAINS: dict[str, Callable[..., Ground]] = {"checker": checker, "plain": plain, "soil": soil}


def make(terrain: str | Ground = "checker", **options) -> Ground:
    """A terrain by name, with that terrain's options, or a :class:`Ground` passed through."""
    if isinstance(terrain, Ground):
        if options:
            raise TypeError("options go to the function that builds the terrain, not next to a finished one")
        return terrain
    if terrain not in TERRAINS:
        raise ValueError(f"unknown terrain {terrain!r}, choose from {sorted(TERRAINS)}")
    return TERRAINS[terrain](**options)
