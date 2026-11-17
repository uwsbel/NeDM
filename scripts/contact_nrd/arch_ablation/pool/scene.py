#!/usr/bin/env python3
"""The pool video's scene, from rollouts.npz, for ``scripts/contact_nrd/arch_ablation/pool/chrono_video.py``: writes scene.npz.

One test shot, the one at the median of ball B's error at 2.0 s (mean of the four models), in four
panels, one per model: the shot as Chrono recorded it (cue ball A white, ball B red, solid) and the
model's rollout of it (see-through, in the model's colour), with ball B's paths. The table is the
study's (``system.json``): 2.54 m x 1.27 m, balls of radius 28.575 mm, a 10 cm grid drawn on the felt.
The errors are millimetres, so each camera follows the recorded ball B from half a metre away.

    python scripts/contact_nrd/arch_ablation/pool/scene.py
    CUDA_VISIBLE_DEVICES=0 <python with pychrono> scripts/contact_nrd/arch_ablation/pool/chrono_video.py \\
        artifacts/contact_nrd/arch_ablation/pool/scene.npz artifacts/contact_nrd/arch_ablation/videos/pool_arch.mp4
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from figures import MODELS, errors

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # layout.py is one folder up
import layout  # noqa: E402

RESULTS = layout.RESULTS / "pool"
FPS, SPEED = 25, 0.25
BALL_COLOR = [(0.93, 0.93, 0.9), (0.78, 0.12, 0.1)]  # A white, B red
FELT, CUSHION, GRID = (0.16, 0.19, 0.24), (0.1, 0.12, 0.16), (0.32, 0.36, 0.42)  # slate: every model colour shows on it


def sphere(radius: float, n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """A UV sphere: vertices [V, 3], faces [F, 3]."""
    theta, phi = np.meshgrid(np.linspace(0, np.pi, n + 1), np.linspace(0, 2 * np.pi, 2 * n, endpoint=False), indexing="ij")
    v = radius * np.stack([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], -1).reshape(-1, 3)
    idx = np.arange(v.shape[0]).reshape(n + 1, 2 * n)
    a, b = idx[:-1], np.roll(idx[:-1], -1, axis=1)
    c, e = idx[1:], np.roll(idx[1:], -1, axis=1)
    faces = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, e], -1).reshape(-1, 3)])
    return v.astype(np.float32), faces.astype(np.int32)


def box(center, size) -> tuple[np.ndarray, np.ndarray]:
    """An axis-aligned box: vertices [8, 3], faces [12, 3]."""
    corners = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
             [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return (corners * size + center).astype(np.float32), np.array(faces, np.int32)


def table(scene: dict) -> list[tuple]:
    """(mesh, colour) of the felt, the cushions and the grid, in the world (the felt's top at z = 0)."""
    lx, ly = scene["half_length_m"], scene["half_width_m"]
    h, t = 2 * scene["cushion_nose_height_m"] - scene["radius_m"], scene["cushion_thickness_m"]
    parts = [(box((0, 0, -0.05), (2 * lx + 2 * t, 2 * ly + 2 * t, 0.1)), FELT)]
    for sx in (-1, 1):
        parts.append((box((sx * (lx + t / 2), 0, h / 2), (t, 2 * ly + 2 * t, h)), CUSHION))
    for sy in (-1, 1):
        parts.append((box((0, sy * (ly + t / 2), h / 2), (2 * lx, t, h)), CUSHION))
    parts += [(box((x, 0, 0.0002), (0.002, 2 * ly, 0.0004)), GRID) for x in np.arange(-lx + 0.1, lx, 0.1)]
    parts += [(box((0, y, 0.0002), (2 * lx, 0.002, 0.0004)), GRID) for y in np.arange(-ly + 0.1, ly, 0.1)]
    return parts


def main() -> None:
    d = dict(np.load(RESULTS / "rollouts.npz"))
    scene = json.loads(str(d["system"]))["physics_config"]["scene"]
    dt, k = float(d["dt"]), int(d["target_step"])
    models = list(MODELS)
    score = np.mean([errors(d, m)[:, k, 1] for m in models], axis=0)
    shot = int(np.argmin(np.abs(score - np.median(score))))
    n_steps = d["GT"].shape[1]
    frames = np.minimum(np.round(np.arange(int((n_steps - 1) * dt / SPEED * FPS)) * SPEED / FPS / dt).astype(int), n_steps - 1)

    ball = sphere(scene["radius_m"])
    meshes = [(ball, BALL_COLOR[0]), (ball, BALL_COLOR[1]), *table(scene)]
    out: dict = {f"v{j}": m[0] for j, (m, _) in enumerate(meshes)}
    out |= {f"f{j}": m[1] for j, (m, _) in enumerate(meshes)}
    out["color"] = np.array([c for _, c in meshes], np.float32)
    out["moving"] = 2
    out["static_pose"] = np.array([[0, 0, 0, 1, 0, 0, 0]] * (len(meshes) - 2), np.float32)

    def poses(positions: np.ndarray) -> np.ndarray:  # [F, 2, 3] -> [F, 2, 7], no turn drawn
        q = np.zeros(positions.shape[:2] + (4,), np.float32)
        q[..., 0] = 1
        return np.concatenate([positions, q], -1)

    gt_b = d["GT"][shot, frames, 1]
    look = np.array([1.0, 1.0, -1.1]) / np.linalg.norm([1.0, 1.0, -1.1])  # from behind-left, above
    colors = {m: [int(MODELS[m][1][i:i + 2], 16) / 255 for i in (1, 3, 5)] for m in models}
    out |= {
        "poses": np.array([[poses(d["GT"][shot, frames]), poses(d[m][shot, frames])] for m in models], np.float32),
        "copy_color": np.array([[[np.nan] * 3, colors[m]] for m in models], np.float32),
        "copy_opacity": np.array([[1.0, 0.45]] * len(models), np.float32),
        "paths": np.array([[d["GT"][shot, :, 1] * [1, 1, 0] + [0, 0, 0.001],
                            d[m][shot, :, 1] * [1, 1, 0] + [0, 0, 0.0012]] for m in models], np.float32),
        "path_width": 0.004,
        "eye": np.repeat((gt_b - 0.5 * look)[None], len(models), 0),
        "target": np.repeat(gt_b[None], len(models), 0),
        "label": np.array([MODELS[m][0] for m in models]),
        "caption": np.array([[f"t = {(f * dt):4.2f} s   ball B error {errors(d, m)[shot, f, 1]:5.2f} mm" for f in frames]
                             for m in models]),
        "legend_name": np.array(["Chrono: ball A white, ball B red", *(m for m in models)]),
        "legend_color": np.array([BALL_COLOR[1], *(colors[m] for m in models)], np.float32),
        "ground_z": -0.8, "ground_size": 8.0, "fps": FPS, "speed": SPEED,
    }
    np.savez_compressed(RESULTS / "scene.npz", **out)
    print(f"shot {shot}: ball B error at 2 s " + ", ".join(f"{m} {errors(d, m)[shot, k, 1]:.2f} mm" for m in models)
          + f"; {len(frames)} frames; wrote {RESULTS / 'scene.npz'}")


if __name__ == "__main__":
    main()
