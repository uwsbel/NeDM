#!/usr/bin/env python3
"""Render a scene.npz in Chrono (Chrono::Sensor, headless) into a 2 x 2 panel mp4.

A scene.npz describes everything drawn; this script knows no study. Its scene is written by a
study's scene script (``scripts/contact_nrd/arch_ablation/pool/scene.py``):

* ``v<j>``, ``f<j>``, ``color``: mesh j's vertices, faces and own colour; the first ``moving`` meshes
  are the system's bodies, the rest fixed meshes at ``static_pose`` (x, y, z, qw, qx, qy, qz).
* per panel p and copy c (a copy is the system once: the recorded run or a model's):
  ``poses`` [P, C, F, bodies, 7]; ``copy_color`` [P, C, 3] (NaN: the meshes' own colours);
  ``copy_opacity`` [P, C]; ``paths`` [P, C, T, 3], drawn as thin boxes ``path_width`` wide.
* per panel and frame: the camera ``eye`` and ``target`` [P, F, 3] and the ``caption`` text; per
  panel the ``label``. ``legend_name`` and ``legend_color`` for the legend; ``fps``, ``speed``;
  ``ground_z`` and ``ground_size`` of the concrete ground.

The panels sit 500 m apart in one Chrono world, each with its own camera.

    CUDA_VISIBLE_DEVICES=0 <python with pychrono> scripts/contact_nrd/arch_ablation/pool/chrono_video.py <scene.npz> <out.mp4>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pychrono as ch
import pychrono.sensor as sens
from PIL import Image, ImageDraw, ImageFont

W, H = 960, 540  # one panel
SPACING = 500.0  # metres between the panels' scenes


def vec(*xyz) -> ch.ChVector3d:
    return ch.ChVector3d(*map(float, xyz))


def frame(pose: np.ndarray, dx: float = 0.0) -> ch.ChFramed:
    """A 7-number pose (x, y, z, qw, qx, qy, qz), moved dx along x."""
    return ch.ChFramed(vec(pose[0] + dx, pose[1], pose[2]), ch.ChQuaterniond(*map(float, pose[3:])))


def facing(direction: np.ndarray) -> ch.ChQuaterniond:
    """The rotation turning +x towards ``direction`` (no roll)."""
    yaw = np.arctan2(direction[1], direction[0])
    pitch = np.arctan2(-direction[2], np.hypot(direction[0], direction[1]))
    return ch.QuatFromAngleZ(float(yaw)) * ch.QuatFromAngleY(float(pitch))


def view(eye: np.ndarray, target: np.ndarray, dx: float) -> ch.ChFramed:
    return ch.ChFramed(vec(eye[0] + dx, eye[1], eye[2]), facing(target - eye))


def write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """An OBJ with area-weighted vertex normals, for smooth shading."""
    normals = np.zeros_like(vertices)
    corners = vertices[faces]
    face_normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    for k in range(3):
        np.add.at(normals, faces[:, k], face_normals)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in normals]
    lines += [f"f {a}//{a} {b}//{b} {c}//{c}" for a, b, c in faces + 1]
    path.write_text("\n".join(lines) + "\n")


def mesh_shape(mesh, color, opacity: float) -> ch.ChVisualShapeTriangleMesh:
    shape = ch.ChVisualShapeTriangleMesh()
    shape.SetMesh(mesh)
    shape.SetColor(ch.ChColor(*map(float, color)))
    shape.SetOpacity(float(opacity))
    return shape


def path_strip(ground: ch.ChBody, points: np.ndarray, width: float, color, dx: float) -> None:
    """The path as thin boxes, one per ``1.5 width`` of its length."""
    distance = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    pts = points[np.unique(np.searchsorted(distance, np.r_[np.arange(0, distance[-1], 1.5 * width), distance[-1]]))]
    for a, b in zip(pts[:-1], pts[1:]):
        length = float(np.linalg.norm(b - a))
        if length < 1e-9:
            continue
        box = ch.ChVisualShapeBox(length + 0.5 * width, width, 0.2 * width)
        box.SetColor(ch.ChColor(*map(float, color)))
        mid = (a + b) / 2
        ground.AddVisualShape(box, ch.ChFramed(vec(mid[0] + dx, mid[1], mid[2]), facing(b - a)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    s = dict(np.load(args.scene))
    n_panels, n_copies, n_frames, moving, _ = s["poses"].shape
    fps = int(s["fps"])

    system = ch.ChSystemNSC()
    ground = ch.ChBody()
    ground.SetFixed(True)
    system.Add(ground)
    manager = sens.ChSensorManager(system)
    manager.scene.SetAmbientLight(ch.ChVector3f(0.35, 0.35, 0.35))
    tmp = tempfile.TemporaryDirectory()  # the renderer reads the files again later: kept to the end
    meshes = []
    for j in range(len(s["color"])):
        obj = Path(tmp.name) / f"{j}.obj"
        write_obj(obj, s[f"v{j}"], s[f"f{j}"])
        meshes.append(ch.ChTriangleMeshConnected.CreateFromWavefrontFile(str(obj), True, False))

    bodies, cameras = [], []
    size = float(s["ground_size"])
    for p in range(n_panels):
        dx = p * SPACING
        center = s["target"][p, 0]
        patch = ch.ChVisualShapeBox(size, size, 0.1)
        patch.SetTexture(ch.GetChronoDataFile("textures/concrete.jpg"), size / 3, size / 3)
        ground.AddVisualShape(patch, ch.ChFramed(vec(center[0] + dx, center[1], float(s["ground_z"]) - 0.05)))
        manager.scene.AddPointLight(ch.ChVector3f(float(center[0] + dx), float(center[1] - 20), 60.0), ch.ChColor(1, 1, 1), 1000)
        for j, pose in enumerate(s["static_pose"]):
            ground.AddVisualShape(mesh_shape(meshes[moving + j], s["color"][moving + j], 1.0), frame(pose, dx))
        copies = []
        for c in range(n_copies):
            tint = s["copy_color"][p, c]
            own = bool(np.isnan(tint).any())
            path_strip(ground, s["paths"][p, c], float(s["path_width"]), s["color"][0] if own else tint, dx)
            parts = []
            for b in range(moving):
                body = ch.ChBody()
                body.SetFixed(True)
                body.AddVisualShape(mesh_shape(meshes[b], s["color"][b] if own else tint, s["copy_opacity"][p, c]))
                system.Add(body)
                parts.append(body)
            copies.append(parts)
        bodies.append(copies)
        cam = sens.ChCameraSensor(ground, fps, view(s["eye"][p, 0], s["target"][p, 0], dx), W, H, 1.0)
        cam.PushFilter(sens.ChFilterRGBA8Access())
        manager.AddSensor(cam)
        cameras.append(cam)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [str(Path(sys.executable).with_name("ffmpeg")), "-loglevel", "error", "-y", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-s", f"{2 * W}x{2 * H}", "-r", str(fps), "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(args.out)],
        stdin=subprocess.PIPE,
    )
    font = ImageFont.load_default(size=22)
    speed = float(s["speed"])
    for f in list(range(n_frames)) + [n_frames - 1] * fps:  # one still second at the end
        for p in range(n_panels):
            for c in range(n_copies):
                for b, body in enumerate(bodies[p][c]):
                    pose = frame(s["poses"][p, c, f, b], p * SPACING)
                    body.SetPos(pose.GetPos())
                    body.SetRot(pose.GetRot())
            cameras[p].SetOffsetPose(view(s["eye"][p, f], s["target"][p, f], p * SPACING))
        system.DoStepDynamics(1.0 / fps)
        manager.Update()
        canvas = Image.new("RGB", (2 * W, 2 * H))
        for p, cam in enumerate(cameras):
            panel = Image.fromarray(np.flipud(cam.GetMostRecentRGBA8Buffer().GetRGBA8Data())[:, :, :3])
            draw = ImageDraw.Draw(panel)
            draw.text((16, 12), str(s["label"][p]), fill="black", font=font)
            draw.text((16, 40), str(s["caption"][p, f]), fill="black", font=font)
            canvas.paste(panel, ((p % 2) * W, (p // 2) * H))
        draw = ImageDraw.Draw(canvas)
        x = 16
        for name, color in zip(s["legend_name"], s["legend_color"]):
            draw.rectangle((x, 2 * H - 34, x + 22, 2 * H - 14), fill=tuple(int(255 * v) for v in color))
            draw.text((x + 30, 2 * H - 38), str(name), fill="black", font=font)
            x += 60 + int(draw.textlength(str(name), font=font))
        if speed != 1.0:
            draw.text((x, 2 * H - 38), f"(played at {speed:g}x)", fill="black", font=font)
        encoder.stdin.write(canvas.tobytes())
    encoder.stdin.close()
    encoder.wait()
    tmp.cleanup()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
