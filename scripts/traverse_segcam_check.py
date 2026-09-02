#!/usr/bin/env python
"""One-shot ChSegmentationCamera validation of the analytic class masks (G0b).

Builds one episode scene (same path as collection), assigns semantic class
IDs to every visual material post-build (collection code untouched):
0 terrain, 1 rock, 2 tree, 3 house, 4 vehicle. Renders a single segmentation
frame and reports per-class IoU vs ``nedm.traverse.masks.label_image``.

Run on newton (OptiX):
  PYTHONPATH=src python scripts/traverse_segcam_check.py --seed 20260910
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

SETTLE_S = 0.8


def set_class(body, class_id: int) -> None:
    import pychrono as chrono

    model = body.GetVisualModel()
    if model is None:
        return
    for i in range(model.GetNumShapes()):
        shape = body.GetVisualShape(i)
        if shape.GetNumMaterials() == 0:
            shape.AddMaterial(chrono.ChVisualMaterial())
        for j in range(shape.GetNumMaterials()):
            mat = shape.GetMaterial(j)
            mat.SetClassID(class_id)
            mat.SetInstanceID(class_id)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arena", default="assets/traverse/arena_v1")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--res", type=int, default=256)
    parser.add_argument("--min-iou", type=float, default=0.75)
    args = parser.parse_args()

    import pychrono as chrono
    import pychrono.sensor as sens

    from nedm.traverse.camera import CameraModel
    from nedm.traverse.layout import sample_episode
    from nedm.traverse.masks import CLASS_NAMES, label_image
    from nedm.traverse.scene import RenderSpec, build_config, build_scene, overhead_camera_pose
    from nedm.traverse.terrain import TerrainMap

    arena_dir = (REPO_ROOT / args.arena).resolve()
    tmap = TerrainMap.from_dir(arena_dir)
    layout, plan = sample_episode(tmap, "segcheck", args.seed)
    start_z = float(tmap.height(*layout.start_xy)) + 0.75
    config = build_config(arena_dir, (*layout.start_xy, start_z), layout.start_yaw)
    render = RenderSpec(width=args.res, height=args.res, plan_markers=False)
    scene = build_scene(config, layout, tmap, arena_dir, plan=None, render=render)
    system = scene.system

    class_of = {"rock": 1, "tree": 2, "house": 3}
    asset_ids = set()
    set_class(scene.patch_body, 0)
    for asset, body in scene.asset_bodies:
        set_class(body, class_of[asset.kind])
        asset_ids.add(body.GetIdentifier())
    patch_id = scene.patch_body.GetIdentifier()
    for body in system.GetBodies():  # everything else (chassis, wheels, ...) = vehicle
        if body.GetIdentifier() not in asset_ids and body.GetIdentifier() != patch_id:
            set_class(body, 4)

    trigger_rate_hz = 1.0 / float(config["simulation"]["step_size_s"])
    seg = sens.ChSegmentationCamera(
        scene.patch_body,
        trigger_rate_hz,
        overhead_camera_pose(render.cam_height_m),
        args.res,
        args.res,
        render.hfov_rad,
    )
    seg.SetName("overhead_seg")
    seg.SetLag(0.0)
    seg.SetCollectionWindow(0.0)
    seg.PushFilter(sens.ChFilterSemanticAccess())
    scene.manager.AddSensor(seg)

    # Settle with brakes so the pose matches recorded frame 0, then render once.
    dt = float(config["simulation"]["step_size_s"])
    hmmwv, terrain = scene.hmmwv, scene.terrain
    driver_inputs = None
    import pychrono.vehicle as veh

    driver_inputs = veh.DriverInputs()
    driver_inputs.m_steering = 0.0
    driver_inputs.m_throttle = 0.0
    driver_inputs.m_braking = 1.0
    for _ in range(int(round(SETTLE_S / dt))):
        ts = float(system.GetChTime())
        terrain.Synchronize(ts)
        hmmwv.Synchronize(ts, driver_inputs, terrain)
        terrain.Advance(dt)
        hmmwv.Advance(dt)

    scene.manager.Update()
    scene.rgb_tap.take()
    if scene.depth_tap is not None:
        scene.depth_tap.take()
    import time

    seg_img = None
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        buf = seg.GetMostRecentSemanticBuffer()
        if buf.HasData():
            data = buf.GetSemanticData()
            arr = np.asarray(data)
            seg_img = arr
            break
        time.sleep(0.001)
    if seg_img is None:
        print("FAIL: no semantic buffer arrived")
        return 1
    print("semantic buffer dtype/shape:", seg_img.dtype, seg_img.shape)

    # Extract class channel (structured or plain), flip rows (bottom-up buffer).
    if seg_img.dtype.names:
        name = [n for n in seg_img.dtype.names if "class" in n.lower()]
        cls = seg_img[name[0]] if name else seg_img[seg_img.dtype.names[0]]
    else:
        cls = seg_img
    cls = np.ascontiguousarray(np.squeeze(cls)[::-1, :]).astype(np.int32)

    ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
    x, y = ref.GetPos().x, ref.GetPos().y
    yaw = float(ref.GetRot().GetCardanAnglesZYX().z)
    cam = CameraModel(width=args.res, height=args.res, hfov_rad=render.hfov_rad)
    analytic = label_image(
        layout.to_json(), tmap.height, cam, vehicle_pose=(x, y, float(tmap.height(x, y)), yaw)
    )

    print("segcam class counts:", dict(zip(*[a.tolist() for a in np.unique(cls, return_counts=True)])))
    print("analytic class counts:", dict(zip(*[a.tolist() for a in np.unique(analytic, return_counts=True)])))
    worst = 1.0
    for cid, cname in CLASS_NAMES.items():
        a = analytic == cid
        b = cls == cid
        union = (a | b).sum()
        iou = (a & b).sum() / union if union else float("nan")
        if union:
            worst = min(worst, iou)
        print(f"  class {cid} {cname:8s}: IoU={iou:.3f}  seg_px={int(b.sum())} analytic_px={int(a.sum())}")
    out = {"seed": args.seed, "worst_iou": worst, "pass": bool(worst >= args.min_iou)}
    print(json.dumps(out))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
