#!/usr/bin/env python
"""Live planning inputs for the terrain challenges: one frame-0 dump per challenge -> planner cache + start poses.

``traverse_wp3_chrono_eval.py --tasks-file`` (with ``dump_frame0``) saves, for each challenge, the fixed camera's
frame at t = 0, the 17-D rest state and the true pose. This script turns those into what the sampling planner
needs, exactly as ``traverse_wp5_live_inputs.py`` does for recorded layouts: the vehicle is masked out of its
own frame at the camera-estimated pose, the frame is encoded with the WP1 stem into the scene map, and a
self-contained cache entry is written (``z1`` = the rest state tiled, ``act`` = brake, ``pose`` = the true pose
tiled, ``map_v2`` = the single-frame map). Start poses come from the pose head on the same frame.
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nedm.traverse import perception as P
from nedm.traverse.camera import CameraModel
from nedm.traverse.terrain import TerrainMap
from traverse_wp2_encode_map import MAP_STAGE, EpisodeMedian
from traverse_wp4_train_posehead import PoseHead, STAGE, pixel_to_world, stage_to_img
from traverse_wp5_live_inputs import fill_masked, vehicle_mask_xy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--challenges", default="artifacts/traverse/wp6_challenges")
    ap.add_argument("--out", default="artifacts/traverse/wp6_challenge_cache")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--encoder", default="artifacts/traverse/wp1_v6/ckpt_warmup.pt")
    ap.add_argument("--posehead", default="artifacts/traverse/wp4_posehead_v1_amd/ckpt_best.pt")
    ap.add_argument("--frames", type=int, default=400, help="tiled length of the rest state (the env reads frames 0..context)")
    ap.add_argument("--norm-arena", default="assets/traverse/arena_v1",
                    help="arena whose height range normalises the encoder's elevation channel: the encoder's TRAINING arena, whatever is planned on")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tmap, cam = TerrainMap.from_dir(Path(args.arena)), CameraModel()
    enc = P.Encoder(z_dim=256, n_q=8).to(dev)
    enc.load_state_dict(torch.load(args.encoder, map_location=dev, weights_only=False)["encoder"], strict=True); enc.eval()
    stem = enc.backbone[:MAP_STAGE]
    payload = torch.load(args.posehead, map_location=dev, weights_only=False)
    pstem = enc.backbone[:STAGE]
    head = PoseHead(width=payload["config"]["width"]).to(dev); head.load_state_dict(payload["head"]); head.eval()
    helper = EpisodeMedian([], Path("artifacts/traverse"), Path(args.arena))  # vehicle mask: true ground under the vehicle
    norm_helper = EpisodeMedian([], Path("artifacts/traverse"), Path(args.norm_arena))  # elevation channel: training normalisation
    ds_helper = P.WP1FrameDataset([], Path(args.norm_arena))  # pose head: elevation channel normalised with its TRAINING arena (fixed 2026-09-06)
    keys, poses, errs = [], {}, []
    for f in sorted(Path(args.challenges).glob("*/frame0.npz")):
        key = f.parent.name
        with np.load(f) as d:
            rgb_u8, depth_mm, z1, pose = d["rgb"], d["depth_mm"], d["z1"], d["pose"]
        rgb = rgb_u8.astype(np.float32) / 255.0
        # start pose from the camera (pose head on the raw frame)
        inp = torch.from_numpy(np.concatenate([rgb.transpose(2, 0, 1), ds_helper._z_map(depth_mm)[None]], 0))[None].to(dev)
        with torch.no_grad():
            _, u_s, v_s, yaw = head(pstem(inp))
        u, v = stage_to_img(u_s.cpu().numpy(), v_s.cpu().numpy())
        x, y = pixel_to_world(cam, tmap, u, v)
        est = (float(x[0]), float(y[0]), math.atan2(float(yaw[0, 0]), float(yaw[0, 1])))
        err = math.hypot(est[0] - pose[0], est[1] - pose[1]); errs.append(err)
        poses[key] = {"est": list(est), "true": [float(v) for v in pose], "err_m": err,
                      "err_deg": math.degrees(abs((est[2] - pose[2] + math.pi) % (2 * math.pi) - math.pi))}
        # scene map from the single frame, vehicle masked at the ESTIMATED pose
        mask = vehicle_mask_xy(helper, *est)
        elev = norm_helper._elevation(depth_mm)
        rgb_f, elev_f = fill_masked(rgb, mask), fill_masked(elev, mask)
        minp = torch.from_numpy(np.concatenate([rgb_f.transpose(2, 0, 1), elev_f[None]]).astype(np.float32))[None].to(dev)
        with torch.no_grad():
            scene_map = stem(minp).float().cpu().numpy().astype(np.float16)[0]
        T = args.frames
        np.savez(out / f"{key}.npz", z1=np.tile(z1[None], (T, 1)).astype(np.float32), act=np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (T, 1)),
                 pose=np.tile(pose[None], (T, 1)).astype(np.float32), map_v2=scene_map, z2=np.zeros((T, 256), np.float32), power=np.zeros((T, 1), np.float32))
        keys.append(key)
        print(f"{key}: rest vx {z1[0]:+.2f} pitch {math.degrees(z1[3]):+.1f} deg, tire loads {np.round(z1[7:11]).astype(int)} N | camera pose err {err:.3f} m {poses[key]['err_deg']:.2f} deg | masked {int(mask.sum())} px")
    (out / "cache_manifest.json").write_text(json.dumps({"episodes": keys, "source": args.challenges, "encoder": args.encoder, "vehicle_masked": True,
                                                         "arena": args.arena, "elevation_norm_arena": args.norm_arena}))
    (out / "start_poses.json").write_text(json.dumps(poses, indent=1))
    print(f"{len(keys)} challenge cache entries -> {out}; camera start pose error mean {np.mean(errs):.3f} m max {np.max(errs):.3f} m")


if __name__ == "__main__":
    main()
