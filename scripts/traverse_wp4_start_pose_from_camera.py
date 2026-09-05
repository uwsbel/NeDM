#!/usr/bin/env python
"""Start pose from the camera (the last privileged input of the imagination chain).

For each held-out episode, run the WP4 pose head on the camera frame at the rollout start
(frame = context) and write the estimated (x, y, yaw) next to the recorded truth. The scorer's
``--start-poses`` option then starts the imagined rollouts from the estimate instead of the
recorded pose.
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nedm.traverse import nrd_data as D, perception as P
from nedm.traverse.camera import CameraModel
from nedm.traverse.storage import EpisodeReader
from nedm.traverse.terrain import TerrainMap
from traverse_wp4_train_posehead import PoseHead, STAGE, pixel_to_world, stage_to_img

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--out", default="artifacts/traverse/wp4_start_poses/val_start_poses.json")
ap.add_argument("--posehead", default="artifacts/traverse/wp4_posehead_v1_amd/ckpt_best.pt")
ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
ap.add_argument("--routes", default="artifacts/traverse/wp3_routes")
ap.add_argument("--stores", default="artifacts/traverse")
ap.add_argument("--split", default="val")
ap.add_argument("--families", nargs="+", default=["oracle"])
ap.add_argument("--episodes", type=int, default=32)
ap.add_argument("--frame", type=int, default=16, help="rollout start frame (= model context)")
args = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
keys = D.load_cache_keys(Path(args.cache))
split = dict(zip(("train", "val", "test"), D.split_keys(keys)))[args.split]
manifest = json.loads((Path(args.routes) / "routes_manifest.json").read_text())
allowed = set().union(*(set(manifest["families"][f]) for f in args.families))
keys = [k for k in split if k in allowed][: args.episodes]
payload = torch.load(args.posehead, map_location=dev, weights_only=False)
enc = P.Encoder(z_dim=256, n_q=8).to(dev)
enc.load_state_dict(torch.load(payload["encoder_ckpt"], map_location=dev, weights_only=False)["encoder"]); enc.eval()
stem = enc.backbone[:STAGE]
head = PoseHead(width=payload["config"]["width"]).to(dev); head.load_state_dict(payload["head"]); head.eval()
arena = Path("assets/traverse/arena_v1"); tmap, cam = TerrainMap.from_dir(arena), CameraModel()
ds_helper = P.WP1FrameDataset([], arena)  # for _z_map
out, errs, yerrs = {}, [], []
for key in keys:
    store, ep = key.split("__", 1)
    reader = EpisodeReader(Path(args.stores) / store / ep)
    win = reader.read_window(args.frame, 1); reader.close()
    fi = {n: i for i, n in enumerate(win["state_fields"])}
    st = win["states"][0]
    x_t, y_t, yaw_t = float(st[fi["pos_x_m"]]), float(st[fi["pos_y_m"]]), float(st[fi["yaw_rad"]])
    rgb = win["rgb"][0].astype(np.float32) / 255.0
    z = ds_helper._z_map(win["depth_mm"][0])
    inp = torch.from_numpy(np.concatenate([rgb.transpose(2, 0, 1), z[None]], 0))[None].to(dev)
    with torch.no_grad():
        _, u_s, v_s, yaw = head(stem(inp))
    u, v = stage_to_img(u_s.cpu().numpy(), v_s.cpu().numpy())
    x, y = pixel_to_world(cam, tmap, u, v)
    yaw_p = math.atan2(float(yaw[0, 0]), float(yaw[0, 1]))
    e = math.hypot(x[0] - x_t, y[0] - y_t); ye = abs((yaw_p - yaw_t + math.pi) % (2 * math.pi) - math.pi)
    errs.append(e); yerrs.append(math.degrees(ye))
    out[key] = {"est": [float(x[0]), float(y[0]), yaw_p], "true": [x_t, y_t, yaw_t], "err_m": e, "err_deg": math.degrees(ye)}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, indent=1))
print(f"{len(keys)} start poses from the camera: xy error mean {np.mean(errs):.3f} m max {np.max(errs):.3f}; yaw mean {np.mean(yerrs):.2f} deg max {np.max(yerrs):.2f} -> {args.out}")
