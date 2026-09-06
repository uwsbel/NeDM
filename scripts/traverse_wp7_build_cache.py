#!/usr/bin/env python
"""Schema-v2 dynamics cache from the plan §28 collection: every recorded run, whatever its outcome, on many arenas.

Input: collection dirs written by ``traverse_wp3_chrono_eval.py --tasks-file`` with ``record`` (``records/<layout>__<run>.npz``:
20 Hz z1 / act / pose / power until the episode ended, status, route) and one frame-0 camera dump per layout
(``frame0/<layout>.npz``), plus ``rows.jsonl`` (per-run Chrono metrics = the episode labels).

Per episode the cache holds the recorded rows only (the trainer pads and masks), the layout's single-frame scene map
built exactly as the planner's live input (vehicle masked at the pose-head estimate, filled, encoded with the frozen
stem, elevation normalised with the encoder's training arena), ``arena`` and ``status``. The manifest maps episodes to
arenas (the split unit), outcomes and layout kinds; ``labels.json`` carries the per-run Chrono metrics for the cheap
predictor; ``start_poses.json`` the camera start-pose estimates.
"""
from __future__ import annotations

import argparse, json, math, sys, time
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


class LiveMapper:
    """frame-0 dump -> (scene map, camera start-pose estimate) for one arena."""

    def __init__(self, arena: Path, norm_arena: Path, encoder: Path, posehead: Path, dev: str):
        self.dev, self.arena = dev, arena
        self.tmap, self.cam = TerrainMap.from_dir(arena), CameraModel()
        enc = P.Encoder(z_dim=256, n_q=8).to(dev)
        enc.load_state_dict(torch.load(encoder, map_location=dev, weights_only=False)["encoder"], strict=True); enc.eval()
        self.stem, self.pstem = enc.backbone[:MAP_STAGE], enc.backbone[:STAGE]
        payload = torch.load(posehead, map_location=dev, weights_only=False)
        self.head = PoseHead(width=payload["config"]["width"]).to(dev); self.head.load_state_dict(payload["head"]); self.head.eval()
        self.helper = EpisodeMedian([], Path("artifacts/traverse"), arena)  # vehicle mask: true ground under the vehicle
        self.norm_helper = EpisodeMedian([], Path("artifacts/traverse"), norm_arena)  # elevation channel: training normalisation
        self.ds_helper = P.WP1FrameDataset([], norm_arena)  # pose head input: training normalisation too

    def __call__(self, frame0: Path) -> tuple[np.ndarray, dict]:
        with np.load(frame0) as d:
            rgb_u8, depth_mm, pose = d["rgb"], d["depth_mm"], d["pose"]
        rgb = rgb_u8.astype(np.float32) / 255.0
        inp = torch.from_numpy(np.concatenate([rgb.transpose(2, 0, 1), self.ds_helper._z_map(depth_mm)[None]], 0))[None].to(self.dev)
        with torch.no_grad():
            _, u_s, v_s, yaw = self.head(self.pstem(inp))
        u, v = stage_to_img(u_s.cpu().numpy(), v_s.cpu().numpy())
        x, y = pixel_to_world(self.cam, self.tmap, u, v)
        est = (float(x[0]), float(y[0]), math.atan2(float(yaw[0, 0]), float(yaw[0, 1])))
        mask = vehicle_mask_xy(self.helper, *est)
        elev = self.norm_helper._elevation(depth_mm)
        rgb_f, elev_f = fill_masked(rgb, mask), fill_masked(elev, mask)
        minp = torch.from_numpy(np.concatenate([rgb_f.transpose(2, 0, 1), elev_f[None]]).astype(np.float32))[None].to(self.dev)
        with torch.no_grad():
            scene_map = self.stem(minp).float().cpu().numpy().astype(np.float16)[0]
        info = {"est": list(est), "true": [float(p) for p in pose], "err_m": math.hypot(est[0] - pose[0], est[1] - pose[1]),
                "err_deg": math.degrees(abs((est[2] - pose[2] + math.pi) % (2 * math.pi) - math.pi)), "masked_px": int(mask.sum())}
        return scene_map, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collections", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", default="artifacts/traverse/wp1_v6/ckpt_warmup.pt")
    ap.add_argument("--posehead", default="artifacts/traverse/wp4_posehead_v1_amd/ckpt_best.pt")
    ap.add_argument("--norm-arena", default="assets/traverse/arena_v1")
    ap.add_argument("--min-frames", type=int, default=20, help="drop runs shorter than this (nothing to learn from)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    episodes, arena_of, status_of, kind_of, arenas, labels, poses = [], {}, {}, {}, {}, {}, {}
    mappers: dict[str, LiveMapper] = {}
    t0 = time.time()
    for coll in args.collections:
        coll = Path(coll)
        rows = {}
        for line in (coll / "rows.jsonl").read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "status" in r:
                rows[(r["key"], r["candidate"])] = r
        kinds = {}
        lay = coll / "layouts.json"
        if lay.exists():
            kinds = {l["id"]: l["kind"] for l in json.loads(lay.read_text())}
        maps: dict[str, np.ndarray] = {}
        n_written = n_short = 0
        for f in sorted((coll / "records").glob("*.npz")):
            try:
                with np.load(f) as z:
                    data = {k: z[k] for k in z.files}
            except Exception as exc:  # a run killed mid-write (batch restart) leaves a truncated file
                print(f"  unreadable record {f.name}: {exc} -- skipped", flush=True); continue
            key, cand = str(data["key"]), str(data["candidate"])
            arena = Path(str(data["arena"])); aid = arena.name
            arenas.setdefault(aid, str(arena))
            if data["z1"].shape[0] < args.min_frames:
                n_short += 1; continue
            if key not in maps:
                if aid not in mappers:
                    mappers[aid] = LiveMapper(arena, Path(args.norm_arena), Path(args.encoder), Path(args.posehead), dev)
                f0 = coll / "frame0" / f"{key}.npz"
                if not f0.exists():
                    print(f"  no frame-0 dump for {key}: skipped", flush=True); continue
                maps[key], poses[key] = mappers[aid](f0)
            ekey = f"{key}__{cand}"
            status = str(data["status"])
            meta = json.loads(Path(str(data["meta_path"])).read_text()) if Path(str(data["meta_path"])).exists() else {}
            np.savez(out / f"{ekey}.npz", z1=data["z1"].astype(np.float32), act=data["act"].astype(np.float32), pose=data["pose"].astype(np.float32),
                     power=data["power"].astype(np.float32), map_v2=maps[key], arena=np.array(aid), status=np.array(status),
                     end_frame=data["end_frame"], candidate=np.array(cand), layout=np.array(key), start_est=np.array(poses[key]["est"], np.float32),
                     layout_json=np.array(json.dumps(meta.get("layout", {}))), max_contact_n=data["max_contact_n"],
                     **{k: v for k, v in data.items() if k.startswith("route_")})
            episodes.append(ekey); arena_of[ekey] = aid; status_of[ekey] = status; kind_of[ekey] = kinds.get(key, meta.get("kind", "unknown"))
            r = rows.get((key, cand), {})
            labels[ekey] = {k: r.get(k) for k in ("status", "completed", "time_s", "energy_kj", "contact", "max_contact_n", "max_roll_deg", "max_pitch_deg",
                                                  "min_tire_fz_n", "stall_s", "stalled", "unload_run_max_s", "airborne_s", "mean_ct_m", "p95_ct_m", "length_m", "recorded_frames")}
            sp = data.get("route_speeds", np.zeros(1))
            labels[ekey].update(arena=aid, layout=key, candidate=cand, kind=kind_of[ekey], mean_speed=float(np.mean(sp)), max_speed=float(np.max(sp)),
                                route_length_m=float(data["route_stations"][-1]) if "route_stations" in data else None)
            n_written += 1
        st = {s: sum(status_of[e] == s for e in episodes if arena_of[e] in {Path(a).name for a in [arenas[k] for k in arenas]} and e.split("__")[0] == Path(arenas[list(arenas)[-1]]).name) for s in set(status_of.values())}
        errs = [poses[k]["err_m"] for k in maps]
        print(f"{coll.name}: {n_written} episodes from {len(maps)} layouts ({n_short} shorter than {args.min_frames} frames dropped); "
              f"camera start-pose error mean {np.mean(errs):.2f} m max {np.max(errs):.2f} m; {time.time() - t0:.0f}s", flush=True)
    by_arena = {a: {s: sum(1 for e in episodes if arena_of[e] == a and status_of[e] == s) for s in sorted(set(status_of.values()))} for a in sorted(arenas)}
    (out / "cache_manifest.json").write_text(json.dumps({"schema": 2, "episodes": episodes, "arena_of": arena_of, "status_of": status_of, "kind_of": kind_of,
                                                         "arenas": arenas, "encoder": args.encoder, "posehead": args.posehead, "norm_arena": args.norm_arena,
                                                         "vehicle_masked": True, "z1_preset": "tire_normal_force_omega_pt", "outcomes_by_arena": by_arena}))
    (out / "labels.json").write_text(json.dumps(labels))
    (out / "start_poses.json").write_text(json.dumps(poses, indent=1))
    print(f"{len(episodes)} episodes -> {out}")
    for a, st in by_arena.items():
        print(f"  {a}: {st}")


if __name__ == "__main__":
    main()
