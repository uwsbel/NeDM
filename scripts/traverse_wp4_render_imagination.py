#!/usr/bin/env python
"""Render planner rollouts inside the NRD as a top-down video (presentation media).

For each held-out episode: k candidate routes (privileged oracle sweeps + the
recorded route) are tracked by the WP3 policy inside the frozen WP2 model from
the episode's real 16-frame start context. Nothing here touches Chrono: every
vehicle you see is imagined by the model. The recorded route's real vehicle is
drawn as a white ghost so the imagined/real gap is visible.

Usage (from repo root, nedm env):
  PYTHONPATH=src python scripts/traverse_wp4_render_imagination.py \
      --policy artifacts/traverse/wp3_tracker_v1 --episodes 6 --cols 3 \
      --out artifacts/traverse/media/planner_imagination_grid.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation, patches
from matplotlib.colors import LightSource

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nedm.traverse import nrd_data as D
from nedm.traverse.nrd_model import DT_S, VX
from nedm.traverse.terrain import TerrainMap
from nedm.traverse.tracker_env import TraverseTrackingEnv, merge_env_cfg
from traverse_wp4_score_candidates import (CANDIDATE_SWEEP, FOOTPRINT_DISCS, FOOTPRINT_HALF_W,
                                           build_candidates, load_policy, route_dict)
from nedm.traverse.planner_b import MapDecoder, occupancy_discs

LABEL = {"recorded": "recorded route", "oracle": "default plan", "shortest": "shortest",
         "energy_averse": "energy-saving", "slow": "slow (4 m/s)", "fast": "fast (9 m/s)",
         "wide_berth": "wide berth"}
COLOR = {"recorded": "#ffffff", "oracle": "#00d5ff", "shortest": "#ff9f1c", "energy_averse": "#2ec4b6",
         "slow": "#b388ff", "fast": "#ff3864", "wide_berth": "#c6ff00"}
VEH_L, VEH_W = 4.7, 2.2


def collect(args) -> dict:
    cache, routes = Path(args.cache), Path(args.routes)
    keys = D.load_cache_keys(cache)
    split = dict(zip(("train", "val", "test"), D.split_keys(keys)))[args.split]
    manifest = json.loads((routes / "routes_manifest.json").read_text())
    allowed = set().union(*(set(manifest["families"][f]) for f in args.families))
    keys = [k for k in split if k in allowed]
    keys = [keys[i] for i in args.select] if args.select else keys[: args.episodes]
    tmap = TerrainMap.from_dir(Path(args.arena))
    horizon = int(round(args.horizon_s / DT_S))

    entries, index, episodes = [], [], {}
    decoder = MapDecoder(Path(args.maphead), Path(args.arena), args.device) if args.candidates == "predicted" else None
    for key in keys:
        store, ep = key.split("__", 1)
        meta = json.loads((Path(args.stores) / store / ep / "meta.json").read_text())
        occ = None
        if decoder is not None:
            with np.load(cache / f"{key}.npz") as d:
                occ, _ = decoder(d["map_v2"])
            discs = occupancy_discs(occ, decoder.size_m, args.occ_threshold, mode="cells")
            cands, layout = build_candidates(meta, tmap, CANDIDATE_SWEEP, obstacles=discs, repair_iterations=40)
        else:
            cands, layout = build_candidates(meta, tmap, CANDIDATE_SWEEP)
        with np.load(routes / f"{key}.npz") as r:
            recorded = {n: r[n] for n in ("waypoints", "speeds", "headings", "stations")}
        with np.load(cache / f"{key}.npz") as d:
            real_pose = d["pose"]
        episodes[key] = {"layout": layout, "routes": {}, "real_pose": real_pose, "context": None, "occ": occ}
        for name, plan in [("recorded", None)] + cands:
            route = recorded if plan is None else route_dict(plan)
            episodes[key]["routes"][name] = route
            entries.append((key, route))
            index.append((key, name))
    print(f"{len(keys)} episodes -> {len(entries)} rollouts", flush=True)

    cfg = merge_env_cfg({"num_envs": len(entries), "device": args.device, "auto_reset": False,
                         "dynamics_checkpoint": args.dynamics_checkpoint, "arena": args.arena,
                         "cache": args.cache, "routes": args.routes, "split": args.split,
                         "fragment_steps_max": horizon})
    env = TraverseTrackingEnv(cfg, device=args.device, entries=entries)
    policy = load_policy(Path(args.policy), env, args.device)
    n, dev = env.num_envs, env.device
    for e in episodes.values():
        e["context"] = env.context
    obst_np = {k: np.asarray(e["layout"].obstacles(), np.float32) for k, e in episodes.items()}
    m = max(len(o) for o in obst_np.values())
    obst = np.full((n, m, 3), -1.0, np.float32)
    for i, (key, _) in enumerate(index):
        obst[i, : len(obst_np[key])] = obst_np[key]
    obst = torch.tensor(obst, device=dev)

    with torch.no_grad():
        env.reset_idx(torch.arange(n, device=dev), episode_ids=torch.arange(n, device=dev),
                      start_frames=torch.full((n,), env.context, device=dev, dtype=torch.long),
                      fragment_steps=torch.full((n,), horizon, device=dev, dtype=torch.long))
        env._compute_observations()
        active = torch.ones(n, dtype=torch.bool, device=dev)
        pose_log = [env.pose.clone()]
        vx_log = [env.z1_phys[:, VX].clone()]
        energy_log = [env.energy_kj.clone()]
        ct_log = [torch.zeros(n, device=dev)]
        active_log = [active.clone()]
        completed = torch.zeros(n, dtype=torch.bool, device=dev)
        collided = torch.zeros(n, dtype=torch.bool, device=dev)
        end_step = torch.full((n,), horizon, device=dev, dtype=torch.long)
        for step in range(horizon):
            act = policy(env.obs_buf)
            _, _, dones, _ = env.step(act)
            err = env._route_errors()
            cos_y, sin_y = torch.cos(env.pose[:, 2]), torch.sin(env.pose[:, 2])
            clear = torch.full((n,), float("inf"), device=dev)
            for off in FOOTPRINT_DISCS:
                cx, cy = env.pose[:, 0] + off * cos_y, env.pose[:, 1] + off * sin_y
                d = torch.hypot(obst[..., 0] - cx[:, None], obst[..., 1] - cy[:, None]) - obst[..., 2] - FOOTPRINT_HALF_W
                d = torch.where(obst[..., 2] >= 0, d, torch.full_like(d, float("inf")))
                clear = torch.minimum(clear, d.min(dim=1).values)
            collided |= active & (clear < 0)
            just_done = active & dones.bool()
            end_step = torch.where(just_done, torch.full_like(end_step, step + 1), end_step)
            completed |= just_done & err["route_end"]
            # freeze logs once an env is done
            prev = pose_log[-1]
            pose_log.append(torch.where(active[:, None], env.pose, prev))
            vx_log.append(torch.where(active, env.z1_phys[:, VX], vx_log[-1]))
            energy_log.append(torch.where(active, env.energy_kj, energy_log[-1]))
            ct_log.append(torch.where(active, err["e_ct"].abs(), ct_log[-1]))
            active &= ~dones.bool()
            active_log.append(active.clone())
            if not active.any():
                break
    out = {
        "index": index, "episodes": episodes, "tmap": tmap, "horizon": horizon,
        "pose": torch.stack(pose_log).cpu().numpy(), "vx": torch.stack(vx_log).cpu().numpy(),
        "energy": torch.stack(energy_log).cpu().numpy(), "ct": torch.stack(ct_log).cpu().numpy(),
        "active": torch.stack(active_log).cpu().numpy(), "completed": completed.cpu().numpy(),
        "collided": collided.cpu().numpy(), "end_step": end_step.cpu().numpy(),
    }
    return out


def vehicle_polygon(x: float, y: float, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    body = np.array([[-VEH_L / 2, -VEH_W / 2], [VEH_L / 2, -VEH_W / 2], [VEH_L / 2 + 0.6, 0.0],
                     [VEH_L / 2, VEH_W / 2], [-VEH_L / 2, VEH_W / 2]])
    rot = np.array([[c, -s], [s, c]])
    return body @ rot.T + np.array([x, y])


def window_polygon(x: float, y: float, yaw: float, half: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    sq = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    return sq @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def load_chrono_rows(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "tracker" in str(r.get("controller", "")) and r.get("completed"):
            out[(r["key"], r["candidate"])] = r
    return out


def render(data: dict, args) -> None:
    keys = list(data["episodes"])
    tmap: TerrainMap = data["tmap"]
    T = data["pose"].shape[0]
    hold = int(round(args.hold_s / DT_S))
    cols = min(args.cols, len(keys))
    rows_n = math.ceil(len(keys) / cols)
    fig_w = args.panel_in * cols
    fig_h = args.panel_in * rows_n * 0.92 + (1.9 if cols == 1 else (1.6 if args.candidates == "predicted" else 1.4))
    fig, axes = plt.subplots(rows_n, cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.patch.set_facecolor("#101418")
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=(0.835 if cols == 1 else (0.85 if args.candidates == "predicted" else 0.865)), wspace=0.02, hspace=0.06)
    half = tmap.half
    ls = LightSource(azdeg=315, altdeg=45)
    rng = float(tmap.height_grid.max() - tmap.height_grid.min())
    shaded = ls.shade(tmap.height_grid, cmap=plt.cm.gist_earth, vert_exag=6.0, blend_mode="soft",
                      vmin=tmap.height_grid.min() - 1.0 * rng, vmax=tmap.height_grid.max() + 0.6 * rng)
    chrono = load_chrono_rows(Path(args.chrono_rows) if args.chrono_rows else None)

    panels = []
    for p, key in enumerate(keys):
        ax = axes[p // cols][p % cols]
        ep = data["episodes"][key]
        ax.imshow(shaded, extent=[-half, half, -half, half], origin="lower", interpolation="bilinear")
        ax.set_facecolor("#101418")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor("#3a4450")
        pred = ep.get("occ") is not None
        for a in ep["layout"].assets:
            fc = {"house": "#8d6e63", "rock": "#9e9e9e", "tree": "#2e7d32"}.get(a.kind, "#888")
            ax.add_patch(patches.Circle((a.x_m, a.y_m), a.footprint_radius_m, fc="none" if pred else fc, ec="k",
                                        lw=0.8 if pred else 0.6, ls="--" if pred else "-", alpha=0.9, zorder=3))
        if pred:
            g = ep["occ"].shape[0]; cell = tmap.size_m / g
            iy, ix = np.nonzero(ep["occ"] >= args.occ_threshold)
            for yy, xx in zip(iy, ix):
                ax.add_patch(patches.Rectangle((xx * cell - half, half - (yy + 1) * cell), cell, cell,
                                               fc="#ff1744", ec="none", alpha=0.75, zorder=3))
        # goal marker at the recorded route end
        rec = ep["routes"]["recorded"]
        gx, gy = rec["waypoints"][-1]
        ax.add_patch(patches.Circle((gx, gy), 2.0, fc="none", ec="#ffd600", lw=1.5, ls="--", zorder=4))
        ax.plot([ep["layout"].start_xy[0]], [ep["layout"].start_xy[1]], marker="o", ms=5, color="w", zorder=4)
        # candidate routes
        idxs = [i for i, (k, _) in enumerate(data["index"]) if k == key]
        allxy = []
        for i in idxs:
            name = data["index"][i][1]
            w = ep["routes"][name]["waypoints"]
            allxy.append(w)
            ax.plot(w[:, 0], w[:, 1], color=COLOR[name], lw=0.9, alpha=0.55, zorder=5,
                    ls="--" if name == "recorded" else "-")
        allxy = np.concatenate(allxy)
        pad = args.pad_m
        x0, x1 = allxy[:, 0].min() - pad, allxy[:, 0].max() + pad
        y0, y1 = allxy[:, 1].min() - pad, allxy[:, 1].max() + pad
        side = max(x1 - x0, y1 - y0)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = min(side, 2 * half)
        cx = float(np.clip(cx, -half + side / 2, half - side / 2))
        cy = float(np.clip(cy, -half + side / 2, half - side / 2))
        ax.set_xlim(cx - side / 2, cx + side / 2); ax.set_ylim(cy - side / 2, cy + side / 2)
        ax.set_aspect("equal")
        # real recorded vehicle ghost
        ghost = patches.Polygon(vehicle_polygon(*ep["real_pose"][ep["context"]]), closed=True, fc="none",
                                ec="w", lw=1.0, ls=":", zorder=7)
        ax.add_patch(ghost)
        arts = []
        for i in idxs:
            name = data["index"][i][1]
            x, y, yaw = data["pose"][0, i]
            poly = patches.Polygon(vehicle_polygon(x, y, yaw), closed=True, fc=COLOR[name], ec="k", lw=0.5,
                                   alpha=0.6 if name == "recorded" else 0.95, zorder=8)
            ax.add_patch(poly)
            win = patches.Polygon(window_polygon(x, y, yaw, args.window_half_m), closed=True, fc="none",
                                  ec=COLOR[name], lw=0.5, ls=":", alpha=0.5, zorder=6)
            ax.add_patch(win)
            trail, = ax.plot([x], [y], color=COLOR[name], lw=2.0, alpha=0.95, zorder=7)
            arts.append((i, name, poly, win, trail))
        short = key.split("__", 1)[1].replace("_oracle", "")
        title = ax.text(0.02, 0.975, f"held-out layout {short}", transform=ax.transAxes, color="w",
                        fontsize=9, va="top", ha="left", fontweight="bold",
                        bbox=dict(fc="#000000", alpha=0.45, ec="none", pad=2))
        pts = np.concatenate([allxy, np.array([[a.x_m, a.y_m] for a in ep["layout"].assets])])
        # scoreboard corner: the one whose box (in axes fraction) covers the fewest route/asset points
        n_lines = len(idxs)
        bw, bh = 0.34 * (7.2 / args.board_font) ** -1 * (5.6 / args.panel_in) ** -1 * 0.55 + 0.34, 0.03 + 0.025 * n_lines
        bw = min(bw, 0.6)
        fr = (pts - np.array([cx - side / 2, cy - side / 2])) / side
        best, best_score = None, None
        for ha_, va_ in (("left", "bottom"), ("right", "bottom"), ("left", "top"), ("right", "top")):
            x_lo = 0.0 if ha_ == "left" else 1.0 - bw
            y_lo = 0.0 if va_ == "bottom" else 0.9 - bh
            inside = ((fr[:, 0] > x_lo - 0.04) & (fr[:, 0] < x_lo + bw + 0.04) &
                      (fr[:, 1] > y_lo - 0.04) & (fr[:, 1] < y_lo + bh + 0.04)).sum()
            centre = np.array([x_lo + bw / 2, y_lo + bh / 2])
            score = (inside, -np.hypot(*(fr - centre).T).min())
            if best_score is None or score < best_score:
                best, best_score = (ha_, va_), score
        ha_, va_ = best
        bx, by = (0.02 if ha_ == "left" else 0.98), (0.02 if va_ == "bottom" else 0.90)
        names = [n for _, n in [(i, data["index"][i][1]) for i in idxs]]
        lines_art = []
        step_fr = 0.028 * args.board_font / 7.2 * (5.6 / args.panel_in) ** 0.5
        for li, n in enumerate(names):
            off = (len(names) - 1 - li) * step_fr if va_ == "bottom" else -li * step_fr
            lines_art.append(ax.text(bx, by + off, "", transform=ax.transAxes, color=COLOR[n], fontsize=args.board_font,
                                     va=va_, ha=ha_, family="monospace", fontweight="bold",
                                     bbox=dict(fc="#000000", alpha=0.6, ec="none", pad=1.5)))
        board = lines_art
        panels.append((key, ax, ghost, arts, board))

    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=COLOR[n], lw=3, ls="--" if n == "recorded" else "-") for n in LABEL]
    fig.legend(handles, [LABEL[n] for n in LABEL], loc="upper center", ncol=7 if cols > 1 else 4,
               bbox_to_anchor=(0.5, (0.9 if args.candidates == "predicted" else 0.915) if cols > 1 else 0.885), frameon=False, labelcolor="w",
               fontsize=9 if cols > 1 else 8.5, handlelength=2.5, columnspacing=1.4)
    sup = fig.text(0.5, 0.975, "", ha="center", va="top", color="w", fontsize=13, fontweight="bold")
    sub = fig.text(0.5, 0.94, "", ha="center", va="top", color="#c0c8d0", fontsize=9.5)

    def scoreboard(key, t_step, final: bool) -> str:
        lines = []
        idxs = [(i, n) for i, (k, n) in enumerate(data["index"]) if k == key]
        best = None
        if final:
            ok = [(i, n) for i, n in idxs if n != "recorded" and data["completed"][i] and not data["collided"][i]]
            pool = ok or [(i, n) for i, n in idxs if n != "recorded"]
            if pool:
                best = min(pool, key=lambda t: data["end_step"][t[0]] * DT_S + data["energy"][-1, t[0]] / 10.0)[1]
        for i, n in idxs:
            s = min(t_step, T - 1)
            done = data["end_step"][i] <= t_step
            t_s = data["end_step"][i] * DT_S if done else t_step * DT_S
            e = data["energy"][s, i]
            if done:
                tag = "OK " if data["completed"][i] else ("HIT" if data["collided"][i] else "X  ")
            else:
                tag = f"{data['vx'][s, i]:3.1f}m/s"
            extra = ""
            cr = chrono.get((key, n))
            if final and cr:
                extra = f" | sim {cr['time_s']:4.1f}s"
            star = " <- pick" if (final and n == best) else ""
            lines.append(f"{LABEL[n]:<15s} {tag:>7s} {t_s:4.1f}s {e:5.0f}kJ{extra}{star}")
        width = max(len(l) for l in lines)
        return "\n".join(l.ljust(width) for l in lines)

    writer = animation.FFMpegWriter(fps=args.fps, bitrate=args.bitrate, codec="libx264",
                                    extra_args=["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p"])
    total = T + hold
    t0 = time.time()
    with writer.saving(fig, args.out, dpi=args.dpi):
        for f in range(total):
            s = min(f, T - 1)
            final = f >= T - 1
            for key, ax, ghost, arts, board in panels:
                ep = data["episodes"][key]
                rp = ep["real_pose"]
                ri = min(ep["context"] + s, len(rp) - 1)
                ghost.set_xy(vehicle_polygon(*rp[ri]))
                for i, name, poly, win, trail in arts:
                    x, y, yaw = data["pose"][s, i]
                    poly.set_xy(vehicle_polygon(x, y, yaw))
                    win.set_xy(window_polygon(x, y, yaw, args.window_half_m))
                    win.set_visible(data["active"][s, i])
                    trail.set_data(data["pose"][: s + 1, i, 0], data["pose"][: s + 1, i, 1])
                    if final:
                        if name == "recorded":
                            continue
                for art, line in zip(board, scoreboard(key, s, final).split("\n")):
                    art.set_text(line)
            mode = "camera-only plans" if args.candidates == "predicted" else "planner rollout"
            sup.set_text(f"{mode.capitalize()} inside the NRD model  —  t = {s * DT_S:4.1f} s")
            sub.set_text(("Every vehicle is imagined by the physics+camera model and driven by the learned tracker; no simulator in the loop.\n"
                         "Dotted square = 8x8 camera-derived terrain window fed to the physics.   "
                         "White dotted outline = the real recorded vehicle on its (dashed white) route.   "
                         "sim = same candidate driven in the Chrono simulator afterwards.").replace("physics.   ", "physics.\n").replace("route.   ", "route.\n") if cols == 1 else
                         ("Every vehicle is imagined by the physics+camera model and driven by the learned tracker; no simulator in the loop.\n"
                          + ("Red cells = obstacles decoded from the camera; the plans were searched on those cells, not on the true map (dashed outlines).\n"
                             if args.candidates == "predicted" else "Dotted square = 8x8 camera-derived terrain window fed to the physics.   ")
                          + "White dotted outline = the real recorded vehicle on its (dashed white) route.   "
                          "sim = same candidate driven in the Chrono simulator afterwards."))
            writer.grab_frame()
            if f % 50 == 0:
                print(f"frame {f}/{total}  {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {args.out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="artifacts/traverse/wp3_tracker_v1")
    ap.add_argument("--dynamics-checkpoint", default="artifacts/traverse/wp2_mapv2_index_amd/ckpt_best.pt")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--routes", default="artifacts/traverse/wp3_routes")
    ap.add_argument("--stores", default="artifacts/traverse")
    ap.add_argument("--split", default="val")
    ap.add_argument("--families", nargs="+", default=["oracle"])
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--select", type=int, nargs="*", default=None, help="indices into the val/oracle list")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--horizon-s", type=float, default=20.0)
    ap.add_argument("--hold-s", type=float, default=3.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--bitrate", type=int, default=6000)
    ap.add_argument("--panel-in", type=float, default=5.6)
    ap.add_argument("--pad-m", type=float, default=7.0)
    ap.add_argument("--window-half-m", type=float, default=5.0)
    ap.add_argument("--chrono-rows", default=None)
    ap.add_argument("--candidates", choices=["oracle", "predicted"], default="oracle")
    ap.add_argument("--maphead", default="artifacts/traverse/wp4_maphead_v2/ckpt_best.pt")
    ap.add_argument("--occ-threshold", type=float, default=0.85)
    ap.add_argument("--board-font", type=float, default=6.6)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    data = collect(args)
    for key in data["episodes"]:
        idxs = [(i, n) for i, (k, n) in enumerate(data["index"]) if k == key]
        print(key, [(n, round(float(data["end_step"][i] * DT_S), 1), bool(data["completed"][i])) for i, n in idxs])
    render(data, args)


if __name__ == "__main__":
    main()
