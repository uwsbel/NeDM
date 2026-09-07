#!/usr/bin/env python
"""A1 arm "cheap": route/terrain/speed features for the direct learned time+work predictor.

Builds ONE feature table for every cached Chrono run so the predictor can be trained
leave-one-arena-out without recomputing terrain queries eleven times.

Everything here is computable BEFORE the drive from inputs every planner in the A1
comparison has: the candidate route polyline, its commanded speed profile, the arena
height field (the same prior the world model's crop projection reads), the layout's
asset list, and the vehicle's initial 17-D state at rest. No Chrono outcome, no
recorded future action and no post-hoc quantity enters a feature.

Channels are grouped so an ablation can switch a group off:
  terrain : height/slope/roughness along the route
  profile : commanded speed, nominal acceleration v dv/ds, curvature
  physics : per-station work-rate proxies (climb power, drag, accel) and their integrals
  assets  : signed distance from the route to the nearest asset footprint

  PYTHONPATH=src python scripts/traverse_wp9_cheap_features.py \
      --caches artifacts/traverse/wp7_cache_v1 artifacts/traverse/wp7_cache_sealed artifacts/traverse/wp8_cache_sealed2 \
      --out artifacts/traverse/wp9_energy/cheap_features.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from nedm.traverse.terrain import TerrainMap  # noqa: E402

STEP_M = 1.0
MAX_STEPS = 120          # longest cached route is 94.3 m
MASS_KG = 25000.0 / 9.81  # the runner's chassis mass bookkeeping
G = 9.81
C_RR = 0.05               # nominal rolling coefficient for the uncalibrated analytic proxy
ETA = 0.85                # nominal driveline efficiency for the same proxy

# per-station channel names, in order
PROFILE_CHANNELS = [
    # terrain
    "dz", "along", "cross", "rough", "cum_climb", "cum_drop",
    # profile / geometry
    "kappa_abs", "kappa_signed", "v_cmd", "a_nom", "a_nom_pos", "a_nom_neg",
    "s_frac", "s_remain",
    # physics proxies (per-metre work rate terms, all >= 0 where signed)
    "p_climb", "p_drop", "v2", "lat_acc", "w_grade_cum", "w_accel_cum",
    # assets
    "clear", "clear_near",
    # mask
    "valid",
]
ASSET_CHANNELS = ("clear", "clear_near")
GROUP_OF = {}
for _c in PROFILE_CHANNELS:
    GROUP_OF[_c] = "assets" if _c in ASSET_CHANNELS else "core"

GLOBAL_NAMES = (
    [f"z1_{i}" for i in range(17)]
    + ["len_m", "v_mean", "v_max", "v_min", "v_end_cmd", "profile_time",
       "climb_tot", "drop_tot", "dz_net", "along_mean_abs", "along_pos_mean", "along_max",
       "rough_mean", "turn_tot", "accel_int", "w_analytic_kj", "log_w_analytic",
       "deadline_s", "profile_slack", "len_over_lmin",
       "clear_min", "clear_p05", "clear_mean"]
)
ASSET_GLOBALS = ("clear_min", "clear_p05", "clear_mean")


def route_features(pts: np.ndarray, speeds: np.ndarray, tmap, assets: np.ndarray) -> tuple[np.ndarray, dict]:
    """(N,2) route + (N,) commanded speeds -> (MAX_STEPS, C) station features and a scalar dict."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    station = np.concatenate([[0.0], np.cumsum(seg)])
    length = float(station[-1])
    s = np.arange(0.0, length + 1e-6, STEP_M)[:MAX_STEPS]
    n = len(s)
    x, y = np.interp(s, station, pts[:, 0]), np.interp(s, station, pts[:, 1])
    v = np.interp(s, station, speeds)
    tx, ty = np.gradient(x), np.gradient(y)
    tn = np.hypot(tx, ty) + 1e-9
    tx, ty = tx / tn, ty / tn
    h = tmap.height(x, y)
    gx, gy = tmap.gradient(x, y)
    along = gx * tx + gy * ty
    cross = -gx * ty + gy * tx
    theta = np.unwrap(np.arctan2(ty, tx))
    dtheta = np.gradient(theta)
    kappa = dtheta / STEP_M
    offs = [(1.5, 0.0), (-1.5, 0.0), (0.0, 1.5), (0.0, -1.5)]
    hh = np.stack([tmap.height(x + a * tx - b * ty, y + a * ty + b * tx) for a, b in offs] + [h])
    rough = hh.std(0)
    dh = np.concatenate([[0.0], np.diff(h)])
    cum_climb = np.cumsum(np.maximum(dh, 0.0))
    cum_drop = np.cumsum(np.maximum(-dh, 0.0))
    dv_ds = np.gradient(v) / STEP_M
    a_nom = v * dv_ds                                  # nominal along-route acceleration, m/s^2
    # per-metre specific work terms (J/kg per metre)
    w_grade = G * np.maximum(dh, 0.0) / STEP_M
    w_accel = np.maximum(a_nom, 0.0)
    if assets.size:
        d = np.hypot(x[:, None] - assets[None, :, 0], y[:, None] - assets[None, :, 1]) - assets[None, :, 2]
        clear = d.min(1)
    else:
        clear = np.full(n, 20.0)

    f = np.zeros((MAX_STEPS, len(PROFILE_CHANNELS)), np.float32)
    col = {c: i for i, c in enumerate(PROFILE_CHANNELS)}
    f[:n, col["dz"]] = (h - h[0]) / 2.0
    f[:n, col["along"]] = np.clip(along, -0.8, 0.8)
    f[:n, col["cross"]] = np.clip(cross, -0.8, 0.8)
    f[:n, col["rough"]] = rough / 0.3
    f[:n, col["cum_climb"]] = cum_climb / 5.0
    f[:n, col["cum_drop"]] = cum_drop / 5.0
    f[:n, col["kappa_abs"]] = np.clip(np.abs(kappa), 0, 0.3) * 5.0
    f[:n, col["kappa_signed"]] = np.clip(kappa, -0.3, 0.3) * 5.0
    f[:n, col["v_cmd"]] = v / 10.0
    f[:n, col["a_nom"]] = np.clip(a_nom, -4, 4) / 2.0
    f[:n, col["a_nom_pos"]] = np.clip(np.maximum(a_nom, 0), 0, 4) / 2.0
    f[:n, col["a_nom_neg"]] = np.clip(np.maximum(-a_nom, 0), 0, 4) / 2.0
    f[:n, col["s_frac"]] = s / 50.0
    f[:n, col["s_remain"]] = (length - s) / 50.0
    f[:n, col["p_climb"]] = v * np.maximum(along, 0.0) / 2.0
    f[:n, col["p_drop"]] = v * np.maximum(-along, 0.0) / 2.0
    f[:n, col["v2"]] = v**2 / 50.0
    f[:n, col["lat_acc"]] = np.clip(np.abs(kappa) * v**2, 0, 10) / 5.0
    f[:n, col["w_grade_cum"]] = np.cumsum(w_grade) * STEP_M / 100.0
    f[:n, col["w_accel_cum"]] = np.cumsum(w_accel) * STEP_M / 100.0
    f[:n, col["clear"]] = np.clip(clear, -2.0, 10.0) / 5.0
    f[:n, col["clear_near"]] = np.clip(3.0 - clear, 0.0, 5.0) / 3.0
    f[:n, col["valid"]] = 1.0

    climb_tot, drop_tot = float(cum_climb[-1]), float(cum_drop[-1])
    accel_int = float(np.sum(np.maximum(a_nom, 0.0)) * STEP_M)     # J/kg
    w_kj = MASS_KG * (G * climb_tot + G * C_RR * length + accel_int) / 1000.0 / ETA
    v_mid = np.maximum(0.5 * (speeds[:-1] + speeds[1:]), 1e-3)
    prof_t = float(np.sum(np.diff(station) / v_mid))
    scal = {
        "len_m": length, "v_mean": float(speeds.mean()), "v_max": float(speeds.max()),
        "v_min": float(speeds.min()), "v_end_cmd": float(speeds[-1]), "profile_time": prof_t,
        "climb_tot": climb_tot, "drop_tot": drop_tot, "dz_net": float(h[-1] - h[0]),
        "along_mean_abs": float(np.abs(along).mean()), "along_pos_mean": float(np.maximum(along, 0).mean()),
        "along_max": float(along.max()), "rough_mean": float(rough.mean()),
        "turn_tot": float(np.abs(dtheta).sum()), "accel_int": accel_int,
        "w_analytic_kj": float(w_kj), "log_w_analytic": float(np.log(max(w_kj, 1.0))),
        "clear_min": float(clear.min()), "clear_p05": float(np.quantile(clear, 0.05)),
        "clear_mean": float(clear.mean()),
    }
    return f, scal


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--terrain", choices=["true", "predicted"], default="true",
                    help="true: the arena height field (the prior the world model's crop projection reads); "
                         "predicted: elevation decoded by the frozen map head from the same camera frame")
    ap.add_argument("--maphead", default="artifacts/traverse/wp4_maphead_v2/ckpt_best.pt")
    args = ap.parse_args()

    t0 = time.time()
    episodes: list[tuple[Path, str]] = []
    arena_dirs: dict[str, str] = {}
    for c in args.caches:
        cache = Path(c)
        m = json.loads((cache / "cache_manifest.json").read_text())
        episodes += [(cache, k) for k in m["episodes"]]
        arena_dirs.update(m["arenas"])
    tmaps = {a: TerrainMap.from_dir(REPO_ROOT / d) for a, d in arena_dirs.items()}
    decoders = {}
    if args.terrain == "predicted":
        from nedm.traverse.planner_b import MapDecoder
        dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        decoders = {a: MapDecoder(REPO_ROOT / args.maphead, REPO_ROOT / d, dev) for a, d in arena_dirs.items()}
    print(f"{len(episodes)} episodes, {len(tmaps)} arenas loaded, terrain={args.terrain} ({time.time()-t0:.0f}s)", flush=True)

    prof = np.zeros((len(episodes), MAX_STEPS, len(PROFILE_CHANNELS)), np.float32)
    z1_0 = np.zeros((len(episodes), 17), np.float32)
    scal_rows: list[dict] = []
    keys, layouts, cands, arenas = [], [], [], []
    asset_cache: dict[str, np.ndarray] = {}
    pred_tmaps: dict[str, object] = {}
    for i, (cache, key) in enumerate(episodes):
        with np.load(cache / f"{key}.npz", allow_pickle=True) as z:
            pts = np.asarray(z["route_waypoints"], np.float64)
            sp = np.asarray(z["route_speeds"], np.float64)
            layout, aid, cand = str(z["layout"]), str(z["arena"]), str(z["candidate"])
            if layout not in asset_cache:
                lj = json.loads(str(z["layout_json"]))
                a = [[x["x_m"], x["y_m"], x.get("footprint_radius_m", 3.5)] for x in lj.get("assets", [])]
                asset_cache[layout] = np.asarray(a, np.float64).reshape(-1, 3)
            z1_0[i] = np.asarray(z["z1"])[0, :17]
            if decoders and layout not in pred_tmaps:
                _, elev = decoders[aid](z["map_v2"])
                pred_tmaps[layout] = decoders[aid].terrain(elev)
        tmap = pred_tmaps[layout] if decoders else tmaps[aid]
        f, sc = route_features(pts, sp, tmap, asset_cache[layout])
        prof[i] = f
        scal_rows.append(sc)
        keys.append(key); layouts.append(layout); cands.append(cand); arenas.append(aid)
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(episodes)} ({time.time()-t0:.0f}s)", flush=True)

    # layout-level scalars that need the whole bank (deadline uses the bank's shortest route)
    lay_min: dict[str, float] = {}
    for lay, sc in zip(layouts, scal_rows):
        lay_min[lay] = min(lay_min.get(lay, 1e9), sc["len_m"])
    V_REF, A_ACCEL, K = 5.0, 1.5, 1.0   # frozen mission rule, identical to traverse_wp9_truth.py
    for lay, sc in zip(layouts, scal_rows):
        sc["deadline_s"] = K * (lay_min[lay] / V_REF + V_REF / A_ACCEL)
        sc["profile_slack"] = (sc["deadline_s"] - sc["profile_time"]) / sc["deadline_s"]
        sc["len_over_lmin"] = sc["len_m"] / lay_min[lay]

    scal_keys = [n for n in GLOBAL_NAMES if not n.startswith("z1_")]
    scal = np.array([[r[k] for k in scal_keys] for r in scal_rows], np.float32)
    glob = np.concatenate([z1_0, scal], 1)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, prof=prof, glob=glob, keys=np.array(keys), layout=np.array(layouts),
        candidate=np.array(cands), arena=np.array(arenas),
        profile_channels=np.array(PROFILE_CHANNELS), global_names=np.array(GLOBAL_NAMES),
        asset_channels=np.array(list(ASSET_CHANNELS)), asset_globals=np.array(list(ASSET_GLOBALS)),
        step_m=STEP_M, max_steps=MAX_STEPS, terrain=args.terrain,
    )
    print(f"wrote {out}  prof{prof.shape} glob{glob.shape}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
