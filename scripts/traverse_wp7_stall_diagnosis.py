#!/usr/bin/env python
"""Why does the imagination accept runs that Chrono stalls? Two parts.

``classify``: every non-feasible recorded run (schema-v2 caches) gets a physical failure class from its own
17-D trace and the true terrain: where the vehicle stopped (progress along the route), what it was doing
(throttle, engine torque, wheel speeds -> slip, tire loads -> wheel lift, pitch/roll and the terrain slope
ahead), and whether it ever launched. Also the per-layout outcome pattern over the direct-crossing speeds
(does a slow crossing stall where a fast one passes?). Cross-tabulated with the imagination rows when given.

``model``: for the same runs, where does a dynamics model's prediction go wrong? (a) from rest with the
RECORDED controls (teacher forcing: is the dynamics wrong?), (b) from the recorded context 2 s before the stop
with the recorded controls (local dynamics error), (c) same start with the tracker driving (controller
feedback), (d) from rest with the tracker (the planner's imagination). Feasible runs on the same layouts are
the matched controls. Plus a sensitivity probe: does the rest-state prediction change when the tire loads are
set to a healthy pattern?
"""
from __future__ import annotations

import argparse, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nedm.traverse.terrain import TerrainMap

DT = 0.05
R_WHEEL = 0.4665  # HMMWV tire radius (m)
FZ, OM = slice(7, 11), slice(11, 15)
UNLOADED_N = 500.0
STOP_MPS = 0.3
WHEELS = ("fl", "fr", "rl", "rr")


def load_episode(cache: Path, key: str) -> dict:
    with np.load(cache / f"{key}.npz") as d:
        return {n: d[n] for n in d.files}


def route_progress(pose: np.ndarray, wps: np.ndarray, stations: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(pose[:, None, :2] - wps[None, :, :2], axis=-1)
    return stations[d.argmin(axis=1)]


def final_stationary_run(vx: np.ndarray, thr: float = STOP_MPS) -> tuple[int, int]:
    """(first frame of the final |vx|<thr run, its length); length 0 if moving at the end."""
    i = len(vx) - 1
    while i >= 0 and abs(vx[i]) < thr:
        i -= 1
    return i + 1, len(vx) - (i + 1)


def slope_ahead_deg(tmap: TerrainMap, x: float, y: float, yaw: float, d: float = 2.0) -> float:
    c, s = math.cos(yaw), math.sin(yaw)
    h1 = float(tmap.height(np.array([x + d * c]), np.array([y + d * s]))[0])
    h0 = float(tmap.height(np.array([x - d * c]), np.array([y - d * s]))[0])
    return math.degrees(math.atan2(h1 - h0, 2 * d))


def stuck_from_displacement(pose: np.ndarray, win: int = 40, thr_m: float = 1.0) -> tuple[int, int]:
    """Frame after which the vehicle never again moves more than ``thr_m`` in any ``win``-frame window (stuck until the
    end), and the length of that stuck tail. Displacement-based: the Chrono stall counter is cumulative, so a stalled
    vehicle may rock with |vx| briefly above 0.3 m/s."""
    n = len(pose)
    if n <= win:
        return 0, n
    fwd = np.linalg.norm(pose[win:, :2] - pose[:-win, :2], axis=1)  # fwd[t] = displacement over [t, t+win]
    stuck = fwd < thr_m
    i = len(stuck) - 1
    while i >= 0 and stuck[i]:
        i -= 1
    start = i + 1  # first t with every later window stuck
    return start, n - start


def classify_episode(ep: dict, label: dict, tmap: TerrainMap) -> dict:
    z1, act, pose, power = ep["z1"], ep["act"], ep["pose"], ep["power"][:, 0]
    status = str(ep["status"]); n = len(z1)
    wps, sta = ep["route_waypoints"], ep["route_stations"]
    prog = route_progress(pose, wps, sta)
    vx = z1[:, 0]
    launched = np.nonzero(vx > 0.5)[0]
    launch_s = float(launched[0] * DT) if len(launched) else None
    end = int(ep["end_frame"]) if int(ep["end_frame"]) >= 0 else n
    stop, run = stuck_from_displacement(pose[:end])
    stuck_end = run * DT >= 2.0 or status == "stall"
    out = {"arena": str(ep["arena"]), "layout": str(ep["layout"]), "candidate": str(ep["candidate"]), "kind": label.get("kind"),
           "status": status, "stalled": bool(label.get("stalled")), "contact": bool(label.get("contact")), "feasible": bool(label.get("completed")) and not label.get("stalled") and not label.get("contact"),
           "n_frames": n, "end_frame": end, "progress_end_m": float(prog[min(end, n) - 1]), "route_len_m": float(sta[-1]), "launch_s": launch_s,
           "vx_peak": float(vx.max()), "mean_speed_cmd": label.get("mean_speed"), "stop_s": float(stop * DT) if stuck_end else None, "stuck_tail_s": float(run * DT),
           "vx_end2s_mean": float(np.abs(vx[max(0, end - 40):end]).mean())}
    tags = []
    if status == "rollover":
        cls = "rollover"
    elif status == "off_route":
        cls = "off_route"
    elif out["feasible"]:
        cls = "feasible"
    elif status == "completed" and not out["stalled"]:
        cls = "contact"  # drove the route; infeasible only because it touched an asset
    elif status == "completed":
        cls = "stall_recovered"  # >= 1 s of cumulative stall, then finished
    elif out["progress_end_m"] < 2.0 or launch_s is None:
        cls = "launch"
    elif stuck_end:
        cls = "stop"  # moved, then stuck en route
    else:
        cls = "crawl"  # timeout while still moving: too slow, not stuck
    if cls == "launch":
        t_c = min(60, n - 1)  # 3 s into the launch attempt
    elif cls == "stop":
        t_c = stop
    else:
        t_c = min(end, n) - 1
    lo, hi = max(0, t_c - 20), min(n, t_c + 21)
    w = slice(lo, hi)
    fz, om = z1[w, FZ], z1[w, OM]
    x, y, yaw = (float(v) for v in pose[t_c])
    slip = float((np.abs(om) * R_WHEEL - np.abs(vx[w])[:, None]).max())
    pre = slice(max(0, t_c - 40), max(1, t_c))
    out.update({"class": cls, "t_center_s": float(t_c * DT), "pitch_deg": float(np.degrees(z1[w, 3].mean())), "roll_deg": float(np.degrees(z1[w, 2].mean())),
                "pitch_min_deg": float(np.degrees(z1[w, 3].min())), "slope_ahead_deg": slope_ahead_deg(tmap, x, y, yaw), "slope_ahead_5m_deg": slope_ahead_deg(tmap, x, y, yaw, 5.0),
                "unloaded_frac": [float((fz[:, i] < UNLOADED_N).mean()) for i in range(4)], "unloaded_frac_any": float((fz < UNLOADED_N).any(axis=1).mean()),
                "slip_mps": slip, "throttle": float(act[w, 1].mean()), "brake": float(act[w, 2].mean()), "steer": float(act[w, 0].mean()),
                "torque_nm": float(z1[w, 16].mean()), "engine_radps": float(z1[w, 15].mean()), "power_kw": float(power[w].mean()),
                "vx_pre2s": float(vx[pre].mean()), "height_m": float(tmap.height(np.array([x]), np.array([y]))[0]),
                "rest_fz": [float(v) for v in z1[0, FZ]], "rest_unloaded_any": bool((z1[0, FZ] < UNLOADED_N).any())})
    if cls in ("launch", "stop", "stall_recovered", "crawl"):
        if out["slope_ahead_deg"] > 8.0 or out["pitch_min_deg"] < -8.0:
            tags.append("slope")
        if max(out["unloaded_frac"]) > 0.5:
            tags.append("lift")
        if slip > 3.0:
            tags.append("spin")
        if out["throttle"] < 0.3 and out["brake"] < 0.1 and cls != "crawl":
            tags.append("low_throttle")
        if out["brake"] > 0.3:
            tags.append("braking")
    if out["contact"] and cls != "contact":
        tags.append("contact")
    out["tags"] = tags
    out["label"] = cls + ("|" + "+".join(tags) if tags else "")
    return out


def speed_patterns(labels: dict, feasible_of: dict) -> dict:
    by = defaultdict(dict)
    for k, c in labels.items():
        cand = c["candidate"]
        if cand.startswith("direct_v") and cand[8:].isdigit():
            by[c["layout"]][int(cand[8:])] = feasible_of[k]
    pats = {}
    for lay, d in by.items():
        vs = sorted(d)
        s = "".join("S" if d[v] else "F" for v in vs)
        ok, bad = [v for v in vs if d[v]], [v for v in vs if not d[v]]
        if not bad:
            p = "all_ok"
        elif not ok:
            p = "all_fail"
        elif max(bad) < min(ok):
            p = "slow_fail_fast_ok"
        elif min(bad) > max(ok):
            p = "fast_fail_slow_ok"
        else:
            p = "mixed"
        pats[lay] = {"pattern": p, "outcomes": s, "speeds": vs}
    return pats


def cmd_classify(args) -> None:
    rows, pats_all = {}, {}
    img = defaultdict(dict)  # key -> model -> img_ok
    for spec in args.imagine:
        name, path = spec.split("=", 1)
        for r in json.loads(Path(path).read_text()):
            img[r["key"]][name] = bool(r["img_ok"])
    for cdir in args.caches:
        cache = Path(cdir)
        man = json.loads((cache / "cache_manifest.json").read_text())
        labels = json.loads((cache / "labels.json").read_text())
        tmaps = {a: TerrainMap.from_dir(Path(p)) for a, p in man["arenas"].items()}
        feasible_of = {k: bool(c.get("completed")) and not c.get("stalled") and not c.get("contact") for k, c in labels.items()}
        pats_all.update(speed_patterns(labels, feasible_of))
        for k in man["episodes"]:
            c = labels[k]
            if feasible_of[k] and not args.all:
                continue
            ep = load_episode(cache, k)
            r = classify_episode(ep, c, tmaps[str(ep["arena"])])
            r["img_ok"] = img.get(k, {})
            rows[k] = r
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "classify.json").write_text(json.dumps({"rows": rows, "speed_patterns": pats_all}, indent=1))
    # ---- summaries
    arenas = sorted(set(r["arena"] for r in rows.values()))
    labs = sorted(set(r["label"] for r in rows.values()))
    print(f"{len(rows)} non-feasible runs; class|tags by arena")
    print(f"{'label':34s}" + "".join(f"{a[-4:]:>7s}" for a in arenas) + f"{'all':>7s}")
    for lab in labs:
        cnt = Counter(r["arena"] for r in rows.values() if r["label"] == lab)
        print(f"{lab:34s}" + "".join(f"{cnt.get(a, 0):7d}" for a in arenas) + f"{sum(cnt.values()):7d}")
    print("\ncoarse class by status:")
    ct = Counter((r["class"], r["status"]) for r in rows.values())
    for (cls, st), n in sorted(ct.items()):
        print(f"  {cls:16s} {st:10s} {n:5d}")
    models = sorted(set(m for r in rows.values() for m in r["img_ok"]))
    if models:
        print("\nimagination acceptance of the non-feasible runs by class (accepted / n with rows):")
        print(f"{'class':18s}" + "".join(f"{m[:22]:>24s}" for m in models))
        for cls in sorted(set(r["class"] for r in rows.values())):
            line = f"{cls:18s}"
            for m in models:
                rs = [r for r in rows.values() if r["class"] == cls and m in r["img_ok"]]
                line += f"{sum(r['img_ok'][m] for r in rs):>12d}/{len(rs):<11d}"
            print(line)
    print("\nper-layout outcome over direct-crossing speeds (v2..v9):")
    pc = Counter((lay.split("__")[0], p["pattern"]) for lay, p in pats_all.items())
    pa = sorted(set(a for a, _ in pc))
    for p in ("all_ok", "slow_fail_fast_ok", "fast_fail_slow_ok", "mixed", "all_fail"):
        print(f"  {p:20s}" + "".join(f"{pc.get((a, p), 0):6d}" for a in pa) + f"{sum(v for (a, q), v in pc.items() if q == p):6d}")
    print("   arenas:", " ".join(a[-4:] for a in pa))
    # launch / stop diagnostics
    for cls in ("launch", "stop", "crawl"):
        rs = [r for r in rows.values() if r["class"] == cls]
        if not rs:
            continue
        f = lambda key: np.array([r[key] for r in rs], float)
        print(f"\n{cls}: n={len(rs)} progress {np.median(f('progress_end_m')):.1f} m (median) of route {np.median(f('route_len_m')):.0f} m | slope ahead {np.median(f('slope_ahead_deg')):+.1f} deg (p25 {np.percentile(f('slope_ahead_deg'), 25):+.1f}, p75 {np.percentile(f('slope_ahead_deg'), 75):+.1f}) "
              f"| pitch {np.median(f('pitch_deg')):+.1f} | slip {np.median(f('slip_mps')):.1f} m/s (p25 {np.percentile(f('slip_mps'), 25):.1f}) | any-wheel-unloaded frac {np.median(f('unloaded_frac_any')):.2f} | throttle {np.median(f('throttle')):.2f} | torque {np.median(f('torque_nm')):.0f} Nm | rest unloaded {np.mean([r['rest_unloaded_any'] for r in rs]):.2f}")


# ----------------------------------------------------------------------------------------------- model tests
def cmd_model(args) -> None:
    import torch
    from nedm.traverse.oracle import PlanCandidate
    from nedm.traverse.tracker_env import TraverseTrackingEnv, merge_env_cfg
    from nedm.traverse.nrd_model import VX
    from traverse_wp4_score_candidates import load_policy, route_dict
    dev = args.device
    diag = json.loads((Path(args.out) / "classify.json").read_text())["rows"]
    results = {}
    for cdir in args.caches:
        cache = Path(cdir)
        man = json.loads((cache / "cache_manifest.json").read_text())
        labels = json.loads((cache / "labels.json").read_text())
        start_est = json.loads((cache / "start_poses.json").read_text())
        for aid in args.arenas:
            if aid not in man["arenas"]:
                continue
            keys_bad = [k for k in man["episodes"] if man["arena_of"][k] == aid and k in diag and diag[k]["class"] in args.classes]
            lay_bad = set(diag[k]["layout"] for k in keys_bad)
            keys_ok = [k for k in man["episodes"] if man["arena_of"][k] == aid and labels[k]["layout"] in lay_bad and k not in diag
                       and labels[k].get("completed") and not labels[k].get("stalled") and not labels[k].get("contact")]
            if args.max_per_arena:
                keys_bad, keys_ok = keys_bad[: args.max_per_arena], keys_ok[: args.max_per_arena]
            keys = keys_bad + keys_ok
            if not keys:
                continue
            eps = {k: load_episode(cache, k) for k in keys}
            entries = []
            for k in keys:
                z = eps[k]
                plan = PlanCandidate(waypoints=z["route_waypoints"].astype(float), speeds=z["route_speeds"].astype(float), headings=z["route_headings"].astype(float),
                                     stations=z["route_stations"].astype(float), meta={"candidate": labels[k]["candidate"]})
                entries.append((k, route_dict(plan)))
            n = len(keys)
            print(f"\n=== {aid}: {len(keys_bad)} non-feasible ({Counter(diag[k]['class'] for k in keys_bad)}) + {len(keys_ok)} feasible controls on the same layouts", flush=True)
            for ckpt in args.dynamics_checkpoints:
                mname = Path(ckpt).parent.name
                cfg = merge_env_cfg({"num_envs": n, "device": dev, "auto_reset": False, "split": "val", "dynamics_checkpoint": ckpt, "arena": man["arenas"][aid],
                                     "cache": str(cache), "routes": "artifacts/traverse/wp3_routes", "fragment_steps_max": 600, "z1_extra_cache": None, "map_key": "map_v2",
                                     "termination": {"max_abs_roll_rad": math.radians(args.roll_limit_deg), "max_abs_pitch_rad": math.radians(args.pitch_limit_deg)}})
                env = TraverseTrackingEnv(cfg, device=dev, entries=entries)
                policy = load_policy(Path(args.policy), env, dev)
                b = env.bank
                ids = torch.arange(n, device=dev)
                rec_vx = (b.z1[:, :, VX] * env.z1_std[VX] + env.z1_mean[VX]).cpu().numpy()  # padded with the last row
                n_rec = np.array([len(eps[k]["z1"]) for k in keys])
                H = int(round(args.horizon_s / DT))

                def wrong_maps():
                    """swap every env's scene map for the next env's (another layout of the same arena, usually)"""
                    env.env_maps[:] = torch.roll(env.env_maps, 1, dims=0)

                def seed_rest(fz_override=None, map_probe=False):
                    env.reset_idx(ids, episode_ids=ids, start_frames=torch.full((n,), env.context, device=dev, dtype=torch.long), fragment_steps=torch.full((n,), H, device=dev, dtype=torch.long))
                    c = env.context
                    z0 = b.z1[ids, 0].clone()
                    if fz_override is not None:
                        z0[:, FZ] = (torch.tensor(fz_override, device=dev) - env.z1_mean[FZ]) / env.z1_std[FZ]
                        z0[:, OM] = (0.0 - env.z1_mean[OM]) / env.z1_std[OM]
                    brake = (torch.tensor([0.0, 0.0, 1.0], device=dev) - env.act_mean) / env.act_std
                    env.z1_hist[:] = z0[:, None, :].expand(-1, c, -1); env.act_hist[:] = brake[None, None, :].expand(n, c, -1)
                    env.pose[:] = b.pose[ids, 0] if not args.camera_start else torch.tensor(np.array([start_est[labels[k]["layout"]]["est"] for k in keys], np.float32), device=dev)
                    if map_probe:
                        wrong_maps()
                    with torch.no_grad():
                        env.token_hist[:] = env.model.cropper(env.env_maps, env.pose[:, None, :].expand(-1, c, -1))
                    env.z1_phys[:] = z0 * env.z1_std + env.z1_mean
                    env.last_actions[:] = torch.tensor([0.0, 0.0, 1.0], device=dev); env.actions[:] = env.last_actions
                    d = (b.route_xy[ids] - env.pose[:, None, :2]).norm(dim=-1)
                    valid = torch.arange(b.route_xy.shape[1], device=dev)[None, :] < b.route_len[ids][:, None]
                    env.route_idx[:] = torch.where(valid, d, torch.full_like(d, float("inf"))).argmin(dim=1)
                    env.start_station_m[:] = b.route_s[ids, env.route_idx]
                    env.episode_length_buf[:] = 0; env.progress_m[:] = 0
                    env._compute_observations()

                def run(mode: str, t0: np.ndarray, steps: int) -> dict:
                    """mode 'rec' = recorded controls (teacher forcing), 'pol' = tracker. t0[i] = frame whose state is the
                    last context frame (rest: 0). Returns vx trajectories (n, steps) and progress."""
                    vx_tr = np.full((n, steps), np.nan, np.float32); pitch_tr = np.full((n, steps), np.nan, np.float32); prog = np.zeros(n, np.float32); done = np.zeros(n, bool); fail = np.zeros(n, bool); complete = np.zeros(n, bool)
                    t0_t = torch.tensor(t0, device=dev)
                    for s in range(steps):
                        if mode == "rec":
                            idx = torch.clamp(t0_t + s, max=b.n_frames - 1)
                            driver = b.act_raw[ids, idx]
                            env._nn_step(driver); env.episode_length_buf += 1
                            err = env._route_errors(); env.last_actions = driver.clone(); env.actions = driver
                            env._compute_observations()
                            fin = torch.isfinite(env.z1_phys).all(dim=-1)
                            dn = (err["route_end"] | ~fin).cpu().numpy()
                            to = np.zeros(n, bool)
                        else:
                            with torch.no_grad():
                                a = policy(env.obs_buf)
                            _, _, dn_t, _ = env.step(a)
                            err = env._route_errors()
                            dn = dn_t.bool().cpu().numpy()
                            to = env.time_out_buf.cpu().numpy()
                        route_end = err["route_end"].cpu().numpy()
                        fail |= (~done) & dn & ~to
                        complete |= (~done) & dn & route_end
                        vx_tr[:, s] = np.where(done, np.nan, env.z1_phys[:, VX].cpu().numpy())
                        pitch_tr[:, s] = np.where(done, np.nan, np.degrees(env.z1_phys[:, 3].cpu().numpy()))
                        prog = np.where(done, prog, env.progress_m.cpu().numpy())
                        done |= dn
                        if done.all():
                            break
                    return {"vx": vx_tr, "pitch": pitch_tr, "progress_m": prog, "completed": complete, "failed": fail, "steps": (~np.isnan(vx_tr)).sum(1)}

                res = {}
                # (a) rest + recorded controls ; (d) rest + tracker ; (e) rest + healthy loads + recorded controls
                seed_rest(); res["rest_rec"] = run("rec", np.zeros(n, int), H)
                seed_rest(); res["rest_pol"] = run("pol", np.zeros(n, int), H)
                seed_rest(fz_override=[6250.0] * 4); res["rest_rec_healthy_fz"] = run("rec", np.zeros(n, int), H)
                seed_rest(map_probe=True); res["rest_rec_wrongmap"] = run("rec", np.zeros(n, int), H)
                # (b)/(c) from the recorded context 2 s before the stop (non-feasible) / at the same station... for controls use mid-route
                # non-feasible: the recorded stop frame; feasible controls: the frame where they passed the station at which
                # the stalled siblings of the same layout stopped (median), so the local test compares like with like
                stop_station = defaultdict(list)
                for k in keys_bad:
                    if diag[k]["class"] == "stop":
                        stop_station[diag[k]["layout"]].append(diag[k]["progress_end_m"])
                t_stop = np.zeros(n, int)
                for i, k in enumerate(keys):
                    if k in diag and diag[k].get("stop_s") is not None:
                        t_stop[i] = int(round(diag[k]["stop_s"] / DT))
                    else:
                        e = eps[k]; prog = route_progress(e["pose"], e["route_waypoints"], e["route_stations"])
                        target = np.median(stop_station[labels[k]["layout"]]) if stop_station[labels[k]["layout"]] else 0.4 * float(e["route_stations"][-1])
                        hit = np.nonzero(prog >= target)[0]
                        t_stop[i] = int(hit[0]) if len(hit) else n_rec[i] - 2
                t0 = np.clip(t_stop - int(round(args.lead_s / DT)), env.context, n_rec - 2)
                def seed_at(t_start: np.ndarray, map_probe: bool = False):
                    st = torch.tensor(t_start, device=dev)
                    env.reset_idx(ids, episode_ids=ids, start_frames=st, fragment_steps=torch.full((n,), H, device=dev, dtype=torch.long))
                    if map_probe:
                        wrong_maps()
                        win = st[:, None] + torch.arange(-env.context, 0, device=dev)[None, :]
                        with torch.no_grad():
                            env.token_hist[:] = env.model.cropper(env.env_maps, b.pose[ids[:, None], win])
                    env._compute_observations()

                for mode in ("rec", "pol"):
                    seed_at(t0); res[f"pre_{mode}"] = run(mode, t0 - 1, int(round(args.post_s / DT)))
                seed_at(t0, map_probe=True); res["pre_rec_wrongmap"] = run("rec", t0 - 1, int(round(args.post_s / DT)))
                # seeded INSIDE the stuck phase (1 s after the stop, vehicle stationary, throttle on): does the model hold the stall?
                t_in = np.clip(t_stop + 20, env.context, n_rec - 2)
                seed_at(t_in); res["stuck_rec"] = run("rec", t_in - 1, int(round(args.post_s / DT)))
                # local k-step teacher-forced errors around the stop: start frames stop-2s .. stop+1s every 0.2 s, 20 steps each
                KS = (1, 4, 8, 20)
                offsets = list(range(-40, 21, 4))
                local = np.full((n, len(offsets), len(KS)), np.nan, np.float32)  # signed vx error pred - rec at frame t+k-1
                for oi, off in enumerate(offsets):
                    t_s = np.clip(t_stop + off, env.context, n_rec - 21)
                    ok_ = (t_stop + off >= env.context) & (t_stop + off <= n_rec - 21)
                    seed_at(t_s)
                    t_t = torch.tensor(t_s, device=dev)
                    for st in range(20):
                        env._nn_step(b.act_raw[ids, t_t - 1 + st]); env.act_hist[:, -1] = env.act_hist[:, -1]
                        if st + 1 in KS:
                            ki = KS.index(st + 1)
                            pred = env.z1_phys[:, VX].cpu().numpy()
                            rec = rec_vx[np.arange(n), np.minimum(t_s + st, b.n_frames - 1)]
                            local[:, oi, ki] = np.where(ok_, pred - rec, np.nan)
                for i, k in enumerate(keys):
                    r = results.setdefault(k, {"arena": aid, "layout": labels[k]["layout"], "candidate": labels[k]["candidate"], "class": diag[k]["class"] if k in diag else "feasible",
                                               "label": diag[k]["label"] if k in diag else "feasible", "n_rec": int(n_rec[i]), "stop_s": diag[k].get("stop_s") if k in diag else None,
                                               "rec_vx": rec_vx[i, : n_rec[i]].round(3).tolist(), "rec_pitch": np.degrees(eps[k]["z1"][:, 3]).round(2).tolist(), "rec_progress": route_progress(eps[k]["pose"], eps[k]["route_waypoints"], eps[k]["route_stations"]).round(2).tolist(), "t0_pre_s": float(t0[i] * DT), "models": {}})
                    m = {}
                    for name, rr in res.items():
                        m[name] = {"vx": np.nan_to_num(rr["vx"][i], nan=-99).round(3).tolist(), "pitch": np.nan_to_num(rr["pitch"][i], nan=-99).round(2).tolist(), "progress_m": float(rr["progress_m"][i]), "completed": bool(rr["completed"][i]), "failed": bool(rr["failed"][i]), "steps": int(rr["steps"][i])}
                    m["local"] = {"offsets_s": [o * DT for o in offsets], "ks": list(KS), "err": np.nan_to_num(local[i], nan=-99).round(3).tolist()}
                    r["models"][mname] = m
                # quick print: predicted vs recorded vx 3 s after rest / 2 s after the stop
                def at(tr, s):
                    return np.array([tr[i, min(s, tr.shape[1] - 1)] if s < tr.shape[1] else np.nan for i in range(n)])
                bad = np.array([k in diag for k in keys])
                for name in ("rest_rec", "rest_rec_healthy_fz", "rest_rec_wrongmap", "rest_pol"):
                    p3 = at(res[name]["vx"], 59); r3 = np.array([rec_vx[i, min(59, n_rec[i] - 1)] for i in range(n)])
                    print(f"  {mname:24s} {name:20s} vx@3s  non-feasible: pred {np.nanmean(p3[bad]):.2f} rec {r3[bad].mean():.2f} | feasible ctrl: pred {np.nanmean(p3[~bad]) if (~bad).any() else float('nan'):.2f} rec {r3[~bad].mean() if (~bad).any() else float('nan'):.2f}"
                          f" | pred completes {res[name]['completed'][bad].sum()}/{bad.sum()} vs {res[name]['completed'][~bad].sum()}/{(~bad).sum()}")
                p_in = at(res["stuck_rec"]["vx"], 79); r_in = np.array([rec_vx[i, min(t_in[i] + 79, n_rec[i] - 1)] for i in range(n)])
                print(f"  {mname:24s} {'stuck_rec':20s} seeded 1 s into the stall, 4 s later: non-feasible pred {np.nanmean(p_in[bad]):.2f} rec {r_in[bad].mean():.2f} (pred<0.5: {(p_in[bad] < 0.5).sum()}/{bad.sum()})")
                for name in ("pre_rec", "pre_rec_wrongmap", "pre_pol"):
                    s2 = int(round((args.lead_s + 2.0) / DT))
                    p = at(res[name]["vx"], s2); r_ = np.array([rec_vx[i, min(t0[i] + s2, n_rec[i] - 1)] for i in range(n)])
                    print(f"  {mname:24s} {name:20s} vx 2 s after the stop: non-feasible pred {np.nanmean(p[bad]):.2f} rec {r_[bad].mean():.2f} (pred<0.5: {(p[bad] < 0.5).sum()}/{bad.sum()}) | ctrl pred {np.nanmean(p[~bad]) if (~bad).any() else float('nan'):.2f} rec {r_[~bad].mean() if (~bad).any() else float('nan'):.2f}")
                del env, policy
                torch.cuda.empty_cache()
    (Path(args.out) / f"model_tests_{args.tag}.json").write_text(json.dumps(results))
    print(f"\nwrote {Path(args.out) / f'model_tests_{args.tag}.json'} ({len(results)} runs)")


def cmd_analyze(args) -> None:
    R = json.loads((Path(args.out) / f"model_tests_{args.tag}.json").read_text())
    models = sorted(set(m for r in R.values() for m in r["models"]))
    classes = ["launch", "stop", "feasible"]
    g = lambda a: np.array(a, float)
    vx_at = lambda tr, s: (tr[s] if s < len(tr) and tr[s] > -90 else np.nan)

    def stat(rows, model, fn):
        v = g([fn(r, r["models"][model]) for r in rows]); v = v[np.isfinite(v)]
        return (v.mean(), len(v)) if len(v) else (np.nan, 0)

    print(f"model tests [{args.tag}]: {len(R)} runs; " + ", ".join(f"{c}={sum(r['class'] == c for r in R.values())}" for c in classes))
    for cls in classes:
        rows = [r for r in R.values() if r["class"] == cls]
        if not rows:
            continue
        print(f"\n--- {cls} (n={len(rows)}) --- recorded vx@3s {np.nanmean([vx_at(r['rec_vx'], 59) for r in rows]):.2f}; recorded vx 2 s after the stop {np.nanmean([vx_at(r['rec_vx'], int(round((r['t0_pre_s'] + 4.0) / DT))) for r in rows]):.2f}")
        hdr = f"{'model':24s} {'rest+rec vx@3s':>15s} {'>1m/s':>6s} {'healthyFz':>10s} {'wrongMap':>9s} {'rest+trk done':>14s} | {'pre+rec vx@+2s':>15s} {'<0.5':>6s} {'wrongMap':>9s} {'pre+trk vx':>10s} {'<0.5':>6s} | k8 err pre/post | k20 err pre/post"
        print(hdr)
        for m in models:
            a, _ = stat(rows, m, lambda r, x: vx_at(x["rest_rec"]["vx"], 59))
            a1, _ = stat(rows, m, lambda r, x: float(vx_at(x["rest_rec"]["vx"], 59) > 1.0))
            h, _ = stat(rows, m, lambda r, x: vx_at(x["rest_rec_healthy_fz"]["vx"], 59))
            w, _ = stat(rows, m, lambda r, x: vx_at(x["rest_rec_wrongmap"]["vx"], 59))
            d, _ = stat(rows, m, lambda r, x: float(x["rest_pol"]["completed"]))
            s2 = int(round(4.0 / DT))  # lead 2 s + 2 s after the stop (post window 6 s)
            p, _ = stat(rows, m, lambda r, x: vx_at(x["pre_rec"]["vx"], s2))
            p1, _ = stat(rows, m, lambda r, x: float(vx_at(x["pre_rec"]["vx"], s2) < 0.5))
            pw, _ = stat(rows, m, lambda r, x: vx_at(x["pre_rec_wrongmap"]["vx"], s2))
            q, _ = stat(rows, m, lambda r, x: vx_at(x["pre_pol"]["vx"], s2))
            si, _ = stat(rows, m, lambda r, x: vx_at(x["stuck_rec"]["vx"], 79) if "stuck_rec" in x else np.nan)
            si1, _ = stat(rows, m, lambda r, x: float(vx_at(x["stuck_rec"]["vx"], 79) < 0.5) if "stuck_rec" in x else np.nan)
            q1, _ = stat(rows, m, lambda r, x: float(vx_at(x["pre_pol"]["vx"], s2) < 0.5))
            err = np.array([r["models"][m]["local"]["err"] for r in rows], float); err[err < -90] = np.nan
            offs = np.array(rows[0]["models"][m]["local"]["offsets_s"]); ks = rows[0]["models"][m]["local"]["ks"]
            pre_m, post_m = offs < 0, offs >= 0
            e8 = (np.nanmean(err[:, pre_m, ks.index(8)]), np.nanmean(err[:, post_m, ks.index(8)]))
            e20 = (np.nanmean(err[:, pre_m, ks.index(20)]), np.nanmean(err[:, post_m, ks.index(20)]))
            print(f"{m:24s} {a:15.2f} {a1:6.2f} {h:10.2f} {w:9.2f} {d:14.2f} | {p:15.2f} {p1:6.2f} {pw:9.2f} {q:10.2f} {q1:6.2f} | {e8[0]:+.2f} / {e8[1]:+.2f}   | {e20[0]:+.2f} / {e20[1]:+.2f}   | in-stall +4s {si:.2f} (<0.5: {si1:.2f})")
    # worked examples
    for cls in ("launch", "stop"):
        rows = [r for r in R.values() if r["class"] == cls][: args.examples]
        for r in rows:
            print(f"\n{r['layout']}__{r['candidate']} [{r['label']}] recorded stop {r['stop_s']} s, pre-test start {r['t0_pre_s']} s")
            ts = [1, 2, 3, 4, 6, 8, 10]
            print(f"   {'t (s)':22s}" + "".join(f"{t:7d}" for t in ts))
            print(f"   {'Chrono vx':22s}" + "".join(f"{vx_at(r['rec_vx'], t * 20 - 1):7.2f}" for t in ts))
            for m in models:
                print(f"   {m[:14] + ' rest+rec':22s}" + "".join(f"{vx_at(r['models'][m]['rest_rec']['vx'], t * 20 - 1):7.2f}" for t in ts))
            if cls == "stop":
                ts2 = [0, 1, 2, 3, 4, 5]
                t0f = int(round(r["t0_pre_s"] / DT))
                print(f"   {'after pre-start (s)':22s}" + "".join(f"{t:7d}" for t in ts2))
                print(f"   {'Chrono vx':22s}" + "".join(f"{vx_at(r['rec_vx'], t0f + t * 20):7.2f}" for t in ts2))
                for m in models:
                    print(f"   {m[:14] + ' pre+rec':22s}" + "".join(f"{vx_at(r['models'][m]['pre_rec']['vx'], t * 20):7.2f}" for t in ts2))
                    print(f"   {m[:14] + ' pre+trk':22s}" + "".join(f"{vx_at(r['models'][m]['pre_pol']['vx'], t * 20):7.2f}" for t in ts2))


# ------------------------------------------------------------------------------------- predictability probe
def cmd_predictable(args) -> None:
    """Is the stall predictable from what the dynamics model is given? Small MLPs on the same 16-frame context
    (17-D state + controls) and the next ``lead`` recorded controls -> 'stuck lead frames later'; trained on the
    training arenas, judged (AUC) on the validation and sealed arenas. Plus a rest-state probe for launch failures."""
    import torch
    from traverse_wp6_imagine_sweep import auc
    diag = json.loads((Path(args.out) / "classify.json").read_text())["rows"]
    C = 16
    eps_by_arena = defaultdict(list)
    for cdir in args.caches:
        cache = Path(cdir)
        man = json.loads((cache / "cache_manifest.json").read_text()); labels = json.loads((cache / "labels.json").read_text())
        tmaps = {a: TerrainMap.from_dir(Path(p)) for a, p in man["arenas"].items()}
        for k in man["episodes"]:
            e = load_episode(cache, k); a = str(e["arena"])
            r = diag.get(k)
            cls = r["class"] if r else "feasible"
            stop = None
            if r and cls == "launch":
                stop = 0
            elif r and cls == "stop":
                stop = int(round(r["stop_s"] / DT))
            x, y, yaw = (float(v) for v in e["pose"][0])
            sta = e["route_stations"]; v5 = float(e["route_speeds"][sta <= 5.0].mean())
            patches = {}
            if args.terrain:
                pose = e["pose"]; tm = tmaps[a]
                for name, (fw, lat) in {"fine": (np.arange(0.0, 6.01, 0.25), np.arange(-1.5, 1.51, 0.25)), "coarse": (np.linspace(-5, 5, 8), np.linspace(-5, 5, 8))}.items():
                    du, dv = np.meshgrid(fw, lat, indexing="ij"); du, dv = du.ravel(), dv.ravel()
                    cy, sy = np.cos(pose[:, 2])[:, None], np.sin(pose[:, 2])[:, None]
                    px = pose[:, 0:1] + cy * du[None] - sy * dv[None]; py = pose[:, 1:2] + sy * du[None] + cy * dv[None]
                    h = tm.height(px.ravel(), py.ravel()).reshape(px.shape) - tm.height(pose[:, 0], pose[:, 1])[:, None]
                    patches[name] = h.astype(np.float32)
            eps_by_arena[a].append({"key": k, "z1": e["z1"], "act": e["act"], "cls": cls, "stop": stop, "n": len(e["z1"]), "layout": labels[k]["layout"], "patches": patches,
                                    "rest": np.concatenate([e["z1"][0], [v5, slope_ahead_deg(tmaps[a], x, y, yaw), slope_ahead_deg(tmaps[a], x, y, yaw, 5.0)]]).astype(np.float32),
                                    "launch_fail": cls == "launch", "feasible": cls == "feasible"})
    train_a = [a for a in eps_by_arena if a in args.train_arenas]
    evals = {name: [a for a in eps_by_arena if a in arenas] for name, arenas in (("val", args.val_arenas), ("sealed", args.test_arenas))}
    dev = args.device

    def samples(arenas, lead, stride=4, only_moving=True):
        X, Y, G = [], [], []
        for a in arenas:
            for e in eps_by_arena[a]:
                z1, act, n = e["z1"], e["act"], e["n"]
                for t in range(C, n - lead, stride):
                    stuck = e["stop"] is not None and t + lead >= e["stop"]
                    if only_moving and e["stop"] is not None and t >= e["stop"]:
                        continue  # already stuck: trivial
                    X.append(np.concatenate([z1[t - C:t].ravel(), act[t - C:t].ravel(), act[t:t + lead].ravel()] + [e["patches"][nm][t - 1] for nm in ("fine", "coarse") if nm in e["patches"]]))
                    Y.append(float(stuck)); G.append(e["cls"])
        return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(G)

    def fit_mlp(X, Y, Xv, Yv, cols, epochs=30, seed=0):
        torch.manual_seed(seed)
        mu, sd = X[:, cols].mean(0), X[:, cols].std(0) + 1e-6
        f = lambda A: torch.tensor((A[:, cols] - mu) / sd, device=dev)
        Xt, Yt, Xvt = f(X), torch.tensor(Y, device=dev), f(Xv)
        net = torch.nn.Sequential(torch.nn.Linear(len(cols), 256), torch.nn.GELU(), torch.nn.Linear(256, 64), torch.nn.GELU(), torch.nn.Linear(64, 1)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        pos_w = torch.tensor([(1 - Y.mean()) / max(Y.mean(), 1e-3)], device=dev)
        best = (-1, None); bs = 1024
        for ep in range(epochs):
            perm = torch.randperm(len(Xt), device=dev)
            net.train()
            for i in range(0, len(Xt), bs):
                idx = perm[i:i + bs]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[idx])[:, 0], Yt[idx], pos_weight=pos_w)
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                pv = net(Xvt)[:, 0].cpu().numpy()
            a_ = auc(pv, Yv > 0.5)
            if a_ > best[0]:
                best = (a_, {k_: v.clone() for k_, v in net.state_dict().items()})
        net.load_state_dict(best[1]); net.eval()
        return lambda A: net(f(A))[:, 0].detach().cpu().numpy(), best[0]

    print(f"predictability of 'stuck <lead> later' from the model's own inputs; train {train_a}, val {evals['val']}, sealed {evals['sealed']}")
    for lead in args.leads:
        X, Y, _ = samples(train_a, lead)
        Xv, Yv, Gv = samples(evals["val"], lead)
        Xs, Ys, Gs = samples(evals["sealed"], lead)
        n_ctx = C * 17 + C * 3
        n_base = n_ctx + lead * 3
        variants = {"context+future controls": list(range(n_base)), "context only": list(range(n_ctx)),
                    "last frame + future controls": list(range(C * 17 - 17, C * 17)) + list(range(C * 17 + C * 3 - 3, X.shape[1])),
                    "last frame, no tire loads/omegas": list(range(C * 17 - 17, C * 17 - 10)) + list(range(C * 17 - 2, C * 17)) + list(range(C * 17 + C * 3 - 3, n_base))}
        if args.terrain:
            n_fine = 25 * 13; last_fut = list(range(C * 17 - 17, C * 17)) + list(range(C * 17 + C * 3 - 3, n_base))
            variants["last frame + future controls + TRUE terrain 0.25 m patch (6 x 3 m ahead)"] = last_fut + list(range(n_base, n_base + n_fine))
            variants["last frame + future controls + coarse 1.4 m patch (+-5 m, the crop's grid)"] = last_fut + list(range(n_base + n_fine, X.shape[1]))
            variants["context + future controls + TRUE terrain 0.25 m patch"] = list(range(n_base + n_fine))
            variants["TRUE terrain 0.25 m patch only"] = list(range(n_base, n_base + n_fine))
        print(f"\nlead {lead * DT:.0f} s: train {len(X)} samples ({100 * Y.mean():.1f} % positive), val {len(Xv)} ({100 * Yv.mean():.1f} %), sealed {len(Xs)} ({100 * Ys.mean():.1f} %)")
        for name, cols in variants.items():
            pred, a_val = fit_mlp(X, Y, Xv, Yv, cols)
            ps = pred(Xs); a_seal = auc(ps, Ys > 0.5)
            # sensitivity at 5 % false positives on the sealed set
            neg = np.sort(ps[Ys < 0.5]); thr = neg[int(0.95 * len(neg))] if len(neg) else np.inf
            sens = float((ps[Ys > 0.5] > thr).mean()) if (Ys > 0.5).any() else float("nan")
            print(f"  {name:36s} AUC val {a_val:.3f}  sealed {a_seal:.3f}  (sealed sensitivity at 5 % FPR: {sens:.2f})")
    # launch probe: rest state (+ commanded first-5 m speed, slope ahead) -> launch failure
    if args.skip_launch:
        return
    R = lambda arenas: (np.stack([e["rest"] for a in arenas for e in eps_by_arena[a]]), np.array([float(e["launch_fail"]) for a in arenas for e in eps_by_arena[a]], np.float32))
    X, Y = R(train_a); Xv, Yv = R(evals["val"]); Xs, Ys = R(evals["sealed"])
    print(f"\nlaunch failure from the rest state: train {len(X)} ({int(Y.sum())} failures), val {len(Xv)} ({int(Yv.sum())}), sealed {len(Xs)} ({int(Ys.sum())})")
    for name, cols in {"17-D rest state + v_cmd + slope": list(range(20)), "rest state only": list(range(17)), "tire loads + omegas only": list(range(7, 15)),
                       "attitude + v_cmd + slope (no loads)": [2, 3, 17, 18, 19], "v_cmd + slope only": [17, 18, 19]}.items():
        pred, a_val = fit_mlp(X, Y, Xv, Yv, cols, epochs=200)
        print(f"  {name:36s} AUC val {a_val:.3f}  sealed {auc(pred(Xs), Ys > 0.5):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("classify")
    c.add_argument("--caches", nargs="+", default=["artifacts/traverse/wp7_cache_v1", "artifacts/traverse/wp7_cache_sealed"])
    c.add_argument("--imagine", nargs="*", default=[], help="NAME=rows.json of an imagination run (cross-tab)")
    c.add_argument("--all", action="store_true", help="classify feasible runs too")
    c.add_argument("--out", default="artifacts/traverse/wp7_stall_diag")
    m = sub.add_parser("model")
    m.add_argument("--caches", nargs="+", default=["artifacts/traverse/wp7_cache_v1"])
    m.add_argument("--arenas", nargs="+", default=["arena_f105"])
    m.add_argument("--classes", nargs="+", default=["launch", "stop"])
    m.add_argument("--dynamics-checkpoints", nargs="+", required=True)
    m.add_argument("--policy", default="artifacts/traverse/wp3_tracker_v1")
    m.add_argument("--horizon-s", type=float, default=30.0)
    m.add_argument("--lead-s", type=float, default=2.0, help="start the pre-stop tests this long before the recorded stop")
    m.add_argument("--post-s", type=float, default=6.0, help="length of the pre-stop tests")
    m.add_argument("--camera-start", action="store_true", help="rest tests start at the camera pose estimate (planner) instead of the true pose")
    m.add_argument("--roll-limit-deg", type=float, default=34.4)
    m.add_argument("--pitch-limit-deg", type=float, default=22.9)
    m.add_argument("--max-per-arena", type=int, default=0)
    m.add_argument("--device", default="cuda")
    m.add_argument("--out", default="artifacts/traverse/wp7_stall_diag")
    m.add_argument("--tag", default="f105")
    pr = sub.add_parser("predictable")
    pr.add_argument("--caches", nargs="+", default=["artifacts/traverse/wp7_cache_v1", "artifacts/traverse/wp7_cache_sealed"])
    pr.add_argument("--train-arenas", nargs="+", default=["arena_f101", "arena_f102", "arena_f103", "arena_f104"])
    pr.add_argument("--val-arenas", nargs="+", default=["arena_f105"])
    pr.add_argument("--test-arenas", nargs="+", default=["arena_f106", "arena_f107"])
    pr.add_argument("--leads", nargs="+", type=int, default=[20, 40, 80])
    pr.add_argument("--terrain", action="store_true", help="add privileged true-terrain patch variants (information ceiling)")
    pr.add_argument("--skip-launch", action="store_true")
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--out", default="artifacts/traverse/wp7_stall_diag")
    an = sub.add_parser("analyze")
    an.add_argument("--out", default="artifacts/traverse/wp7_stall_diag")
    an.add_argument("--tag", default="f105")
    an.add_argument("--examples", type=int, default=3)
    args = ap.parse_args()
    {"classify": cmd_classify, "model": cmd_model, "analyze": cmd_analyze, "predictable": cmd_predictable}[args.cmd](args)


if __name__ == "__main__":
    main()
