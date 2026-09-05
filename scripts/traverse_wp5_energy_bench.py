#!/usr/bin/env python
"""Offline benchmark of imagined time / energy against every route the tracker has driven in Chrono.

Each Chrono batch (``rows.jsonl`` + the route file it drove) becomes a set of (route, Chrono time,
Chrono energy) triples. Every route is re-imagined with the PPO tracker inside one or more dynamics
checkpoints from the same start pose the planner used (camera estimate for camera-localised batches,
recorded start otherwise), and the estimators are scored: ratio (Chrono / imagined), correlation,
MAE, and -- where a batch holds several candidates per layout -- the within-layout rank correlation
and top-1 agreement of the combined cost time + energy/10. Ensemble estimators (mean / max / mean+std
over checkpoints) come for free. This is the test bed for "make the imagined energy accurate" that
needs no simulator time.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nedm.traverse.nrd_model import DT_S
from nedm.traverse.power_calib import KINDS, PowerModel
from nedm.traverse.tracker_env import TraverseTrackingEnv, merge_env_cfg
from traverse_wp4_score_candidates import load_policy, rollout


def load_batch(batch: Path, controller_substr: str) -> list[dict]:
    args = json.loads((batch / "summary.json").read_text())["args"]
    if not args.get("route_file"):
        return []
    routes = json.loads(Path(args["route_file"]).read_text())
    rows = []
    for line in (batch / "rows.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if controller_substr not in r["controller"] or not r.get("completed"):
            continue
        match = [c for c in routes.get(r["key"], []) if c["candidate"] == r["candidate"]]
        if not match:
            continue
        route = {k: np.asarray(match[0][k], np.float32) for k in ("waypoints", "speeds", "headings", "stations")}
        rows.append({"batch": batch.name, "key": r["key"], "candidate": r["candidate"], "route": route,
                     "camera": args.get("localisation") == "camera", "chrono_time": r["time_s"], "chrono_energy": r["energy_kj"]})
    return rows


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def score(name: str, est: np.ndarray, rows: list[dict], key_of=lambda r: (r["batch"], r["key"])) -> dict:
    ch = np.array([r["chrono_energy"] for r in rows])
    t_ch = np.array([r["chrono_time"] for r in rows]); t_img = np.array([r["_time"] for r in rows])
    out = {"estimator": name, "n": len(rows), "ratio": float(ch.mean() / max(est.mean(), 1e-6)),
           "corr": float(np.corrcoef(est, ch)[0, 1]), "mae_kj": float(np.abs(est - ch).mean()),
           "median_log_ratio": float(np.median(np.log(np.maximum(ch, 1) / np.maximum(est, 1))))}
    # within-layout ranking on the combined cost (only groups with >= 3 candidates)
    groups: dict = {}
    for i, r in enumerate(rows):
        groups.setdefault(key_of(r), []).append(i)
    sp, agree, n_g = [], 0, 0
    for idx in groups.values():
        if len(idx) < 3:
            continue
        idx = np.array(idx)
        c_img = t_img[idx] + est[idx] / 10.0; c_ch = t_ch[idx] + ch[idx] / 10.0
        sp.append(spearman(c_img, c_ch)); agree += int(np.argmin(c_img) == np.argmin(c_ch)); n_g += 1
    out.update({"groups": n_g, "spearman": float(np.nanmean(sp)) if sp else None, "top1_agree": agree})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batches", nargs="+", default=[
        "artifacts/traverse/wp5_chrono_sample_planner", "artifacts/traverse/wp5_chrono_sample_planner_v2",
        "artifacts/traverse/wp4_chrono_allsensor", "artifacts/traverse/wp4_chrono_loc_camera_pred_occ",
        "artifacts/traverse/wp4_chrono_pred_full_r40", "artifacts/traverse/wp4_chrono_fallback"])
    ap.add_argument("--dynamics-checkpoints", nargs="+", default=["artifacts/traverse/wp2_mapv2_dagger2_ro8_amd/ckpt_best.pt"])
    ap.add_argument("--z1-extra-cache", nargs="*", default=None,
                    help="sidecar per checkpoint ('' for none); one entry applies to all")
    ap.add_argument("--policy", default="artifacts/traverse/wp3_tracker_v1")
    ap.add_argument("--start-poses", default="artifacts/traverse/wp4_start_poses/val_start_poses.json")
    ap.add_argument("--power-calib", nargs="+", default=["artifacts/traverse/wp4_power_calib/power_calib.json"])
    ap.add_argument("--cache", default="artifacts/traverse/wp2_z2_cache_v6")
    ap.add_argument("--routes", default="artifacts/traverse/wp3_routes")
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--controller", default="tracker")
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = [r for b in args.batches for r in load_batch(Path(b), args.controller)]
    print(f"{len(rows)} Chrono-driven routes from {len(args.batches)} batches: "
          + ", ".join(f"{b.split('/')[-1]} {sum(r['batch'] == b.split('/')[-1] for r in rows)}" for b in args.batches), flush=True)
    start_est = json.loads(Path(args.start_poses).read_text())
    horizon = int(round(args.horizon_s / DT_S))
    sidecars = args.z1_extra_cache or [""]
    if len(sidecars) == 1:
        sidecars = sidecars * len(args.dynamics_checkpoints)
    power_models = {}
    for pc in args.power_calib:
        tag = "" if pc == args.power_calib[0] else "@" + Path(pc).parent.name.replace("wp5_power_calib_", "").replace("wp4_power_calib", "rec")
        for k in KINDS:
            power_models[k + tag] = PowerModel.load(Path(pc), k)
    entries = [(r["key"], r["route"]) for r in rows]
    per_ckpt = {}
    for ckpt, sidecar in zip(args.dynamics_checkpoints, sidecars):
        t0 = time.perf_counter()
        cfg = merge_env_cfg({"num_envs": len(entries), "device": args.device, "auto_reset": False, "split": "val",
                             "dynamics_checkpoint": ckpt, "arena": args.arena, "cache": args.cache, "routes": args.routes,
                             "fragment_steps_max": horizon, "z1_extra_cache": sidecar or None})
        env = TraverseTrackingEnv(cfg, device=args.device, entries=entries)
        policy = load_policy(Path(args.policy), env, args.device)
        # default start = the recorded pose at the end of the context window (the env's initial reset picks a
        # random fragment, so it must be re-seated before cloning); camera batches start from the camera estimate
        n_env = torch.arange(len(entries), device=env.device)
        env.reset_idx(n_env, episode_ids=n_env, start_frames=torch.full_like(n_env, env.context),
                      fragment_steps=torch.full_like(n_env, horizon))
        sp = env.pose.clone()
        for i, r in enumerate(rows):
            if r["camera"] and r["key"] in start_est:
                sp[i] = torch.tensor(start_est[r["key"]]["est"], device=env.device, dtype=sp.dtype)
        dummy = torch.full((len(entries), 1, 3), -1.0, device=env.device)
        res = rollout(env, policy, horizon, dummy, dummy, power_models, sp)
        res = {k: v.cpu().numpy() for k, v in res.items() if not k.startswith("_")}
        name = Path(ckpt).parent.name
        per_ckpt[name] = res
        print(f"{name}: imagined {len(entries)} routes in {time.perf_counter() - t0:.0f}s; completed {res['completed'].mean():.3f}", flush=True)
        del env, policy; torch.cuda.empty_cache()

    t_ch = np.array([r["chrono_time"] for r in rows])
    results, table = {"n_routes": len(rows), "checkpoints": list(per_ckpt), "time": {}, "energy": []}, []
    for name, res in per_ckpt.items():
        ok = res["completed"]
        results["time"][name] = {"completed": float(ok.mean()), "bias": float((res["time_s"][ok] / t_ch[ok]).mean() - 1),
                                 "corr": float(np.corrcoef(res["time_s"][ok], t_ch[ok])[0, 1]), "by_batch": {}}
        for b in sorted({r["batch"] for r in rows}):
            m = np.array([r["batch"] == b for r in rows]) & ok
            results["time"][name]["by_batch"][b] = {"completed": float(ok[np.array([r["batch"] == b for r in rows])].mean()),
                                                     "bias": float((res["time_s"][m] / t_ch[m]).mean() - 1) if m.any() else None}
        print(f"time {name}: completed {ok.mean():.3f} bias {results['time'][name]['bias']:+.3f} corr {results['time'][name]['corr']:.3f} | "
              + " ".join(f"{b.split('_')[-1][:6]}:{v['bias']:+.2f}" for b, v in results["time"][name]["by_batch"].items() if v["bias"] is not None))
    # estimators: per checkpoint (head, calibrated models, pessimistic), then ensembles over checkpoints
    ests: dict[str, np.ndarray] = {}
    for name, res in per_ckpt.items():
        short = name.replace("wp2_mapv2_", "")
        ests[f"{short}/head"] = res["energy_kj"]
        for k in res:
            if k.startswith("energy_") and k != "energy_kj":
                ests[f"{short}/{k[7:-3]}"] = res[k]
        ests[f"{short}/pess(head,act)"] = np.maximum(res["energy_kj"], res["energy_act_kj"])
    if len(per_ckpt) > 1:
        heads = np.stack([r["energy_kj"] for r in per_ckpt.values()])
        ests["ens/head_mean"] = heads.mean(0); ests["ens/head_max"] = heads.max(0)
        ests["ens/head_mean+std"] = heads.mean(0) + heads.std(0)
        acts = np.stack([r["energy_act_kj"] for r in per_ckpt.values()])
        ests["ens/pess_all"] = np.maximum(heads, acts).max(0)
    # use the first checkpoint's imagined time for the combined-cost ranking
    first = next(iter(per_ckpt.values()))
    for r, t in zip(rows, first["time_s"]):
        r["_time"] = float(t)
    complete_all = np.all([res["completed"] for res in per_ckpt.values()], axis=0)
    sel = [r for r, c in zip(rows, complete_all) if c]
    for name, e in ests.items():
        s = score(name, e[complete_all], sel)
        s["by_batch"] = {}
        for b in sorted({r["batch"] for r in sel}):
            m = np.array([r["batch"] == b for r in sel])
            ch = np.array([r["chrono_energy"] for r in sel])[m]; ee = e[complete_all][m]
            s["by_batch"][b] = {"ratio": float(ch.mean() / max(ee.mean(), 1e-6)), "corr": float(np.corrcoef(ee, ch)[0, 1]) if m.sum() > 2 else None}
        results["energy"].append(s); table.append(s)
    print(f"\n{'estimator':34s} {'ratio':>6s} {'corr':>6s} {'MAE':>6s} {'medLR':>6s} | {'groups':>6s} {'rho':>6s} {'top1':>5s} | per-batch ratio")
    for s in table:
        pb = " ".join(f"{b.split('_')[-1][:6]}:{v['ratio']:.2f}" for b, v in s["by_batch"].items())
        print(f"{s['estimator']:34s} {s['ratio']:6.2f} {s['corr']:6.3f} {s['mae_kj']:6.1f} {s['median_log_ratio']:+6.2f} | "
              f"{s['groups']:6d} {s['spearman'] if s['spearman'] is not None else float('nan'):6.3f} {s['top1_agree']:5d} | {pb}")
    np.savez_compressed(out / "per_route.npz", batch=np.array([r["batch"] for r in rows]), key=np.array([r["key"] for r in rows]),
                        candidate=np.array([r["candidate"] for r in rows]), chrono_time=t_ch,
                        chrono_energy=np.array([r["chrono_energy"] for r in rows]),
                        **{f"{n}__{k}": v for n, res in per_ckpt.items() for k, v in res.items()})
    (out / "energy_bench.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
