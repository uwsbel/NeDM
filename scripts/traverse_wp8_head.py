#!/usr/bin/env python
"""Design (a) of plan §32, cheapest decisive form (closeout review 2026-09-07): does a small predictor that reads the
IMAGINED trajectory (17-D state, pose, 256-D crop tokens along the imagined path) predict the recorded outcome of a route
better than the same predictor reading only the NOMINAL route (profile features + the same tokens along the nominal path),
and does either beat the terrain-profile predictor? Trained on f101-f104 (early stopping leave-one-arena-out), judged on
f105 with layout-cluster bootstrap CIs. Pre-registered endpoints in notes §13.12.

  dump    --model CKPT --policy tracker|pure_pursuit|openloop --out DIR   (imagined sequences from rest, camera start pose)
  nominal --model CKPT --out DIR                                          (nominal profile + tokens along the nominal path)
  train   --arms NAME=DIR ... --seeds 3                                   (heads, endpoints)
"""
from __future__ import annotations

import argparse, json, math, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nedm.traverse.terrain import TerrainMap

CACHE = Path("artifacts/traverse/wp7_cache_v1")
ARENAS = ["arena_f101", "arena_f102", "arena_f103", "arena_f104", "arena_f105"]
TRAIN, VAL = ARENAS[:4], ["arena_f105"]
DIAG = Path("artifacts/traverse/wp7_stall_diag/classify.json")
SUB = 4  # record every 4th step (0.2 s)
H = 600


def labels_and_classes(caches=None):
    """route labels over one or more caches (the classify.json rows cover every cache that has been classified)"""
    diag = json.loads(DIAG.read_text())["rows"]
    out, man = {}, {"episodes": [], "arena_of": {}, "arenas": {}}
    for cdir in (caches or [CACHE]):
        labels = json.loads((Path(cdir) / "labels.json").read_text()); m = json.loads((Path(cdir) / "cache_manifest.json").read_text())
        man["episodes"] += m["episodes"]; man["arena_of"].update(m["arena_of"]); man["arenas"].update(m["arenas"])
        for k in m["episodes"]:
            out[k] = None; out[k] = (labels, m)
    res = {}
    for k, (labels, m) in out.items():
        c = labels[k]; r = diag.get(k)
        feas = bool(c.get("completed")) and not c.get("stalled") and not c.get("contact")
        res[k] = {"arena": m["arena_of"][k], "layout": c["layout"], "candidate": c["candidate"], "feasible": feas, "cls": r["class"] if r else ("feasible" if feas else "infeasible"),
                  "contact_only": bool(c.get("completed")) and not c.get("stalled") and bool(c.get("contact")), "cache": str(cdir_of(k, caches)),
                  "mean_speed": c["mean_speed"], "cost": c["time_s"] + c["energy_kj"] / 10, "time_s": c["time_s"], "energy_kj": c["energy_kj"]}
    return res, man


def cdir_of(k, caches):
    for cdir in (caches or [CACHE]):
        if (Path(cdir) / f"{k}.npz").exists():
            return Path(cdir)
    return CACHE


def cmd_dump(args) -> None:
    import torch
    from nedm.traverse.oracle import PlanCandidate
    from nedm.traverse.tracker_env import TraverseTrackingEnv, merge_env_cfg, pure_pursuit_actions
    from traverse_wp4_score_candidates import load_policy, route_dict
    from traverse_wp7_stall_diagnosis import tracker_action_center
    dev = args.device
    lab, man = labels_and_classes(args.caches)
    start_est = {}
    for cdir in args.caches:
        start_est.update(json.loads((Path(cdir) / "start_poses.json").read_text()))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for aid in args.arenas:
        keys = [k for k in man["episodes"] if man["arena_of"][k] == aid]
        if not keys:
            continue
        cache = Path(lab[keys[0]]["cache"])
        entries, starts = [], []
        for k in keys:
            with np.load(cache / f"{k}.npz") as z:
                plan = PlanCandidate(waypoints=z["route_waypoints"].astype(float), speeds=z["route_speeds"].astype(float), headings=z["route_headings"].astype(float),
                                     stations=z["route_stations"].astype(float), meta={})
            entries.append((k, route_dict(plan))); starts.append(start_est[lab[k]["layout"]]["est"])
        n = len(keys)
        cfg = merge_env_cfg({"num_envs": n, "device": dev, "auto_reset": False, "split": "val", "dynamics_checkpoint": args.model, "arena": man["arenas"][aid],
                             "cache": str(cache), "routes": "artifacts/traverse/wp3_routes", "fragment_steps_max": H, "z1_extra_cache": None, "map_key": "map_v2",
                             "termination": {"max_abs_roll_rad": math.radians(60), "max_abs_pitch_rad": math.radians(60)},
                             "action_center": tracker_action_center(Path("artifacts/traverse/wp3_tracker_v1"))})
        env = TraverseTrackingEnv(cfg, device=dev, entries=entries); b = env.bank; ids = torch.arange(n, device=dev); c = env.context
        if args.policy == "openloop":
            ar = {"n": None}
            def policy(obs):
                pp = env._scale_policy_actions(pure_pursuit_actions(env))  # physical: steering from pure pursuit
                e = torch.randn(n, device=dev)
                ar["n"] = e if ar["n"] is None else 0.3 * ar["n"] + math.sqrt(1 - 0.09) * e
                thr = (0.4 + 0.0265 * ar["n"]).clamp(0, 1)
                phys = torch.stack([pp[:, 0], thr, torch.zeros(n, device=dev)], -1)
                return env.physical_to_policy(phys)
        else:
            policy = load_policy(Path("artifacts/traverse/wp3_tracker_v1" if args.policy == "tracker" else args.policy), env, dev)
        # rest start at the camera start pose, as the planner
        env.reset_idx(ids, episode_ids=ids, start_frames=torch.full((n,), c, device=dev, dtype=torch.long), fragment_steps=torch.full((n,), H, device=dev, dtype=torch.long))
        z0 = b.z1[ids, 0]; brake = (torch.tensor([0.0, 0.0, 1.0], device=dev) - env.act_mean) / env.act_std
        env.z1_hist[:] = z0[:, None, :].expand(-1, c, -1); env.act_hist[:] = brake[None, None, :].expand(n, c, -1)
        env.pose[:] = torch.tensor(np.asarray(starts, np.float32), device=dev)
        with torch.no_grad():
            env.token_hist[:] = env.crop(env.env_maps, env.pose[:, None, :].expand(-1, c, -1))
        env.z1_phys[:] = z0 * env.z1_std + env.z1_mean
        env.last_actions[:] = torch.tensor([0.0, 0.0, 1.0], device=dev); env.actions[:] = env.last_actions
        d = (b.route_xy[ids] - env.pose[:, None, :2]).norm(dim=-1)
        valid = torch.arange(b.route_xy.shape[1], device=dev)[None, :] < b.route_len[ids][:, None]
        env.route_idx[:] = torch.where(valid, d, torch.full_like(d, float("inf"))).argmin(dim=1); env.start_station_m[:] = b.route_s[ids, env.route_idx]
        env.episode_length_buf[:] = 0; env.progress_m[:] = 0; env._compute_observations()
        T = H // SUB
        seq = np.zeros((n, T, 17 + 3 + env.model.token_dim + 1), np.float16); active = np.zeros((n, T), bool)
        done = np.zeros(n, bool); completed = np.zeros(n, bool); t_end = np.full(n, H, int)
        for s in range(H):
            with torch.no_grad():
                a = policy(env.obs_buf)
            _, _, dn_t, _ = env.step(a); dn = dn_t.bool().cpu().numpy()
            if s % SUB == 0:
                j = s // SUB
                seq[:, j, :17] = env.z1_phys.cpu().numpy(); seq[:, j, 17:20] = env.pose.cpu().numpy()
                seq[:, j, 20:-1] = env.token_hist[:, -1].cpu().numpy(); seq[:, j, -1] = env.progress_m.cpu().numpy()
                active[:, j] = ~done
            re = env._route_errors()["route_end"].cpu().numpy()
            completed |= (~done) & dn & re; t_end = np.where((~done) & dn, s + 1, t_end); done |= dn
            if done.all():
                break
        np.savez_compressed(out / f"{aid}.npz", keys=np.array(keys), seq=seq, active=active, completed=completed, t_end=t_end, energy=env.energy_kj.cpu().numpy())
        print(f"{aid}: {n} routes, imagined completion {completed.mean():.2f}, mean end {t_end.mean() * 0.05:.1f} s", flush=True)
        del env, policy; torch.cuda.empty_cache()


def cmd_nominal(args) -> None:
    import torch
    from nedm.traverse.nrd_model import load_map_model
    from traverse_wp7_cheap_predictor import profile_features
    dev = args.device
    lab, man = labels_and_classes(args.caches)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tmaps = {a: TerrainMap.from_dir(Path(p)) for a, p in man["arenas"].items()}
    for aid in args.arenas:
        keys = [k for k in man["episodes"] if man["arena_of"][k] == aid]
        if not keys:
            continue
        cache = Path(lab[keys[0]]["cache"])
        model, norm, _ = load_map_model(args.model, man["arenas"][aid], dev)
        hm = torch.tensor(tmaps[aid].height_grid, dtype=torch.float32, device=dev)[None, None]
        prof = np.zeros((len(keys), 120, 7), np.float32); tok = np.zeros((len(keys), 60, model.token_dim), np.float16)
        for i, k in enumerate(keys):
            with np.load(cache / f"{k}.npz") as z:
                w, sp, hd, st, m = z["route_waypoints"], z["route_speeds"], z["route_headings"], z["route_stations"], z["map_v2"]
            prof[i] = profile_features(w, sp, tmaps[aid])
            s = np.arange(0.0, st[-1] + 1e-6, 2.0)[:60]
            x, y = np.interp(s, st, w[:, 0]), np.interp(s, st, w[:, 1]); yaw = np.interp(s, st, np.unwrap(hd))
            pose = torch.tensor(np.stack([x, y, yaw], -1), dtype=torch.float32, device=dev)[None]
            with torch.no_grad():
                t = model.cropper(torch.tensor(m, device=dev).float()[None], pose, hm)[0]
            tok[i, : len(s)] = t.cpu().numpy()
        np.savez_compressed(out / f"{aid}.npz", keys=np.array(keys), prof=prof, tok=tok)
        print(f"{aid}: {len(keys)} routes nominal features", flush=True)


def cmd_train(args) -> None:
    import torch, torch.nn as nn
    from traverse_wp6_imagine_sweep import auc
    dev = args.device
    TRAIN_, VAL_ = args.train_arenas, args.eval_arenas
    lab, man = labels_and_classes(args.caches)
    arms = {}
    for spec in args.arms:
        name, d = spec.split("=", 1); d = Path(d)
        arms[name] = {}
        for aid in TRAIN_ + VAL_:
            if not (d / f"{aid}.npz").exists():
                continue
            z = np.load(d / f"{aid}.npz")
            keys = [str(k) for k in z["keys"]]
            if "seq" in z.files:
                seq, act = z["seq"].astype(np.float32), z["active"]
                X = np.concatenate([seq, act[..., None].astype(np.float32)], -1)
                if name.endswith(":state"):  # ablation: imagined state + pose + progress only, no tokens
                    X = np.concatenate([seq[..., :20], seq[..., -1:], act[..., None].astype(np.float32)], -1)
            else:
                prof, tok = z["prof"], z["tok"].astype(np.float32)
                if name.endswith(":profile"):
                    X = prof
                elif name.endswith(":speed"):
                    X = prof[..., 4:5] * prof[..., 6:7]
                else:  # profile at 1 m + tokens at 2 m, aligned by repeating each token twice
                    tok2 = np.repeat(tok, 2, axis=1)[:, :120]
                    X = np.concatenate([prof, tok2], -1)
            for k, x in zip(keys, X):
                arms[name][k] = x
    results = {}
    val_keys = [k for k in lab if lab[k]["arena"] in VAL_ and k in next(iter(arms.values()))]
    y_stall = np.array([lab[k]["cls"] in ("launch", "stop") for k in val_keys]); y_feas = np.array([lab[k]["feasible"] for k in val_keys])
    y_inf = ~y_feas; lay = np.array([lab[k]["layout"] for k in val_keys]); spd = np.array([lab[k]["mean_speed"] for k in val_keys])
    contact_only = np.array([lab[k]["contact_only"] for k in val_keys])
    print(f"eval {VAL_}: {len(val_keys)} routes, {y_stall.sum()} stall+launch, {y_inf.sum()} infeasible ({contact_only.sum()} contact-only), {y_feas.sum()} feasible; train {TRAIN_}")

    def fit(name, seed):
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        D = arms[name]
        def XY(arenas):
            ks = [k for k in lab if lab[k]["arena"] in arenas and k in D]
            X = np.stack([D[k] for k in ks]); y = np.array([not lab[k]["feasible"] for k in ks], np.float32)
            return ks, X, y
        _, Xtr, _ = XY(TRAIN_); mu, sd = Xtr.reshape(-1, Xtr.shape[-1]).mean(0), Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
        f = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32, device=dev)
        class Head(nn.Module):
            def __init__(s_, d):
                super().__init__(); s_.g = nn.GRU(d, 64, batch_first=True); s_.o = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
            def forward(s_, x):
                h, _ = s_.g(x); return s_.o(h.mean(1))[:, 0]  # mean over time (padding is zeros after standardisation ~ neutral)
        preds_val = []
        for hold in TRAIN_:  # leave-one-arena-out early stopping
            ks_tr, Xa, ya = XY([a for a in TRAIN_ if a != hold]); ks_ho, Xh, yh = XY([hold])
            net = Head(Xa.shape[-1]).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
            Xt, yt, Xht = f(Xa), torch.tensor(ya, device=dev), f(Xh)
            pos_w = torch.tensor([(1 - ya.mean()) / max(ya.mean(), 1e-3)], device=dev)
            best = (-1, None)
            for ep in range(args.epochs):
                perm = torch.randperm(len(Xt), device=dev); net.train()
                for i in range(0, len(Xt), 64):
                    idx = perm[i:i + 64]
                    loss = nn.functional.binary_cross_entropy_with_logits(net(Xt[idx]), yt[idx], pos_weight=pos_w)
                    opt.zero_grad(); loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    a_ = auc(net(Xht).cpu().numpy(), yh > 0.5)
                if a_ > best[0]:
                    best = (a_, {k_: v.clone() for k_, v in net.state_dict().items()})
            net.load_state_dict(best[1]); net.eval()
            with torch.no_grad():
                preds_val.append(net(f(np.stack([D[k] for k in val_keys]))).cpu().numpy())
        return np.mean(preds_val, 0)

    def boot_ci(score, y, n_boot=300, seed=0, lay_sub=None):
        lay_ = lay if lay_sub is None else lay_sub
        rng = np.random.default_rng(seed); L = np.unique(lay_); vals = []
        for _ in range(n_boot):
            pick = rng.choice(L, len(L)); idx = np.concatenate([np.nonzero(lay_ == l)[0] for l in pick])
            if y[idx].any() and (~y[idx]).any():
                vals.append(auc(score[idx], y[idx]))
        return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

    fastest_ok = sum(1 for l in np.unique(lay) if lab[max((k for k in val_keys if lab[k]["layout"] == l), key=lambda k: lab[k]["mean_speed"])]["feasible"])
    n_lay = sum(1 for l in np.unique(lay) if any(lab[k]["feasible"] for k in val_keys if lab[k]["layout"] == l))
    print(f"{'arm':28s} seed  AUC stall|feas [CI]        AUC infeasible [CI]      within-speed  feas.rej@50%inf  P(stall|acc)  gate+fastest (heuristic {fastest_ok}/{n_lay})")
    for name in arms:
        for seed in range(args.seeds):
            sc = fit(name, seed)
            m = y_stall | y_feas
            a1 = auc(sc[m], y_stall[m]); a2 = auc(sc, y_inf); c1 = boot_ci(sc[m], y_stall[m], lay_sub=lay[m]); c2 = boot_ci(sc, y_inf)
            # within commanded-speed bins (1 m/s wide): mean AUC over bins with both classes
            bins = np.round(spd); ws = [auc(sc[(bins == b_) & m], y_stall[(bins == b_) & m]) for b_ in np.unique(bins) if y_stall[(bins == b_) & m].any() and y_feas[(bins == b_)].any()]
            thr = np.sort(sc[y_inf])[len(sc[y_inf]) // 2] if args.threshold is None else float(args.threshold)
            rej_feas = float((sc[y_feas] >= thr).mean()); acc = sc < thr
            p_stall_acc = float(y_stall[acc].mean()) if acc.any() else float("nan")
            n_ok, fixes, breaks, fix_lay = 0, 0, 0, []
            for l in np.unique(lay):
                ks = [i for i, k in enumerate(val_keys) if lab[k]["layout"] == l]
                if not any(lab[val_keys[i]]["feasible"] for i in ks):
                    continue
                accd = [i for i in ks if sc[i] < thr] or ks
                pick = max(accd, key=lambda i: lab[val_keys[i]]["mean_speed"]); fast = max(ks, key=lambda i: lab[val_keys[i]]["mean_speed"])
                n_ok += lab[val_keys[pick]]["feasible"]
                if lab[val_keys[pick]]["feasible"] and not lab[val_keys[fast]]["feasible"]:
                    fixes += 1; fix_lay.append((l, "contact-only" if lab[val_keys[fast]]["contact_only"] else "stall/timeout"))
                if lab[val_keys[fast]]["feasible"] and not lab[val_keys[pick]]["feasible"]:
                    breaks += 1
            results.setdefault(name, []).append({"seed": seed, "scores": sc.round(4).tolist(), "threshold": float(thr), "fixes": fixes, "breaks": breaks, "fix_layouts": fix_lay, "auc_stall": a1, "ci_stall": c1, "auc_inf": a2, "ci_inf": c2, "within_speed": float(np.mean(ws)) if ws else None, "rej_feas": rej_feas, "p_stall_acc": p_stall_acc, "gate_fastest": n_ok})
            print(f"{name:28s} {seed:4d}  {a1:.3f} [{c1[0]:.2f},{c1[1]:.2f}]   {a2:.3f} [{c2[0]:.2f},{c2[1]:.2f}]   {np.mean(ws) if ws else float('nan'):.3f}      {rej_feas:.2f}          {p_stall_acc:.2f}        {n_ok}/{n_lay}  fixes {fixes} ({sum(1 for _, t in fix_lay if t == 'stall/timeout')} stall/timeout) breaks {breaks}", flush=True)
    results["_keys"] = val_keys
    Path(args.out).write_text(json.dumps(results, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump"); d.add_argument("--model", required=True); d.add_argument("--policy", default="tracker"); d.add_argument("--out", required=True); d.add_argument("--device", default="cuda")
    d.add_argument("--caches", nargs="+", default=[str(CACHE)]); d.add_argument("--arenas", nargs="+", default=ARENAS)
    n = sub.add_parser("nominal"); n.add_argument("--model", required=True); n.add_argument("--out", required=True); n.add_argument("--device", default="cuda")
    n.add_argument("--caches", nargs="+", default=[str(CACHE)]); n.add_argument("--arenas", nargs="+", default=ARENAS)
    t = sub.add_parser("train"); t.add_argument("--arms", nargs="+", required=True); t.add_argument("--seeds", type=int, default=3); t.add_argument("--epochs", type=int, default=25)
    t.add_argument("--caches", nargs="+", default=[str(CACHE)]); t.add_argument("--train-arenas", nargs="+", default=TRAIN); t.add_argument("--eval-arenas", nargs="+", default=VAL)
    t.add_argument("--threshold", type=float, default=None, help="fixed gate threshold (pre-registered from the validation arena); default: rejects 50 %% of the eval set's infeasible routes")
    t.add_argument("--out", default="artifacts/traverse/wp8_head/results.json"); t.add_argument("--device", default="cuda")
    args = ap.parse_args()
    {"dump": cmd_dump, "nominal": cmd_nominal, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
