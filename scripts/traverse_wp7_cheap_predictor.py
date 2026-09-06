#!/usr/bin/env python
"""Cheap crossing predictor (plan §28 step 2): success / time / energy of a proposed route-and-speed from the terrain
profile along it, the start state and the speed profile -- the matched control for the fine-tuned world model.

Inputs per episode (from the schema-v2 cache):
  * the route, resampled at 1 m: height relative to the start, along- and cross-slope, curvature, commanded speed,
    local roughness -- read from a height field (``--terrain true``: the arena heightmap, the same prior the
    world model's crop projection reads; ``predicted``: the map head's elevation decoded from the same camera frame);
  * the start state (17-D rest state, normalised) and route length.
Targets: feasible (completed, no stall, no contact), log time, log energy (the last two on feasible runs only).
Model: a small GRU over the profile + MLP on the globals; trained on the train arenas, selected on the val arena.
Reports val AUC of infeasibility, time / energy errors on feasible runs, per layout kind; writes predictions.

  PYTHONPATH=src python scripts/traverse_wp7_cheap_predictor.py --caches artifacts/traverse/wp7_cache_v1 \
      --val-arenas arena_f105 --out artifacts/traverse/wp7_cheap_v1
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse.terrain import TerrainMap

STEP_M = 1.0
MAX_STEPS = 120
N_PROFILE = 7  # dz, along slope, cross slope, |kappa|, speed, roughness, valid


def profile_features(pts: np.ndarray, speeds: np.ndarray, tmap) -> np.ndarray:
    """(N, 2) route, (N,) commanded speeds -> (MAX_STEPS, N_PROFILE) profile at 1 m stations (zero-padded, last col = valid)."""
    seg = np.hypot(*np.diff(pts, axis=0).T); station = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, station[-1] + 1e-6, STEP_M)[:MAX_STEPS]
    x, y = np.interp(s, station, pts[:, 0]), np.interp(s, station, pts[:, 1])
    v = np.interp(s, station, speeds)
    tx, ty = np.gradient(x), np.gradient(y); tn = np.hypot(tx, ty) + 1e-9; tx, ty = tx / tn, ty / tn
    h = tmap.height(x, y); gx, gy = tmap.gradient(x, y)
    along = gx * tx + gy * ty; cross = -gx * ty + gy * tx
    kappa = np.abs(np.gradient(np.unwrap(np.arctan2(ty, tx)))) / STEP_M
    offs = [(1.5, 0.0), (-1.5, 0.0), (0.0, 1.5), (0.0, -1.5)]
    hh = np.stack([tmap.height(x + a * tx - b * ty, y + a * ty + b * tx) for a, b in offs] + [h])
    rough = hh.std(0)
    f = np.zeros((MAX_STEPS, N_PROFILE), np.float32)
    n = len(s)
    f[:n, 0] = (h - h[0]) / 2.0; f[:n, 1] = np.clip(along, -0.8, 0.8); f[:n, 2] = np.clip(cross, -0.8, 0.8)
    f[:n, 3] = np.clip(kappa, 0, 0.3) * 5.0; f[:n, 4] = v / 10.0; f[:n, 5] = rough / 0.3; f[:n, 6] = 1.0
    return f


class CheapPredictor(nn.Module):
    def __init__(self, n_global: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(N_PROFILE, hidden, batch_first=True)
        self.glob = nn.Sequential(nn.Linear(n_global, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))  # logit feasible, log t, log E
        self.register_buffer("g_mean", torch.zeros(n_global)); self.register_buffer("g_std", torch.ones(n_global))

    def forward(self, prof: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(prof)
        n = prof[..., -1].sum(1).long().clamp(min=1) - 1  # last valid step
        last = out[torch.arange(len(out), device=out.device), n]
        g = self.glob((glob - self.g_mean) / self.g_std)
        return self.head(torch.cat([last, g], -1))


def global_features(z1_0: np.ndarray, length_m: float, speeds: np.ndarray) -> np.ndarray:
    return np.concatenate([z1_0, [length_m / 50.0, speeds.mean() / 10.0, speeds.max() / 10.0]]).astype(np.float32)


def auc(score, label):
    s, l = np.asarray(score, float), np.asarray(label, int)
    if l.sum() == 0 or l.sum() == len(l):
        return float("nan")
    order = np.argsort(s); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        idx = np.nonzero(s == v)[0]; ranks[idx] = ranks[idx].mean()
    n1, n0 = l.sum(), (1 - l).sum()
    return float((ranks[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_dataset(caches: list[Path], terrain: str, maphead: str | None, device: str):
    manifest, labels = {"episodes": [], "arenas": {}}, {}
    for cache in caches:
        m = json.loads((cache / "cache_manifest.json").read_text())
        manifest["episodes"] += [(cache, k) for k in m["episodes"]]; manifest["arenas"].update(m["arenas"])
        labels.update(json.loads((cache / "labels.json").read_text()))
    tmaps = {a: TerrainMap.from_dir(Path(d)) for a, d in manifest["arenas"].items()}
    decoder = None
    if terrain == "predicted":
        from nedm.traverse.planner_b import MapDecoder
        decoder = MapDecoder(Path(maphead), Path(manifest["arenas"][next(iter(manifest["arenas"]))]), device)
    rows = []
    pred_tmaps: dict[str, object] = {}
    for cache, key in manifest["episodes"]:
        with np.load(cache / f"{key}.npz") as z:
            pts, sp = z["route_waypoints"], z["route_speeds"]
            z1_0 = z["z1"][0]
            layout = str(z["layout"]); aid = str(z["arena"])
            if decoder is not None and layout not in pred_tmaps:
                _, elev = decoder(z["map_v2"]); pred_tmaps[layout] = decoder.terrain(elev)
        tm = pred_tmaps[layout] if decoder is not None else tmaps[aid]
        lab = labels[key]
        feasible = bool(lab.get("completed")) and not bool(lab.get("stalled")) and not bool(lab.get("contact"))
        rows.append({"key": key, "arena": aid, "kind": lab.get("kind", "unknown"), "layout": layout,
                     "prof": profile_features(pts, sp, tm), "glob": global_features(z1_0, float(lab.get("length_m") or 0.0), sp),
                     "feasible": feasible, "time_s": float(lab.get("time_s") or np.nan), "energy_kj": float(lab.get("energy_kj") or np.nan),
                     "status": lab.get("status")})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", nargs="+", required=True, help="schema-v2 caches (e.g. the training cache and the sealed-arena cache)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-arenas", nargs="+", required=True)
    ap.add_argument("--test-arenas", nargs="*", default=[])
    ap.add_argument("--terrain", choices=["true", "predicted"], default="true")
    ap.add_argument("--maphead", default="artifacts/traverse/wp4_maphead_v2/ckpt_best.pt")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = load_dataset([Path(c) for c in args.caches], args.terrain, args.maphead, dev)
    train = [r for r in rows if r["arena"] not in args.val_arenas and r["arena"] not in args.test_arenas]
    val = [r for r in rows if r["arena"] in args.val_arenas]
    print(f"{len(rows)} episodes ({time.time() - t0:.0f}s): train {len(train)} feasible {np.mean([r['feasible'] for r in train]):.2f} | "
          f"val {len(val)} feasible {np.mean([r['feasible'] for r in val]):.2f}", flush=True)

    def tensors(rs):
        prof = torch.tensor(np.stack([r["prof"] for r in rs]), device=dev)
        glob = torch.tensor(np.stack([r["glob"] for r in rs]), device=dev)
        feas = torch.tensor([r["feasible"] for r in rs], dtype=torch.float32, device=dev)
        lt = torch.tensor([math.log(r["time_s"]) if r["feasible"] else 0.0 for r in rs], device=dev)
        le = torch.tensor([math.log(max(r["energy_kj"], 1.0)) if r["feasible"] else 0.0 for r in rs], device=dev)
        return prof, glob, feas, lt, le

    tr, va = tensors(train), tensors(val)
    model = CheapPredictor(tr[1].shape[1], args.hidden).to(dev)
    model.g_mean.copy_(tr[1].mean(0)); model.g_std.copy_(tr[1].std(0).clamp_min(1e-3))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def loss_fn(t, out):
        prof, glob, feas, lt, le = t
        l_f = F.binary_cross_entropy_with_logits(out[:, 0], feas)
        m = feas > 0.5
        l_t = F.huber_loss(out[m, 1], lt[m], delta=0.5) if m.any() else out.sum() * 0
        l_e = F.huber_loss(out[m, 2], le[m], delta=0.5) if m.any() else out.sum() * 0
        return l_f + l_t + l_e, (float(l_f), float(l_t), float(l_e))

    rng = np.random.default_rng(args.seed)
    best, log = (float("inf"), None), []
    for step in range(args.steps):
        idx = torch.tensor(rng.integers(0, len(train), args.batch), device=dev)
        batch = tuple(x[idx] for x in tr)
        loss, parts = loss_fn(batch, model(batch[0], batch[1]))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if (step + 1) % 200 == 0:
            model.eval()
            with torch.no_grad():
                vl, vparts = loss_fn(va, model(va[0], va[1]))
            model.train()
            log.append({"step": step + 1, "train": float(loss), "val": float(vl), "val_parts": vparts})
            if float(vl) < best[0]:
                best = (float(vl), step + 1); torch.save({"model": model.state_dict(), "n_global": tr[1].shape[1], "hidden": args.hidden, "terrain": args.terrain, "step": step + 1}, out / "ckpt_best.pt")
            print(f"step {step + 1}: train {float(loss):.3f} val {float(vl):.3f} (feas {vparts[0]:.3f} t {vparts[1]:.3f} E {vparts[2]:.3f})", flush=True)
    model.load_state_dict(torch.load(out / "ckpt_best.pt", map_location=dev)["model"]); model.eval()
    with torch.no_grad():
        o = model(va[0], va[1]).cpu().numpy()
    p_feas = 1 / (1 + np.exp(-o[:, 0])); t_pred, e_pred = np.exp(o[:, 1]), np.exp(o[:, 2])
    feas = np.array([r["feasible"] for r in val]); infeasible = ~feas
    res = {"n_val": len(val), "best_step": best[1], "val_loss": best[0],
           "auc_infeasible": auc(-p_feas, infeasible), "infeasible_rate": float(infeasible.mean()),
           "precision_at_reject_20pct": float(infeasible[np.argsort(p_feas)[: max(1, len(val) // 5)]].mean()),
           "time_mape_feasible": float(np.mean(np.abs(t_pred[feas] - np.array([r["time_s"] for r in val])[feas]) / np.array([r["time_s"] for r in val])[feas])) if feas.any() else None,
           "energy_mape_feasible": float(np.mean(np.abs(e_pred[feas] - np.array([r["energy_kj"] for r in val])[feas]) / np.array([r["energy_kj"] for r in val])[feas])) if feas.any() else None,
           "energy_ratio_feasible": float(np.array([r["energy_kj"] for r in val])[feas].mean() / e_pred[feas].mean()) if feas.any() else None,
           "by_kind": {}}
    for kind in sorted(set(r["kind"] for r in val)):
        m = np.array([r["kind"] == kind for r in val])
        res["by_kind"][kind] = {"n": int(m.sum()), "infeasible_rate": float(infeasible[m].mean()), "auc_infeasible": auc(-p_feas[m], infeasible[m])}
    # baseline: feasibility by commanded speed alone (does the profile add anything?)
    vmax = np.array([r["glob"][-1] for r in val])
    res["auc_infeasible_speed_only"] = auc(vmax, infeasible)
    (out / "readout.json").write_text(json.dumps(res, indent=1)); (out / "train_log.json").write_text(json.dumps(log))
    def dump(rs, name):
        if not rs:
            return
        t = tensors(rs)
        with torch.no_grad():
            oo = model(t[0], t[1]).cpu().numpy()
        (out / name).write_text(json.dumps([{"key": r["key"], "arena": r["arena"], "layout": r["layout"], "p_feasible": float(1 / (1 + np.exp(-o_[0]))), "time_pred": float(np.exp(o_[1])),
                                             "energy_pred": float(np.exp(o_[2])), "feasible": bool(r["feasible"]), "time_s": r["time_s"], "energy_kj": r["energy_kj"],
                                             "status": r["status"], "kind": r["kind"]} for r, o_ in zip(rs, oo)], indent=0))
    dump(val, "val_predictions.json")
    dump([r for r in rows if r["arena"] in args.test_arenas], "test_predictions.json")  # sealed arenas: predictions only, never looked at during selection
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
