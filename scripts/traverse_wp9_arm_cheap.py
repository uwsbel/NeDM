#!/usr/bin/env python
"""A1 arm: the DIRECT LEARNED COST PREDICTOR -- positive shaft work, time and mission-compliance of a
proposed route-and-speed, from the route profile alone. This is the inexpensive competitor the NRD
imagination has to beat, so it is built to be strong, not to lose.

Differences from ``traverse_wp7_cheap_predictor.py`` (kept intact; this is a separate script):

1. TARGET. It regresses log ``w_pos_kj`` -- positive mechanical shaft work from
   ``artifacts/traverse/wp9_energy/truth.json`` -- not log of the legacy signed energy. Time and the two
   mission gates (Chrono feasibility, full mission compliance) are regressed jointly.
2. LEAVE-ONE-ARENA-OUT. The old script only ever dumped predictions for its own val/test arenas, so an
   all-arena table was impossible. Here every one of the eleven arenas is held out in turn: trained on
   nine, model selected on a tenth, predicted on the eleventh. EVERY prediction in the emitted table is
   out-of-sample for the arena it scores.
3. ENCODER + FEATURES. The old encoder was a single unidirectional GRU whose *last hidden state* was the
   whole route summary, over seven pooled-free channels with no nominal acceleration. Work is an INTEGRAL
   along the route, so this replaces it with a conv stem + 2-layer bidirectional GRU read out by
   mean/max/end pooling, adds nominal acceleration ``a = v dv/ds``, per-station work-rate proxies and
   their running integrals (``traverse_wp9_cheap_features.py``), and gives the work/time heads an
   explicit integral form: a non-negative per-metre increment summed over the route. ``--encoder legacy``
   reproduces the old architecture and channel set for the ablation.

Emits ``arm_cheap.json``: one prediction per (layout, candidate) plus the per-arena held-out fit.

  PYTHONPATH=src python scripts/traverse_wp9_arm_cheap.py \
      --features artifacts/traverse/wp9_energy/cheap_features.npz \
      --truth artifacts/traverse/wp9_energy/truth.json \
      --out artifacts/traverse/wp9_energy/arm_cheap.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

LEGACY_CHANNELS = ["dz", "along", "cross", "kappa_abs", "v_cmd", "rough", "valid"]
LEGACY_GLOBALS = [f"z1_{i}" for i in range(17)] + ["len_m", "v_mean", "v_max"]


# --------------------------------------------------------------------------------------- metrics
def auc(score, label) -> float:
    s, l = np.asarray(score, float), np.asarray(label, int)
    if l.sum() == 0 or l.sum() == len(l):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):
        idx = np.nonzero(s == v)[0]
        ranks[idx] = ranks[idx].mean()
    n1, n0 = l.sum(), (1 - l).sum()
    return float((ranks[l == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a)).astype(float), np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------------------- model
class StrongPredictor(nn.Module):
    """conv stem -> bidirectional GRU -> {mean, max, end} pooling, with integral heads for time/work."""

    def __init__(self, n_ch: int, n_global: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.1, integral: bool = True):
        super().__init__()
        self.integral = integral
        self.stem = nn.Sequential(nn.Conv1d(n_ch, hidden, 5, padding=2), nn.GELU(),
                                  nn.Conv1d(hidden, hidden, 3, padding=1), nn.GELU())
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.glob = nn.Sequential(nn.Linear(n_global, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        d = 6 * hidden + hidden
        self.trunk = nn.Sequential(nn.Linear(d, 2 * hidden), nn.GELU(), nn.Dropout(dropout),
                                   nn.Linear(2 * hidden, hidden), nn.GELU())
        self.cls = nn.Linear(hidden, 2)                     # logit feasible, logit compliant
        if integral:
            self.step_head = nn.Linear(2 * hidden + hidden, 2)   # per-metre log time / log work increment
        self.pool_head = nn.Linear(hidden, 2)               # log time, log work (residual on the integral)
        self.register_buffer("g_mean", torch.zeros(n_global))
        self.register_buffer("g_std", torch.ones(n_global))

    def forward(self, prof, glob, mask):
        h = self.stem(prof.transpose(1, 2)).transpose(1, 2)
        out, _ = self.gru(h)
        m = mask.unsqueeze(-1)
        n = mask.sum(1).clamp(min=1)
        mean = (out * m).sum(1) / n.unsqueeze(-1)
        mx = (out.masked_fill(m == 0, -1e4)).max(1).values
        last = out[torch.arange(len(out), device=out.device), (n.long() - 1)]
        g = self.glob((glob - self.g_mean) / self.g_std)
        z = self.trunk(torch.cat([mean, mx, last, g], -1))
        cls = self.cls(z)
        pool = self.pool_head(z)
        if not self.integral:
            return cls, pool
        gx = g.unsqueeze(1).expand(-1, out.shape[1], -1)
        inc = F.softplus(self.step_head(torch.cat([out, gx], -1)))         # (B, T, 2) >= 0
        tot = (inc * m).sum(1).clamp(min=1e-3)
        return cls, torch.log(tot) + 0.1 * pool                            # integral + small learned residual


class LegacyPredictor(nn.Module):
    """The wp7 architecture: one unidirectional GRU, last valid hidden state, MLP on globals."""

    def __init__(self, n_ch: int, n_global: int, hidden: int = 64, **_):
        super().__init__()
        self.gru = nn.GRU(n_ch, hidden, batch_first=True)
        self.glob = nn.Sequential(nn.Linear(n_global, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 4))
        self.register_buffer("g_mean", torch.zeros(n_global))
        self.register_buffer("g_std", torch.ones(n_global))

    def forward(self, prof, glob, mask):
        out, _ = self.gru(prof)
        n = mask.sum(1).clamp(min=1).long() - 1
        last = out[torch.arange(len(out), device=out.device), n]
        g = self.glob((glob - self.g_mean) / self.g_std)
        o = self.head(torch.cat([last, g], -1))
        return o[:, :2], o[:, 2:]


# --------------------------------------------------------------------------------------- data
def build_arrays(args):
    d = np.load(args.features, allow_pickle=True)
    chans = [str(c) for c in d["profile_channels"]]
    gnames = [str(c) for c in d["global_names"]]
    prof_all, glob_all = d["prof"], d["glob"]
    lay, cand, arena = [np.array([str(x) for x in d[k]]) for k in ("layout", "candidate", "arena")]

    if args.encoder == "legacy":
        keep_c = LEGACY_CHANNELS
        keep_g = LEGACY_GLOBALS
    else:
        keep_c = [c for c in chans if not (args.no_assets and c in {str(x) for x in d["asset_channels"]})]
        keep_g = [g for g in gnames if not (args.no_assets and g in {str(x) for x in d["asset_globals"]})]
    ci = [chans.index(c) for c in keep_c]
    gi = [gnames.index(g) for g in keep_g]
    prof = np.ascontiguousarray(prof_all[:, :, ci])
    glob = np.ascontiguousarray(glob_all[:, gi])
    mask = prof_all[:, :, chans.index("valid")].copy()

    truth = {(r["key"], r["candidate"]): r for r in json.loads(Path(args.truth).read_text())}
    miss = [(l, c) for l, c in zip(lay, cand) if (l, c) not in truth]
    if miss:
        raise SystemExit(f"{len(miss)} cached runs missing from the truth table, e.g. {miss[:3]}")
    rows = [truth[(l, c)] for l, c in zip(lay, cand)]
    tgt = {
        "feasible": np.array([r["feasible"] for r in rows], np.float32),
        "compliant": np.array([r["compliant"] for r in rows], np.float32),
        "time_s": np.array([r["time_s"] for r in rows], np.float64),
        "w_pos_kj": np.array([r["w_pos_kj"] for r in rows], np.float64),
        "deadline_s": np.array([r["deadline_s"] for r in rows], np.float64),
        "route_len_m": np.array([r["route_len_m"] for r in rows], np.float64),
        "profile_time_s": np.array([r["profile_time_s"] for r in rows], np.float64),
        "v_cmd_mean": np.array([r["v_cmd_mean"] for r in rows], np.float64),
        "status": np.array([r["status"] for r in rows]),
    }
    return dict(prof=prof, glob=glob, mask=mask, layout=lay, candidate=cand, arena=arena,
                channels=keep_c, globals=keep_g, tgt=tgt)


def fold_metrics(pred_w, pred_t, p_feas, p_comp, tgt, sel, layout) -> dict:
    """Held-out accuracy on the arena's runs; the primary block is on MISSION-COMPLIANT runs."""
    feas = tgt["feasible"][sel] > 0.5
    comp = tgt["compliant"][sel] > 0.5
    w, t = tgt["w_pos_kj"][sel], tgt["time_s"][sel]
    lay = layout[sel]

    def reg(m):
        if m.sum() < 3:
            return None
        return {
            "n": int(m.sum()),
            "work_mae_kj": float(np.mean(np.abs(pred_w[m] - w[m]))),
            "work_mape": float(np.mean(np.abs(pred_w[m] - w[m]) / w[m])),
            "work_r": pearson(pred_w[m], w[m]),
            "work_log_r": pearson(np.log(pred_w[m]), np.log(w[m])),
            "work_spearman": spearman(pred_w[m], w[m]),
            "time_mae_s": float(np.mean(np.abs(pred_t[m] - t[m]))),
            "time_mape": float(np.mean(np.abs(pred_t[m] - t[m]) / t[m])),
            "time_r": pearson(pred_t[m], t[m]),
            "time_spearman": spearman(pred_t[m], t[m]),
        }

    # within-layout ranking of work among the compliant candidates -- what A1 selection actually needs
    rho, regret, n_lay = [], [], 0
    for L in np.unique(lay):
        m = (lay == L) & comp
        if m.sum() < 3:
            continue
        n_lay += 1
        rho.append(spearman(pred_w[m], w[m]))
        chosen = w[m][np.argmin(pred_w[m])]
        regret.append(float(chosen / w[m].min() - 1.0))
    # ---- the A1 selection the arm would actually make, on the oracle FEASIBLE gate (plan A1: scorers get
    # the Chrono-confirmed feasible subset, but NOT the deadline-compliant subset -- deadline prediction
    # stays their job). Rule, frozen: among feasible candidates take the smallest predicted work among
    # those predicted on time; if none is predicted on time, take the smallest predicted time.
    dl = tgt["deadline_s"][sel]
    sel_ok, sel_reg, sel_late, n_sel = 0, [], 0, 0
    ref_len_reg, ref_len_ok = [], 0
    route_len = tgt["route_len_m"][sel]
    prof_t = tgt["profile_time_s"][sel]
    v_mean = tgt["v_cmd_mean"][sel]
    for L in np.unique(lay):
        m = (lay == L) & feas
        if m.sum() < 2 or not (comp & (lay == L)).any():
            continue
        n_sel += 1
        best_w = w[(lay == L) & comp].min()
        pw, pt, d = pred_w[m], pred_t[m], dl[m]
        ontime = pt <= d
        j = int(np.argmin(np.where(ontime, pw, np.inf))) if ontime.any() else int(np.argmin(pt))
        if comp[m][j]:
            sel_ok += 1
            sel_reg.append(float(w[m][j] / best_w - 1.0))
        else:
            sel_late += 1
        # model-free reference: among feasible candidates whose OWN commanded profile meets the deadline,
        # the shortest route, slowest profile. Uses no fitted model and no Chrono outcome.
        pm = prof_t[m] <= dl[m]
        cost = route_len[m] + 0.01 * v_mean[m]
        jl = int(np.argmin(np.where(pm, cost, np.inf))) if pm.any() else int(np.argmin(prof_t[m]))
        if comp[m][jl]:
            ref_len_ok += 1
            ref_len_reg.append(float(w[m][jl] / best_w - 1.0))
    a1 = {
        "n_layouts": n_sel, "selected_compliant": sel_ok, "selected_noncompliant": sel_late,
        "selected_compliant_rate": (sel_ok / n_sel) if n_sel else None,
        "work_regret_mean": float(np.mean(sel_reg)) if sel_reg else None,
        "work_regret_median": float(np.median(sel_reg)) if sel_reg else None,
        "ref_shortest_ontime_compliant_rate": (ref_len_ok / n_sel) if n_sel else None,
        "ref_shortest_ontime_regret_mean": float(np.mean(ref_len_reg)) if ref_len_reg else None,
    }
    return {
        "n": int(sel.sum()), "n_feasible": int(feas.sum()), "n_compliant": int(comp.sum()),
        "a1_selection": a1,
        "auc_feasible": auc(p_feas, feas), "auc_compliant": auc(p_comp, comp),
        "brier_feasible": float(np.mean((p_feas - feas) ** 2)),
        "on_compliant": reg(comp), "on_feasible": reg(feas),
        "within_layout_work_spearman": float(np.nanmean(rho)) if rho else None,
        "within_layout_work_regret_mean": float(np.mean(regret)) if regret else None,
        "within_layout_work_regret_median": float(np.median(regret)) if regret else None,
        "n_layouts_scored": n_lay,
    }


def train_fold(data, train_i, val_i, test_i, args, dev) -> tuple[np.ndarray, dict]:
    prof, glob, mask, tgt = data["prof"], data["glob"], data["mask"], data["tgt"]
    ch = data["channels"]
    valid_col = ch.index("valid") if "valid" in ch else None

    # standardize profile channels on the TRAINING fold, over valid stations only
    mtr = mask[train_i] > 0.5
    ptr = prof[train_i]
    mu = np.zeros(prof.shape[2], np.float32)
    sd = np.ones(prof.shape[2], np.float32)
    for c in range(prof.shape[2]):
        if c == valid_col:
            continue
        v = ptr[:, :, c][mtr]
        mu[c], sd[c] = v.mean(), max(v.std(), 1e-3)
    profn = ((prof - mu) / sd).astype(np.float32) * mask[:, :, None]
    if valid_col is not None:
        profn[:, :, valid_col] = mask

    P = torch.tensor(profn, device=dev)
    G = torch.tensor(glob, device=dev)
    M = torch.tensor(mask, device=dev)
    y_f = torch.tensor(tgt["feasible"], device=dev)
    y_c = torch.tensor(tgt["compliant"], device=dev)
    y_t = torch.tensor(np.log(np.maximum(tgt["time_s"], 1e-2)), dtype=torch.float32, device=dev)
    y_w = torch.tensor(np.log(np.maximum(tgt["w_pos_kj"], 1.0)), dtype=torch.float32, device=dev)
    tr = torch.tensor(np.nonzero(train_i)[0], device=dev)
    va = torch.tensor(np.nonzero(val_i)[0], device=dev)

    # within-layout pairs: A1 selection is a comparison BETWEEN candidates of one layout, and almost all
    # of the marginal variance in the pooled regression is BETWEEN layouts. Sampling the batch as layout
    # pairs lets a differenced loss put capacity on the comparison that actually decides the route.
    lay_tr: dict[str, list[int]] = {}
    for i in np.nonzero(train_i)[0]:
        lay_tr.setdefault(str(data["layout"][i]), []).append(int(i))
    pair_lays = [L for L, v in lay_tr.items() if len(v) >= 2]
    pair_sz = np.array([len(lay_tr[L]) for L in pair_lays], np.int64)
    pair_mat = np.zeros((len(pair_lays), int(pair_sz.max()) if len(pair_lays) else 1), np.int64)
    for r, L in enumerate(pair_lays):
        pair_mat[r, : len(lay_tr[L])] = lay_tr[L]
    pair_p = pair_sz.astype(np.float64) / max(pair_sz.sum(), 1)

    def sample_pairs(rng_, h):
        """(h,) layouts sampled proportional to bank size -> two distinct candidate indices each."""
        L = rng_.choice(len(pair_sz), size=h, p=pair_p)
        n = pair_sz[L]
        a = (rng_.random(h) * n).astype(np.int64)
        b = (a + 1 + (rng_.random(h) * (n - 1)).astype(np.int64)) % n
        return pair_mat[L, a], pair_mat[L, b]

    # a fixed val-fold pair set, so checkpoint selection sees the ranking objective too
    lay_va: dict[str, list[int]] = {}
    for i in np.nonzero(val_i)[0]:
        lay_va.setdefault(str(data["layout"][i]), []).append(int(i))
    vrng = np.random.default_rng(12345)
    va_a, va_b = [], []
    for L, v in lay_va.items():
        if len(v) < 2:
            continue
        for _ in range(min(6, len(v))):
            i, j = vrng.choice(len(v), size=2, replace=False)
            va_a.append(v[i]); va_b.append(v[j])
    va_pair = torch.tensor(np.concatenate([np.array(va_a, np.int64), np.array(va_b, np.int64)]), device=dev) if va_a else None

    def loss_on(idx, model, pair=False):
        cls, reg = model(P[idx], G[idx], M[idx])
        lf = F.binary_cross_entropy_with_logits(cls[:, 0], y_f[idx])
        lc = F.binary_cross_entropy_with_logits(cls[:, 1], y_c[idx])
        m = y_f[idx] > 0.5                                  # time/work targets only where the run finished
        if m.any():
            lt = F.huber_loss(reg[m, 0], y_t[idx][m], delta=0.5)
            lw = F.huber_loss(reg[m, 1], y_w[idx][m], delta=0.5)
        else:
            lt = lw = reg.sum() * 0
        total = lf + lc + args.w_time * lt + args.w_work * lw
        lp = torch.zeros((), device=reg.device)
        if pair and args.w_pair > 0:
            h = len(idx) // 2
            both = m[:h] & m[h:]
            if both.any():
                dp = reg[:h][both] - reg[h:][both]
                dy = torch.stack([y_t[idx][:h][both] - y_t[idx][h:][both],
                                  y_w[idx][:h][both] - y_w[idx][h:][both]], -1)
                lp = F.huber_loss(dp, dy, delta=0.3)
                total = total + args.w_pair * lp
        return total, (float(lf.detach()), float(lc.detach()), float(lt.detach()), float(lw.detach()), float(lp.detach()))

    ens_pred, ens_log = [], []
    for seed in range(args.seeds):
        torch.manual_seed(1000 * seed + args.seed)
        Model = LegacyPredictor if args.encoder == "legacy" else StrongPredictor
        model = Model(prof.shape[2], glob.shape[1], hidden=args.hidden, layers=args.layers,
                      dropout=args.dropout, integral=not args.no_integral).to(dev)
        with torch.no_grad():
            model.g_mean.copy_(G[tr].mean(0))
            model.g_std.copy_(G[tr].std(0).clamp_min(1e-3))
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
        rng = np.random.default_rng(1000 * seed + args.seed)
        best, best_state, log = float("inf"), None, []
        for step in range(args.steps):
            if args.w_pair > 0 and len(pair_sz):
                a, b = sample_pairs(rng, args.batch // 2)
                idx = torch.tensor(np.concatenate([a, b]), device=dev)
            else:
                idx = tr[torch.tensor(rng.integers(0, len(tr), args.batch), device=dev)]
            loss, _ = loss_on(idx, model, pair=True)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if (step + 1) % args.eval_every == 0:
                model.eval()
                with torch.no_grad():
                    vl, parts = loss_on(va, model)
                    if va_pair is not None and args.w_pair > 0:
                        _, pp = loss_on(va_pair, model, pair=True)
                        vl = vl + args.w_pair * pp[4]
                model.train()
                log.append({"step": step + 1, "val": float(vl), "parts": parts})
                if float(vl) < best:
                    best = float(vl)
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            cls, reg = model(P, G, M)
        ens_pred.append(np.concatenate([torch.sigmoid(cls).cpu().numpy(), reg.cpu().numpy()], 1))
        ens_log.append({"best_val": best, "curve": log[-1] if log else None})
    pred = np.mean(ens_pred, 0)
    return pred, {"best_val": float(np.mean([e["best_val"] for e in ens_log])), "seeds": args.seeds}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="artifacts/traverse/wp9_energy/cheap_features.npz")
    ap.add_argument("--truth", default="artifacts/traverse/wp9_energy/truth.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder", choices=["strong", "legacy"], default="strong")
    ap.add_argument("--no-integral", action="store_true", help="pooled regression head instead of the per-metre integral")
    ap.add_argument("--no-assets", action="store_true", help="drop the route-to-asset clearance channels")
    ap.add_argument("--arenas", nargs="*", default=None, help="restrict the leave-one-out loop (debug)")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--w-time", type=float, default=1.0)
    ap.add_argument("--w-work", type=float, default=1.0)
    ap.add_argument("--w-pair", type=float, default=1.0, help="within-layout differenced time/work loss (0 disables layout-paired batching)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    data = build_arrays(args)
    arena = data["arena"]
    all_arenas = sorted(set(arena.tolist()))
    loop = args.arenas or all_arenas
    print(f"{len(arena)} runs, {len(all_arenas)} arenas, {len(data['channels'])} channels, "
          f"{len(data['globals'])} globals, device {dev}", flush=True)

    predictions: dict[str, dict] = {}
    per_arena, folds = {}, {}
    for held in loop:
        # val arena: the next arena in the cycle, never the held-out one -- fixed, outcome-blind
        others = [a for a in all_arenas if a != held]
        val_a = others[(all_arenas.index(held)) % len(others)]
        test_i = arena == held
        val_i = arena == val_a
        train_i = ~(test_i | val_i)
        tf = time.time()
        pred, info = train_fold(data, train_i, val_i, test_i, args, dev)
        p_feas, p_comp = pred[:, 0], pred[:, 1]
        pred_t, pred_w = np.exp(pred[:, 2]), np.exp(pred[:, 3])
        m = fold_metrics(pred_w[test_i], pred_t[test_i], p_feas[test_i], p_comp[test_i], data["tgt"], test_i, data["layout"])
        m["val_arena"] = val_a
        m["n_train"] = int(train_i.sum())
        m["fit_s"] = round(time.time() - tf, 1)
        per_arena[held] = m
        folds[held] = {"held_out": held, "val_arena": val_a,
                       "train_arenas": [a for a in all_arenas if a not in (held, val_a)], **info}
        for i in np.nonzero(test_i)[0]:
            predictions[f"{data['layout'][i]}|{data['candidate'][i]}"] = {
                "pred_w_pos_kj": float(pred_w[i]), "pred_time_s": float(pred_t[i]),
                "p_feasible": float(p_feas[i]), "p_compliant": float(p_comp[i]),
                "arena": str(arena[i]), "held_out_from": held,
            }
        oc = m["on_compliant"] or {}
        print(f"[{held}] val={val_a} n={m['n']} AUCfeas={m['auc_feasible']:.3f} AUCcomp={m['auc_compliant']:.3f} "
              f"| compliant n={oc.get('n')} workMAE={oc.get('work_mae_kj', float('nan')):.1f}kJ "
              f"r={oc.get('work_r', float('nan')):.3f} timeMAE={oc.get('time_mae_s', float('nan')):.2f}s "
              f"r={oc.get('time_r', float('nan')):.3f} rho_layout={m['within_layout_work_spearman']} "
              f"({m['fit_s']}s)", flush=True)

    def agg(path, key):
        vals = [per_arena[a][path][key] for a in loop if per_arena[a].get(path)]
        return float(np.nanmean(vals)) if vals else None

    overall = {
        "n_predictions": len(predictions),
        "arenas_predicted": loop,
        "all_predictions_out_of_sample": True,
        "mean_auc_feasible": float(np.nanmean([per_arena[a]["auc_feasible"] for a in loop])),
        "mean_auc_compliant": float(np.nanmean([per_arena[a]["auc_compliant"] for a in loop])),
        "compliant_work_mae_kj": agg("on_compliant", "work_mae_kj"),
        "compliant_work_mape": agg("on_compliant", "work_mape"),
        "compliant_work_r": agg("on_compliant", "work_r"),
        "compliant_work_spearman": agg("on_compliant", "work_spearman"),
        "compliant_time_mae_s": agg("on_compliant", "time_mae_s"),
        "compliant_time_mape": agg("on_compliant", "time_mape"),
        "compliant_time_r": agg("on_compliant", "time_r"),
        "within_layout_work_spearman": float(np.nanmean([per_arena[a]["within_layout_work_spearman"]
                                                         for a in loop if per_arena[a]["within_layout_work_spearman"] is not None])),
        "within_layout_work_regret_mean": float(np.nanmean([per_arena[a]["within_layout_work_regret_mean"]
                                                            for a in loop if per_arena[a]["within_layout_work_regret_mean"] is not None])),
    }
    A = [per_arena[a]["a1_selection"] for a in loop]
    overall["a1_selection"] = {
        "n_layouts": int(sum(x["n_layouts"] for x in A)),
        "selected_compliant": int(sum(x["selected_compliant"] for x in A)),
        "selected_noncompliant": int(sum(x["selected_noncompliant"] for x in A)),
        "selected_compliant_rate": float(sum(x["selected_compliant"] for x in A) / max(sum(x["n_layouts"] for x in A), 1)),
        "work_regret_mean_over_arenas": float(np.nanmean([x["work_regret_mean"] for x in A if x["work_regret_mean"] is not None])),
        "ref_shortest_ontime_compliant_rate": float(sum(x["ref_shortest_ontime_compliant_rate"] * x["n_layouts"] for x in A) / max(sum(x["n_layouts"] for x in A), 1)),
        "ref_shortest_ontime_regret_mean_over_arenas": float(np.nanmean([x["ref_shortest_ontime_regret_mean"] for x in A if x["ref_shortest_ontime_regret_mean"] is not None])),
    }
    out = {
        "predictions": {k: {kk: vv for kk, vv in v.items()} for k, v in predictions.items()},
        "fit": {
            "target": "log w_pos_kj (positive shaft work) from artifacts/traverse/wp9_energy/truth.json",
            "protocol": "leave-one-arena-out; train on 9 arenas, select on a 10th, predict the 11th",
            "encoder": args.encoder, "integral_head": not args.no_integral,
            "channels": data["channels"], "globals": data["globals"],
            "hyperparams": {k: getattr(args, k) for k in
                            ("steps", "batch", "lr", "hidden", "layers", "dropout", "weight_decay", "w_time", "w_work", "w_pair", "seeds", "seed")},
            "folds": folds, "per_arena": per_arena, "overall": overall,
            "wall_s": round(time.time() - t0, 1),
        },
    }
    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out))
    print(json.dumps(overall, indent=1))
    print(f"wrote {op} ({len(predictions)} predictions, {time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
