#!/usr/bin/env python
"""Analytic energy baseline for the night-2 energy head (artifacts/traverse/crm_night2_v1, Study 1 stage B).

Refits the 09-07 A1 analytic work model (scripts/traverse_wp9_analytic.py: per-route physics terms from the route
waypoints, the commanded speed profile and the arena heightmap -- positive tractive work and its square, low-/high-speed
variants, profile duration, cornering, cross-slope, ... -- fitted by non-negative least squares on the level plus an
alpha-weighted copy of the design centred within each route group) on the f104 twin-matched datasets, one fit per world,
and evaluates it next to two trivial baselines on the held-out goal-reached routes with the read-outs the learned energy
head is judged on: log-RMSE of route energy, median absolute percentage error, within-group Spearman and the
optimiser's-curse ratio (pred/true at the route with the lowest predicted energy over the group's mean pred/true).

Target E_last = first-arrival cumulative W+ (kJ) at the furthest station with a value (the goal circle is 2.5 m, so the
last observed station is typically 86-91 of 95 and E_last = the episode total W+ for 90 % of goal-reached rows); a
variant fits the episode total W+. Fit rows: split == 'train' & fail == 0 (goal reached; unsafe-but-reached rows are
included, a clean-only fit is reported as a variant). The md5 dev fold (int(md5(group), 16) % 5 == 0) inside the training
groups chooses the term pool and alpha by the original script's declared score (0.5 MAE/30 kJ + 0.5 within-group
selection regret/20 kJ); the final fit uses every training row. Held-out rows: split != 'train' & fail == 0.
Rigid and CRM twins share identical waypoints and speed profiles (checked on every 50th id), so the physics terms are
extracted once per route id and cached in route_terms.npz.

Differences from the original's inputs, handled here: (i) the f104 profiles start at the full commanded speed (no launch
ramp) and taper to zero at the goal, so the original's ``tract`` term does not see the standing-start kinetic energy --
a ``tract_launch`` pool (tract pool + ``accel`` + ``kemax``) is offered alongside the original pools; (ii) the original
``resample`` drops the last partial station (< 0.5 m, in the v -> 0 taper) -- kept as is, negligible; (iii) the original
selected by leave-one-arena-out over four arenas -- here there is one arena, so the dev fold does that job.

  PYTHONPATH=src python scripts/n2_energy_analytic.py [--out artifacts/traverse/crm_night2_v1/energy_baselines]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nedm.traverse.terrain import TerrainMap  # noqa: E402


def _load_wp9():
    spec = importlib.util.spec_from_file_location("traverse_wp9_analytic", REPO / "scripts" / "traverse_wp9_analytic.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


WP9 = _load_wp9()
NAMES: list[str] = list(WP9.TERMS) + list(WP9.EXTRA)
POOLS: dict[str, list[str]] = {
    "tract": list(WP9.POOLS["tract"]),
    "tract_launch": list(WP9.POOLS["tract"]) + ["accel", "kemax"],
    "separated": list(WP9.POOLS["separated"]),
    "full": list(WP9.POOLS["full"]),
}
ALPHA_GRID: list[float] = list(WP9.ALPHA_GRID)
K = REPO / "artifacts/traverse/crm_night2_v1"
RUN_DIRS = {
    "crm": [REPO / "artifacts/traverse/crm_f104_v1/collect_v1/runs"],
    "rigid": [REPO / "artifacts/traverse/fdm_f104_50h_20260909/production_v3/runs",
              REPO / "artifacts/traverse/fdm_f104_50h_20260909/production_v4/runs"],
}
ARENA = REPO / "assets/traverse/arena_f104_50h_v1"
MIN_GROUP = 3
WORLDS = ("rigid", "crm")


def dev_fold(group: str) -> bool:
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % 5 == 0


# ---------------------------------------------------------------- terms (once per route id)


def find_ref(world: str, rid: str) -> Path | None:
    for d in RUN_DIRS[world]:
        p = d / rid / "command_reference.npz"
        if p.exists():
            return p
    return None


def extract_terms(ids: np.ndarray, tmap: TerrainMap, cache: Path, rebuild: bool, check_every: int = 50) -> np.ndarray:
    if cache.exists() and not rebuild:
        with np.load(cache, allow_pickle=True) as z:
            if [str(n) for n in z["names"]] == NAMES and len(z["id"]) == len(ids) and (z["id"].astype(str) == ids).all():
                print(f"terms cache: {len(ids)} routes from {cache}")
                return np.asarray(z["X"], np.float64)
    X = np.full((len(ids), len(NAMES)), np.nan)
    t0, n_check = time.time(), 0
    for i, rid in enumerate(ids):
        pc, pr = find_ref("crm", rid), find_ref("rigid", rid)
        src = pc or pr
        if src is None:
            raise FileNotFoundError(f"no command_reference.npz for {rid}")
        with np.load(src) as z:
            wp, spd = np.asarray(z["reference_waypoints"], np.float64), np.asarray(z["reference_speeds"], np.float64)
        if pc is not None and pr is not None and i % check_every == 0:   # twins must share the geometry
            with np.load(pr) as z:
                if not (np.allclose(wp, z["reference_waypoints"]) and np.allclose(spd, z["reference_speeds"])):
                    raise RuntimeError(f"rigid/CRM twins differ in route geometry: {rid}")
            n_check += 1
        row = WP9.route_terms(WP9.resample(wp, spd, tmap))
        X[i] = [row[n] for n in NAMES]
        if (i + 1) % 5000 == 0:
            print(f"  terms {i + 1}/{len(ids)} ({time.time() - t0:.0f} s)", flush=True)
    assert np.isfinite(X).all()
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, id=ids, names=np.array(NAMES), X=X)
    print(f"wrote {cache}: {len(ids)} routes, twin geometry checked on {n_check} ids ({time.time() - t0:.0f} s)")
    return X


# ---------------------------------------------------------------- fits (the original's NNLS machinery)


def make_terms(ids: np.ndarray, groups: np.ndarray, X: np.ndarray) -> dict:
    return {(str(g), str(i)): {n: float(X[k, j]) for j, n in enumerate(NAMES)}
            for k, (g, i) in enumerate(zip(groups, ids))}


def make_rows(idx: np.ndarray, ids: np.ndarray, groups: np.ndarray, y: np.ndarray) -> list[dict]:
    return [{"key": str(groups[i]), "candidate": str(ids[i]), "w_pos_kj": float(y[i])} for i in idx]


def predict(rows: list[dict], terms: dict, names: list[str], w: np.ndarray) -> np.ndarray:
    return np.maximum(WP9.design(rows, terms, names) @ w, 0.0)


def fit_len(L: np.ndarray, y: np.ndarray) -> float:
    """kJ/m x route length: least squares through the origin."""
    return float((L * y).sum() / (L * L).sum())


def fit_affine(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """a + b x t: ordinary least squares."""
    A = np.stack([np.ones_like(t), t], 1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a), float(b)


# ---------------------------------------------------------------- metrics


def metrics(pred: np.ndarray, true: np.ndarray, groups: np.ndarray, min_n: int = MIN_GROUP) -> dict:
    pred = np.maximum(np.asarray(pred, np.float64), 0.0)
    true = np.asarray(true, np.float64)
    out: dict = {"n": int(len(true))}
    if len(true) < 3:
        return out
    res = pred - true
    ss = float(((true - true.mean()) ** 2).sum())
    out.update({
        "log_rmse": float(np.sqrt(np.mean((np.log1p(pred) - np.log1p(true)) ** 2))),
        "mdape_pct": float(100.0 * np.median(np.abs(res) / true)),
        "mape_pct": float(100.0 * np.mean(np.abs(res) / true)),
        "mae_kj": float(np.abs(res).mean()), "bias_kj": float(res.mean()),
        "r2": float(1.0 - float((res ** 2).sum()) / ss) if ss > 0 else float("nan"),
        "spearman_pooled": WP9.spearman(pred, true), "true_median_kj": float(np.median(true)),
    })
    g: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(groups):
        g[str(k)].append(i)
    rho, r_arg, r_mean, reg, top = [], [], [], [], []
    for ii in g.values():
        if len(ii) < min_n:
            continue
        ii = np.array(ii)
        r = pred[ii] / true[ii]
        j = int(np.argmin(pred[ii]))
        rho.append(WP9.spearman(pred[ii], true[ii]))
        r_arg.append(float(r[j]))
        r_mean.append(float(r.mean()))
        reg.append(float(true[ii][j] - true[ii].min()))
        top.append(bool(true[ii][j] == true[ii].min()))
    out["n_groups"] = len(rho)
    if rho:
        ra, rm = np.array(r_arg), np.array(r_mean)
        out.update({
            "spearman_within_group": float(np.nanmean(rho)),
            "curse_ratio": float(np.mean(ra / rm)),          # < 1: the pick looks cheaper than the group average does
            "curse_diff": float(np.mean(ra - rm)),           # the 09-07 form (pred/true at the pick minus over all)
            "pred_over_true_at_argmin": float(np.mean(ra)), "pred_over_true_mean": float(np.mean(rm)),
            "regret_kj": float(np.mean(reg)), "top1_rate": float(np.mean(top)),
        })
    return out


def sel_score(m: dict) -> float:
    """The original script's declared selection score (MAE / 30 kJ and within-group regret / 20 kJ, equal weight)."""
    return 0.5 * m["mae_kj"] / WP9.MAE_SCALE_KJ + 0.5 * m.get("regret_kj", np.nan) / WP9.REGRET_SCALE_KJ


# ---------------------------------------------------------------- one world


def run_world(world: str, X: np.ndarray, ids_ref: np.ndarray, out: Path) -> dict:
    d = np.load(K / "datasets" / f"twin_{world}.npz", allow_pickle=True)
    ids, groups, split = d["id"].astype(str), d["group"].astype(str), d["split"].astype(str)
    assert (ids == ids_ref).all(), "twin file id order differs from the terms cache"
    source, profile = d["source"].astype(str), d["profile"].astype(int)
    fail, unsafe = d["fail"].astype(int), d["unsafe"].astype(int)
    E, T = np.asarray(d["E"], np.float64), np.asarray(d["T"], np.float64)
    total_wplus, L = np.asarray(d["total_wplus"], np.float64), np.asarray(d["route_len"], np.float64)
    n = len(ids)
    fin = np.isfinite(E)
    lastj = np.where(fin.any(1), E.shape[1] - 1 - np.argmax(fin[:, ::-1], 1), -1)
    ar = np.arange(n)
    E_last = np.where(lastj >= 0, E[ar, np.maximum(lastj, 0)], np.nan)
    T_last = np.where(lastj >= 0, T[ar, np.maximum(lastj, 0)], np.nan)
    t_cmd = X[:, NAMES.index("_t_profile")]

    ok = fail == 0
    train, heldout = (split == "train") & ok, (split != "train") & ok
    dev = np.array([dev_fold(g) for g in groups]) & train
    fit_only, clean_train = train & ~dev, train & (unsafe == 0)
    assert np.isfinite(E_last[ok]).all() and (E_last[ok] > 0).all()
    print(f"\n[{world}] rows {n}: train goal-reached {train.sum()} (dev {dev.sum()} rows / "
          f"{len(set(groups[dev]))} groups; unsafe-but-reached {int((train & (unsafe == 1)).sum())}), "
          f"held-out goal-reached {heldout.sum()} in {len(set(groups[heldout]))} groups; "
          f"last observed station p10/50/90 {np.percentile(lastj[ok], [10, 50, 90])}; "
          f"E_last == total W+ on {np.mean(np.isclose(E_last[ok], total_wplus[ok])):.1%} of goal-reached rows")

    terms = make_terms(ids, groups, X)
    rows_fit = make_rows(np.flatnonzero(fit_only), ids, groups, E_last)
    rows_dev = make_rows(np.flatnonzero(dev), ids, groups, E_last)

    # ---- (pool, alpha) on the dev fold, original score
    table = []
    for pool, names in POOLS.items():
        for alpha in ALPHA_GRID:
            w = WP9.fit_work(rows_fit, terms, names, alpha)
            m = metrics(predict(rows_dev, terms, names, w), E_last[dev], groups[dev])
            m.update({"pool": pool, "alpha": alpha, "score": sel_score(m)})
            table.append(m)
            print(f"  dev pool={pool:13s} alpha={alpha:3.1f} logRMSE {m['log_rmse']:.4f} MdAPE {m['mdape_pct']:5.1f}% "
                  f"MAE {m['mae_kj']:6.1f} rho_g {m.get('spearman_within_group', np.nan):+.3f} "
                  f"regret {m.get('regret_kj', np.nan):6.1f} curse {m.get('curse_ratio', np.nan):.3f} score {m['score']:.4f}",
                  flush=True)
    chosen = min(table, key=lambda r: r["score"])
    by_logrmse = min(table, key=lambda r: r["log_rmse"])
    names, alpha = POOLS[chosen["pool"]], chosen["alpha"]
    print(f"  SELECTED pool={chosen['pool']} alpha={alpha} (score {chosen['score']:.4f}); "
          f"dev log-RMSE optimum would be pool={by_logrmse['pool']} alpha={by_logrmse['alpha']} ({by_logrmse['log_rmse']:.4f})")

    # ---- final fits on all training goal-reached rows
    rows_all = make_rows(ar, ids, groups, E_last)
    rows_train = make_rows(np.flatnonzero(train), ids, groups, E_last)
    w_main = WP9.fit_work(rows_train, terms, names, alpha)
    w_wplus = WP9.fit_work(make_rows(np.flatnonzero(train), ids, groups, total_wplus), terms, names, alpha)
    w_clean = WP9.fit_work(make_rows(np.flatnonzero(clean_train), ids, groups, E_last), terms, names, alpha)
    c_len = fit_len(L[train], E_last[train])
    a_t, b_t = fit_affine(t_cmd[train], E_last[train])
    a_T, b_T = fit_affine(t_cmd[train], T_last[train])          # time baseline for the time head (secondary)

    pred = {
        "analytic": predict(rows_all, terms, names, w_main),
        "analytic_wplus": predict(rows_all, terms, names, w_wplus),
        "analytic_clean": predict(rows_all, terms, names, w_clean),
        "len": np.maximum(c_len * L, 0.0),
        "time": np.maximum(a_t + b_t * t_cmd, 0.0),
    }
    pred_T = np.maximum(a_T + b_T * t_cmd, 0.0)

    # ---- held-out evaluation
    ho_clean = heldout & (unsafe == 0)
    ev = {"heldout": {}, "heldout_clean": {}, "dev_chosen": {}}
    for k, p in pred.items():
        tru = total_wplus if k == "analytic_wplus" else E_last
        ev["heldout"][k] = metrics(p[heldout], tru[heldout], groups[heldout])
        ev["heldout_clean"][k] = metrics(p[ho_clean], tru[ho_clean], groups[ho_clean])
    ev["heldout"]["analytic_wplus_vs_E_last"] = metrics(pred["analytic_wplus"][heldout], E_last[heldout], groups[heldout])
    ev["dev_chosen"]["analytic"] = {k: v for k, v in chosen.items() if k not in ("pool", "alpha", "score")}
    ev["dev_chosen"]["len"] = metrics(np.maximum(fit_len(L[fit_only], E_last[fit_only]) * L[dev], 0), E_last[dev], groups[dev])
    a0, b0 = fit_affine(t_cmd[fit_only], E_last[fit_only])
    ev["dev_chosen"]["time"] = metrics(np.maximum(a0 + b0 * t_cmd[dev], 0), E_last[dev], groups[dev])
    ev["time_baseline_heldout"] = metrics(pred_T[heldout], T_last[heldout], groups[heldout])

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{world}_analytic_pred.npz",
        id=ids, group=groups, split=split, source=source, profile=profile, fail=fail, unsafe=unsafe,
        pred_analytic=pred["analytic"].astype(np.float32), pred_analytic_wplus=pred["analytic_wplus"].astype(np.float32),
        pred_analytic_clean=pred["analytic_clean"].astype(np.float32),
        pred_len=pred["len"].astype(np.float32), pred_time=pred["time"].astype(np.float32),
        pred_T_from_cmd_time=pred_T.astype(np.float32),
        true_E=E_last.astype(np.float32), true_wplus=total_wplus.astype(np.float32), true_T=T_last.astype(np.float32),
        t_cmd=t_cmd.astype(np.float32), route_len=L.astype(np.float32), last_station=lastj.astype(np.int16),
        heldout=heldout, train=train, dev=dev, terms_names=np.array(names), terms=X[:, [NAMES.index(n) for n in names]].astype(np.float32),
    )
    coef = lambda w: {nm: float(c) for nm, c in zip(names, w)}  # noqa: E731
    summary = {
        "counts": {"rows": int(n), "train_goal_reached": int(train.sum()), "train_dev_rows": int(dev.sum()),
                   "train_dev_groups": len(set(groups[dev])), "train_unsafe_but_reached": int((train & (unsafe == 1)).sum()),
                   "heldout_goal_reached": int(heldout.sum()), "heldout_groups": len(set(groups[heldout])),
                   "heldout_groups_ge3": ev["heldout"]["analytic"].get("n_groups", 0),
                   "heldout_clean": int(ho_clean.sum()), "heldout_unsafe_but_reached": int((heldout & (unsafe == 1)).sum()),
                   "train_clean_median_kj": float(np.median(E_last[clean_train])),
                   "train_unsafe_but_reached_median_kj": float(np.median(E_last[train & (unsafe == 1)])) if (train & (unsafe == 1)).any() else None,
                   "last_station_p10_50_90": [float(v) for v in np.percentile(lastj[ok], [10, 50, 90])],
                   "E_last_equals_total_wplus_frac": float(np.mean(np.isclose(E_last[ok], total_wplus[ok])))},
        "selection": {"protocol": "md5(group) % 5 == 0 dev fold inside train; fit on the other training rows; score = "
                                  "0.5*MAE/30kJ + 0.5*within-group regret/20kJ (traverse_wp9_analytic.py); final fit on all train",
                      "chosen": {"pool": chosen["pool"], "alpha": alpha, "score": chosen["score"]},
                      "dev_log_rmse_optimum": {"pool": by_logrmse["pool"], "alpha": by_logrmse["alpha"],
                                               "log_rmse": by_logrmse["log_rmse"]},
                      "table": table},
        "analytic": {"terms": names, "alpha": alpha,
                     "coefficients": coef(w_main), "coefficients_nonzero": {k: v for k, v in coef(w_main).items() if v > 0},
                     "coefficients_wplus_target": {k: v for k, v in coef(w_wplus).items() if v > 0},
                     "coefficients_clean_fit": {k: v for k, v in coef(w_clean).items() if v > 0},
                     "fit_rows": {"main": int(train.sum()), "wplus": int(train.sum()), "clean": int(clean_train.sum())}},
        "trivial": {"len_kj_per_m": c_len, "time_a_kj": a_t, "time_b_kj_per_s": b_t,
                    "time_of_time_a_s": a_T, "time_of_time_b": b_T},
        "metrics": ev,
    }
    for k in ("analytic", "len", "time", "analytic_wplus", "analytic_clean"):
        m = ev["heldout"][k]
        print(f"  held-out {k:15s} n {m['n']:4d} groups {m.get('n_groups', 0):3d} logRMSE {m['log_rmse']:.4f} "
              f"MdAPE {m['mdape_pct']:5.1f}% MAPE {m['mape_pct']:5.1f}% rho_g {m.get('spearman_within_group', np.nan):+.3f} "
              f"rho_pooled {m['spearman_pooled']:+.3f} curse {m.get('curse_ratio', np.nan):.3f} "
              f"(diff {m.get('curse_diff', np.nan):+.3f}) regret {m.get('regret_kj', np.nan):5.1f} kJ bias {m['bias_kj']:+6.1f}")
    return summary


# ---------------------------------------------------------------- README


def write_readme(path: Path, S: dict) -> None:
    def row(world: str, label: str, key: str, blk: str = "heldout") -> str:
        m = S[world]["metrics"][blk][key]
        return (f"| {world} | {label} | {m['n']} | {m.get('n_groups', 0)} | {m['log_rmse']:.3f} | {m['mdape_pct']:.1f} | "
                f"{m['mape_pct']:.1f} | {m.get('spearman_within_group', float('nan')):+.3f} | {m['spearman_pooled']:+.3f} | "
                f"{m.get('curse_ratio', float('nan')):.3f} | {m.get('curse_diff', float('nan')):+.3f} | "
                f"{m.get('regret_kj', float('nan')):.1f} | {m['bias_kj']:+.1f} |")

    hdr = ("| world | model | n | groups>=3 | log-RMSE | MdAPE % | MAPE % | rho within group | rho pooled | curse ratio | "
           "curse diff | regret kJ | bias kJ |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [
        "# Analytic energy baselines (night 2, Study 1 stage B)", "",
        "Produced by `scripts/n2_energy_analytic.py` from `datasets/twin_{rigid,crm}.npz` (CPU only). The analytic model is",
        "`scripts/traverse_wp9_analytic.py`'s route physics terms (route waypoints + commanded speed profile + f104 heightmap;",
        "`resample`/`route_terms` imported unchanged) fitted by its NNLS with the within-group centred copy of the design",
        "(weight alpha), per world. Fit rows: `split == 'train' & fail == 0`; (pool, alpha) chosen on the md5(group) % 5 == 0",
        "dev fold by the original's score (0.5 MAE/30 kJ + 0.5 within-group regret/20 kJ), then refitted on all training rows.",
        "Held-out: `split != 'train' & fail == 0` (goal reached). Target E_last = first-arrival cumulative W+ at the furthest",
        "observed station (kJ). Curse ratio = mean over groups (>= 3 goal-reached routes) of (pred/true at the route with the",
        "lowest predicted energy) / mean(pred/true over the group); < 1 means the pick looks cheaper than it is. curse diff is",
        "the 09-07 form (difference instead of ratio).", "",
        "## Held-out metrics (goal-reached routes, unsafe-but-reached included)", "", hdr,
    ]
    for w in WORLDS:
        lines += [row(w, "analytic (E_last)", "analytic"), row(w, "kJ/m x route length", "len"),
                  row(w, "a + b x commanded time", "time"),
                  row(w, "analytic, total W+ target (scored vs total W+)", "analytic_wplus"),
                  row(w, "analytic, clean-only fit (unsafe == 0)", "analytic_clean")]
    lines += ["", "## Held-out metrics, clean routes only (fail == 0 & unsafe == 0)", "", hdr]
    for w in WORLDS:
        lines += [row(w, "analytic (E_last)", "analytic", "heldout_clean"), row(w, "kJ/m x route length", "len", "heldout_clean"),
                  row(w, "a + b x commanded time", "time", "heldout_clean"),
                  row(w, "analytic, clean-only fit", "analytic_clean", "heldout_clean")]
    lines += ["", "## Fitted coefficients", ""]
    for w in WORLDS:
        s = S[w]
        c, t, sel = s["analytic"], s["trivial"], s["selection"]["chosen"]
        lines += [f"**{w}** -- pool `{sel['pool']}`, alpha {sel['alpha']} (dev score {sel['score']:.4f}; dev log-RMSE optimum: "
                  f"`{s['selection']['dev_log_rmse_optimum']['pool']}` alpha {s['selection']['dev_log_rmse_optimum']['alpha']}); "
                  f"fit on {c['fit_rows']['main']} rows, held-out {s['counts']['heldout_goal_reached']} rows / "
                  f"{s['counts']['heldout_groups']} groups ({s['counts']['heldout_groups_ge3']} with >= 3).", "",
                  "- analytic (E_last), non-zero terms: " + ", ".join(f"{k} {v:.4g}" for k, v in c["coefficients_nonzero"].items()),
                  "- analytic (total W+ target): " + ", ".join(f"{k} {v:.4g}" for k, v in c["coefficients_wplus_target"].items()),
                  "- analytic (clean-only fit, %d rows): " % c["fit_rows"]["clean"]
                  + ", ".join(f"{k} {v:.4g}" for k, v in c["coefficients_clean_fit"].items()),
                  f"- kJ/m x length: {t['len_kj_per_m']:.3f} kJ/m",
                  f"- a + b x commanded time: a {t['time_a_kj']:.1f} kJ, b {t['time_b_kj_per_s']:.2f} kJ/s",
                  f"- time baseline (secondary, for the time head): T = {t['time_of_time_a_s']:.2f} s + {t['time_of_time_b']:.3f} x "
                  f"commanded time; held-out log-RMSE {s['metrics']['time_baseline_heldout']['log_rmse']:.3f}, "
                  f"MdAPE {s['metrics']['time_baseline_heldout']['mdape_pct']:.1f} %, within-group rho "
                  f"{s['metrics']['time_baseline_heldout'].get('spearman_within_group', float('nan')):+.3f}", ""]
    lines += ["## Caveats", ""]
    for w in WORLDS:
        c = S[w]["counts"]
        if c["train_unsafe_but_reached"] >= 0.05 * c["train_goal_reached"]:
            lines += [f"- **{w}**: the prescribed fit rows (`fail == 0`) mix {c['train_unsafe_but_reached']} unsafe-but-reached "
                      f"routes (median E_last {c['train_unsafe_but_reached_median_kj']:.0f} kJ, "
                      f"{100 * c['train_unsafe_but_reached'] / c['train_goal_reached']:.0f} % of the rows) with clean successes "
                      f"(median {c['train_clean_median_kj']:.0f} kJ). A level NNLS fit on that mixture sits far above the clean "
                      f"median, which is what the MdAPE of the main rows measures; the clean-only fit is the like-for-like "
                      f"baseline for an energy head trained on clean-driving (event-censored) targets, and the held-out set "
                      f"holds {c['heldout_unsafe_but_reached']} unsafe-but-reached routes whose struggle energy no route-geometry "
                      f"model (or clean-driving head) predicts."]
        else:
            lines += [f"- **{w}**: {c['train_unsafe_but_reached']} unsafe-but-reached route(s) among the {c['train_goal_reached']} "
                      f"fit rows and {c['heldout_unsafe_but_reached']} among the held-out rows; main and clean-only fits coincide."]
    lines += ["- Selection picked alpha = 0 in both worlds: the within-group centred copy of the design (the original's "
              "within-mission weighting) does not help here.",
              "- The original's `tract` term carries no standing-start kinetic energy on f104 (profiles begin at the commanded "
              "speed); the `tract_launch` pool offered `accel`/`kemax` and NNLS never used them.",
              "- `resample` drops the last partial station (< 0.5 m, in the v -> 0 taper); goal-reached episodes end 5-9 "
              "stations short of station 95 (2.5 m goal circle), so E_last is the energy at the furthest observed station.",
              "", "## Files", "",
              "- `{world}_analytic_pred.npz`: id, group, split, source, profile, fail, unsafe, pred_analytic, pred_analytic_wplus,",
              "  pred_analytic_clean, pred_len, pred_time, pred_T_from_cmd_time, true_E (E_last), true_wplus, true_T, t_cmd, route_len,",
              "  last_station, heldout / train / dev masks, the selected terms per route (terms, terms_names). All rows of the twin",
              "  file, same order; failed rows carry predictions too (their true_E is the energy at the censoring station).",
              "- `summary.json`: counts, the full dev selection table, coefficients, all metrics.",
              "- `route_terms.npz`: the 23 physics terms for every route id (cache; identical for both worlds)."]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(K / "energy_baselines"))
    ap.add_argument("--rebuild-terms", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with np.load(K / "datasets" / "twin_rigid.npz", allow_pickle=True) as d:
        ids = d["id"].astype(str)
    with np.load(K / "datasets" / "twin_crm.npz", allow_pickle=True) as d:
        assert (d["id"].astype(str) == ids).all(), "twin files are not aligned"
    tmap = TerrainMap.from_dir(ARENA)
    X = extract_terms(ids, tmap, out / "route_terms.npz", args.rebuild_terms)

    S = {w: run_world(w, X, ids, out) for w in WORLDS}
    S["_provenance"] = {
        "script": "scripts/n2_energy_analytic.py", "analytic_source": "scripts/traverse_wp9_analytic.py (resample, route_terms, "
        "fit_work, nnls, design, spearman imported unchanged)", "arena": str(ARENA.relative_to(REPO)),
        "datasets": ["artifacts/traverse/crm_night2_v1/datasets/twin_rigid.npz", "artifacts/traverse/crm_night2_v1/datasets/twin_crm.npz"],
        "route_geometry": {w: [str(p.relative_to(REPO)) for p in RUN_DIRS[w]] for w in WORLDS},
        "pools": POOLS, "alpha_grid": ALPHA_GRID, "min_group_routes": MIN_GROUP,
        "constants": {"mass_kg": WP9.MASS_KG, "crr_nom": WP9.CRR_NOM, "cda_nom_m2": WP9.CDA_NOM, "step_m": WP9.STEP_M,
                      "v_scale": WP9.V_SCALE},
        "notes": [
            "f104 designed profiles start at the full commanded speed and taper to 0 at the goal (no launch ramp): the original "
            "'tract' term has no standing-start kinetic energy, so the 'tract_launch' pool adds 'accel' (= launch KE + positive "
            "kinetic-energy increments) and 'kemax'.",
            "resample() drops the last partial station (< 0.5 m, inside the v -> 0 taper); kept unchanged.",
            "The goal circle is 2.5 m, so goal-reached episodes end 5-9 stations before station 95; E_last is the energy at the "
            "furthest observed station and equals the episode total W+ on ~90 % of goal-reached rows.",
            "MASS_KG = 25000/9.81 is the original's constant, not the f104 vehicle's stored mass (none is stored); NNLS rescales "
            "the tractive terms, the constant only fixes the grade/rolling/aero mix inside the positive-force clip.",
            "Terrain frame checked: recorded pitch vs TerrainMap along-track slope at the recorded pose r = -0.98 (mirrored y: -0.16).",
        ],
    }
    (out / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    write_readme(out / "README.md", S)
    print(f"\nwrote {out / 'summary.json'}, {out / 'README.md'}, {out}/{{rigid,crm}}_analytic_pred.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
