#!/usr/bin/env python
"""A1 arm: the calibrated ANALYTIC work/time baseline -- the plan's "A* route/profile sweep plus
calibrated analytic work estimate" row (docs/vision/hmmwv_traverse/energy_and_crater_experiment_plan.md).

For every (layout, candidate) in the A1 truth table this predicts POSITIVE shaft work (``w_pos_kj``) and
arrival time from the planned route geometry, the arena heightmap and the commanded speed profile ONLY.
No Chrono, no learned network, no recorded state/action/pose. This is the strong classical competitor
NRD imagination has to beat, so it is fitted properly rather than assumed.

PHYSICS. Every term is an integral along the planned route (0.5 m stations); ``v`` is the COMMANDED
speed, ``dh`` the height change from ``TerrainMap.height``, ``kappa`` the path curvature, m = 2548 kg:

  tract       sum(max(0, F) ds), F = m g sin + Crr m g cos + 1/2 rho Cd A v^2 + m a
              the per-segment POSITIVE tractive work: grade, rolling resistance, aerodynamic drag and
              acceleration combined with the clip that matters -- an engine cannot be paid for a
              descent, and gravity assist only cancels drag while the net force stays positive.
              a comes from the commanded profile (launch from rest and the terminal taper included).
  tract2      sum(max(0,F)^2 ds) / (m g)      slip / resistive loss, which grows with force squared
  tract_lowv  sum(max(0,F) (5/v) ds)          low-speed driveline penalty: the same tractive work costs
                                              more engine work when it is delivered slowly (torque
                                              converter slip, poor operating point). This term is what
                                              lets the estimate rank a speed sweep on one fixed route.
  tract_hiv   sum(max(0,F) (v/5) ds)          the opposite sign of that effect
  time        sum(ds/v)                       profile duration; its coefficient is a parasitic power
  corn        m sum(v^2 kappa ds)             lateral tyre force through the corner: steering scrub
  cross       m g sum(|cross-slope| ds)       side-slope scrub
  one         1                               standing-start / shutdown overhead

The separated grade / rolling / acceleration / aerodynamic terms, a roughness term, peak kinetic energy
and a quadratic cornering term are all OFFERED to the fit as alternative pools; which pool is used is
decided by the leave-one-arena-out protocol below, never by looking at f105-f111.

FIT. Non-negative least squares (a negative physical term would be nonsense, and non-negativity is what
stops the fit from paying a route for going downhill) against ``w_pos_kj`` on COMPLIANT runs of the
TRAINING arenas f101-f104 only. The design is augmented with a copy of itself centred within each
layout, weighted ``alpha``: the arm's job is both to state the level of the work and to separate the
candidates of ONE mission from each other, and the within-mission spread (~16 kJ s.d.) is four times
smaller than the between-mission spread (~70 kJ). ``alpha`` and the term pool are chosen together by
minimising a declared leave-one-arena-out score over the four training arenas,
``0.5 * MAE/30kJ + 0.5 * selection_regret/20kJ``.

Every emitted prediction is out-of-arena: f105-f111 use the pooled f101-f104 fit, and each training
arena is predicted by a fit that excludes it. Nothing is ever scored by a fit that saw its arena.

TIME is a separate non-negative fit on FEASIBLE runs (arrival time is only defined there; restricting it
to runs that already met the deadline would truncate the target and bias the factor down). Chrono runs
a few percent longer than the rule-based profile promises, and longer still where the tractive demand
is high, so ``tract2`` is offered alongside the profile duration.

FLOOR. ``nedm.traverse.energy_floor.route_features`` refitted against w_pos_kj on the same training
arenas, emitted as a separate column for the plan's NRD+floor arm. The shipped
``artifacts/traverse/wp5_energy_floor/energy_floor.json`` is fitted against SIGNED work on arena_v1 and
under-predicts this arena family, so it never binds; it is NOT modified -- a new json is written here.

TERRAIN. Heights come from ``TerrainMap.from_dir``, which re-orients the BMP into the frame Chrono
simulates. The authored feature list in ``meta["features"]`` is in the generation frame and is mirrored
in y; this script never reads it. ``--mirror-terrain`` refits on a y-mirrored heightfield as a control:
if the heights this script reads were the mirror image of the driven terrain, the mirrored fit would be
just as good.

  PYTHONPATH=src python scripts/traverse_wp9_analytic.py \
      --caches artifacts/traverse/wp7_cache_v1 artifacts/traverse/wp7_cache_sealed artifacts/traverse/wp8_cache_sealed2 \
      --truth artifacts/traverse/wp9_energy/truth.json --out artifacts/traverse/wp9_energy
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from nedm.traverse.energy_floor import route_features  # noqa: E402
from nedm.traverse.terrain import TerrainMap  # noqa: E402

G = 9.81
MASS_KG = 25000.0 / 9.81   # the chassis mass the truth builder uses
RHO_AIR = 1.2
CRR_NOM = 0.01             # HMMWV tyre default rolling resistance (nominal; the fit rescales it)
CDA_NOM = 3.5              # HMMWV Cd * frontal area, m^2 (nominal)
V_SCALE = 5.0              # reference speed the low/high-speed weights are written against
STEP_M = 0.5               # route resampling step (the planner's own sample_step_m)
TRAIN_ARENAS = ("arena_f101", "arena_f102", "arena_f103", "arena_f104")

TERMS = ["grade", "desc", "roll", "accel", "brake", "aero", "time", "corn", "corn2", "cross",
         "tract", "tract2", "tract_lowv", "tract_hiv", "grade_lowv", "rough", "rough_v", "tlow",
         "kemax", "undul", "one"]
EXTRA = ["_L", "_t_profile"]

# The pools offered to the leave-one-arena-out selection. "separated" is the textbook additive
# decomposition; "tract" is the same physics with the per-segment positive-force clip applied first;
# "full" offers both and lets non-negative least squares choose.
POOLS = {
    "separated": ["grade", "desc", "roll", "accel", "brake", "aero", "time", "corn", "cross",
                  "kemax", "grade_lowv", "rough", "rough_v", "one"],
    "tract": ["tract", "tract2", "tract_lowv", "tract_hiv", "time", "corn", "cross", "one"],
    "full": ["grade", "desc", "roll", "accel", "brake", "aero", "time", "corn", "corn2", "cross",
             "tract", "tract2", "tract_lowv", "tract_hiv", "grade_lowv", "rough", "rough_v",
             "kemax", "undul", "one"],
}
ALPHA_GRID = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
TIME_POOL = ["_t_profile", "tract", "tract2", "grade", "corn", "_L", "tlow", "one"]

MAE_SCALE_KJ = 30.0        # declared normalisers of the selection score
REGRET_SCALE_KJ = 20.0


# ---------------------------------------------------------------- route -> physical terms


def resample(pts: np.ndarray, speeds: np.ndarray, tmap: TerrainMap, step_m: float = STEP_M) -> np.ndarray:
    """(s, x, y, v_cmd, h, kappa, |cross slope|, roughness) at uniform stations along the route."""
    p = np.asarray(pts, np.float64)
    seg = np.hypot(*np.diff(p, axis=0).T)
    st = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, float(st[-1]) + 1e-9, step_m)
    x, y = np.interp(s, st, p[:, 0]), np.interp(s, st, p[:, 1])
    v = np.interp(s, st, np.asarray(speeds, np.float64))
    h = np.asarray(tmap.height(x, y), np.float64)
    tx, ty = np.gradient(x), np.gradient(y)
    n = np.hypot(tx, ty) + 1e-9
    tx, ty = tx / n, ty / n
    kappa = np.abs(np.gradient(np.unwrap(np.arctan2(ty, tx)))) / step_m
    gx, gy = tmap.gradient(x, y)
    gx, gy = np.asarray(gx, np.float64), np.asarray(gy, np.float64)
    cross = np.abs(-gx * ty + gy * tx)
    offs = [(1.5, 0.0), (-1.5, 0.0), (0.0, 1.5), (0.0, -1.5)]  # wheel-track sized probes
    rough = np.stack([np.asarray(tmap.height(x + a * tx - b * ty, y + a * ty + b * tx)) for a, b in offs] + [h]).std(0)
    return np.stack([s, x, y, v, h, kappa, cross, rough])


def route_terms(prof: np.ndarray) -> dict[str, float]:
    """Physically motivated work terms (kJ, except ``time``/``tlow`` in s and ``one`` = 1)."""
    s, _x, _y, vc, h, kap, cr, rg = prof
    ds = np.maximum(np.diff(s), 1e-9)
    L = float(s[-1])
    v = np.maximum(vc, 0.3)
    vm = 0.5 * (v[1:] + v[:-1])
    dh = np.diff(h)
    hyp = np.hypot(ds, dh)
    sin_t, cos_t = dh / hyp, ds / hyp
    dke = 0.5 * MASS_KG * np.diff(v**2)
    a_seg = np.diff(v**2) / (2.0 * ds)                     # acceleration the profile demands
    force = (MASS_KG * G * sin_t + CRR_NOM * MASS_KG * G * cos_t
             + 0.5 * RHO_AIR * CDA_NOM * vm**2 + MASS_KG * a_seg)
    fp = np.maximum(force, 0.0)
    dt = ds / vm
    km, crm, rgm = 0.5 * (kap[1:] + kap[:-1]), 0.5 * (cr[1:] + cr[:-1]), 0.5 * (rg[1:] + rg[:-1])
    kj, r5 = 1e-3, vm / V_SCALE
    return {
        "grade": kj * MASS_KG * G * float(np.maximum(dh, 0).sum()),
        "desc": kj * MASS_KG * G * float(np.maximum(-dh, 0).sum()),
        "roll": kj * CRR_NOM * MASS_KG * G * L,
        "accel": kj * (float(np.maximum(dke, 0).sum()) + 0.5 * MASS_KG * float(v[0]) ** 2),
        "brake": kj * float(np.maximum(-dke, 0).sum()),
        "aero": kj * 0.5 * RHO_AIR * CDA_NOM * float((vm**2 * ds).sum()),
        "time": float(dt.sum()),
        "corn": kj * MASS_KG * float((vm**2 * km * ds).sum()),
        "corn2": kj * MASS_KG * float(((vm**2 * km) ** 2 * ds).sum()) / 100.0,
        "cross": kj * MASS_KG * G * float((crm * ds).sum()),
        "tract": kj * float((fp * ds).sum()),
        "tract2": kj * float((fp**2 * ds).sum()) / (MASS_KG * G),
        "tract_lowv": kj * float((fp * ds / r5).sum()),
        "tract_hiv": kj * float((fp * ds * r5).sum()),
        "grade_lowv": kj * MASS_KG * G * float((np.maximum(dh, 0) / r5).sum()),
        "rough": kj * MASS_KG * G * float((rgm * ds).sum()),
        "rough_v": kj * MASS_KG * G * float((rgm * r5 * ds).sum()),
        "tlow": float(dt[vm < 3.0].sum()),
        "kemax": kj * 0.5 * MASS_KG * float(v.max()) ** 2,
        "undul": kj * MASS_KG * G * float(np.abs(dh).sum()),
        "one": 1.0,
        "_L": L,
        "_t_profile": float(dt.sum()),
    }


# ---------------------------------------------------------------- NNLS (no scipy in this env)


def nnls(A: np.ndarray, b: np.ndarray, tol: float = 1e-10, max_iter: int = 400) -> np.ndarray:
    """Lawson-Hanson active-set non-negative least squares."""
    A, b = np.asarray(A, np.float64), np.asarray(b, np.float64)
    n = A.shape[1]
    P = np.zeros(n, bool)
    x = np.zeros(n)
    w = A.T @ (b - A @ x)
    for _ in range(max_iter):
        free = ~P
        if not free.any() or w[free].max() <= tol:
            break
        idx = np.flatnonzero(free)
        P[idx[np.argmax(w[idx])]] = True
        for _ in range(max_iter):
            s = np.zeros(n)
            s[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if s[P].min() > tol:
                x = s
                break
            bad = P & (s <= tol)
            den = x[bad] - s[bad]
            safe = np.abs(den) > 1e-15
            alpha = np.min(np.where(safe, x[bad] / np.where(safe, den, 1.0), np.inf))
            x = x + alpha * (s - x)
            P = P & (x > tol)
            if not P.any():
                x = np.zeros(n)
                break
        w = A.T @ (b - A @ x)
    return x


def fit_nnls(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NNLS with columns pre-scaled to unit RMS (conditioning only; non-negativity is preserved)."""
    scale = np.sqrt((A**2).mean(0))
    scale[scale <= 0] = 1.0
    return nnls(A / scale, b) / scale


def fit_work(rows: list[dict], terms: dict, names: list[str], alpha: float) -> np.ndarray:
    """NNLS on the level, plus ``alpha`` copies of the design centred within each layout."""
    A = design(rows, terms, names)
    y = np.array([r["w_pos_kj"] for r in rows], np.float64)
    if alpha <= 0:
        return fit_nnls(A, y)
    g: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        g[r["key"]].append(i)
    Ad, yd, keep = A.copy(), y.copy(), []
    for ii in g.values():
        if len(ii) < 2:
            continue
        Ad[ii] -= A[ii].mean(0)
        yd[ii] -= y[ii].mean()
        keep += ii
    if not keep:
        return fit_nnls(A, y)
    keep = np.array(sorted(keep))
    s = np.sqrt(len(y) / len(keep))               # keep the two blocks comparably weighted
    return fit_nnls(np.vstack([A, alpha * s * Ad[keep]]), np.concatenate([y, alpha * s * yd[keep]]))


def design(rows: list[dict], terms: dict, names: list[str]) -> np.ndarray:
    return np.array([[terms[(r["key"], r["candidate"])][n] for n in names] for r in rows], np.float64)


# ---------------------------------------------------------------- extraction / cache


def extract_terms(caches: list[str], assets: Path, needed: set[tuple[str, str]], mirror: bool = False) -> dict:
    tmaps: dict[str, TerrainMap] = {}
    out: dict[tuple[str, str], dict] = {}
    for cache in caches:
        files = sorted(Path(cache).glob("*.npz"))
        for i, f in enumerate(files):
            with np.load(f, allow_pickle=True) as d:
                key, cand, arena = str(d["layout"]), str(d["candidate"]), str(d["arena"])
                if (key, cand) not in needed or (key, cand) in out:
                    continue
                pts = np.asarray(d["route_waypoints"], np.float64)
                spd = np.asarray(d["route_speeds"], np.float64)
            if arena not in tmaps:
                t = TerrainMap.from_dir(assets / arena)
                tmaps[arena] = TerrainMap(np.flipud(t.height_grid).copy(), t.size_m, t.meta) if mirror else t
            prof = resample(pts, spd, tmaps[arena])
            row = route_terms(prof)
            row["_floor"] = route_features(prof[1:3].T, prof[3], prof[0], tmaps[arena]).tolist()
            out[(key, cand)] = row
            if (i + 1) % 1000 == 0:
                print(f"  {Path(cache).name}: {i+1}/{len(files)}", flush=True)
        print(f"{cache}: {len(files)} files, {len(out)} routes so far", flush=True)
    return out


def save_terms(path: Path, terms: dict) -> None:
    keys = sorted(terms)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        key=np.array([k for k, _ in keys]), candidate=np.array([c for _, c in keys]),
        names=np.array(TERMS + EXTRA),
        X=np.array([[terms[k][n] for n in TERMS + EXTRA] for k in keys], np.float64),
        floor=np.array([terms[k]["_floor"] for k in keys], np.float64),
    )


def load_terms(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as d:
        names = [str(n) for n in d["names"]]
        X, floor = d["X"], d["floor"]
        out = {}
        for i, (k, c) in enumerate(zip(d["key"], d["candidate"])):
            row = {n: float(X[i, j]) for j, n in enumerate(names)}
            row["_floor"] = floor[i].tolist()
            out[(str(k), str(c))] = row
    return out


# ---------------------------------------------------------------- metrics


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rank(z: np.ndarray) -> np.ndarray:
    o = np.argsort(z, kind="mergesort")
    r = np.empty(len(z), np.float64)
    r[o] = np.arange(len(z), dtype=np.float64)
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(_rank(a), _rank(b))


def eval_level(pred: np.ndarray, true: np.ndarray) -> dict:
    if len(pred) < 3:
        return {"n": int(len(pred))}
    res = pred - true
    ss = float(((true - true.mean()) ** 2).sum())
    return {"n": int(len(pred)), "mae_kj": float(np.abs(res).mean()), "bias_kj": float(res.mean()),
            "mape_pct": float(100.0 * np.mean(np.abs(res) / np.maximum(true, 1e-6))),
            "pearson": pearson(pred, true), "spearman": spearman(pred, true),
            "r2": float(1.0 - float((res**2).sum()) / ss) if ss > 0 else float("nan"),
            "true_mean_kj": float(true.mean())}


def eval_rank(rows: list[dict], pred: np.ndarray, min_n: int = 4) -> dict:
    """Ranking where the decision is made: among one layout's own compliant candidates."""
    g: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        g[r["key"]].append(i)
    t = np.array([r["w_pos_kj"] for r in rows], np.float64)
    rho, reg, rnd, top = [], [], [], []
    for ii in g.values():
        if len(ii) < min_n:
            continue
        ii = np.array(ii)
        rho.append(spearman(pred[ii], t[ii]))
        j = ii[int(np.argmin(pred[ii]))]
        reg.append(float(t[j] - t[ii].min()))
        rnd.append(float(t[ii].mean() - t[ii].min()))
        top.append(bool(t[j] == t[ii].min()))
    if not rho:
        return {"n_layouts": 0}
    return {"n_layouts": len(rho), "mean_spearman": float(np.nanmean(rho)),
            "top1_rate": float(np.mean(top)), "mean_regret_kj": float(np.mean(reg)),
            "median_regret_kj": float(np.median(reg)),
            "mean_regret_of_a_random_pick_kj": float(np.mean(rnd)),
            "regret_vs_random_pct": float(100.0 * np.mean(reg) / max(np.mean(rnd), 1e-9))}


def loao_score(by_arena: dict[str, list[dict]], terms: dict, names: list[str], alpha: float) -> dict:
    """Leave-one-arena-out over the training arenas: level error and selection regret."""
    mae, reg, rho = [], [], []
    arenas = sorted(by_arena)
    for held in arenas:
        tr = [r for a in arenas if a != held for r in by_arena[a]]
        te = by_arena[held]
        if len(tr) < 20 or len(te) < 5:
            continue
        w = fit_work(tr, terms, names, alpha)
        p = np.maximum(design(te, terms, names) @ w, 0.0)
        t = np.array([r["w_pos_kj"] for r in te])
        mae.append(float(np.abs(p - t).mean()))
        rk = eval_rank(te, p)
        reg.append(rk.get("mean_regret_kj", np.nan))
        rho.append(rk.get("mean_spearman", np.nan))
    m, g = float(np.mean(mae)), float(np.nanmean(reg))
    return {"mae_kj": m, "regret_kj": g, "spearman": float(np.nanmean(rho)),
            "score": 0.5 * m / MAE_SCALE_KJ + 0.5 * g / REGRET_SCALE_KJ}


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", nargs="*", default=[
        "artifacts/traverse/wp7_cache_v1", "artifacts/traverse/wp7_cache_sealed", "artifacts/traverse/wp8_cache_sealed2"])
    ap.add_argument("--truth", default="artifacts/traverse/wp9_energy/truth.json")
    ap.add_argument("--assets", default="assets/traverse")
    ap.add_argument("--out", default="artifacts/traverse/wp9_energy")
    ap.add_argument("--terms-cache", default="artifacts/traverse/wp9_energy/analytic_terms.npz")
    ap.add_argument("--rebuild-terms", action="store_true")
    ap.add_argument("--k-sigma", type=float, default=1.5, help="energy-floor slack, in residual sigmas")
    ap.add_argument("--mirror-terrain", action="store_true",
                    help="control: read a y-MIRRORED heightfield. Writes arm_analytic_mirrored.json; if this "
                         "fits as well as the real one, the terrain the script reads is not the driven terrain.")
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    args = ap.parse_args()

    rows = json.loads(Path(args.truth).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("_mirrored" if args.mirror_terrain else "")

    cache_path = Path(args.terms_cache if not args.mirror_terrain
                      else str(args.terms_cache).replace(".npz", "_mirrored.npz"))
    needed = {(r["key"], r["candidate"]) for r in rows}
    if cache_path.exists() and not args.rebuild_terms:
        terms = load_terms(cache_path)
        print(f"terms cache: {len(terms)} routes from {cache_path}")
    else:
        terms = extract_terms(args.caches, Path(args.assets), needed, mirror=args.mirror_terrain)
        save_terms(cache_path, terms)
        print(f"wrote {cache_path} ({len(terms)} routes)")
    missing = [r for r in rows if (r["key"], r["candidate"]) not in terms]
    if missing:
        print(f"WARNING: {len(missing)} truth rows have no cached route geometry; dropped", file=sys.stderr)
        rows = [r for r in rows if (r["key"], r["candidate"]) in terms]

    comp_by_arena = {a: [r for r in rows if r["arena"] == a and r["compliant"]] for a in TRAIN_ARENAS}
    feas_by_arena = {a: [r for r in rows if r["arena"] == a and r["feasible"]] for a in TRAIN_ARENAS}
    train_comp = [r for a in TRAIN_ARENAS for r in comp_by_arena[a]]
    train_feas = [r for a in TRAIN_ARENAS for r in feas_by_arena[a]]
    print("training compliant per arena:", {a: len(v) for a, v in comp_by_arena.items()})
    print("training feasible  per arena:", {a: len(v) for a, v in feas_by_arena.items()})

    # ---- choose (term pool, within-mission weight) by the declared LOAO score, training arenas only
    sel_table = []
    for pool_name, pool in POOLS.items():
        for alpha in ALPHA_GRID:
            s = loao_score(comp_by_arena, terms, pool, alpha)
            sel_table.append({"pool": pool_name, "alpha": alpha, **s})
            print(f"  pool={pool_name:10s} alpha={alpha:4.1f} LOAO mae {s['mae_kj']:6.2f} kJ "
                  f"regret {s['regret_kj']:6.2f} kJ rho {s['spearman']:+.3f} score {s['score']:.4f}", flush=True)
    chosen = min(sel_table, key=lambda d: d["score"])
    names, alpha = POOLS[chosen["pool"]], chosen["alpha"]
    print(f"SELECTED pool={chosen['pool']} alpha={alpha} (LOAO score {chosen['score']:.4f})")

    w_full = fit_work(train_comp, terms, names, alpha)
    w_loao = {a: fit_work([r for b in TRAIN_ARENAS if b != a for r in comp_by_arena[b]], terms, names, alpha)
              for a in TRAIN_ARENAS}

    # ---- time: forward selection on LOAO MAE, fitted on FEASIBLE runs
    def t_loao(nm: list[str]) -> float:
        e = []
        for held in TRAIN_ARENAS:
            tr = [r for a in TRAIN_ARENAS if a != held for r in feas_by_arena[a]]
            te = feas_by_arena[held]
            w = fit_nnls(design(tr, terms, nm), np.array([r["time_s"] for r in tr]))
            e.append(np.abs(design(te, terms, nm) @ w - np.array([r["time_s"] for r in te])))
        return float(np.concatenate(e).mean())

    tnames, t_best, t_trace = ["_t_profile"], t_loao(["_t_profile"]), []
    t_trace.append({"terms": list(tnames), "loao_mae_s": t_best})
    while True:
        cands = [(t_loao(tnames + [n]), n) for n in TIME_POOL if n not in tnames]
        if not cands:
            break
        gain, n = min(cands)
        if t_best - gain < 0.01:
            break
        tnames, t_best = tnames + [n], gain
        t_trace.append({"added": n, "loao_mae_s": t_best})
    tw_full = fit_nnls(design(train_feas, terms, tnames), np.array([r["time_s"] for r in train_feas]))
    tw_loao = {a: fit_nnls(design([r for b in TRAIN_ARENAS if b != a for r in feas_by_arena[b]], terms, tnames),
                           np.array([r["time_s"] for b in TRAIN_ARENAS if b != a for r in feas_by_arena[b]]))
               for a in TRAIN_ARENAS}

    # ---- energy floor: the WP5 route-geometry features refitted against w_pos_kj (ordinary least
    #      squares, as in WP5: floor = fit - k*sigma must be allowed to sit on the data)
    FLOOR_NAMES = ["length_m", "v2_length/100", "climb_m", "v_peak^2/10", "accel^2/10", "one"]
    Ff = np.array([terms[(r["key"], r["candidate"])]["_floor"] for r in train_comp], np.float64)
    yf = np.array([r["w_pos_kj"] for r in train_comp])
    fw_full, *_ = np.linalg.lstsq(Ff, yf, rcond=None)
    sig_full = float((Ff @ fw_full - yf).std())
    fw_loao, sig_loao = {}, {}
    for held in TRAIN_ARENAS:
        tr = [r for a in TRAIN_ARENAS if a != held for r in comp_by_arena[a]]
        F = np.array([terms[(r["key"], r["candidate"])]["_floor"] for r in tr], np.float64)
        y = np.array([r["w_pos_kj"] for r in tr])
        v, *_ = np.linalg.lstsq(F, y, rcond=None)
        fw_loao[held], sig_loao[held] = v, float((F @ v - y).std())

    # ---- predict everywhere, always with a fit that did not see the arena
    preds: dict[str, dict] = {}
    for r in rows:
        a, t = r["arena"], terms[(r["key"], r["candidate"])]
        ww = w_loao[a] if a in TRAIN_ARENAS else w_full
        tt = tw_loao[a] if a in TRAIN_ARENAS else tw_full
        ff, sg = (fw_loao[a], sig_loao[a]) if a in TRAIN_ARENAS else (fw_full, sig_full)
        preds[f"{r['key']}|{r['candidate']}"] = {
            "pred_w_pos_kj": float(max(0.0, np.array([t[n] for n in names]) @ ww)),
            "pred_time_s": float(max(0.0, np.array([t[n] for n in tnames]) @ tt)),
            "floor_w_pos_kj": float(max(0.0, np.asarray(t["_floor"]) @ ff - args.k_sigma * sg)),
        }

    def pw(rs): return np.array([preds[f"{r['key']}|{r['candidate']}"]["pred_w_pos_kj"] for r in rs])
    def pt(rs): return np.array([preds[f"{r['key']}|{r['candidate']}"]["pred_time_s"] for r in rs])
    def tw(rs): return np.array([r["w_pos_kj"] for r in rs])

    arenas = sorted({r["arena"] for r in rows})
    per_arena = {}
    for a in arenas:
        comp = [r for r in rows if r["arena"] == a and r["compliant"]]
        feas = [r for r in rows if r["arena"] == a and r["feasible"]]
        blk = {"fit_saw_this_arena": False,
               "held_out_by": "leave-one-arena-out within f101-f104" if a in TRAIN_ARENAS
                              else "never entered any fit (arena outside f101-f104)"}
        if comp:
            blk["work_compliant"] = eval_level(pw(comp), tw(comp))
            blk["rank_within_layout"] = eval_rank(comp, pw(comp))
        if feas:
            blk["work_feasible"] = eval_level(pw(feas), tw(feas))
            p, t = pt(feas), np.array([r["time_s"] for r in feas])
            dl, ot = np.array([r["deadline_s"] for r in feas]), np.array([r["on_time"] for r in feas])
            blk["time_feasible"] = {"n": len(feas), "mae_s": float(np.abs(p - t).mean()),
                                    "bias_s": float((p - t).mean()), "pearson": pearson(p, t),
                                    "mape_pct": float(100.0 * np.mean(np.abs(p - t) / np.maximum(t, 1e-6))),
                                    "deadline_call_accuracy": float(np.mean((p <= dl) == ot)),
                                    "deadline_base_rate_on_time": float(np.mean(ot))}
            fl = np.array([preds[f"{r['key']}|{r['candidate']}"]["floor_w_pos_kj"] for r in feas])
            blk["floor_violation_rate"] = float(np.mean(tw(feas) < fl))
        per_arena[a] = blk

    all_comp = [r for r in rows if r["compliant"]]
    all_feas = [r for r in rows if r["feasible"]]
    seal = [r for r in all_comp if r["arena"] not in TRAIN_ARENAS]
    coef = {n: float(c) for n, c in zip(names, w_full)}
    fit = {
        "target": "w_pos_kj -- positive mechanical shaft work, kJ (NOT fuel)",
        "inputs": "planned waypoints, commanded speed profile, TerrainMap heights/gradients. No Chrono "
                  "output, no recorded state/action/pose, no learned model.",
        "mirrored_terrain_control": bool(args.mirror_terrain),
        "fitted_on": {"arenas": list(TRAIN_ARENAS), "subset": "compliant runs", "n": len(train_comp)},
        "prediction_scheme": "out-of-arena everywhere: f101-f104 predicted by a leave-one-arena-out fit, "
                             "f105-f111 by the pooled f101-f104 fit",
        "selection": {"protocol": "minimise 0.5*LOAO_MAE/30kJ + 0.5*LOAO_selection_regret/20kJ over the four "
                                  "training arenas; f105-f111 never used", "chosen": chosen, "table": sel_table},
        "work_terms": names,
        "work_coefficients": coef,
        "work_coefficients_nonzero": {k: v for k, v in coef.items() if v > 0},
        "work_coefficients_loao": {a: {n: float(c) for n, c in zip(names, w_loao[a])} for a in TRAIN_ARENAS},
        "within_mission_weight_alpha": alpha,
        "term_units": {"work terms": "kJ at the tyre, nominal Crr=0.01 CdA=3.5 m^2 m=%.0f kg" % MASS_KG,
                       "time/tlow": "s (the 'time' coefficient is a parasitic power in kW)", "one": "1"},
        "implied_physics": {
            "powertrain_factor_on_tractive_work": coef.get("tract"),
            "low_speed_penalty_kJ_per_kJ_at_1mps": (coef.get("tract_lowv", 0.0) * V_SCALE),
            "parasitic_power_kW": coef.get("time"),
            "Crr_over_eta_if_roll_used": coef.get("roll", 0.0) * CRR_NOM or None,
            "CdA_over_eta_m2_if_aero_used": coef.get("aero", 0.0) * CDA_NOM or None,
            "standing_start_overhead_kJ": coef.get("one"),
        },
        "time_terms": tnames,
        "time_coefficients": {n: float(c) for n, c in zip(tnames, tw_full)},
        "time_coefficients_loao": {a: {n: float(c) for n, c in zip(tnames, tw_loao[a])} for a in TRAIN_ARENAS},
        "time_selection_trace": t_trace,
        "time_fitted_on": {"arenas": list(TRAIN_ARENAS), "subset": "feasible runs", "n": len(train_feas)},
        "floor": {"features": FLOOR_NAMES, "w": [float(v) for v in fw_full], "sigma_kj": sig_full,
                  "k_sigma": args.k_sigma,
                  "fitted_on": "compliant runs of f101-f104, target w_pos_kj, ordinary least squares"},
        "per_arena": per_arena,
        "overall": {
            "work_compliant_all_arenas": eval_level(pw(all_comp), tw(all_comp)),
            "work_compliant_f105_f111": eval_level(pw(seal), tw(seal)),
            "work_feasible_all_arenas": eval_level(pw(all_feas), tw(all_feas)),
            "rank_within_layout_all_arenas": eval_rank(all_comp, pw(all_comp)),
            "rank_within_layout_f105_f111": eval_rank(seal, pw(seal)),
        },
    }
    (out / f"arm_analytic{tag}.json").write_text(json.dumps({"predictions": preds, "fit": fit}, indent=1))
    if not args.mirror_terrain:
        (out / "energy_floor_wpos.json").write_text(json.dumps({
            "features": FLOOR_NAMES, "w": [float(v) for v in fw_full], "sigma_kj": sig_full,
            "n_fit": len(train_comp), "target": "w_pos_kj", "fitted_on": list(TRAIN_ARENAS),
            "loao": {a: {"w": [float(v) for v in fw_loao[a]], "sigma_kj": sig_loao[a]} for a in TRAIN_ARENAS},
            "note": "refit of nedm.traverse.energy_floor.route_features against POSITIVE shaft work on the "
                    "f-family arenas. artifacts/traverse/wp5_energy_floor/energy_floor.json (signed work, "
                    "arena_v1) is left untouched; load this one with EnergyFloor.load for the f arenas.",
        }, indent=1))

    print("\nwork terms:", {n: round(float(c), 4) for n, c in zip(names, w_full) if c > 0})
    print("time terms:", {n: round(float(c), 4) for n, c in zip(tnames, tw_full)})
    print(f"\n{'arena':12s} {'n':>5s} {'MAE kJ':>7s} {'bias':>7s} {'MAPE%':>6s} {'r':>6s} {'rho':>6s} {'R2':>6s} "
          f"{'true':>6s} | {'rho_lay':>7s} {'top1':>5s} {'regret':>6s} {'rand':>6s} | {'tMAE':>5s} {'tbias':>6s} {'dl acc':>6s}")
    for a in arenas:
        b = per_arena[a]
        if "work_compliant" not in b:
            continue
        c, rk, tm = b["work_compliant"], b.get("rank_within_layout", {}), b.get("time_feasible", {})
        nan = float("nan")
        print(f"{a:12s} {c['n']:5d} {c['mae_kj']:7.1f} {c['bias_kj']:+7.1f} {c['mape_pct']:6.1f} "
              f"{c['pearson']:6.3f} {c['spearman']:6.3f} {c['r2']:6.3f} {c['true_mean_kj']:6.1f} | "
              f"{rk.get('mean_spearman', nan):7.3f} {rk.get('top1_rate', nan):5.2f} "
              f"{rk.get('mean_regret_kj', nan):6.1f} {rk.get('mean_regret_of_a_random_pick_kj', nan):6.1f} | "
              f"{tm.get('mae_s', nan):5.2f} {tm.get('bias_s', nan):+6.2f} {tm.get('deadline_call_accuracy', nan):6.3f}")
    o = fit["overall"]
    for lab, k in [("ALL compliant", "work_compliant_all_arenas"), ("f105-f111    ", "work_compliant_f105_f111")]:
        e = o[k]
        print(f"\n{lab} n={e['n']:5d}  MAE {e['mae_kj']:.1f} kJ ({e['mape_pct']:.1f}%)  r {e['pearson']:.3f}  "
              f"rho {e['spearman']:.3f}  R2 {e['r2']:.3f}  (mean true work {e['true_mean_kj']:.1f} kJ)")
    for lab, k in [("all arenas", "rank_within_layout_all_arenas"), ("f105-f111 ", "rank_within_layout_f105_f111")]:
        r = o[k]
        print(f"within-layout selection, {lab}: {r['n_layouts']} layouts, rho {r['mean_spearman']:.3f}, "
              f"top-1 {r['top1_rate']:.3f}, regret {r['mean_regret_kj']:.2f} kJ vs {r['mean_regret_of_a_random_pick_kj']:.2f} kJ "
              f"for an average candidate ({r['regret_vs_random_pct']:.0f}%)")
    print(f"\nwrote {out/('arm_analytic'+tag+'.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
