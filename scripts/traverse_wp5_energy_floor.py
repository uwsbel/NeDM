#!/usr/bin/env python
"""Route-geometry energy model from the Chrono-driven routes (a floor against the optimiser's curse).

Fits Chrono energy ~ [length, sum of v^2 along the speed profile weighted by arc length, positive climb,
number of speed ramps] on every route the tracker has driven in Chrono, reports R^2 and the residual
spread, and writes coefficients the planner can use as a *lower bound* on imagined energy:
``floor = fit - k * sigma``. A sampled route whose imagined energy falls far below what any route of
that length / speed / climb has ever cost in Chrono is the model being exploited, not a bargain.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse.terrain import TerrainMap


from nedm.traverse.energy_floor import route_features  # shared with the planner


FEATURES = ["length_m", "v2_length/100", "climb_m", "v_peak^2/10", "accel^2/10", "one"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", nargs="+", default=[
        "artifacts/traverse/wp5_chrono_sample_planner", "artifacts/traverse/wp5_chrono_sample_planner_v2",
        "artifacts/traverse/wp5_chrono_sample_planner_v3", "artifacts/traverse/wp5_chrono_sample_planner_v4_pt",
        "artifacts/traverse/wp4_chrono_allsensor", "artifacts/traverse/wp4_chrono_loc_camera_pred_occ",
        "artifacts/traverse/wp4_chrono_pred_full_r40", "artifacts/traverse/wp4_chrono_fallback"])
    ap.add_argument("--arena", default="assets/traverse/arena_v1")
    ap.add_argument("--out", default="artifacts/traverse/wp5_energy_floor/energy_floor.json")
    ap.add_argument("--holdout", default="wp5_chrono_sample_planner_v4_pt", help="batch held out of the fit")
    args = ap.parse_args()
    tmap = TerrainMap.from_dir(Path(args.arena))
    X, y, names, batch = [], [], [], []
    for b in args.batches:
        b = Path(b)
        a = json.loads((b / "summary.json").read_text())["args"]
        if not a.get("route_file"):
            continue
        routes = json.loads(Path(a["route_file"]).read_text())
        for line in (b / "rows.jsonl").read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "tracker" not in r["controller"] or not r["completed"]:
                continue
            m = [c for c in routes.get(r["key"], []) if c["candidate"] == r["candidate"]]
            if not m:
                continue
            X.append(route_features(m[0]["waypoints"], m[0]["speeds"], m[0]["stations"], tmap)); y.append(r["energy_kj"]); names.append((r["key"], r["candidate"])); batch.append(b.name)
    X, y, batch = np.array(X), np.array(y), np.array(batch)
    fit = batch != args.holdout
    w, *_ = np.linalg.lstsq(X[fit], y[fit], rcond=None)
    pred = X @ w
    res = y - pred
    r2 = lambda m: 1 - ((res[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()
    sigma = float(res[fit].std())
    print(f"{len(y)} routes ({fit.sum()} fit, {(~fit).sum()} held out: {args.holdout})")
    print("coefficients:", {n: round(float(c), 3) for n, c in zip(FEATURES, w)})
    print(f"fit R^2 {r2(fit):.3f}  sigma {sigma:.1f} kJ | held-out R^2 {r2(~fit):.3f}  bias {res[~fit].mean():+.1f} kJ  MAE {np.abs(res[~fit]).mean():.1f} kJ")
    for b in np.unique(batch):
        m = batch == b
        print(f"  {b:34s} n {m.sum():3d} Chrono {y[m].mean():6.1f} fit {pred[m].mean():6.1f} MAE {np.abs(res[m]).mean():5.1f} corr {np.corrcoef(pred[m], y[m])[0, 1]:.3f}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"features": FEATURES, "w": w.tolist(), "sigma_kj": sigma, "n_fit": int(fit.sum()),
                               "r2_fit": float(r2(fit)), "r2_holdout": float(r2(~fit))}, indent=1))


if __name__ == "__main__":
    main()
