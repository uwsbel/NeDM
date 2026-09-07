#!/usr/bin/env python
"""A1 ground truth: per-candidate mechanical work, mission compliance and the frozen deadline rule.

The traversal study has always scored energy as SIGNED shaft work -- the runner integrates
``engine.GetOutputMotorshaftTorque() * transmission.GetOutputMotorshaftSpeed()`` with its sign at every
physics substep (``traverse_wp3_chrono_eval.py``), so a route that coasts downhill is credited for the
overrun. That is not the quantity an energy claim should rest on, and it is not rank-neutral: on the
cached banks the best candidate changes on a third of layouts when the sign is respected (plan A0).

This rebuilds the truth column for every cached run from the 20 Hz ``power`` series:

* ``w_pos_kj``   = integral of max(P, 0) dt   -- the primary quantity (positive mechanical shaft work)
* ``w_neg_kj``   = integral of min(P, 0) dt   -- reported separately, never netted off
* ``w_signed_kj``= w_pos + w_neg              -- the legacy measure, kept as a secondary column
* ``ke_term_kj`` = 1/2 m v_end^2              -- kinetic energy carried through the goal, so a candidate
  cannot "save" work by arriving fast; ``w_pos_ke_kj`` = w_pos + ke_term is the terminal-matched variant.

The quantity is mechanical work at the engine-transmission motorshaft. It is NOT fuel: the installed
Chrono engine is a torque map with a losses map and has no consumption model. Call it work.

MISSION (frozen before any arm is scored, computed from route geometry and commanded speeds only --
inputs every planner has; no Chrono outcome enters the deadline):

    deadline_s = K * (L_min / V_REF + V_REF / A_ACCEL)

``L_min`` is the shortest route length in the layout's own candidate bank, ``V_REF`` 5 m/s and
``A_ACCEL`` 1.5 m/s^2 (``oracle.PlannerParams``). ``K`` is chosen on the TRAINING arenas f101-f104 as the
smallest value on a declared grid for which at least ``--k-target`` of bank candidates are compliant by
their own profile-implied duration, then frozen and applied everywhere. ``--k`` overrides and pins it.

A candidate counts as MISSION-COMPLIANT when it completed, made no asset contact, kept the declared
clearance, and its Chrono time is within the deadline. Work is only ever compared among compliant
candidates; failures and abstentions are reported as counts, never as cheap energy.

  PYTHONPATH=src python scripts/traverse_wp9_truth.py \
      --caches artifacts/traverse/wp7_cache_v1 artifacts/traverse/wp7_cache_sealed artifacts/traverse/wp8_cache_sealed2 \
      --collect-dirs artifacts/traverse/wp7_collect_f10* artifacts/traverse/wp8_collect_f1* \
      --out artifacts/traverse/wp9_energy
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DT_S = 0.05                # cache record rate (CTRL_DT_S)
MASS_KG = 25000.0 / 9.81   # chassis weight used by the runner's clearance/attitude bookkeeping
V_REF_MPS = 5.0            # declared reference cruise speed for the deadline
A_ACCEL = 1.5              # oracle.PlannerParams.a_accel
K_GRID = [round(1.0 + 0.1 * i, 1) for i in range(11)]  # 1.0 .. 2.0
TRAIN_ARENAS = ("arena_f101", "arena_f102", "arena_f103", "arena_f104")
FEASIBLE_STATUS = "completed"


def profile_time_s(stations: np.ndarray, speeds: np.ndarray) -> float:
    """Duration implied by the commanded speed profile: integral of ds / v."""
    s, v = np.asarray(stations, np.float64), np.maximum(np.asarray(speeds, np.float64), 1e-3)
    ds = np.diff(s)
    v_mid = 0.5 * (v[:-1] + v[1:])
    return float(np.sum(ds / np.maximum(v_mid, 1e-3)))


def read_episode(path: Path) -> dict | None:
    with np.load(path, allow_pickle=True) as d:
        end = int(d["end_frame"])
        p = d["power"].ravel().astype(np.float64)
        z1 = d["z1"]
        n_rec = len(p)
        # end_frame is the frame the route end was reached; -1 means the run never got there.
        # Work is integrated over the DRIVEN window only, so the post-arrival parking tail (which
        # carries only negative shaft power) cannot discount a route that brakes hard at the goal.
        stop = end + 1 if end >= 0 else n_rec
        p = p[:stop]
        if p.size == 0:
            return None
        v_end = float(abs(z1[min(stop, len(z1)) - 1, 0]))
        row = {
            "key": str(d["layout"]),
            "candidate": str(d["candidate"]),
            "arena": str(d["arena"]),
            "status": str(d["status"]),
            "frames": int(stop),
            "time_s": float(stop) * DT_S,
            "w_pos_kj": float(np.maximum(p, 0.0).sum() * DT_S),
            "w_neg_kj": float(np.minimum(p, 0.0).sum() * DT_S),
            "v_end_mps": v_end,
            "ke_term_kj": 0.5 * MASS_KG * v_end**2 / 1000.0,
            "max_contact_n": float(d["max_contact_n"]),
            "route_len_m": float(np.asarray(d["route_stations"])[-1]),
            "profile_time_s": profile_time_s(d["route_stations"], d["route_speeds"]),
            "v_cmd_max": float(np.asarray(d["route_speeds"]).max()),
            "v_cmd_mean": float(np.asarray(d["route_speeds"]).mean()),
        }
        row["w_signed_kj"] = row["w_pos_kj"] + row["w_neg_kj"]
        row["w_pos_ke_kj"] = row["w_pos_kj"] + row["ke_term_kj"]
        return row


def load_collect_rows(dirs: list[str]) -> dict[tuple[str, str], dict]:
    """(key, candidate) -> the collection row, for fields the cache builder drops."""
    out: dict[tuple[str, str], dict] = {}
    for d in dirs:
        p = Path(d) / "rows.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "status" in r:
                out[(r["key"], r["candidate"])] = r
    return out


def choose_k(rows: list[dict], target: float) -> tuple[float, dict]:
    """Smallest K on the grid for which >= target of TRAINING-arena bank candidates comply by their own
    profile-implied duration. Outcome-blind: only route geometry and commanded speeds are used."""
    train = [r for r in rows if r["arena"] in TRAIN_ARENAS]
    diag = {}
    for k in K_GRID:
        ok = [r["profile_time_s"] <= k * r["deadline_base_s"] for r in train]
        frac = float(np.mean(ok)) if ok else 0.0
        diag[str(k)] = round(frac, 4)
        if frac >= target:
            return k, diag
    return K_GRID[-1], diag


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--collect-dirs", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=None, help="pin the deadline slack factor instead of fitting it")
    ap.add_argument("--k-target", type=float, default=0.5, help="fraction of training-arena candidates the deadline must admit")
    ap.add_argument("--min-clearance-m", type=float, default=0.3, help="mission clearance requirement")
    args = ap.parse_args()

    rows: list[dict] = []
    for cache in args.caches:
        files = sorted(Path(cache).glob("*.npz"))
        for f in files:
            r = read_episode(f)
            if r is not None:
                r["cache"] = Path(cache).name
                rows.append(r)
        print(f"{cache}: {len(files)} episodes", flush=True)
    if not rows:
        print("no episodes read", file=sys.stderr)
        return 1

    collect = load_collect_rows([d for pat in args.collect_dirs for d in sorted(glob.glob(pat))])
    n_join = 0
    for r in rows:
        cr = collect.get((r["key"], r["candidate"]))
        if cr is None:
            r["min_clearance_m"] = None
            r["energy_kj_legacy"] = None
            continue
        n_join += 1
        r["min_clearance_m"] = cr.get("min_clearance_m")
        r["energy_kj_legacy"] = cr.get("energy_kj")

    # ---- mission: deadline base from the layout's own bank geometry, then the frozen K
    by_layout: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_layout[r["key"]].append(r)
    for key, group in by_layout.items():
        l_min = min(r["route_len_m"] for r in group)
        base = l_min / V_REF_MPS + V_REF_MPS / A_ACCEL
        for r in group:
            r["deadline_base_s"] = base
    k, k_diag = (args.k, {"pinned": args.k}) if args.k is not None else choose_k(rows, args.k_target)
    for r in rows:
        r["deadline_s"] = k * r["deadline_base_s"]
        clear_ok = r["min_clearance_m"] is None or r["min_clearance_m"] >= args.min_clearance_m
        r["feasible"] = r["status"] == FEASIBLE_STATUS and r["max_contact_n"] <= 1.0 and clear_ok
        r["on_time"] = r["time_s"] <= r["deadline_s"]
        r["compliant"] = bool(r["feasible"] and r["on_time"])

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "truth.json").write_text(json.dumps(rows))

    # ---- report
    arenas = sorted({r["arena"] for r in rows})
    n_lay = len(by_layout)
    comp = [r for r in rows if r["compliant"]]
    feas = [r for r in rows if r["feasible"]]
    lay_with = sum(1 for g in by_layout.values() if any(r["compliant"] for r in g))
    summary = {
        "n_runs": len(rows), "n_layouts": n_lay, "arenas": arenas,
        "joined_collect_rows": n_join,
        "deadline": {"K": k, "V_REF_MPS": V_REF_MPS, "A_ACCEL": A_ACCEL, "rule": "K*(L_min/V_REF + V_REF/A_ACCEL)",
                     "fit_on": list(TRAIN_ARENAS), "k_target": args.k_target, "grid_compliance": k_diag},
        "min_clearance_m": args.min_clearance_m,
        "n_feasible": len(feas), "n_compliant": len(comp),
        "layouts_with_a_compliant_candidate": lay_with,
        "work_kj": {
            "all":       {"w_pos": float(np.mean([r["w_pos_kj"] for r in rows])), "w_neg": float(np.mean([r["w_neg_kj"] for r in rows])), "w_signed": float(np.mean([r["w_signed_kj"] for r in rows]))},
            "compliant": {"w_pos": float(np.mean([r["w_pos_kj"] for r in comp])), "w_neg": float(np.mean([r["w_neg_kj"] for r in comp])), "w_signed": float(np.mean([r["w_signed_kj"] for r in comp]))} if comp else None,
        },
        "terminal_speed_mps_compliant": {
            "mean": float(np.mean([r["v_end_mps"] for r in comp])), "p95": float(np.quantile([r["v_end_mps"] for r in comp], 0.95)),
            "ke_over_wpos_median": float(np.median([r["ke_term_kj"] / max(r["w_pos_kj"], 1e-6) for r in comp])),
        } if comp else None,
    }

    # how often the metric choice changes the winner, among compliant candidates
    flips = {"w_pos_vs_signed": 0, "w_pos_ke_vs_w_pos": 0, "n": 0}
    for g in by_layout.values():
        c = [r for r in g if r["compliant"]]
        if len(c) < 3:
            continue
        flips["n"] += 1
        wp = np.array([r["w_pos_kj"] for r in c]); ws = np.array([r["w_signed_kj"] for r in c]); wk = np.array([r["w_pos_ke_kj"] for r in c])
        flips["w_pos_vs_signed"] += int(np.argmin(wp) != np.argmin(ws))
        flips["w_pos_ke_vs_w_pos"] += int(np.argmin(wk) != np.argmin(wp))
    summary["metric_flips_among_compliant"] = flips
    (out / "truth_summary.json").write_text(json.dumps(summary, indent=1))

    print(json.dumps(summary, indent=1))
    print(f"\nwrote {out/'truth.json'} ({len(rows)} runs, {n_lay} layouts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
