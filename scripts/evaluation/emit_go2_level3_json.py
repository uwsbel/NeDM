"""Machine-readable level-3 blocks, for a status page that plots rather than reads.

FACTS ONLY. Every field is copied or recomputed from the eval JSON and the
verdict comes from report_go2_level3.py, not from this script and not from
whoever is writing the prose. If this file ever disagrees with the report, the
report is right.

Emits one block per (checkpoint, arm), plus trajectories for the best and worst
references on the primary arm -- three world-frame paths each, so a figure can
show the reference, the open-loop replay drifting off it, and whether the policy
pulled back. ~8 KB of numbers, and it shows what a camera cannot.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def decimate(points: list[list[float]], target: int) -> list[list[float]]:
    if len(points) <= target:
        return [[round(v, 4) for v in p] for p in points]
    idx = np.linspace(0, len(points) - 1, target).round().astype(int)
    return [[round(v, 4) for v in points[i]] for i in idx]


def reference_paths(reference_path: str, indices: list[int], start: int, span: int) -> dict[int, float]:
    """Path length of each reference over the evaluated window, for the blocks."""
    from nedm.rl.references import load_reference_set

    rs = load_reference_set(reference_path)
    out = {}
    for i in indices:
        p = rs.poses[i, start:start + span, :2]
        out[i] = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    return out


def block(path: Path, checkpoint: str, arm: str, verdict_json: Path | None) -> dict[str, Any]:
    data = json.loads(path.read_text())
    floor = {r["reference_id"]: r for r in data["replay_baseline"]["per_reference"]}
    ids = [r["reference_id"] for r in data["policy"]["per_reference"]]
    steps = data["policy"]["per_reference"][0]["steps"]
    try:
        paths = reference_paths(data["reference_path"], ids, 127, steps * 5)
    except Exception:
        paths = {}

    rows = []
    for r in data["policy"]["per_reference"]:
        b = floor[r["reference_id"]]
        rows.append({
            "family": r["scenario_family"].removesuffix("_command").split("_", 2)[2],
            "floor_m": round(b["mean_position_error_m"], 5),
            "policy_m": round(r["mean_position_error_m"], 5),
            "ratio": round(r["mean_position_error_m"] / b["mean_position_error_m"], 4),
            "diff_m": round(r["mean_position_error_m"] - b["mean_position_error_m"], 5),
            "path_m": round(paths.get(r["reference_id"], float("nan")), 4),
        })

    ratios = [r["ratio"] for r in rows]
    diffs = [r["diff_m"] for r in rows]
    out: dict[str, Any] = {
        "checkpoint": checkpoint,
        "arm": arm,
        "horizon_s": round(steps * data["policy"]["per_reference"][0].get("dt_s", 0.05), 2)
        if "dt_s" in data["policy"]["per_reference"][0] else round(steps * 0.05, 2),
        "per_reference": rows,
        "median_ratio": round(float(np.median(ratios)), 4),
        "count_better": int(sum(1 for r in ratios if r < 1.0)),
        "count_total": len(rows),
        "median_paired_diff_m": round(float(np.median(diffs)), 5),
        "pooled_floor_m": round(data["replay_baseline"]["mean_position_error_m"], 5),
        "pooled_policy_m": round(data["policy"]["mean_position_error_m"], 5),
    }
    # The verdict and the structure come from the report, never recomputed here.
    if verdict_json and verdict_json.is_file():
        v = json.loads(verdict_json.read_text())
        key = {"primary_6s_random_train": "primary",
               "bracket_6s_least_moving": "least_moving",
               "generalisation_6s_val": "generalisation"}.get(arm)
        node = v.get(key) if key else None
        if node:
            st = node.get("structure", {})
            out.update({
                "verdict": node["verdict"],
                "slope": round(st.get("slope", float("nan")), 4),
                "intercept_m": round(st.get("intercept", float("nan")), 5),
                "loo_min": round(st.get("loo_min", float("nan")), 4),
                "loo_max": round(st.get("loo_max", float("nan")), 4),
            })
        out["checklist_all_pass"] = bool(v.get("checklist_passed"))
    return out


def trajectories(path: Path, n: int = 120) -> list[dict[str, Any]]:
    """Best and worst reference by ratio, three world-frame paths each."""
    data = json.loads(path.read_text())
    floor = {r["reference_id"]: r for r in data["replay_baseline"]["per_reference"]}
    scored = sorted(
        data["policy"]["per_reference"],
        key=lambda r: r["mean_position_error_m"] / floor[r["reference_id"]]["mean_position_error_m"])
    out = []
    for label, r in (("best", scored[0]), ("worst", scored[-1])):
        b = floor[r["reference_id"]]
        if not r.get("path_xy"):
            continue
        out.append({
            "which": label,
            "family": r["scenario_family"].removesuffix("_command").split("_", 2)[2],
            "episode_id": r["episode_id"],
            "ratio": round(r["mean_position_error_m"] / b["mean_position_error_m"], 4),
            "dt_s": round(float(r.get("dt_s", 0.05)), 4),
            "ref_xy": decimate(r["ref_xy"], n),
            "policy_xy": decimate(r["path_xy"], n),
            "replay_xy": decimate(b.get("path_xy") or [], n),
        })
    return out


def ppo_curve(log: Path, n: int = 100) -> dict[str, Any]:
    import re

    text = log.read_text(errors="replace")
    recs, cur = [], {}
    for line in text.splitlines():
        m = re.search(r"Learning iteration (\d+)/", line)
        if m:
            if len(cur) == 3:
                recs.append(cur)
            cur = {"it": int(m.group(1))}
            continue
        for key, pat in (("pos", r"/tracking/position_error_m:\s+([\d.]+)"),
                         ("noise", r"Mean action noise std:\s+([\d.]+)")):
            mm = re.search(pat, line)
            if mm and key not in cur:
                cur[key] = float(mm.group(1))
    if len(cur) == 3:
        recs.append(cur)
    if not recs:
        return {}
    idx = np.linspace(0, len(recs) - 1, min(n, len(recs))).round().astype(int)
    best = min(recs, key=lambda r: r["pos"])
    return {
        "series": [[recs[i]["it"], round(recs[i]["pos"], 5), round(recs[i]["noise"], 4)] for i in idx],
        "columns": ["iteration", "position_error_m", "action_noise_std"],
        "best_iteration": best["it"], "best_position_error_m": round(best["pos"], 5),
        "final_iteration": recs[-1]["it"], "final_position_error_m": round(recs[-1]["pos"], 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="append", required=True, metavar="CKPT:ARM:PATH")
    ap.add_argument("--verdict", type=Path, default=None)
    ap.add_argument("--trajectories-from", type=Path, default=None)
    ap.add_argument("--ppo-log", type=Path, default=None)
    a = ap.parse_args()

    blocks = []
    for spec in a.eval:
        ckpt, arm, path = spec.split(":", 2)
        blocks.append(block(Path(path), ckpt, arm, a.verdict))
    doc: dict[str, Any] = {"blocks": blocks}
    if a.trajectories_from:
        doc["trajectories"] = trajectories(a.trajectories_from)
    if a.ppo_log:
        doc["ppo_curve"] = ppo_curve(a.ppo_log)
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
