"""Quality and diversity report for a packed pool campaign (JSON + figures).

Quality: rejection reasons, penetration, height above the cloth, vertical
speed, energy (never rising between contacts), B at rest before the A-B hit,
A-B hit count. Diversity: action-grid coverage, B departure direction, B and
A cushion hits by the target time, spin state of A at impact (sliding vs
rolling), B's position at the target time (the reachable set).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from nedm.pool_ball.evaluate import cushion_hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    root = args.data
    index = json.loads((root / "campaign_index.json").read_text())
    config, campaign = index["config"], index["campaign"]
    scene = config["scene"]
    R = scene["radius_m"]
    with np.load(root / "model_data.npz") as packet:
        states, contacts, heights = packet["states"], packet["contacts"], packet["heights"]
        polar, splits = packet["polar"], packet["splits"]
    record = index["record_step_s"]
    target = round(campaign["target_time_s"] / record)
    episodes = index["episodes"]
    worst = {k: [e["worst"][k] for e in episodes] for k in episodes[0]["worst"]}
    b_cushions = np.array([cushion_hits(s[::10, 7:], scene, target // 10) for s in states])
    a_cushions = np.array([cushion_hits(s[::10, :7], scene, target // 10) for s in states])
    first_ab = np.array([e["first_ab_time_s"] for e in episodes])
    ab_steps = np.round(first_ab / record).astype(int)
    a_at_impact = states[np.arange(len(states)), ab_steps - 1, :7]
    slip = np.hypot(a_at_impact[:, 2] - R * a_at_impact[:, 5], a_at_impact[:, 3] + R * a_at_impact[:, 4])
    b_after = states[np.arange(len(states)), np.minimum(ab_steps + 5, states.shape[1] - 1), 7:]
    direction = np.degrees(np.arctan2(b_after[:, 3], b_after[:, 2]))
    b_target = states[:, target, 7:9]
    report = {
        "episodes": len(episodes), "attempts": index["attempt_count"], "rejections": index["rejections"],
        "split_counts": index["split_counts"], "cells_short_of_quota": len(index["cells_short_of_quota"]),
        "quality": {k: {"max": float(np.max(v)), "p99": float(np.quantile(v, 0.99))} for k, v in worst.items()},
        "max_height_above_cloth_m": float(np.abs(heights[..., 0] - R).max()),
        "ab_hits": dict(Counter(e["ab_contacts"] for e in episodes)),
        "first_ab_time_s": {"min": float(first_ab.min()), "max": float(first_ab.max())},
        "b_cushion_hits_by_target": dict(Counter(int(x) for x in b_cushions)),
        "a_cushion_hits_by_target": dict(Counter(int(x) for x in a_cushions)),
        "a_slip_at_impact_mps": {"rolling_fraction(<0.01)": float((slip < 0.01).mean()), "min": float(slip.min()),
                                 "median": float(np.median(slip)), "max": float(slip.max())},
        "b_departure_direction_deg": {"min": float(direction.min()), "max": float(direction.max()),
                                      "histogram_10deg": np.histogram(direction, bins=np.arange(-90, 91, 10))[0].tolist()},
        "b_at_target": {"x_range": [float(b_target[:, 0].min()), float(b_target[:, 0].max())],
                        "y_range": [float(b_target[:, 1].min()), float(b_target[:, 1].max())]},
        "contact_windows_per_episode": {k: float(v) for k, v in zip(["AB", "A_cushion", "B_cushion"],
                                        [contacts[..., 0].any(-1).mean(), contacts[..., 1:5].sum((1, 2)).mean(), contacts[..., 5:9].sum((1, 2)).mean()])},
    }
    (root / "data_report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; figures skipped")
        return
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes[0, 0]
    ax.scatter(polar[:, 0], np.degrees(polar[:, 1]), s=1, c=splits, cmap="viridis")
    ax.set(xlabel="launch speed (m/s)", ylabel="aim from A->B line (deg)", title="action coverage (colour = split)")
    ax = axes[0, 1]
    ax.hist(direction, bins=72)
    ax.set(xlabel="B departure direction (deg)", ylabel="episodes", title="B departure direction")
    ax = axes[0, 2]
    hits = sorted(set(b_cushions) | set(a_cushions))
    ax.bar(np.array(hits) - 0.2, [np.sum(b_cushions == h) for h in hits], 0.4, label="B")
    ax.bar(np.array(hits) + 0.2, [np.sum(a_cushions == h) for h in hits], 0.4, label="A")
    ax.set(xlabel=f"cushion hits by t={campaign['target_time_s']} s", ylabel="episodes", title="cushion hits")
    ax.legend()
    ax = axes[1, 0]
    sc = ax.scatter(b_target[:, 0], b_target[:, 1], s=1, c=b_cushions, cmap="plasma")
    hx, hy = scene["half_length_m"], scene["half_width_m"]
    ax.plot([-hx, hx, hx, -hx, -hx], [-hy, -hy, hy, hy, -hy], "k-")
    ax.plot(*scene["ball_a_xy_m"], "wo", mec="k")
    ax.plot(*scene["ball_b_xy_m"], "ro")
    ax.set_aspect("equal")
    ax.set(title=f"B at t={campaign['target_time_s']} s (colour = B cushion hits)")
    plt.colorbar(sc, ax=ax, fraction=0.03)
    ax = axes[1, 1]
    ax.hist(slip, bins=60)
    ax.set(xlabel="A slip speed at impact (m/s); 0 = rolling", ylabel="episodes", title="A spin state at impact")
    ax = axes[1, 2]
    for k in np.linspace(0, len(states) - 1, 40).astype(int):
        ax.plot(states[k, : target + 1, 7], states[k, : target + 1, 8], lw=0.6)
        ax.plot(states[k, : target + 1, 0], states[k, : target + 1, 1], lw=0.4, ls=":", color="gray")
    ax.plot([-hx, hx, hx, -hx, -hx], [-hy, -hy, hy, hy, -hy], "k-")
    ax.set_aspect("equal")
    ax.set(title="40 sample shots: B solid, A dotted")
    fig.tight_layout()
    fig.savefig(root / "data_report.png", dpi=110)
    print(f"wrote {root / 'data_report.png'}")


if __name__ == "__main__":
    main()
