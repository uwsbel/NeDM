#!/usr/bin/env python3
"""Build the rigid/CRM/bumpy Chrono comparison table+plot for the OFAT L8 policies.

Reads the per-terrain chrono_eval summary.json files for the three
crm2000mix{00,25,100}_onehot_ofatL8 RL runs (dynamics backbone = the L8-depth
OFAT arm, best_val checkpoint) and writes CSV/JSON/PNG/PDF into
artifacts/rl_runs/chrono_eval_comparisons/, replacing the prior 6-layer
model_500 comparison.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path("/home/harry/NeDM")
OUT_DIR = REPO_ROOT / "artifacts/rl_runs/chrono_eval_comparisons"
OUT_STEM = "onehot_policy_3x3_chrono_xy_rmse_median_iqr_steerlim010_ofatL8_model1000"

RUN_DIRS = {
    "rigid_only": REPO_ROOT
    / "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix00_onehot_ofatL8_bestval61_rigid20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010_1000it",
    "mixture": REPO_ROOT
    / "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix25_onehot_ofatL8_bestval51_flat20crm20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010"
    / "hmmwv_rl_15d_crm2000mix25_onehot_ofatL8_bestval51_flat20crm20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010",
    "crm_only": REPO_ROOT
    / "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix100_onehot_ofatL8_bestval54_crmonly20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010_1000it",
}

# eval sub-dir name per (policy, terrain); the mixture run trained to 1800 iters
# but was checkpoint-evaluated at 1000 to match the two specialists' final iter.
EVAL_SUBDIRS = {
    ("rigid_only", "rigid_flat"): "chrono_eval_tracking_model_999_rigid_val20_rest_start_steerlim010_pre0",
    ("rigid_only", "crm"): "chrono_crm_eval_model_999_val20_rest_start_min10_steerlim010_pre0",
    ("rigid_only", "bumpy"): "chrono_bumpy_eval_model_999_val20_rest_start_steerlim010_pre0",
    ("crm_only", "rigid_flat"): "chrono_eval_tracking_model_999_rigid_val20_rest_start_steerlim010_pre0",
    ("crm_only", "crm"): "chrono_crm_eval_model_999_val20_rest_start_min10_steerlim010_pre0",
    ("crm_only", "bumpy"): "chrono_bumpy_eval_model_999_val20_rest_start_steerlim010_pre0",
    ("mixture", "rigid_flat"): "chrono_eval_tracking_model_1000_rigid_val_rest_start_steerlim010_pre0",
    ("mixture", "crm"): "chrono_crm_eval_model_1000_val20_rest_start_min10_steerlim010_pre0",
    ("mixture", "bumpy"): "chrono_bumpy_eval_model_1000_val20_rest_start_steerlim010_pre0",
}

POLICY_LABELS = {"mixture": "Generalist (mix25)", "rigid_only": "Rigid only", "crm_only": "CRM only"}
POLICY_ORDER = ["mixture", "rigid_only", "crm_only"]
TERRAIN_ORDER = ["rigid_flat", "crm", "bumpy"]
TERRAIN_TITLES = {"rigid_flat": "Rigid flat", "crm": "CRM (deformable soil)", "bumpy": "Bumpy rigid"}

# dataviz categorical palette (fixed order): slot1 blue, slot2 aqua, slot3 yellow
COLORS_LIGHT = {"mixture": "#2a78d6", "rigid_only": "#1baf7a", "crm_only": "#eda100"}


def load_stats(policy: str, terrain: str) -> dict:
    subdir = EVAL_SUBDIRS[(policy, terrain)]
    path = RUN_DIRS[policy] / subdir / "summary.json"
    d = json.loads(path.read_text())
    xy = np.array([r["xy_rmse_m"] for r in d["rollouts"]], dtype=float)
    steps = np.array([r["steps"] for r in d["rollouts"]], dtype=float)
    max_steps = steps.max()
    n_total = len(xy)
    n_early = int((steps < max_steps).sum())
    n_success = n_total - n_early
    q1, median, q3 = np.percentile(xy, [25, 50, 75])
    return {
        "summary_path": str(path),
        "steering_rate_limit": d.get("steering_rate_limit"),
        "n_total": n_total,
        "n_success": n_success,
        "n_early": n_early,
        "mean_xy_rmse_m": float(xy.mean()),
        "q1_xy_rmse_m": float(q1),
        "median_xy_rmse_m": float(median),
        "q3_xy_rmse_m": float(q3),
        "label": POLICY_LABELS[policy],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    stats = {t: {p: load_stats(p, t) for p in POLICY_ORDER} for t in TERRAIN_ORDER}

    # ---- CSV ----
    csv_lines = [
        "terrain,policy,n_total,n_success,n_early,mean_xy_rmse_m,median_xy_rmse_m,"
        "q1_xy_rmse_m,q3_xy_rmse_m,summary_path"
    ]
    for t in TERRAIN_ORDER:
        for p in POLICY_ORDER:
            s = stats[t][p]
            csv_lines.append(
                f"{t},{p},{s['n_total']},{s['n_success']},{s['n_early']},"
                f"{s['mean_xy_rmse_m']:.9f},{s['median_xy_rmse_m']:.9f},"
                f"{s['q1_xy_rmse_m']:.9f},{s['q3_xy_rmse_m']:.9f},{s['summary_path']}"
            )
    (OUT_DIR / f"{OUT_STEM}.csv").write_text("\n".join(csv_lines) + "\n")

    # ---- JSON ----
    out_json = {
        "description": (
            "Chrono HMMWV policy comparison for the L8-depth OFAT dynamics backbone "
            "(ablation_ofat/L8_H8_E256_ctx128{,_mix00,_mix100}, best_val checkpoints "
            "epoch 51/61/54). RL policies trained steering_rate_limit=0.1 for 1000 "
            "PPO iterations (mixture run continued training to 1800 but is evaluated "
            "at the matching iteration-1000 checkpoint); all three eval terrains "
            "apply steering_rate_limit=0.1."
        ),
        "policies": POLICY_ORDER,
        "policy_labels": POLICY_LABELS,
        "terrains": TERRAIN_ORDER,
        "stats": stats,
    }
    (OUT_DIR / f"{OUT_STEM}.json").write_text(json.dumps(out_json, indent=2) + "\n")

    # ---- PNG/PDF plot ----
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    n_terrain = len(TERRAIN_ORDER)
    n_policy = len(POLICY_ORDER)
    group_width = 0.72
    bar_width = group_width / n_policy
    x = np.arange(n_terrain)

    for i, p in enumerate(POLICY_ORDER):
        offsets = x - group_width / 2 + bar_width * (i + 0.5)
        medians = [stats[t][p]["median_xy_rmse_m"] for t in TERRAIN_ORDER]
        q1s = np.array([stats[t][p]["q1_xy_rmse_m"] for t in TERRAIN_ORDER])
        q3s = np.array([stats[t][p]["q3_xy_rmse_m"] for t in TERRAIN_ORDER])
        yerr = np.vstack([np.array(medians) - q1s, q3s - np.array(medians)])
        ax.bar(
            offsets,
            medians,
            width=bar_width * 0.88,
            color=COLORS_LIGHT[p],
            label=POLICY_LABELS[p],
            zorder=3,
        )
        ax.errorbar(
            offsets,
            medians,
            yerr=yerr,
            fmt="none",
            ecolor="#0b0b0b",
            elinewidth=1.2,
            capsize=3,
            zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TERRAIN_TITLES[t] for t in TERRAIN_ORDER], color="#0b0b0b")
    ax.set_ylabel("Median XY RMSE (m), IQR error bars", color="#0b0b0b")
    ax.set_title(
        "OFAT L8 dynamics backbone: Chrono tracking, median XY RMSE by policy \xd7 terrain\n"
        "(steering_rate_limit=0.1, 20 rollouts/cell)",
        color="#0b0b0b",
        fontsize=11,
    )
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{OUT_STEM}.png", facecolor=fig.get_facecolor())
    fig.savefig(OUT_DIR / f"{OUT_STEM}.pdf", facecolor=fig.get_facecolor())
    plt.close(fig)

    print("Wrote:")
    for ext in ("csv", "json", "png", "pdf"):
        print(" ", OUT_DIR / f"{OUT_STEM}.{ext}")

    print("\nSummary table (median [IQR] XY RMSE, m):")
    header = "terrain".ljust(12) + "".join(POLICY_LABELS[p].ljust(20) for p in POLICY_ORDER)
    print(header)
    for t in TERRAIN_ORDER:
        row = TERRAIN_TITLES[t].ljust(12)
        for p in POLICY_ORDER:
            s = stats[t][p]
            cell = f"{s['median_xy_rmse_m']:.3f} [{s['q1_xy_rmse_m']:.3f}-{s['q3_xy_rmse_m']:.3f}]"
            row += cell.ljust(20)
        print(row)


if __name__ == "__main__":
    main()
