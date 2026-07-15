#!/usr/bin/env python3
"""Single combined closed-loop XY-trajectory figure (L8 backbone), all 9 evals.

3x3 grid: rows = terrain (rigid flat, CRM soil, rigid bumpy), columns = the
same three representative held-out maneuvers used per terrain (sustained-turn
loop, multi-steer zigzag, sine-steer wave). Each cell is the XY path only
(reference + all three policies), with per-cell color-coded XY RMSE labels, so
all 9 evaluations from plot_l8_policy_trajectories.py survive in one compact,
one-page figure instead of three separate one-eval-per-row pages.

Usage:
    python plot_l8_policy_trajectories_grid.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]

RUNS = {
    "mix00": {
        "rigid": REPO_ROOT
        / "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix00_onehot_ofatL8_bestval61_rigid20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010_1000it"
        / "chrono_eval_tracking_model_999_rigid_val20_rest_start_steerlim010_pre0",
        "bumpy": REPO_ROOT
        / "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix00_onehot_ofatL8_bestval61_rigid20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010_1000it"
        / "chrono_bumpy_eval_model_999_val20_rest_start_steerlim010_pre0",
        "crm": REPO_ROOT
        / "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix00_onehot_ofatL8_bestval61_rigid20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010_1000it"
        / "chrono_crm_eval_model_999_val20_rest_start_min10_steerlim010_pre0",
    },
    "mix25": {
        "rigid": REPO_ROOT
        / "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix25_onehot_ofatL8_bestval51_flat20crm20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010"
        / "chrono_eval_tracking_model_1000_rigid_val_rest_start_steerlim010_pre0",
        "bumpy": REPO_ROOT
        / "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix25_onehot_ofatL8_bestval51_flat20crm20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010"
        / "chrono_bumpy_eval_model_1000_val20_rest_start_steerlim010_pre0",
        "crm": REPO_ROOT
        / "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix25_onehot_ofatL8_bestval51_flat20crm20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010"
        / "chrono_crm_eval_model_1000_val20_rest_start_min10_steerlim010_pre0",
    },
    "mix100": {
        "rigid": REPO_ROOT
        / "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix100_onehot_ofatL8_bestval54_crmonly20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010_1000it"
        / "chrono_eval_tracking_model_999_rigid_val20_rest_start_steerlim010_pre0",
        "bumpy": REPO_ROOT
        / "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix100_onehot_ofatL8_bestval54_crmonly20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010_1000it"
        / "chrono_bumpy_eval_model_999_val20_rest_start_steerlim010_pre0",
        "crm": REPO_ROOT
        / "artifacts/rl_runs/hmmwv_rl_15d_crm2000mix100_onehot_ofatL8_bestval54_crmonly20_K16_64steps_ar02_state_vxvyyr_pos2_yaw2_steerlim010_1000it"
        / "chrono_crm_eval_model_999_val20_rest_start_min10_steerlim010_pre0",
    },
}

POLICIES = ["mix25", "mix00", "mix100"]
POLICY_LABELS = {
    "mix25": "Generalist (mix25)",
    "mix00": "Rigid only",
    "mix100": "CRM only",
}
POLICY_SHORT = {"mix25": "G", "mix00": "R", "mix100": "C"}
POLICY_COLORS = {
    "mix25": "#2a78d6",
    "mix00": "#eb6834",
    "mix100": "#008300",
}
REF_COLOR = "#333333"

TERRAIN_ROWS = [
    ("rigid", "(a) Rigid flat"),
    ("crm", "(b) CRM soil"),
    ("bumpy", "(c) Rigid bumpy (zero-shot)"),
]

# Same three maneuver families, hand-picked per terrain (episode idx differs
# per terrain's own held-out set), matching plot_l8_policy_trajectories.py.
SELECTIONS = {
    "rigid": [(12, "Sustained turn (loop)"), (3, "Multi-steer (zigzag)"), (19, "Sine-steer (wave)")],
    "crm": [(18, "Sustained turn (loop)"), (3, "Multi-steer (zigzag)"), (1, "Sine-steer (wave)")],
    "bumpy": [(18, "Sustained turn (loop)"), (9, "Multi-steer (zigzag)"), (1, "Sine-steer (wave)")],
}

STEP_DT_S = 0.05

LABEL_FONTSIZE = 12
TICK_FONTSIZE = 9
LEGEND_FONTSIZE = 13
ROW_TITLE_FONTSIZE = 13
COL_TITLE_FONTSIZE = 13
RMSE_FONTSIZE = 8.5
REF_LINEWIDTH = 2.6
POLICY_LINEWIDTH = 2.0


def load_episode(run_dir: Path, idx: int) -> dict:
    npz = np.load(run_dir / f"chrono_tracking_{idx:02d}.npz")
    summary = json.loads((run_dir / "summary.json").read_text())
    rollout = summary["rollouts"][idx]
    return {
        "pose": npz["pose"],
        "ref_pose": npz["ref_pose"],
        "reference": rollout["reference"],
        "xy_rmse_m": rollout["xy_rmse_m"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/analysis/manuscript_figs"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(12.6, 12.6))

    manifest_rows = []
    for row, (terrain, row_title) in enumerate(TERRAIN_ROWS):
        for col, (idx, maneuver_tag) in enumerate(SELECTIONS[terrain]):
            ax = axes[row, col]
            episodes = {pol: load_episode(RUNS[pol][terrain], idx) for pol in POLICIES}
            ref_pose = episodes[POLICIES[0]]["ref_pose"]
            ref_name = episodes[POLICIES[0]]["reference"]

            ax.plot(ref_pose[:, 0], ref_pose[:, 1], color=REF_COLOR, ls="--",
                     lw=REF_LINEWIDTH, label="Reference", zorder=1)

            row_rmse = {}
            for pol in POLICIES:
                pose = episodes[pol]["pose"]
                ax.plot(pose[:, 0], pose[:, 1], color=POLICY_COLORS[pol],
                         lw=POLICY_LINEWIDTH, label=POLICY_LABELS[pol], zorder=2)
                row_rmse[pol] = episodes[pol]["xy_rmse_m"]

            ax.plot(*ref_pose[0, :2], marker="o", ms=7, mfc="white", mec="black", mew=1.3, zorder=3)
            ax.plot(*ref_pose[-1, :2], marker="s", ms=6.5, mfc="black", mec="black", zorder=3)
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=TICK_FONTSIZE)

            rmse_text = "  ".join(
                f"{POLICY_SHORT[pol]}={row_rmse[pol]:.2f}" for pol in POLICIES
            )
            ax.text(
                0.03, 0.03, rmse_text, transform=ax.transAxes, fontsize=RMSE_FONTSIZE,
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
            )

            if col == 0:
                ax.set_ylabel("y [m]", fontsize=LABEL_FONTSIZE)
            if row == len(TERRAIN_ROWS) - 1:
                ax.set_xlabel("x [m]", fontsize=LABEL_FONTSIZE)
            if row == 0:
                ax.set_title(maneuver_tag, fontsize=COL_TITLE_FONTSIZE, pad=8)
            if col == 0:
                ax.text(
                    -0.34, 1.12, row_title, transform=ax.transAxes,
                    fontsize=ROW_TITLE_FONTSIZE, fontweight="bold",
                    ha="left", va="bottom",
                )

            manifest_rows.append({
                "terrain": terrain, "episode_idx": idx, "maneuver_tag": maneuver_tag,
                "reference": ref_name, "xy_rmse_m": row_rmse,
            })

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False,
        fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.95))
    fig.subplots_adjust(hspace=0.45, wspace=0.32)

    png_path = out_dir / "hmmwv_policy_trajectories_L8_grid.png"
    pdf_path = out_dir / "hmmwv_policy_trajectories_L8_grid.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")

    manifest = {
        "policies": POLICY_LABELS, "policy_short": POLICY_SHORT, "step_dt_s": STEP_DT_S,
        "rows": manifest_rows,
    }
    manifest_path = out_dir / "hmmwv_policy_trajectories_L8_grid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
