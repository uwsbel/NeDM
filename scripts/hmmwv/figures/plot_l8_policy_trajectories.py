#!/usr/bin/env python3
"""Representative closed-loop x/y/theta trajectory-tracking figures (L8 backbone).

For each terrain (rigid flat, rigid bumpy, CRM), picks 3 hand-selected
"good looking" complex references (a sustained-turn loop, a multi-steer
zigzag, a sine-steer wave) and plots the closed-loop rollout of all three
onehot policies (generalist mixture, rigid-only, CRM-only) against the
reference, as x/y trajectories + x(t) + y(t) + heading theta(t).

Companion figure to hmmwv_policy_transfer_bars_L8.py (same matched OFAT-L8
triplet: mix00_bestval61, mix25_bestval51, mix100_bestval54, all ~iter 1000).

Usage:
    python plot_l8_policy_trajectories.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

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
# dataviz-skill validated 3-slot categorical (light mode): blue / orange / green.
# Matches hmmwv_policy_transfer_bars_L8.py's C0/C1/C2 (mixture/rigid/crm) mapping.
POLICY_COLORS = {
    "mix25": "#2a78d6",
    "mix00": "#eb6834",
    "mix100": "#008300",
}
REF_COLOR = "#333333"

# Hand-picked "good looking" complex references per terrain: (episode_idx, short tag).
# Chosen for maneuver diversity (loop / zigzag / wave) and visually rich shape
# (see scratch survey: total heading swing + bbox size), not for outcome.
SELECTIONS = {
    "rigid": [(12, "sustained turn (loop)"), (3, "multi-steer (zigzag)"), (19, "sine-steer (wave)")],
    "bumpy": [(18, "sustained turn (loop)"), (9, "multi-steer (zigzag)"), (1, "sine-steer (wave)")],
    "crm": [(18, "sustained turn (loop)"), (3, "multi-steer (zigzag)"), (1, "sine-steer (wave)")],
}

STEP_DT_S = 0.05  # dt_s(0.01) * action_repeat(5), from dynamics checkpoint metadata


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


LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 14
REF_LINEWIDTH = 3.2
POLICY_LINEWIDTH = 2.4


def plot_terrain(terrain: str, out_dir: Path) -> dict:
    selections = SELECTIONS[terrain]
    fig, axes = plt.subplots(len(selections), 4, figsize=(17.0, 10.5))

    manifest_rows = []
    for row, (idx, tag) in enumerate(selections):
        ax_xy, ax_x, ax_y, ax_theta = axes[row]
        episodes = {pol: load_episode(RUNS[pol][terrain], idx) for pol in POLICIES}
        ref_pose = episodes[POLICIES[0]]["ref_pose"]
        ref_name = episodes[POLICIES[0]]["reference"]

        t = np.arange(ref_pose.shape[0]) * STEP_DT_S
        ref_theta_deg = np.rad2deg(np.unwrap(ref_pose[:, 2]))

        ax_xy.plot(ref_pose[:, 0], ref_pose[:, 1], color=REF_COLOR, ls="--", lw=REF_LINEWIDTH, label="Reference", zorder=1)
        ax_x.plot(t, ref_pose[:, 0], color=REF_COLOR, ls="--", lw=REF_LINEWIDTH, label="Reference", zorder=1)
        ax_y.plot(t, ref_pose[:, 1], color=REF_COLOR, ls="--", lw=REF_LINEWIDTH, label="Reference", zorder=1)
        ax_theta.plot(t, ref_theta_deg, color=REF_COLOR, ls="--", lw=REF_LINEWIDTH, label="Reference", zorder=1)

        row_rmse = {}
        for pol in POLICIES:
            pose = episodes[pol]["pose"]
            color = POLICY_COLORS[pol]
            theta_deg = np.rad2deg(np.unwrap(pose[:, 2]))
            ax_xy.plot(pose[:, 0], pose[:, 1], color=color, lw=POLICY_LINEWIDTH, label=POLICY_LABELS[pol], zorder=2)
            ax_x.plot(t, pose[:, 0], color=color, lw=POLICY_LINEWIDTH, label=POLICY_LABELS[pol], zorder=2)
            ax_y.plot(t, pose[:, 1], color=color, lw=POLICY_LINEWIDTH, label=POLICY_LABELS[pol], zorder=2)
            ax_theta.plot(t, theta_deg, color=color, lw=POLICY_LINEWIDTH, label=POLICY_LABELS[pol], zorder=2)
            row_rmse[pol] = episodes[pol]["xy_rmse_m"]

        ax_xy.plot(*ref_pose[0, :2], marker="o", ms=9, mfc="white", mec="black", mew=1.5, zorder=3)
        ax_xy.plot(*ref_pose[-1, :2], marker="s", ms=8, mfc="black", mec="black", zorder=3)
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.set_ylabel("y [m]", fontsize=LABEL_FONTSIZE)

        ax_x.set_ylabel("x [m]", fontsize=LABEL_FONTSIZE)
        ax_y.set_ylabel("y [m]", fontsize=LABEL_FONTSIZE)
        ax_theta.set_ylabel(r"$\theta$ [deg]", fontsize=LABEL_FONTSIZE)

        for ax in (ax_xy, ax_x, ax_y, ax_theta):
            ax.grid(True, alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=TICK_FONTSIZE)

        if row == len(selections) - 1:
            ax_xy.set_xlabel("x [m]", fontsize=LABEL_FONTSIZE)
            ax_x.set_xlabel("t [s]", fontsize=LABEL_FONTSIZE)
            ax_y.set_xlabel("t [s]", fontsize=LABEL_FONTSIZE)
            ax_theta.set_xlabel("t [s]", fontsize=LABEL_FONTSIZE)

        manifest_rows.append(
            {
                "episode_idx": idx,
                "maneuver_tag": tag,
                "reference": ref_name,
                "xy_rmse_m": row_rmse,
            }
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False,
        fontsize=LEGEND_FONTSIZE, bbox_to_anchor=(0.5, 1.0),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    png_path = out_dir / f"hmmwv_policy_trajectories_L8_{terrain}.png"
    pdf_path = out_dir / f"hmmwv_policy_trajectories_L8_{terrain}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {pdf_path}")
    return {"terrain": terrain, "rows": manifest_rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts/analysis/manuscript_figs"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"policies": POLICY_LABELS, "step_dt_s": STEP_DT_S, "terrains": {}}
    for terrain in ["rigid", "bumpy", "crm"]:
        manifest["terrains"][terrain] = plot_terrain(terrain, out_dir)

    manifest_path = out_dir / "hmmwv_policy_trajectories_L8_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
