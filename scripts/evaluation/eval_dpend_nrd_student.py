"""Teacher-vs-student evaluation on identical held-out NRD pairs (distillation plan, section 10).

Runs, on the same (context, goal) pairs from the validation bank:
  1. the privileged z1 teacher in control;
  2. the z2-history student in control (beta = 0);
and reports, for each, success at the tolerance, closest-approach curves, distance,
time-to-success and termination statistics, action statistics, and the
teacher-student action agreement on the states visited under each controller,
plus the success overlap. Writes summary.json, per_pair.json and a figure.

    PYTHONPATH=src python scripts/evaluation/eval_dpend_nrd_student.py \
        --student-run-dir artifacts/rl_runs/<student run> [--student-checkpoint student_best.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.nrd.context_bank import load_context_bank  # noqa: E402
from nedm.rl.dpend_distill import StudentPolicy, action_agreement, load_teacher, rollout_pairs  # noqa: E402
from nedm.rl.dpend_nrd_reach_env import DPendNRDReachEnv, make_eval_pairs  # noqa: E402

TOLERANCES_M = [0.01, 0.015, 0.02, 0.03, 0.05]


def summarize(rec: dict[str, np.ndarray], tolerance: float) -> dict[str, Any]:
    success = rec["success"] > 0.5
    return {
        "success_rate": float(success.mean()),
        "timeout_rate": float(rec["time_out"].mean()),
        "spin_rate": float(rec["spin"].mean()),
        "ood_rate": float(rec["ood"].mean()),
        "nonfinite_rate": float(rec["nonfinite"].mean()),
        "closest_approach_success": {f"{t * 1000:g}mm": float((rec["min_distance_m"] <= t).mean()) for t in TOLERANCES_M},
        "final_distance_m": {"median": float(np.median(rec["final_distance_m"])), "p90": float(np.percentile(rec["final_distance_m"], 90))},
        "min_distance_m": {"median": float(np.median(rec["min_distance_m"])), "p90": float(np.percentile(rec["min_distance_m"], 90))},
        "time_to_success_s": {
            "median": float(np.median(rec["success_time_s"][success])) if success.any() else float("nan"),
            "p90": float(np.percentile(rec["success_time_s"][success], 90)) if success.any() else float("nan"),
        },
        "action_abs_mean": float(rec["action_abs_mean"].mean()),
        "action_slew_mean": float(rec["action_slew_mean"].mean()),
        "action_saturated_frac": float(rec["action_saturated_frac"].mean()),
    }


def load_student(checkpoint: Path, device: str) -> tuple[StudentPolicy, dict[str, Any]]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = payload["config"]
    student = StudentPolicy(int(cfg["history_len"]), (int(cfg["student_input_dim"]) - 2) // int(cfg["history_len"]), tuple(cfg["hidden_dims"])).to(device)
    student.load_state_dict(payload["model_state_dict"])
    student.eval()
    return student, payload


def write_figure(path: Path, teacher_rec, student_rec, tolerance: float, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"teacher": "#2a78d6", "student": "#1baf7a"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    tols = np.geomspace(0.002, 0.2, 60)
    ax = axes[0]
    for name, rec in (("teacher", teacher_rec), ("student", student_rec)):
        ax.plot(tols * 1000, [(rec["min_distance_m"] <= t).mean() for t in tols], color=colors[name], lw=2,
                label=f"{name} ({rec['success'].mean() * 100:.0f}% @ {tolerance * 1000:.0f} mm)")
    ax.axvline(tolerance * 1000, color="#8a8a85", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("tolerance on closest approach [mm]")
    ax.set_ylabel("fraction of held-out pairs")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    ax = axes[1]
    x = np.clip(teacher_rec["min_distance_m"] * 1000, 0.5, None)
    y = np.clip(student_rec["min_distance_m"] * 1000, 0.5, None)
    ax.scatter(x, y, s=14, color="#4a4a46", alpha=0.7)
    lim = (0.5, max(float(x.max()), float(y.max())) * 1.2)
    ax.plot(lim, lim, color="#8a8a85", lw=1, ls=":")
    ax.axvline(tolerance * 1000, color=colors["teacher"], lw=1, ls="--")
    ax.axhline(tolerance * 1000, color=colors["student"], lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("teacher closest approach [mm]")
    ax.set_ylabel("student closest approach [mm]")
    ax.grid(alpha=0.25, which="both")
    ax = axes[2]
    mask = student_rec["active_mask"].astype(bool)
    ax.scatter(student_rec["teacher_actions"][mask], student_rec["student_actions"][mask], s=4, alpha=0.25, color="#4a4a46")
    ax.plot([-1, 1], [-1, 1], color="#8a8a85", lw=1, ls=":")
    ax.set_xlabel("teacher action on student-visited states")
    ax.set_ylabel("student action")
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.25)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student-run-dir", type=Path, required=True)
    parser.add_argument("--student-checkpoint", type=str, default="student_best.pt")
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--pairs-seed", type=int, default=20260826)
    parser.add_argument("--context-bank", type=Path, default=None, help="default: the bank named in the student config")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = args.device
    torch.set_float32_matmul_precision("high")
    run_dir = args.student_run_dir.resolve()
    student_ckpt = (run_dir / args.student_checkpoint) if not Path(args.student_checkpoint).is_absolute() else Path(args.student_checkpoint)
    student, payload = load_student(student_ckpt, device)
    cfg = payload["config"]
    teacher_run = Path(cfg["teacher_run_dir"])
    teacher_train_cfg = json.loads((teacher_run / "env_cfg.json").with_name("train_cfg.json").read_text())
    bank_path = args.context_bank or Path(cfg["eval"]["context_bank"])

    env_cfg = dict(cfg["env_cfg"])
    env_cfg.update({"num_envs": int(args.num_pairs), "device": device, "auto_reset": False, "observe_z2": False, "context_bank": str(bank_path)})
    env = DPendNRDReachEnv(env_cfg, device=device)
    if not torch.allclose(env.z2_mean.cpu(), payload["z2_mean"].cpu(), atol=1e-6):
        raise ValueError("student checkpoint z2 normalization does not match the NRD checkpoint")
    teacher, teacher_iter = load_teacher(teacher_train_cfg, Path(cfg["teacher_checkpoint"]), env, device)
    bank = load_context_bank(bank_path)
    context_ids, goals = make_eval_pairs(bank, int(args.num_pairs), int(args.pairs_seed), env.cfg["goal"], env.link_lengths, env.success_tolerance)
    goals_np = goals.numpy()
    H = int(cfg["history_len"])
    max_steps = int(env.max_episode_length)

    teacher_rec = rollout_pairs(env, teacher, student, "teacher", context_ids, goals, max_steps, H)
    student_rec = rollout_pairs(env, teacher, student, "student", context_ids, goals, max_steps, H)
    t_ok, s_ok = teacher_rec["success"] > 0.5, student_rec["success"] > 0.5
    summary = {
        "student_run": run_dir.name,
        "student_checkpoint": str(student_ckpt),
        "student_iteration": int(payload["iteration"]),
        "teacher_checkpoint": cfg["teacher_checkpoint"],
        "teacher_iteration": teacher_iter,
        "num_pairs": int(args.num_pairs),
        "pairs_seed": int(args.pairs_seed),
        "context_bank": str(bank_path),
        "tolerance_m": env.success_tolerance,
        "history_len": H,
        "teacher": summarize(teacher_rec, env.success_tolerance),
        "student": summarize(student_rec, env.success_tolerance),
        "agreement_on_teacher_visited_states": action_agreement(teacher_rec),
        "agreement_on_student_visited_states": action_agreement(student_rec),
        "overlap": {
            "both": int((t_ok & s_ok).sum()),
            "teacher_only": int((t_ok & ~s_ok).sum()),
            "student_only": int((~t_ok & s_ok).sum()),
            "neither": int((~t_ok & ~s_ok).sum()),
            "success_gap_pp": float((s_ok.mean() - t_ok.mean()) * 100.0),
            "closest_approach_median_diff_mm": float((np.median(student_rec["min_distance_m"]) - np.median(teacher_rec["min_distance_m"])) * 1000.0),
        },
    }
    acceptance = {
        "success_within_5pp": abs(summary["overlap"]["success_gap_pp"]) <= 5.0,
        "failure_rates_within_1pp": all(summary["student"][k] - summary["teacher"][k] <= 0.01 for k in ("spin_rate", "ood_rate", "nonfinite_rate")),
        "closest_approach_median_within_5mm": abs(summary["overlap"]["closest_approach_median_diff_mm"]) <= 5.0,
    }
    summary["acceptance"] = acceptance
    output_dir = args.output_dir.resolve() if args.output_dir else run_dir / f"student_vs_teacher_eval_{student_ckpt.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    per_pair = []
    for i in range(int(args.num_pairs)):
        per_pair.append({
            "pair": i, "context_id": int(context_ids[i]), "goal_x_m": float(goals_np[i, 0]), "goal_z_m": float(goals_np[i, 1]),
            **{f"{name}_{key}": float(rec[key][i]) for name, rec in (("teacher", teacher_rec), ("student", student_rec))
               for key in ("success", "time_out", "spin", "ood", "success_time_s", "final_distance_m", "min_distance_m", "action_abs_mean", "action_slew_mean")},
        })
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "per_pair.json").write_text(json.dumps(per_pair, indent=2))
    np.savez_compressed(output_dir / "trajectories.npz",
                        teacher_tips=teacher_rec["tips"], student_tips=student_rec["tips"],
                        teacher_rollout_teacher_actions=teacher_rec["teacher_actions"], teacher_rollout_student_actions=teacher_rec["student_actions"],
                        student_rollout_teacher_actions=student_rec["teacher_actions"], student_rollout_student_actions=student_rec["student_actions"],
                        goals=goals_np, context_ids=context_ids.numpy())
    write_figure(output_dir / "student_vs_teacher.png", teacher_rec, student_rec, env.success_tolerance,
                 f"{run_dir.name} / {student_ckpt.name}: teacher vs student on {args.num_pairs} held-out NRD pairs")
    t, s, ov = summary["teacher"], summary["student"], summary["overlap"]
    agr = summary["agreement_on_student_visited_states"]
    print(f"teacher: success {t['success_rate']:.2f} spin {t['spin_rate']:.2f} mind {t['min_distance_m']['median'] * 1000:.1f} mm tts {t['time_to_success_s']['median']:.2f} s")
    print(f"student: success {s['success_rate']:.2f} spin {s['spin_rate']:.2f} ood {s['ood_rate']:.2f} mind {s['min_distance_m']['median'] * 1000:.1f} mm tts {s['time_to_success_s']['median']:.2f} s")
    print(f"overlap both/teacher-only/student-only/neither = {ov['both']}/{ov['teacher_only']}/{ov['student_only']}/{ov['neither']} gap {ov['success_gap_pp']:+.1f} pp")
    print(f"agreement (student-visited): mae {agr['mae']:.3f} rmse {agr['rmse']:.3f} p95 {agr['p95_abs_error']:.3f} sign-disagree {agr['sign_disagreement_rate']:.3f}; "
          f"(teacher-visited): mae {summary['agreement_on_teacher_visited_states']['mae']:.3f}")
    print(f"acceptance: {acceptance}")
    print(f"wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
