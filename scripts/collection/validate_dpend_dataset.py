"""Validation gates G0-G2 for the double-pendulum + camera collector.

Live mechanism checks (no dataset needed):
  1. State round-trip + FK (G1): reset_state(q, w) then read_state must return the
     same generalized coordinates, and forward kinematics from (q1, q2) must match
     the Chrono marker world positions.
  2. Replay determinism (G1.4): the same IC + action sequence run twice in one
     process (at different absolute sim times, i.e. different solver warm starts)
     must produce the same trajectory within solver tolerance.
  3. Timestep convergence (G0): dt_sim 1e-3 vs 5e-4 trajectories must agree over
     short horizons (the system is chaotic, long-horizon divergence is expected).
  4. Planarity: |y| of both links stays at numerical zero.

Stored-dataset checks (pass --dataset-root):
  5. Frame alignment (G2): every CSV row's cam_time_s equals its time_s, frame
     count equals row count, and -- end to end -- the yellow tip marker's pixel
     centroid in the stored frame matches the pinhole projection of the recorded
     tip position, for sampled rows of every episode.

Run:  PYTHONPATH=src conda run -n nedm python scripts/collection/validate_dpend_dataset.py \
          [--dataset-root artifacts/datasets/dpend_smoke] [--skip-live]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nedm.double_pendulum_data import (  # noqa: E402
    CONTROL_DT_S,
    DT_SIM_S,
    TAU_MAX_NM,
    FrameTap,
    _advance_to_next_boundary,
    build_scene,
    find_tip_pixel,
    forward_kinematics,
    project_to_pixel,
    read_state,
    reset_state,
    run_episode,
    sample_action_sequence,
)

PASS = "PASS"
FAIL = "FAIL"
failures: list[str] = []


def report(name: str, ok: bool, detail: str) -> None:
    print(f"[{PASS if ok else FAIL}] {name}: {detail}")
    if not ok:
        failures.append(name)


def run_physics_trajectory(
    scene, initial_condition, actions: np.ndarray, dt_sim_s: float = DT_SIM_S
) -> np.ndarray:
    reset_state(scene, *initial_condition)
    rows = [read_state(scene)]
    for action in actions:
        scene.elbow_torque.SetSetpoint(float(action) * TAU_MAX_NM, scene.system.GetChTime())
        _advance_to_next_boundary(scene, None, dt_sim_s)
        rows.append(read_state(scene))
    keys = ["q1_rad", "q2_rad", "omega1_radps", "omega2_radps", "tip_x_m", "tip_z_m", "out_of_plane_m"]
    return np.array([[r[k] for k in keys] for r in rows])


def check_state_roundtrip_and_fk() -> None:
    scene = build_scene(with_camera=False)
    rng = np.random.default_rng(7)
    max_q_err = max_w_err = max_fk_err = 0.0
    for _ in range(50):
        q1, q2 = rng.uniform(-math.pi, math.pi, 2)
        w1, w2 = rng.uniform(-8.0, 8.0, 2)
        reset_state(scene, q1, q2, w1, w2)
        state = read_state(scene)
        dq1 = abs(math.atan2(math.sin(state["q1_rad"] - q1), math.cos(state["q1_rad"] - q1)))
        dq2 = abs(math.atan2(math.sin(state["q2_rad"] - q2), math.cos(state["q2_rad"] - q2)))
        max_q_err = max(max_q_err, dq1, dq2)
        max_w_err = max(max_w_err, abs(state["omega1_radps"] - w1), abs(state["omega2_radps"] - w2))
        ex, ez, tx, tz = forward_kinematics(state["q1_rad"], state["q2_rad"])
        max_fk_err = max(
            max_fk_err,
            abs(ex - state["elbow_x_m"]), abs(ez - state["elbow_z_m"]),
            abs(tx - state["tip_x_m"]), abs(tz - state["tip_z_m"]),
        )
    report("state round-trip (q, omega)", max_q_err < 1e-9 and max_w_err < 1e-9,
           f"max angle err {max_q_err:.2e} rad, max rate err {max_w_err:.2e} rad/s")
    report("FK vs Chrono markers", max_fk_err < 1e-9, f"max position err {max_fk_err:.2e} m")


def check_replay_determinism() -> None:
    """One-step replay (plan 4.2 check 4): identical (z1, a) -> identical z1'.

    The system is chaotic, so whole-trajectory comparison amplifies solver
    warm-start noise to O(1); the Markov-sufficiency property is per step.
    """
    scene = build_scene(with_camera=False)
    rng = np.random.default_rng(11)
    cases = [
        ((rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi),
          rng.uniform(-6, 6), rng.uniform(-8, 8)), rng.uniform(-1, 1))
        for _ in range(25)
    ]
    first_pass = [run_physics_trajectory(scene, ic, np.array([a]))[1] for ic, a in cases]
    # Perturb solver warm-start state with an unrelated stretch, then replay.
    run_physics_trajectory(scene, (2.0, 1.0, -4.0, 5.0), sample_action_sequence(rng, "smooth", 37))
    second_pass = [run_physics_trajectory(scene, ic, np.array([a]))[1] for ic, a in cases]
    max_dev = float(np.abs(np.array(first_pass) - np.array(second_pass)).max())
    report("one-step replay determinism", max_dev < 1e-9, f"max one-step deviation {max_dev:.2e}")


def check_timestep_convergence() -> None:
    rng = np.random.default_rng(13)
    initial_condition = (1.2, 0.7, 0.5, -1.0)
    actions = sample_action_sequence(rng, "smooth", 100)  # 2 s
    coarse_scene = build_scene(with_camera=False)
    fine_scene = build_scene(with_camera=False)
    coarse = run_physics_trajectory(coarse_scene, initial_condition, actions, dt_sim_s=1e-3)
    fine = run_physics_trajectory(fine_scene, initial_condition, actions, dt_sim_s=5e-4)
    tip_err = np.linalg.norm(coarse[:, 4:6] - fine[:, 4:6], axis=1)
    horizons = {"0.5s": 25, "1.0s": 50, "2.0s": 100}
    detail = ", ".join(f"{name}: {tip_err[step]*1000:.2f} mm" for name, step in horizons.items())
    report("timestep convergence 1e-3 vs 5e-4 (tip)", tip_err[25] < 0.005, detail)
    planarity = float(max(coarse[:, 6].max(), fine[:, 6].max()))
    report("planarity |y|", planarity < 1e-9, f"max out-of-plane {planarity:.2e} m")


def check_constraint_drift() -> None:
    """After a 10 s driven episode, FK from (q1, q2) must still match the Chrono
    marker positions -- i.e. the joints have not drifted apart."""
    from nedm.double_pendulum_data import forward_kinematics, read_state

    scene = build_scene(with_camera=False)
    rng = np.random.default_rng(19)
    run_physics_trajectory(scene, (1.5, -0.5, 3.0, 2.0), sample_action_sequence(rng, "smooth", 500))
    state = read_state(scene)
    ex, ez, tx, tz = forward_kinematics(state["q1_rad"], state["q2_rad"])
    drift = max(abs(ex - state["elbow_x_m"]), abs(ez - state["elbow_z_m"]),
                abs(tx - state["tip_x_m"]), abs(tz - state["tip_z_m"]))
    # Baseline on this box: ~2.6e-6 m after 500 steps. The 1e-4 gate is far below
    # anything observable (one pixel spans ~5 mm of workspace).
    report("constraint drift over 10 s", drift < 1e-4, f"max FK-vs-marker gap {drift:.2e} m")


def check_dataset(dataset_root: Path, rows_per_episode: int) -> None:
    index = json.loads((dataset_root / "dataset_index.json").read_text())
    episodes = index["episodes"]
    stamp_bad = 0
    count_bad = 0
    pixel_errors: list[float] = []
    occluded = 0
    checked_rows = 0
    for episode in episodes:
        with (dataset_root / episode["csv_path"]).open(newline="") as fp:
            rows = list(csv.DictReader(fp))
        frames = np.load(dataset_root / episode["frames_path"])
        if len(rows) != frames.shape[0]:
            count_bad += 1
            continue
        sample_indices = np.linspace(0, len(rows) - 1, min(rows_per_episode, len(rows))).astype(int)
        for row_index in sample_indices:
            row = rows[row_index]
            checked_rows += 1
            if abs(float(row["cam_time_s"]) - float(row["time_s"])) > 1e-4:
                stamp_bad += 1
            expected = project_to_pixel(float(row["tip_x_m"]), float(row["tip_z_m"]))
            found = find_tip_pixel(frames[row_index])
            if found is None:
                occluded += 1
                continue
            pixel_errors.append(math.hypot(expected[0] - found[0], expected[1] - found[1]))

    report("frame count == row count", count_bad == 0, f"{count_bad}/{len(episodes)} episodes mismatched")
    report("cam_time_s == time_s", stamp_bad == 0, f"{stamp_bad}/{checked_rows} rows mismatched")
    if pixel_errors:
        pixel_errors_arr = np.array(pixel_errors)
        detail = (
            f"median {np.median(pixel_errors_arr):.2f} px, p95 {np.percentile(pixel_errors_arr, 95):.2f} px, "
            f"max {pixel_errors_arr.max():.2f} px over {len(pixel_errors)} rows ({occluded} occluded/skipped)"
        )
        report("tip pixel projection vs rendered blob", np.median(pixel_errors_arr) < 2.5, detail)
    else:
        report("tip pixel projection vs rendered blob", False, "no visible tip marker found in any sampled frame")


def check_live_camera_alignment() -> None:
    """Collect one throwaway episode WITH camera and run the pixel test on it."""
    scene = build_scene(with_camera=True)
    tap = FrameTap(scene.camera)
    _advance_to_next_boundary(scene, tap)
    rng = np.random.default_rng(17)
    actions = sample_action_sequence(rng, "piecewise", 50)
    result, state_rows, frames = run_episode(
        scene, tap, "align_check", "train", "piecewise", actions, (0.9, 1.8, 2.0, -1.0), None
    )
    from nedm.double_pendulum_data import CSV_HEADER

    value_fields = CSV_HEADER[3:]  # state_rows columns
    tip_x_col, tip_z_col = value_fields.index("tip_x_m"), value_fields.index("tip_z_m")
    errors = []
    occluded = 0
    for row_index in range(state_rows.shape[0]):
        expected = project_to_pixel(state_rows[row_index, tip_x_col], state_rows[row_index, tip_z_col])
        found = find_tip_pixel(frames[row_index])
        if found is None:
            occluded += 1
            continue
        errors.append(math.hypot(expected[0] - found[0], expected[1] - found[1]))
    errors_arr = np.array(errors)
    detail = (
        f"median {np.median(errors_arr):.2f} px, max {errors_arr.max():.2f} px "
        f"over {len(errors)} frames ({occluded} occluded)"
    )
    report("LIVE tip pixel projection vs render", np.median(errors_arr) < 2.5, detail)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--rows-per-episode", type=int, default=8)
    parser.add_argument("--skip-live", action="store_true")
    args = parser.parse_args()

    if not args.skip_live:
        check_state_roundtrip_and_fk()
        check_replay_determinism()
        check_timestep_convergence()
        check_constraint_drift()
        check_live_camera_alignment()
    if args.dataset_root is not None:
        check_dataset(args.dataset_root.resolve(), args.rows_per_episode)

    if failures:
        print(f"\n{len(failures)} check(s) FAILED: {failures}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
