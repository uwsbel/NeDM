# NRD double-pendulum teacher–student distillation — implementation notes

Implements `NRD_double_pendulum_teacher_student_distillation_plan.md`
(2026-08-26). Teacher: the state-only reaching policy
`artifacts/rl_runs/dpend_nrd_reach_z1_armreward_lowerhalf_seed1_20260826/model_1499.pt`
(see `rl_implementation_notes.md`). Everything runs inside the frozen NRD.

## What exists

| Piece | Path |
|---|---|
| Distillation library: teacher loader, `StudentPolicy`, `StudentHistory`, FIFO `ReplayBuffer`, paired rollouts + action agreement | `src/nedm/rl/dpend_distill.py` |
| Online DAgger runner (TensorBoard, JSON log, checkpoints) | `scripts/training/distill_dpend_nrd_student.py` |
| Teacher-vs-student evaluation on identical held-out pairs | `scripts/evaluation/eval_dpend_nrd_student.py` |
| One-job-at-a-time multi-seed launcher | `scripts/training/launch_dpend_distill_seeds.sh` |

The structure follows `~/Genesis/examples/manipulation/behavior_cloning.py`
(teacher = RL runner's inference policy, FIFO experience buffer, supervised
action imitation with Adam + grad clipping, TensorBoard, periodic checkpoints);
the executed action is the plan's β-mixture rather than Genesis' closeness gate.

## Plan → code

- **Teacher (3.1):** the env built with `observe_z2=False`; its `obs_buf` is the
  10-D privileged observation; teacher = `runner.get_inference_policy` (stored
  empirical normalizer, deterministic), clamped to [−1, 1].
- **Student (3.2, 4):** `[4 × normalized z2, g/L]` = 258-D → MLP 256-128-64 ELU →
  tanh. Normalization uses the NRD checkpoint's `z2_mean/z2_std` (fingerprint
  stored in the student checkpoint and verified by the evaluator). No z1, tip,
  goal error, reward, or teacher action reaches the student.
- **History (5):** `StudentHistory` keeps `(N, 4, 64)` normalized latents; on
  reset it is filled from the recorded context at indices `[0, 5, 10, 15]`
  (four latents 0.1 s apart); after each policy step it appends the env's latest
  recursively predicted latent; auto-reset envs are re-seeded from their new
  context so histories never cross episodes (unit-checked).
- **DAgger (6–8):** per policy step the teacher labels the visited state (always
  stored), the executed action is teacher with probability β per env and
  student otherwise; β = max(0, 1 − i/50); 4096 envs × 24 steps per iteration,
  FIFO buffer 500 k samples, Adam 1e-3, batch 8192, 5 epochs per iteration,
  SmoothL1, grad-norm 1.0; 200 iterations (50 decay + 150 fully
  student-controlled). Actual teacher/student control fractions are logged.
- **Task (9):** the teacher's env config verbatim (lower-half goals, 2 cm, 10 Hz,
  5 s) and the teacher's train context bank.
- **Evaluation (10):** teacher-controlled and student-controlled rollouts on the
  same 100 val-bank pairs (`pairs_seed 20260826` — the exact set the teacher's
  87 % was measured on); both policies are queried at every step so action
  agreement is measured on teacher-visited and student-visited states.

Pre-flight checks (plan 13): history init/shift/reset exact; the teacher
through this path reproduces `nrd_chrono_transfer_eval_iter1499/per_pair.json`
per pair (87/100, closest-approach difference 0.000 mm); synthetic-batch
regression converges; FIFO wrap correct.

## Results (3 seeds, 200 DAgger iterations each, ≈ 6 min per seed)

Held-out NRD pairs (100, identical for teacher and every student), success at 2 cm:

| seed | student (best ckpt, iter) | student (last ckpt, iter 199) | teacher | overlap both / T-only / S-only / neither | closest approach S vs T | action MAE on S-visited states (p95) | sign disagreement |
|---|---|---|---|---|---|---|---|
| 1 | **88 %** (169) | 88 % | 87 % | 86 / 1 / 2 / 11 | 14.9 vs 15.2 mm | 0.021 (0.066) | 1.1 % |
| 2 | **90 %** (109) | 86 % | 87 % | 86 / 1 / 4 / 9 | 16.4 vs 15.2 mm | 0.025 (0.067) | 1.6 % |
| 3 | **87 %** (69) | 85 % | 87 % | 84 / 3 / 3 / 10 | 15.4 vs 15.2 mm | 0.027 (0.074) | 1.7 % |

Spin / OOD / non-finite: 0 for every student. Time to success (median):
1.40–1.44 s student vs 1.44 s teacher. Action magnitude and slew identical to
the teacher (|a| 0.50–0.51, slew 0.24–0.25). Closest-approach curves overlap
the teacher's over the whole range (seed 1: 1 cm 25 % vs 14 %, 1.5 cm 51 % vs
48 %, 3 cm 91 % vs 90 %, 5 cm 95 % vs 94 %).

Held-out student-controlled success during training (β = 0 from iteration 50):
seed 1 83 % → 84 → 86 → 87 → 82 → 87 → 85 → 86 → 88 → 86 (every 20
iterations); seeds 2 and 3 similar (76–90 %). The student is stable after β
reaches zero (no collapse; the ±3 pp wobble is the same as the teacher's own
checkpoint-to-checkpoint variation).

**Acceptance (plan 11): all five criteria hold on all three seeds** — success
within 5 pp (+1.0 / +3.0 / +0.0 pp with best checkpoints; +1 / −1 / −2 pp with
the unselected last checkpoints), stable after β = 0, failure rates not above
the teacher's, median closest approach within 5 mm (−0.3 / +1.2 / +0.2 mm),
reproduced on three seeds.

Caveat on "best": `student_best.pt` is selected by the held-out success itself
(no separate selection split), so the last-checkpoint column is the
selection-free number; both columns pass.

## Interpretation

Four 10 Hz camera latents plus the goal reproduce the privileged controller
almost exactly (action MAE 0.02 on a ±1 action range, 1–2 % sign disagreement,
identical closed-loop statistics). As the plan anticipated, this is a
controlled distillation result on a scene that is nearly determined by z1, not
evidence that vision adds task information. Note the student's latents inside
the NRD are the model's recursively predicted ones; a Chrono transfer of the
student (camera frame → frozen encoder → 4-step history) is the natural next
check and is outside this plan's scope.

## Artifacts

`artifacts/rl_runs/dpend_nrd_student_z2hist4_from_z1_armreward_lowerhalf_seed{1,2,3}_20260826/`:
`distill_cfg.json`, TensorBoard events (`loss/*`, `dagger/*`, `rollout/*`,
`eval_student/*`), `training_log.json`, `summary.json`, checkpoints
`student_best.pt` / `student_last.pt` / `student_0025..0200.pt`, and
`student_vs_teacher_eval_student_best/` (`summary.json`, `per_pair.json`,
`trajectories.npz`, `student_vs_teacher.png`).
