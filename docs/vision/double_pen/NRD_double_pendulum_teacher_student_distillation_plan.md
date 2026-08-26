# NRD Double-Pendulum Teacher–Student Distillation Plan

**Status:** proposed next experiment after the successful state-only reaching policy  
**Teacher:** `artifacts/rl_runs/dpend_nrd_reach_z1_armreward_lowerhalf_seed1_20260826/model_1499.pt`  
**Scope:** online teacher–student distillation inside the frozen NRD only

## 1. Objective

Distill the successful privileged `z1` reaching policy into a student policy that receives only:

1. a fixed four-observation history of camera latents `z2`; and
2. the Cartesian goal position.

The student must output the same normalized elbow-torque action as the teacher. The experiment tests whether a short history of the frozen camera representation contains enough pose and motion information to reproduce the privileged controller in closed loop.

This is a new vision-only policy experiment. It is distinct from the original Study-1 comparison, where `z2` was appended to `z1` rather than used in place of it.

## 2. Fixed experiment boundary

The first implementation deliberately keeps the training recipe small:

- use a fixed history length `H = 4`;
- initialize the student randomly;
- generate teacher labels online during DAgger rollouts;
- use supervised action imitation only;
- keep the camera encoder, NRD transition model, and teacher frozen;
- train and evaluate inside NRD;
- use the teacher's lower-half goal distribution and 2 cm success tolerance.

The first implementation does **not** include:

- an observability or state-decoding probe;
- offline teacher-label generation;
- auxiliary `z1`, velocity, or tip-position prediction;
- PPO or other reinforcement-learning fine-tuning of the student;
- changes to the NRD dynamics or camera encoder.

## 3. Teacher and student contracts

### 3.1 Teacher

The frozen teacher is the final successful state-only policy. At policy step `k`, it receives the existing 10-dimensional observation

\[
o^T_k =
\left[
\operatorname{normalize\_state}(z1_k),
\frac{g_k}{L},
\frac{g_k-\operatorname{tip}(z1_k)}{L}
\right],
\]

where

\[
z1=[\cos q_1,\sin q_1,\cos q_2,\sin q_2,\omega_1,\omega_2],
\qquad L=0.6\ \mathrm{m}.
\]

The teacher is loaded with its stored empirical observation normalization and queried in deterministic inference mode. Its action label is the action applied by the environment:

\[
a^T_k=\operatorname{clamp}(\pi_T(o^T_k),-1,1).
\]

### 3.2 Student

The student receives four normalized camera latents at the 10 Hz policy rate and the normalized Cartesian goal:

\[
o^S_k =
\left[
\widetilde{z2}_{k-3},
\widetilde{z2}_{k-2},
\widetilde{z2}_{k-1},
\widetilde{z2}_{k},
\frac{g_k}{L}
\right],
\]

with

\[
\widetilde{z2}=\frac{z2-z2_{\mathrm{mean}}}{z2_{\mathrm{std}}}.
\]

The `z2` statistics must come from the frozen NRD checkpoint. Raw `z2` must not be used because the encoder latents contain a large shared constant component and only become meaningful after per-dimension normalization.

Each `z2` has 64 dimensions, so the student input width is

\[
4\times64+2=258.
\]

The student must not receive `z1`, angular velocity, tip position, the exact goal-error vector, reward, termination information, or the teacher action as an input.

The student outputs one normalized elbow action:

\[
a^S_k=\pi_S(o^S_k)\in[-1,1],
\qquad
\tau_k=1.5a^S_k\ \mathrm{N\,m}.
\]

## 4. Student architecture

Use a simple fixed-history MLP for the baseline:

```text
flattened [4 x normalized z2, normalized goal] = 258
    -> Linear(258, 256) + ELU
    -> Linear(256, 128) + ELU
    -> Linear(128, 64) + ELU
    -> Linear(64, 1)
    -> tanh
```

Initialize all student parameters randomly using the experiment seed. A recurrent student is unnecessary for the baseline because the fixed four-observation window already exposes temporal information explicitly.

## 5. Student-history construction

The NRD transition model advances at 50 Hz while the policy acts at 10 Hz with `action_repeat = 5`. The student history is updated once per policy decision, not once per NRD transition.

### 5.1 Reset

Each reset context already contains 16 aligned `z2` values at 50 Hz. Initialize the four student observations from context indices

```text
[0, 5, 10, 15]
```

so that the student begins with four latents spaced by 0.1 s and spanning 0.3 s. Normalize each latent with the checkpoint's `z2_mean` and `z2_std`.

### 5.2 Rollout

After applying one policy action for five NRD transitions:

1. take the latest recursively predicted `z2` from the NRD history;
2. drop the oldest student-history entry; and
3. append the latest normalized `z2`.

When an environment resets, replace all four history entries from its newly selected recorded context. Histories must never cross episode boundaries.

## 6. Online DAgger algorithm

The teacher and student run against the same frozen vectorized NRD environment. The environment retains `z1` internally for dynamics, reward, termination, and teacher queries, but `z1` never enters the student observation.

For DAgger iteration `i`:

1. collect a vectorized rollout;
2. query the teacher and student at every policy step;
3. store the student observation with the teacher action label;
4. choose the executed action from a teacher–student mixture;
5. advance the NRD and update the student `z2` history;
6. train the student on the aggregated replay buffer; and
7. repeat with a smaller teacher-execution probability.

At every policy step:

```python
teacher_obs = build_teacher_observation(current_z1, goal)
teacher_action = clamp(teacher(teacher_obs), -1.0, 1.0)

student_obs = build_student_observation(z2_history, goal)
student_action = student(student_obs)

replay_buffer.add(student_obs, teacher_action)

use_teacher = random_uniform() < beta
executed_action = where(use_teacher, teacher_action, student_action)

env.step(executed_action)
update_student_z2_history()
```

The teacher is queried and stored even when the student action is executed. This is the essential DAgger operation: the teacher labels states visited under the evolving student-controlled distribution.

## 7. Teacher-mixture schedule

Use a simple linear schedule over the first 50 DAgger iterations:

\[
\beta_i=\max\left(0,1-\frac{i}{50}\right),
\]

where `beta_i` is the probability of executing the teacher action independently for each environment and policy step.

- Iteration 0 is fully teacher-controlled and supplies the randomly initialized student's first supervised batch.
- Student control increases continuously during iterations 1–49.
- From iteration 50 onward, rollouts are fully student-controlled while the teacher continues to label every visited state.

Log the actual teacher-control and student-control fractions. The final student must be evaluated with `beta = 0`.

## 8. Replay buffer and optimization

Store only the information needed for supervised distillation:

```text
student observation: float32 [258]
teacher action:       float32 [1]
```

Use a FIFO replay buffer so that the dataset aggregates several recent student policies without growing without bound. Initial settings:

```yaml
dagger:
  num_envs: 4096
  rollout_steps_per_iteration: 24
  teacher_decay_iterations: 50
  replay_capacity_samples: 500000

student_training:
  optimizer: Adam
  learning_rate: 0.001
  batch_size: 8192
  epochs_per_iteration: 5
  max_grad_norm: 1.0
```

The single baseline loss is action imitation:

\[
\mathcal L_{\mathrm{action}}
=
\operatorname{SmoothL1}(a^S_k,a^T_k).
\]

Shuffle samples before every training epoch. Save the student model, optimizer, iteration, schedule state, and configuration at regular intervals. The replay buffer itself need not be included in routine checkpoints if it is too large.

## 9. Goal and environment distribution

Match the successful teacher experiment exactly:

```yaml
goal:
  theta_range_deg: [180, 360]
  r_min_frac: 0.5
  r_max_frac: 0.8

task:
  action_repeat: 5
  policy_rate_hz: 10
  max_episode_steps: 50
  success_tolerance_m: 0.02
```

Use the same training context bank and frozen NRD checkpoint as the teacher. Do not expand to upper-half or full-circle goals during this distillation experiment because the selected teacher was trained and validated on lower-half goals.

## 10. NRD evaluation

Evaluate the deterministic teacher and deterministic student on the same held-out `(context, goal)` pairs from the validation context bank.

Run at least:

1. teacher in NRD with privileged `z1` observation;
2. student in NRD with four-step `z2` history and goal;
3. action-agreement evaluation on the states visited by the teacher;
4. action-agreement evaluation on the states visited by the student.

Report:

- reaching success at 2 cm;
- closest-approach success curves at 1, 1.5, 2, 3, and 5 cm;
- final and minimum goal distance;
- time to success;
- timeout, spin, OOD, and non-finite rates;
- action magnitude, slew, and saturation;
- teacher–student action MAE, RMSE, and p95 absolute error;
- action sign disagreement rate; and
- success overlap: both, teacher-only, student-only, and neither.

Action imitation error is a diagnostic. Closed-loop success under fully student-controlled rollouts is the primary result.

## 11. Acceptance criteria

The first distillation experiment passes when all of the following hold on the fixed held-out NRD pairs:

1. student success is within 5 percentage points of the teacher;
2. the student remains stable after `beta` reaches zero;
3. student spin, OOD, and non-finite rates do not exceed the teacher by more than 1 percentage point;
4. median closest approach is within 5 mm of the teacher; and
5. the result is reproduced with at least three student initialization seeds.

The teacher's previously reported 87% result was measured on 100 paired cases. The distillation comparison must re-evaluate both teacher and student on the exact same selected held-out pair set rather than treating 87% as an immutable reference value.

## 12. Implementation changes

### Environment support

Extend `DPendNRDReachEnv` or add a thin distillation wrapper that exposes separate observation paths:

```text
teacher observation = normalized z1 + normalized goal + exact goal error
student observation = four normalized z2 values + normalized goal
```

The existing `build_observation` cannot be reused directly for the student because it always includes normalized `z1` and the `z1`-derived exact goal error.

Add a per-environment four-step student latent history initialized from the recorded reset context and updated at the policy rate.

### Distillation runner

Add one training entry point responsible for:

- loading the frozen teacher and its empirical normalizer;
- constructing the randomly initialized student;
- managing the teacher-mixture schedule;
- collecting online labels;
- maintaining the replay buffer;
- optimizing and checkpointing the student; and
- writing TensorBoard and JSON summaries.

### Evaluation

Extend the paired NRD evaluator or add a student-specific evaluator that reconstructs the same four-step history from each held-out context and updates it at the policy rate.

## 13. Execution order

1. Add separate teacher and student observation builders.
2. Add and unit-test four-step history initialization and episode-reset behavior.
3. Load the teacher and verify that its deterministic actions match the existing evaluator.
4. Add the student MLP and supervised update test on a synthetic batch.
5. Add the online DAgger rollout and replay buffer.
6. Run a small smoke test with few environments and iterations.
7. Run the full DAgger schedule until student-controlled success plateaus.
8. Evaluate teacher and student on identical held-out NRD pairs.
9. Repeat the full student training for three random seeds.

## 14. Interpretation

The fixed Study-1 camera scene is nearly determined by `z1`, so this is a controlled policy-distillation test rather than evidence that vision contributes new task information. A successful result shows that the frozen camera latent and a short temporal window can act as the observation interface for a policy originally trained with privileged physical state.

If the student fails, first distinguish between:

- high teacher-action imitation error, indicating an optimization or information problem; and
- low imitation error but poor closed-loop success, indicating compounding error or insufficient student-state coverage.

The latter case should be addressed first by running more fully student-controlled DAgger iterations and checking replay-buffer coverage, without changing `H`, adding privileged student inputs, or changing the NRD model in the baseline experiment.
