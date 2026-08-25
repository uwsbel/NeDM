# Why the arm ROM takes `[q, qd]` as input (and not `qd` alone)

Reviewer-style challenge: *"`q` is the integral of `qd`. Why feed both to the arm
dynamics model instead of `qd` only, recovering `q` by integration — the way the
vehicle ROM drops `(x, y, yaw)` and integrates them outside the network?"*

Short answer: the ROM's input is the set of coordinates the **one-step dynamics
depend on**; coordinates the dynamics are *invariant* to (cyclic/symmetry
coordinates) are dropped and integrated outside. For the vehicle on flat ground
those are `(x, y, yaw)`; the vehicle model *keeps* roll and pitch, which gravity
and the tire forces depend on. For the arm every joint angle is a non-cyclic
coordinate — gravity `g(q)`, inertia `M(q)`, Coriolis `C(q, qd)`, joint limits,
hull/self-collision **and the actuator itself** (`tau = clamp(Kp (qcmd - q) - Kd qd)`)
all depend on `q` — so `q` is a state, not a nuisance coordinate. Recovering it
from `qd` needs the initial condition, and inside a finite context window (16 x
0.02 s) a `qd`-only network can only see `delta q` over 0.32 s, never *where the
arm is* in the gravity field or how far it is from its PD set-point.

The "range" argument the reviewer might expect is real but secondary: `q` lives
in a compact, well-covered box (joint limits `q0 in [-pi, pi]`, `q1 in
[-0.65, 1.57]`, `q2, q3 in [-1.57, 1.57]` rad), so normalisation and coverage
are well-posed, whereas vehicle `(x, y)` is unbounded and any finite dataset
covers a vanishing fraction of it — a network given raw `(x, y)` learns spurious
position dependence and cannot extrapolate. So: `q` is *necessary* because the
dynamics depend on it, and *safe to include* because it is bounded. (Even where
the vehicle dynamics do depend on position — bumpy/CRM terrain — the fix is local
terrain features / terrain conditioning, never raw world position.)

Note that "predict `delta qd`, integrate `q` outside" is *already* how the ROM
propagates: the deployed 8-D model predicts `[delta q, delta qd]` and
`delta q` is 99.4 % explained by the trapezoid `½ (qd_t + qd_{t+1}) dt` on the
data. The disagreement is only about the *network input*, and that is what the
two experiments below isolate.

## 1. One-step evidence: how much of the acceleration is in `q`?

`scripts/ablations/probe_arm_q_input_information.py` fits matched-capacity MLPs
(3 x 256, 60 epochs, 3 seeds) to the normalised per-step `delta qd` (joint
acceleration) on the processed 8-D cache `arm_dyn_v3_8d_seq16_v1` from different
input feature sets. Held-out (val split, 110 697 windows) MSE in normalised units,
`R^2` per joint (mean over seeds; seed sd <= 0.0013):

| input to the regressor                    | dim | val MSE(`Δqd`) | R² per joint            |
|-------------------------------------------|----:|---------------:|-------------------------|
| `[q, qd, qcmd]`  (deployed token)         |  12 | **0.0240**     | .964 .985 .984 .970     |
| `[qd, qcmd]`     (drop `q`)               |   8 | 0.4371 (18×)   | .580 .650 .492 .371     |
| `[qd, qcmd]` × 16 steps (qd-only ctx-16)  | 128 | 0.0360 (1.5×)  | .952 .983 .967 .948     |
| `[q, qd, qcmd]` × 16 steps                | 192 | 0.0338         | .962 .983 .961 .949     |
| `[qd, qcmd − q]` (PD error, no abs. `q`)  |   8 | 0.1818 (7.6×)  | .706 .924 .814 .832     |
| `[q, qd, qcmd − q]`                       |  12 | 0.0191         | .970 .988 .990 .976     |
| `[q1..q3, qd, qcmd − q]` (drop base yaw)  |  11 | 0.0188         | .970 .988 .991 .976     |

Reading:

* **Without `q` the one-step map is not identifiable**: the same `(qd, qcmd)`
  produces very different accelerations depending on `q` (18× the error, `R^2`
  0.37–0.65).
* The decomposition separates the two roles of `q`. Supplying the PD error
  `qcmd − q` but not absolute `q` recovers only part of the gap (0.44 → 0.18):
  the actuator term needs `q`, and so do gravity/inertia (0.18 → 0.02).
* A `qd`-only regressor **with a 16-step history** recovers most of the one-step
  accuracy (0.036 vs 0.024). This is expected and is exactly why the one-step
  metric alone does not settle the question: under a PD actuator the response
  history leaks `qcmd − q`, so the network can do in-context system
  identification of `q`. That shortcut is fragile (torque clamps, joint limits,
  collisions, and — in closed loop — a history made of its own predictions),
  which is what the rollout ablation below tests.
* Dropping base yaw `q0` (given the PD error) costs nothing (0.0191 → 0.0188):
  base yaw is the one near-cyclic joint, as the physics predicts (gravity and
  inertia are yaw-invariant). We still keep it because the deployed action is the
  *absolute* command `qcmd` (so `qcmd − q0` needs `q0`), the joint has a ±π
  limit, hull collision breaks the symmetry, and it costs one input channel.

## 2. Closed-loop evidence: transformer ablation on open-loop rollouts

Same architecture / data / recipe / seed as the deployed 8-D ROM
(`configs/arm_transformer_8d_v1.json`: 5L/8H/256, ctx 16, 80 × 2000 steps,
selection on 0.5 s FK end-effector drift `rollout_sel`). Two new config knobs
(`model.blind_state_fields`, `model.integrated_state_fields`; see
`src/nedm/training/model.py`) implement the reviewer's proposal faithfully:

* **qd-only** (`configs/ablations/arm_transformer_8d_qdonly_v1.json`): the
  network never sees `q_0..q_3`; `q` is propagated outside the network by the
  trapezoid integral of the predicted `qd`, starting from the true `q` at the end
  of the context window; loss on the `Δq` head is zeroed.
* **control** (`configs/ablations/arm_transformer_8d_integq_v1.json`): the
  network sees `[q, qd]` but `q` is propagated by the same trapezoid rule — this
  isolates the *input* effect from the *propagation* scheme.

Open-loop FK end-effector error against the Chrono-recorded `ee_base`
(`scripts/evaluation/eval_arm_rollout.py`, val split, same 400-episode seed-0
subset for all three, checkpoint = `best_val.pt` selected on `rollout_sel`):

| model (network input → propagation of `q`)                 | one-step EE RMSE | 0.5 s (n=216)        | 1.0 s (n=180)        | 2.0 s (n=66)          | trainer `rollout_sel` (best ep) |
|-------------------------------------------------------------|-----------------:|---------------------:|---------------------:|----------------------:|-------------------------------:|
| **deployed**: `[q, qd]` → network predicts `Δq`             | **0.0012 m**     | **0.017 m** (p90 .027) | **0.033 m** (p90 .043) | **0.066 m** (p90 .080)  | 0.0021 (ep 76) |
| control: `[q, qd]` → `q` by trapezoid of predicted `qd`     | 0.0038 m         | 0.061 m (p90 .088)   | 0.106 m (p90 .154)   | 0.295 m (p90 .424)    | 0.0079 (ep 57) |
| **reviewer**: `[qd]` only → `q` by trapezoid of predicted `qd` | 0.0038 m      | 0.182 m (p90 .297)   | 0.534 m (p90 .873)   | 1.710 m (p90 2.386)   | 0.0133 (ep 79) |

(EE RMSE in metres over the rollout horizon; the arm reaches ~2–3 m, and the
mean reference EE displacement over the 2 s horizon is 5.7 m, so 1.7 m RMSE at 2 s is a loss of the
trajectory. `n` = episodes long enough for the horizon.)

Reading:

* **Input effect, propagation held fixed** (rows 2 vs 3): blinding the network
  to `q` costs 3.0× at 0.5 s, 5.0× at 1 s and 5.8× at 2 s. The one-step EE
  error is *identical* (0.0038 m) — the `qd`-only network does learn to infer
  `q`-dependent accelerations from the 16-step `qd`/`qcmd` history at one step,
  exactly as the probe in §1 predicts — but in closed loop that history consists
  of its own predictions and the shortcut compounds; the errors grow super-
  linearly (0.18 → 0.53 → 1.71 m) where the control grows ~linearly. During
  training its `rollout_sel` also got *worse* between epochs 3 and 10 (0.08 →
  0.19) while its one-step loss improved — the one-step fit and the closed-loop
  behaviour decouple without `q` in the input.
* **Propagation effect, input held fixed** (rows 1 vs 2): even the "integrate `q`
  from `qd`" half of the proposal loses 3.6–4.4×. At 50 Hz the recorded `qd` is
  a sample of a signal that moves *within* the 20 ms step (the PD gains are
  Kp = 4000–30000, so a joint responds to a command in a few ms): the trapezoid
  per-step `Δq` residual is 0.0026 / 0.0049 rad on joints 2 / 3 versus 0.0015 /
  0.0014 rad for the network's own `Δq` head, which can use `qcmd_next`. Letting
  the network predict `Δq` (the deployed design) is the better integrator.
* Compounded, the reviewer's variant is 11× (0.5 s) to 26× (2 s) worse than the
  deployed ROM, at identical parameter count, data, schedule and seed.

So the design rule is consistent across both study cases: **the network sees
every coordinate the one-step dynamics depend on and nothing the dynamics are
invariant to** — the vehicle drops `(x, y, yaw)` and keeps roll/pitch and body
rates; the arm keeps all of `q` (base yaw only approximately cyclic, and cheap).
And the "range" intuition is the sufficient condition that makes keeping `q`
harmless: it is bounded by the joint limits and densely covered by the data.


## Reproduce

```bash
PYTHONPATH=src python scripts/ablations/probe_arm_q_input_information.py \
    --output artifacts/ablations/arm_q_input/probe_one_step.json
PYTHONPATH=src python scripts/training/train_hmmwv_dynamics.py --device cuda \
    --config configs/ablations/arm_transformer_8d_qdonly_v1.json
PYTHONPATH=src python scripts/training/train_hmmwv_dynamics.py --device cuda \
    --config configs/ablations/arm_transformer_8d_integq_v1.json
for run in arm_transformer_8d_v1 arm_transformer_8d_integq_v1 arm_transformer_8d_qdonly_v1; do
  PYTHONPATH=src python scripts/evaluation/eval_arm_rollout.py --device cuda \
      --checkpoint artifacts/training_runs/$run --horizons-s 0.5 1.0 2.0 \
      --output artifacts/ablations/arm_q_input/rollout_eval_$run.json
done
```
