# Bouncing-ball NRD study

The parallel study lives in `/home/harry/NeDM-bouncing-ball`, branch
`study/bouncing-ball`. Collection, training, preparation, and substantial
evaluation ran on AMD. Luffy was used for source edits, small checks, and
viewing returned artifacts.

## Current precision result, 2026-10-01

The preferred checkpoint is now `event_nrd_v3`, saved locally at
`artifacts/training_runs/bouncing_ball_precision_20261001_amd/runs/certified_v2/best.pt`.
On **900 newly collected test trajectories**, the matched 100 Hz comparison
reduces p95 whole-trajectory position RMSE from **30.91 mm to 0.806 mm (38.3x)**
and p95 full-horizon endpoint error from **60.36 mm to 1.355 mm (44.5x)**.
The model was selected on validation before reading this fresh test cohort.
All trajectories retain the required bounce sequence. Checking the same 900
trajectories at the raw 0.5 ms recording rate gives a maximum position error
of **2.141 mm** over every episode and time sample.

The five existing 1.7 s targets now have independent physical Chrono misses
of **0.101–1.166 mm**. Training, simulation, evaluation, and rendering ran on
AMD; the shared checkout and original baseline artifacts remain unchanged.
See [precision experiments and reproduction](bouncing_ball_precision.md) and
[launch optimization](bouncing_ball_launch_optimization.md) for the model,
losses, ablations, exact targets, and limitations. The tenfold milestone is
met; a uniform hundredfold trajectory improvement is not established.

## Original baseline, retained

AMD root: `/work1/dannegrut/harry/experiments/ball_span_v1_20260930`.
The accepted model is `runs/nrd_v2_certified/best.pt`; its local copy is
`artifacts/training_runs/bouncing_ball_nrd_v2_amd/best.pt`.

The final evaluation uses 900 validation episodes and **900 fresh test
episodes**, with full recursive rollouts from launch. The model was fixed
before the fresh test evaluation.

| Fresh-test metric | p95 | Maximum |
|---|---:|---:|
| Whole-trajectory position RMSE | 3.09 cm | 4.10 cm |
| Endpoint position error | 6.02 cm | 8.05 cm |
| Velocity MAE | 0.0213 m/s | 0.0276 m/s |
| Spin MAE | 0.150 rad/s | 0.208 rad/s |
| Penetration | 0.338 mm | 1.924 mm |

Every fresh-test episode has finite predictions and the intended ground →
wall → rebound sequence. Velocity/spin errors exclude +/-20 ms around true
impacts; position errors include every saved state. Contact classification
uses surface geometry and velocity reversal: a wall's tangential impulse can
reverse vertical velocity. Two regression tests cover this distinction.

`evaluation.json` stores episode metrics and checkpoint/data/source hashes.
`impact_timing.json` compares 100 Hz completed-impact frame boundaries against
raw Chrono event times. `launch_gradient_check.json` checks initial-velocity
autograd against finite differences. `figures/` contains coverage, held-out
errors, and comparisons with the five approved episodes.

At 1.7 s, the float64 local derivative check matches finite differences to a
maximum absolute error of 2.11e-7 on 16 validation launches. A larger 0.001 m/s
perturbation can cross a contact sampling frame and give a different secant
slope. The first five-target launch-optimization experiment now has independent
Chrono replay; see [launch optimization](bouncing_ball_launch_optimization.md).

## Physical and data contract

The state is `[x_m, z_m, vx_mps, vz_mps, omega_y_radps]`. The launch action
`[vx_0, vz_0]` is applied once; subsequent motion is passive. The scene has a
0.1 m radius, 1 kg sphere launched at z=1 m, a floor at z=0, and a wall front
at x=5 m. A planar joint permits x/z translation and y rotation. Initial spin
is zero. Geometry and materials are fixed.

`configs/bouncing_ball/chrono_v1.json` specifies Chrono 10.0.0, NSC/ADMM,
touching-only contacts, zero collision envelope, friction 0.03, rolling
friction **0.001 m**, spinning friction 0.0001 m, and restitution 0.9. Physics
uses 0.125 ms steps and records every 0.5 ms. Acceptance requires a ground
bounce, wall collision, rebound, and stopping before the second ground impact,
with penetration and planar-motion checks. Rejections are preserved.

The primary campaign collected **7,200/7,200 accepted episodes** on a 30 by 15
grid: vx=[4,7] m/s and vz=[-10.5,-9] m/s. Each cell has 12 train, 2 validation,
and 2 initial test episodes, assigned before simulation. An additional
**900/900 accepted** episodes from seed 202609302 provide two fresh test
launches per cell. Thus 8,100 physical episodes were collected; the original
900 test episodes remain as diagnostics.

`final_data/model_data.npz` preserves the 5,400 training and 900 validation
episodes and replaces the original test cohort with the 900 fresh episodes.
Model data are 100 Hz and stop at least **20 ms before the second ground
impact**, giving horizons of 1.88–2.18 s. This keeps evaluation inside the
intended two-impact domain. Complete original raw traces remain on AMD.
Padding is masked in losses/metrics. Contact labels supervise training and
are not inference inputs.

`campaign_index.json` maps packed indices to metadata, raw source roots, CSV
hashes, and simulator provenance. Local final data are under
`artifacts/datasets/bouncing_ball_final_amd`. The full-horizon packet used to
fit neural weights remains in `artifacts/datasets/bouncing_ball_span_v1_amd`;
the checkpoint records that training-data hash separately from the final
evaluation-data hash.

## Original model and shared scope

`src/nedm/bouncing_ball/model.py` reuses the shared continuous transformer with
one state token. Gravity and free-flight integration are explicit. The neural
head learns each contact's `[delta_vx, delta_vz, delta_omega_y]` and gate
calibration. Estimated impact time determines the impulse's position effect.
Two internal time-to-impact features derive from the five state values.
Inference requires no target, true contact, future state, restitution,
friction, or simulator call.

Accepted weights come from 30,000 contact-balanced one-step updates.
Normalization uses training episodes only. Selection and localization
calibration use complete validation rollouts. Final smooth gate width is
3e-8 m. PyTorch autograd propagates through recursive rollouts; Chrono supplies
data rather than derivatives.

Validation covers interpolation within the launch range and fixed scene.
Geometry/material changes, a different initial-spin range, or further bounces
require new data and validation. A fixed 1.7 s rollout is after wall rebound
and before the next ground contact throughout this domain. This endpoint is
used by the first launch-action optimization experiment.

## Reproduction and provenance

Cluster scripts use the existing Chrono 10 binary bundle and the PyTorch
2.10.0 module/Python 3.12 venv on `mi3501x`. Each attempt has separate output
and code snapshots. Workflow reference:
`/home/harry/NeDM/.claude/skills/launch-amd-cluster-training/SKILL.md`.

The sequence is preflight, primary collection, GPU smoke, fitting with
`nrd_v2_train.json` and `--validation-only`, validation calibration, fresh
collection with `certification_v1.json`, final-data preparation, fixed-model
evaluation, and plots. `BALL_DATA`, `BALL_CAMPAIGN`, `BALL_CHECKPOINT`, and
`BALL_RUN` select roots. Outputs refuse overwrite; download current accepted
outputs rather than rerunning into them.

Key jobs: preflight **445715**; primary collection **445723**; fitting
**445758** (selected warmup update 30000); fresh collection **445798**; final
data **445802**; calibration **445803**; certification **445806**.
Plotting and gradient checks completed in job **445880**; plotting dependencies
are isolated under the campaign's `plot_deps/`, with versions in
`plot_runtime.txt`.
Preflight reproduced the five approved Chrono trajectories/event times
exactly. The shared AMD checkout and original local studies were untouched.

Earlier independent position/velocity correction models and full-rollout
refinement attempts failed or degraded validation and remain archived. The
first test cohort exposed rare partial-impulse failures and is diagnostic
only. The early horizon probe used an incorrect contact-count metric; it is
superseded by the geometry-aware evaluator and fresh certification. Raw
physical data and contact settings were unchanged during model refinement.

## Loading

```python
import torch
from nedm.bouncing_ball.model import load_model

model, metadata = load_model(checkpoint_path)
launch = torch.tensor([[5.5, -9.75]], requires_grad=True)
initial = torch.cat([torch.tensor([[0., 1.]]), launch, torch.zeros(1, 1)], dim=1)
trajectory = model.rollout(initial, round(1.7 / model.dt))
# New precision model: [1, 35, 5]. Original model: [1, 171, 5].
launch_gradient = torch.autograd.grad(trajectory[:, -1, :2].sum(), launch)[0]
```
