# Pool-ball NRD: results (night of 2026-10-02)

## Summary

- **System.** Two pool balls on a 9-ft table with four cushions, simulated in Chrono (penalty contact). Ball A is launched with `(vx, vy)` from the head spot and hits ball B, which rests on the centre spot; B usually rebounds off a cushion. Real pool physics: sliding then rolling, spin, cushion nose at 1.27 R.
- **Data.** 24,000 shots (16.7 simulated hours), plus a sealed cohort of 2,400 shots, all collected on AMD in about 35 minutes of wall time. Quality and diversity checks pass.
- **Model.** The NRD idea from the bouncing ball: a Transformer backbone for smooth motion (run per ball, output anchored at rest), learned contact switches that read only the predicted state, and bounce networks for the correction at each contact. The selected design has one ball-ball module and one module per cushion: 11 networks, 3.85 M parameters.
- **Accuracy on the sealed cohort.** B's error at t = 2.0 s from a free 200-step rollout: median 1.85 mm, p95 9.1 mm. Contact events correct in 99.1 % of shots. The literal port of the ball-study winner gives median 50 mm; Transformer-only gives 351 mm.
- **Gradient targeting, checked in Chrono.** 100 sealed targets: choose `(vx, vy)` so that B is at the target at t = 2 s.
  - With Levenberg-Marquardt on the model's autograd Jacobian: **99 / 100 within 1 cm, median miss 1.4 mm**. The 1 cm milestone is met.
  - Plain gradient descent, the originally planned method: 83 / 100.
  - With both start spots fixed, interpolating the 19,200 training shots is more precise still (median 0.26 mm). With 1,920 shots, the model (94 / 100) and the interpolated lookup (95 / 100) are about level.

## Data (AMD, Chrono SMC, 12.5 us physics step, 1 ms records)

| Item | Value |
|---|---|
| Main campaign | 24,000 shots: 19,200 training, 2,400 validation, 2,400 in-campaign test (diagnostic only) |
| Sealed cohort | 2,400 shots, new seed 202610029. Only its quality/diversity summary was produced at collection time; no model saw it until the 15 candidates were frozen |
| Accepted | 24,000 / 24,000 and 2,400 / 2,400 (no rejections) |
| Launch range | speed 1.5–3.0 m/s; aim within the 65° cut-angle cone; 30 x 40 grid, jittered within cells |
| Episode | 2.5 s; target time t = 2.0 s |
| Simulated time | 16.7 h (main) + 1.7 h (sealed) |
| Collection wall time | main: 25 min on 5 MI350X nodes (120 workers); sealed: 8 min on 2 nodes |

Quality across all 24,000 shots:
- Penetration is at most 0.57 mm.
- Height above the cloth is at most 0.49 mm.
- Energy never rises between contacts.
- B moves less than 0.1 mm/s before the hit (acceptance check).
- 0.6 % of shots have a second A-B hit.

Diversity: see `artifacts/pool_ball/data_v1/data_report.png`.
- B's departure direction spans ±58°.
- By t = 2 s, B has hit 0 / 1 / 2 cushions in 28 / 52 / 20 % of shots.
- A is already rolling at impact in 42 % of shots, and sliding at up to 1.6 m/s in the rest.
- B's position at t covers the right half of the table.

Sample Chrono shots: `artifacts/pool_ball/data_v1/sample_shots.mp4`.

## Training iterations tonight, and what each one showed

1. **First full run, recipe ported from the ball study.**
   - The smooth Transformer core trains well: p95 one-step error is about 1e-5 m in position and about 2e-4 m/s in velocity per 10 ms.
   - The contact stage failed: the bounce network's data-fitted linear part had ±11,574 weights on two nearly identical inputs (A's and B's side spin after a hit). A tiny drift of A's spin in a rollout then produced a 363 m/s ball.
   - Fixed with a ridge-regularised fit and per-channel input floors.
2. **L-BFGS on the bounce networks** (it gave the ball study its precision) destroyed rollouts here every time. It was dropped.
3. **Longer bounce training made rollouts worse.**
   - Diagnosis, on the literal-port run (v3): from 10k to 80k updates the one-step loss fell from 132k to 53k, while B's p95 error at t (target-worthy validation shots) grew from 144 to 398 mm.
   - In rollouts, the learned contact switch sometimes fires one window early or late. The bounce networks had only been trained on windows that contain a contact, so they extrapolated wildly there.
4. **Fix: also train each bounce network on windows up to 30 ms before and after its contacts**, with the switch forced on and the true (near-zero) change as the target. The correction then fades out at the edge of the contact region.
5. **More accurate smooth core.** Another 100k core steps at batch 4096 and learning rate 1e-4 (decaying).
   - p95 one-step velocity error for B fell from 1.8e-4 to 6.5e-5 m/s.
   - The sliding-phase bias fell 10x, to -3e-6 m/s per step.
   - On top of the same contact stages, B's p95 error at t (target-worthy validation shots) went from 23 to 17 mm.
6. **Longer rollout refinement of the bounce networks** helps here, unlike in the ball study. Half-second or one-second rollouts were run for 2,000-3,000 updates instead of 300. B's p95 error at t on target-worthy validation shots fell a further:
   - 12-15 % on the refined-core models (per-cushion: 10.9 to 9.5 / 9.4 mm);
   - 28-32 % on the first-core model.
7. **One cushion module per cushion** (4 modules shared by both balls, each with its own learned switch) instead of one shared cushion module.
   - B's p95 error at t went from 17 to 10.9 mm on the same core, before the long refinement.
   - Events were right in 98.75 % of validation shots.
   - A single cushion network had to cover four cushions, two balls and every phase within the step. Per-cushion networks each get a simpler function.
   - The -x cushion module (behind A's start) never sees a contact in this data, so only three cushion modules are actually trained.

### Validation summary (all 2,400 validation shots)

Columns: B's p95 error at t on the 1,088 target-worthy validation shots, and events right on all validation shots.

| Model | p95 at t (target-worthy) | events right (all) |
|---|---|---|
| Transformer only (no contact branch) | 643 mm | 0 % (by the cushion-count check) |
| Literal port: joint core, one switch, one shared bounce NN, v4 recipe (v3 recipe: 110 mm, 83.5 %) | 172 mm | 77 % |
| Per-ball core, one switch, one shared bounce NN, v4 recipe (v3 recipe: 106 mm) | 263 mm | 74 % |
| Structured: A-B module + shared cushion module (v4) | 23 mm | 97 % |
| Same with 10 % of the training shots | 27 mm | 96.5 % |
| Structured on the refined core | 17 mm | 98 % |
| Per-cushion modules on the refined core | 10.9 mm | 98.8 % |

The one-switch baselines were frozen with the v4 recipe. Its near-contact training was designed for the structured model, and it made those baselines 1.6-2.5x worse on validation than their v3 versions. Even their better v3 numbers (about 105-110 mm) are about 10x the per-cushion model's.

## Selection and sealed certification

The final model was chosen by validation score alone: **per-cushion contact modules on the refined Transformer core, with one-second rollout refinement** (`structured4_refcore_rr10`). It is the checkpoint at update 1,500 of a 2,000-update refinement, picked by validation score: 0.01272. The half-second-refinement twin scored 0.01282, a gap within checkpoint-to-checkpoint noise.

All 15 candidates were then frozen (checkpoints and SHA256s in `frozen/` on AMD; local copy of the hashes in `artifacts/pool_ball/certification/`). After that, each was evaluated on the 2,400 sealed shots (seed 202610029).

One more model, the per-cushion small-data run, was frozen and scored on the sealed cohort later (04:12), after the main results had been seen. Its configuration was fixed at 02:56, and it played no part in selection. Its sealed numbers are a second look and are reported separately below.

B's error at t = 2.0 s, free rollout from launch, 10 ms grid, sealed cohort (n = 2,400; target-worthy subset n = 1,088):

| Model | median | p95 | target-worthy p95 | shots > 10 mm | B path error (median / p95) | events right |
|---|---|---|---|---|---|---|
| **Selected: per-cushion modules, refined core, 1 s refinement** | **1.85 mm** | **9.1 mm** | **7.5 mm** | **105 (4.4 %)** | **1.1 / 5.1 mm** | **99.1 %** |
| Same, 0.5 s refinement | 2.67 | 9.7 | 8.2 | 112 | 1.5 / 5.9 | 99.2 % |
| Per-cushion modules, refined core, standard refinement | 2.40 | 12.1 | 9.6 | 156 | 1.5 / 7.2 | 99.0 % |
| Shared cushion module, refined core, 1 s refinement | 3.88 | 20.2 | 15.2 | 410 | 1.9 / 9.5 | 98.3 % |
| Shared cushion module, first core | 9.49 | 30.7 | 22.4 | 1,116 | 5.9 / 15.9 | 96.9 % |
| Same with 10 % of the training shots | 8.82 | 36.8 | 26.1 | 980 | 4.8 / 17.3 | 96.5 % |
| Per-ball core, one switch, one shared bounce NN (v4 recipe) | 59.8 | 545 | 235 | 2,323 | 31.7 / 222 | 72.8 % |
| Literal port of the ball-study winner (joint core, one switch, shared bounce NN; v4 recipe) | 50.3 | 348 | 188 | 2,230 | 25.3 / 139 | 76.7 % |
| Transformer only (no contact branch) | 351 | 741 | 647 | 2,398 | 215 / 414 | 0 % |

The sealed numbers follow the validation ranking, and selection was not revisited after seeing them.

## Data and compute cost

| Item | Amount |
|---|---|
| Simulated time | 16.7 h main campaign (24,000 x 2.5 s) + 1.7 h sealed cohort (2,400 x 2.5 s) |
| Physics cost | about 7.6 CPU-seconds per 2.5 s shot (12.5 us steps, Python stepping loop) |
| Collection wall time | 25 min for the main campaign on 5 MI350X nodes (120 CPU workers); 8 min for the sealed cohort on 2 nodes |
| Selected model, one MI350X | 2.1 h end to end: core 25 min, core refinement 23 min, switches + bounce networks + standard refinement 37 min, long refinement 40 min |
| Selected model size | 3.85 M parameters (float64) |
| All training tonight | about 12 node-hours (40 training jobs, most of them diagnosis and controls) |
| Everything tonight (AMD) | 16.2 node-hours (allocation now about 823 / 1,500 used) |

## Gradient targeting through the frozen model, checked in Chrono (sealed targets)

**Task.** Choose ball A's launch `(vx, vy)` so that ball B is at a target `(x, y)` at t = 2.0 s.

**Targets.** 100 targets from the sealed cohort, a seeded draw from shots that pass the predeclared filter:
- |cut| ≤ 55°;
- one A-B hit;
- B moved at least 0.3 m;
- B at least 8 cm from every cushion at t;
- at most one B cushion hit before t, at least 0.15 s before t and at least 15 cm from a corner;
- A at least 12 cm from B at t;
- source launch inside the search box.

Half have no cushion hit before t and half have one. The optimiser never sees the source launch.

**Search.**
- Loss `|p_B(t) - target|^2` is differentiated through the 200-step model rollout with PyTorch autograd.
- Variables: normalised speed and cut angle, with `(vx, vy)` computed from them inside the graph. Box: speed 1.55-2.95 m/s, |cut| ≤ 60°.
- 18 fixed starts per target: speeds 1.8 / 2.25 / 2.7 m/s x cuts ±8 / ±25 / ±45°.
- **Levenberg-Marquardt** (Gauss-Newton on the 2x2 Jacobian from two backward passes, damped, 40 iterations) and **plain projected gradient descent** with backtracking (as in Newton's diffsim example and the ball study, 100 iterations).
- Among converged starts (model distance ≤ 0.2 mm, one A-B contact predicted), the pick is model-only: fewest B-cushion switch firings, then the least sensitive solution. If no start converges (1 target with Levenberg-Marquardt, model distance 0.25 mm), the smallest model residual is taken.
- **Gradient descent was the planned method** (design notes). Levenberg-Marquardt was promoted to primary after two rehearsals on 30 *validation* targets, before anything was frozen: model convergence 26/30 and 29/30 against 15/30 and 17/30 for gradient descent. The sealed cohort played no part in that choice.
- Every chosen launch was then replayed in Chrono on one node, together with:
  - every other distinct converged launch;
  - both data-only baselines (nearest training shot; local linear fit over the 12 training launches nearest to it);
  - each target's own source launch, as a control. That control measures cross-node reproducibility: the cohort was recorded on other nodes.

| Method (Chrono miss of B at t) | median | p90 | max | within 1 cm | within 5 mm |
|---|---|---|---|---|---|
| **NRD + Levenberg-Marquardt** | **1.41 mm** | **2.97 mm** | 15.0 mm | **99 / 100** | 97 / 100 |
| NRD + gradient descent | 1.97 mm | 13.7 mm | 94 mm | 83 / 100 | 73 / 100 |
| Nearest of 19,200 training shots | 2.94 mm | 7.1 mm | 16.1 mm | 98 / 100 | 75 / 100 |
| Local linear fit around the nearest shot (19,200 shots) | 0.26 mm | 0.79 mm | 6.9 mm | 100 / 100 | 97 / 100 |
| Control: replay of the target's own launch | 0.00 mm | 0.05 mm | 0.33 mm | 100 / 100 | 100 / 100 |

- **The 1 cm milestone is met with the model:** 99 of 100 sealed targets, median miss 1.4 mm.
- **By the target's own route** (its source shot):
  - no cushion before t: 50 / 50 within 1 cm (median 1.26 mm);
  - one cushion: 49 / 50 (median 1.68 mm).
- **By the route the chosen launch actually takes in Chrono:**
  - 77 reach the target with no cushion;
  - 23 reach it after one cushion: 22 / 23 within 1 cm, median 1.85 mm.
- **The single miss** (15 mm) is a model error: the model itself was 0.07 mm from the target.
- **Many targets have several launches** that reach them.
  - Levenberg-Marquardt converged from a median of 7 of the 18 starts, to a median of 2 distinct launches.
  - All 227 distinct converged launches were replayed: 192 (85 %) were within 1 cm.
  - The chosen launch takes a different route from the target's source shot for 29 targets. 28 of them are reached with no cushion where the source used one; one goes the other way, with no alternative.
- **Gradient descent is the weaker optimiser here.** It reached the model tolerance on only 59 of 100 targets.
  - For most of the others it stalled a few millimetres from the target, usually by iteration 15-45: model distance median 3.2 mm. Its step cap, backtracking and stall rule then stopped it.
  - The conditioning at the solutions is moderate: the median condition number of the 2x2 Jacobian in the search variables is 2.6 (90th percentile 27).
  - Levenberg-Marquardt uses the same model gradients (the 2x2 Jacobian, from two backward passes) and converged on 99.
- **With both balls' starting spots fixed, the data alone is a very strong competitor.** Interpolating between the 19,200 recorded shots around the nearest one is about 5 times more precise than the model-gradient search. That is expected for a two-input problem sampled this densely.
- **What the model offers beyond the lookup:**
  - it can be differentiated and rolled out from any state, not only from the recorded launch grid;
  - it carries the full trajectories and events, not only B's position at one time.

### Second look: the same targets with 10 % of the data (post-certification)

A per-cushion model trained on 1,920 of the 19,200 training shots (`structured4_small_data`, same recipe, long refinement in the same run) was frozen at 04:12. It was then run on the same 100 sealed targets with Levenberg-Marquardt. Its baselines use only the same 1,920 shots.

| Method, 1,920 training shots (Chrono miss of B at t) | median | p90 | max | within 1 cm |
|---|---|---|---|---|
| NRD + Levenberg-Marquardt | 4.19 mm | 7.96 mm | 24.3 mm | 94 / 100 |
| Local linear fit around the nearest shot | 2.31 mm | 7.57 mm | 79.9 mm | 95 / 100 |
| Nearest of 1,920 shots | 10.49 mm | 19.4 mm | 26.7 mm | 48 / 100 |

The model's own sealed accuracy also drops: B's error at t has median 4.1 mm and p95 15.0 mm, against 1.85 / 9.1 mm with all the data.

With 10 % of the data, the model and the interpolated lookup are about level: the lookup has the better median, the model the better worst case. The model does not win clearly here; with fixed start spots, its data-efficiency advantage is not demonstrated.

**Search cost on one MI350X.**
- Batched over 100 targets x 18 starts: 499 s for 40 Levenberg-Marquardt iterations (12.5 s per iteration), and 1,238 s for 100 gradient-descent iterations.
- A single target alone: one 2 s rollout takes 0.87 s; rollout plus 2x2 Jacobian 2.6 s; one full iteration with 16 backtracking candidates about 7 s.
- At small batch, GPU kernel-launch overhead dominates (about 150 small kernels per 10 ms step).

## Limitations and caveats

- **Fixed start positions make the data a strong competitor.** With A and B always starting on the same spots, the launch-to-B(t) map has only two inputs. 19,200 shots sample it densely enough that interpolating between recorded shots beats the model-gradient search (0.26 vs 1.41 mm median). The natural next step is to vary B's start, or both balls' starts. A dynamics model should generalise there, while a lookup table would need far more data.
- **The physics is Chrono with explicit modelling choices, not a measured table:**
  - penalty (SMC Hertz) contact, because rigid NSC friction makes sliding balls hop;
  - cloth rolling and spinning resistance added as torques;
  - the "None" tangential model, because OneStep depends on the timestep;
  - cushion nose at 1.27 R, cloth restitution 0.05;
  - per-pair friction and restitution as in `configs/pool_ball/chrono_v1.json`.
- **Hard 0/1 switches make the rollout only piecewise differentiable.**
  - At three test launches, autograd matches central finite differences to about 1e-7 relative at small steps. At one launch the gap grows as the step squared (0.31 at 1e-5 m/s), which means the model's map is sharply curved there, not that a switch was crossed.
  - Larger steps can cross a switch window, where the slopes jump.
  - Plain gradient descent stalls short of the target; Levenberg-Marquardt copes.
- **Tails.** 105 of 2,400 sealed shots (4.4 %) have B more than 10 mm off at t, and the worst is 268 mm.
  - Only 3 of these 105 have a wrong event count. Their B-cushion counts are 0 / 1 / 2 in 28 / 48 / 29 of them. The cause of the rest (contact-timing errors within the correct events, or response error) was not diagnosed.
  - Errors are measured at the 10 ms samples; between samples, near impacts, they are larger and were not measured.
- **Structural choices are learned, not hand-coded physics, but they are choices:**
  - the core runs per ball and is anchored at rest;
  - there are one ball-ball module and four cushion modules, so the model knows there are four cushions (not where they are).
  - Inference uses no table geometry, time-to-impact or friction law: the switches decide from the predicted state alone.
- **The final design came from a sequence of decisions on validation data.** These were the near-contact training, the refined core, per-cushion modules, the long refinement, and Levenberg-Marquardt as the primary search.
  - The sealed cohort was collected early, to save time. Only its quality/diversity summary was produced then.
  - No model saw it until the 15 candidates were frozen. All 15 were then certified, and the selected model was run on 100 sealed targets.
  - A 16th model (small data) was scored on it afterwards, as a labelled second look.
- **Dropped:** the context-8 Transformer variant was cancelled (its core L-BFGS stage was too slow tonight). The ball study found context length made no difference.
- **Nothing is committed.** The code, configs and docs are uncommitted in `~/NeDM-pool-ball` (branch `study/pool-ball`). Large data and all checkpoints are on AMD under `/work1/dannegrut/harry/experiments/pool_ball_20261002`.
