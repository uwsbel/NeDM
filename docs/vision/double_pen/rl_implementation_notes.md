# NRD double-pendulum goal-reaching RL — implementation notes

Implements `NRD_double_pendulum_RL_task.md` against the Study-1 joint NRD
(`artifacts/training_runs/dpend_nrd_full_v1`, see `implementation_notes.md`).
Built 2026-08-26.

## What exists

| Piece | Path | Notes |
|---|---|---|
| NRD checkpoint loader | `src/nedm/nrd/checkpoint.py` | shared by the RL env and `scripts/evaluation/eval_nrd_dpend.py` |
| Reset context bank | `src/nedm/nrd/context_bank.py`, `scripts/preprocess/build_dpend_nrd_context_bank.py` | recorded 16-step `[z1, z2, a]` windows; latents from the frozen encoder, encoded once (16 384 train windows in 11 s) |
| RL environment | `src/nedm/rl/dpend_nrd_reach_env.py` | `DPendNRDReachEnv` (rsl_rl `VecEnv`), mirrors `ArmReachingEnv` |
| Training | `scripts/training/train_dpend_nrd_rl_reach.py`, `scripts/training/launch_dpend_nrd_rl_pair.sh` | `--policy-obs z1` (Policy A) / `z1z2` (Policy B); PPO recipe copied from the arm reach study |
| Paired NRD + Chrono evaluation | `scripts/evaluation/eval_dpend_nrd_rl_reach.py` | same held-out (context, goal) pairs for every policy and both simulators; GIFs, closest-approach figure, `summary.json`, `per_pair.json` |

Banks (local-only, `artifacts/rl_reference_sets/`):
`dpend_nrd_full_v1_train_contexts_16384_seed20260826.npz` (resets) and
`dpend_nrd_full_v1_val_contexts_512_seed20260826.npz` (held-out evaluation).
A bank is tied to the NRD checkpoint that encoded it; the env refuses a bank
whose `z2_mean` fingerprint differs from the model's.

## Plan → code mapping

- **Transition model (plan 4).** Every 20 ms: `NRDDynamicsModel.predict_next` on
  the rolling 16-token window → next `z1` (trig pairs renormalized before
  anything else touches them) and next raw `z2`, both fed back. Decoder never
  called. Reset = a recorded window from the bank; the window's last recorded
  action is replaced by the policy's.
- **Policy rate (plan 2).** `action_repeat = 5` → 10 Hz; the action is clipped to
  `[-1, 1]` and held. Pre-clip saturation is logged (`action_saturated_frac`).
  The data's piecewise dwell is 0.1–0.5 s, so 0.1 s holds sit at its short edge.
- **Goals (plan 3).** `θ ~ U(0, 2π)`, `r ~ U(0.5L, 0.8L)`, `x = r cos θ`,
  `z = r sin θ`; resampled while the initial distance is within tolerance.
  `goal.theta_range_rad` / `r_*_frac` are config knobs (`--goal-theta-range-deg`).
- **Observation (plan 5).** `build_observation(z1, z2, goal)` =
  `[normalize_state(z1), normalize_z2(z2)?, g/L, e/L]` with the checkpoint's
  statistics; 10-D (A) / 74-D (B). The Chrono evaluation calls the same
  function with true `z1` and the camera frame's encoding.
- **Reward (plan 7).** Per 20 ms transition, summed over the 5-step hold:
  `-w_d d/L + w_p (d_{t-1}-d_t)/L + B·1[first d ≤ 1 cm] - w_Δω Σ((Δω_i)/σ_{ω_i})²`
  with `σ_ω = state_std[ω]` from the checkpoint (2.92, 6.98 rad/s) and the plan's
  weights (1, 5, 25, 0.01). Once an env has succeeded/failed mid-hold its
  remaining substep rewards are masked. No action penalties (plan: log only).
- **Termination (plan 8).** Success (bonus once, terminate), 5 s timeout
  (50 policy steps), non-finite state/latent, `|ω| > 35 rad/s`, and an OOD guard
  on the predicted latent: any `|normalize_z2(ẑ2)|` above `1.5 ×` the per-dim
  maximum over the bank's encoded real frames (median per-dim max 3.5σ → guard
  ≈ 5.2σ). Timeout / spin / OOD / non-finite are logged separately; only
  timeouts bootstrap the value target.
- **Logged (plan 9).** success, timeout, OOD, spin rates; final/min distance;
  time to success; RMS / max normalized Δω; |a| and slew; reward by term;
  success split by upper/lower-half goals; `min_distance_within_{1,2,5,10}cm`.
- **Chrono transfer (plan 10).** `eval_dpend_nrd_rl_reach.py`: mechanism reset to
  the context's final state (`atan2` of the trig pairs, ω as recorded) with the
  collector's `reset_state` (Setup+Update, bitwise-deterministic replay), camera
  rendered by the manual-trigger pattern once per policy step (100 substeps
  between renders — verified no skipped launches), true `z1` read from the
  bodies, success checked at every 20 ms boundary exactly as in the NRD, and
  additionally the 1 ms closest approach (`min_distance_fine_m`) for the
  discrete-sampling discussion below.

## Two soundness findings (flagged before training)

1. **The goal annulus is not uniformly covered by the training data.** Over the
   full cache (461 k rows) 18 % of tip positions fall inside `r ∈ [0.30, 0.48]`,
   but only 4.7 % of those have `z > 0`: per 10° bin the lower half holds
   1 000–7 700 rows and the top (60°–120°) 65–100 rows. The plan's `θ ~ U(0, 2π)`
   puts ~50 % of goals where the NRD has ~1 % of its data, and the mechanism
   (passive shoulder, 1.5 N·m elbow) has to swing up to reach them — the data
   only visits those poses passively from the uniform-angle initial conditions.
   Under a random policy the env shows success 3 % (lower) vs 0 % (upper).
   The first experiment keeps the plan's distribution; success is logged by half
   so the upper-half ceiling is visible, and `--goal-theta-range-deg` exists for
   a lower-half follow-up.
2. **1 cm tolerance vs. the model and the sampling.** One-hold (0.1 s) NRD tip
   error on 64 val episodes: RMSE 6.9 mm, median 3.3 mm, p90 9.6 mm — the same
   size as the tolerance. And the tip travels a median 34 mm per 20 ms transition
   in the annulus, so a 2 cm-diameter disc is easily stepped over by the 50 Hz
   pointwise check unless the policy slows the tip. The eval therefore reports
   success-vs-tolerance curves from the closest approach (both simulators) and
   Chrono's 1 ms closest approach, so the sensitivity to this choice is explicit.

## Deviation from the plan: failure terminations are charged (found in the first runs)

The plan's reward is negative almost everywhere (`-w_d d/L` per transition,
≈ −5 per second at a typical 0.5 L distance), and the spin guard terminates the
episode without cost. That makes "saturate the elbow torque and spin past
35 rad/s" the cheapest strategy: return ≈ −24 for a 0.6 s episode versus
≈ −150 for surviving 5 s. The first paired runs (`*_plan_v1_*`, logs kept)
found it: by iteration 100 the z1+z2 policy had spin rate 0.99, mean episode
length 6 steps and 93 % saturated actions, and the z1 policy was on the same
path (spin 0.50 and rising while its success rate stalled at 14 %).

Fix (`reward.failure_penalty_mode = "remaining_distance"`, default): a spin /
OOD / non-finite termination is charged `w_d · (d_fail/L) · (remaining
transitions)`, i.e. exactly the distance penalty the episode would have kept
paying at its failing distance until the 5 s horizon — the failure is made
value-equivalent to standing still for the rest of the episode, and no other
transition's reward changes. Verified with a bang-bang controller (spin rate
0.34, failure term −74 per spinning episode). `"none"` restores the plan's
literal reward. The plan's own escape hatches (action-magnitude/rate penalties)
would not have closed this: the exploit is about the horizon, not the action.

## Pivot (2026-08-26, user direction): state-only first, arm-study reward

The plan's shaping did not get off the ground even with the failure charge:
the `plan_v2` pair plateaued at 13–17 % success (lower-half goals 26–34 %,
upper-half 0 %) by iteration 200, with rsl_rl's adaptive-KL learning rate
slamming between 1e-2 and its 1e-5 floor. The user asked to (1) run one job
at a time and get the state-only policy working before touching z1+z2, and
(2) take the reward from the arm reach study
(`arm_reach_adaptivekl005_lr1e4_tol005_ep150_bonus150_sigma015_8d_rom_20260727`),
which has no termination penalty at all.

Env `reward.type = "exponential"` (trainer `--reward-preset arm`), scale-mapped
from the arm (workspace ≈ 1.5 m → L = 0.6 m):

| | arm study | pendulum preset |
|---|---|---|
| reach reward | `exp(-e / 0.15 m)` per step | `exp(-d / 0.06 m)`, averaged over the 5-transition hold |
| action-rate penalty | `0.02 · Δa²` | same (a ∈ [-1, 1]) |
| success bonus / tolerance / steps | 150 / 5 cm / 1 | 150 / 2 cm / 1 |
| failure charge | none | none (all other terms ≥ 0, so early termination has no incentive) |
| Δω smoothness | – | weight 0 |

PPO `--ppo-preset arm`: desired_kl 0.005, lr 1e-4, entropy 0.001, 3 epochs ×
16 minibatches, init noise 0.3, 64 steps/env, 4096 envs, 1500 iterations.
Goals: lower half only (`--goal-theta-range-deg 180 360`) — the same move the
arm study made when it kept goals inside the data-covered workspace (see
finding 1). Run: `dpend_nrd_reach_z1_armreward_lowerhalf_seed1_20260826`.
The plan's 1 cm rate is still reported by the eval from the closest-approach
curve.

## Throughput

`DPendNRDReachEnv`, 4096 envs, full 16-token context, decoder off, RTX 4090:
330 k NRD transitions/s (66 k policy steps/s) alone; ~30 k policy steps/s each
when both paired runs share the GPU. One PPO iteration (4096 × 24 steps) ≈ 3.3 s
shared; 800 iterations ≈ 45 min.

## Results — state-only policy, arm-recipe reward, lower-half goals

Run `dpend_nrd_reach_z1_armreward_lowerhalf_seed1_20260826` (66 k policy
steps/s alone, ≈ 4 s/iteration). In-NRD training success (rsl_rl episode
average): 37 % @ it 23, 64 % @ 100, 73 % @ 200, 82 % @ 300, 84.5 % @ 400, then a plateau at 85 ± 0.5 % through 1500; spin
terminations < 0.1 %, OOD 0. Wall time 1 h 45 min for 1500 iterations (393 M policy steps).

Paired NRD / Chrono evaluation on the held-out val-bank pairs
(`nrd_chrono_transfer_eval_iter*/summary.json`; deterministic policy, 2 cm
tolerance, success checked at every 20 ms boundary in both simulators):

| checkpoint | pairs | NRD success | Chrono success | both / NRD-only / Chrono-only / neither | closest approach median NRD / Chrono | time to success (median) |
|---|---|---|---|---|---|---|
| model_100 | 50 | 54 % | 54 % | 23 / 4 / 4 / 19 | 19.6 / 19.7 mm | 1.9 / 1.5 s |
| model_400 | 100 | 84 % | 86 % | 81 / 3 / 5 / 11 | 15.8 / 16.2 mm | 1.4 / 1.4 s |
| model_800 | 100 | 84 % | 89 % | 82 / 2 / 7 / 9 | 15.2 / 15.0 mm | 1.4 / 1.4 s |
| model_1499 (final) | 100 | **87 %** | **87 %** | 84 / 3 / 3 / 10 | 15.2 / 15.3 mm | 1.4 / 1.4 s |

Success vs. tolerance on the closest approach (model_1499, NRD → Chrono):
1 cm 14 % → 14 %, 1.5 cm 48 % → 48 %, 2 cm 87 % → 87 %, 3 cm 90 % → 90 %,
5 cm 94 % → 94 %. Per-pair |closest approach NRD − Chrono| median 3.3 mm,
p90 10.0 mm — the same size as the NRD's own one-hold error (finding 2), which
is why the plan's 1 cm sits on the steep part of the curve.

Chrono GIFs (`chrono_pair_00*.gif`) show the learned strategy: fold the elbow
to the radius of the goal, let the passive shoulder swing the folded pair
through it (pair #1: 7.8 mm at 2.16 s).

Δω / action statistics are identical between simulators (normalized Δω RMS
0.34 / 0.34, max 0.85 / 0.85; |a| 0.50 / 0.54; slew 0.25 / 0.30), i.e. the
policy does not behave differently on the true plant. The 10 "neither" pairs
are timeouts in both simulators with closest approach 3–10 cm — goals the
folded swing does not pass through within 5 s from that context, not model
error.

**Status of the plan's success criterion.** The state-only half is done: a
policy trained entirely inside the frozen decoder-free NRD transfers to Chrono
with no measurable gap (87 % / 87 %, 84 of 87 successes shared). Not yet run,
by user direction: Policy B (`--policy-obs z1z2`) under the same recipe, the
full-circle goal distribution, and the 1 cm tolerance. Earlier runs kept for
the record: `*_plan_v1_*` (spin exploit, killed at it ~100/110),
`*_plan_v2_*` (failure charge, plateau 13–17 %, killed at it ~250) and
`*_lowerhalf_v2_*` (plan reward, lower half, killed at it ~150).
