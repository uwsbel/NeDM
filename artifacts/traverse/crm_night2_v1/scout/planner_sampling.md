# Scout: the candidate sampler, and what an iterated resampling planner needs

Read-only reconnaissance, 2026-09-17/18, worktree `/home/harry/NeDM-traverse_mppi` (branch `traverse_mppi`). Two
small offline measurements were run on the workstation (RTX 5090, `/home/harry/miniconda3/envs/nedm/bin/python`):
a timing of the deployed 256-candidate plan on the CRM map/ensemble, and a scoring-only pilot of iterated resampling
(no simulation, nothing written under `artifacts/` except this file; scratch at `/tmp/time_planner.py`,
`/tmp/cem_pilot.py`, `/tmp/cem_pilot.json`).

## 1. Prior results that settle or bound the question (do not repeat)

| Date / record | What was tested | Conclusion that matters tonight |
|---|---|---|
| 09-10 "MPPI closed-loop demo", memory `f104-fixed-arena-study-state` (group_0200, `ReferenceMPPI.optimize`, 506 proposals, 48 driven, 43 failed) | the real iterated MPPI in `src/nedm/traverse/fdm_mppi.py` with the multi-head model | failure was the **sin^2 envelope** of `deform_reference` (`fdm_mppi.py:88`): lateral and speed perturbations forced to zero at both route ends, so all 506 proposals were the same plan 5 m before the goal (0.155 m lateral spread, 1.77-2.28 m/s). Not a verdict on iteration. |
| 09-10 "confirmed fix" (292 episodes, 73 groups, 2 m/s) | tent (`np.interp`) vs smoothstep knot interpolation | smooth C1 knots: model-best failure 23.3% -> 12.3%, median lateral reach 0.88 -> 2.53 m. |
| 09-10 **sampling-width A/B** (219 episodes, 73 groups, 2 m/s) | narrow (knots=3, 2.53 m) vs wide (knots=2, 4.70 m) vs band 2.5-4.5 m | 12.3% / 16.4% / 15.1%, indistinguishable; **knot count, not sigma, controls reach**; the ceiling was the model's geometry discrimination (AUC ~0.77), not the search. A post-hoc "3-4.5 m sweet spot" did NOT replicate. |
| 09-12 night 2, `night2_v1/{LOG,REPORT}.md` | night-1 proposal (3 knots, sigma 2.6, clip +-6, sin^2 speed envelope) vs night-2 sine-basis sampler, 184 fresh + 300 hazard groups | proposal change with model fixed: unsafe 8.2 -> 0.5% (11 vs 0, p=0.001); hazard set 9.7 -> 0.3% (28 vs 0). Night-1 acceptance was 48.8% (the "2%" claim is wrong); what limited it was the +-6 m clip and the envelope. The 9 designed anchors contribute nothing (+0.3 pts, 1 vs 0, p=1.0). **Pessimistic ensemble (max over members) was the only arm with 0 failures / 0 unsafe.** |
| 09-15 gen_v1 (`gen_v1/REPORT.md`, `test_results.json`) | deployed one-shot planner on f104 + 5 sibling arenas, 200 groups each | **f104 speed-free is at the floor: 0/200 unsafe** (rule 0.5%, straight-6 1.0%). Headroom only at fixed 2 m/s (f104 2.5%, new arenas 5.9%) and speed-free on new arenas (1.3%). Time is not in the objective; the model is ~4 s slower per goal than the hand rule. |
| 09-17 CRM `crm_f104_v1/REPORT.md` | same one-shot planner, CRM soil, 200 hazard pairs | CRM-trained planner goal reached 91.0% (pess 92.0%), rigid-trained 68.0%, straight-6 66.5%, fixed-2 75.0%. **9% failures = the only large realised headroom on f104 for a planner-only change.** Failures cluster at ~12 spots / 9 features. |
| mppi_claude sibling (memory `mppi-claude-route-scorer-state`, `~/NeDM-mppi-claude/src/nedm/traverse/rc_planner.py`) | route MPPI with 3 offset knots + 4 speed knots, ESS-tempered softmax, 3 iterations x 64 first plan | sealed: scorer+MPPI 356/393 vs heuristic 315, time-only MPPI 344; "most closed-loop gain = route re-optimisation (direct/A* at high speed) rather than the learned model"; replanning added nothing. Reusable code, not a result about this pipeline. |

Nothing has ever tested **iteration** (resampling around elites) with the night-2 parameterisation: every night-2,
gen_v1, sensor, nav and CRM pick is a single draw of 256 (`f104_n2_cand.py:49-55` builds even the night-1 control by
single draws, not by `ReferenceMPPI.optimize`).

## 2. The deployed sampler, precisely

### 2.1 Files and signatures

- `scripts/f104_n2_sampler.py` (108 lines): `A_ACC, A_DEC = 1.5, 2.0` (l.23), `V_MIN, V_MAX = 0.5, 6.0` (l.24).
  - `smooth_interp(f, k, vals)` l.27: C1 smoothstep knots, zero at both ends (unused by the deployed path).
  - `speed_knots(f, k, vals)` l.35: k knots at `linspace(0,1,k)` (k=4 -> f = 0, 1/3, 2/3, 1), smoothstep between them, **free ends**.
  - `shape(xy, station, speed, lat, dv)` l.43: `pts = xy + lat*normal`; new stations from the deformed points; `v = clip(speed+dv, 0.5, 6)`; terminal cone `v <= sqrt(2*2.0*(L-s))`; forward pass `v[j] <= sqrt(v[j-1]^2 + 2*1.5*ds)`; backward pass with 2.0. Returns `{'waypoints','speeds','stations','headings','meta'}`. `v[0]` is NOT pinned to rest (anchors at 6 m/s command 6 at station 0; the validator only checks accel between waypoints).
  - `lateral_profile(f, L, rng, sigma=5.0, modes=3, kappa_max=0.125, budget=0.55)` l.58: `a_j ~ N(0, sigma/j)`, clipped to `cap_j = 0.55*kappa_max*L^2/(j*pi)^2`; returns `(sum_j a_j sin(j pi f), a)`.
  - `sample_one(base, rng, knots=4, lat_sigma=5.0, lat_clip=10.0, sp_sigma=1.5, sp_clip=4.0, base_speed=None, modes=3, kappa_max=0.125)` l.66: lateral profile clipped to +-10 m; `dv_k ~ N(0, 1.5)` clipped +-4; `f` is the fraction of the BASE station; `meta = {'candidate':'n2_wide','max_lateral_m','mean_speed_mps'}`.
  - `anchors(base, offsets=(0,-4,4), speeds=(2,4,6))` l.81: 9 designed-style routes, `lat = off*sin(pi f)^2`, constant speed + cones; `meta['candidate']='n2_anchor'`.
  - `propose(base, anchor_pose, rng, n=256, validate=None, cfg=None, **kw)` l.96: anchors first (validated), then rejection sampling until n valid or `8*n` tries; returns `(list, tries)`.
- `scripts/gen_planner.py` (323 lines): `CFG = MPPIConfig(max_speed_mps=6., min_speed_mps=0.0, max_curvature_inv_m=.125, arena_half_extent_m=40.)` l.30; `N_CAND = 256` l.32; `set_map(arena_dir)` l.35 (heightmap -> `DS.G`); `base_route(pose, goal, ...)` l.102 (frozen generator route_00 Hermite at 2 m/s, then Hermite scales / Dubins arc fallbacks); `safe_validate(route, obstacles, cfg, anchor)` l.94 (ValueError -> invalid); `proposal_pool(base, pose, rng, n=256)` l.124; `fixed2_pool(base, pose, rng, n=256)` l.130 (geometry only, `sp_sigma=0, base_speed=2`); `corridors(cands) -> (X (n,5,96,32) float32, L (n,))` l.141; `geom_ctx(start_xy, goal_xy, start_yaw, L) -> (n,5)` l.149 = [dx, dy, dist, yaw, L]; `RiskModel(pattern, device)` l.154 with `score(X, ctx5, bs=256) -> (z, p)` l.170: per member normalise `x[:, :4]` by `ck['norm']`, append a constant channel (cin=6), `route_logit` = log sum softplus(hazard); **z = ensemble MEAN logit, p = 1-exp(-exp(z))**; `plan(pose, goal, rng, model, rule, mode)` l.211: `i = argmin(z)`; `HandRule` l.187.
- `scripts/f104_n2_dataset.py`: `station_tensor(wp, sp, st) -> (X (5,96,32), route_len)` l.45; channels `[elev - e0, along grade (clip +-2), cross slope (clip +-2), commanded speed, valid]`, 96 stations x 32 lateral over +-6 m, bilinear `sample_map` l.27 on `G['rgbd'][3]` (elevation channel) x `elev_scale`. `init_map(root)` l.17 reads `<root>/static_map_v1/observation.{json,npz}`.
- `src/nedm/traverse/fdm_mppi.py`: `MPPIConfig` l.18 (samples 128, iterations 2, temperature 1.0, knots 3, lateral_sigma 0.35, speed_sigma 0.4, max_lateral 1.0, max_speed_delta 1.0, noise_correlation 0.6, curvature 0.125, accel 1.5 / decel 2.0, half_width 1.3 / half_length 2.6 (+0.1 margin), arena 40, path_step 0.25); `deform_reference` l.69 (**sin^2 envelope l.88 and tent `np.interp` l.90-91 — the two defects diagnosed on 09-10; do not reuse**); `validate_reference(route, obstacles, config, anchor_pose)` l.121; **`ReferenceMPPI` l.171 exists** (`optimize(route, anchor_pose, obstacles, score, initial_mean)` l.176: `iterations` rounds of `samples` correlated-Gaussian draws, row 0 = mean, row 1 = zero, a quarter speed-only; softmax weights `exp(-(cost-min)/temperature)`; keeps the best validated sample; re-validates the weighted mean at the end and takes it only if its cost is <= the best sample's; reports ESS). It iterates, but over the defective `deform_reference` and with tiny sigmas; no night-2 result used it.
- `~/NeDM-mppi-claude/src/nedm/traverse/rc_planner.py` (363 lines): `knot_profile(points, knots, params, v0=None, switch=None)` l.35 (**v0-aware**: `v[0] = v0` when replanning while moving, launch clamp only from rest; curvature cap `sqrt(a_lat_max/kappa)`; terminal taper), `_pchip` l.72 (monotone cubic offsets, tangential ends), `deform_route` l.96, `make_candidate` l.125 (`_repair_curvature` before rejecting), `MPPIConfig` l.213 (3 offset + 4 speed knots, 64 samples x 3 iters first plan, 32 x 1 replan, `ess_frac=0.25`, sigma floor 0.25 x, offset bound min(12 m, 0.3 L), offset sigma min(2, 0.06 L)), `RouteMPPI.plan` l.279 (per-seed-family Gaussian, mean always in the population, quarter speed-only, softmax update l.320-329, family means re-validated l.331-343), `_weights(J, ess_frac)` l.350 (**bisection on the temperature to hit a target ESS** — the piece to lift verbatim). Cost `J = P*min(t,30) + (1-P)*30 + w*work/10` l.250-252.

### 2.2 Route parameterisation and parameter vector

theta = (a_1, a_2, a_3, dv_1, dv_2, dv_3, dv_4) in R^7 (for `fixed2_pool`: R^3).
Prior: `a_j ~ N(0, (5/j)^2)` = sd (5, 2.5, 1.67) m, `dv_k ~ N(0, 1.5^2)` m/s, clipped +-4.
Caps (validator-derived) `cap_j = 0.55 * 0.125 * L^2 / (j pi)^2`:

| L (m) | cap_1 | cap_2 | cap_3 | share of a_2 / a_3 draws that hit the cap |
|---|---|---|---|---|
| 30 | 6.27 | 1.57 | 0.70 | ~47% / ~67% |
| 40 | 11.1 (then +-10 profile clip) | 2.79 | 1.24 | ~26% / ~46% |
| 50 | 17.4 (profile clip +-10 binds) | 4.35 | 1.93 | ~8% / ~25% |

So on the 29-52 m eval routes the higher modes sit at their caps much of the time: the effective prior in
(a_2, a_3) is bimodal, and a Gaussian refit on raw draws would drift past the caps. Refit on the *clipped* values
that were actually built (the pilot below does).
Speed: `v = clip(2 + dv(f), 0.5, 6)` then cones, so dv > +4 and dv < -1.5 are saturated; the reachable speed set is
[0.5, 6] m/s with the 1.5/2.0 m/s^2 cones and the terminal stop cone. Route time (commanded) `T = sum ds / v_mid`.

### 2.3 Validity (`validate_reference`, `fdm_mppi.py:121-168`)

Crop to the waypoint nearest `anchor_pose` (l.128-130; or `meta['fdm_station']`), then: 3-point discrete curvature
of the waypoints (`planner_s._curvature_max`, l.65) `<= 0.125 /m`; speeds in [0, 6]; `diff(v^2)/(2 ds)` within
+1.5 / -2.0 m/s^2; footprint corners (half 2.6 x 1.3 m + 0.1 m margin) of the 0.25 m-resampled path inside
+-40 m (nav_v1 uses a 37 m planning bound); obstacle clearance (none in f104). The nav_v1 fold-back rule (reject > 45 deg
turn between consecutive points) lives in `scripts/nav_online.py:65-81`, wrapping the frozen validator; it never fires
for single-goal starts that face the goal.

**Measured acceptance** (8 CRM eval cases, L 29-48 m): 56-89% per case, mean 76% (256 valid in 277-444 tries);
2,048 raw draws over 4 cases: 82.3% valid, rejections **98% curvature, 2% arena**, 0 speed/accel (the cones make
speed always valid). Night-2 LOG quoted 73-93% on the longer 184-group set.

### 2.4 Scoring

`z = mean_m log sum_s softplus(h_ms)` over 5 members (`RiskModel.score`); pick = `argmin z` (ties -> lowest index,
anchors come first). **Time and energy are not in the objective.** Pessimistic variant (`crm_pools.py:79`,
`member_logits` l.47-55): `argmin max_m z_m`. On the CRM eval one-shot picks (n=200, `eval_v1/picks/*.json` vs
`runs/*/outcome.json`): median pick logit -6.42 (p = 0.16%); realised failure by pick-logit quartile
**0/50, 1/50, 6/50, 11/50** (z < -7.07 / -6.42 / -4.80 / above); AUC(pick logit -> failed) = 0.84. So on CRM a
lower pick logit does translate into fewer failures, and the 9% sits almost entirely in the groups where the
one-shot pool's best candidate is mediocre (z > -4.8: 22% fail).

## 3. Compute cost, measured (RTX 5090 workstation, CRM map root + `CRM_N2_s*.pt`, 8 eval cases)

| step | per 256-candidate plan | per candidate |
|---|---|---|
| `base_route` | 0.4 ms | — |
| `proposal_pool` (sample + validate, ~330 tries) | **156 ms** | sample_one 0.25 ms, validate 0.19 ms |
| `corridors` (`station_tensor` loop) | **156 ms** | 0.61 ms |
| `RiskModel.score` (5 members, GPU) | 60-70 ms warm (first call 314 ms) | 0.26 ms |
| total | **~0.38-0.40 s** | ~1.5 ms all-in (incl. rejects) |

Cross-check: nav_v1 bench on an MI350X (`nav_v1/LOG.md:54-60`): candidates 0.130 s, corridors 0.181 s, model
0.042 s -> 0.42 s per 256; campaign CPU node 1.4-2.4 s. The corridor loop can be batched exactly like
`scripts/nav_online.py:422 corridors12_batch` (`DS.sample_map` already accepts (n,96,32) arrays), which would take
~0.1 s off each 256-batch; sampling/validation is pure numpy per route (~0.45 ms per try) and parallelises trivially.
An iterated planner therefore costs ~1.5 ms per evaluated candidate: 4x64 = 0.40 s (measured), 8x64 = 0.75 s,
4x256 = 1.56 s.

## 4. Offline pilot: does resampling around elites find lower predicted risk? (24 CRM eval cases, scoring only)

`/tmp/cem_pilot.py`: `from_params(base, theta)` rebuilds `sample_one` from an explicit theta (caps applied as a
projection); round 0 = 9 anchors + Gaussian draws from the prior; elites = best 15% of all valid candidates so far
(min 4) by ensemble-mean z; CEM refit `mu = mean(elites)`, `sd = max(std(elites), 0.15 x prior sd)`; MPPI variant =
softmax over all valid samples with ESS-targeted temperature (ESS = 25% of n, `rc_planner._weights`); rejection
sampling to n valid per round (cap 8n tries); pick = argmin z over everything evaluated. Same rng seed per case.

| arm | evaluations | mean z of pick | delta vs one-shot (mean / median) | cases improved | pess-mean gap of the pick | commanded route time | mean speed | wall |
|---|---|---|---|---|---|---|---|---|
| one-shot 256 (deployed) | 256 | -5.81 | 0 | — | 0.67 | 17.9 s | 3.30 m/s | 0.43 s |
| CEM 4x64 (equal budget) | 256 | -6.57 | **-0.77 / -0.28** | 22/24 | 0.65 | 17.6 s | 3.38 | 0.40 s |
| MPPI-weighted 4x64 | 256 | -6.46 | -0.65 / -0.19 | 20/24 | 0.75 | 16.7 s | 3.51 | 0.39 s |
| CEM 8x64 | 512 | -6.99 | **-1.18 / -0.54** | 24/24 | 0.52 | 15.6 s | 3.66 | 0.75 s |
| CEM 4x256 | 1024 | -6.91 | -1.11 / -0.63 | 24/24 | 0.58 | 15.2 s | 3.56 | 1.56 s |

Reading. (i) At equal budget the iterated sampler lowers the pick's logit in 22/24 cases; the gain is concentrated
where the one-shot pool was poor (case 0003: -1.93 -> -5.76 / -6.68; 0004: -4.36 -> -7.23; 0011: -4.28 -> -6.09;
0013: -3.04 -> -4.93; 0017: -4.32 -> -5.96; 0023: -3.83 -> -5.28) — exactly the quartile that realises 22% failures
on CRM. (ii) Acceptance stays 84-89% through the rounds (the refit sd does not collapse: last-case sd
[1.1, 1.7, 0.6, 0.75, 0.9, 1.0, 0.5] after 4 rounds). (iii) The iterated picks are not slower (route time 15-17.6 s
vs 17.9 s) and command slightly more speed — on CRM, where straight-6 beats fixed-2 by a wide margin, part of the
gain may be re-discovering momentum. (iv) Optimiser's-curse check: the member-disagreement gap (max member minus
mean) of the pick does not grow with search effort (0.67 -> 0.65 / 0.52), so the search is not preferentially
exploiting one optimistic member; whether the new low scores are *real* (the model is right about routes it has
not seen) is what only Chrono can answer. The night-1 lesson stands: the 3 confident misses were each "the only
candidate under 5% among ~100% candidates" — an elite-focused search hunts such lone minima harder, which argues
for running the pessimistic objective as an arm and for scoring the refit mean (not just samples).

## 5. Specification of the iterated sampler (what to build)

New module, e.g. `scripts/f104_n2_iter.py`, importing `f104_n2_sampler` (`shape`, `speed_knots`, `anchors`) and
`gen_planner` (`CFG`, `safe_validate`, `corridors`, `geom_ctx`, `RiskModel`):

1. `from_params(base, theta, lat_clip=10, sp_clip=4, kappa_max=0.125, budget=0.55) -> route` — deterministic map
   theta -> route (as in the pilot; `meta['theta']` stored so refits use the clipped values).
2. `objective(z_mean, z_pess, T, E, mode, lam_t, lam_e)`; default `J = z_mean` (reproduces the deployed planner at
   K=1, n=256). Options: `z_pess`; the expected-cost form `J = P*H + (1-P)*T + lam_e*E` with `P = 1-exp(-exp(z))`,
   H = 60-120 s (this is `rc_planner.cost`, and it removes the arbitrary lam_t); lam_t in logit units otherwise
   (the one-shot pool spans z ~ -7.5 .. +3 and T ~ 6 .. 40 s; lam_t ~ 0.05-0.1 per second is the range where
   time starts to move picks). **E needs the energy head from study 1** (per-route positive work; the label
   exists per episode as `outcome.json positive_work_kj` / `trajectory.npz positive_work_kj_per_interval`); do not
   fake it with a corridor proxy.
3. `plan_iter(base, pose, goal, score_fn, rounds=4, n=64, elite_frac=0.15, sd_floor=0.15, weighting='cem'|'mppi',
   ess_frac=0.25, rng, anchors=True)`:
   - round 0: anchors + `n - 9` prior draws (prior sd (5, 2.5, 1.67, 1.5 x4)); every round: rejection-sample to n
     valid (cap 8n tries), build corridors in one batch, one `score` call (5 members -> z_mean, z_pess), compute J.
   - update: elites = best `ceil(elite_frac * n)` over all evaluated so far (CEM), or ESS-tempered softmax weights
     over all valid samples (MPPI, `rc_planner._weights`); `mu` = (weighted) mean of clipped thetas,
     `sd = max(weighted sd, sd_floor * prior sd)`; optional: keep a quarter of the draws speed-only (lateral at
     mu) as `rc_planner.py:298` and `fdm_mppi.py:224` do — it kept acceptance high when the base route is already
     near the curvature limit.
   - final: build and score `mu` itself (as `ReferenceMPPI` l.238 and `RouteMPPI` l.331 do); pick = argmin J over
     every validated candidate incl. the means; log per-round `zmin`, ESS, acceptance, sd; write the pick with
     `meta = {'candidate': 'n2_iter', 'theta', 'round', 'rank'}` so the analysis can tell rounds apart.
   - multi-start (optional, matches `RouteMPPI` families): one Gaussian per anchor cruise speed (2/4/6) to keep the
     population multimodal; the pilot did fine with one Gaussian because the elites are re-selected from the union.
4. Determinism: `np.random.default_rng(md5(group + tag))` with a NEW tag per arm (the one-shot tags are
   `'crm_proposal'`/`'crm_fixed2'` in `crm_pools.py:36-37` and `'gen_night2'`/`'gen_fixed2'` in `gen_pools.py:42-43`);
   score from the same float16-rounded X the one-shot arm uses (`crm_pools.py:41-43` stores X as float16) so that
   arm A reproduces the eval_v1 picks bit-for-bit.
5. Replanning while moving (for study 1's velocity input): the base route starts at the measured pose, `v[0]` must
   equal the current speed and the forward cone must start from it (`rc_planner.knot_profile(v0=...)` l.53-56 is
   the reference), and the validator's anchor crop (`fdm_mppi.py:128-130`) already drops the waypoints behind the
   vehicle. `shape()` would need a `v0` argument (today `v[0]` is whatever the knot says).

Gotchas: `sample_one` draws lateral then speed from one rng stream, so any change in the sampler changes the
stream — never compare arms by "same seed", compare by group. `f` (knot positions) is the BASE route's station
fraction while the cones use the deformed station — harmless, but keep it identical in `from_params`. Ties in
`argmin` go to the first index (anchors). The GPU's first `score` call costs ~0.3 s (warm up before timing).

## 6. Evaluation design

**Groups.** Primary: CRM, the 200 `cases_eval` hazard pairs (`artifacts/traverse/crm_f104_v1/cases_eval/cases/
f104_crm_eval_group_XXXX.json`, fields `layout.start_xy/start_yaw`, `goal_xy`, `evaluation_stratum`, base route
`routes/<g>/route_00.json`) plus 200-400 fresh pairs from `scripts/gen_cases.py --strata feature --avoid
<cases_night2, cases_eval>` (new seed). Because CRM episodes are **bit-identical across GPU types** (REPORT s.3),
the eval_v1 one-shot `crm` arm drives (and `crm_pess`, `straight6`) are reusable on the first 200 pairs: arm A
needs no new drives there, only the new arms' distinct picks (identical (pool,index) picks share a route file and
are driven once, `crm_pools.py:86-95`). Secondary, rigid: f104 speed-free has **no headroom** (0/200 unsafe), so
use `gen_v1/cases_test_{g228,g203,g217,g216,g231}` (200 each; one-shot 1.3% unsafe speed-free, 5.9% at 2 m/s) and
`cases_test_f104` at fixed 2 m/s (2.5%); rigid arms of a group must share a node (`gen_pools.py:80` shard =
md5(group) % 24; `gen_array.sbatch` one node per array task).

**Arms** (all from the same base route and the same map/ensemble; picks hashed before driving as in
`eval_v1/PICKS_LOCKED.sha256`):
A one-shot 256 (deployed; reproduces eval_v1) · B iterated equal budget, CEM 4x64 · C iterated more budget,
CEM 8x64 (512) · **D one-shot 512 with the same seed family (budget control: separates "iteration" from "more
samples")** · E = B with the pessimistic objective · F (optional, objective study) = B with the expected-cost
objective `P*H + (1-P)*T`, H = 60 s, to measure what time-awareness costs in safety.
Fixed-2 m/s versions of A/B/D on the rigid arenas (theta in R^3).

**Metrics.** Offline, all groups (free): pick logit z_mean and z_pess, rank of A's pick inside B's evaluated set,
commanded route time, max lateral. Realised: CRM primary = goal not reached (`outcome.json status`), rigid =
unsafe (failed or slid back, `f104_n2_dataset.one` rule) and failed; secondary time to goal, max tilt,
`positive_work_kj` (energy — already in `outcome.json`, no model needed to *measure* it). Paired exact McNemar per
pre-registered contrast (B vs A primary; C vs A; D vs A; E vs B), Holm over the family; report the terrain-feature
clustered CI as REPORT s.1 does (failures sit at ~12 spots). Power: A fails 18/200 on CRM; if B moves the worst
two quartiles to the best two quartiles' rate the expectation is ~5/200, i.e. discordant ~14 vs 1 (p ~ 0.001);
with 400-600 pairs the clustered test also resolves.

**Launch.** Rigid: `scripts/gen_pools.py`-style pick pass -> `tasks_test.json` rows `{id, group, arena, case,
route, shard, run}` (paths relative to `GEN_ROOT`) -> `scripts/gen_array.sbatch` (3 h, `GEN_WORKERS = cpus-2`,
`PYTHONPATH` to the frozen `gen_v1/source`, `FDM_RUNTIME_FINGERPRINT`) -> `gen_runner.py` -> `gen_collect.py
--horizon-s 120` -> `gen_analyze_test.py`. 4,000 rigid drives take ~15 wall minutes on ~2,000 workers. CRM:
`scripts/crm_pools.py --cases cases_eval/cases --map-root map_root --models crm=...CRM_N2_s*.pt --case-prefix
cases_eval/cases --route-prefix eval_v2/routes --out eval_v2` (extend with the new arms) -> `tasks.json` rows
`{id, group, case, route, run, tier, episode_seed}` -> `scripts/crm_launch.sh <tasks.json> <out> configs/crm_main.json
[hours]` (submits `crm_collect.sbatch` arrays on 7 GPU partitions; `crm_worker.py` claims by mkdir, resumable) ->
`crm_qa.py` -> `crm_analyze.py`. eval_v1 drove 1,240 episodes (6.6 simulated h) in ~25 wall minutes on ~111 GPUs;
~1,500-2,000 new drives fit in under an hour. Offline picks for 600 groups x 6 arms: ~600 x 4 s = 40 min on the 5090
(or the MI350X).

## 7. Recommendations

1. Build the iterated sampler as a 7-d CEM/MPPI over (a_1..3, dv_1..4) with the projection-to-caps and refit-on-
   clipped-theta rules above; reuse `rc_planner._weights` and the "score the mean too" step; do not touch
   `deform_reference`.
2. Run it on CRM first (the only place with real headroom, 9% failures, and reusable one-shot drives), with the
   budget control (one-shot 512) and the pessimistic arm; treat the rigid sibling arenas at 2 m/s as the second test.
3. Report the predicted-logit gain (already ~ -0.8 at equal budget in the pilot) next to the realised change; if
   the realised change is nil, the gap is the model's extrapolation error on elite-refined routes — then the fix is
   on-policy data from the iterated sampler's own picks (the night-2 A2 lesson), not more search.
4. Keep time out of the primary contrast; add it as arm F so the trade is measured, not assumed. Energy scoring
   waits for study 1's energy head; the realised `positive_work_kj` can be reported for every arm tonight.
