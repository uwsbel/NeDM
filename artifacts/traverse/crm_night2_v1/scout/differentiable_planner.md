# Gradient-based route optimisation through the risk network — reconnaissance (2026-09-17/18)

Read-only scout. Numerical checks were run locally (RTX 5090, torch 2.12, `/home/harry/miniconda3/envs/nedm/bin/python`);
no simulation, no cluster job. Test code: `/tmp/diffcorr/diff_corridor.py` (torch re-implementation + landscape +
descent), `/tmp/diffcorr/followup.py` (float64 re-validation, geometry/speed ablation, eval-pick head-room); outputs
`/tmp/diffcorr/sweeps.npz`, `/tmp/diffcorr/descent_{top-8,random-8}.npz`. Repo root `/home/harry/NeDM-traverse_mppi`.

## 0. Verdict in one paragraph

A torch corridor that reproduces `f104_n2_dataset.station_tensor` to float32 round-off (max |diff| 2.4e-7 on all 256
demo-pool routes; ensemble logits within 2.3e-3 of `RiskModel.score`, identical argmin) and `f104_n2_sampler.shape`
to 1e-14 is straightforward (about 90 lines, given in section 3). The route logit is differentiable w.r.t. the sine
amplitudes a_j and speed knots dv_k with autograd matching central finite differences (dv exact to 1e-8 relative; a to
3e-4 median / 2e-2 worst because of piecewise-linear interpolation kinks). The landscape is usable: gradients are
O(0.3 logit/m) and O(0.1 logit per m/s), Lipschitz-ish (pairwise |dz|/|dtheta| median 0.07, p99 0.53, max 1.45), only a
few slope-sign flips per 1-D axis, no discontinuities from the `valid` channel on the arena interior. Adam descent from
the pool's top-8 lowers the ensemble-mean logit by 5.8 on average AND lowers the held-out member (leave-one-member-out)
by 5.6, the pessimistic max-member by 6.1 and the independently trained rigid-ground ensemble by 4.6, with 100 % of
(fold, start) pairs improving — so on this case descent is not fitting one member's noise. Two honest caveats decide the
study design: (i) most of the gain is "drive faster" (speed-only descent gets 80-95 % of the joint gain; dv is pushed to
its +4 m/s clip and mean commanded speed goes from 3.5 to 4.7-5.7 m/s), which is the CRM network's learned cue
("faster is safer" AUC 0.76; straight-6 = 66.5 % vs straight-2 = 13 %), not a terrain insight; geometry-only descent
gains 0.2-2.4 logit; (ii) head-room is limited: 153 of the 200 locked CRM evaluation picks already have P(unsafe) < 1 %,
only 47 have P > 1 % (15 with P > 20 %, of which 7 failed), and 7 of the 18 driven failures had P < 1 % (blind spots no
optimiser fixes and the optimiser's curse can worsen). Recommendation: build it as a *refinement stage after the 256-draw*
(multi-start from the top-k), optimise an expected-cost objective (C_fail * P + time + energy) rather than the bare logit,
constrain with the validator's own curvature quantity at weight 1e5 plus float64 re-shaping and `validate_reference`,
guard with leave-one-member-out + pessimistic re-scoring + the rigid ensemble as a second opinion, and evaluate paired
against the already-driven pool-argmin arm (identical pools, exact reuse of the eval_v1 drives).

## 1. What exists (paths, lines, signatures, shapes)

### 1.1 Corridor construction (numpy) — `scripts/f104_n2_dataset.py`
- Constants `:12-13`: `N_STATION, N_LATERAL, HALF_WIDTH_M = 96, 32, 6.0`.
- `init_map(root)` `:17-24`: loads `<root>/static_map_v1/observation.{json,npz}`; `G['rgbd']` = (4, 512, 512) float32
  (channel 3 = elevation / `elevation_scale_m`=10, invalid sentinel −2.0), `G['mpp']` = 2·110·tan(0.8203/2)/512 ≈ 0.187 m/px,
  `G['ctr']` = 255.5. CRM map: `artifacts/traverse/crm_f104_v1/map_root/static_map_v1 -> ../maps/arena_f104_50h_v1`;
  ch3 range [−2.0, 0.390]; 30.3 % of pixels invalid = outside the 80 x 80 m arena (field of view 95.7 m).
- `sample_map(x, y)` `:27-34`: row = ctr − y/mpp, col = ctr + x/mpp; bilinear with corner indices clipped to [0, n−2] and
  fractions clipped to [0, 1] (== border padding); `valid` = all four corners > −1.999 (boolean, per sample).
- `resample_route(wp, stations, n)` `:37-42`: uses the given `stations` if monotone and same length else cumulative chord
  length; `grid = linspace(s0, s_end, 96)`; `np.interp` per coordinate.
- `station_tensor(wp, sp, st) -> (X (5,96,32) float32, route_len)` `:45-60`: tangent = `np.gradient(pts)` (unit spacing,
  central interior, one-sided ends), left normal (−ty, tx), offsets `linspace(−6, 6, 32)`; elev = patch·10 where valid;
  e0 = elev[0,16] (or nan-mean of row 0); fill invalid with e0; `ds = route_len/95`, `dl = 12/31`;
  channels `[fill − e0, clip(grad_s, ±2), clip(grad_l, ±2), v repeated laterally, valid]`; v = `np.interp` of `sp` over the
  chord-length stations onto `linspace(0, L, 96)`. Clips are never active on f104 (0 of 256·96·32 samples).

### 1.2 Route parameterisation (numpy) — `scripts/f104_n2_sampler.py`
- `A_ACC, A_DEC = 1.5, 2.0`, `V_MIN, V_MAX = 0.5, 6.0` `:23-24`.
- `speed_knots(f, k, vals)` `:35-40`: smoothstep between k=4 knots at `linspace(0,1,4)`, free ends.
- `shape(xy, station, speed, lat, dv)` `:43-55`: pts = xy + lat·normal(xy) (normal from `np.gradient` of the BASE route,
  so it is a constant matrix), stations recomputed, `v = clip(speed+dv, 0.5, 6)`, terminal cone `v <= sqrt(4 (L−s))`,
  forward pass `v_j <= sqrt(v_{j-1}^2 + 3 ds)`, backward pass with 4 ds. v[0] is free (0.5-6 m/s from rest).
- `lateral_profile(f, L, rng, sigma=5, modes=3, kappa_max=.125, budget=.55)` `:58-63`: `lat = sum_j a_j sin(j pi f)`,
  a_j ~ N(0, 5/j) clipped to `cap_j = 0.55·0.125·L^2/(j pi)^2` (for the demo base L = 52.0 m: caps [18.9, 4.7, 2.1] m).
  GOTCHA: the caps are per mode; they do not guarantee curvature feasibility of the sum, and even a_1 alone is
  curvature-feasible on only 38 % of [−cap_1, cap_1] (measured, section 4). Curvature is 100 % of sampler rejections.
- `sample_one` `:66-78`: `lat = clip(lat, ±10)`, `dv = speed_knots(f, 4, clip(N(0,1.5), ±4))`.
- `anchors` `:81-93`: 9 designed routes (offsets 0/−4/+4 m as `off·sin^2(pi f)`, 2/4/6 m/s). `propose` `:96-108`.

### 1.3 Network — `scripts/gen_riskmodel.py`
- `route_logit(haz) = log(sum_s softplus(haz_s) + 1e-6)` `:10-12` = log cumulative hazard = cloglog P(event anywhere).
- `Net(cin=6, nctx=5, arch='gru', width=64, layers=2)` `:15-57`: 4 Conv2d(3x3, stride (1, 1/2/2/2) = station-preserving,
  lateral 32->4) + BatchNorm + GELU; lateral mean+max -> Linear(192,96); ctx MLP(5->32); + station position;
  Conv1d(k=5) + Dropout(0.1); BiGRU(96->2x64) (or `tx`: 2-layer TransformerEncoder d=96 with learned pos, or `mlp`);
  Linear -> per-station hazard (B, 96). In eval mode everything is deterministic and differentiable (GELU smooth; `amax`
  gives a sub-gradient).
- Checkpoints (`torch.load(..., weights_only=False)`): keys `state, arch, layers, channels, norm{mu, sd (4,), cont_index},
  ctx_cols [17..21], ctx_mu, ctx_sd (5,), cin 6, nctx 5`. CRM ensemble
  `artifacts/traverse/crm_f104_v1/train_v1/deploy/CRM_N2_s{0..4}.pt`; rigid `artifacts/traverse/fdm_f104_50h_20260909/night2_v1/final/N2_s{0..4}.pt`.
  Norm (CRM s0): mu [0.613, 0.0073, 0.0043, 3.108], sd [1.014, 0.177, 0.187, 1.739]; `valid` is NOT normalised; a 6th
  all-ones channel is appended (`gen_planner.py:177`).

### 1.4 Planner — `scripts/gen_planner.py`
- `CFG = MPPIConfig(max_speed 6, min_speed 0, max_curvature .125, arena_half_extent 40)` `:30`; `N_CAND = 256` `:32`.
- `safe_validate` `:94-99` (ValueError -> `{'valid': False, 'reasons': [msg]}` and NO `max_curvature` key — gotcha).
- `base_route(pose, goal)` `:102-121`: Hermite `propose_route_families` route_00 at 2 m/s (0.5 m step; 106 points for the
  demo), fallbacks for starts facing away. `proposal_pool` `:124-127`, `corridors(cands) -> (X (n,5,96,32), L (n,))` `:141-146`,
  `geom_ctx(start_xy, goal_xy, start_yaw, L) -> (n,5) [rel_x, rel_y, |rel|, start_yaw, route_len]` `:149-151`.
- `RiskModel.score(X, ctx5) -> (mean logit, 1−exp(−exp z))` `:170-184`, float32, `torch.no_grad`.
- `plan(...)` `:211-241`: argmin of the ensemble mean over the single 256-draw. There is no iteration and no gradient.
- The old `ReferenceMPPI` (`src/nedm/traverse/fdm_mppi.py:171-246`, iterations=2, sin^2 envelope) and the CEM step
  `planner_s.resample_routes` (`src/nedm/traverse/planner_s.py:144-`) exist for older route families and are unused here.

### 1.5 Validator — `src/nedm/traverse/fdm_mppi.py:121-168` `validate_reference(route, obstacles, config, anchor_pose)`
- Cuts the route at the waypoint nearest the anchor (`:128-130`), then:
  curvature `_curvature_max(xy)` (`planner_s.py:65-72`, three-point circumscribed-circle curvature on the RAW 0.5 m waypoints,
  not on the dense resample) `> 0.125 + 1e-6`; speed outside [0, 6] ± 1e-6; `acceleration = diff(v^2)/(2 diff(s))`
  `> 1.5 + 1e-6` or `< −2.0 − 1e-6`; arena: footprint corners (half-length 2.6 + 0.1, half-width 1.3 + 0.1 m) of the
  0.25 m dense resample outside ±40 m. Tolerances are 1e-6 — float32 routes whose acceleration limits are ACTIVE fail
  (measured: float32 torch route accel range [−2.00001, +1.008]; the same parameters re-shaped in float64 numpy pass).

## 2. Prior results that constrain tonight

- **The spurious low-risk well (memory `f104-fixed-arena-study-state.md:64-90`, 2026-09-10).** Old CNN + argmin over 506
  MPPI draws found a P(fail)=0.00 pick that stalled; causal ablation: varying only the corridor patch moved P 0.96 -> 0.04;
  along the deformation direction the model swung 0.80 per 0.1 m patch RMS where training pairs swing 0.079 — a 10x
  local-sensitivity anomaly. The fix was architectural (station-preserving CNN -> BiGRU, survival loss): spurious wells in
  the argmin tail 5-16 -> 0-2, closed-loop failures 17.1 % -> 6.5 % (p=.004), and pessimistic ensembling was recommended to
  blunt the optimiser's curse. A gradient optimiser is a *stronger* argmin than 256 draws, so this is the failure mode to
  measure first (section 5 shows it did not appear on the demo case with the current network; section 8 says how to check
  it per case).
- **CRM night (`artifacts/traverse/crm_f104_v1/REPORT.md`):** pessimistic ensemble 92.0 % vs mean 91.0 % (n.s.); the
  CRM-trained planner's picks detour more (max lateral 6.4 m median, 78/200 beyond 8 m) and command more speed on climbs;
  "faster is safer" scores AUC 0.76 alone; a nearest-10-training-routes lookup predicts the evaluation failures with AUC
  0.885 (interpolation regime). Pools are rebuilt bit-for-bit from md5 seeds (`scripts/crm_pools.py:25-44`).
- **Head-room in the 200 locked picks (computed tonight from `eval_v1/picks/*.json` + `eval_v1/results.json`):** pick logit
  quantiles [−7.6, −7.35, −7.07, −6.42, −4.8, −2.35, +1.23] (min, p10, p25, p50, p75, p90, max); P(unsafe) of the pick:
  153 in [0, 1 %) (7 driven failures, 4.6 %), 19 in [1, 5 %) (2 failures), 13 in [5, 20 %) (2), 15 in [20 %, 1] (7 failures,
  46.7 %). The 18 failures had predicted P {0.001, 0.002, 0.002, 0.004, 0.005, 0.005, 0.006, 0.02, 0.034, 0.052, 0.089,
  0.248, 0.315, 0.599, 0.658, 0.872, 0.889, 0.967}. So a better optimiser can act on ~47 pairs and at most ~11 of the 18
  failures; the 7 sub-1 % failures are network blind spots. Pool picks: mean speed median 2.98 m/s (p25-p75 2.56-3.66),
  max lateral median 7.0 m (p75 8.7, max 10.0 = `lat_clip`).
- **Energy (memory `crm-night2-plan-state.md`, commit 6abf6ee):** positive shaft work W+; a learned energy predictor did not
  beat the calibrated analytic work model (`scripts/traverse_wp9_analytic.py:81-160`: terms tract = sum max(0, m g sin +
  Crr m g cos + 1/2 rho CdA v^2 + m a) ds, tract2, tract_lowv, time, corn = m sum v^2 kappa ds, cross = m g sum |cross| ds;
  m = 2548 kg, Crr 0.01, CdA 3.5; NNLS-fitted coefficients). Every term is an integral over grade / cross-slope / v /
  curvature along the route — exactly the differentiable corridor quantities — so a differentiable energy term is available
  tonight WITHOUT the energy head; the head (other scout) can replace it later.

## 3. The differentiable chain (verified design)

Chain: `(a (B,3), dv (B,4)) -> t_shape -> (pts (B,106,2), v (B,106), st (B,106)) -> t_station_tensor -> X (B,5,96,32), L (B,)
-> normalise + ones channel -> Net.eval() x 5 members -> route_logit -> mean / max`. Full code in `/tmp/diffcorr/diff_corridor.py:18-113`.

- `npgrad(f, h, dim)`: `cat([f1−f0, (f[2:]−f[:-2])/2, f[-1]−f[-2]]) / h` = `np.gradient` exactly (divide AFTER `movedim`
  when h is batched (B,1,1)).
- `tinterp(xq (B,Q), xp (B,N), fp (B,N))`: `torch.searchsorted(xp.detach(), xq.detach(), right=True) − 1` clamped to
  [0, N−2], `w = ((xq−x0)/(x1−x0)).clamp(0,1)`, lerp. Gradient flows through the values AND through the breakpoints
  (`xp` = cumulative chord length, a function of the points); the segment index is a constant per evaluation (kinks).
- `TMap.sample(x, y)`: `F.grid_sample(R (1,1,512,512), grid (1, B·96·32, 1, 2), mode='bilinear', padding_mode='border',
  align_corners=True)` with `gx = col/(n−1)·2−1`, `gy = row/(n−1)·2−1`. This equals the numpy clipped bilinear exactly
  (max |diff| 1.2e-7 m on elevation). `valid` = the numpy four-corner test on `floor(row/col).detach()` — a constant in
  the graph (it is a boolean; gradient through it is zero by construction).
- `t_station_tensor`: as numpy, batched; `e0 = where(valid[:,0,16], elev[:,0,16], masked mean of row 0)`; `ds = (L/95).clamp_min(1e-3)`;
  `X[:,3] = v[...,None].expand(B,96,32)`; `X[:,4] = valid.float()`.
- `t_speed_knots(f (N,), vals (B,4))` and `t_shape(bxy, bst, bsp, a, dv)`: mirror of `S.shape` with `torch.minimum`
  recurrences (two python loops of 105 steps). NaN GOTCHA: `sqrt(2·A_DEC·(L−s))` is `sqrt(0)` at the last station (v[-1]=0
  always) -> `inf·0 = NaN` in the a-gradient; use `.clamp_min(1e-12)` inside every sqrt. Verified fix.
- `t_curv_max(pts)`: the validator's three-point curvature on the raw 106 points, batched (used for the penalty).
- ctx: `[rel_x, rel_y, |rel|, start_yaw, L]` with `L = st[:,-1]` differentiable (ctx_sd for L is 10.3 m, so its influence is small).
- Ensemble: `m.to(dev).eval()`; **cuDNN GOTCHA (confirmed):** `backward()` through a cuDNN GRU in eval mode raises
  `RuntimeError: cudnn RNN backward can only be called in training mode`. Either `m.mix.train()` (the GRU alone; it has no
  inter-layer dropout so its numerics are unchanged, BatchNorm/Dropout stay eval) or `with torch.backends.cudnn.flags(enabled=False)`.
- Vectorising the speed passes (recommended): the forward recurrence `v_j^2 = min(u_j^2, v_{j-1}^2 + 2 a ds_j)` equals
  `v_j^2 = min_{k<=j} (u_k^2 + 2 a (s_j − s_k))` (min-plus prefix over the unclamped u), a masked (B,106,106) `amin`; same
  backward with A_DEC. Removes 210 sequential kernels + their backward.
- Precision: do the FD checks and the final re-shape in float64; float32 is fine for descent (the pool logits differ by
  2.3e-3 between the float32 `RiskModel.score` and the float64 chain).

**Reproduction results (256 demo-pool routes, group `f104_crm_eval_group_0013`, base = anchor offset 0 at 2 m/s, L = 52.04 m):**

| check | result |
|---|---|
| `station_tensor` torch (f64) vs numpy (f32), 256 routes | max diff elev 1.2e-7, grade 3.0e-8, cross 3.0e-8, speed 2.4e-7, valid 0, route_len 1.9e-6 |
| ensemble logits vs `pool.npz` (`RiskModel.score`) | mean max diff 2.3e-3, per-member 4.9e-3, argmin 145 = 145 |
| `t_shape` vs `S.shape`, 16 random (a, dv) | waypoints 7e-15 m, speeds 1e-14 m/s |
| recovery of (a, dv) for all 256 pool routes by least squares | waypoint error median 4e-15, max 0.13 m (one route hit `lat_clip`); dv not identifiable where clips/cone bind (max 4 m/s) |
| autograd vs central FD (f64, h=1e-4, top-8) | dv: rel err median 1.2e-8, max 5.2e-5; a: median 3.3e-4, max 2.0e-2 (2/56 entries > 1e-2, both with tiny FD values −0.002, −0.18) |
| forward+backward, 5 members, cudnn off, float32 5090 | B=256: 350 ms; B=16: 282 ms (loop-bound, not batch-bound) |

## 4. Loss landscape (ensemble-mean logit z; 1-D sweeps of 401 points around the pool argmin, statistics restricted to the curvature-feasible part)

| axis (range) | z range | median / max |dz/dt| | slope-sign flips | p99 |d2z/dt2| | largest single step | curvature-feasible share |
|---|---|---|---|---|---|---|
| a_1 [−18.9, +18.9] m | [−5.60, +3.38] | 0.43 / 6.0 logit/m | 9 / 152 | 15 | 0.57 logit per 0.094 m | 0.38 |
| a_2 [−4.7, +4.7] m | [−5.65, +3.05] | 0.66 / 3.4 | 4 / 399 | 12 | 0.08 | 1.00 |
| a_3 [−2.1, +2.1] m | [−5.62, −0.76] | 1.42 / 3.5 | 1 / 365 | 24 | 0.04 | 0.92 |
| dv_1..dv_4 [−4, +4] m/s | e.g. dv_4 [−6.27, +3.24] | 0.06-0.67 / 2.8-4.9 logit per m/s | 0-2 / 399 | 6-17 | ≤ 0.10 | 1.00 |

- Gradient magnitudes at the top-8: |dz/da| median 0.27, max 2.5 logit/m; |dz/ddv| median 0.10, max 2.3 logit per m/s;
  no exactly-zero dv gradients at the top-8 (clips inactive there) — but after descent dv sits on the +4 clip, where the
  gradient is zero by construction (use a box projection, not a clamp inside the graph, so Adam's state stays sane).
- 512 random parameter draws inside the sampler's box: z quantiles [−2.1, 2.35, 3.42, 3.96, 4.36] (min, p10, p50, p90,
  max) — the random box is overwhelmingly high-risk (median P ≈ 1), 90 % curvature-feasible; pairwise |dz|/|dtheta| median
  0.07, p99 0.53, max 1.45 logit per unit (m, m/s).
- `valid` channel: all 256 corridors are 100 % valid; along the a_1/a_2 sweeps the valid count changes 0-27 times (routes
  reaching the arena edge) and none of these coincided with the largest step jump. The channel is a piecewise constant;
  only routes within ~6 m of the ±40 m edge see it. Clipping of grade/cross (±2) is never active on f104.
- Input sensitivity (L2 norm of dz/dX per channel at the top-8): grade 6.35 (max 21.3), cross 4.05 (12.3), valid 0.56,
  elev_rel 0.29, speed 0.26 — the network reads slopes; the 10x-anomaly probe of the old CNN should be repeated per case
  as the ratio of the descent's dz to the corridor-RMS change it induced (section 8).
- Multi-modality: a_1 has ~9 local extrema across its feasible range; descent needs multi-start (the pool's top-k gives it).

## 5. Descent experiments (Adam, penalties: 1e3·relu(kappa−0.125)^2 summed over waypoints + 10·relu(|pts|−37)^2; box projection to the sampler caps and ±4 m/s)

**5a. Leave-one-member-out, top-8 starts (pool idx [145, 5, 164, 150, 2, 183, 8, 153], logits before [−5.59, +3.37, −2.19,
−1.20, +3.65, −0.48, +3.26, +0.46]), 60 steps, lr 0.05 m / 0.1 m/s, objective = mean of 4 members.** Mean change over
5 folds x 8 starts: fit members −5.76 logit, held-out member −5.64, pessimistic (max of all 5) −6.13, rigid-trained
ensemble −4.64; the held-out member improved in 40/40 cases, never worsened by > 0.5. Start-wise held-out change
[−1.25, −7.12, −4.67, −5.17, −6.53, −6.34, −8.77, −5.30] tracks the fit change [−1.29, −7.47, −4.74, −5.22, −6.63, −6.38,
−8.61, −5.74]. Interpretation: no member-specific exploitation on this case; the descent direction is shared by five CRM
seeds and by a network trained on rigid ground.

**5b. All-5-member objective, 100 steps:** top-8 mean logit after [−6.91, −4.13, −6.79, −6.81, −3.11, −6.87, −6.12, −5.59]
(pool argmin −5.59, i.e. P 0.37 % -> 0.10 % for the best start; seven mediocre starts become as good as the argmin);
pessimistic after [−5.97 … −2.52]; rigid before -> after e.g. [+2.7 -> −6.6, +3.5 -> −5.3]. Random-8 starts (median start
logit +3.3): after [−4.29, −3.06, −4.11, −6.94, −4.31, −3.22, −4.29, −4.10] — descent from random box points reaches the
pool-argmin level in 100 steps only once; the pool's top-k are the right starts.

**5c. What the optimiser actually does (ablation, top-8, 100 steps):** geometry-only (dv frozen) [−5.81, +3.09, −3.52,
−2.73, +2.49, −0.66, +2.74, −1.91] with mean v unchanged (2.0-4.0 m/s); speed-only (a frozen) [−6.76, −2.58, −6.35, −5.99,
−0.44, −6.58, +0.91, −3.71] with mean v 4.2-5.7 m/s; both [−6.91, −4.13, −6.79, −6.81, −3.11, −6.87, −6.12, −5.59] with
mean v 4.75-5.7 m/s, dv on the +4 clip on 2-4 of the 4 knots. => on this pair 80-95 % of the joint gain is "command more
speed". Final routes: max |lateral| 2-10 m (two at the 10 m `lat_clip`), L 52-58 m.

**5d. Validity after descent:** float32 torch routes fail `validate_reference` with 'acceleration'/'deceleration' on 15/16
(round-off 1e-5 against the 1e-6 tolerance because the passes are active everywhere); the same (a, dv) re-shaped by numpy
`S.shape` in float64 pass except curvature: 1/8 top-8 (kappa 0.190 from the argmin start, a = [6.7, −4.7, 0.6]) and 1/8
random (0.142) with weight 1e3; with weight 1e5 and lr_a 0.02, 8/8 pass (kappa ≤ 0.093) at logits [−6.85, −4.14, −6.91,
−6.55, −2.97, −6.89, −5.65, −3.92] (the argmin start loses 0.06 logit vs the unconstrained result).

## 6. Recommended optimiser specification

- **Parameters** theta = (a_1..a_3 [m], dv_1..dv_4 [m/s]) on the planner's base route (`gen_planner.base_route`), i.e. the
  sampler's own family; start and goal are pinned by the sine basis. Optional later: 5 modes and 6 knots (the network was
  trained on 3/4; keep 3/4 tonight so the support matches the pool).
- **Multi-start**: the top-16 of the 256-pool (ensemble-mean ranking), plus the pool argmin's mirror in a_1 (sign flip) to
  cover the other side of a hill. Batch them (B=16-32 costs the same as B=256).
- **Optimiser**: Adam, lr_a = 0.02 m, lr_dv = 0.10 m/s, betas (0.9, 0.99), 100 steps, gradient-norm clip 10; box projection
  after each step (|a_j| ≤ cap_j, |dv_k| ≤ 4) done with `clamp_` under `no_grad` (not inside the graph). Keep the best-so-far
  iterate by the PESSIMISTIC objective (max member), early-stop after 15 steps without pessimistic improvement. Do not use
  L-BFGS: the chain is only piecewise smooth (interp segment switches, `torch.minimum` ties, box projections; FD/autograd
  disagreement up to 2e-2 on a; |d2z| p99 up to 24 logit/m^2) and its line search will stall; Adam with these steps
  behaved monotonically in 60-100 steps.
- **Constraints as penalties** (all on the raw 106-point route, i.e. the validator's own quantities):
  curvature 1e5·sum relu(kappa_i − 0.95·0.125)^2 (1e3 was insufficient: 2/16 violations); arena 10·sum relu(|pts| − 37)^2
  (37 m = 40 − footprint half-length 2.7 − margin); speed/acceleration/deceleration hold by construction of `t_shape`
  (clamp + cone + passes) — verify in float64; trust region 0.02·||theta − theta_start||^2 (ablate: it is the knob that
  limits exploitation; report results with and without).
- **Projection back onto validity**: re-shape the final (a, dv) with numpy `S.shape` (float64), run `safe_validate(route,
  [], CFG, anchor)`; discard invalid finals; choose among {16 valid finals} ∪ {pool} by the pessimistic objective; ties or
  < 0.3 logit improvement over the pool argmin -> keep the pool argmin (cheap insurance against the curse).
- **Cost**: today 0.3 s per step -> 30 s per planning call (100 steps) on the 5090; with the vectorised passes and the
  cuDNN trick expect ≤ 10 s; the current planner is ~0.4 s per call (2.4 Hz on MI350X, memory). Fine for offline picks on
  200 pairs (≤ 2 h); NOT for 1-2 Hz replanning unless cut to ~10 steps from the moving-vehicle top-k.

## 7. Objective: risk + energy + time

Use an expected-cost form, not the bare logit (the logit rewards pushing P from 0.4 % to 0.1 % as much as from 40 % to
10 %, and it is what drives dv to the clip):

  J(theta) = C_fail · P(theta) + T(theta) + lambda_E · E(theta) / P_ref  + penalties,
  P = 1 − exp(−exp(z_mean)) (or the pessimistic z_max), T = sum_j ds_j / max(0.5·(v_j + v_{j−1}), 0.25) [s],
  E = analytic positive work on the 96 stations (m g sin(atan grade)·ds + Crr m g ds + 1/2 rho CdA v^2 ds + m a ds, clipped at 0,
  plus m v^2 kappa ds cornering) [kJ] with the `traverse_wp9_analytic.py` constants; P_ref converts kJ to seconds
  (e.g. 1 kJ = 0.2 s makes a 50 kJ saving worth 10 s; sweep lambda_E in {0, 0.5, 1, 2}).
  C_fail in seconds: 120 s (the episode horizon) is the natural value; 60 s makes the planner accept ~1.5 % more risk
  for 1 s saved. Report picks for C_fail ∈ {60, 120, 300}.
- Gradient of P w.r.t. z is exp(z)·exp(−exp z): it vanishes at both ends (P ≈ 0: correct, no incentive to go lower;
  P ≈ 1: bad starts stall). Starting from the pool's top-k (P < 0.5 typically) avoids the second; if a start has P > 0.9 use
  z for the first 20 steps then switch.
- Time and energy are strong counter-forces to "go faster": T falls with v, but E's v^2 drag and acceleration terms rise,
  and the CRM network's P is monotone in v. Expect the optimum to sit at 4-5 m/s rather than the 6 m/s cap. This is also
  where the energy head (other scout) plugs in: replace E by the head's output, keep the analytic model as the benchmark.
- Once the velocity input exists (other scout), theta becomes (a, dv) on the route ahead of the moving vehicle with the
  initial speed fixed; the chain is unchanged except that the start speed and the start station are constants.

## 8. Failure modes to guard, with the concrete check for each

1. **Optimiser's curse / spurious wells** (the 09-10 finding). Check per case: (i) leave-one-member-out — optimise on 4,
   read the 5th; flag if held-out gain < 0.5 · fit gain or held-out worsens by > 0.5 logit (tonight: 0/40); (ii)
   pessimistic z_max must improve; (iii) rigid-trained ensemble as an independent second opinion (agreed on all 8 starts);
   (iv) local-sensitivity ratio: |Δz| per unit corridor-RMS change along the descent path vs the training-pair baseline
   (the 10x anomaly signature); (v) nearest-training-route sanity: the failure rate of the 10 most-overlapping training
   routes at similar speed (AUC 0.885 on eval failures) must not be worse than for the pool argmin.
2. **Speed exploitation**: dv -> +4 clip, mean v 4.7-5.7 m/s vs pool picks' median 2.98 m/s. Keep V_MAX = 6 (training
   support ends there), use the expected-cost objective with energy, and report tilt30 (straight-6 had 29/200 tilt events
   vs 12/200 for the CRM planner) and downhill overshoot (max 8.5 m/s seen on the straight-6 arm).
3. **Leaving the sampler's support**: enforce |lat| ≤ 10 (lat_clip; the clamp inside `t_shape` already does), the per-mode
   caps and ±4 m/s; log max |lat| and mean v of every final.
4. **Curvature**: per-mode caps admit infeasible sums; penalty weight 1e5, final `validate_reference`, drop invalid finals.
5. **float32 vs 1e-6 validator tolerance**: always re-shape the final parameters in float64 numpy before validating.
6. **NaN** from sqrt(0) at the terminal cone: `clamp_min(1e-12)` in every sqrt (silently poisons Adam otherwise).
7. **cuDNN eval-mode GRU backward**: `m.mix.train()` or cudnn off.
8. **Zero-gradient traps**: dv on a clip, grade/cross clips (never active on f104), `amax` ties; use projection not
   in-graph clamps; monitor the fraction of zero gradient entries.
9. **Route topology**: the sine family cannot switch sides of an obstacle except via a_1's sign; multi-start covers it.
10. **ctx gaming**: L is an input; changes of a few metres shift ctx by < 0.3 sd; ignore but log.
11. **Base-route dependence**: everything is relative to `base_route`; on mission legs with fallbacks (arc-line bases) the
    normal field is not smooth at the arc/line joint — the corridor is still fine (it resamples), but a_j semantics change.

## 9. Validation protocol (offline first, then Chrono; rigid CPU first per the user's decision, then CRM)

Offline (no driving, ~1 h on the 5090 for 200 cases):
- For each of the 200 `cases_eval` pairs rebuild the locked pool with `crm_pools.build` (md5 seeds), run the descent from the
  top-16 under three objectives (logit-only, expected cost with C_fail 120 s and lambda_E 0, expected cost with energy),
  record per case: fit/held-out/pessimistic/rigid logit changes, validator pass rate, kappa_max, mean v, max |lat|, T, E,
  nearest-training-route failure rate, and the projected pick (pessimistic argmin over finals ∪ pool). Pre-register the
  exploitation flags of section 8.1 and the abstention rule (< 0.3 logit gain -> keep the pool pick).
- Report how many of the 47 pairs with pool-pick P > 1 % (and the 15 with P > 20 %) change pick, and to what predicted P.

Closed loop (paired, same 200 pairs, one physics config, picks hashed before the first drive as in `eval_v1/PICKS_LOCKED.sha256`):
- Arms: pool argmin (already driven — exact reuse of `eval_v1/runs` since pools are bit-identical), gradient pick (mean
  objective), gradient pick (pessimistic), gradient pick with energy, the iterated-sampling arm from the other planner scout,
  straight-6 (already driven). Identical picks share one drive.
- Primary: goal reached, exact McNemar on discordant pairs, clustered by terrain feature (the CRM report's honest CI);
  secondary: time to goal, positive work, tilt30, max speed; count "optimiser-created failures" (final P_pess < 1 % that fail)
  vs the pool arm's 7.
- Power: the pool arm is at 91 %; at most ~11 of its 18 failures are addressable and the faster routes may add tilt
  failures, so a 3-5 point gain is the realistic ceiling and 200 pairs will give a wide CI; say so in the PLAN. Consider
  enriching with the 47 P > 1 % pairs plus 100 fresh hazard-enriched pairs where the pool pick is risky.
- Rigid Chrono first (CPU, cheap): the same pipeline with the rigid ensemble (`N2_s*.pt`) and rigid map
  (`gen_planner.set_map` heightmap path) on the gen_v1/nav_v1 case sets; then CRM confirmation on the 200.

## 10. Open items / what not to repeat

- Do not re-test "does the argmin tail of the OLD CNN have spurious wells" — settled 09-10 (architecture fix). Do test it
  for the descent with the current network, per case, with the section-8 flags.
- Do not fit a learned energy predictor as the energy term for the planner before the analytic model is beaten (09-07).
- The `valid` channel and the arena edge: untested tonight for routes near ±34 m (none in the demo pool); if the offline
  pass shows finals near the edge, add a soft penalty on distance to the map's invalid region.
- Sensor-driven (nav-style, 10/12-channel) corridors: `sensor_dataset.tensor10` / `sensor_dataset_v2.tensor12` are the same
  construction with more channels; the torch chain extends by adding `grid_sample` calls per channel. Not needed tonight.
