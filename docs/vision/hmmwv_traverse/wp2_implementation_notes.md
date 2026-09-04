# WP2 implementation notes — NRD dynamics subsystem (G3 triad)

**Date:** 2026-09-03 · **Modules:** `nedm/traverse/nrd_data.py`
**Scripts:** `traverse_wp2_encode.py`, `traverse_wp2_add_power.py`,
`traverse_wp2_train.py`, `traverse_wp2_analyze.py`
**Artifacts:** `artifacts/traverse/wp2_z2_cache_v6/`, `wp2_g3{,b}_{state,joint,priv}_amd/`,
`wp2_g3b_analysis.json`

## Setup

Frozen WP1 v6 encoder → per-episode latent cache (`z2` 256-D, `z1` 15-D
`tire_normal_force_omega`, `act` 3-D, `pose` 3-D, `power` 1-D kW). 9518
episodes, 4.0 GB, ~1 KB/frame instead of 0.3 MiB — the dynamics runs do no
zstd decode and no convolution. Encode: 8 min on one MI350.

Splits are reproduced from the cache manifest with the *same* permutation
`perception.split_episodes` uses (seed 20260902, 70/15/15), so WP2's held-out
layouts are exactly the encoder's held-out layouts. Counts verified
6662/1427/1429 against v6's config. Without this, encoder-training layouts leak
into WP2 val/test and "held out" stops meaning held out for the stack.

Three matched variants share backbone (6L/256, ctx 16 @ 0.05 s), data, split
and budget (40k steps, batch 256); only the token differs:
`state [z1,a]` · `joint [z1,z2,a]` · `priv [z1,(x,y,sin ψ,cos ψ),a]`.
The auxiliary power head (plan §4) is identical in all three and never feeds
back into the token, so state-only stays exactly 15-D for RQ2.

Rollout eval is autonomous: z1 and (joint) z2 both predicted recursively, only
actions come from the episode. `priv` is fed ground-truth pose every step —
that is what makes it a ceiling rather than a fourth model.

## Iteration log

| Run | Change | Selection metric | Result |
|---|---|---|---|
| g3 (state/joint/priv) | first triad; selection = pose err @1.0 s | pose_err_m@20 | state 0.118 / joint 0.134 / priv 0.123 m; CV baseline 0.582 |
| g3b | + power head, horizons to 5 s, selection = `z1_mae_norm` | z1_mae_norm@100 | state 0.617 / **joint 0.578** / **priv 0.574** |

## The first triad's lesson: pose rollout is the wrong discriminator

Joint had the *best* aggregate z1 MAE and the *worst* pose error. Pose
integrates only vx, vy and yaw-rate; 8 of 15 channels are tire Fz and spindle
ω. Per-channel MAE at 1.0 s showed z2 improving 12 of 15 channels — pitch 58 %,
body-y angular velocity 42 %, tire Fz 13–20 % — and losing only on vx, which
dominates pose error.

The privileged row is what proved the metric insensitive rather than z2 weak:
`state vs priv` on pose error was −0.006 m, CI95 [−0.021, +0.009]. **Perfect
ground-truth localization does not move dead-reckoned pose error at 1 s.** A
metric a privileged ceiling cannot move cannot adjudicate G3.

## G3b result (paired bootstrap, 256 held-out episodes)

Aggregate dynamics fidelity, `z1_mae_norm` (negative delta ⇒ first is better):

| Comparison | 1.0 s | 5.0 s |
|---|---|---|
| state vs joint | +0.048 CI [+0.032, +0.065] | +0.039 CI [+0.000, +0.076] |
| state vs priv | −0.006 CI [−0.015, +0.003] | +0.043 CI [+0.015, +0.071] |
| joint vs priv | **−0.054 CI [−0.071, −0.038]** | +0.004 CI [−0.029, +0.038] |

Terrain-coupled channels (attitude + tire Fz) at 1.0 s: `state vs joint`
+0.095 CI [+0.069, +0.122]; `joint vs priv` −0.094 CI [−0.121, −0.068];
`state vs priv` +0.001 CI [−0.014, +0.015].

**G3 passes on the corrected metric, and the privileged row is not an upper
bound.** joint beats priv outright at 1 s and matches it at 5 s. The terrain
row explains why: z2 improves the terrain-coupled channels decisively while
(x,y,ψ) does not improve them at all. Position alone only buys terrain if the
model memorizes the fixed heightmap; z2 hands the terrain over directly.
Plan §8.3's optional "+ privileged local terrain patch" is therefore the real
ceiling and has not been run.

## Batch A: seeds, integrity control, and the real ceiling

Three seeds per arm, plus the plan §8.3 integrity control and the privileged
terrain row (`wp2_privterr_amd`, pose + true 8x8 ego terrain patch every step).

| run | z1mae@1 s | z1mae@5 s | terrain@1 s | terrain@5 s |
|---|---|---|---|---|
| state (3 seeds) | 0.3281 ± 0.0080 | 0.6082 ± 0.0125 | 0.4946 ± 0.0044 | 0.816–0.832 |
| joint (3 seeds) | **0.2867 ± 0.0141** | 0.6256 ± 0.0452 | **0.4153 ± 0.0184** | 0.789–0.872 |
| priv (pose only) | 0.3263 | 0.5741 | 0.4890 | 0.7915 |
| **privterr (ceiling)** | **0.2211** | **0.3090** | **0.3172** | **0.3710** |
| joint, z2 SHUFFLED | 0.3693 | 0.6974 | 0.5338 | 0.8555 |

**1. The 1 s win is robust; the 5 s win was seed luck.** At 1 s the state–joint
gap (0.0414) is ~5x the pooled seed SD, and the terrain-channel gap (0.0793) is
larger still. At 5 s joint is 0.6256 ± 0.0452 against state's 0.6082 ± 0.0125 —
**no advantage, and 3.6x the variance**. The single-seed g3b reading (joint
0.578 vs state 0.617 at 5 s) does not replicate and is withdrawn. This lines up
with G4: as the predicted latent degrades, so does its usefulness.

**2. The integrity control passes cleanly.** z2 drawn from a *different* layout
scores 0.3693 — worse than state-only (0.3281), not merely worse than joint. A
latent that actively hurts when mismatched is being read for layout-specific
content, not used as extra capacity. This settles §8.3's integrity row.

**3. The pose-only "upper bound" is not one; the terrain row is.** `privterr`
reaches 0.2211 at 1 s and 0.3090 at 5 s — roughly twice as good as anything
else at 5 s. So there is a large amount of terrain information available to
dynamics, and:

> **z2 captures ~38 % of it at 1 s** — (0.3281 − 0.2867) / (0.3281 − 0.2211) —
> **and ~0 % by 5 s.**

That is the bounded form the G3 claim should take. It is a real, seed-robust,
integrity-controlled effect, and it is a minority of the available headroom
over a short horizon.

## Does PREDICTING z2 beat HOLDING it? Yes, at 1 s (plan §8.3 baseline)

`traverse_wp2_z2mode.py` rolls the *same* trained joint checkpoint out twice —
feeding back its own predicted z2, vs freezing the last encoded z2 — paired
over 256 held-out episodes. On a static map persistence is the strong baseline
the plan warns about, and G4 had made the prediction branch look worthless.

z1mae@1 s, predict − persist (negative ⇒ predicting is better), 3 seeds:
**−0.0298 [−0.0381, −0.0217] · −0.0100 [−0.0170, −0.0030] · −0.0120
[−0.0181, −0.0061]** — all three exclude zero. Terrain channels likewise
(−0.052, −0.018, −0.018). At 5 s it degenerates: seed 3 flips sign
(terrain +0.053 [+0.009, +0.100], i.e. freezing is *better*).

Decomposing the 1 s gain over state-only (0.3281):

| configuration | z1mae@1 s | share of the gain |
|---|---|---|
| state (no z2) | 0.3281 | — |
| joint, z2 **frozen** | 0.3040 | 56 % — just *having* z2 in context |
| joint, z2 **predicted** | 0.2867 | +42 % — from rolling it forward |

**The z2-prediction branch is load-bearing at 1 s**, which is the narrow thing
§9.5 needs. Read together with G4 the picture is coherent rather than
contradictory: the predicted latent is **a good dynamics conditioner and a bad
picture**. What dynamics needs is "what terrain am I on *now*", which changes as
the vehicle moves — freezing it means conditioning on stale terrain — and that
survives even though the latent is no longer decodable into a scene.

## Batch B input ablations: z2 is an RGB encoder, and elevation is being wasted

Re-encoded the whole 9518-episode set twice with the frozen WP1 encoder, once
with the elevation channel zeroed and once with RGB zeroed, then retrained
joint on each (plan §8.3 input-ablation row).

| run | z1mae@1 s | terrain@1 s | closes ceiling gap |
|---|---|---|---|
| state (3 seeds) | 0.3281 ± 0.0080 | 0.4946 ± 0.0044 | — |
| joint RGB+D (3 seeds) | 0.2867 ± 0.0141 | 0.4153 ± 0.0184 | 38.7 % |
| **joint RGB only (depth zeroed)** | **0.2797** | **0.4014** | **45.2 %** |
| joint depth only (RGB zeroed) | 0.3500 | 0.5149 | −20.4 % |
| joint z2 shuffled | 0.3693 | 0.5338 | −38.5 % |
| privterr (ceiling) | 0.2211 | 0.3172 | 100 % |

**Removing the elevation channel costs nothing** — 0.2797 vs 0.2867 ± 0.0141
is half a seed SD, i.e. indistinguishable. **Removing RGB is catastrophic**:
0.3500, worse than having no z2 at all. Corroborated at the latent level:
zeroing depth moves z2 by 0.42 mean-abs, zeroing RGB by 2.45.

So the WP1 encoder is effectively an **RGB encoder that ignores its depth
channel**, and everything z2 contributes to dynamics comes from appearance.

*Caveat:* zeroing a channel is off-distribution for an encoder trained on both.
That weakens the depth-only row (badly OOD) but not the RGB-only row — zeroing
depth barely perturbs z2 precisely *because* the encoder was already ignoring
it. A clean per-modality comparison needs encoders retrained per input, a
WP1-scale run.

**This is the most actionable result in WP2.** The elevation channel is the
direct terrain observation, `privterr` shows true local elevation is worth
0.3281 → 0.2211, and z2 reaches only 0.2867 *without using elevation at all*.
It supersedes the expectation — recorded before the run — that elevation would
be carrying the win.

**Why the channel goes unused is NOT low contrast.** Measured over 240 sampled
frames, the elevation channel's within-image std is **0.311**, about 2x any RGB
channel (R 0.072, G 0.101, B 0.186); RGB/elevation contrast ratio is 0.48x.
Rescaling or renormalizing the input is therefore not the fix, and the
"strengthen the elev recon weight" idea is attacking the wrong thing too.

The explanation consistent with every result so far is **the pooling stage, not
the channel**. Elevation's value is intrinsically *spatial and local* — "what
is the slope right where I am" — which is exactly the class of information
`AttnPool` destroys (WP1: BEV 0.38 pooled vs 0.88 from the pre-pooling map).
RGB can still contribute through global appearance statistics that survive
pooling; a height field cannot. Four independent lines now converge on this:

1. WP1 — pooling costs 0.88 → 0.38 BEV IoU, and pos-enc/slot-z do not fix it;
2. `privterr` — a *local, ego-aligned* terrain patch is worth 0.328 → 0.221;
3. Batch B — the global latent cannot exploit the elevation channel at all;
4. G4 — the predicted global latent is not decodable back into a scene.

The indicated fix is therefore the §5 fallback applied to the *dynamics* token,
not just the planner: give the transition model a **local/spatial
representation** — an ego-centric crop of the 16x16 spatial map at the vehicle's
dead-reckoned position, the learned analogue of `privterr`'s patch — instead of
a single pooled vector. This is an architecture-contract change (z2 is
specified as a global token in §8.1) and is left as a decision, not a
unilateral run.

## Negative result: the power head did not convert (and the test cannot work open-loop)

The hypothesis was that z2's terrain advantage would show up in energy, which
is what Planner-C scoring consumes. It did not: `state vs joint` on energy
error at 1.0 s is -0.561 kJ, CI95 [-0.818, -0.311] -- **joint is significantly
worse**, and priv beats joint too (+0.342 CI [+0.126, +0.565]).

The target is right: engine power is `engine_motorshaft_torque_nm x
trans_motorshaft_speed_radps`, and those two speed channels are bit-identical
in the store (max abs difference 0.000e+00, corr 1.0), so this *is* engine
torque x engine speed -- the energy the vehicle actually consumes. Wheel work
would be a different quantity and slip-contaminated; it is not the fix.

The reason the test is uninformative is that the target is almost entirely
determined by an input the model gets for free (200 full_v1 episodes,
39 928 samples):

| Predictor of engine power | R^2 |
|---|---|
| throttle only | 0.692 |
| throttle + engine speed | 0.693 |
| + their products | 0.856 |

and the residual after that carries no terrain signal at all --
`corr(residual, pitch) = +0.017`. (`corr(engine_speed, vx) = 0.380`: gear
changes decouple them, so this is not a vx effect either.)

**Deeper problem: the rollout feeds ground-truth actions**, so the recorded
throttle already encodes how the driver responded to the terrain. The model is
handed terrain-informed throttle for free, which is exactly why nothing is left
for z2 to explain. Energy scoring therefore cannot be evaluated open-loop: in a
real Planner-C rollout the tracker *generates* throttle, and terrain reaches
energy through the policy's response, not through a power head reading z2. The
energy question is downstream of WP3 (tracker), which independently agrees with
the v1.2 decision to order tracker before planner.

## Batch C: the spatial-token fix (the diagnosis was right)

`z2` replaced by an ego-indexed crop of the encoder's stage-2 scene map
(`map_crop.MapCropper`, `traverse_wp2_train_map.py`). Same backbone, data,
split, budget; only the sensor token changes.

| run | z1mae@1 s | terrain@1 s | z1mae@5 s | pose_m@1 s | closes gap @1 s / @5 s |
|---|---|---|---|---|---|
| state (3 seeds) | 0.3281 ±0.0080 | 0.4946 | 0.6082 ±0.0125 | 0.1434 | — |
| pooled z2 (3 seeds) | 0.2867 ±0.0141 | 0.4153 | 0.6256 ±0.0452 | 0.1602 | 38.7 % / −5.8 % |
| MAP index, leaky maps (2) | 0.2518 ±0.0008 | 0.3728 | 0.4390 ±0.0114 | 0.1235 | 71.3 % / 56.5 % |
| **MAP index, clean maps (2)** | **0.2478 ±0.0078** | **0.3641** | **0.4555 ±0.0294** | **0.1166** | **75.0 % / 51.0 %** |
| MAP predict (2 seeds) | 0.3886 ±0.0257 | 0.5787 | 1.0408 ±0.0074 | 0.1800 | −56.5 % / −144.6 % |
| privterr ceiling | 0.2211 | 0.3172 | 0.3090 | 0.1280 | 100 % |

**Pooling was the whole problem.** Indexing roughly doubles the terrain
information captured at 1 s (39 % -> 75 %) and converts a *negative* 5 s result
into 51 % of the ceiling. The leak-free maps (Batch D) score slightly *better*
at 1 s than the leaky ones, so the advantage is genuine terrain information and
not a parking-spot shortcut; Batch A's shuffle control said the same thing from
the other direction.

On pose error at 1 s the clean crop (0.1166 m) beats even the privileged
ceiling (0.1280 m): `privterr` is handed ground-truth pose and true elevation,
but its patch is bare terrain, whereas the learned crop also carries appearance
(obstacles, surface type). It also repairs the pose-channel regression (0.1235 m, better than
both state and pooled z2, where pooled z2 was worse than no sensor at all), and
the seed spread collapses to ±0.0008 from the pooled path's ±0.0141.

Note the privilege direction favours the result: `map index` is given
ground-truth pose only at t=0 and then dead-reckons from its own predicted z1
to decide where to crop, while `privterr` is handed ground-truth pose *and* the
true heightmap every step. The less privileged method closes 71 % of the more
privileged one's advantage.

**Index beats predict decisively, and that is a finding, not a tuning detail.**
On a static map the right operation is to *index* the scene at the predicted
pose, not to roll a sensor latent forward — the plan's own §18.3 argument, now
with dynamics-side evidence. Predict's collapse also has a specific cause worth
recording: its head chases the cropper's own output while the cropper is still
training, so the target representation drifts underneath it. Pooled-z2 predict
did not suffer this because its target came from a frozen, precomputed encoder.
Any future attempt at an autoregressive spatial token must freeze the projection
first.

### Leakage bug in the first scene maps (Batch D re-run)

The first scene maps used a plain temporal median of 9 frames, on the reasoning
that the vehicle is somewhere different in each frame and medians away. That is
false for a large minority of episodes. Measured over 400 episodes: the minimum
pairwise separation among the 9 sampled vehicle positions has median 4.4 m
(about one vehicle length) and 5th percentile 0.0 m, and **108/400 (27 %) have
>= 5 of 9 samples within one vehicle length** — vehicles park at route ends,
settle at the start, and sit still in contact episodes. In those the vehicle
wins the median vote and is baked into the map.

That is leakage rather than a cosmetic artifact: a blob at the parking spot
encodes *where the vehicle spends its time*, and the model can learn "blob under
me ⇒ I am stopping" and score better on z1 without any terrain understanding.

Fix: rasterize the vehicle footprint per frame from its recorded pose (the same
`_box_corners` / `_rasterize_solids` path WP1 uses), dilate 3 px for shadow and
antialiasing, and take the median **only over frames in which the vehicle was
elsewhere**; 16 frames instead of 9. Over the full 9518-episode set this leaves
**2171 pixels total that the vehicle never vacated — 0.2 per episode of
65 536** — which fall back to the plain median and are counted in the log.

Clean maps are written under cache key `map_v2`, leaving the original `map`
intact, so `wp2_map_index_*` (leaky) and `wp2_mapv2_index_*` (clean) differ only
in the leak. The Batch C figure of 71.3 % is provisional until that comparison
lands.

### Cluster note (cost a run)

The crop introduced the first convolution in any WP2 job. MIOpen keeps a shared
SQLite kernel cache and four jobs starting simultaneously raced on it, killing
one with `miopenStatusInternalError`. Fix: per-job
`MIOPEN_USER_DB_PATH=/tmp/miopen_$SLURM_JOB_ID`. Separately, the sbatch
templates ended with `echo "exit: $?"`, so SLURM reported **COMPLETED** for a
job that had crashed 17 s in — the templates now capture and re-raise the real
status. Job state alone was not a trustworthy success signal until that fix.

## G4 cross-modal: FAILS beyond ~0.5 s

`traverse_wp2_g4.py` applies the frozen WP1 `LatentProbe` (trained on ENCODED
z2) to the AUTONOMOUSLY PREDICTED z2 -- the dynamics model never saw the probe
and the probe never saw a predicted latent. 128 held-out episodes:

| metric | 0.5 s | 1.0 s | 2.0 s | 5.0 s |
|---|---|---|---|---|
| z2 cosine vs encoded truth | 0.995 | 0.983 | 0.938 | 0.597 |
| decoded blob vs true pose (m) | 2.57 | 4.01 | 6.43 | 11.90 |
| **same, decoded from ENCODED z2** | **1.40** | **1.41** | **1.37** | **1.30** |
| dead-reckoned pose vs truth (m) | 0.045 | 0.154 | 0.512 | 2.29 |
| BEV IoU from predicted z2 | 0.367 | 0.336 | 0.268 | **0.097** |
| **same, from ENCODED z2** | **0.384** | **0.385** | **0.387** | **0.382** |

**G4 does not pass.** The predicted latent stops carrying decodable scene
content almost immediately: at 0.5 s the decoded vehicle blob is already at
2.57 m against the encoded latent's flat 1.40 m, and BEV occupancy decays
0.37 -> 0.10 while the encoded reference holds at 0.38. The two branches do not
agree either -- `blob_vs_deadrec` tracks `blob_vs_gt` almost exactly, i.e. the
whole discrepancy is the decoded blob's error, not the dead-reckoned pose's
(which is 50x better at 0.5 s).

**Latent cosine is a misleading proxy for scene retention.** At 1.0 s the
predicted z2 still has cosine 0.983 against the truth while its decoded blob
error has already tripled. A small angular error in a 256-D latent destroys the
fine spatial content the probe reads. This is a hard confirmation of the plan
§8.3 warning that aggregate latent metrics alone are uninformative here, and it
means `z2_cos` must not be used as the horizon criterion.

Per-class permanence is not reported: WP1 established z2 carries essentially no
rock content (IoU 0.005, recall 0.007), so there is nothing for it to lose.
That question belongs to the spatial-map path.

### What this does and does not invalidate

It does **not** touch G3. z2's contribution there is to *condition the
dynamics*, and the terrain-channel win is measured directly on z1, not through
a decoder (the plan's own architecture table lists the HMMWV decoder as
"diagnostic only").

It **does** bound every use that reads the scene back out of a predicted
latent. Concretely, §9.5's Planner-C scorer rolls z2 forward over "short
imagined tracking segments"; any part of that scoring which depends on the
rolled-forward z2 still depicting the layout is limited to well under 1 s.
Scoring that only needs z2 to condition z1 is unaffected. This is the
measurement the §9.5 promotion was missing.

## z2 rollout horizon

Autonomous z2 cosine against the encoded truth: 0.995 (0.5 s) → 0.983 (1.0 s)
→ 0.940 (2.0 s) → **0.634 (5.0 s)**. z2's advantage over state narrows in step
with this decay (terrain-channel delta +0.095 at 1 s → +0.043 at 5 s). This
bounds how far Planner-C can imagine before the latent stops carrying the
scene, and it should be measured directly against the §9.5 segment length.

## Open

- Single seed throughout; the 5 s deltas have CIs that graze zero.
- Energy scoring is not testable open-loop (above); revisit with the WP3
  tracker in the loop generating actions.
- Privileged + local terrain patch row = the actual upper bound.
- §8.3 integrity ablations (z2 shuffled between layouts) not run; judged
  low-priority since 4 % more parameters cannot plausibly produce 30–60 %
  per-channel gains concentrated on terrain-coupled channels.

## Addendum 2026-09-04 (afternoon): pose-drift test and the two-stage prediction head

Both items from the session brief's "next steps" ran; scripts:
`traverse_wp2_train_map.py --eval-only` (pose drift) and
`--init-from ... --freeze-cropper` (two-stage). Checkpoint selection in both map
trainers is at the **longest** horizon (5 s), not the plan §8.2's 0.5–1.0 s; it
is consistent across every arm so the comparisons stand, but every "@1 s" number
in these tables is for a 5 s-selected checkpoint.

### Pose drift explains less than half of the 5 s shortfall

Same clean-map index checkpoints, 1024 held-out episodes, crop taken at the
dead-reckoned pose (honest) vs at the true pose (`posedrift_readout.json`):

| seed | z1mae@1 s dr / gt | z1mae@5 s dr / gt | terrain@5 s dr / gt | pose_m@5 s dr / gt |
|---|---|---|---|---|
| 1 | 0.2487 / 0.2482 | 0.4546 / **0.3936** | 0.6226 / 0.5164 | 1.879 / 1.879 |
| 2 | 0.2558 / 0.2552 | 0.4534 / **0.3995** | 0.6186 / 0.5275 | 1.890 / 1.878 |

At 1 s the crop position does not matter at all. At 5 s reading the map at the
true pose recovers **0.06 of the 0.146 gap** to the privileged ceiling (0.309),
i.e. ~40 % of the long-horizon shortfall is "reading the wrong place"; the rest
is accumulated state error that a correctly placed crop does not repair. Note
the true-pose crop does not improve the dead-reckoned pose itself (1.88 m
either way): better terrain context does not feed back into localization.

### Two-stage prediction head: works one-step, loses autoregressively

Starting from the clean-map index checkpoints, the map projection was frozen,
crop statistics fitted (mean |.| 0.024, std 0.052 — the raw-scale token loss in
Batch C was ~60x smaller than the z1 loss, which is why that head never
trained), and a next-crop head trained on **normalized** targets for 20k steps.

| run | z1@1 s | z1@5 s | token cos @0.5 / 1 / 5 s | train token loss (norm. MSE) |
|---|---|---|---|---|
| index (reference, 2 seeds) | 0.2478 | 0.4555 | exact | — |
| two-stage predict, seed 1 | 0.3511 | 0.7202 | 0.79 / 0.57 / 0.52 | 0.044 |
| two-stage predict, seed 2 | 0.3253 | 0.6852 | 0.77 / 0.53 / 0.49 | 0.045 |
| head only (backbone frozen) | 0.3880 | 0.8685 | 0.70 / 0.61 / 0.48 | 0.438 |
| **persistence** (hold the crop) | — | — | **0.87 / 0.76 / 0.58** | — |
| state-only | 0.3281 | 0.6082 | — | — |

Teacher-forced one-step prediction is fine (4 % residual variance), but fed
back autoregressively the predicted crop falls **below persistence within
0.5 s** and drags z1 down to state-only level. The crop token itself is
fast-moving: measured on held-out episodes, holding it fixed gives cosine 0.87
at 0.5 s and 0.76 at 1 s (mean-centred 0.72 / 0.50), a 10° yaw rotation of the
window alone costs 0.055 cosine. On a static map the next crop is a
deterministic function of the next pose, so an autoregressive head can only add
error to what indexing gives exactly. **Conclusion: the ẑ₂ branch does not
earn its place on a static, fully observed scene, and this time the test was
fair** (stationary target, normalized loss, two seeds, persistence baseline).
v1.3's "index, don't predict" stands on evidence; the branch returns when the
scene moves or the camera does (§16 deferred ladder).

### Static-scene assumption: holds at the map level (`traverse_wp2_static_check.py`)

64 held-out episodes, two scene maps each from disjoint halves of the recording
(frames 0–199 vs 200–399, same masked-median procedure, 16 frames each):

| comparison | same episode, two halves | different layout (floor) |
|---|---|---|
| whole-map feature cosine | **0.996** (p5 0.974) | 0.798 |
| per-cell feature cosine | 0.999 | — |
| ego-window token cosine along the recorded trajectory | 0.948 (p5 0.625) | 0.876 |

The scene is static: two maps built from non-overlapping time ranges agree to
0.996 against a 0.80 floor. The window-level tail (p5 0.63) is the masked
median's residual, not the scene — with only half the frames the vehicle vacates
fewer pixels, so a parked or settling vehicle leaks into one half's map and
shows up in the windows read near that spot. It argues for building the
production map from all 16 frames spread over the episode (as done) and is a
caveat on map construction, not on the assumption.
