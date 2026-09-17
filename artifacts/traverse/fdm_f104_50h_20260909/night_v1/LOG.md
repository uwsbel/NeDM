# night session 2026-09-10T23:25:34-05:00

## [step 1, ~23:32] data
- 12,542 driven routes indexed (11,721 designed + 821 Chrono MPPI variants). 0 errors.
- unsafe = NOT(goal & no rollback & min vx > -0.30): 23.8% train vs 11.0% fail.
- 96.5% of first danger events are rollbacks (user's negative-vx intuition confirmed).  [CORRECTED 09-11, step 11: 83.0% rollback, 16.7% effortful near-stop, 0.3% terminal; the 96.5% came from reading the steering column as throttle]
- Label noise floor ~1%: 99.0% of 1,233 (2,4,6 m/s) same-route triples are monotone in speed.
- Pre-event path drift: unsafe p95 5.9 m vs clean 0.9 m -> candidate down-weighting.
- station_ds.npz: 5 ch x 96 st x 32 lat (elev_rel, grade_along, grade_cross, speed, valid) + old features.

## [step 2, ~23:38] dev sweep (partial, 3 seeds each; dev = 20% of train groups, 307 unsafe / 201 fail same-speed pairs)
- A0 old/fail  G_unsafe 0.711  G_fail 0.726  P_fail 0.876   (reproduces known ~0.72 terrain-only baseline)
- A1 old/unsafe 0.863 / 0.786 ; A2 +speed_drop 0.852 / 0.756 ; A3 +pair 0.830 / 0.700  -> dropout & pair HURT old arch
- H1 hazard     0.947 / 0.879  P_fail 0.934 ; H2 +speed_drop 0.946 / 0.877
- SUSPICIOUS size. Must ablate route-length shortcut (ctx L varies within same-speed pairs) + valid mask before trusting.

## [step 3, ~23:40] shortcut ablation (DEV, H1 seed0) -> gain is genuine terrain reading
- route length -> group mean: G unchanged (0.948/0.886); length swapped: unchanged. NO length shortcut.
- terrain swapped within same-group same-speed cell: vs OWN label 0.176/0.229 (inverts), vs PARTNER label 0.948/0.886 (== baseline).
  => within same-speed pairs the hazard model's score is a function of terrain only.
- old A1: patch swap vs partner 0.671 (baseline 0.873) -> old model's signal half in scalar summaries, not the map.

## [step 4, ~23:43] attribution (dev G_fail, true failures, same group same speed)
A0 0.726 -> A1 (unsafe label) 0.786 -> H6 (station arch, route label) 0.816 -> H3 (clip-at-event hazard label) 0.852 -> H1 (no sdrop/pair) 0.879
Speed dropout and pair loss HURT on both archs. Clip-at-event (user idea) ~ +0.036-0.05, comparable to the architecture change.
SELECTION RULE (written before test): max dev G_unsafe; |diff|<0.01 = tie -> G_fail. Currently H1 (H3 0.952 vs H1 0.947 tie; H1 wins G_fail).

## [step 5, 23:47] SELECTION LOCKED (before any test-set look): H1 (plain hazard, clip-at-event labels)
dev G_unsafe: H3 .952 max; H1 .947 H2 .946 H4 .949 H5 .946 all within .01 -> tie -> G_fail: H1 .879 > H4 .866 > H3/H5 .852.
Dev G_fail is coarse (201 pairs, ~0.01 steps): H-variants are 2-3 steps apart; robust effect is A->H.

## [step 6, ~23:55] one-shot TEST (73 groups; 1441 unsafe / 1264 fail same-speed pairs), deployed OLD vs NEW
paired group-bootstrap NEW-OLD: unsafe terrain-only +0.185 [+.145,+.229]; fail terrain-only +0.055 [-.001,+.105] (1-sided p .027);
fail within-group +0.073 [+.041,+.101]; fail pooled +0.073 [+.047,+.102]. Dev had overstated fail terrain-only (+.15) -> winner's curse.
speed/terrain/interaction share of within-group risk (73 full 3x3 test tables): TRUTH unsafe 48.8/23.5/27.6;
OLD 61.3/17.5/21.2; NEW 53.8/22.5/23.8; A1 (old arch new label) 61.3/17.4 = unchanged -> architecture does the rebalancing.
Chrono paired job 414101: 123 groups (73 test + 50 val), 238 episodes; models disagree on 115/123.

## [step 7, 00:07] CLOSED-LOOP CHRONO, paired (same 256-candidate sets, same node), 123 held-out groups, 2 m/s geometry-only
UNSAFE: OLD 44.7% -> NEW 17.9%  (+26.8 pts CI [+18.7,+35.8]; discordant 34 vs 1; McNemar p<1e-4)
FAIL:   OLD 15.4% -> NEW  6.5%  (+8.9 pts CI [+1.6,+16.3];  discordant 17 vs 6; p=0.035)
test-only fail +5.5 (p=0.39, n.s.); val fail +14.0 (p=0.065). OLD test fail 12.3% == earlier smooth-arm baseline (harness reproduces).
Test-set attribution correction: unsafe terrain-only gain = label +0.117, arch +0.052, clip-at-event -0.001.

## [step 8, 00:12] AUDIT (2 of 3 back): weakened, not refuted. No leakage; splits clean; numbers reproduce.
- ~71% of +0.185 unsafe terrain-only gain is the LABEL (A1-OLD +0.131). Architecture-only NEW-A1: +0.053 [+.018,+.093].
- Fail terrain-only vs label-matched A1: +0.021 [-.020,+.061] n.s.; designed-only fail NEW-OLD +0.060 n.s.
- RETRACT decomposition (C3): n.s. and vanishes with per-group weighting (OLD 50.9 vs NEW 50.2 speed share).
- C2 largely by construction; valid content = no length shortcut; TEST own-label 0.41-0.46 not 0.18-0.23.
- Process: H1_full started 23:42 before tie rule written / all dev runs done; test groups seen in earlier session.
NEXT: closed-loop A1 (old arch, unsafe label) vs NEW, same candidates + same node -> isolates architecture.

## [step 9, 00:36] 3-ARM CLOSED LOOP (123 groups, same candidates, same node): OLD / A1 (old arch+unsafe label, 5-seed full) / NEW
UNSAFE 44.7 / 26.0 / 17.9 ; FAIL 15.4 / 17.1 / 6.5
label+data+ens (OLD->A1): unsafe +18.7 [+10.6,+26.8] p<1e-4 ; fail -1.6 [-8.1,+4.9] p=.81 (NO effect on failures)
architecture (A1->NEW):   unsafe +8.1 [+2.4,+14.6] p=.021 ; fail +10.6 [+4.1,+17.9] p=.0044
test-only architecture n.s. (unsafe p=.34, fail p=.18); val-only significant; pooled significant.
Reproducibility: 0/123 OLD and 0/123 NEW unsafe flips when re-driven in a second job.
=> offline audit said architecture's fail gain n.s.; closed loop (argmin tail) says it is the part that cuts real failures.

## [step 10, 01:06] REPLICATION under SPEED+GEOMETRY MPPI (the regime of the 90%-failure demo), fresh seeds, 123 groups, 3 arms same node
UNSAFE OLD 39.8 / A1 18.7 / NEW 10.6 ; FAIL 13.8 / 10.6 / 3.3
total OLD->NEW: unsafe +29.3 (36 vs 0, p<1e-4); fail +10.6 (14 vs 1, p=.001)
architecture A1->NEW: unsafe +8.1 (10 vs 0, p=.002); fail +7.3 (10 vs 1, p=.012)
labels OLD->A1: unsafe +21.1 (27 vs 1); fail +3.3 (p=.45, n.s.)  -> same division of labour as the fixed-speed test.
val-only NEW: 0/50 failures.

## [step 11, 2026-09-11 afternoon] EXAMPLES + THROTTLE-COLUMN BUG (found by a 14-agent verification pass)
BUG: action columns are [steering, throttle, braking] (scripts/traverse_fdm_rgbd_diverse_chrono.py:235). f104_night_index.py,
f104_night_paired_analyze.py, f104_night_examples.py, f104_vx_clean_vs_dirty.py read column 0 (steering) as throttle. Fixed (column 1).
Impact, verified by re-running:
- Chrono closed-loop labels: 0 flips in all three tests; every rate and McNemar count in steps 7, 9, 10 is unchanged (re-run printed identical numbers).
- Training labels (episodes.json, 12,542 routes): 16 flip safe->unsafe (11 train / 2 val / 3 test), 0 the other way, 0 fail changes;
  event station changes for 121 routes (16 new events, 105 move 1-8 stations earlier; 86 in train). About 1% of labels.
  The deployed H1/A1 models were trained on the old labels: episodes_v0_steering_col.json keeps them. station_ds.npz NOT rebuilt, no retrain.
- First danger event: 83.0% rollback / 16.7% effortful near-stop / 0.3% terminal (was reported as 96.5% rollback).
- Every stored back_s ("seconds rolling back under throttle") was computed from steering; real values are larger (e.g. 0751 0 -> 5.55 s).
- Physics: every slide frame (598,563 frames, vx < -0.30) has throttle >= 0.999 on 99.94%, brake 0, forward gear; one wheel spins
  forward while the others turn backwards: the wheel-lift + open-differential stall. The speed controller sees signed speed.
Also confirmed: arena_meta.json 'features' are y-mirrored vs the Chrono world (world y = -listed y); the camera raster used in figures is correct.

EXAMPLES (examples/examples.png, scripts/f104_night_examples.py; speed+geometry test, NEW picks):
- Predicted risk vs outcome: <0.1%: 84 picks, 0 slid/0 failed; 0.1-1%: 19, 1/0; 1-5%: 6, 2/0; 5-20%: 3, 0/0; >=20%: 11, 10/4.
- GOOD (clean, no zone entered, 2 m/s straight line crossed a zone and was unsafe, min vx >= 1 m/s; 8 groups fit; 5 lowest tilt):
  0201 0549 0681 0915 0899. Verified fair. Caveats: 0549 is a crater-rim detour (hill clip only 4 cm); 0681 OLD/A1 picked similar clean detours;
  straight line at 6 m/s was clean in most; one-wheel lift ~1 s is normal here (95% of clean runs > 0.2 s). v1 rule (min speed only) picked
  routes with ~30 deg sustained side tilt on hill flanks; revised to rank by tilt.
- SURPRISES (unsafe with risk < 5%): 0751 (0.8%), 1106 (3.4%), 0045 (3.4%). Each pick was the ONLY candidate under 5% (next best 12%, 8%, 20%;
  most candidates ~100%): a lone low score found by the argmin, in start/goals the model otherwise rated very risky. 0751: goal on a hill; NEW did
  best of the three arms. All three slid at full throttle after a wheel lifted, then reached the goal on a later attempt.
- FAILURES (4/123): 1381 (53%), 0469 (59%), 1006 (99.9%), 0200 (99.9%). Every candidate scored >= 53%. Cause is the sampler: lateral spread
  <= ~3-4 m and route-average speed <= ~3.3 m/s, and the sin^2 envelope returns speed to 2 m/s near start and goal (goal on a hill in 0200,
  at a crater rim in 1006). The straight 6 m/s line was clean in all 4; in 1381, 7 of 12 designed routes were clean.
- Warning flag (post hoc): best OR second-best candidate >= 5% flags 17 groups = 13/13 unsafe + 4/110 clean. With the straight 6 m/s line
  as fallback when flagged (offline, original-collection drives): unsafe 3/123, failed 0/123 (3 flagged groups had no 6 m/s drive -> kept pick).
