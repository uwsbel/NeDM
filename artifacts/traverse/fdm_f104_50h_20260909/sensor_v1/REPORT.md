# sensor_v1 — depth camera channels as the network's input (2026-09-15)

Pre-registered in `PLAN.md` (primary test, then a declared follow-up after the primary failed). Model architecture,
loss, training rows and schedule are the deployed night-2 model's; only the corridor channels change. Everything
trained on the AMD cluster (MI350), driven in Chrono on the cluster, all arms of a start/goal on one node.

## What the "current input" actually was
Already one overhead RGB-D capture per arena (camera 110 m up, 1024 x 1024): depth was converted to height with
the camera's known geometry, then the network got that height plus two hand-computed slopes (grade, cross-slope);
the colour channels were never used. Tonight's change: capture the five sibling arenas too (the f104 re-capture is
byte-identical to the original), and feed the network the camera's own channels.

## Offline (ranking AUC on the unsafe label)
| Input | f104 dev fold, same speed | new arenas, within start/goal | new arenas, same speed |
|---|---|---|---|
| deployed model (height + slopes) | — | 0.950 | 0.893 |
| E: height + slopes, retrained (control) | 0.978 | 0.949 | 0.890 |
| E0: height from depth only, no slopes | 0.979 | 0.944 | 0.882 |
| D: raw depth + camera ray angle | **0.982** | 0.932 | 0.861 |
| RGBD: colour + raw depth | 0.967 | 0.865 | 0.672 |
| RGB: colour only | 0.968 | 0.851 | 0.626 |
New-arena columns: 9,000 routes collected there in gen_v1, never trained on. Colour helps a little on f104 and hurts
a lot on new terrain: the network latches onto shading patterns of the one arena it saw.

## Closed loop, test 1 (pre-registered primary; 1,200 fresh start/goals over f104 + 5 new arenas; 5,428 runs)
Deployed pick by the rule "best f104 dev score among sensor variants" = **D (raw depth)**.
| Unsafe (failed or slid) | D | current | difference [95% CI] | discordant | p |
|---|---|---|---|---|---|
| **fixed 2 m/s, all arenas (primary, margin +2.0)** | 9.42% | 7.17% | +2.25 [+0.50, +4.00] | 72 vs 45 | 0.016 — **failed** |
| speed free, all | 2.17% | 1.08% | +1.08 [+0.25, +2.00] | 20 vs 7 | 0.019 |
| f104 only, 2 m/s / speed free | 2.5% / 0.5% | 2.5% / 0.5% | 0 / 0 | 1v1 / 0v0 | 1.0 |
f104 five-goal missions (100): D 96% vs current 99% (1 vs 4, p = 0.38); D 7 s faster per mission.

## Closed loop, test 2 (declared follow-up; another 1,200 fresh start/goals; 6,759 runs)
| Unsafe | E0 (height from depth, no slopes) | D (raw depth, replication) | current |
|---|---|---|---|
| fixed 2 m/s, all arenas | 6.50% | 9.25% | 5.67% |
| speed free, all arenas | 1.00% | 1.58% | 0.75% |
| fixed 2 m/s, f104 | 2.0% | 2.0% | 1.5% |
E0 vs current, 2 m/s: +0.83 [-0.58, +2.25], 42 vs 32, p = 0.30 — not different, but the upper bound misses the
+2.0 margin by 0.25, so non-inferiority is **not formally shown**. Speed free: +0.25 [-0.33, +0.83], **non-inferior**
at the +1.0 margin. Failures and tilt: no significant differences. D replicates test 1: +3.58 [+1.83, +5.33] at 2 m/s
(78 vs 35, p < 0.001), +0.83 speed free (p = 0.03). Both sensor models beat the 6 m/s straight line (p < 0.001).

## Reading
- Raw depth works on the arena it trained on (identical to the current model on f104, twice) but generalises worse:
  from a camera looking down, raw depth mixes terrain height with the viewing angle, which changes across the image,
  and one arena's worth of views is not enough for the network to learn that geometry. Replicated on 2,400 start/goals.
- Undoing the viewing angle with the camera's known intrinsics (no terrain knowledge, no hand-made slopes) recovers
  essentially all of the performance; the hand-computed slope channels are not needed.
- Colour channels make the model worse on new terrain (offline); not driven.

## Caveats
- The camera is a single static overhead capture per arena (perfect pose, no noise, no occlusion), not an onboard sensor.
- Test 2 was designed after test 1 failed; it used fresh start/goals and was declared before any of its runs.
- The analysis script was refactored between the tests; re-running test 1 through it reproduces the primary exactly
  (bootstrap CIs of secondary rows can move by ~0.1 point because the random draws are consumed in a different order).
- The current model's own 2 m/s rate differs between the two tests (7.2% vs 5.7%) and from gen_v1 (5.9%, heightmap):
  different start/goals; comparisons are only made within a test.
