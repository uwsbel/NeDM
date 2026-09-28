# Would a bigger or wider route search help the Gator on soil? (offline probe, 2026-09-27)

**The question.** The Gator-trained soil planner reached the goal on 539 of 800 soil start/goal pairs on f104 when
the Gator drove its picks. On 134 of its 261 failures, the route it picked was already rated more than 50 % likely
to fail by its own model. Does a bigger search, or a wider set of route shapes, find routes that the same model
rates safe on those pairs?

Everything here is offline. Routes are judged only by the model's own prediction; nothing was driven, nothing was
sent to the cluster, nothing was committed.

## Short answer

- **Yes for about half of the "known risky" pairs, and mostly because of wider route shapes, not more search effort.**
  - **The key count:** on **71 of the 134 pairs** where the original planner knew its best route was risky, the
    wider search found a route the model rates safe (P < 0.05). Here "the wider search" means sideways swings up to
    20 m and 5 bend terms, with the much bigger 16 x 512 search.
    - 58 of the 71 are rated below 1 %.
    - 56 of the 134 routes are rated below 5 % by all five networks of the ensemble.
  - **Within the original route shapes**, more search helps less (safe counts out of 134):
    - 8 x 256 sampling: 20.
    - 16 x 512 sampling: 27.
    - Gradient refinement: 40 starting from the original search, 33 starting from the 16 x 512 search.
    - Any of the five bigger or wider searches: 73 (a best-of-five count, so optimistic).
  - **On 46 of the 134 pairs even the widest search's best route is still rated above 50 %.** For those pairs, no
    route of this shape family looks passable to the model.
- **The wider limit on sideways swing matters most.** Both halves of the wider family help:
  - The sideways limit alone (20 m, 3 bend terms): 60 safe.
  - The extra bend terms alone (5 terms, 10 m): 47 safe.
  - Both: 71. Neither (same 16 x 512 budget): 27.
  - 47 of the 71 safe routes swing more than 10 m sideways. That is beyond what the original route shapes could
    reach, and beyond every Gator training route (caveat 4).
  - Along those 71 routes, the steepest 2 m uphill grade falls from 29 % to 21 % (median). The total climb falls
    from 3.05 m to 2.46 m.
  - So in the model's eyes, the safe routes mainly skirt the steepest part of the hill rather than avoiding climbing
    altogether.
- **Half of the failures cannot be addressed by search at all.** The other 127 failed pairs were pairs the model did
  not see coming: 84 of them already had a pick rated below 5 %. A bigger search only lowers those ratings further
  (to below 5 % on 121 of 127 with the wider search). That says nothing about the drives; those failures need a
  better model, not a bigger search.
- **The controls are safe.** On 100 pairs the Gator completed, every search keeps the rated risk low:
  - The wider search rates all 100 routes below 1 %.
  - Routes are not absurdly long: the median is 1.06 times the straight-line distance, the longest 1.23.
  - Routes are faster: the median commanded time is 0.77 of the driven route's.
- **The main caveat.** Model-rated safety is not driven safety, and this model has been confidently wrong before.
  Of the driven picks it rated 1-5 %, **52 % failed** (32 of 62). Of those rated below 1 %, 9.6 % failed (52 of 539).
  A bigger search chooses routes *because* the model likes them, so it also collects the model's mistakes. Only
  driving these routes can tell whether any of the 71 are real.

## What was done

- **The model.** The Gator-trained soil ensemble on f104 (study tag G_full): five networks trained on all 15,235
  Gator soil drives. Checkpoints were checked against the original pick manifest by sha256.
  - "P" is the ensemble-mean predicted probability that the drive fails.
  - "Worst network" is the most pessimistic of the five.
- **The decision.** Exactly as in the evaluation:
  - The vehicle stands still at the case start pose, with an empty history.
  - Candidate routes bend the straight route to the goal sideways (a sum of sine-shaped bends that is zero at both
    ends) and re-time it (4 speed knots).
  - The route checker is unchanged: turning curvature at most 0.125 1/m (radius 8 m), speed 0-6 m/s, acceleration
    at most 1.5 m/s^2, braking at most 2 m/s^2, vehicle footprint inside +-40 m.
- **The objective.** The same as the original: the lowest ensemble-mean risk.
- **The pairs.**
  - FAIL: all 261 pairs the Gator failed with the original pick.
  - FAIL P>0.5: the 134 of those whose pick was rated above 50 %.
  - CONTROL: 100 of the 539 completed pairs, the 100 lowest md5(pair id).
- **Seeding.** Every search is seeded from the pair id and a search name, so it re-runs identically.

| label | the search, in words | routes scored per pair |
|---|---|---|
| A0 | the original search, re-run: 4 rounds of 64 routes. Each round keeps the best 15 % and re-centres the sampling on them (cross-entropy method) | 257 |
| A1 | bigger: 8 rounds of 256 | about 2,050 |
| A2 | much bigger: 16 rounds of 512 | about 8,200 |
| A3 | gradient refinement: 64 starting routes (the original pick plus the original search's first-round pool), each pushed downhill in predicted risk through the model for 300 steps. Best route kept by ensemble-mean risk; the start route is kept if nothing beats it | 257 + 64 x 300 steps |
| A3b | the same refinement, started from the 64 best distinct routes of A2 | 64 x 300 steps on top of A2 |
| A4 | wider route shapes: 5 sideways bend terms instead of 3 and a sideways limit of 20 m instead of 10 m, with the 16 x 512 search. Each bend's size is still capped by the same curvature formula, and the route checker is unchanged | about 8,200 |
| A4a | ablation of A4 (the 134 pairs only): 3 bend terms, 20 m limit, 16 x 512 | about 8,200 |
| A4b | ablation of A4 (the 134 pairs only): 5 bend terms, 10 m limit, 16 x 512 | about 8,200 |

**How the gradient refinement was called.**
- The existing refinement code (scripts/ci_grad.py) was written for moving decision states with a driving history.
- Called in-process with no decision-state file, it takes exactly the standing start:
  - the case start pose and the straight base route;
  - the recorded speed is 0 ("standing start");
  - the history window is empty. The Gator models use no history input at all, so the gradient chain gets no
    history context.
- A3 is that function (`plan_group`) with 64 starts, 300 steps, stop after 50 steps without improvement, learning
  rates 0.02 (sideways) / 0.10 (speed), and best-route choice by the ensemble-mean risk with no abstention margin.
  The default was the worst-network risk with a 0.3 margin.
- A3b uses the same pieces (route chain, optimiser, route checker, final re-score) from A2's routes. A self-test
  shows this path reproduces the existing refinement's pick exactly when given the same starts.

## Results

### 1. The 134 pairs the planner already rated risky (the main table)

Each cell counts the 134 pairs by the predicted failure probability P of the chosen route. "All 5 networks < 0.05"
means even the most pessimistic network agrees. The stage-1 model is a second opinion, explained below.

| search | P<0.01 | P<0.05 | P<0.2 | P>0.5 | median P | all 5 networks < 0.05 | mean < 0.05 but worst network > 0.5 | stage-1 model also < 0.05 |
|---|---|---|---|---|---|---|---|---|
| recorded pick = A0 | 0 | 0 | 0 | 134 | 0.999 | 0 | 0 | 0 |
| A1 8 x 256 | 14 | 20 | 34 | 82 | 0.795 | 14 | 1 | 12 |
| A2 16 x 512 | 21 | 27 | 44 | 72 | 0.660 | 16 | 1 | 21 |
| A3 gradient from A0 | 24 | **40** | 57 | 62 | 0.348 | 22 | 6 | 28 |
| A3b gradient from A2 | 23 | 33 | 52 | 63 | 0.401 | 22 | 3 | 23 |
| **A4 wider shapes, 16 x 512** | **58** | **71** | **83** | **46** | **0.033** | **56** | 5 | **54** |
| A4a 3 terms, 20 m | 46 | 60 | 69 | 49 | 0.145 | 46 | 3 | 43 |
| A4b 5 terms, 10 m | 37 | 47 | 61 | 62 | 0.347 | 26 | 5 | 33 |

P deciles on these 134 pairs (0 %, 10 %, ..., 100 %):

| search | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recorded = A0 | 0.507 | 0.730 | 0.836 | 0.962 | 0.995 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| A1 | 1.7e-4 | 0.010 | 0.122 | 0.277 | 0.510 | 0.795 | 0.985 | 0.999 | 1.000 | 1.000 | 1.000 |
| A2 | 6.2e-5 | 0.004 | 0.048 | 0.123 | 0.315 | 0.660 | 0.947 | 0.997 | 1.000 | 1.000 | 1.000 |
| A3 | 1.0e-4 | 0.001 | 0.016 | 0.051 | 0.173 | 0.348 | 0.853 | 0.980 | 0.999 | 1.000 | 1.000 |
| A3b | 6.2e-5 | 0.002 | 0.014 | 0.083 | 0.202 | 0.401 | 0.865 | 0.990 | 0.999 | 1.000 | 1.000 |
| A4 | 4.1e-5 | 2.4e-4 | 7.0e-4 | 0.002 | 0.006 | 0.033 | 0.136 | 0.701 | 0.925 | 0.991 | 1.000 |

Worst-network P deciles are in tables.md.

**By terrain type** (the evaluation stratum of each pair; rated safe by A2 / by A4):

| terrain type | pairs | A2 | A4 |
|---|---|---|---|
| across a hill slope | 64 | 10 | 37 |
| onto a hill, off across the slope | 28 | 8 | 15 |
| across a crater slope | 20 | 3 | 5 |
| into a crater, off across the slope | 9 | 2 | 4 |
| long traverse | 9 | 2 | 6 |
| roughness change | 4 | 2 | 4 |

The wider shapes help most on hill slopes. Crater pairs stay hard.

### 2. All 261 failures, and the 127 failures the model did not see coming

| search | all 261: P<0.05 | all 261: P>0.5 | all 261: median P | 127 unforeseen: P<0.05 | 127 unforeseen: median P |
|---|---|---|---|---|---|
| recorded = A0 | 84 | 134 | 0.529 | 84 | 0.018 |
| A1 | 130 | 82 | 0.054 | 110 | 0.004 |
| A2 | 143 | 72 | 0.020 | 116 | 0.002 |
| A3 | 160 | 62 | 0.014 | 120 | 0.001 |
| A3b | 153 | 63 | 0.013 | 120 | 0.002 |
| A4 | 192 | 46 | 0.002 | 121 | 0.001 |

On the 127 unforeseen failures the model was already wrong. Pushing its rating even lower is not evidence of
anything.

### 3. Control: 100 pairs the Gator completed

| search | P<0.01 | P<0.05 | P>0.5 | median P | worst network > 0.5 | length / straight line, median [max] | length / driven route, max | commanded time / driven route, median |
|---|---|---|---|---|---|---|---|---|
| recorded = A0 | 90 | 96 | 0 | 5.4e-4 | 4 | 1.06 [1.15] | 1.00 | 1.00 |
| A1 | 94 | 99 | 0 | 2.2e-4 | 3 | 1.06 [1.16] | 1.16 | 0.84 |
| A2 | 97 | 99 | 0 | 1.8e-4 | 1 | 1.06 [1.17] | 1.16 | 0.78 |
| A3 | 98 | 99 | 0 | 1.5e-4 | 1 | 1.06 [1.17] | 1.16 | 0.74 |
| A3b | 99 | 99 | 0 | 1.5e-4 | 1 | 1.06 [1.16] | 1.16 | 0.76 |
| A4 | 100 | 100 | 0 | 1.2e-4 | 0 | 1.06 [1.23] | 1.17 | 0.77 |

Bigger search does not trade the completed pairs away, in the model's eyes. Routes stay short and get faster.

### 4. Route shape of the picks (the 134 risky pairs; median [90th percentile, max])

| search | length / straight line | commanded time s | largest sideways offset m | at the sideways limit | a bend term at its curvature cap | mean speed m/s | mean speed > 5 m/s | >= 3 of 4 speed knots at the +-4 m/s limit | within 6 m of the arena edge |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 1.07 [1.12, 1.18] | 17.2 [27.1, 43.4] | 6.7 [9.8, 10.0] | 10 | 43 | 3.03 [3.82, 5.43] | 1 | 0 | 3 |
| A1 | 1.09 [1.13, 1.17] | 14.5 [21.0, 30.2] | 7.7 [9.9, 10.0] | 9 | 58 | 3.68 [4.34, 4.71] | 0 | 0 | 6 |
| A2 | 1.10 [1.14, 1.17] | 13.4 [21.9, 44.2] | 7.8 [10.0, 10.0] | 20 | 67 | 4.00 [4.83, 5.48] | 8 | 0 | 11 |
| A3 | 1.10 [1.14, 1.18] | 12.6 [23.1, 36.2] | 8.6 [10.0, 10.0] | 29 | 22 | 4.43 [5.48, 5.74] | 36 | 19 | 8 |
| A3b | 1.10 [1.14, 1.17] | 13.0 [19.0, 46.5] | 7.9 [10.0, 10.0] | 29 | 18 | 4.40 [5.18, 5.74] | 23 | 16 | 11 |
| A4 | 1.13 [1.20, 1.24] | 13.6 [18.3, 30.7] | 9.2 [13.6, 16.1] | 0 (limit 20 m) | 69 | 4.03 [4.82, 5.61] | 8 | 0 | 14 |

- "At the sideways limit" means within 0.05 m of 10 m (20 m for A4). No A4 pick reached 20 m; the largest offset
  on any failed pair is 18.4 m.
- The route checker still bounds every pick. The largest |x| or |y| of any waypoint is 38.2 m (in A4a); the
  checker requires the vehicle footprint to stay within +-40 m.
- In the wider family, the curvature cap on a bend term is binding in half the picks (69 of 134). The 8 m
  turning-radius limit, not the 20 m clip, is now what stops the shapes from swinging wider.
- The gradient search runs the speed to its limits. A3 has 36 of 134 routes averaging above 5 m/s, and 19 with at
  least 3 speed knots pinned at +4 m/s. This is the classic sign of an optimiser pressing against the edge of the
  parameter box. On soil the Gator failed 83 % of constant 6 m/s drives (REPORT section 3.2), so fast picks deserve
  suspicion.

### 5. Terrain along the picks (from the planner's own elevation map)

| pairs | search | total climb m, median [90th pct] | steepest 2 m uphill grade %, median [90th pct] | picks climbing > 0.5 m less / more than the driven route |
|---|---|---|---|---|
| 134 risky | recorded = A0 | 3.12 [4.47] | 30.9 [39.3] | - |
| 134 risky | A2 | 2.99 [4.24] | 29.6 [38.0] | 19 / 13 |
| 134 risky | A3 | 3.08 [4.36] | 29.0 [38.7] | 18 / 15 |
| 134 risky | A4 | 2.87 [4.34] | 26.8 [36.3] | 26 / 9 |
| 71 made safe by A4 | recorded | 3.05 | 29.2 | - |
| 71 made safe by A4 | A4 | 2.46 | 21.3 | 41 of 71 have a steepest grade more than 5 points lower |
| 100 control | recorded / A4 | 2.08 / 1.99 | 17.8 / 18.5 | 5 / 1 |

### 6. Planning time per pair

These times were measured on this workstation, which was shared with other jobs throughout: load about 20-24 on 16
threads, and 3 of my processes at a time. For comparison, the original search took 0.39 s per pair unloaded.

| search | mean s per pair | total minutes (361 pairs; 134 for ablations) | relative to A0 on the same machine |
|---|---|---|---|
| A0 | 1.0 | 6 | 1x |
| A1 | 4.9 | 30 | 5x |
| A2 | 19.8 | 119 | 20x |
| A3 | 14.0 (includes its own A0 search) | 84 | 14x |
| A3b | 12.5 (plus A2's 19.8 it starts from) | 75 | 32x with A2 |
| A4 | 21.8 | 131 | 22x |
| A4a | 18.4 | 41 | - |
| A4b | 22.1 | 49 | - |

Nothing was subsampled; A2 ran on all 361 pairs.

## How much to trust "rated safe" (caveats)

1. **Rated safety is not driven safety, and the model's low ratings have been too optimistic.** This is how the
   driven picks on all 800 pairs did, by their rating:

   | rated P | picks | failed | failure rate |
   |---|---|---|---|
   | < 0.01 | 539 | 52 | 9.6 % |
   | 0.01-0.05 | 62 | 32 | 52 % |
   | 0.05-0.2 | 28 | 18 | 64 % |
   | 0.2-0.5 | 29 | 25 | 86 % |
   | 0.5-0.9 | 38 | 32 | 84 % |
   | > 0.9 | 104 | 102 | 98 % |

   Those were picks of the original, small search. A bigger search selects hard for low ratings, so its low ratings
   are *more* likely to be the model's errors than the original's were. Do not turn these counts into a predicted
   completion rate.
2. **A bigger search can exploit model errors.** Two warning signs are measured above.
   - Ensemble disagreement: the mean is below 5 % while one network says above 50 %. This happens on 1-6 of the
     134 picks, depending on the search.
   - Picks pressing against parameter limits: fast routes and pinned speed knots, most of all in the gradient
     searches.
   - The wider family's safe picks look the most robust of all by these signs: all five networks agree on 56, and
     the stage-1 model agrees on 54.
3. **The second opinion is not independent.** The stage-1 Gator model was trained on tiers 0-6, a subset of the same
   drives. It agrees with G_full on most of these routes, but that only shows the finding is not an artefact of one
   training run. It does not show the routes work.
4. **The wider routes are outside the training data.** I measured a random sample of 2,000 of the 15,235 Gator
   training drives (checks/training_route_extent.json):
   - The designed routes never go more than 4 m sideways of the straight route.
   - The planner-proposed routes stay within 10 m (90 % within 8 m).
   - Not one training route goes beyond 10 m.

   47 of the wider family's 71 "safe" routes on the risky pairs do go beyond 10 m. There the model is extrapolating,
   which is exactly where it is least trustworthy. The gain from wider shapes is therefore the least certain part of
   this result, and the most important one to drive.
5. **The route family is still narrow.** Every route is a smooth sideways bend of the straight line, pinned at both
   ends and capped by curvature. On 46 of the 134 pairs even the widest search's best route stays above 50 % risk.
   Genuinely different routes (a detour around the far side of a hill, or an approach from another direction) are
   not in any family tried here.
6. **The timings were taken on a busy shared machine.** They are comparable between searches, not absolute.

## Checks

- **The original search re-runs exactly.** On all 361 pairs, A0 gives the recorded pick: identical route file bytes,
  identical pick records, and a difference in P of 0.
- **Self-tests** (checks/selftest.json):
  - The re-parameterised route family with 3 terms and 10 m reproduces the original functions exactly (random
    stream, route bytes, the re-centring projection) and re-runs A0 exactly through the swapped-in functions. The
    swap is undone afterwards.
  - The refinement-from-given-starts path used for A3b reproduces ci_grad's pick exactly.
- **Determinism** (checks/rerun.json): each search was re-planned on 4 pairs in a fresh process and compared byte for
  byte.
  - All sampling searches, A0-A4 and both ablations, were identical.
  - The gradient searches were identical on 3 of 4 pairs at first. On pair f104_crm_eval_group_0121, P was 6.06e-4
    in the main run against 6.39e-4 on re-run. That pair was planned in the minute the GPU ran out of memory (see
    below). A fresh process with the same order reproduced the re-run value, and 6 more pairs re-ran identically.
    That pair's two gradient routes were re-planned under normal conditions (the old files are kept in
    checks/replaced_0121/). Afterwards all re-checks were identical: 10 of 10 pairs for each gradient search.
- **Scoring is batch-stable.** Scoring every search's pick of a pair together in one batch changes the risk logit
  by at most 0.0035 compared with the search's own scoring.
- **Every written route is checked.** Each route file passes the unchanged route checker, starts within 0.25 m of
  the case pose and ends within 0.25 m of the goal. Its content hash matches its pick record.

## Files (all under artifacts/traverse/search_probe_20260927/)

| path | contents |
|---|---|
| `scripts/sp_search_probe.py` (repo) | the probe. Commands: `groups`, `selftest`, `plan`, `rerun`, `finalize`, `rescore`, `report`. No existing file was edited |
| `groups/` | fail.txt (261), fail_p50.txt (134), control.txt (100), all.txt (361), groups.json (selection rules and the driven outcome of every pair) |
| `A0_cem4x64_repro/`, `A1_cem8x256/`, `A2_cem16x512/`, `A3_grad64x300_fromA0/`, `A3b_grad64x300_fromA2/`, `A4_wide5m20_cem16x512/` | one folder per search, 361 pairs each. Contents: picks/ and routes/ (the planner file formats), tasks.json (the original tasks row format), summary.json, PICKS_LOCKED.sha256, groups.txt, ag_picks.json (manifest with models and sha256, map check, per-pair picks). A2 also has pool/, its best 64 routes per pair, which are A3b's starts |
| `A4a_wide3m20_cem16x512/`, `A4b_wide5m10_cem16x512/` | the two ablations, the 134 risky pairs only, same file set |
| `rescore.json` | per pair: the driven pick and every search's pick scored together by all five networks and by the stage-1 model, plus the terrain along each route |
| `results.json`, `tables.md` | every number above, and the full tables (worst-network deciles, per-set geometry) |
| `checks/` | selftest.json, rerun.json (with the diagnosis note), replaced_0121/, training_route_extent.json |
| `logs/` | planning logs |

**Driving these later (not done).**
- **Path layout.** tasks.json paths are relative to this folder. The case paths point to the same case files as the
  original evaluation (`../generalist_20260921/A_adapt/suite/cases/<pair>.json`).
- **Building the rows.** The folders are in the format the evaluation row builder reads. A local test run (output
  in /tmp, deleted) built 1,083 Gator soil rows from three searches without complaint:

      PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
      $PY scripts/ag_eval_tasks.py build --world crm \
          --arm SPA4_wide_gator="artifacts/traverse/search_probe_20260927/A4_wide5m20_cem16x512"@gator \
          --arm SPA2_free_gator="artifacts/traverse/search_probe_20260927/A2_cem16x512"@gator \
          --out <rows.json>          # then `ag_eval_tasks.py stage --tasks <rows.json>` (dry run first)

- **What to drive.**
  - A0's routes are the ones already driven; pass the original rows with `--existing` so they are reused, not
    re-driven.
  - The most informative drive is A4 on the 134 risky pairs. Add A2 or A4a on the same pairs to separate "more
    search" from "wider shapes", and A4 on the control set to check for harm.

## Incident: GPU memory

My first launch ran 6 planning processes at once. At about 10:24 they plus the other jobs on the GPU filled its
memory. Three of my processes crashed with an out-of-memory error. One Isaac evaluation episode of the twinfactory
experiment that started at that moment (episode 202709316) logged a Warp "CUDA error 2: out of memory" at start-up.
It then ran to the end and reported exit code 0; I did not check whether its results were affected. After that I ran
at most 3 processes (1.1-1.7 GB each) and released cached GPU memory after every search. There were no further
memory errors.
