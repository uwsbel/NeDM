# Verification: `scripts/ci_grad.py` (gradient route refinement, arm G)

VERDICT: pass. Adversarial review 2026-09-22 13:05-13:45 by a second agent. I read `NOTES_ci_grad.md`, the 919 lines
of `scripts/ci_grad.py` and every module it imports, re-ran the self-test outputs' checks independently, and tried to
break the six things the task named: gradient correctness, history context, joint-transformer handling, route validity
under the collector's own reader, the abstain rule, determinism, and the "only 10 suite groups" rule. Every claim in
the module note and in the builder's report that I could test held. No correctness defect found. The points below are
small improvements and caveats, none of them a reason to hold the 800-group run.

Everything I ran wrote only to `/tmp/vgrad/` and `/tmp/vgrad/pd/`; the only file I added to the artefact tree is this
one. No cluster job, no Chrono, GPU peak 0.51 GB.

## What I re-ran, with numbers

**1. Gradient, including the part the self-test does not cover.** The module's own T1 differentiates only the
ensemble-mean logit. I differentiated the FULL objective the optimiser actually minimises (mean logit + curvature
penalty + arena penalty), in float64, at four kinds of point per route, 6 routes x 2 groups
(`/tmp/vgrad/fd.py`, `/tmp/vgrad/fd.json`):

| point | max curvature | penalty active | cosine(autograd, central difference) | worst relative error of the gradient vector |
|---|---|---|---|---|
| start parameters | 0.114-0.123 | 0-1 of 6 rows | 0.948-0.973 | 0.26-0.47 |
| generic nearby point | 0.108-0.122 | 0-1 of 6 | 0.999997-1.0 | 1.3e-4 / 3.5e-3 |
| amplitudes at 0.95 of the cap | 0.77-1.20 | 4-6 of 6 | 0.977-1.0 | 2.4e-1 / 2.4e-7 |
| amplitudes at the cap | 1.02-1.23 | 4-6 of 6 | 0.996-1.0 | 8.9e-2 / 1.8e-6 |
| speed knots at the +4 m/s clip | 0.108-0.122 | 0-1 of 6 | 0.971-0.978 | 0.23-0.28 |

The low cosines are the note's claim that the family has kinks. I checked that directly instead of taking it on trust
(`/tmp/vgrad/kink.py`): per component I compared autograd against the forward and the backward difference separately.

- components where the two one-sided slopes agree (a smooth point): autograd matches the central difference to
  **3e-5 to 8e-4 relative**, at all four kinds of point;
- components where the one-sided slopes disagree (a real kink: the 0.5/6 m/s speed clamp, the terminal cone, the
  two acceleration passes, the +-10 m lateral clip, the penalty's relu): autograd equals **one of the two one-sided
  slopes to 1.5e-4 / 8.6e-3 / 5.3e-4 / 4.9e-3 relative** in the four cases.

So the chain returns a correct (sub)gradient everywhere I could push it, including with the curvature penalty active
at and beyond the cap, which is where an Adam step is most likely to go wrong. float32 gradients (what the run uses)
agree with float64 to cosine 0.984-1.0.

**2. The float64 re-shape at the OPTIMISED parameters** (the self-test only checks it at the start parameters, where
the answer is trivially exact). After 60 real Adam steps on 17 rows (`/tmp/vgrad/batch.py`): waypoints 5.3e-8 m,
speeds 1.0e-6, stations 9.4e-8 versus the torch route at the same parameters; the corridor built by the deployed numpy
builder versus the torch chain 1.5e-7 (elevation), 5.6e-8 (grade), 4.6e-8 (cross), 1e-6 (speed channel), route length
1.9e-6. The deployed scorer's mean logit on those finals versus the torch mean logit at the same parameters: median
8.1e-4, max 1.0e-2 - the documented float16-corridor + TF32 gap, 30x below the 0.3 abstain margin. So the route that
gets written and re-scored is the route that was optimised.

**3. The 17 rows really are independent inside the one batch.** Scoring each kept row alone (batch of 1) versus in the
batch of 17: float64 max difference **1.8e-12**, float32 4.0e-3 (the same cuDNN/TF32 batch-shape noise the note
measures as 5.7e-3-1.1e-2). No cross-row leakage through the shared map, the e0 fill or the min-plus passes.

**4. History context: computed once, held fixed, identical to the deployed one** (`/tmp/vgrad/hist.py`,
`/tmp/vgrad/hist_h.py`, counters wrapped around `ci_train.encode_history`, `ci_train.CIModel.encode`,
`ci_train.TxJoint.embed_hist` and `ga_train.GAModel.encode`). One whole group (CEM + 10 gradient steps + the deployed
re-score):

- joint transformer ensemble: `encode_history` 5, `CIModel.encode` 5, `TxJoint.embed_hist` 5 - exactly once per
  member; the history tokens stored in the differentiable copy are bit-equal to the scorer's (max |difference| 0,
  mask equal) and carry no gradient;
- short-window GRU ensemble: 5 / 5 / 0, z bit-equal (0);
- the K1 GRU ensemble (ga_train members): `encode_calls` 1 per member, z bit-equal (0); this is also the case the
  module asserts on in `summary.json` (50 = 5 members x 10 groups in the self-test).

All member parameters have `requires_grad False`, the deployed models stay in eval mode, and the copies put only the
GRUs in train mode (their dropout modules stay in eval, and both GRUs are single-layer, so eval and train arithmetic
are the same; the independent evidence is T1's 5.5e-6 agreement with the deployed scorer, which runs them in eval).

**5. Joint transformer.** Handled through the same path the deployed scorer uses: the context is the pair (tokens,
mask), expanded over the batch, passed as `z=` so `CIModel.forward` skips `embed_hist` (confirmed by the counter
above). The velocity and tag contexts are `None` for these members because their checkpoints are geometry-context,
history-conditioned (`cond hist_aux`, `ctx_mode geom`) - the code only fills them for `geom_vel` / `cond tag`
members, which matches `ci_train.score`.

**6. Validity under the collector's own reader.** I extracted `read_route` from
`scripts/traverse_fdm_rgbd_diverse_chrono.py` and ran it over **every one of the 127 route files** the self-test
wrote (all three ensembles, all runs): **127/127 pass**, no exception. Measured over those files: station array versus
the geometry recomputed from the waypoints **max difference 0.0** (the reader's tolerance is 1e-4), minimum segment
length 0.49 m (the reader rejects <= 0), minimum 62 waypoints (>= 3 needed), speeds 0.0-6.0 (allowed range 0-10).
Across all 99 pick files: end distance to the goal <= 3.6e-15 m, start distance to the recorded pose 0.0 m, curvature
<= 0.1227 (limit 0.125 + 1e-6), acceleration within [-2, 1.5] to 1e-14, every route file hashes to the hash in its
pick, and the `__B.json` file exists exactly when G differs from B.

**7. Abstain rule.** Re-derived from the pick files, 99 picks: `abstained == (no valid final or gain < 0.3)` in
**99/99**, abstain always emits the B route and records it as the same route, and `J_G <= J_B` in 99/99. The
never-exercised branch is "no valid final at all" (all 30 group-ensemble runs had 17/17 valid finals, and even with
`--lr 5,5` the parameter box keeps all 17 valid), so I forced it by patching the contract to always fail
(`/tmp/vgrad/invalid.py`): the module abstains cleanly, `gain: null`, `n_valid_finals: 0`, emits the B route, no
crash.

**8. Determinism, harder than T6.** T6 replans the same first 3 groups in the same order. I replanned groups #6 and
#9 of the 10 **alone**, twice, with the joint transformer (the attention path): the G route hash, the theta, the gain
and z_mean to 9 decimals are identical to the 10-group run and to each other. Global cuDNN/TF32 flags are unchanged
after a full group (checked before/after in process), which is why arm B stays bit-identical to the deployed planner.

**9. Arm B is the deployed pick.** `--ref-b-picks` matched 10/10 with equal `z_mean` for all three ensembles in the
self-test; I also confirmed the reference set `s2/L1/picks_crm_X` is the real 800-group `free`-family arm-B set for
the same models and decision states (its `summary.json`: family `free`, 800 groups, same model glob, same map root).
Running the report's full command verbatim with `--limit 1` (which selects group `f104_crm_eval_group_0000`, one of
the 10) reproduces ref B 1/1.

**10. Downstream.** I copied the self-test output to `/tmp/vgrad/pd/picks_crm_HG` and ran
`scripts/ga_a5_pass2_tasks.py --world crm --arms HG --branch-frame 20 ...` unchanged: 10 rows, 10 branch-route files,
correct `--mode branch --branch-frame 20` extras. The task rows the module writes have the same fields and the same
relative-path convention as the reference `ci_planner` set (I diffed row 0 of both). Abstained groups get
`run: false` in `tasks_new_only.json`, and the downstream builder independently de-duplicates by (group, sha256), so
an abstained G is driven once when both arms are built together.

**11. Cost and the 800-group extrapolation.** Self-test summaries: joint transformer 3.36 s median / 3.49 mean /
4.63 max per group (first group 4.6 s is warm-up; mean excluding it 3.36), GRU ensembles 2.59 and 2.64 s median; GPU
reserved peak 0.48-0.54 GB. My own runs reproduce 3.34-3.36 s per warm group. 800 x 3.36 s = **44.8 min** for the
joint transformer, **34.5-35.2 min** for a GRU ensemble, as claimed. I checked the one way this could be optimistic:
the 10 self-test groups are slightly shorter than the suite (base route 80.5 waypoints mean versus 87.6 over 60
randomly drawn other groups; suite maximum 167). Measured scaling on synthetic long base routes (no extra suite group
planned): 60 Adam steps on 17 rows cost 2.88 s at 80 waypoints, 2.97 s at 121, 2.99 s at 151, 2.96 s at 153 - the
cost is dominated by the 96-station networks, not the waypoint count, so the estimate stands. Disk: about 50 KB per
group, so about 40 MB for 800 groups.

**12. Only the 10 groups.** The self-test groups are exactly every 80th of the 800 sorted pose keys. Every pick and
route file in `ci_grad_selftest/` belongs to those 10. Every leftover development output on the machine
(`/tmp/cigrad_*`, `/tmp/cig_*`, `/tmp/pd/picks_crm_HG`) contains only groups from that same set of 10 - the union over
all of them is exactly 10 groups, none outside. Nothing under `s2/`, `deploy_v1/` or the suite case directory was
modified (mtimes predate the module's work), `git status` shows `scripts/ci_grad.py` as the only new file under
`scripts/` and no tracked script modified by this effort, and the module writes only under its `--out` directory.
It contains no `sbatch`/`ssh`; its only subprocess call is re-invoking itself for the self-test.

## Problems and suggested fixes (none blocking)

1. **The extra 16 starts are ranked by the ensemble mean even when the keep criterion is the worst member.**
   `plan_group` sorts the round-0 pool by `res['Z_mean']` regardless of `--keep`. Under the default
   (`pessimistic`) a candidate that is excellent on the worst member but mediocre on the mean can be left out of the
   starts. Fix (one line): sort by `res['Z_pess']` when `opts.keep == 'pessimistic'`. Low impact (all 64 round-0
   candidates are within a few logits of each other), but it makes the arm internally consistent.

2. **`route_contract` relies on the validator to reject a folded route; it does not itself check what the collector
   checks.** The collector's `read_route` requires strictly increasing stations, a stations array that matches the
   geometry to 1e-4, and >= 3 waypoints. `route_contract` checks the validator, the start/goal distances, finiteness
   and the speed range; duplicate waypoints are caught only indirectly (the validator raises and `safe_validate`
   turns it into `valid: False`). Empirically all 127 emitted files pass the reader today. Fix: add the three reader
   conditions explicitly to `route_contract` (recompute the station array from the waypoints and compare, atol 1e-4),
   so a future family change cannot slip an unreadable route past the module.

3. **Early stopping never fires, because it needs every row to stall.** `since >= patience` is required for all 17
   rows, so every run pays the full 60 steps (steps_run 60 in 30/30 runs; 42-66 of 170 rows were still improving at
   step 60, so the budget binds as much as the patience does). Two possible fixes: freeze stalled rows (drop them
   from the batch) to cut cost, or raise `--steps` for the rows that are still descending. As it stands, treat 60
   steps as a fixed budget, not as convergence - worth one line in the note, since "best_step == 60" in a third of
   the rows means the reported gains are a lower bound.

4. **`tier` is the index within the planned subset.** If the 800 groups are ever planned in chunks with `--groups`,
   tiers restart at 0 per chunk and no longer line up with the reference arm-B set. Fix: either document "run the 800
   in one invocation" (the report's command does) or derive `tier` from the position in the full sorted case list.

5. **Caveat for the closed loop, measured, not a code defect.** On the changed picks (10 / 8 / 2 groups), the
   refinement raises the mean commanded speed (3.21 -> 3.36, 3.58 -> 3.96, 3.39 -> 4.26 m/s) and the start heading
   error (25.9 -> 30.2, 20.8 -> 26.7, 31.7 -> 41.9 deg), while the branch speed step mostly falls at the median but
   rises in 3/10, 3/8 and 2/2 picks (H maximum 3.03 -> 3.81 m/s). The study's own diagnosis says a branch speed step
   above 1.5 m/s fails 37 % of the time and that the risk model does not capture the handover (E2). So the arm is
   free to move exactly along the axis the model is known to be blind to. Sample sizes are tiny; the point is that
   the first 800-group drive should report the speed-step and heading distributions of G next to B, not only the
   logit gain.

6. **Two presentational nits.** (a) The note's "4.8e-6" for the torch networks versus the deployed scorer is the
   figure for one ensemble; the maximum over all three in `RESULTS.json` is 5.5e-6. (b) `--frame 60` is the 3 s
   protocol's default and is dead for this poses file (every entry carries `frame: 20`, and the pick files record
   `source.frame: null` with the v0 cross-check at 0.0). With a poses file that lacks per-entry frames it would
   silently cross-check v0 against the wrong pass-1 frame; worth an assertion or a note.

## Commands used for this verification

    PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
    cd /home/harry/NeDM-traverse_mppi
    $PY /tmp/vgrad/fd.py --ens X --groups 2 --rows 6      # full-objective finite differences, 4 kinds of point
    $PY /tmp/vgrad/kink.py --ens X                        # one-sided vs central differences at the kinks
    $PY /tmp/vgrad/batch.py --ens X                       # row independence + float64 re-shape at the optimum
    $PY /tmp/vgrad/hist.py --ens X|Hn ; $PY /tmp/vgrad/hist_h.py     # history-encode counters
    $PY /tmp/vgrad/invalid.py                             # forced "no valid final" path
    $PY /tmp/vgrad/scaling.py                             # cost vs base-route length (synthetic routes)
    $PY /tmp/vgrad/flags.py                               # global torch flags / eval mode untouched
    # collector reader over every emitted route, abstain/hash/contract audit over every pick: inline python, see above
    # determinism: scripts/ci_grad.py ... --groups f104_pair_group_0200,f104_pair_group_0440 --out /tmp/vgrad/solo_X_{a,b}
    # rigid smoke test and the report's command with --limit 1: /tmp/vgrad/rigid1, /tmp/vgrad/full_cmd
