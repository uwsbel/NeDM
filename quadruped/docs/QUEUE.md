# Queue

Ordered. Top item is next. Move an item to STATE.md when done, with its result.

## Done 2026-09-22 (results in STATE.md)

Replication across surrogates, multi-path evaluation, env scaling, rollout-trained
ensembles, rigid-ground check, and the frozen recipe in nine 1 s rollout-trained
surrogates; 2048 envs (three seeds) and seed 7's 1 s surrogate at 1024 envs.

## Now

- **Re-score the recipe at iteration 1000 in ten arms** (hpcfund 433924: seed 7 at two PPO
  seeds, north s1, euler seeds 2-5, plus the iteration-1000 snapshots of seeds 6, 8, 9 and
  north s0). A watcher submits the paired paths scoring when the fine-tunes finish. Its
  table replaces the nine-surrogate (dw 4.0) table in STATE and on the page.
- **Capacity and data study** (euler 67372): 25% and 50% subsamples of the corpus and a
  smaller and a larger surrogate, two seeds each, then PPO and paths scoring on hpcfund.
  Answers whether a bigger model or less data moves transfer, after doubling the corpus
  did not.
- **Keep the write-up page current** (https://claude.ai/artifact/G8PHCRfk7M8Mu7b8rW2PbU).
  It is maintained, not published once: when a number here changes, republish the page in
  the same move. Source lives in the session scratchpad (`site/`), and the trajectory
  animations are regenerated with `evaluate.py --dump-traj` on sbel.

## Waiting on Kyle

- A Chrono cost profile (SPH vs multibody vs contact) for the cost section.
- Terminology, and the write-up format for Dan.

## Decided

- **No harder-push corpus** (Kyle, 2026-09-24). The current corpus stays; the claim is
  scoped to tracking, with push robustness held at the base policy's level, not improved.

## Done 2026-09-23 (results in STATE.md)

The stopping budget (stop at ~iteration 1000), push robustness evaluation, disturbance
training (negative), guard calibration on induced failures (6 trips in 60 runs), the
doubled corpus (no gain), OOD coverage through iteration 3000, the GPU fault investigation.

## Next

1. **Write it up** as the Study 4 results: the iteration-1000 table (ten arms, CRM
   paths, straight, rigid, pushes), the ablations that justify each piece (branch length, rollout
   training, env count, ensembles, stopping point, corpus size), per-family breakdown, cross-machine agreement, cost.
2. **Sync nodes from git.** Branch is on GitHub; have each node clone and pull instead of
   receiving `git archive` copies, so every run's code is a commit.
3. **Paths shorter than 6 s** for the three fast paths skipped identically in every arm
   (the CRM bed particle cap), so the path set is 40 of 40.

## Open questions, not blocking

- **Why was a3's one-step surrogate the outlier?** (+57%, +80% forward at 0.30 s.) Moot for
  the recipe, which does not use one-step surrogates or 0.30 s branches.
- **Analytic PPO gap.** Analytic fell in 16/16 at dw 2.89; PPO without the OOD penalty
  still transfers. The old pipeline's analytic recipe worked with the policy's own reward
  terms and dw 1.0; ours uses a squared tracking loss and a longer budget. Not pursued.
- **Staged copies on north/a3/sbel/hpcfund** still hold the old top-level diagnostics
  (rsync without --delete); harmless, clean on the next staging.
- **Truncation bias.** v2 truncates 18.3% of episodes, 17.6% on `off_bed` (v1 12.6%), keeping
  94.4% of rows (first 391 episodes). Drift-correlated, as before.
- **The Chrono GPU fault** (illegal memory access in `SphBceManager.cu`): investigated,
  contained, cause narrowed but unproven (STATE). Five occurrences in ~3,600 episodes,
  not force-related (two came in ordinary collection), not deterministic. Optional next steps, in order: backport
  PR #829 so the `calcHashD` error flag is trustworthy, add an index guard in
  `CalcRigidForces_D` that prints the offending marker instead of crashing, and only then
  report upstream -- a bare "illegal access at line 543" is unactionable.
