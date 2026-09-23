# Queue

Ordered. Top item is next. Move an item to STATE.md when done, with its result.

## Done 2026-09-22 (results in STATE.md)

Replication across surrogates, multi-path evaluation, env scaling, rollout-trained
ensembles, rigid-ground check, and the frozen recipe in nine 1 s rollout-trained
surrogates; 2048 envs (three seeds) and seed 7's 1 s surrogate at 1024 envs.

## Now

- **Re-score the recipe at the new stop.** The curve says iteration 1000, not dw 4.0:
  twice the tracking gain, rigid ground better, robustness still at the base policy's
  level (STATE). Every number in the frozen-recipe table is at the old stop, so the table
  has to be rebuilt in the nine surrogates before the write-up quotes it.
- **Does the disturbance hold robustness?** hpcfund 432532 fine-tunes with a robot_lab
  style kick in 15% and 50% of branches. If it does, the recipe gains one flag and the
  stop can go later; if not, the stop stands at 1000 on the robustness limit.
- **Calibrate the guard on induced failures** (hpcfund 432599, OOD penalty removed) and
  replace the spike statistic with the fraction of branch-steps outside, which does not
  depend on batch size.

- **Keep the write-up page current** (https://claude.ai/artifact/G8PHCRfk7M8Mu7b8rW2PbU).
  It is maintained, not published once: when a number here changes, republish the page in
  the same move. Source lives in the session scratchpad (`site/`), and the trajectory
  animations are regenerated with `evaluate.py --dump-traj` on sbel.

## Next

1. **Write it up** as the Study 4 results: the frozen-recipe table (nine surrogates, CRM
   paths, straight, rigid), the ablations that justify each piece (branch length, rollout
   training, env count, ensembles), per-family breakdown, cross-machine agreement, cost.
2. **Sync nodes from git.** Branch is on GitHub; have each node clone and pull instead of
   receiving `git archive` copies, so every run's code is a commit.
3. **Paths shorter than 6 s** for the three fast paths skipped identically in every arm
   (the CRM bed particle cap), so the path set is 40 of 40.

## Open questions, not blocking

- **Why was a3's one-step surrogate the outlier?** (+57%, +80% forward at 0.30 s.) Moot for
  the recipe, which does not use one-step surrogates or 0.30 s branches.
- **Analytic PPO gap.** Analytic fell in 16/16 at dw 2.89; PPO without the OOD penalty
  still transfers. The old pipeline's analytic recipe worked with the policy's own reward
  terms and dw 1.0; ours uses a squared tracking loss and dw 4.0. Not pursued.
- **Staged copies on north/a3/sbel/hpcfund** still hold the old top-level diagnostics
  (rsync without --delete); harmless, clean on the next staging.
- **Truncation bias.** v2 truncates 18.3% of episodes, 17.6% on `off_bed` (v1 12.6%), keeping
  94.4% of rows (first 391 episodes). Drift-correlated, as before.
- **The Chrono GPU fault** (illegal memory access in `SphBceManager.cu`): investigated,
  contained, cause narrowed but unproven (STATE). Three occurrences in ~2,400 episodes,
  none below 300 N pushes, not deterministic. Optional next steps, in order: backport
  PR #829 so the `calcHashD` error flag is trustworthy, add an index guard in
  `CalcRigidForces_D` that prints the offending marker instead of crashing, and only then
  report upstream -- a bare "illegal access at line 543" is unactionable.
