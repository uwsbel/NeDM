# Queue

Ordered. Top item is next. Move an item to STATE.md when done, with its result.

## Done 2026-09-22 (results in STATE.md)

Replication across surrogates, multi-path evaluation, env scaling, rollout-trained
ensembles, rigid-ground check, and the frozen recipe in nine 1 s rollout-trained
surrogates; 2048 envs (three seeds) and seed 7's 1 s surrogate at 1024 envs.

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
- **The Chrono GPU fault** (illegal memory access in `SphBceManager.cu`, one in ~1,950
  episodes). Report if it recurs.
