# Queue

Ordered. Top item is next. Move an item to STATE.md when done, with its result.

## Now (running overnight 2026-09-22; results land through the morning)

1. **Replicate the 2 s recipe across surrogates.** The -50%/-53% forward result is one
   surrogate x two seeds. hpcfund 430862 multi-step fine-tunes 8 surrogates (50-step on
   h6, h8, h9, v2n, v2n_s1; a second seed of h7; 100-step on h7 and v2n_s1); a launcher
   then runs 2 s PPO in each and the multi-step ensemble (0.3 s and 2 s, with and without
   the disagreement penalty). euler 66341/66343/66346: 50-step fine-tunes of e2-e5.
2. **Multi-path evaluation.** hpcfund 430923: 8 policies x 40 held-out paths (4 per command
   family, 15 s). Does the straight-line gain hold on turning, sideways and speed-change
   paths, and does 2 s still beat 0.3 s there?
3. **Env scaling.** The 2 s recipe at 64, 256, 512, 1024 envs (hpcfund 430927) and 2048
   (sbel seed 0, north seed 1), same weight budget; each scored straight and on 40 paths.
   Massive parallel rollouts are what the surrogate offers that Chrono cannot.

## Next

4. **Freeze the recipe** from 1-3 and evaluate its policies on rigid ground over the paths
   (one check so far: forward tracking slightly worse on rigid, yaw better).
5. **Write it up** as the Study 4 results table: straight and per-family paired changes,
   seeds and surrogates as replicates, cross-machine agreement, cost.
6. **Sync nodes from git.** Branch is on GitHub; have each node clone and pull instead of
   receiving `git archive` copies, so every run's code is a commit.

## Open questions, not blocking

- **Why is a3's surrogate the outlier?** Forward tracking worse in both seeds (+57%, +80%)
  where seven other one-step surrogates improve it. Not the selected epoch. The multi-step
  recipe and the ensemble may make the question moot; check a3's surrogate after multi-step
  fine-tuning.
- **Analytic PPO gap.** Analytic fell in 16/16 at dw 2.89; PPO without the OOD penalty
  still transfers. The old pipeline's analytic recipe worked with the policy's own reward
  terms and dw 1.0; ours uses a squared tracking loss and dw 4.0. Not pursued.
- **Merged manifests** copy shard 0's `family_balance` and `long_episodes` instead of
  aggregating (data unaffected). Fix in merge_corpus.py.
- **Staged copies on north/a3/sbel/hpcfund** still hold the old top-level diagnostics
  (rsync without --delete); harmless, clean on the next staging.
- **Truncation bias.** v2 truncates 19.4% of episodes on `off_bed` (v1 12.6%), keeping 94.4%
  of rows. Drift-correlated, as before.
- **The Chrono GPU fault** (illegal memory access in `SphBceManager.cu`, one in ~1,950
  episodes). Report if it recurs.
