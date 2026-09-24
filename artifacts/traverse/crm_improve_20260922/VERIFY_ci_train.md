# Verification: route-risk trainer `scripts/ci_train.py`

Written 2026-09-22 by an adversarial verifier. Inputs read: PLAN.md, NOTES_ci_train.md, `scripts/ci_train.py` (751 lines),
`scripts/ga_train.py` (unchanged, md5 `61598dd089816624e558feeff21c793b`), `n2_arch_train.TokenTx`,
`ga_planner.history_from_trajectory`, and the trainer calls in `scripts/ci_planner.py`. Every check below was run on the
local RTX 5090 with `OMP_NUM_THREADS=6`. Peak GPU allocation in my runs was 2.1-3.0 GB. No cluster jobs were
submitted and there were no Chrono runs. Scratch outputs went to `/tmp/vci/` and are not part of the repo.

**Verdict: pass with issues.** I found no correctness bug in the paths the plan will use:

- The trainer reproduces ga_train bit for bit on real data, including when the data is split across two files.
- The history-window and velocity conventions hold on all four real training files.
- The planner's own call path agrees with the trainer's logits.

The issues are latent safety gaps in multi-file loading, a confound in the data-fraction arms (H4), and a few
corrections to the note.

## 1. What I re-ran, and what held

| check | command / method | result |
|---|---|---|
| ga_train reproduction, synthetic | ga_train through the notes' deterministic wrapper vs `ci_train --arch gru --hist-enc gru --ctx geom --deterministic`, syn_a, 2 seeds x 3 epochs, cond none / tag / hist / hist_aux; my own comparator (below) | logits 0.0 for every seed and condition. 1,412 metric values identical (1,436 with the domain head), dev and ensemble blocks included, split hash equal |
| ga_train reproduction, real data | same, on a 12,844-row group subset of `mixed_reanchor.npz` (110 train, 16 val, 8 test groups, both worlds, 9,502 moving rows); 2 seeds x 2 epochs; cond hist_aux and tag | logits 0.0; 1,436 / 1,412 metric values identical; fit rows 8,721 in both |
| multi-file concatenation | the same subset cut into two files (6,422 + 6,422 rows) with `--ds p1 --ds p2`, compared against ga_train on the single file | logits 0.0, metrics identical: the two-file path is exactly the one-file path |
| architecture matrix | the notes' 14 runs (syn_a + syn_b, hist_aux, 2 epochs, `--roundtrip-check`) | all exit 0. Round trip on aligned batches 0.0 everywhere. Other batch shapes at most 1.9e-5 (gru with the transformer encoder), 2.4e-7 for the station transformers |
| expected failures | `--strict-keys` on syn_a + syn_b; the same file given twice; three train groups renamed `f104_pair_group_*` | all refused with the intended messages |
| history convention on the real files | mixed_reanchor, branch_both, short_anchor, anchor_k40_60_80 (all rows) | valid-frame count = min(k, 40) on 100 % of rows. Valid frames always form the newest block of the window. Every k = 0 window is all masked. No NaN in valid frames |
| velocity source | `vel_from_history` vs stored anchor state `ctx[:,0]`, `ctx[:,6]` and `vx_anchor`, all moving rows of the four files (85,820 + 4,614 + 89,998 + 88,797) | identical after float16 rounding (0.0); at most 2.5e-3 m/s from the float32 ctx value. State column 0 is vx and 6 is yaw rate (`ga_domain_probe.STATE12_NAMES`) |
| planner window = training window | `ga_planner.history_from_trajectory`: row t = [state[j] on the 12 columns, action[j-1]], j = k-39+t, valid for 1 <= j <= k | the brief's convention. The newest frame is the decision-frame state, so training and planning take the velocity from the same frame |
| S2 combination on real data | `--ds real_sub --ds short_sub` (10,012 short-anchor rows, k 10/20/30), `--hist-T 20 --ctx geom_vel --cond hist_aux --roundtrip-check`: gru/gru, gru/tx, txjoint, tx96_2/tx | all train. Round trip on aligned batches 0.0; the other checks at most 2.1e-6. Anchor buckets k0 / k10_30 / k40p all populated |
| mask-mode invariance on real rows | those 4 checkpoints, 200 held-out rows with k >= 40 each: the stored window vs the same window with the older 20 frames masked and filled with random numbers | 0.0 for all four: a 3 s-approach window and a 1 s-approach window with the same newest 20 frames give identical inputs |
| planner call path | the CIScorer sequence (`encode_history` once on the raw 40-frame window, `vel_from_history` on the raw window, `score(z=, vel=)` one row at a time) vs the trainer's logits, 200 held-out rows per checkpoint | at most 3.0e-4 to 4.7e-4. The cause is TF32, see issue 5 |
| same-start-state metrics | my own recomputation (episode from the source files by id) on the shipped `real_gru_hist_aux_plus_branch_xhost` run | AUC, pair counts, pick failure and cell counts identical for held-out and dev, both worlds, both labels |
| hand-written transformer layer | `Block` vs `nn.TransformerEncoderLayer(norm_first=True, gelu)` with copied weights and a key mask | 0.0 |
| history transformer masking | masked frames filled with 1e3-scale noise; all-masked window with changed content | read-out unchanged (0.0) in both cases |
| sign conventions | route logit = log of the summed softplus hazard (higher means riskier); AUC counts positive rows scored above negative ones; the pick is the argmin | consistent with ga_train |
| suite blacklist, splits | groups of all four files | no suite group anywhere. Every group has one split across all files. All 865 fit groups hold rows of both worlds (so equal data fractions really are per-domain equal) |

The shipped comparator `compare_ga.py` has a hole: when one side is NaN and the other a number, the difference is
NaN, which is not greater than 0, so the value is counted as equal. My comparator
(`/tmp/vci/strict_compare.py`) counts that case as a difference, also compares the fit, dev and held-out masks, and
still finds 0 differences. The claims in the notes therefore stand.

## 2. Issues

1. **Silent defaults when a per-row key is missing from one file (medium, latent).** A key missing from any one `--ds`
   file is dropped for all files. The trainer then fills in a default, and only prints the list of dropped keys:
   - `anchor_frame` becomes 0: every row counts as a standing start;
   - `source` becomes `'unknown'`: `--row-weight` does nothing (a warning) and the per-source blocks disappear;
   - `episode` disappears: the same-start-state metrics disappear;
   - `profile` becomes -1: the same-speed AUC is lost;
   - `domain` is re-derived from the id suffix.

   Reproduced with the real subset split in two and `anchor_frame` removed from the second file:
   - all 12,072 used rows had anchor frame 0, although 8,930 of them had a visible window;
   - `--startup-only` kept 12,072 rows (8,721 fit) instead of 3,142 (2,264 fit);
   - every metric labelled "startup" and every "k0" block would silently mix in moving rows.

   The four current files all carry these keys, and the planned S3 labelling (`ga_branch_dataset.py` format) writes
   them, so nothing is wrong today. Fix: fill defaults per file before intersecting the key sets, or refuse a file
   that lacks any of `anchor_frame`, `domain`, `source`, `episode`, `profile`.
2. **No check that each group has one split across files (medium, latent).** In a copy of the second half I gave group
   `f104_v2_group_0020` split `val`. It loaded without error or warning as 47 fit rows plus 47 held-out val rows. That
   is leakage between fit and selection rows, against the K1 rule that the twin group split is the only split. All
   current files agree; a one-line assert on (group -> unique split) would close this.
3. **The per-domain data fraction also changes the training length (medium, for H4).** An epoch is n_fit // 256 steps
   and batches stay 128 rigid + 128 soil, so dropping soil data shortens training for both worlds:
   - Synthetic (flags/): soil fraction 0.25 gives 8 steps against 12; rigid 0.25 with soil 0.5 gives 4.
   - Full K1 file: rigid 42,108 + soil 0.25 x 41,432 gives about 205 steps per epoch against 326, i.e. 37 % fewer
     steps. The rigid rows are also seen 37 % less, although their data did not change.

   A soil-fraction curve therefore mixes "less soil data" with "shorter training", and any rigid change in those arms
   is confounded. Suggest an option that fixes the step count to the full-data run (or defines an epoch on the fit set
   before the fractions are applied), or report the curves at equal steps.
4. **Blacklist checks groups only (low, latent).** A row whose id contains `f104_pair_group_0001_route_00` but whose
   group is a training group was accepted (`/tmp/vci/fail/suite_in_id_only.npz`, exit 0). The K1 builders check ids
   and groups. Checking `id` and `episode` against the same patterns would cost nothing.
5. **Scoring in other batch shapes differs by up to about 5e-4 on real data, not 2e-5 (documentation).** The planner
   sequence scored one row at a time differs from the trainer by 3.0e-4 to 4.7e-4. The cause is cuDNN's default TF32
   convolutions (`torch.backends.cudnn.allow_tf32 = True`), which pick different kernels for batch size 1:
   - gru: batch 1 vs 512 differs by 2.1e-4 with TF32, 6.0e-7 without;
   - txjoint: 2.6e-4 with TF32, 4.8e-7 without;
   - batch 256 vs 512: 0.0 for both.

   The planner scores candidates in batches of 256, so this is harmless, and ga_train behaves the same. But the
   note's statement that other batch shapes differ by at most 2e-5 holds only on the synthetic set; on real data it
   is below 1e-3.
6. **The joint transformer's domain head depends on the candidate route (low, comparability).** Its history tokens
   attend to the station tokens, so the domain logit changes with the corridor:
   - with one fixed window, the logit's standard deviation across corridors is 1.60 (3.75 with each row's own
     window);
   - it is still a history read-out: giving every row one rigid or one soil window drops the AUC from 0.981 to
     0.545 / 0.533.

   Its "domain-head AUC" is therefore not the same quantity as the GRU or transformer encoder's AUC, which is a
   function of the window alone. It must not stand in for the W-probe AUC in the predeclared "adaptive" rule
   (>= 0.95). `ci_planner` reports no `p_crm` for txjoint members, which is consistent.
7. **Data mixing limits for S2/S3 (low, usability; the failures are loud).**
   - `anchor_k40_60_80.npz` shares 38,997 ids (its k 40 and k 80 rows) with `mixed_reanchor.npz`, so the trainer
     refuses the pair. Frame-60 rows, the 3 s protocol's decision frame, cannot be added without building a filtered
     file first. There is no per-file anchor-frame filter and no de-duplication option.
   - The short-anchor and k40/60/80 rows carry source `designed` / `on_policy` like the re-anchored rows. So
     `--row-weight` and the `by_source` blocks cannot tell them apart; only `by_anchor` can. A per-file or
     per-anchor-bucket weight would be needed to up-weight short-anchor rows.
   - The default tag does not include the dataset list, lr, wd, epochs, data seed, hist-drop or seed0. Runs that
     differ only in these overwrite each other in the same `--out`. Always pass `--tag` (or a separate `--out`).
8. **Memory for the S2 combination (low, planning).** K1 file + short-anchor file = 110,587 + 85,844 = 196,431 used
   rows, i.e. 7.2 GB even with `--x-half`. Locally only `--x-host` fits under the 6 GB rule; the note's estimate
   (4.1 GB with `--x-half`) covers the K1 file alone. The trainer has not been run on the AMD cluster yet (ROCm,
   `expandable_segments`, boolean-mask attention). One short cluster smoke before the B/A sweep is advisable.
9. **Minor.**
   - The note's "standing-start rows have stored |vx| at most 0.17 m/s" holds on the 10 % smoke set only; on the whole
     K1 file it is 0.335 m/s. The velocity input is still 0 for those rows, as it is in the planner.
   - A batch quota larger than one domain's fit rows silently shrinks the batch: soil fraction 0.02 left 8 soil rows
     per batch instead of 192. This cannot happen at the planned fractions (0.25 of 41k soil rows).
   - `--startup-only --ctx geom_vel` standardises two all-zero columns with sd 1e-6, so such a model would give garbage
     at a moving start (never a planned use).
   - The held-out same-start-state metrics rest on few pairs (53-54 per world in the 10 % smoke). Report `Sn_*` with
     them in S3.

## 3. Do the identity tests compare like with like?

- **Reproduction test: yes.** Same data, seeds and deterministic flag, and the same row order. It covers only gru +
  GRU encoder + geometry context with the 0.5 batch split. Everything new (the transformers, the joint model, the
  history transformer, the velocity input, other batch splits, data fractions, row weights) is checked only against
  itself, plus the hand checks in the notes. That is the expected limit, as there is no external reference. The layer
  equivalence and masking checks above narrow it.
- **Round trip: yes.** `score()` gets the raw float16 corridor, raw geometry and the raw 40-frame window, and has to
  rebuild the standardisation, window rule, velocity and model from the checkpoint alone. It is compared with the
  trainer on identical 512-row batches. The planner's path (precomputed code, window shared by all candidates) is
  linked by the shared-window and precomputed-z checks, and I checked it directly against the trainer's logits
  (issue 5).
- **Velocity check: yes.** Float16 of the stored anchor state against the newest valid frame of the float16 window.

## Commands (scratch, reproducible)

```
cd /home/harry/NeDM-traverse_mppi; export OMP_NUM_THREADS=6 PYTHONPATH=src:scripts; PY=/home/harry/miniconda3/envs/nedm/bin/python
# real subset: 110 train / 16 val / 8 test groups of mixed_reanchor.npz (rng 5), whole file and two contiguous halves -> /tmp/vci/real_sub*.npz
$PY -c "import sys, torch; torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; sys.argv=['ga_train.py','--ds','/tmp/vci/real_sub.npz','--out','/tmp/vci/rrepro/ga','--cond','hist_aux','--seeds','2','--epochs','2','--no-save','--tag','ga_hist_aux']; import ga_train; ga_train.main()"
$PY scripts/ci_train.py --ds /tmp/vci/real_sub_p1.npz --ds /tmp/vci/real_sub_p2.npz --out /tmp/vci/rrepro/ci2 --arch gru --hist-enc gru --ctx geom --cond hist_aux --seeds 2 --epochs 2 --no-save --deterministic --tag ci2_hist_aux
$PY /tmp/vci/strict_compare.py /tmp/vci/rrepro/ga/ga_hist_aux /tmp/vci/rrepro/ci2/ci2_hist_aux
$PY scripts/ci_train.py --ds /tmp/vci/real_sub.npz --ds /tmp/vci/short_sub.npz --out /tmp/vci/s2 --arch txjoint --ctx geom_vel --cond hist_aux --hist-T 20 --seeds 1 --epochs 2 --roundtrip-check --tag s2_txjoint_gru
```
