# Verification of `scripts/ga_train.py` (plan A2 shared risk trainer)

Verifier run 2026-09-21 on luffy (RTX 5090, conda `nedm`, torch 2.12.0+cu130). Scratch outputs in `/tmp/verify_ga_train/`
(not part of the artefact tree). No cluster submissions, no Chrono runs, GPU peak 5.54 GB (all real-data runs used
`--x-half` to stay under the 8 GB cap given to the verifier). No file other than this one was written under the repo.

## Verdict: PASS with minor issues

The module does what the plan and the note say. Every claimed number I re-ran reproduces; the checkpoint contract, split
rules, history layout, timing convention and blacklist are respected; the round trip `load_ga_model` + `score` equals the
trainer's `predict` exactly for all five conditions. The issues below are all minor (none changes a result); the two
worth fixing before the real arms are (1) the sealed test-split logits written into every `_logits.npz` and (2) the
9.5 GB default device footprint.

## What was reproduced

| check | command / method | result |
|---|---|---|
| Synthetic smoke, 5 conditions | `ga_train.py --ds selftest/train/synthetic.npz --cond {none,tag,hist,hist_aux,hist_rma} --epochs 2 --seeds 1 --split-eval val` into `/tmp/verify_ga_train/out_*` | Final losses 4.3244 / 4.4692 / 4.7057 / 5.0543 / 4.2624 = the note's 4.32 / 4.47 / 4.71 / 5.05 / 4.26 to 4 decimals; nctx 5/7/5/5/5; peak GPU 1.41-1.43 GB; hist_aux domain head established AUC 0.4976, startup 0.5 (identical to the stored log) |
| Round trip, 5 conditions | `roundtrip_check.py` on my fresh checkpoints | max \|score - predict\| = 0.00e+00 for all five; contract keys present; startup all-masked window == explicit zero window == precomputed z |
| Determinism | same command twice (hist_aux) | logits differ by max 2.1e-4 (cuDNN GRU/CNN non-determinism); metrics identical to the printed precision. My fresh runs vs the implementer's stored `member_logits`: max 1.6e-4 to 3.5e-4 (2.3e-3 for hist_rma), corr 1.0000 |
| Real data, hist_aux (self-test 2) | same command as the note plus `--x-half` (f32 needs 9.5 GB) | fit 8,147 (4,133 rigid / 4,014 CRM), dev 21,653, val 5,394, 93 steps, split_hash 0738de0c. Held-out val: rigid P_unsafe 0.907 / 0.928 (startup / established), W 0.895 / 0.915, pick_fail 0.000 / 0.000, Brier 0.208 / 0.125, ECE 0.269 / 0.132; CRM P 0.926 / 0.930, W 0.916 / 0.919, pick_fail 0.196 / 0.161, Brier 0.114 / 0.097, ECE 0.116 / 0.042; domain head established AUC 0.8913 acc 0.810 (n 3,990), startup 0.5. All equal to the note's table within 0.001 (one W value 0.895 vs 0.896). Peak GPU 5.52 GB, RSS 4.9 GB, 25 s wall |
| Real data, hist_rma (NOT run by the implementer on real rows) | same flags, `--cond hist_rma --x-half` | Three stages run: teacher 93 steps (loss 40.5 -> 1.66; held-out P_unsafe rigid 0.900 / 0.923, CRM 0.930 / 0.936), encoder fit 93 steps (MSE 0.262 -> 0.139, z R^2 on established held-out rows 0.424), fine-tune 155 steps at lr/4 (1.89 -> 1.62). Student held-out P_unsafe rigid 0.926 / 0.939, CRM 0.930 / 0.934; peak GPU 5.54 GB, 28 s wall. The heavy-tailed CRM slip column (max 75.9) is standardised without clipping and the teacher still trains |
| Checkpoint contract on the real checkpoint | probe of `real_hist_aux/..._s0.pt` + `_logits.npz` against the dataset | all 15 contract keys; cond hist_aux, cin 6, nctx 5, zdim 16, hist_cols = 12 state + [0,1,2], hist_T 40, train_rows 8,147; `split_hash` recomputed from the sorted fit ids = 0738de0c (matches); `hist_mu`/`hist_sd` recomputed over the valid steps of the fit rows agree to 6e-6 / 4e-8 relative; `ctx_mu`/`ctx_sd` agree exactly; `ensemble_logit == mean(member_logits)` |
| Split rules / leakage | same probe | fit rows are split == train only, dev rows train-only (md5 fold), held-out rows val only; fit groups ∩ val groups = 0, fit ∩ test groups = 0, fit ∩ dev groups = 0; `--subsample 0.1` kept 86 of the ~865 non-dev train groups; no `f104_crm_eval_group_*`, `f104_g1_test_group_*`, `f104_pair_group_*` name among the 1,200 groups or the ids |
| Dataset invariants the trainer relies on | `mixed_reanchor.npz` scan | anchor frames are multiples of 40 (k = 0 rows all-masked, 30,048 / 30,048; established rows 40/40 valid); `hist` f16 max abs 275 (engine speed), far below the f16 range, no NaN/inf, masked steps are exactly 0; unsafe == 1 iff event_idx >= 0 (no discordant rows); one split per group (1,089 / 56 / 55) |
| History layout vs the builder | code read | builder: `si = k-(T-1)+t`, `ai = k-T+t`, valid iff `ai >= 0`; checkpoint `hist_layout` string says the same; the planner's T5 check (`state[21..60]` / `action[20..59]` at frame 60) is the same rule. Causal: state at frame j is paired with the action applied during interval j-1, so `action[k]` (computed at the end of interval k-1 and applied from interval k) is not in the window ending at k |
| Geometry source | `f104_n2_dataset.init_map` reads `static_map_v1/observation.npz` (the OptiX static depth map); the mixed file keeps the night-2 reanchor corridors byte-identical | consistent with the plan's one-geometry-source rule; not something the trainer can change |
| Helper functions | CPU probe (`GA_DEVICE=cpu`) | `survival_nll` at haz = 0: event at station 10 gives 11 ln 2, censored gives 96 ln 2 (correct discrete-time survival NLL); `auc` on all-tied scores 0.5, perfect 1.0, inverted 0.0; `calibration` bin clipping at p = 1 works; `score` with a (1,2) one-hot equals (n,2), and the one-hot changes the logits; a shared (1,40,15) window, per-row copies and a precomputed z give identical logits; NaN under the mask does not change z; `--hist-drop 0.2` masks whole windows only (21.7 % of 1,000 rows, no partial windows); balanced batches with 700 / 300 fit rows are 128 + 128 |
| Parent fidelity | diff against `n2_arch_train.py` | network, loss, OneCycle, grad clip 5, AdamW wd 1e-4, dev fold, ctx cols 17-21 (`CTX_COLS['geom']`) unchanged; `auc` tie handling changed from argsort order to average rank (correct; the old one gave 1.0 on all-tied scores); held-out set changed from `split != train` to `split == --split-eval` (plan R1/R12) |

## Issues (all minor; smallest fix given)

1. **Sealed test-split predictions are written into every `_logits.npz`.** With `--split-eval val` the npz still carries
   `member_logits` / `ensemble_logit` for all 5,281 test-split rows (checked on the real self-test file). Nothing reads them
   today, but the plan seals the 55 test groups for the final report and any later analysis can compute test metrics from
   these files without a deliberate step. Fix: in `main()` set the logits of rows with `split not in {train, split_eval}`
   to NaN before `np.savez_compressed` (one line), or save only fit/dev/held-out rows.
2. **Default device footprint 9.5 GB** (f32 corridor tensor for all 115,868 rows, including rows never evaluated). The
   implementer's own real self-test ran at 9.5 GB; under the 8 GB rule given to the verifier the real arms must use
   `--x-half` (5.5 GB measured; metrics identical to 3 decimals and logits within ~1e-5 in the note's check). Fix: make
   `--x-half` the default, or standardise per batch from the f16 array (the raw f16 is 4.3 GB) instead of storing the f32
   standardised copy.
3. **Not bit-reproducible.** Same seed, same command: logits differ by up to 2e-4 (cuDNN). Metrics reproduce to the
   printed precision, so this only matters if anyone diff-tests checkpoints. Fix if wanted: `torch.backends.cudnn.deterministic
   = True` and `torch.use_deterministic_algorithms(True)` in `train_one` (slower GRU backward), and say so in the note.
4. **Silent batch shrink when a domain has fewer than bs/2 fit rows.** `Batches` with 950 rigid / 50 CRM fit rows returns
   178-row batches (the whole small domain every step). Cannot happen on the real arms (4k+ rows per domain at 10 %),
   only with extreme `--subsample`. Fix: assert `min(len(p) for p in parts) >= q` or fall back to proportional draws.
5. **`hist_rma` stage 2 fits the encoder on all-masked rows too.** ~26 % of rows (all startup rows) have an empty window
   but a per-row teacher target `e` (it contains the domain one-hot and the frame-0 privileged means), so a quarter of
   the MSE is unlearnable by construction and the reported stage-2 loss is inflated; `z_r2_established_heldout` correctly
   excludes them. Fix: weight the MSE by `hm.any(1)` (one line) and report the established-only MSE.
6. **The `all` regime pools startup and established anchors of one group|domain into a single pick cell** (probe: the pick
   among rows at k = 0 and k = 40 of the same episode is chosen). The note flags this for `established`; the plan's
   metrics are per {startup, established}, so `all` should be labelled as an aggregate, not a route-choice statistic.
7. **Documentation nits.** `--x-half` argparse help says "predict differs from score at ~1e-3" while the measured and
   documented gap is 1.07e-5. In `--mode deploy` the JSON `dev` block is in-sample (the dev fold is in the fit set) and is
   not labelled as such (inherited from the parent). `Data` decompresses the whole `hist` array just to read `hist_T` for
   the `none`/`tag` conditions (~1 s, harmless).

## Contract deviations reported by the implementer, assessed

- RMA teacher takes the one-hot through `penc(Linear(10 -> 16))` instead of the ctx vector: acceptable and preferable
  (student and teacher share the backbone shape, nctx stays 5, `score()` refuses the teacher file). Verified working on
  real rows (above).
- `hist_aux` domain BCE only on rows with a visible window: correct (an all-masked window can only carry the base rate).
- Extra checkpoint keys: harmless; the planner's `check_contract` ignores them and only warns on `hist_cols` differences.
- Builder stores 12 `hist_cols`, trainer appends the 3 action indices and writes 15: consistent with the plan's
  "12 state + 3 action indices" and with the planner's `HIST_COLS`.

## Not verified here

- No 30-epoch, 5-seed arm has been trained (as the implementer states); the numbers above are 93-step plumbing runs.
- `ga_planner.py`'s own network copy was cross-checked by the planner module's T8 test, not re-run here.
- Rigid reanchor corridors: the trainer inherits whatever `ga_build_mixed.py` copied from the night-2 files; the
  `init_map` code path reads the OptiX static map, but which `--root` was passed when `reanchor_rigid.npz` was built is
  not recorded in the night-2 log (a ga_data question, not a ga_train one).
