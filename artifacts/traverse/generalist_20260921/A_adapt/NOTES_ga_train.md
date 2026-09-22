# Module note: shared rigid/CRM risk trainer (`scripts/ga_train.py`, plan A2)

Written 2026-09-21. One file: `/home/harry/NeDM-traverse_mppi/scripts/ga_train.py` (512 lines). Self-test artefacts under
`/home/harry/NeDM-traverse_mppi/artifacts/traverse/generalist_20260921/A_adapt/selftest/train/`.

## What it is

A fork of `scripts/n2_arch_train.py` that keeps only the deployed CNN-GRU route-risk network, the discrete-time survival
loss on `event_idx`, the ranking metrics and the one-JSON-row-per-seed output, drops the energy/time heads and the other
architectures, and adds domain-tag and observation-history conditioning for the mixed rigid/CRM dataset
(`A_adapt/datasets/mixed_reanchor.npz`, 115,868 rows).

Model (`GAModel`): corridor CNN -> per-station features (96 x 96) + a 32-d context embedding broadcast to every station +
station position -> Conv1d(k=5) -> bidirectional GRU (width 64) -> one hazard logit per station. Route logit =
log sum softplus(hazard) (`gen_riskmodel.route_logit`), P(unsafe) = 1 - exp(-exp(route logit)).

Conditions (`--cond`):

| cond | context input to the 32-d embedding | extra loss |
|---|---|---|
| `none` | 5 geometry columns (ctx cols 17-21: goal dx, dy, distance, start yaw, route length), standardised on fit rows | - |
| `tag` | geometry 5 + one-hot domain (2) -> nctx 7 (oracle reference) | - |
| `hist` | geometry 5 + z (16): causal GRU(16 -> 32) over the 2 s history (40 frames x 15 channels standardised per channel on the valid steps of fit rows; masked steps zeroed; the validity mask appended as the 16th input channel), final hidden -> Linear -> tanh | - |
| `hist_aux` | as `hist` | + 0.5 x BCE(domain) on a Linear(16 -> 1) head over z, on rows whose window is at least partly visible |
| `hist_rma` | stage 1: teacher = geometry 5 + e (16) = tanh(Linear(10 -> 16)([privileged 8 standardised, one-hot domain 2])) trained with the risk loss; stage 2: student with the teacher's backbone copied and frozen, the history encoder fitted so z reproduces e (MSE, no history drop); stage 3: everything unfrozen, risk loss, 5 epochs at lr/4 with history drop | stage losses recorded in the JSON row (`stages`) |

The 15 history channels are state columns [0-6, 11-15] (vx, vy, roll, pitch, roll/pitch/yaw rate, four spindle speeds,
engine speed) followed by the three applied actions (steer, throttle, brake). The dataset stores `hist_cols` as the 12
state columns; the trainer appends the 3 action indices and writes the 15-entry list into the checkpoint. Startup rows
(`anchor_frame == 0`) have an all-masked window, so every startup row shares one z.

Training: AdamW (lr 2e-3, wd 1e-4), OneCycle, batch 256, grad clip 5, 30 epochs by default; batches are balanced
(128 rigid + 128 CRM from two independent per-domain permutations, each re-drawn when exhausted; the last < 128 rows
of each permutation are skipped per cycle, 0.3 % of a 42k-row domain) whenever both domains are in the fit rows.
`--hist-drop 0.2` masks the whole window of 20 % of the training rows so a masked history at a moving anchor is
in-distribution. Rows: train = `split == 'train'` (minus the md5 dev fold of train groups in `--mode holdout`; all train
rows in `--mode deploy`); evaluated held-out rows = `--split-eval val|test` only; `--startup-only` keeps
`anchor_frame == 0` rows everywhere (the twin-equivalent arms); `--domain-filter crm|rigid` drops the other domain
before anything is put on the device; `--subsample F` keeps a fraction F of the training groups.

Metrics (JSON `members[i].heldout` and `.dev`, and `ensemble.heldout`): per domain (`rigid`, `crm`, and `both` pooled)
x regime (`startup`, `established`, `all`): pooled AUC `P_*`, within-cell AUC `W_*` with cells keyed `group|domain`,
same-speed cells `G_*` (`group|domain|profile`, designed routes only), lowest-risk pick failure per cell (`pick_fail`
vs `random_fail` / `oracle_fail`, and `pick_fail_unsafe`), Brier and 10-bin equal-width ECE of P(unsafe) against
`unsafe` (`brier_unsafe`, `ece_unsafe`; the fitted event) and against `fail` (secondary), `unsafe_not_fail_rate`,
event rates and mean predicted probability. `hist_aux` rows also carry `domain_head` (AUC / accuracy of the domain
logit; startup is 0.5 by definition and is not a finding). AUC ranks ties by their average rank (the night-2 helper
ranked ties by array order, which gave a fake 1.0 for the all-tied startup domain logits).

## Checkpoint contract and the importable API

`{out}/{tag}_s{seed}.pt` = dict with `model_kind='ga_train'`, `cond`, `state`, `cin` (6), `nctx` (5, or 7 for tag),
`zdim` (16 for the history conditions, 0 otherwise), `hist_cols` (15 ints), `hist_T` (40), `norm` (corridor mu/sd for
channels 0-3, `cont_index`, channel names), `ctx_mu`/`ctx_sd` (5), `hist_mu`/`hist_sd` (15), `train_rows`, `split_hash`
(md5 of the sorted fit-row ids). Extras: `width`, `hist_cols_state`, `hist_cols_action`, `hist_layout`, `geom_cols`,
`split_eval`, `mode`, `domain_filter`, `startup_only`, `hist_drop`, `seed`, `tag`, `ds`, `args`. The `hist_rma` run also
writes `{tag}_s{seed}_teacher.pt` (`cond='rma_teacher'`, plus `priv_mu`/`priv_sd`); `score()` refuses it because it
needs privileged inputs. State-dict keys: `front.cnn.*`, `lat.*`, `ctx.0.*`, `tconv.0.*`, `mix.*`, `head.*`
[+ `henc.*`, `hz.*` for history models, `dom.*` for hist_aux].

```python
from ga_train import load_ga_model, score, encode_history
model, ck = load_ga_model(path, 'cuda')                       # eval mode
z = encode_history(model, ck, hist, hmask)                    # (1,16) from one raw (1,40,15) window; hist=None -> startup (all-masked)
logits = score(model, ck, X, geom5, hist, hmask, domain_onehot=None, z=None)   # X raw (n,5,96,32) f32, geom5 raw (n,5)
```
`score` standardises everything from the checkpoint; `hist`/`hmask` (or `z`) with a leading dimension of 1 are shared by
all n candidates (the planner computes z once per decision); `domain_onehot` (n or 1, 2) is required for `cond='tag'`.
Forward contract: `model(X (B,6,96,32) standardised, ctx (B,nctx), hist (B,T,15) standardised, hmask (B,T)) ->
{'haz': (B,96) [, 'z', 'dom']}`; `model(..., z=z)` skips the encoder.

## How to run

```
cd /home/harry/NeDM-traverse_mppi
PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python scripts/ga_train.py \
  --ds artifacts/traverse/generalist_20260921/A_adapt/datasets/mixed_reanchor.npz --out <dir> \
  --cond hist_aux --domain-filter both --split-eval val --mode holdout --seeds 5 --epochs 30 [--startup-only] [--subsample F] [--x-half] [--no-save]
```
Outputs: `{tag}.json` (args, per-seed rows, ensemble metrics), `{tag}_logits.npz` (ids, ensemble and member logits, fit /
dev / held-out masks, domain, anchor_frame, group, labels), `{tag}_s{seed}.pt`. Default tag =
`ga_{cond}_{domain_filter}[_startup]_{mode}_{split_eval}[_subF]`. Device via `GA_DEVICE` (default cuda).

## What was tested (all on luffy, RTX 5090 shared with a 1.2 GB process)

1. Synthetic smoke, `selftest/train/make_synthetic.py synthetic.npz 2000 0` (noise flavour: 2,000 rows, both domains,
   random corridor / ctx / history / labels, planted signal: rows with visible mean vx > 0 and domain 1 fail more; fail
   0.773, unsafe 0.794, 41 % startup rows, fail 0.67 rigid / 0.88 CRM, 0.93 with vx > 0 vs 0.71 otherwise). Every
   `--cond` for 2 epochs, 1 seed (`log_{cond}.txt`, `out_{cond}/`): no errors, finite losses (4.32 / 4.47 / 4.71 /
   5.05 / 4.26 for none / tag / hist / hist_aux / hist_rma; hist_rma stage losses 4.69 teacher, 0.202 encoder MSE,
   4.26 fine-tune), logits (1, 2000) finite, 1.9-2.6 s per seed, peak GPU 1.4 GB. `roundtrip_check.py` on each
   checkpoint: all contract keys present, nctx 5/7, `load_ga_model` + `score` on the raw arrays equals the trainer's
   `predict()` logits with max |difference| = 0.00e+00 (all five conditions); for history models the all-masked
   startup window scored through `hist=None`, an explicit zero window and a precomputed `z` agree to < 1e-5.
2. Learning check, structured flavour (`synthetic6k.npz`, 6,000 rows; corridor = smooth bump whose amplitude raises the
   fail logit, history vx offset and a CRM-only spindle slip so the window identifies the domain), 40 epochs, 1 seed
   (`learn_log_{cond}.txt`): the corridor-only model ranks the bump at startup (within-cell AUC 0.76 rigid / 0.89 CRM);
   the hist_aux domain head reaches AUC 0.98 on established held-out rows and 0.50 at startup (all-masked window, as it
   must); hist_rma stage 2 R^2 of z against the teacher embedding 0.11 (7 of the 8 synthetic privileged columns are
   noise the history cannot reproduce). The held-out risk AUCs on this set are noisy (0.42-0.70) because the tiny random
   corridors are memorised (final loss 0.08); it is a plumbing check, not a benchmark.
3. Flags (`selftest/train/flags/`): `--startup-only` (810 rows kept, no established cells), `--domain-filter crm` /
   `rigid` (single-domain fit and held-out sets, no balancing), `--split-eval test` (held-out 210 test rows instead of
   202 val), `--mode deploy` (fit 1,588 = train incl. the dev fold, different split_hash), `--subsample 0.5` (fit 625),
   `--seeds 2` (two member rows + ensemble), `hist_rma` with `--domain-filter crm --startup-only` (three stages run),
   `--x-half` (score vs predict gap 1.07e-5, i.e. the float16 device copy costs ~1e-5 in the logit). Balanced batches
   asserted directly: with 700 rigid / 300 CRM fit rows every one of 20 batches is exactly 128 + 128.
4. Real data (self-test 2): `--cond hist_aux --subsample 0.1 --epochs 3 --seeds 1 --split-eval val` on
   `mixed_reanchor.npz` (`selftest/train/real_hist_aux/`): rows 115,868, fit 8,147 (4,133 rigid / 4,014 CRM, 10 % of
   the train groups), dev 21,653, held-out val 5,394; 93 steps in 3.8 s of training, 38 s wall including loading,
   peak GPU 9.5 GB (float32 corridor tensor 8.5 GB), peak RSS 4.9 GB. Held-out val, seed 0:

   | domain | regime | n | P_unsafe | W_unsafe | G_unsafe | P_fail | pick_fail (random / oracle) | Brier_u | ECE_u | Brier_f | ECE_f | unsafe-not-fail |
   |---|---|---|---|---|---|---|---|---|---|---|---|---|
   | rigid | startup | 702 | 0.907 | 0.896 | 0.582 | 0.861 | 0.000 (0.174 / 0.000) | 0.208 | 0.269 | 0.333 | 0.440 | 0.171 |
   | rigid | established | 2,027 | 0.928 | 0.915 | 0.729 | 0.872 | 0.000 (0.176 / 0.000) | 0.125 | 0.132 | 0.224 | 0.302 | 0.170 |
   | crm | startup | 702 | 0.926 | 0.916 | 0.807 | 0.926 | 0.196 (0.723 / 0.107) | 0.114 | 0.115 | 0.114 | 0.115 | 0.000 |
   | crm | established | 1,963 | 0.930 | 0.919 | 0.813 | 0.930 | 0.161 (0.722 / 0.107) | 0.097 | 0.042 | 0.097 | 0.042 | 0.000 |

   Domain head on held-out established rows: AUC 0.891, accuracy 0.81 (n 3,990); startup 0.5 by construction. The
   rigid startup calibration (mean P 0.61 vs unsafe rate 0.35) reflects 93 optimiser steps, not a property of the
   arm; the fail-calibration gap on rigid is the label artefact the plan review noted (unsafe-not-fail 0.17).

## Cost estimate for the real arms

0.041 s per step at batch 256 on the shared 5090 -> holdout mode (~84k fit rows, 328 steps/epoch, 30 epochs ~ 9.8k
steps) ~ 7 min per seed, ~35 min per 5-seed arm; deploy mode (~105k rows) ~ 8.5 min per seed; `hist_rma` about 2.2x
(teacher + encoder fit + 5 fine-tune epochs). Loading the mixed set takes ~30 s and 9.5 GB of device memory per
process (f32); `--x-half` halves the corridor tensor (~5 GB) at a ~1e-5 logit cost, if two arms must share the GPU.

## Known limits and choices a reviewer should know

- `hist_rma` teacher: the domain one-hot enters through the privileged branch (Linear(10 -> 16) on [privileged 8,
  one-hot 2]) rather than through the ctx vector, so the student has exactly the teacher's architecture (nctx 5 + z 16)
  and no weight surgery is needed when z replaces e; the brief wrote Linear(8 -> 16) with the tag in ctx. The
  dataset's privileged column 8 is `is_crm` anyway. Stage 3 runs at lr/4 (`--rma-lr3`), a choice, not a contract.
- `hist_aux` applies the domain BCE only to rows whose window is at least partly visible after the history drop; on
  all-masked rows the head can only learn the base rate.
- The history drop is applied in the single-stage conditions and in RMA stage 3, not in RMA stage 2 (an encoder fitted
  to reproduce a per-row target from an empty window would only learn the mean).
- Within-group cells on established rows pool anchors of different episodes and frames of the same group (as the
  night-2 re-anchored metrics did); `G_*` further splits by speed profile (designed routes only).
- All rows of the file are put on the device (test rows too, even when evaluating val), so per-process memory is 9.5 GB
  regardless of `--subsample`; `--domain-filter` and `--startup-only` subset before upload.
- The ECE uses 10 equal-width bins; with P concentrated near 0 or 1 most bins are empty and the ECE reduces to a few
  bin gaps.
- Synthetic sets are not benchmarks: the noise flavour is untrainable by design, the structured one is memorised.
