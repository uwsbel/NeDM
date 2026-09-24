# Module note: route-risk trainer for this effort (`scripts/ci_train.py`)

Written 2026-09-22. One file: `/home/harry/NeDM-traverse_mppi/scripts/ci_train.py` (about 750 lines). It is the trainer for
the offline balance and architecture arms (S0 B/A), the short-approach models (S2) and the retrain on new data (S3).
`scripts/ga_train.py` is imported and was not edited (md5 `61598dd089816624e558feeff21c793b`, `git diff` empty).
Self-test inputs, logs and JSON results: `artifacts/traverse/crm_improve_20260922/ci_train_selftest/`.

## What it is

It is a fork of ga_train that reuses these pieces from ga_train: the corridor CNN, the survival loss, input
standardisation, `regime_metrics`, and the domain-head read-out. On top of that it adds:

- several datasets in one run;
- four network families and a choice of history encoder;
- a history window of adjustable length, and a velocity input;
- controls for the rigid/soil mix in each batch and for the amount of data per domain;
- per-source loss weights;
- extra metric blocks.

The domain-tag and history conditions work as in ga_train: `--cond none | tag | hist | hist_aux`. The three-stage
teacher/student condition (`hist_rma`) was not carried over.

### Networks (`--arch`)

Parameter counts below are for `--cond hist_aux`, geometry context.

| `--arch` | what it is | parameters |
|---|---|---|
| `gru` | ga_train's CNN-GRU: same modules, same construction order and same state-dict keys as `ga_train.GAModel`, so the same seed gives the same initial weights | 262,018 (+96,960 with `--hist-enc tx`) |
| `tx96_2`, `tx128_4` (any `txD_L`) | the night-2 tokenised transformer (`n2_arch_train.TokenTx`, imported). Each station's CNN column becomes a token with a sinusoidal station position; the context vector (plus the 16-d history code for the history conditions) is an extra [CTX] token; pre-LN encoder with a final LayerNorm | 363,042 / 939,426 |
| `txjoint` (`txjointD_L`, default d 96, 2 layers, 3 heads) | one transformer over [CTX] + 96 station tokens + the history frames as tokens. A frame token is Linear([frame, mask]) plus a learned embedding of the frame's age plus a learned type embedding (ctx / station / history). Attention runs jointly over stations and time; invalid frames are masked as keys. With `--cond none/tag` it has no history tokens | 362,018 |

### History encoder and window

- **`--hist-enc gru`** (gru / tx arches only): ga_train's causal GRU (16 -> 32) followed by tanh(Linear) to a 16-d code.
- **`--hist-enc tx`** (gru / tx arches only): a small causal transformer (d 64, 2 layers, 4 heads).
  - Frame tokens are Linear([frame, mask]) plus a sinusoid of the frame age; a learned read-out token sits after the
    newest frame.
  - Each token attends to itself and to valid frames at or before it. The read-out gives the code:
    z = tanh(Linear(LayerNorm(read-out))).
  - An all-masked window (standing start) therefore gives one constant code, as with the GRU.
- **`--hist-T N`** keeps only the newest N of the 40 stored 50 ms frames.
  - `--hist-window mask` (default, as the brief says): keeps the 40-frame window and masks the older frames.
  - `--hist-window cut`: feeds only N frames.
  - For the transformer encoders and `txjoint` the two give the same function. For the GRU they do not: with `mask`,
    the GRU steps through the masked frames first.
  - Check (`flags/log_f_T10*`): changing the frames older than the newest 10 changes the logits by exactly 0.0 in both
    modes and all three encoders; changing visible frames changes them by 0.015-0.2.

### Velocity input and conditions

- **`--ctx geom_vel`** adds two columns to the five geometry columns (ctx cols 17-21): vx and yaw rate (history channels
  0 and 6) taken from the newest valid history frame. They are zero when the window is empty, and all seven columns are
  standardised on the fit rows.
  - Checked on real data: on all 26,036 re-anchored rows of the 10 % smoke set, these values equal the stored anchor
    state (ctx cols 0 and 6, rounded to float16) exactly.
  - Startup rows get 0; their stored |vx| is at most 0.17 m/s.
  - The velocity is not affected by `--hist-drop`.
- **`--cond`**: as ga_train. `hist_aux` puts its domain head on z; for `txjoint` the head sits on the masked mean of the
  output history tokens.

### Data (`--ds`, repeatable)

- **Loading several files.** The npz files are concatenated in memory. Each row keeps its own split, group, domain and
  source.
  - Ids must be unique across all rows (hard error).
  - Keys that are only present in some files are dropped and listed. Example: `cls` and `branch_run` when
    `mixed_reanchor.npz` is combined with `branch_both.npz`.
  - `--strict-keys` turns the key-set difference into an error. I relaxed "identical key sets" on purpose so that the
    re-anchored and branch files combine without a rebuild.
  - `hist_cols` must be equal across files.
- **Evaluation-suite groups.** Groups matching `f104_pair_group_*`, `f104_crm_eval_group_*` or `f104_g1_test_group_*`
  (the builders' blacklist) are refused among the fit rows with a hard error, and give a warning among evaluated rows.
- **Which rows go on the GPU.** Only rows that are fitted or evaluated: the fit rows, the md5 dev fold and the
  `--split-eval` split.
  - `--keep-all-rows` restores ga_train's behaviour (every row); it is used only for the reproduction check.
  - `--x-half` stores the standardised corridor as float16.
  - `--x-host` keeps it in pinned host memory and moves each batch to the GPU. On real data this made no measurable
    difference to step time: 0.036 s/step vs 0.028-0.038 for runs with the corridor on the GPU.

### Balance and weighting

- **`--crm-batch-frac F`**: each batch has floor(256 x (1 - F)) rigid rows and the rest soil rows. They come from two
  per-domain shuffles, each reshuffled when used up, as in ga_train.
  - F = 0.5 reproduces ga_train's 128 + 128 exactly. F = 0.75 gives 64 + 192 (checked in the JSON `batch_quota`).
  - An epoch is still n_fit // 256 steps, so a higher F revisits the soil rows more often.
- **`--data-frac-crm` / `--data-frac-rigid`**: keep a fraction of the TRAIN groups per domain.
  - Both domains use one seeded shuffle of the fit groups (`--data-seed`, default 0), so smaller fractions are subsets
    of larger ones, and equal fractions keep the same twin groups.
  - Checked: soil 0.25 gives 15 groups, all inside the 31 groups kept at 0.5; rigid 0.25 keeps the same 15 groups.
  - Applied after `--subsample` (ga_train's group subsample). The dev and evaluated rows are unchanged.
- **`--row-weight 'source=w,...'`**: weights the survival loss per row by the npz `source` field.
  - The batch loss is sum(w x nll) / sum(w).
  - Checked on one batch with 65 branch rows out of 256: 34.525604 vs 34.525606 computed by hand (unweighted 34.3245).
  - The domain-head loss is not weighted.

### Learning rate, weight decay, determinism

- Defaults: lr 2e-3 and wd 1e-4 for `gru`; lr 1e-3 and wd 0.05 for the transformers (the night-2 settings).
  `--lr` / `--wd` override them.
- `--deterministic` switches cuDNN to deterministic mode. Two identical 3-epoch runs on syn_a + syn_b with the flag
  were bit-identical for gru with the transformer encoder and for tx96_2, but not for txjoint (max logit difference
  2.3e-5; its attention uses a per-row key mask). Without the flag, two `--hist-enc tx` runs ended at losses 3.1755
  and 3.1756.

## Metrics (JSON `members[i].dev / .heldout`, and `ensemble`)

- **Structure.** Each block is `{'all': ..., 'by_source': {source: ...}, 'by_anchor': {bucket: ...}}`.
  - Every entry has ga_train's layout: domain (rigid / crm / both) x (startup / established / all), with all of
    ga_train's fields: pooled / within-group / same-speed AUC, lowest-risk pick failure vs random and best route,
    Brier and ECE against unsafe and fail, rates, `unsafe_not_fail_rate`.
  - Anchor-frame buckets: `k0` (k = 0), `k10_30`, `k40p` (k >= 40), plus `k_other` if any rows fall outside these.
- **Same-start-state fields.** Every regime block also carries `S_unsafe` / `S_fail` (with pair counts `Sn_*`),
  `S_pick_fail[_unsafe]`, `S_random_fail*`, `S_oracle_fail*` and `S_cells` / `S_rows`.
  - They are computed within cells of rows that start from one identical state (keyed episode | anchor frame | domain).
    This is the branch-data question: several continuations driven from the same decision state. Single-row cells are
    skipped.
  - In the branch data the original route's re-anchored row at the same frame joins its cell.
- **Domain head.** `hist_aux` runs also report `domain_head` as in ga_train.
- **Cost.** 7.9 s per seed for the 21,653 dev rows and 0.8 s for the 5,394 val rows.

## Checkpoint contract and API

`{out}/{tag}_s{seed}.pt` is a dict with these fields:

- **Identity and network:** `model_kind='ci_train'`, `arch`, `arch_kind`, `tx_d`, `tx_layers`, `hist_enc` (`joint` for
  txjoint), `hist_tx` (d / layers / heads), `cond`, `domain_vocab` {rigid 0, crm 1}, `state`, `cin` (6), `nctx`
  (5 / 7 / 9), `zdim`, `width`.
- **Context:** `ctx_mode`, `ctx_names`, `geom_cols`, `vel_hist_channels` [0, 6].
- **History window:** `hist_cols` (15), `hist_cols_state`, `hist_cols_action`, `hist_T` (frames the model consumes),
  `hist_valid_T` (the newest frames it sees), `hist_window`, `hist_T_stored`, `hist_layout`.
- **Standardisation:** `norm` (corridor mu / sd), `ctx_mu` / `ctx_sd` (5 or 7), `hist_mu` / `hist_sd` (15).
- **Data and run:** `train_rows`, `train_rows_by_domain`, `split_hash` (md5 of the sorted fit ids), `split_eval`,
  `mode`, `domain_filter`, `startup_only`, `hist_drop`, `crm_batch_frac`, `data_frac_*`, `data_seed`, `row_weight`,
  `lr`, `wd`, `epochs`, `seed`, `tag`, `ds` (list), `args`.
- **Compatibility:** for `--arch gru --hist-enc gru` the state-dict keys and shapes equal `ga_train.GAModel`'s
  (checked).

```python
import ci_train as C                                  # PYTHONPATH=src:scripts
model, ck = C.load_ci_model(path, 'cuda')             # eval mode
z = C.encode_history(model, ck, hist, hmask)          # once per decision; hist (T0,15) or (1,T0,15) raw, None = standing start
logits = C.score(model, ck, X, geom5, hist=None, hmask=None, vel=None, domain_onehot=None, z=z)
```

- **`score` inputs.** It takes the raw corridor X (n,5,96,32) f32 and the raw geometry (n,5). It standardises
  everything from the checkpoint.
  - The window can be (T0,15), (1,T0,15) shared by all candidates, or (n,T0,15) per row. The model window rule is
    applied to it: newest `hist_T` frames, older-than-`hist_valid_T` masked. A shorter window is left-padded with
    masked frames.
  - Per-row windows are encoded in the same 512-row chunks as the trainer's `predict()`.
- **Velocity.** For `geom_vel` models the velocity is taken from the window. It is zero for the standing-start window.
  When only a precomputed `z` is passed it must be given as `vel`; `vel_from_history(H, M)` is exported.
- **What `encode_history` returns.** gru / tx arches: z (m,16) f32. txjoint: `{'tok': (m,T,96), 'mask': (m,T)}`, the
  embedded history tokens, which do not depend on the candidate. No-history models: None.
- **Planner.** `scripts/ga_planner.py` still only loads legacy and `ga_train` checkpoints (see the last item under
  known limits).

## How to run

```
cd /home/harry/NeDM-traverse_mppi
OMP_NUM_THREADS=6 PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python scripts/ci_train.py \
  --ds artifacts/traverse/generalist_20260921/A_adapt/datasets/mixed_reanchor.npz [--ds more.npz ...] --out <dir> \
  --arch gru|tx96_2|tx128_4|txjoint --hist-enc gru|tx --hist-T N --hist-window mask|cut --ctx geom|geom_vel \
  --cond none|tag|hist|hist_aux --crm-batch-frac 0.5 --data-frac-crm 1 --data-frac-rigid 1 --row-weight branch=2 \
  --domain-filter both --split-eval val --mode holdout --seeds 3 --epochs 30 [--x-half|--x-host] [--roundtrip-check] [--tag T]
```

- **Outputs:** `{tag}.json` (arguments, per-file row counts, dropped keys, data-fraction report, per-seed rows,
  ensemble), `{tag}_logits.npz` (used rows only: ids, ensemble and member logits, fit / dev / held-out masks, domain,
  anchor frame, group, source, labels, file index), `{tag}_s{seed}.pt`.
- **Default tag:** `ci_{arch}_{hist_enc}_{cond}_{ctx}_T{N|all}[cut]_{domain_filter}[_startup]_{mode}_{split_eval}` plus
  `_cbF`, `_fcF`, `_frF`, `_rw...`, `_subF` when those options are used.
- **Checkpoint check:** `--roundtrip-check` reloads each checkpoint and compares `score()` with the trainer's logits.

## Tests (all on the shared RTX 5090, torch 2.12, `OMP_NUM_THREADS=6`)

### Inputs

Built with ga_train's generator:

```
python artifacts/traverse/crm_improve_20260922/ci_train_selftest/make_ci_synthetic.py artifacts/traverse/crm_improve_20260922/ci_train_selftest
```

- **`syn_a.npz`:** 2,000 rows, generator seed 0, structured flavour, with unique ids. The K1 file `synthetic.npz`
  repeats ids, because the generator draws (group, route, domain, anchor) with replacement, so ci_train refuses it.
- **`syn_b.npz`:** 800 rows re-labelled as branch data:
  - source `branch`, with the same groups and splits as syn_a;
  - 200 blocks of 4 rows from one start state;
  - anchor frames 10 / 20 (short windows) and 60;
  - an extra `cls` key.

### 1. Every architecture x history encoder x context, on syn_a + syn_b

Settings: `--cond hist_aux`, 2 epochs, 1 seed, `--roundtrip-check`. All 14 finished with finite losses
(`ci_train_selftest/matrix/`). The run concatenated 2 files, dropped `cls`, used 2,482 rows, fit 1,704 (863 rigid /
841 soil), with 456 branch fit rows.

"Same batches" is the batch-aligned comparison of `score()` on the reloaded checkpoint against the trainer's logits.

| arch | hist enc | ctx | params | loss first / final | train s | GPU peak GB | rows checked | same batches | precomputed z / standing-start z | other batch shapes | one shared window / zero window |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gru | gru | geom | 262,018 | 36.80 / 3.132 | 1.4 | 1.45 | 1,970 | 0 | 0 / 0 | 0 | 0 / 0 |
| gru | gru | geom_vel | 262,082 | 36.75 / 3.348 | 1.4 | 1.45 | 1,970 | 0 | 0 / 0 | 0 | 0 / 0 |
| gru | tx | geom | 358,978 | 36.63 / 3.176 | 1.4 | 1.54 | 1,970 | 0 | 0 / 0 | 2.0e-5 | 0 / 0 |
| gru | tx | geom_vel | 359,042 | 36.80 / 3.355 | 1.4 | 1.54 | 1,970 | 0 | 0 / 0 | 8.6e-6 | 2.4e-7 / 2.4e-7 |
| tx96_2 | gru | geom | 363,042 | 48.20 / 3.234 | 1.4 | 1.49 | 1,970 | 0 | 0 / 0 | 2.1e-7 | 1.3e-7 / 1.3e-7 |
| tx96_2 | gru | geom_vel | 363,234 | 27.52 / 3.185 | 1.4 | 1.49 | 1,970 | 0 | 0 / 0 | 2.3e-7 | 8.9e-8 / 8.9e-8 |
| tx96_2 | tx | geom | 460,002 | 48.20 / 3.223 | 1.4 | 1.58 | 1,970 | 0 | 0 / 0 | 2.4e-7 | 8.9e-8 / 8.9e-8 |
| tx96_2 | tx | geom_vel | 460,194 | 27.47 / 3.134 | 1.4 | 1.58 | 1,970 | 0 | 0 / 0 | 2.8e-7 | 8.9e-8 / 8.9e-8 |
| tx128_4 | gru | geom | 939,426 | 36.42 / 3.081 | 1.5 | 2.08 | 1,970 | 0 | 0 / 0 | 2.3e-7 | 1.2e-7 / 1.2e-7 |
| tx128_4 | gru | geom_vel | 939,682 | 33.66 / 3.266 | 1.5 | 2.08 | 1,970 | 0 | 0 / 0 | 2.4e-7 | 2.4e-7 / 2.4e-7 |
| tx128_4 | tx | geom | 1,036,386 | 36.52 / 3.070 | 1.5 | 2.17 | 1,970 | 0 | 0 / 0 | 2.2e-7 | 1.2e-7 / 1.2e-7 |
| tx128_4 | tx | geom_vel | 1,036,642 | 33.57 / 3.241 | 1.5 | 2.17 | 1,970 | 0 | 0 / 0 | 2.4e-7 | 1.5e-7 / 1.5e-7 |
| txjoint | joint | geom | 362,018 | 40.05 / 3.199 | 1.4 | 1.62 | 1,970 | 0 | 0 / 0 | 0 | 0 / 0 |
| txjoint | joint | geom_vel | 362,210 | 41.06 / 3.118 | 1.4 | 1.62 | 1,970 | 0 | 0 / 0 | 0 | 0 / 0 |

- **What must match to 1e-5:** the batch-aligned comparison ("same batches") and the precomputed-z paths. Every arm
  gives exactly 0 on these.
- **Scoring in other batch shapes** ("other batch shapes", and the one-window / zero-window columns) only needs 1e-3.
  These differ by rounding alone.
- **Why the history-transformer arms reach 2e-5 there.** Their attention kernel rounds differently for different batch
  sizes: the code z moves by about 4e-7, and the trained net amplifies that. Adding 1e-7 noise to z moved single
  logits by up to 3e-5; the median change was 0.
- **The first failing version.** It compared a random row sample (not whole batches) and failed at 4.7e-5 for exactly
  this reason. So the check now scores whole trainer batches (`PRED_BS` 512), and `score()` encodes per-row windows in
  the same chunks.

Other cond and flag checks (`ci_train_selftest/flags/`):

- **Other conditions:** `txjoint` with cond none, `tx96_2` with tag (nctx 7), `txjoint` with hist, and `gru` with the
  transformer encoder and hist all passed the round trip (0.0 on the same batches).
- **History window:** `--hist-T 10` in mask and cut modes, with gru / tx / joint encoders, passed.
- **Velocity without history:** `--cond none --ctx geom_vel` passed.
- **Batch mix and data fractions:** `--crm-batch-frac 0.75` gives a batch quota of rigid 64 / soil 192.
  `--data-frac-crm 0.5 --data-frac-rigid 0.25` gives fit 214 rigid / 440 soil (from 863 / 841).
- **Loss weights:** `--row-weight branch=2` ran.
- **Float16 corridor:** `--x-half` gives an aligned difference of 8.3e-5 (float16 corridor; tolerance 1e-3) and halves
  the data on the GPU (0.09 vs 0.18 GB).
- **Host storage:** `--x-host` gives logits identical to the on-GPU run (both `--deterministic`: difference 0.0).
- **Deploy mode, test split:** `--mode deploy --split-eval test` gives fit 2,192 and 318 held-out test rows.
- **Soil only:** `--domain-filter crm` gives a single-domain batch.
- **Startup only:** `--startup-only` keeps 724 rows (no branch rows).
- **Several seeds:** `--seeds 2` gives two member rows and an ensemble.
- **Expected failures** (all failed as intended): `--strict-keys` on syn_a + syn_b (key sets differ: `cls`); the same
  file given twice (2,000 duplicate ids); a file whose train groups were renamed `f104_pair_group_*` (suite groups
  among the fit rows).

### 2. Reproduction of ga_train (same data, same seeds)

ga_train is not bit-reproducible on CUDA run to run. Two plain runs differ in the logit by up to 5.6e-5 after 2 epochs
(K1's `synthetic.npz`, hist_aux) and by 5.5e-3 after 3 epochs (`syn_a.npz`, hist_aux). So both trainers were compared in cuDNN deterministic mode. ga_train was run
through a wrapper that sets the flag, because the file cannot be edited:

```
S=artifacts/traverse/crm_improve_20260922/ci_train_selftest
python -c "import sys, torch; torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False;
sys.argv=['ga_train.py','--ds','$S/syn_a.npz','--out','$S/repro/ga','--cond','$c','--seeds','2','--epochs','3','--no-save','--tag','ga_$c'];
import ga_train; ga_train.main()"
python scripts/ci_train.py --ds $S/syn_a.npz --out $S/repro/ci_all  --arch gru --hist-enc gru --ctx geom --cond $c --seeds 2 --epochs 3 --no-save --deterministic --keep-all-rows --tag ci_$c
python scripts/ci_train.py --ds $S/syn_a.npz --out $S/repro/ci_used --arch gru --hist-enc gru --ctx geom --cond $c --seeds 2 --epochs 3 --no-save --deterministic --tag ci_$c
python $S/compare_ga.py $S/repro/ga/ga_$c $S/repro/ci_all/ci_$c        # likewise ci_used
```

Result (`repro/compare.txt`):

- **Identical in both settings.** For each of cond none, tag, hist and hist_aux, seeds 0 and 1, the member logits are
  identical (max difference 0.000e+00). This holds with `--keep-all-rows` (2,000 rows) and with the default
  used-rows-only upload (1,790 common rows; the 210 test rows are not loaded).
- **Everything else matches.** All 1,170 metric values ga_train reports (1,194 for hist_aux, including the domain head)
  are identical. So are the final losses (e.g. hist_aux 3.2975144386 / 3.1821029186), the step counts (12), fit rows
  (1,248), parameter counts and the split hash.
- **Without deterministic mode** the two trainers differ by as much as ga_train differs from itself. At 3 epochs:
  ga_train vs ci_train 9.2e-3 / 3.7e-3 (seeds 0 / 1); ga_train vs ga_train 5.5e-3 / 5.1e-3; ga_train default vs
  deterministic 6.9e-3 / 7.2e-3.

### 3. Real-data smoke (`ci_train_selftest/real/`)

```
python scripts/ci_train.py --ds artifacts/traverse/generalist_20260921/A_adapt/datasets/mixed_reanchor.npz \
  --arch txjoint --subsample 0.1 --epochs 2 --seeds 1 --out <dir> [--cond hist_aux --roundtrip-check]
```

The data: 115,868 rows in the file, 35,194 used. That is fit 8,147 (4,133 rigid / 4,014 soil; 10 % of the train
groups), dev 21,653 and val 5,394. The corridor on the GPU takes 2.4-2.5 GB. Loading takes 10.9-12.4 s; peak resident
host memory is 5.2 GB.

| run | steps | train s | s / step | predict s | wall s | GPU peak allocated / reserved GB | process on nvidia-smi |
|---|---|---|---|---|---|---|---|
| txjoint, cond none (the brief's command) | 62 | 1.8 | 0.029 | 0.6 | 32 | 3.71 / 3.82 | 4,556 MiB |
| txjoint, cond hist_aux, round trip | 62 | 2.1 | 0.034 | 0.6 | 33 | 3.94 / 4.09 | 4,840 MiB |
| gru, cond none, geom_vel, round trip | 62 | 1.7 | 0.028 | 0.4 | 33 | 3.69 / 3.88 | 4,614 MiB |
| gru hist_aux, mixed + branch_both, `--row-weight branch=2 --x-host`, round trip | 66 | 2.4 | 0.036 | 1.3 | 40 | 1.36 / 1.57 | 2,260 MiB |

- **Round trips:** 0.0 on 2,048 rows for every real run.
- **Early memory problem, fixed.** The first version peaked at 6.6-6.7 GB on nvidia-smi. Allocator fragmentation came
  from the 1,024-row prediction batches and the loading temporaries. The fixes: prediction batch 512, cache released
  after loading and before prediction, and `expandable_segments`.
- **Held-out val after 62 steps** (plumbing only, not a result; hist_aux run):
  - within-group unsafe AUC: soil established 0.922, rigid established 0.902;
  - soil lowest-risk pick failure 0.161 (random 0.722, best route 0.107);
  - domain-head AUC on established rows 0.944.
- **Multi-file run** (mixed + branch_both): 36,649 rows used, 314 branch fit rows, dropped keys `branch_run` and `cls`.
  The branch source block reports same-start-state cells: soil `S_unsafe` 1.000, rigid 0.842.

## Cost estimate for the real arms

- **Steps.** Holdout mode on the mixed file has about 84k fit rows (K1 note), so 328 steps per epoch and 9,840 steps
  at 30 epochs.
- **Time per seed.** At the measured 0.028-0.038 s per step on the shared 5090 that is about 5-6 minutes per seed for
  gru and txjoint. Add about 9 s of metrics per seed and 11-16 s of loading. `tx128_4` was not timed on real data
  (night-2 measured it at roughly 4x the GRU per epoch).
- **Memory for a full local run.** About 111k used rows put the corridor at 8.2 GB in float32, which is over the 6 GB
  local limit. Use `--x-half` (4.1 GB) or `--x-host` (the corridor stays in host memory). On the cluster the default
  float32 on-GPU store fits.

## Known limits and choices a reviewer should know

- `--hist-T` with the default `mask` mode follows the brief (older frames masked); `cut` is the alternative. For the
  GRU encoder the two differ, as explained above.
- Keys missing from some files are dropped rather than refused (`--strict-keys` refuses them). Ids must be unique, which
  is why K1's `synthetic.npz` cannot be used.
- The velocity columns ignore `--hist-drop` (a dropped window still sees the anchor speed).
- `txjoint` with cond none / tag is a station transformer with its own layer code (not TokenTx) and no history tokens.
- txjoint is not bit-deterministic even with `--deterministic` (2.3e-5 between two runs). The history-transformer
  encoder's logits depend at about 1e-5 on how rows are batched.
- `hist_rma` was not ported.
- The same-start-state metric keys cells by episode | anchor frame | domain. On re-anchored-only data almost all cells
  are single rows and are skipped. The synthetic generator repeats such keys, so synthetic blocks show values too.
- The logits file holds only the used rows (ga_train writes every row).
- `scripts/ga_planner.py` loads only legacy and `ga_train` checkpoints. Driving a ci_train model in closed loop needs
  the planner to call `ci_train.load_ci_model` / `encode_history` / `score`. That is not done here: not my file.
