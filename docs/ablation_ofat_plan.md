# OFAT Architecture Ablation — flat+CRM one-hot generalist dynamics model

One-Factor-At-A-Time (OFAT) sweep around the current best flat+CRM one-hot
generalist dynamics model, to see how much the transformer's four architecture
axes matter for the domain-balanced open-loop rollout metric that also predicts
RL tracking quality.

## Anchor (the current "latest and greatest" model)

Run: `artifacts/training_runs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm2000_mix25_rebal_rollout_onehot`
Config: `configs/hmmwv_transformer_v07_tire_normal_force_omega_300g_crm2000_mix25_rebal_rollout_onehot.json`

### Architecture

GPT-style causal transformer over continuous 15-D `(state, action)` tokens at
100 Hz (`dt = 0.01 s`), predicting the next-step state delta. Terrain-conditioned:
a 2-D one-hot `[flat, crm]` code is concatenated to every token, so
`input_dim = state(15) + action(3) + terrain(2) = 20`.

| Axis | Anchor value |
|---|---|
| `n_layer` (depth) | 6 |
| `n_head` | 8 (head_dim = 256/8 = 32) |
| `n_embd` (width) | 256 |
| `block_size` (train context) | 128 |
| `head_hidden_dim` | 256 |
| params | ~4.83 M |

Backbone: pre-norm blocks (LayerNorm, causal self-attention, MLP `4·n_embd`),
learned position embedding of size `block_size`, linear input projection
`20→n_embd`, 2-layer GELU readout `n_embd→head_hidden_dim→15`. `dropout=0`,
`bias=false`. Code: `src/nedm/training/model.py`, `model_transformer.py`.

### Training recipe (held fixed across the whole sweep)

- Data: 75% flat / 25% CRM batches. Flat cache
  `hmmwv_tire_rigid_300g_normal_force_omega_seq_v1` (128 M train transitions);
  CRM cache `hmmwv_crm_2000_normal_force_omega_seq_v1` (2.28 M train). batch=64
  → 48 flat + 16 CRM per step.
- Loss: Huber (δ=1.0) with `equal_domain_combined_std` per-channel weights (so
  CRM's ~30× larger Fz deltas don't dominate).
- Selection: `checkpoint_metric = rollout_sel` — the weight-averaged (0.5 flat /
  0.5 CRM) **10 s open-loop rollout err/dist**. `best_val.pt` is the epoch that
  minimizes it. Dual-domain rollout eval runs 12 episodes/domain at 5 s & 10 s.
- Optimizer: AdamW lr 3e-4 → 3e-5 cosine, warmup 1000, wd 0.1, betas (0.9,0.95),
  grad-clip 1.0. 80 epochs × 2000 steps, seed 2026061801.
- **Memory: `load_dataset_into_memory: true` + `pin_memory: false`.** This is the
  verified pairing used by *every* v07 run (flat-only 300g, crm100, and all three
  crm2000 one-hot runs — confirmed at each run's git commit). The flat processed
  cache is fully loaded into RAM (~21 GB; the box has ~48 GB free) rather than
  mmap-streamed. `pin_memory` **must** stay false when loading into memory,
  otherwise the DataLoader page-locks the already-in-RAM batches and exhausts
  physical memory — that is the memory bug to avoid. `load_into_memory=false`
  (mmap streaming) is also the mode the `newton-instability` memory ties to the
  box's random native SIGABRT/SIGSEGV, so `true` is both correct and safer here.

## The OFAT grid — 14 unique configs (anchor shared)

Anchor = 6L / 8H / 256 / ctx128, counted once. Each arm varies exactly one axis;
everything else stays at the anchor + fixed recipe above.

| Arm | Values (anchor in **bold**) | Notes |
|---|---|---|
| Depth `n_layer` | 2, 4, 8, 12, (**6**) | H8 / E256 / ctx128 |
| Width `n_embd` | 128, 192, 384, 512, (**256**) | `n_head` scaled to head_dim 32 (128/4, 192/6, 384/12, 512/16); `head_hidden_dim` tied to `n_embd` |
| Heads `n_head` | 4, 16, (**8**) | at E256 → head_dim 64 / 16 (/32) |
| Context `block_size` | 32, 64, 256, (**128**) | L6 / H8 / E256 |

Total: 1 anchor + 4 + 4 + 2 + 3 = **14**. The anchor is the existing trained run
(not retrained); the other **13 are trained fresh** at seed 2026061801.

Built param counts (validated, no data load, via `validate_configs.py`):

```
arm      spec                    L   H    E  head_dim  ctx      params
anchor   L6_H8_E256_ctx128       6   8  256    32      128   4,829,455
depth    L2 / L4 / L8 / L12      *   8  256    32      128   1.68M / 3.26M / 6.40M / 9.55M
width    E128/E192/E384/E512     6   *   *     32      128   1.22M / 2.73M / 10.83M / 19.23M
heads    H4 / H16                6   *  256   64/16    128   4,829,455 (identical params)
context  ctx32 / ctx64 / ctx256  6   8  256    32     */*    4.80M / 4.81M / 4.86M
```

## Two-stage protocol

**Stage A — screen (14 configs × 1 seed).** Train all 13 new configs (anchor
reused). Rank by the domain-balanced score **S = min-over-epochs `rollout_sel`**
(the best_val checkpoint's score). ~68 min/run for the anchor-size model; bigger
(L12, E512) and longer-context (ctx256) runs are proportionally slower. Serial on
one 4090 → roughly 1 GPU-day. Rank with `rank_stage_a.py`.

**Stage B — confirm (top ~4 × 2 more seeds).** Regenerate the top-4 configs at 2
extra seeds and retrain (`gen_configs.py --seed <s> --suffix _s2/_s3`), for the
across-seed variance and full metrics incl. zero-shot bumpy transfer. ~8 runs.

Total ≈ 22 runs / ~1–1.5 GPU-days on the full 300 GB flat + crm_2000 cache.

## Input-feature ablation at the L8 winner (2026-07-15)

Stage A varied the architecture at a fixed feature set. This pair varies the
**input features** at the fixed Stage-A winner (`L8_H8_E256_ctx128`, S=0.0456),
asking what each block of the input actually buys:

| Run | Change | Input | Readout |
|---|---|---|---|
| `L8_H8_E256_ctx128` (baseline) | — | 20-D = state 15 + action 3 + one-hot 2 | 15 |
| `L8_H8_E256_ctx128_no_onehot` | `terrain_conditioning.enabled` → `false` | **18-D** (no terrain key) | 15 |
| `L8_H8_E256_ctx128_no_tireforce_omega` | drop 4 tire Fz + 4 spindle omega from state **and** target | **12-D** (7-D state) | **7** |

Everything else — L8/8H/E256/ctx128, 75/25 flat/CRM mix, equal-domain-combined-std
Huber, `rollout_sel` selection, AdamW 3e-4→3e-5, 80×2000 steps, seed 2026061801,
`load_dataset_into_memory=false` — is held at the L8 config, so the feature set is
the only variable. Configs are generated as deep copies of the L8 base by
`gen_feature_ablation_configs.py` (`--print-diff` shows exactly what moved).

**What each asks.** `no_onehot` removes the terrain label, so one shared backbone
must infer flat-vs-CRM from the state history alone — it prices the terrain key
against the [[crm-flat-onehot-ablation]] result that one-hot conditioning is what
lets the generalist avoid the flat tax. `no_tireforce_omega` removes the
tire-contact block (the channels that CRM's slip/sinkage physics moves most),
leaving only body kinematics `[vx, vy, roll, pitch, roll_rate, pitch_rate, yaw_rate]`.

**Judge on `rollout_sel`, not `val_loss`.** `rollout_sel` is integrated from
vx/vy/yaw_rate — channels every arm keeps — so it stays comparable across all
three. `no_tireforce_omega`'s `val_loss` is over 7 channels instead of 15 (and its
channel weights re-normalize over 7), so it is on a different scale and is
meaningless as a cross-arm comparison. `rank_feature_ablation.py` omits it
deliberately.

Both are kept **out of `manifest.json`**: Stage-A ranking compares architectures at
a fixed feature set, and the 7-D run is not commensurable there.

### The 7-D caches (`body7`)

`no_tireforce_omega` needs caches whose `state_fields` are the 7-D `default`
preset. Rather than re-running `preprocess.py` over ~300 GB of raw episodes,
`derive_state_subset_dataset.py` **column-slices the existing processed memmaps** —
which is exactly how `*_normal_force_omega_seq_v1` was itself derived from
`*_force_omega_seq_v1` (hence its symlinked `actions`/`rollout`/`episode` arrays).
Only `states`/`targets` depend on the field list; everything else is symlinked to
the source's real files, and per-channel `normalization` entries are exact row
subsets. `--verify` re-reads every written row against the source slice. Takes
~20 s for the 160 M-row flat cache and ~9 GB of disk, versus hours to re-preprocess.

Caveat for downstream work: the RL/Chrono envs feed a 15-D state, so the 7-D
checkpoint is **not** drop-in for `eval_hmmwv_rl_chrono_tracking.py` without an
env-side change. These two runs are an offline-metric study unless that is built.

## Separate, free axis — inference-K sweep (no retraining)

The train-context choice (`block_size`) is distinct from the *deploy* context. On
the winning checkpoint, sweep the number of last tokens fed at inference with
`scripts/throughput/bench_context_accuracy.py` (K = 128…1) — no retraining. This keeps the
"dynamics is near-Markovian → K=16 at deploy is as good as full context"
argument (see the RL dynamics-context speedup work) independent of how much
context the model was *trained* with.

## Reproduce

```bash
PY=/home/harry/anaconda3/envs/nedm/bin/python
# 1. (re)generate Stage A configs + manifest
$PY scripts/ablations/gen_configs.py
# 2. sanity-build every model, no data load
$PY scripts/ablations/validate_configs.py
# 3. launch the serial sweep in tmux (idempotent, resumable, skips completed)
bash scripts/ablations/launch_sweep.sh
tail -f artifacts/training_runs/ablation_ofat/sweep.log
# 4. rank when runs finish (updates live)
$PY scripts/ablations/rank_stage_a.py
# 5. Stage B: pick top-4, then e.g.
$PY scripts/ablations/gen_configs.py --seed 2026061802 --suffix _s2 --only L8_H8_E256_ctx128 ...
MANIFEST=configs/ablation_ofat/manifest_s2.json bash scripts/ablations/run_sweep.sh
```

Input-feature ablation (2026-07-15):

```bash
PY=/home/harry/anaconda3/envs/nedm/bin/python
# 1. derive the 7-D caches (once; ~20 s + ~9 GB, --verify checks every row)
for d in tire_rigid_300g crm_2000; do
  $PY scripts/ablations/derive_state_subset_dataset.py \
    --source-dir artifacts/training_datasets/hmmwv_${d}_normal_force_omega_seq_v1 \
    --output-dir artifacts/training_datasets/hmmwv_${d}_body7_seq_v1 \
    --state-field-preset default --verify
done
# 2. generate the two configs from the L8 base
$PY scripts/ablations/gen_feature_ablation_configs.py --print-diff
# 3. train both serially (idempotent, resumable, skips completed)
tmux new-session -d -s l8_feature_ablation \
  'cd ~/NeDM && bash scripts/ablations/run_l8_feature_ablation.sh; exec bash'
tail -f artifacts/training_runs/ablation_ofat/l8_feature.log
# 4. compare against the L8 baseline (updates live)
$PY scripts/ablations/rank_feature_ablation.py
```

Configs: `configs/ablation_ofat/*.json` (+ `manifest.json`). Runs:
`artifacts/training_runs/ablation_ofat/<spec>/`.
