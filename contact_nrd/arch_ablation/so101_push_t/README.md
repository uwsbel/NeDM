# SO101 push-T ablation: one network or three, transformer or MLP

> **Record of the original runs.** The numbers and figures below come from the first implementation of this
> ablation, in the NRDLab code base. Scripts, run folders and `nrdlab` commands named in this text are that
> implementation's and are not in this repository. The NeDM implementation is `nedm.contact_nrd.arch_ablation`
> ([`contact_nrd/arch_ablation/README.md`](../README.md)). Result files named here are in `artifacts/contact_nrd/arch_ablation/so101_push_t/`.

The SO101 counterpart of [`../g1_locomotion`](../g1_locomotion/README.md): a 2 × 2 on the push-T NRD
(**1T, 1M, 3T, 3M**), all at the 3T's parameter budget and gradient-update budget, on the push-T
study's NRD corpus (`nrd_data`, 144 112 training / 7 828 validation episodes). Only the trunk
changes between T and M; only the number of networks changes between 1 and 3.

The **3T is the released `so101/rot_sc8_s61`** and is not retrained. 1T, 1M and 3M are trained
here, with seed 61.

## The one difference from G1

The G1 has one moving body. SO101 has two that push each other: the arm (10 channels) and the T
(13). The 1T and 1M therefore read **both bodies as one token per step**: `[arm | T]`, each
padded to 13 channels (26 in total), plus the arm's command. One head outputs the change of
both bodies. As in the 3T, joint angles, the T's position and its quaternion are not taken from
the network: `assemble` integrates them from the new rates. The G1 1T's adapter
(`g1/fair_1t/train.py`) handles any number of moving bodies, so no class is
duplicated here.

## The four models

| | 3T (released) | 3M | 1T | 1M |
| --- | --- | --- | --- | --- |
| Networks | core, collision, contact | the same | one, both bodies | the same |
| Trunk | `TokenEncoder`: 4 layers, 4 heads, width 128, FFN 512 | residual MLP over each network's flattened tokens: width 256, 2 blocks, hidden solved per network | Go2 1T `StudyBackbone`: 6 layers, 8 heads, causal, no bias, width solved | residual MLP over the 4 flattened step tokens: width 512, 2 blocks, hidden solved; token width = the 1T's |
| Parameters | 2 420 190 | ≈ 2.42 M (in the log) | ≈ 2.42 M (in the log) | ≈ 2.42 M (in the log) |
| Commands read | the current one | the current one | all 4 | all 4 |
| Selection | official score | official score | motion score | motion score |

The solved sizes and the exact parameter counts are the first line of each job's log, and they
are stored in the run's `manifest.json` and in `best.pt`'s `config`.

The residual MLP block is `x + Linear(GELU(Linear(LayerNorm(x))))`, and padded tokens are zeroed
before flattening. See `common/nets.py`.

## Training

The SO101 contact NRD's `train.json` ([`configs/contact_nrd/train.json`](../../../configs/contact_nrd/train.json)): seed
61, history 4, 20 ms step, checks every 10 000 updates on 1 500 validation episodes, and the
best check is kept.

| | 3M | 1T / 1M |
| --- | --- | --- |
| Fits | core ridge linear start, input statistics | the 3T's fits on an untrained 3T (windows, contact scale, noise), then the network's own statistics |
| Collision | 30 000 updates, LR 1e-3, batch 2 048, warm-up 1 000 | – (no gate) |
| Co-training | 120 000 updates, LR 1e-3, warm-up 2 000, cosine, floor 0.01, clip 1.0 | **150 000** updates, otherwise the same |
| Losses | core (4 096) + contact (4 096) + rollout (128, every 8, weight 8) | the same; the contact labels only choose windows |
| Rollout horizon | 0.1 → 1.0 s over the first half | the same |
| Gradient updates | 150 000 | 150 000 |

The 1-network models have no collision stage, so their co-training is lengthened to the 3T's
150 000 total. The G1 ablation kept 120 000. Warm-up, push start and ramps are fractions of the
stage, so they stretch with it.

**Caveat**: the released 3T was trained with an older schedule (`core_clean`/`core_noisy` 30 000
each, contact 160 000, co-training 20 000 at LR 1e-4, warm-started from `rot_sc7_s61`). A
recipe-matched 3T is `nrdlab train-model ContactNRD-SO101PushT --dataset nrd_data`.

**Motion score**: the official free-rollout score (p95 target position + 0.5 · p95 path RMS +
5 cm · p95 target angle + 0.05 · (1 − events right) + non-finite) without its events term. The
1-network models predict no contacts, so they are selected and compared on it; the 3T and 3M get
both.

## Running

With the NeDM implementation, from the repository root ([`reproduce.sh`](../../../scripts/contact_nrd/arch_ablation/reproduce.sh)):

```bash
scripts/contact_nrd/arch_ablation/reproduce.sh so101_push_t   # download, 3m, 1t, 1m, test
```

As the original runs (the larger push-T corpus, its 3T, and 150 000 updates for the one-network models):

```bash
TRAIN_DATA=<...>/so101_push_rl/nrd_data/train MODEL_3T=<...>/so101_push_rl/nrd/rot_sc8_s61/best.pt ONE_UPDATES=150000 \
    scripts/contact_nrd/arch_ablation/reproduce.sh so101_push_t 3m 1t 1m test
```

Scores go to `test_scores_nedm.json`; the files of the original runs are `test_scores.json`, `test_scores_<run>.json`,
`horizon_errors.json` and `horizon_error.png`.
