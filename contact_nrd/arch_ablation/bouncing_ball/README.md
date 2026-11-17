# Bouncing ball: the 3T, Go2's 1T and the MLP variants, trained alike

> **Record of the original runs.** The numbers and figures below come from the first implementation of this
> ablation, in the NRDLab code base. Scripts, run folders and `nrdlab` commands named in this text are that
> implementation's and are not in this repository. The NeDM implementation is `nedm.contact_nrd.arch_ablation`
> ([`contact_nrd/arch_ablation/README.md`](../README.md)). Result files named here are in `artifacts/contact_nrd/arch_ablation/bouncing_ball/`.

The ball counterpart of [`../go2`](../go2/README.md). The contact NRD
study's ball model (`ContactNRD-Ball`) is three transformers: core, collision and contact
`TokenNet`s (NRDLab's `src/nrdlab/models/contact.py`). This folder trains **Go2's one transformer, unchanged**,
inside the ball 3T's own objective. That way the data, sampling, losses, pushes, rollouts, optimiser,
update budget and selection are the same, and the network is the difference.

| | 3T | this 1T |
| --- | --- | --- |
| Network | core, collision, contact: three `TokenNet`s, 4 layers, 4 heads, width 128 each | `Go2SurrogateModel` (the study's 1T): 6 layers, 8 heads, width 256, no bias. These are the settings of Go2's fair 1T |
| History | 4 states (`configs/contact_nrd/train.json`) | 4 (context 4; Go2's fair 1T used its 3T's history of 8) |
| Tokens | per kind of input (state, differences, query, partner) | one per time step, `[state_t]` (the ball has no commands) |
| Output | `Δs_core + Σ_pairs g · Δs_pair`; the position follows the new velocity (`assemble`) | `Δv, Δω` from the newest token; the position follows the new velocity through the same `assemble` |
| Contact | explicit switches per pair (ball–floor, ball–wall) | implicit; no switches |
| Stages | `core` fit, `collision`, `cotrain` | `cotrain` only. The 3T's fits run on an untrained 3T to produce the windows and `ρ_c` |
| Core loss scale | `core.out_std` | the 1T's `target_std` |
| History noise | starts from the fitted core's error | starts from the 1T's own error |
| Everything else in `cotrain` | the study's `train.json` | = 3T |

Parameter counts for both are logged at start-up and saved in the run's `manifest.json`.

## The second 1T: the 3T core's network

`train_core.py` trains a second 1T. Its network is the 3T's own core `TokenNet`, alone: the same
class, width 128, 4 heads, 4 layers, FFN 512, and the same tokens (current state, history
differences, query). It also keeps the core's ridge-fitted linear path and zero-initialised heads,
and has about 0.8M parameters. There is no collision or contact network: every step, contact or
not, is `assemble(cur, core(...), 0)`. Its core loss scale and starting history noise are the 3T's
own. The only differences from the 3T are one network instead of three, no collision stage, and the
contact and rollout terms training the core. Runs go to
`runs/ContactNRD-Ball/train-model-1t-core/`. The checkpoint is a `contact_nrd` bundle with untrained
collision and contact networks; `evaluate.py --core-t` steps it by its core alone.

## The MLP variants: 3M and 1M

The 3T's `TokenNet`s with MLP trunks. Every `TokenNet` keeps its tokens, embeddings, zero-initialised
heads, ridge-fitted linear path and output scale. Only the trunk changes: the masked token sequence is
flattened, projected to the width (128) and passed through as many pre-LN residual MLP blocks
(width, FFN 512, width, GELU) as the transformer had layers (4).

- **3M**: core, collision and contact all MLPs, on the 3T's full schedule (`train_3m.py`, runs in
  `train-model-3m`). Parameters: core 648 556, collision 729 515, contact 732 422.
- **1M**: the core MLP alone, on the 3T's co-training, as the second 1T (`train_1m.py`, runs in
  `train-model-1m`). Parameters: 648 556.
- `train_1m.py --loss unified`: option C below. It also refits the core's `out_std` to the unified
  unit (`train-model-1m-unified`).

## Running

With the NeDM implementation, from the repository root ([`reproduce.sh`](../../../scripts/contact_nrd/arch_ablation/reproduce.sh)):

```bash
scripts/contact_nrd/arch_ablation/reproduce.sh bouncing_ball   # download, 3m, 1t, 1m, test
```

The NeDM implementation trains the size-matched design of the G1 and SO101 cases: a 3M whose trunks have the size of
the 3T's, and a 1T and 1M of that design (the Go2 1T's network) at the 3T's parameter count. The variants above (the 3T core network
alone as a second 1T, the core MLP alone as the 1M, MLP trunks of the 3T's width) and the unified loss below are the
original implementation's and are not ported. The stored files are `horizon_errors.json` and `horizon_error.png`.

## Option C: the unified loss

The original implementation could also train the single networks with one supervised loss in place of the
co-training's three balanced terms (why: [`loss_balancing_explained.md`](loss_balancing_explained.md)). It is not
the 3T's objective, is not part of the 2 × 2, and is not ported.
