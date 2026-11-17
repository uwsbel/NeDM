# Architecture ablation: training setups compared

How each model of the ablation was trained, side by side. The names:

| | Three networks (core, collision, contact) | One network |
| --- | --- | --- |
| **Transformer trunk** | **3T**: the contact NRD as designed ([`contact_nrd/`](../README.md)) | **1T** |
| **MLP trunk** | **3M** | **1M** |

Sections 1 to 6 describe the **original runs**, the ones the stored results come from. Every number is from the
case's README, its config, or the run's `manifest.json` (`$NRDLAB_DATA/runs/<task>/<verb>/<run>/`). Section 7 says
how the NeDM implementation (`nedm.contact_nrd.arch_ablation`) differs. Results are in `artifacts/contact_nrd/arch_ablation/`, not here.

## 1. The five systems

| | Bouncing ball | Pool | SO101 push-T | G1 locomotion | Go2 (rigid ground) |
| --- | --- | --- | --- | --- | --- |
| Folder | [`bouncing_ball/`](bouncing_ball/README.md) | [`pool/`](pool/README.md) | [`so101_push_t/`](so101_push_t/README.md) | [`g1_locomotion/`](g1_locomotion/README.md) | [`go2/`](go2/README.md) |
| Task name of the original runs | `ContactNRD-Ball` | `ContactNRD-Pool` | `ContactNRD-SO101PushT` | `ContactNRD-G1Locomotion` | `Go2-Crm-Velocity` |
| Training data | `contact_nrd` (`bouncing_ball-train`) | `pool-train` | `nrd_data`: 144 112 training, 7 828 validation episodes | `g1_locomotion-train` (19 GB) | `go2_rigid_v2_contact_v3`: 2 372 training, 613 validation episodes |
| Scored on | 1 800 test shots | 4 800 test shots | 1 000 test episodes | the G1 test episodes | the 613 validation episodes (no test split) |
| Model step | 20 ms | 20 ms | 20 ms | 20 ms | 10 ms |
| History | 4 | 4 | 4 | 4 | 8 |
| Seed | 61 | 61 | 61 | 61 | 8 |
| Base config | `train.json` | `train.json` | `train.json` | `train_g1_locomotion.json` (`ridge` 0.1) | `go2_model_cfg`, `contact_scale="step"` |
| Moving bodies in a one-network model | 1 | 2 (both balls) | 2 (arm and T, one token per step) | 1 | 1 |

`train.json` is [`configs/contact_nrd/train.json`](../../configs/contact_nrd/train.json).

## 2. Which model was trained, and its size

"Released" means the checkpoint of the contact NRD release, not retrained for this study.

| System | 3T | 3M | 1T | 1M |
| --- | --- | --- | --- | --- |
| Bouncing ball | released, seed 61; 3 networks of about 0.8 M each | trained: 648 556 + 729 515 + 732 422 = 2 110 493 | trained, two of them: Go2's 1T at 4 795 648, and the 3T's core network alone at 798 956 | trained: 648 556 (the core MLP alone) |
| Pool | released, seed 61; 2 404 916 | trained: 5 275 828 | trained: 799 619 | trained: 1 647 363 |
| SO101 push-T | released `so101/rot_sc8_s61`; 2 420 190 | trained: 2 420 086 | trained: 2 487 496 (width 184) | trained: 2 421 128 (token width 184, hidden 929) |
| G1 locomotion | released, seed 61; 2 530 941 | trained: 2 531 477 | trained earlier ([`g1_locomotion/`](g1_locomotion/README.md)): 4 829 696 (width 256) | trained: 4 833 024 (width 256, hidden 1 986; `--budget original`) |
| Go2 | trained earlier (`rigid_step_h8_s8`): 2 465 269 | started: 2 464 525; the run was not finished when these files were copied, so it has no score | trained earlier ([`go2/`](go2/README.md)): 4 814 336 | trained: 4 814 336 (hidden 1 730) |

Size matching differs by system:

| System | What the M and one-network models are matched to |
| --- | --- |
| SO101 push-T | All four at the 3T's parameter count (within 3 %) and at 150 000 gradient updates |
| G1 locomotion | 3M to the 3T (2.53 M). 1M to the existing 1T (4.83 M), which is about twice the 3T |
| Go2 | 3M to the 3T (2.47 M). 1M to the 1T (4.81 M), which is about twice the 3T |
| Bouncing ball | Not matched. The MLP trunks keep the 3T's width, FFN and block count, so they come out smaller |
| Pool | Not matched. The variants keep the study's network settings: 3M is 2.2 times the 3T, the one-network models are smaller |

## 3. Network

| | 3T | 3M | 1T | 1M |
| --- | --- | --- | --- | --- |
| Networks | core, collision, contact; no shared weights | the same three | one | one |
| Trunk | `TokenEncoder`: 4 layers, 4 heads, width 128, FFN 512 | residual MLP over each network's flattened tokens: width 256, 2 blocks, hidden width solved per network (ball: width 128, 4 blocks, FFN 512) | Go2's `StudyBackbone`: 6 layers, 8 heads, causal, no bias; width 256 (ball, G1, Go2) or 184 (SO101). Ball also: the 3T's core `TokenNet` alone | residual MLP over the flattened step tokens, width 512, 2 blocks, hidden width solved for the parameter target |
| MLP block | – | `x + Linear(GELU(Linear(LayerNorm(x))))`; padded tokens zeroed before flattening | – | the same |
| Tokens | one per kind of input: state, history differences, command, query, partner | the same | one per time step: `[state_t, command_t]` | the same |
| Commands read | the current one | the current one | the whole history (4, or 8 for Go2) | the whole history |
| Contact | explicit: the collision network switches each pair on or off, the contact network adds its change | the same | implicit: no switches | implicit: no switches |
| Start of training | zero heads on a ridge-fitted linear path | the same | normal(0, 0.02) (ball core-only 1T: as the 3T) | as its 1T |
| Integrated channels | positions, quaternions and joint angles follow from the new rates (`assemble`) | the same | the same | the same |

The pool's one-network models are variants of the original code (`layout: single`) that keep the 3T's network
settings (width 128); the table's 1T and 1M trunk rows describe the other four systems.

## 4. Training schedule

Shared by every model unless a row says otherwise: AdamW, learning rate 1e-3 with cosine decay to
1 % of it, gradient clip 1.0, no weight decay, running-mean balancing of the loss terms, a check
every 10 000 updates, and the best check kept.

| | 3T | 3M | 1T | 1M |
| --- | --- | --- | --- | --- |
| Fits before training | core ridge linear start, input statistics, contact scale `ρ_c` | the same | the 3T's fits on an untrained 3T (for the windows, `ρ_c`, noise), then the network's own statistics | the same |
| Collision stage | 30 000 updates, LR 1e-3, batch 2 048, warm-up 1 000, then frozen | the same | none | none |
| Co-training | 120 000 updates, warm-up 2 000 | the same | 120 000 (SO101: 150 000) | as its 1T |
| Gradient updates in all | 150 000 | 150 000 | 120 000 (SO101: 150 000) | as its 1T |
| Batch per update | 4 096 contact-free windows + 4 096 contact windows | the same | the same | the same |
| Core loss | one-step error on contact-free steps, core network only, in `core.out_std` | the same | the same steps, whole network, in its `target_std` | as its 1T |
| Contact loss | one-step error on contact steps in `ρ_c`, measured labels as gates | the same | the same; the labels only choose the windows | as its 1T |
| Rollout loss | every 8th update, batch 128, weight 8, in `4ρ_c`, ramped in over the first quarter, horizon 0.1 s to 1.0 s over the first half | the same | the same, no gates | as its 1T |
| History noise and dropout | noise on the history differences, at the network's own one-step error, measured again at every check; history entries dropped at random | the same | the same; the first scale comes from a random start, so it is larger for the first 10 000 updates | as its 1T |
| Pushes | after the first quarter, half of the contact windows use 1 to 3 of the model's own steps as history | the same | the same | the same |
| Objective code | the contact NRD's (`nedm.contact_nrd.train`, which the original code reproduces) | the same, trunks swapped before the optimiser is built | the same, with one network behind the 3T's calls (ball core-only: the core network alone) | as its 1T |

Per-system differences from this table:

| System | Difference |
| --- | --- |
| Bouncing ball | Checks use all 900 validation episodes. The 6-layer 1T and the 1M can also be trained with one unified loss (`--loss unified`, [`loss_balancing_explained.md`](bouncing_ball/loss_balancing_explained.md)); that is not the 3T's objective and is not part of the 2 × 2 |
| Pool | 3M, 1T and 1M are variants built into the original code (`ContactNRD-Pool-3Mlp`, `-1Mlp`, `-1Transformer`), trained with the study's full pool schedule |
| SO101 push-T | The one-network models' co-training is lengthened to 150 000 so that all four have the same number of updates. The released 3T was trained with an older schedule (`core_clean` and `core_noisy` 30 000 each, contact 160 000, co-training 20 000 at LR 1e-4, warm-started from `rot_sc7_s61`) |
| G1 locomotion | `ridge` 0.1 for the core's linear start (1e-6 elsewhere). The released 3Ts were trained by `nedm.contact_nrd.train` |
| Go2 | History 8 and 10 ms steps, so history dropout keeps 2 to 8 states; `go2_model_cfg` with `contact_scale="step"` in place of `train.json` |

## 5. Checkpoint selection and the score reported

| System | 3T and 3M selected on | 1T and 1M selected on | Compared on |
| --- | --- | --- | --- |
| Bouncing ball | free-rollout score on the validation episodes | the same | ball position error against horizon up to 1.7 s, 1 800 test shots |
| Pool | free-rollout score on up to 1 500 validation episodes | the same | ball B position at 2.0 s, path RMS, contact events, 4 800 test shots |
| SO101 push-T | official score: p95 target position + 0.5 · p95 path RMS + 5 cm · p95 target angle + 0.05 · (1 − events right) + non-finite | motion score: the same without the events term | motion score for all four; official score for 3T and 3M |
| G1 locomotion | released 3T: official score (7 terms). 3M: motion score | motion score: the 5 terms without contact events | motion score for all four; official score for 3T and 3M |
| Go2 | p95 of the 1 s velocity error on the 613 validation episodes | the same | open-loop NRMSE at 1, 2 and 10 s from the validation starts with 8 rows of history |

## 6. What is not matched

- **Commands.** One-network models read the command history, three-network models only the current
  command. A one-or-three comparison changes both at once.
- **Size.** Only SO101 has all four models at one size. In G1 and Go2 the one-network models are
  about twice the three-network ones; ball and pool are not matched at all (section 2).
- **Updates.** One-network models have no collision stage, so they get 120 000 updates against
  150 000, except in SO101.
- **Selection.** One-network models predict no contact events, so they cannot be selected on a score
  with an events term. Where the released 3T was selected on the official score (SO101, G1), the
  others were selected on the motion score.
- **Released 3Ts.** The SO101 3T used an older schedule and the G1 3Ts the study's own trainer. A
  recipe-matched SO101 3T is `python -m nedm.contact_nrd.train` on `so101_push_rl/nrd_data/train`.
- **Seeds.** One seed for every model: 61, and 8 for Go2.
- **Convergence.** Every pool variant's validation score was still falling at its last check.
- **Go2 3M.** Not finished, so the Go2 2 × 2 has three scored models.

## 7. The NeDM implementation against the original runs

`nedm.contact_nrd.arch_ablation` trains one design on every case; `scripts/contact_nrd/arch_ablation/reproduce.sh <case>` runs it.

| | Original runs | `nedm.contact_nrd.arch_ablation` |
| --- | --- | --- |
| 3M | SO101, G1, Go2: MLP trunks of width 256, 2 blocks, hidden width solved per network. Ball: width 128, 4 blocks. Pool: the original code's variant, 2.2 times the 3T | the SO101 / G1 design on every case: each trunk at the size of the Encoder it replaces |
| 1T | SO101, G1, Go2: 6 layers, 8 heads, width solved or 256. Ball: the same at width 256, and the core network alone. Pool: one `TokenNet` over both balls, 0.80 M | 6 layers, 8 heads on every case; width solved for the 3T's parameter count, or `--width` |
| 1M | SO101, G1, Go2: residual MLP of width 512, 2 blocks, hidden width solved. Ball: the core MLP alone. Pool: 1.65 M | the SO101 / G1 design on every case; `--width`, `--parameters` |
| Size target | section 2 | the 3T's count by default; G1 as the original: `--width 256` (1T), `--width 256 --parameters 4829696` (1M) |
| One-network updates | 120 000; SO101 150 000 | the config's 120 000; `--updates 150000` |
| Schedule, losses, noise, pushes, selection | sections 4 and 5 | the same: the 3M is `nedm.contact_nrd.train` itself, the 1T and 1M its co-training with the differences of section 4 |
| Selection of the 3M | SO101 official score, G1 motion score | official score (it runs the 3T's trainer) |
| Systems | ball, pool, SO101, G1, Go2 | ball, pool, SO101, G1 |
| Unified loss (ball) | optional | not ported |
| Verified | full runs, the stored results | sizes equal the original's to the parameter (SO101, G1); short runs on subsets of the four cases; no full run |
