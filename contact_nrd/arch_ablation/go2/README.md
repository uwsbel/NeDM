# Go2 rigid: the 3T and two 1Ts

> **Record of the original runs.** The numbers and figures below come from the first implementation of this
> ablation, in the NRDLab code base. Scripts, run folders and `nrdlab` commands named in this text are that
> implementation's and are not in this repository. The NeDM implementation is `nedm.contact_nrd.arch_ablation`
> ([`contact_nrd/arch_ablation/README.md`](../README.md)). Result files named here are in `artifacts/contact_nrd/arch_ablation/go2/fair_1t/`.

Three NRD models of the Go2 on rigid ground, trained on the same episodes with seed 8. The 3T and this 1T share their training and differ in the network. This 1T and the earlier 1T share the network and differ in their training. So, side by side, they separate what the network does from what the training does. How the 3T works is written up in full in NRDLab's `nrd.md`.

| Model | Run | Trained by | Network code | Objective code |
| --- | --- | --- | --- | --- |
| **3T** (three networks) | `train-model/2026-10-29_21-46-21_rigid_step_h8_s8` | `nrdlab train-model Go2-Crm-Velocity` (`go2_model_cfg`, `contact_scale="step"`) | `ContactNRD` / `ReducedContactNRDModel` (`src/nrdlab/models/contact.py`, `reduced_contact.py`) | `ContactNRDObjective` (`src/nrdlab/models/contact_train.py`) |
| **this 1T** (one transformer) | `train-model-1t-fair/2026-11-06_18-08-35_rigid_1t_h8_fair_s8` | `train.py` in this folder | `Go2SurrogateModel` (`src/nrdlab/tasks/go2/model.py`) | the 3T's, subclassed as `OneTransformerObjective` in `train.py` |
| **earlier 1T** (one transformer) | `train-model/2026-11-03_16-04-43_rigid_one_transformer_h8_b4096_s8` | NRDLab's `reproduce/go2_single_h8/train.py` | the same as this 1T | `RolloutObjective`, the default of `src/nrdlab/models/trainer.py` |

---

## 1. What all three share

- **Episodes:** `go2_rigid_v2_contact_v3` (manifest sha256 `a593eb43…`), 2 372 training and 613 validation episodes, 10 ms steps. It is converted from the study's `go2_rigid_v2` and keeps its episodes and split; the 3T's `cfg.yaml` names `go2_rigid_v2` and its `manifest.json` records the resolved `go2_rigid_v2_contact_v3`.
- **History:** 8 states. **Seed:** 8. **Validation:** Go2's corpora have no test split, so the validation episodes both choose the checkpoints and score them.

## 2. How they differ

### 2.1 Side by side

"= 3T" and "= this 1T" mean the same as that column. Bold marks the differences that matter most in §3.

| | 3T | this 1T | earlier 1T |
| --- | --- | --- | --- |
| **Network** | | | |
| Networks | core `f_θ`, collision `c_ξ`, contact `h_φ`: three `TokenNet`s, 4 layers, 4 heads, width 128 | one causal transformer: 6 layers, 8 heads, width 256, no bias | = this 1T |
| History | 8 | = 3T | = 3T |
| Parameters | 813 192 + 807 685 + 844 392 = **2 465 269** | **4 814 336** | 4 814 336 |
| Tokens | one per *kind* of input: state, 7 differences `d_k`, action, query, (pairs) partner | one per *time step*: `[state_t, command_t]` × 8 | = this 1T |
| Commands read | **only the current one** | **all 8** | all 8 |
| Output | `Δs_core + Σ_f g_f · Δs_f`, with gates `g_f = 1[z_f ≥ 0]` | `Δs` from the newest token | = this 1T |
| Contact | explicit: per-foot gates and changes | implicit | implicit |
| Initialisation | zero heads: starts as a ridge-fitted linear shortcut | normal(0, 0.02) | = this 1T |
| **Training** | | | |
| Stages | `core` fit, `collision` 30 000 (then frozen), `cotrain` 120 000 | `cotrain` 120 000 (the 3T's fits run on an untrained 3T, for the windows and `ρ_c`) | **one-step** 2 500, then **rollout** 125 from the one-step `best.pt` |
| Updates in all | 150 000 | 120 000 | **2 625** |
| Windows per update | 4096 contact-free + 4096 with a foot down | = 3T | 4096 |
| Optimiser | AdamW, 1e-3 → 1e-5 cosine, 2 000 warm-up, no weight decay, `running_mean` balancing | = 3T | AdamW, 3e-4 → 3e-5, then 1e-4 → 1e-5; weight decay 0.1; no balancing |
| Contact-free steps | `core` term, **core network only**, in `core.out_std` | `core` term, **whole network**, in `target_std` | one MSE in `target_std`, all steps alike, at all 8 positions |
| Steps with a foot down | `contact` term in `ρ_c`, **measured labels as gates** | = 3T, gates ignored | (the same MSE) |
| Rollout loss | from the start (ramped in over ¼): Huber in `4ρ_c`, up to 1 s, batch 128, weight 8, every 8th update | = 3T, no gates | **last 125 updates only**: MSE, 1 s, batch 256, weight 1 |
| Training on its own errors | history noise, history dropout (keep 2–8), pushes after 25% | = 3T | **none** |
| **Selection** | | | |
| Score | p95 of the 1 s velocity error, 613 episodes | = 3T | mean 2 s NRMSE, 256 fixed windows |
| How often | every 10 000 updates | = 3T | every 32 / 16 updates |
| `best.pt` | update 90 000 | update 90 000 | update 160 of the one-step phase, then 125 rollout updates on top: about 285 updates of training in all (§3.4) |

`ρ_c = 0.1 · RMS(s_{t+1} − s_t)` over steps with a foot down (`nrd.md` §7). It comes from the data alone, so the 3T and this 1T use the same number.

### 2.2 Notes on the training the 3T and this 1T share

- **The 3T's core learns from few steps.** Only 1.2% of training steps have no foot down (400 training episodes: 0 feet 1.2%, 1 foot 9.1%, 2 feet 50.7%, 3 feet 31.0%, 4 feet 8.1%). On the other 98.8% the 3T predicts core plus the per-foot terms; in this 1T, every update of both terms goes into the same weights.
- **Which losses compare.** The `contact` loss is in the same units for both (`ρ_c`, the same objective code). The `core` loss is not: the 3T divides by `core.out_std`, this 1T by its `target_std`.
- **History noise.** Gaussian noise on the history differences `d_k`, never on the current state, at the network's RMS one-step error on contact-free validation steps, measured again at every check. The 3T's first scale comes from its fitted linear core; this 1T's from its random initialisation, so its first 10 000 updates run with larger noise.
- **This 1T inside the 3T objective.** `train.py` swaps the objective's class for `OneTransformerObjective`, so every sampler, loss, push, rollout and check is the 3T's code. The objective passes only the current command, so `OneTransformerNet.past` holds the 7 earlier ones `[B, 7, 12]`, looked up by `commands_before(episode, time)` for every batch, push and check window and shifted along one step in a rollout. An untrained 3T network object supplies `dt`, `advance`, `code` and the other attributes the objective reads. Runs go to `train-model-1t-fair`, so `--resume` of a 3T run can never pick one up.

---

## 3. Results

### 3.1 Open-loop accuracy

All three are scored by `score_rollout` with `all_eval.py` (`all_windows.json`) on the same 4096 validation starts: those with 128 rows of history. Each model reads the last 8 rows of it. Mean NRMSE over the horizon, then NRMSE at single steps (lower is better):

| | mean 0–1 s | mean 0–2 s | mean 0–10 s | 1 step | 5 steps | at 2 s | at 10 s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3T `best.pt` (90k) | 0.1040 | 0.1381 | 0.2993 | 0.0245 | 0.0416 | 0.2060 | 0.4083 |
| **this 1T** `best.pt` (90k) | **0.0913** | **0.1238** | **0.2444** | **0.0166** | **0.0310** | **0.1894** | 0.3506 |
| earlier 1T `best.pt` | 0.3250 | 0.3331 | 0.3113 | 0.1146 | 0.2354 | 0.3575 | **0.3403** |
| 3T `last.pt` (120k) | 0.0975 | 0.1590 | 0.5053 | 0.0231 | 0.0378 | 0.3105 | 0.6816 |
| this 1T `last.pt` (120k) | 0.0834 | 0.1496 | 0.5643 | 0.0150 | 0.0267 | 0.3191 | 0.7934 |

(figure: `horizon_error.png`)

`plot_horizon.py` draws the 10 s scoring. Those starts need 1000 rows after them, so the short-horizon points differ a little from the table.

### 3.2 Eval-time ablations of the 3T and this 1T

These re-score `best.pt` with an input changed; nothing is retrained. They were run on the starts with 8 rows of history (`matched_eval.py`, `matched_windows.json`), so the unchanged rows differ by about 1% from §3.1. (As a check that the label gates take effect, forcing every 3T gate off gives 0.209 at 10 steps, against 0.044.)

| | mean 0–1 s | mean 0–2 s | mean 0–10 s | 1 step |
| --- | --- | --- | --- | --- |
| 3T | 0.1055 | 0.1368 | 0.2962 | 0.0259 |
| 3T, gates = the recorded contact labels | 0.1035 | 0.1360 | 0.3125 | 0.0257 |
| this 1T | 0.0917 | 0.1228 | 0.2457 | 0.0174 |
| this 1T, earlier commands replaced by the current one | 0.1417 | 0.2311 | 0.5691 | 0.0407 |

### 3.3 What the three show

| # | Finding | Evidence |
| --- | --- | --- |
| 1 | **The ranking depends on the horizon.** Up to 2 s: this 1T, the 3T, then far behind the earlier 1T. At 10 s the three are close, and the earlier 1T is lowest. | Mean 0–1 s: 0.091, 0.104, 0.325. At 2 s: 0.189, 0.206, 0.358. At 10 s: 0.351, 0.408, 0.340. |
| 2 | **Training matters more than the network.** | Same network, different training: this 1T vs the earlier 1T, one step 0.017 vs 0.115 (7×). Same training, different network: 3T vs this 1T, 0.025 vs 0.017 (1.5×). |
| 3 | **With the same training, this 1T beats the 3T, and the gates are not why.** | Perfect gates barely change the 3T (§3.2); its collision network misses about 1.7% of contacts per foot. This 1T's one-step error is a third lower on every channel group (velocity, body rates, roll/pitch, joints). |
| 4 | **This 1T uses the command history.** | Without it, this 1T falls below the 3T (§3.2). Joint targets change every 20 ms and the joints respond to the recent ones; the 3T sees only the newest. |
| 5 | **The 3T fits the training data less well.** | Final `contact` loss 0.47 vs 0.22, same units and batches: a limit of the additive core-plus-feet structure, not of generalisation. |
| 6 | **Past update 90 000 both models trained by `ContactNRDObjective` get worse over long horizons.** | `last.pt` over 0–10 s: 0.51 (3T) and 0.56 (this 1T), against 0.30 and 0.24 for `best.pt`. Use `best.pt` for both. |

### 3.4 Why the earlier 1T is so far behind

From its run's `metrics.jsonl` (checks on 256 fixed validation windows):

| Step | What happened | Numbers |
| --- | --- | --- |
| 1. One-step phase, 2 500 updates | The model became accurate per step but unstable in rollout. | One-step validation error fell steadily from 0.31 to 0.029. The 2 s rollout error stayed between 1.0 and 1.4 at every check, no better than holding the last state (1.25); 1.23 at the last check. |
| 2. Selection | Selection by the 2 s error kept **update 160**, when the network had barely started. The 2 340 updates after it were thrown away. | 2 s score 0.98, one-step error 0.12. The phase's own record: `passed: false`, ratio to persistence 0.79. |
| 3. Rollout phase, 125 updates | Started from update 160 and was too short to recover. It stopped the divergence but did not fix the per-step error. The delivered model has about **285 updates** of training in all. | Learning rate at most 1e-4. The 2 s error fell from 0.98 to 0.35; the one-step error stayed at 0.12. |

The 3T and this 1T avoid all three problems. Rollout loss, history noise and pushes run in the same updates as the one-step terms, so the models become accurate and stable together. Their selection score fell at every check up to update 90 000 (this 1T: from 0.40 at 10 000 to 0.072), so the checkpoint they keep is the trained one. More one-step updates alone would not have been enough: 2 500 of them never brought the earlier 1T's 2 s error below that of holding the last state.

### 3.5 What is not yet separated

- **Between the 3T and this 1T.** The ablation only shows that this 1T *uses* its command history: it was trained with it, so removing it at eval time takes it outside its training data. This 1T also has about twice the parameters, scales its core term differently and starts from its own noise scale. Three retrainings would separate the causes: (1) this 1T with its earlier commands masked in training (is it the command history?); (2) the 3T given the command history (can it match this 1T?); (3) a 1T of about 2.5 M parameters (is it the size?).
- **Between this 1T and the earlier 1T.** Which of this 1T's training choices matter most once training is long enough: the rollout weight and schedule, history noise, pushes, the contact scale `ρ_c`, the learning rate or the weight decay. One retraining per choice would tell. (A batch-64 run of the earlier 1T's script, `2026-11-03_15-56-25_rigid_one_transformer_h8_s8`, was stopped during its one-step phase and delivered no model.)

## 4. Reproducing

Go2 is not part of the NeDM implementation: this repository has no Go2 model, data loader or objective, so the Go2
models can only be trained with the original code. The stored files are
`all_windows.json`, `matched_windows.json` and `horizon_error.png`; `scripts/contact_nrd/arch_ablation/go2/plot_horizon.py`
redraws the figure from `all_windows.json`.
