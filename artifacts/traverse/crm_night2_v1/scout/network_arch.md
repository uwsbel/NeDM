# Scout: network architecture and prior ablations (read-only, 2026-09-17/18)

Worktree `/home/harry/NeDM-traverse_mppi` (branch `traverse_mppi`). All paths below are relative to it unless absolute.
Checks were run with `/home/harry/miniconda3/envs/nedm/bin/python` (torch 2.12.0+cu130, RTX 5090 32 GB). No simulation or cluster job was run.

## 0. What is already settled, and what is not (so tonight does not repeat it)

| Question | Status | Where |
|---|---|---|
| GRU vs transformer vs no-sequence MLP, same front end, lr 2e-3 for all, night-1 data (8,160 fit rows) | GRU .958 > tx .947 > MLP .944 (G_unsafe, means over 9 runs = 3 seeds x 3 ctx variants). **Superseded by the audit** | `night2_v1/sweep.json`; `night2_v1/LOG.md:14-20`, `:160-166` |
| Same, re-run with 8 seeds (audit) | GRU .9528, tx .9444, MLP .9394; **at the deployed context (geometry only) GRU and tx tie exactly, .9551 each**; at **lr 1e-3** a 5-seed tx ensemble scores **.9641 vs GRU ensemble .9608**. Verdict recorded: "no benefit at equal tuning, no penalty either". **Text only** - no JSON/checkpoint survives (grep of `artifacts/`, `docs/` for .9641/.9608/.9551 hits only `night2_v1/REPORT.md:148-150`); the data file (`station_ds_fix.npz`, 8,160 rows) was deleted in the 09-15 cleanup; the script was edited after `sweep.json` was written (`LOG.md:183-184`) | `night2_v1/REPORT.md:148-150`, `LOG.md:160-166` |
| Vehicle state as context (full 17 / chassis 7 / none) | Uninformative (R^2 .003 on 523 held-out groups). BUT every episode starts settled at rest (vx, roll, pitch IQR ~.07) and the dev metric has zero within-cell state variance, so the ablation **could not** have detected a velocity effect. Nothing about *moving* starts is known. | `LOG.md:167-171`; `REPORT.md:16-20,151-154` |
| Data quantity (designed routes, GRU, ctx none) | 2,491 rows .905 -> 4,930 .939 -> 9,773 .965 -> 19,463 .980 (still rising; ~1/5 of the gain is dev-group crowding). Adding on-policy rows: designed-dev unchanged (.976-.980) but on-policy-dev .933 -> .969 | `night2_v1/scaling.json`, `onpolicy_eval.json`; `LOG.md:28-32,45-50,172-180` |
| Same architecture from scratch on CRM soil | 13,821 rows, 5 seeds, ~75 s/seed on MI350X; held-out val+test groups pooled AUC .988 / within-group .986 / same-speed .955; closed loop 91 % vs 68 % goal reached vs the frozen rigid ensemble | `crm_f104_v1/REPORT.md:13-15`, `train_v1/deploy/CRM_N2_deploy.json` |
| Energy as an output | **Never trained or evaluated** in this line. The old FDM line (`src/nedm/traverse/fdm_data.py:171-215`) had a horizon-wise positive-work target, never in the route-risk model. Tonight's energy_scan scout has already built per-station targets (`crm_night2_v1/datasets/energy_{rigid,crm}.npz`) | this report section 6 |
| Velocity as an input | **Never** (no row with a moving start exists in any station dataset) | section 5 |
| Parameter-matched comparison, proper tokenisation, transformer-specific lr/epochs, energy/velocity heads | **Open** - this is tonight's study | sections 4-7 |

Bottom line of the prior work: the only architecture evidence is on 8k rows of night-1 data, at one learning rate, with a conv-transformer hybrid that lacks a final LayerNorm (see 1.3). It is a tie in text and a small GRU win in the surviving JSON. Nothing at 14-34k rows, nothing on CRM, nothing parameter-matched.

## 1. The current network, exactly

### 1.1 Definition
`scripts/gen_riskmodel.py:15-57` (`class Net`), byte-identical to `scripts/f104_n2_train.py:27-69`; the night-1 ancestor is `scripts/f104_night_train.py:30-55` (`HazardNet`, GRU only).

```
Net(cin, nctx, arch='gru', width=64, layers=2, heads=4)
forward(x: (B, cin, 96, 32), ctx: (B, nctx)) -> hazard logits (B, 96)
```
- Input `x`: 6 channels = `[elev_rel, along_grade, cross_slope, commanded_speed, valid, ones]`. The first four are z-scored with fit-set per-channel mean/sd over all rows and positions (`f104_n2_train.py:85-86`, `crm_train.py:78-80`); `valid` is left as is; the 6th constant-1 channel is appended at load time (`f104_n2_train.py:87`, `crm_train.py:82`) - a legacy "speed known" flag from the night-1 speed-drop regulariser (`f104_night_train.py:126,199`). `gen_planner.RiskModel.score` (`scripts/gen_planner.py:170-183`) **hard-codes** `x[:, :4]` normalisation and the appended ones; `norm['cont_index']` written by `crm_train`/`sensor_train_v2` is ignored there.
- Corridor geometry (`scripts/f104_n2_dataset.py:12,45-60`): 96 stations resampled uniformly along the route, 32 lateral samples over +-6 m (0.387 m spacing), bilinear map sampling; grades are `np.gradient` of the filled elevation, clipped to +-2; speed channel = reference speed interpolated at each station, repeated laterally.
- `ctx`: geometry-only 5-d `[goal_dx, goal_dy, |d|, start_yaw, route_len]` = columns 17..21 of the stored 22-d `ctx` (`f104_n2_train.py:22-24`, `crm_train.py:18`). Columns 0..16 are the settled initial state in `nedm.training.constants` preset `tire_normal_force_omega_pt` order: 0 vel_body_x, 1 vel_body_y, 2 roll, 3 pitch, 4 roll_rate, 5 ang_vel_body_y (pitch rate), 6 yaw_rate, 7-10 tire fz FL/FR/RL/RR, 11-14 spindle omega x4, 15 engine_motor_speed_radps, 16 engine_motorshaft_torque_nm (`src/nedm/training/constants.py`, `scripts/traverse_wp9_arm_nrd.py:62`). ctx is z-scored with fit-set stats.
- CNN (`gen_riskmodel.py:18-23`): 4 x [Conv2d 3x3, BatchNorm2d, GELU], channels 6->32->64->64->96, stride (1,1) then (1,2) x3: the station axis is never pooled, the lateral axis goes 32->16->8->4. Output (B, 96, 96 stations, 4). 113,088 params.
- Lateral pooling + projection (`:24, :44-45`): cat(mean, max over the 4 lateral cells) -> (B, 96, 192) -> Linear(192, 96) + GELU. 18,528 params.
- Context path (`:25, :48`): Linear(nctx, 32) + GELU, broadcast to every station (192 params for nctx = 5).
- Position: one scalar channel `linspace(0, 1, 96)` (`:47`).
- `tconv` (`:27-28, :49-50`): Conv1d(96+32+1 = 129 -> 96, k = 5, pad 2) + GELU + Dropout(0.1); 62,016 params (k = 1 for `mlp`: 12,480).
- Mixer (`:29-40, :51-56`):
  - `gru`: BiGRU(96 -> 64, 1 layer, bidirectional) -> 128-d; head Linear(128, 1). 62,208 + 129 params.
  - `tx`: learned positional table (1, 96, 96) ~ N(0, .02) **added** to the tconv output; `nn.TransformerEncoderLayer(d_model=96, nhead=4, dim_feedforward=192 (= 2d), dropout=0.1, batch_first, norm_first=True (pre-LN), activation='gelu')` x `layers` (2 in every run so far); **no final LayerNorm**; head Linear(96, 1). Mixer 149,568 (2 layers) / 299,136 (4 layers) params, i.e. 74,784 per layer. `width` is unused for `tx`; `heads` is only used by `tx`.
  - `mlp`: Linear(96, 128)-GELU-Linear(128, 128)-GELU, head Linear(128, 1): no cross-station mixing beyond the k = 5 conv (or none at all, since k = 1).
- Output -> route score (`gen_riskmodel.py:10-12`): `route_logit = log(sum_j softplus(h_j) + 1e-6)` = cloglog of P(event anywhere); `P(unsafe) = 1 - exp(-exp(z))` (`gen_planner.py:184`).

### 1.2 Parameter counts (verified by instantiation)
| variant | nctx=5 (deployed) | nctx=12 (chassis) | nctx=22 (full) |
|---|---|---|---|
| gru (w=64) | **256,161** | 256,385 | 256,705 |
| tx, 2 layers | 352,705 | 352,929 | 353,249 |
| tx, 4 layers | 502,273 | 502,497 | 502,817 |
| mlp | 173,345 | 173,569 | 173,889 |

`docs/progress.md:349` says "256,677 parameters": that is `state_dict` numel = 256,161 trainable + 516 BatchNorm buffers (running mean/var 2 x 256 + 4 counters). Checkpoints (`night2_v1/final/N2_s*.pt`) store `state, arch, layers, ctx_variant, ctx_cols, norm{mu,sd}, ctx_mu, ctx_sd, cin=6, nctx=5` (`f104_n2_train.py:124-127`).

### 1.3 What the existing `tx` variant is, and is not
It is a **conv-transformer hybrid**, not a tokenised transformer: tokens are the outputs of the k = 5 Conv1d over [CNN column features, broadcast ctx, scalar position], so local mixing is already done by the conv, the context is concatenated per token instead of being a token, the positional signal enters twice (scalar channel + learned table), `dim_feedforward = 2d` (192), pre-LN blocks with **no final norm** before the linear head (the head reads the un-normalised residual stream; standard pre-LN stacks end with a LayerNorm), and it was trained at the GRU's lr 2e-3 with the same 30-epoch OneCycle. The audit already showed lr 1e-3 flips the sign of the comparison. Tonight's transformer arm should fix all five points (section 4).

## 2. How the models were trained and scored

### 2.1 Recipe (identical in every trainer)
`f104_n2_train.py:95-128 train_one`, `crm_train.py:98-128`, `sensor_train_v2.py:92-127`, `f104_night_train.py:162-260`:
AdamW(lr 2e-3, wd 1e-4); `OneCycleLR(max_lr=lr, total_steps = epochs * (n_fit // 256))` (pct_start .3 default); batch 256; **30 epochs**; grad-norm clip 5.0; `torch.manual_seed(seed)`; last-epoch weights are used (no early stopping, no validation-based selection); loss = mean discrete-time survival NLL. Corridors are kept as a pinned CPU tensor and gathered per batch (`f104_n2_train.py:100-103`); ctx and event_idx live on the GPU. Training is **not seed-deterministic** (cuDNN GRU; `LOG.md:183`).
Gotcha: the comment at `f104_n2_train.py:100-101` ("~27 GB as float32 on device") is wrong by 10x: 36,199 x 6 x 96 x 32 x 4 B = 2.67 GB (1.33 GB as fp16). The whole tensor fits on the 5090 (and on an MI350X); a device-resident X removes the host gather and roughly halves the step time (see 2.5).

### 2.2 Loss
`f104_night_train.py:63-72 survival_nll(haz, ev)` (copied in `crm_train.py:52-59`): with `ev` = event station (0..95) or -1 (censored = clean route): `nll = sum_{j < ev} softplus(h_j) + [ev >= 0] * softplus(-h_ev)`; censored rows sum softplus over all 96 stations. Event station = `round(high_water_mark_progress / L * 95)` at the first rollback (vx < -0.1 with throttle > 0.3, or vx < -0.3) or 1 s of near-stop under throttle, else the last frame for failed runs (`f104_n2_dataset.py:100-123`). `unsafe` = not clean (fail OR any rollback), `fail` = status != goal_reached.

### 2.3 Dev protocol and metrics
- Dev fold: groups with `md5(group) % 5 == 0` (`f104_night_train.py:26-27`), designed-source rows only (`f104_n2_train.py:82`). Selection rule fixed before night 2: maximise `G_unsafe`, ties within .01 broken by `G_fail` (`f104_n2_train.py:12-13`).
- Metrics (`f104_night_train.py:75-102`, tolerant copy `crm_train.py:21-49`): `P_` pooled AUC; `W_` within-start/goal-group AUC (all positive-negative pairs pooled over groups, micro-averaged); `G_` within (group, speed profile) cell AUC = the "same-speed" number; `Gn_/Wn_` = pair counts. `profile = route index % 4` from the id (`f104_n2_dataset.py:119`); planner-proposal rows have profile -1 and are excluded from `G_`. Profile -> speed: 0 = 2 m/s, 1 = 4 m/s, 2 = 6 m/s, 3 = 2-6-2 m/s (matches the per-profile unsafe rates, rigid .563/.197/.014/.169, CRM .862/.563/.335/.546, against `crm_f104_v1/REPORT.md:54`).
- Route-choice metric (`sensor_train_v2.py:30-49 route_choice`): within each (group, profile) cell pick the argmin score and report picked-unsafe, avoidable-unsafe (a safe sibling existed), oracle and random-pick rates. This is the planner-relevant read-out and should be reported for every arm tonight.
- G_ on CRM is noisy: unsafe == fail there (`crm_f104_v1/REPORT.md:144-145`), and at a 68 % fail rate many 2 m/s cells are all-fail, so `Gn` is small (307 pairs dev, 132 held-out) and per-seed G_unsafe ranged .947-.962 (`CRM_N2_deploy.json`). Prefer `W_` and route-choice on CRM.

### 2.4 The night-2 sweep as it was actually run (`night2_v1/sweep.json`, 27 rows, 2026-09-12 01:40)
`python scripts/f104_n2_train.py --ds night2_v1/station_ds_fix.npz --archs gru,tx,mlp --ctxs full,chassis,none --seeds 3 --epochs 30 --layers 2`; n_fit 8,160 designed night-1 rows (dev fold excluded), lr 2e-3, 8-12 s per run on the 5090. G_unsafe mean +- sd over 3 seeds (my aggregation):

| arch | ctx=full (22) | ctx=chassis (12) | ctx=none (5) | mean over ctx | G_fail (full/chassis/none) |
|---|---|---|---|---|---|
| gru | .9608 +- .0065 | .9586 +- .0019 | .9542 +- .0098 | **.9579** | .876 / .876 / .881 |
| tx (2L) | .9390 +- .0100 | .9466 +- .0082 | .9564 +- .0082 | .9473 | .862 / .871 / .896 |
| mlp | .9444 +- .0057 | .9564 +- .0094 | .9303 +- .0050 | .9437 | .866 / .861 / .857 |

Column means: full .9481, chassis .9539, none .9470. Per-config sd .002-.010 with 3 seeds: differences under ~.01 are not resolvable. Note `tx` is *best* at ctx=none in this JSON (.9564 vs gru .9542), consistent with the audit's tie. The audit's 8-seed and lr-1e-3 numbers (section 0) are the better evidence but exist only as prose.

### 2.5 Timing records (for the cost estimates)
| run | rows | steps | s/seed | ms/step | device |
|---|---|---|---|---|---|
| sweep (8,160 rows, 30 ep) | 8,160 | 930 | gru 9, tx 11-12, mlp 8 | ~10-13 | 5090 |
| scaling (`f104_n2_final.py`) | 2,491 / 4,930 / 9,773 / 19,463 / 26,783 | | 3 / 6 / 12 / 25 / 66 | | 5090 |
| deployed N2 (`final/N2_meta.json`) | 31,851 | 3,720 | 41-79 | 11-21 | 5090 |
| N2T tilt ensemble | 31,851 | 3,720 | 41-56 | 11-15 | 5090 |
| sensor_v2 12-channel variants | 54,180 | 6,330 | 190-296 | 30-47 | 5090 |
| CRM deploy (`CRM_N2_deploy.json`) | 13,821 | 1,590 | 67-92 | 42-58 | MI350X |
| CRM holdout | 10,974 | 1,260 | 39-55 | 31-44 | MI350X |

My GPU-resident micro-benchmark (batch 256, fwd+bwd+AdamW, 5090): gru 9.4 ms, tx-2L 11.8, tx-4L 14.9, mlp 8.8; the CNN front end alone is 7.8 ms, i.e. ~80 % of a step - the mixer choice moves the cost by only +-3 ms. The gap between 9.4 ms (resident) and 11-21 ms (real trainer) is the pinned-host gather, not the model. The MI350X is 2.5-3x slower per step than the 5090 on this latency-bound workload; pack several runs per node (or several seeds per GPU) there.

## 3. Data available tonight (what the arms can be trained and read out on)

| dataset | rows | groups | split | labels | notes |
|---|---|---|---|---|---|
| `fdm_f104_50h_20260909/night2_v1/station_ds_all.npz` | 36,199 (X fp16 (N,5,96,32), ctx (N,22)) | 2,700 | train 33,840 / val 764 / test 1,595 | unsafe .324, fail .168, censored .676 | sources: designed 26,069, on_policy 9,309, misc 821; waves night1 12,542 / A1 14,348 / A2 9,309; no `status` key (dropped by `f104_n2_merge.py`); **every night-2 group is forced to split=train** (`f104_n2_merge.py:16-17`) |
| `crm_f104_v1/datasets/station_ds_crm_v1.npz` | 15,235 | 1,200 | train 13,821 / val 713 / test 701 | unsafe = fail = .681, censored .319 | designed 9,168 + on_policy 6,067; ids are the night-2 pool reused verbatim (`f104_v2_group_*`), so **15,024 CRM rows have a rigid twin with the same id** in station_ds_all (211 have none) |
| `fdm_f104_50h_20260909/gen_v1/station_ds_gen_v1.npz` | 15,639 | | | has `arena` | arenas f104 1,103, g203/g216/g217/g228/g231 ~2,900 each; sources designed 9,000 + gen_test_arm 6,639 -> zero-shot arena read-out for rigid arms |
| raw episodes (local) | production_v2 11,412, v3 14,400, v4 9,600, gen_v1 9,000, CRM collect_v1 15,235 | | | | each has `trajectory.npz` (20 Hz: state (n,17), action (n,3), pose (n,3), power_kw, positive_work_kj_per_interval, parked, dt_s), `outcome.json` (status, positive_work_kj, goal_time_s ...), `command_reference.npz` (reference_waypoints (~90,2), _stations, _speeds, per-frame desired_speed_mps), `anchor_state.npz`, `case.json` |
| `crm_night2_v1/datasets/energy_{rigid,crm}.npz` (built tonight by the energy scout, `scripts/n2_energy_targets.py`) | 35,412 / 15,235 | | | `E (N,96)` cumulative W+ kJ at first arrival at station j, `T (N,96)` s, NaN beyond furthest station; `total_wplus, elapsed, route_len, reach_frac, fail, mean_speed_cmd, max_dev` | join to the station datasets **by id** |

Verified: `power_kw == state[:,15] * state[:,16] / 1000` exactly (corr 1.000, median ratio 1.000 on a rigid and a CRM episode), and `sum(positive_work_kj_per_interval) == outcome.positive_work_kj` (124.28 / 1028.93 kJ). "Engine speed x torque integrated" is literally the recorded W+.

Energy-target facts that decide the loss design (my checks on the two npz files):
- **Station 95 is observed for 0.4 % of rigid rows and 0.1 % of goal-reached rows.** The goal radius is 2.5 m, so the high-water mark stops at ~0.94 L: last observed station p5/50/95 = 87/89/91 for goal-reached routes (both worlds). Route energy must be read at the last observed station (or at `j_max = floor((L - 2.5) / L * 95)` ~ 89), and a per-station loss must mask NaN and must not confuse "parked inside the goal radius" with "failed".
- Rigid, goal-reached, E at last station: p5/50/95 = 122 / 246 / 1,830 kJ (log10 sd .35, heavy right tail = routes that stalled and recovered); 2.6 / 5.7 / 42 kJ/m. corr(E, L) = .01; **Spearman(E, commanded mean speed) = -.07**; within same (group, profile) cells (8,055 cells) max/min ratio p50 1.44, p90 6.2; within-group Spearman(speed, E) median -.05. So on rigid ground energy is neither a length nor a speed proxy, and lateral choice alone moves it by ~44 % typically: there is something for a network to learn.
- CRM, goal-reached: 366 / 529 / 820 kJ (log10 sd .11); 8.4 / 11.7 / 18.4 kJ/m; corr(E, L) = .44, Spearman(E, speed) = +.23, within-group Spearman(speed, E) = +.37; same-cell ratio p50 1.15 / p90 1.48. Faster costs more on soil; the within-cell spread is small, so rank correlations will be modest.
- **Failed routes burn most of their work after the stall**: rigid median 183 kJ up to the last reached station vs 1,924 kJ total (1,414 kJ spinning in place); CRM 331 vs 730. Only the pre-event prefix is a valid energy label; `total_wplus`/`outcome.positive_work_kj` must not be used for failed routes.
- Per-station increments (goal-reached): p1/50/99 = 0 / 0.7 / 49 kJ rigid (33 % of increments <= .01 kJ: W+ is positive-part-only, downhill/coasting gives exact zeros), 0 / 4.8 / 55 kJ CRM (8.7 % zeros). Increments are zero-inflated and heavy-tailed: do not regress kJ with MSE.

## 4. Proposed architecture arms (equal data, matched parameters)

All arms keep the survival loss, the 96-station output and `route_logit`, so every checkpoint stays loadable by a planner with one extra head. Parameter counts and step times below were measured by instantiating each arm with **both** heads (hazard + per-station energy), batch 256, device-resident, 5090:

| id | arm | params | ms/step | purpose |
|---|---|---|---|---|
| A0 | deployed `Net(arch='gru')` + energy head | 256,290 | 12.1 | control (identical to N2/CRM_N2 up to the extra head) |
| A1 | GRU hidden 120 (`nn.GRU(96,120,bidirectional)`) | 351,266 | 13.3 | **parameter-matched to the existing tx-2L (352.7k)** |
| A2 | GRU hidden 96, 2 stacked layers | 472,994 | 13.8 | matched to tx-4L (502k) |
| A3 | existing `tx` d96 L2 as is, but lr 1e-3 and a final LayerNorm | 352,705 | 11.8 | reproduces the audit's "edges ahead" claim on today's data |
| A4 | **Tokenised transformer** d96 L2 h4: token_j = Linear(192->d)(cat(mean,max) of CNN column j) + PE_j; **[CTX] token** prepended (Linear(5->d)); no k=5 conv; ff = 4d; pre-LN; final LayerNorm; dropout .1 | 356,258 | 15.3 | proper tokenisation at the same size as A1/A3 |
| A5 | same, d96 L4 | 579,938 | 19.7 | depth |
| A6 | same, d128 L4 h4 | 932,162 | 21.6 | width |
| A6' | A6 with learned PE instead of sinusoidal | 944,450 | 21.4 | PE ablation (one seed set only) |
| A7 | same, d192 L4 h8 | 1,931,522 | 31.6 | 7.5x the GRU: does capacity matter at 14-34k rows? |
| A8 | **Patch transformer, no CNN**: patch = 4 stations x 32 lateral x 6 ch -> Linear(768->128), 24 tokens, L4 | 892,802 | 11.1 | tests whether the station-preserving CNN is load-bearing; fastest arm |
| A8' | patch 2 -> 48 tokens, d128 L6 | 1,240,194 | 15.7 | finer tokens |
| A9 | `mlp` (no cross-station mixing) | 173,345 | 8.8 | re-establishes the sequence-modelling value (audit: "CI includes zero") |

Design notes for A4-A7: sinusoidal PE (`sin/cos(pos / 1e4^(2i/d))`) as default because the station axis is a metric 1-D sequence; a relative-position bias (ALiBi-style per-head slope on |i-j|) is a cheap third option and usually beats absolute PE at length 96 - include it only if A4 vs A6' shows PE sensitivity. Heads = d/32. Attention over 97 tokens (96 + CTX) at batch 256 is tiny; nothing here needs flash attention. Pass `enable_nested_tensor=False` to `nn.TransformerEncoder` to silence the pre-LN warning. Weight decay .05 for transformer arms is the usual choice; keep 1e-4 for the GRU control (its tuned value) and report both for A4 if budget allows. Learning rate: **{1e-3, 2e-3}** for every transformer arm (OneCycle, same schedule); the audit showed the sign of the GRU-vs-tx comparison depends on it. Epochs: 30 as the protocol, plus a 60-epoch replicate for A4/A6 on the matched 13.8k-row set (1,620 steps at 30 epochs is short for a transformer). BatchNorm in the shared CNN is fine at batch 256 (unchanged from the control).

Which to keep if budget is tight: A0, A1, A4, A6, A8, A9 (6 arms). A1 vs A4 is the single cleanest "GRU vs transformer at equal parameters" pair; A0 vs A1 tells whether GRU width matters at all; A8 tells whether the CNN matters.

## 5. Velocity input (for replanning while moving)

### 5.1 Facts
- No existing row carries velocity information (every episode starts settled at rest; `ctx[:, 0]` = vx has IQR ~.07 m/s), so all velocity arms need **re-anchored samples** from the raw trajectories, and the read-out must be on re-anchored held-out rows (the designed-route dev metric is structurally blind to state: `LOG.md:170-171`).
- Everything needed is in the run folders: `state[:, 0]` = vel_body_x (also vy, roll, pitch, rates at 1-6), `pose`, `command_reference.npz` waypoints/stations/speeds, and `f104_n2_dataset.project(pose, wp)` (`:71-79`) maps each frame to a station `s_k` and lateral deviation.
- The proposal already carries momentum physics: `f104_n2_sampler.shape` (`:43-55`) projects the commanded speed through the acceleration cone `v[j] <= sqrt(v[j-1]^2 + 2 A_ACC ds)` from `v[0]`. At replanning time the first commanded speed must be set to the current speed **before** the corridor is cut, so the speed channel itself encodes what is reachable; the network then only needs `v0` to separate "commanded 4 m/s at station 0" from "actually doing 4 m/s".

### 5.2 Re-anchoring recipe (dataset build, numpy, ~10-15 min for 3 anchors x 35k episodes with 16 workers)
For each episode and anchor frame k (choose 3: progress fractions ~.15/.35/.55 of L, subject to: before the event frame, `|dev_k| < 1.0 m`, `parked[k] == False`, `vx_k > 0.3 m/s`):
1. remaining route = the projection point at `s_k` followed by the waypoints with station > `s_k`; speeds = reference speeds on that sub-route with `v[0] := vx_k` and the acceleration/deceleration cones re-applied (`f104_n2_sampler.shape` logic);
2. `X' = station_tensor(remaining route)` (96 stations over `L - s_k`; note `ds` shrinks, so along-grade is finer than in start-anchored rows - this is fine, `ds` is already route-length dependent);
3. `event_idx' = round((hwm - s_k) / (L - s_k) * 95)` if an event occurs after k, else -1; `unsafe/fail` unchanged;
4. `ctx' = [goal - pose_k (2), |.|, yaw_k, L - s_k]` + velocity features;
5. energy prefix `E' = E - E[k]`, `T' = T - t_k` on the new grid (from `energy_*.npz` or recomputed from `positive_work_kj_per_interval`).
Keep the group split; cap 3 anchors per episode (rows within an episode share the outcome and are strongly correlated); include the start-anchored row (v0 = 0) so the model also covers the standing start. Size: 3 x 35k = ~105k rigid rows x 5 x 96 x 32 fp16 = 3.2 GB (fits on the 5090 device-resident; the CRM set is 15k x 3).

### 5.3 How to feed `v0` - three arms plus a zero-parameter baseline
| id | mechanism | params added | comment |
|---|---|---|---|
| V0 | none: only the speed channel starts at `v0` (5.1, third bullet) | 0 | baseline; expected to capture most of the effect |
| V1 | `v0` (optionally vy, yaw-rate, pitch, roll: 5 chassis values) appended to the 5-d ctx -> Linear(6 or 10 -> 32) broadcast to all stations | 32-160 | matches how the chassis variant was fed; enters *after* the CNN |
| V2 | `v0` as a **7th input plane** (constant `v0/6` over the corridor, inserted before the ones channel and listed in `norm['cont_index']`) | 288 (first conv) | lets the CNN and tconv form local `v0 vs commanded speed vs grade` features - the interaction that matters for stalls and tilt |
| V3 | transformer only: `v0` inside the [CTX] token (A4-A7) | 0 extra | the natural tokenised form of V1 |
Recommend V0 + V1 + V2 on the GRU control and V0 + V3 on the best transformer. Gotcha: `gen_planner.RiskModel.score` must be generalised for a 7-channel model (it hard-codes `x[:, :4]` and appends ones); `SensorRiskModel` (`gen_planner.py:268-300`) is the template that reads `norm['cont_index']`.

### 5.4 Read-outs for the velocity arms
(i) within-group AUC on re-anchored held-out rows, per `v0` bin {0-1, 1-3, 3-6 m/s}, for each arm vs the velocity-blind V0 fitted on the same rows; (ii) start-anchored G_unsafe must not drop (regression check); (iii) a physics probe, not a metric: for fixed remaining routes, `P(unsafe)` as a function of `v0` should fall with `v0` on uphill entries (stall) and rise on cross-slopes (tilt) - if the probe is flat the extra input is being ignored. The closed-loop "small rigid moving-start check" the user asked for is outside this scout; note the collectors start every episode from rest, and the nav_v1 runner (`scripts/nav_runner.py`) is the only existing code path that plans while moving.

## 6. Energy head (second output next to risk)

### 6.1 Targets
Use the energy scout's `E (N,96)` joined by id (35,412 rigid ids cover production_v2/v3/v4; the 821 misc-source rows of station_ds_all have no target unless their run dirs are added; gen_v1's 9,000 local runs can be added the same way). Per-station **increments** `dE_j = E_j - E_{j-1}` (kJ, j >= 1; dE_0 = E_0), observed mask `m_j = isfinite(E_j) & isfinite(E_{j-1})`. Censoring is automatic: NaN beyond the furthest station reached, and for failed routes that is the pre-event prefix (section 3). Also carry `dT_j` (time increments) - the same head structure gives the time term the planner study wants.

### 6.2 Head and loss
- Head: per-station `e_j` from the same mixer features (Linear(2w or d -> 1)); non-negative increment `\hat{dE}_j = softplus(e_j)` (kJ); cumulative `\hat{E}_j = cumsum`. Route energy at planning time = `sum_{j <= j_max} \hat{dE}_j` with `j_max = floor((L - 2.5) / L * 95)` (never supervised stations 90-95 must not be summed).
- Loss (recommended): per-station Huber (delta 1) on `log1p` of increments, masked: `L_E = sum_j m_j * huber(log1p(\hat{dE}_j) - log1p(dE_j)) / sum_j m_j`, plus a route-level term `huber(log \hat{E}_{j*} - log E_{j*})` at the last observed station `j*` (this is what the planner ranks on). Total `L = L_surv + lambda * L_E`, lambda in {0.1, 0.3, 1.0}; report hazard AUC for every lambda - the first read-out of the energy study is whether the second head **costs** ranking quality (multi-task interference). Alternatives, cheaper but weaker: (a) route-level only, Gaussian NLL on `log E_{j*}` (no per-station signal, useless for re-anchored/MPC partial routes); (b) Poisson deviance on kJ increments (handles zeros, but the 49-55 kJ p99 tail dominates). Do not use MSE on kJ.
- Censoring for failed routes: supervise only `j < event_idx` (the prefix); never the total. Optional extra label "energy to the event" is exactly the prefix sum and is available for free.
- Normalisation: increments in kJ are already ~1 (median .7 rigid / 4.8 CRM); `log1p` keeps zeros at zero.

### 6.3 Read-outs
On held-out groups, goal-reached routes only (observed to `j*`): RMSE of `log E_{j*}` (rigid log10 sd .35 = the trivial predictor's RMSE ~.8 in natural log; CRM .11 -> .25); within-group Spearman between predicted and realised `E_{j*}` (rigid same-cell spread 44 % - a rank correlation of .6-.8 is a realistic target; CRM 15 % - expect .3-.5); **energy-pick regret**: among the routes the risk head calls safe (P below the group's median, or the true-safe siblings for an oracle-conditioned version), pick the lowest predicted energy and report realised kJ vs oracle and random; calibration by speed profile (energy must not collapse into a speed reader on rigid, where Spearman(E, speed) = -.07). Also report `T` the same way if the time head is trained.

## 7. Recommended sweep design and cost

### 7.1 Data design ("both worlds at matched size", the user's decision in `crm_night2_v1/LOG.md`)
- **Twin-matched sets**: the CRM training split (13,821 rows, 1,089 groups) and its rigid twins (same ids from station_ds_all's A1/A2 waves; 211 CRM rows have no twin - drop them from both, ~13.6k rows each). Same routes, same groups, same dev fold (`md5 % 5`), two worlds: architecture x world effects are then not confounded by route distribution. Held-out read-out: CRM val+test groups (111 groups, 1,414 routes) and their rigid twins; for rigid arms additionally the gen_v1 sibling arenas (zero-shot, within-group AUC on `gen_test_arm` rows).
- A second rigid tier at full size (33,840 rows) for the two best arms only, to check the ranking does not flip with 2.5x data (the scaling curve says rankings at 8k rows need not hold at 34k).

### 7.2 Stages, seeds, budget
| stage | arms | runs | 5090 (device-resident X) | 5090 (current pinned-host loader) | MI350X |
|---|---|---|---|---|---|
| A: architecture x world, 30 ep, lr {2e-3 GRU; 1e-3, 2e-3 tx}, **5 seeds** | A0 A1 A4 A6 A8 A9 (+A3 A5 A7 A8' if time) | 6 arms x ~1.5 lr x 2 worlds x 5 = ~90 | ~30 s/run (1,590 steps x 12-22 ms) -> ~45 min | ~60 s/run -> 1.5 h | ~2.5 min/run serial -> 3.7 h, or ~30 min packed 8-wide on mi3501x |
| A2: 60-epoch replicate for A4/A6, both worlds, 3 seeds | 12 | 12 min | 25 min | 1 h serial |
| B: energy head on A0 and the best transformer, lambda {0.1, 0.3, 1}, 2 worlds, 5 seeds | 60 | 30 min | 1 h | 2.5 h serial / 20 min packed |
| C: velocity V0-V3 on re-anchored sets (3x rows -> ~3x steps), 2 worlds, 5 seeds | ~40 | ~1.3 h | ~2.5 h | 5 h serial / 40 min packed |
| D: full-size rigid replicate of the top 2 arms, 8 seeds | 16 | ~20 min | 40 min | 1.5 h |
| dataset builds | re-anchored corridors (numpy, 16 workers) | | ~15 min | | |
Total ~3 h on the 5090 with a device-resident loader (~6 h with the current loader), or ~1.5-2 billed node-hours on the cluster if packed. Policy note (memory: train-on-cluster): these are 30-120 s jobs whose per-step cost is latency-bound; the 5090 is faster per run, the cluster is faster in aggregate only if 8+ runs are packed per node. Either way one script, one JSON row per run, as `f104_n2_train.py` does.

### 7.3 Seeds and statistics
Per-config sd of G_unsafe was .002-.010 at 8k rows (3 seeds) and ~.006 at 14k rows on CRM (5 seeds); 5 seeds give an SE of ~.003-.004, enough to resolve .01 differences, not .005. Report per-arm mean +- sd **and** the 5-seed ensemble score (the deployed quantity; the audit's tx "win" was an ensemble comparison); paired bootstrap over held-out groups for arm-vs-arm ensemble differences; 8 seeds for the final two contenders (stage D). Pre-register the selection rule (max held-out `W_unsafe` on the twin-matched held-out groups, ties by route-choice avoidable-unsafe) before looking.

### 7.4 Read-out table per run (one JSON row)
`arch, world, lr, epochs, seed, params, secs, n_fit` + dev and held-out `P/W/G_unsafe`, `G_fail`, `Gn/Wn` counts, route-choice `picked_unsafe / avoidable_unsafe / random / unavoidable`; energy arms add `E_rmse_log, E_spearman_within_group, E_pick_regret_kJ`; velocity arms add AUC per `v0` bin and the delta vs V0; for the planner study also log `mean ||dP/dX||` over held-out rows (input-gradient norm; the old model's 10x local-sensitivity anomaly was what broke MPPI's argmin, `memory f104 note`, and smoothness will matter for gradient-based route optimisation).

## 8. Gotchas (collected)
1. The only surviving architecture numbers are `night2_v1/sweep.json` (8,160 rows, one lr) and prose; the audited 8-seed/lr-1e-3 result cannot be re-derived (data deleted, script changed).
2. cuDNN GRU training is non-deterministic per seed; do not expect bit-reproducible rows.
3. `f104_n2_train.py:100-101` overstates the tensor size 10x; a device-resident fp16/fp32 X (2.7 GB for 36k rows, ~8 GB for the re-anchored set) halves the step time.
4. The existing `tx` is a conv-transformer hybrid without a final LayerNorm, ff = 2d, learned PE added on top of a scalar position channel, trained at the GRU's lr; it is not a tokenised transformer.
5. `G_` is designed-rows-only and blind to any per-row state; velocity read-outs must use re-anchored rows.
6. Deployed N2 fit = `D.fit | D.dev` where dev is designed-only, so 1,989 on-policy dev rows were excluded from the deployed fit (`f104_n2_deploy.py:21`) - relevant if you compare against N2 checkpoints.
7. `f104_n2_merge.py:16-17` forces every night-2 group to split=train; the rigid val/test splits in station_ds_all are night-1 groups only. For twin-matched held-out groups use the CRM split labels, not station_ds_all's.
8. CRM `unsafe == fail`; `event_idx` is the stall station; 68 % positives -> small `Gn`, noisy `G_`.
9. Energy: station 95 is essentially never observed (goal radius 2.5 m); sum predicted increments only to `j_max ~ 89`; failed routes: prefix only.
10. `energy_rigid.npz` has 35,412 ids (v2/v3/v4 raw folders); 821 misc-source rows and gen_v1 arenas need their own target build.
11. Any new input channel must be inserted before the constant-ones channel and registered in `norm['cont_index']`; `gen_planner.RiskModel.score` hard-codes 4 normalised channels and must be generalised (template: `SensorRiskModel`).
12. MI350X is 2.5-3x slower per step than the 5090 here; pack runs.
13. `nn.TransformerEncoder` warns about nested tensors with `norm_first=True`; pass `enable_nested_tensor=False`.
14. Profile-to-speed mapping (0: 2, 1: 4, 2: 6, 3: 2-6-2 m/s) is inferred from `route index % 4` and the per-profile outcome rates; confirm against `case.json`/`f104_episode.json` if a speed-stratified read-out is reported.

## 9. File index
- Model: `scripts/gen_riskmodel.py:10-12` (route_logit), `:15-57` (Net); twin `scripts/f104_n2_train.py:27-69`; ancestor `scripts/f104_night_train.py:30-72` (HazardNet, survival_nll), `:75-102` (auc, cell_auc, metrics), `:26-27` (dev_group).
- Trainers: `scripts/f104_n2_train.py:72-92` (Data, ctx variants), `:95-128` (train_one), `:141-162` (sweep main); `scripts/f104_n2_final.py:33-43` (group-fraction subsets); `scripts/f104_n2_deploy.py`; `scripts/crm_train.py:66-128`; `scripts/sensor_train_v2.py:30-49` (route_choice), `:52-127`.
- Data: `scripts/f104_n2_dataset.py:45-60` (station_tensor), `:71-79` (project), `:91-123` (labels), `:147-151` (npz keys); `scripts/f104_n2_merge.py`; `scripts/n2_energy_targets.py`.
- Planner side: `scripts/gen_planner.py:141-153` (corridors, geom_ctx), `:154-184` (RiskModel: mean of member logits), `:211-247` (plan: argmin); `scripts/f104_n2_sampler.py:43-55` (speed cones), `:58-63`, `:66-78`, `:81-93`, `:96-108`.
- Results: `artifacts/traverse/fdm_f104_50h_20260909/night2_v1/{sweep.json,scaling.json,onpolicy_eval.json,PLAN.md,LOG.md,REPORT.md,final/N2_meta.json,final/N2T_meta.json}`; `artifacts/traverse/crm_f104_v1/{REPORT.md,LOG.md,PLAN.md,train_v1/deploy/CRM_N2_deploy.json,train_v1/holdout/CRM_N2_holdout.json,train_v1/offline_heldout.json}`; `artifacts/traverse/fdm_f104_50h_20260909/sensor_v2/matched/matched_*.json` (5090 timings at 54k rows); `docs/progress.md:336-425`.
- Tonight: `artifacts/traverse/crm_night2_v1/{LOG.md,datasets/energy_rigid.npz,datasets/energy_crm.npz,scout/energy_scan/}`.
