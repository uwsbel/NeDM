# Scout map: route-risk model, dataset tensors, trainer (plan A, steps 1-3)

Read-only, 2026-09-21. Paths relative to `/home/harry/NeDM-traverse_mppi`; every claim was checked in code or by loading the npz files.

## 1. Model inputs, outputs, loss (`scripts/n2_arch_train.py`)

- Corridor `X`: stored `(n, 5, 96, 32)` float16, channels `[elev_rel, grade, cross, speed, valid]` (`scripts/f104_n2_dataset.py:45-60`): 96 stations, 32 lateral cells over +-6 m, elevation relative to the first station, along/cross gradients clipped to +-2, commanded speed repeated laterally, validity mask. The trainer standardises channels 0-3 on fit rows and appends a ones plane -> `(n, 6, 96, 32)` float32 on device (`n2_arch_train.py:200-214`); `--vplane` inserts vx/6 as a 7th plane (`:208-210`). Ranges: elev_rel -1.8..4.1 m, grade/cross +-0.71, speed 0-6 m/s.
- Context `ctx`: stored 22-d = `anchor_state.npz:state` (17) + `[goal dx, goal dy, |goal|, start yaw, route_len]` (`f104_n2_dataset.py:97,115-117`). The 17 state fields are `tire_normal_force_omega_pt` (`src/nedm/training/constants.py:3-54`): 0 vx, 1 vy, 2 roll, 3 pitch, 4 roll rate, 5 pitch rate, 6 yaw rate, 7-10 tyre normal forces fl/fr/rl/rr (N, ~3-12 kN), 11-14 spindle omegas, 15 engine speed, 16 motorshaft torque. Variants select columns: `geom` [17-21], `vel` [0,1,6 + geom], `chassis` [0-6 + geom] (`n2_arch_train.py:19`); standardised per column on fit rows (`:215-216`), then `Linear(nctx,32)+GELU`, broadcast to every station and concatenated with the 96-d station feature and a 0..1 position scalar (`:60,72-73`). In the twin datasets the state part is at rest (vx in +-0.09 m/s CRM, -0.2..0.33 rigid), so `vel`/`chassis` only carry information on re-anchored rows.
- Outputs: one hazard logit per station (`Heads`, `:43-54`); route logit = log sum softplus(haz) = cloglog of P(event anywhere) (`scripts/gen_riskmodel.py:10-12`). Optional energy/time heads (softplus kJ/m, s/m) with `--energy LAMBDA`.
- Loss: discrete-time survival NLL on `event_idx` (station of the first rollback/near-stop event, -1 = censored/clean) (`:228-233`), plus lambda x Huber-on-log1p energy/time increments masked to the clean prefix (`:236-246`). AdamW, OneCycle, batch 256, 30 epochs, grad-clip 5 (`:258-276`).
- Labels (`f104_n2_dataset.py:100-123`): `fail` = status != goal_reached; event = first rollback or 1 s near-stop under throttle after a 1 s settle; `unsafe` = not clean; event station = high-water-mark progress / route length x 95.

## 2. Grouping and splits

- `group` = start-goal case id (`f104_v2_group_NNNN`), `id` = route id (`..._route_KK` designed, `..._op_KK` planner proposals); `profile` = route index % 4 (0: 2 m/s, 1: 4, 2: 6, 3: 2-6-2; -1 for proposals) (`f104_n2_dataset.py:118-120`).
- Fit = `split == 'train'` minus dev fold `md5(group) % 5 == 0`; held-out = val+test groups (`n2_arch_train.py:132,200-202`). Metrics: pooled, within-group and same-speed AUC, lowest-risk pick failure per group (`:151-166`).
- Twin files: `twin_crm.npz` and `twin_rigid.npz` have identical `id`, `group`, `split`, `source`, `profile` arrays in the same order (15,024 rows; train 13,629 / val 702 / test 693; 1,089 train groups, 111 held-out); corridors identical to float16 rounding (max |diff| 0.002). Labels differ: fail 67.8 % CRM vs 19.4 % rigid; P(fail CRM | fail rigid) = 0.996, P(fail CRM | ok rigid) = 0.60. Rigid split labels come from the CRM case files (station_ds_all forced all of these groups to train, `scripts/f104_n2_merge.py:13-17`).
- Planner evaluation missions are absent from every training npz: the 200 CRM pairs `f104_crm_eval_group_*` (`crm_f104_v1/cases_eval/cases`, >= 2.03 m from every training group, `crm_f104_v1/LOG.md:26`) and the rigid `f104_g1_test_group_*` (`gen_v1/cases_test_f104`) have 0 overlap with twin/reanchor groups. Their drives exist as raw episodes (`crm_night2_v1/planner/eval_*`, `moving_v1/out/runs`) and must stay out of any new builder glob.

## 3. The anchor "history" and what per-step data exists

- `anchor_state.npz:history` (16, 24) is padding, not telemetry: `build_history(state[None], [[0,0,1]], pose[None], 0)` at frame 0 (`scripts/crm_collect.py:229-231`, rigid `scripts/traverse_fdm_rgbd_diverse_chrono.py:239-242`) repeats the t=0 state 16 times with a synthetic previous action (steer 0, throttle 0, brake 1) and ego (0,0,0,1) (`src/nedm/traverse/fdm_data.py:79-91`; columns `:29`). Verified on `crm_f104_v1/demo_v1/run_000` and `production_v3/runs/f104_v2_group_0000_route_02`: all 16 rows identical.
- No dataset holds a measured window. `reanchor_*.npz` ctx is the single frame `state[k]` (`scripts/n2_reanchor_dataset.py:71-72`), but `episode` and `anchor_frame` (k = 0 or multiples of 40 = 2 s) are stored, so windows can be cut from raw episodes.
- Raw per-step data is complete and local: `trajectory.npz` has `state (n,17)`, `action (n,3)` = [steering, throttle, braking], `pose`, `parked`, `positive_work_kj_per_interval` at 50 ms (`crm_collect.py:334-338`). CRM `crm_f104_v1/collect_v1/runs` 15,235 episodes; rigid `fdm_f104_50h_20260909/production_v3/runs` 14,400 + `production_v4/runs` 9,600 + `production_v2/runs` 11,412; the same route id exists raw in both worlds (checked `f104_v2_group_0000_route_02`, `_op_00`).
- Action convention: `action[k]` is the PID output at substep 0 of interval k, the same substep as `state[k]` (`crm_collect.py:208-216`); the PID keeps updating within the 50 ms but only that value is logged. `build_history` pairs `state[t]` with `action[t-1]` (`fdm_data.py:86-88`); a causal window ending at k uses `state[k-T+1..k]`, `action[k-T..k-1]`.

## 4. Paired rows and a domain-balanced mixed set

- Twin: row i of `twin_crm` is the same route as row i of `twin_rigid` (same id, corridor, group, split). Mixed file = concatenate both with `domain` int8 (0 rigid, 1 CRM) and ids suffixed `@crm`/`@rigid`; groups stay identical, so the md5 dev fold and val/test are paired automatically. `E/T` exist in both.
- Reanchor: same 15,024 episodes, 1,200 groups and per-group split in both worlds, but anchors differ (57,444 vs 58,424 rows; 44,990 (episode, frame) pairs coincide). Concatenation with a tag keeps groups paired; anchors need not be.
- Device size (float32 X): mixed twin 2.2 GB, mixed reanchor 8.5 GB (+0.4 GB for 40x20 history) - fits the 5090 (32 GB) and MI350X.

## 5. Minimal code changes

(a) Domain tag: in `Data.__init__` append a one-hot of `self.d['domain']` to `ctx` before standardisation (`n2_arch_train.py:206,215`); `build()` sizes `nctx` from `D.ctx.shape[1]` (`:261`). Balanced batches: replace the single `randperm` (`:268-271`) by two per-domain permutations interleaved bs/2 each. Metrics: `cell_auc` keys on `group` strings (`:152-157`), identical across domains, so held-out cells would mix rigid and CRM rows - key on `group|domain` or run `metrics` per domain mask.
(b) History branch: builder emits `H (n, T, dh)` and `hmask (n, T)` (T = 20-40) cut from `trajectory.npz` by (`episode`, `anchor_frame`); k = 0 rows get an all-masked window (startup). Deployable columns: state 0-6, 11-16 plus 3 actions (13+3); columns 7-10 are tyre contact forces, excluded by the plan's own "simulator-only forces" rule. Standardise per column on fit rows. In `GRUNet`: `self.hist = nn.GRU(dh, 32, batch_first=True)`, z = last hidden (padded steps zeroed or packed); `self.ctx` becomes `Linear(nctx + 32, 32)` and `forward(x, ctx, H)` concatenates z to ctx before it (`:60,69-75`); `predict`/`train_one` pass `D.H[k]` (`:253,271`).
(c) Domain head: `self.dom = nn.Linear(32, 2)` on z; `loss += lambda_dom * CE(dom, domain)` at `:272-273`; log accuracy for startup vs established rows. RMA-style teacher: train (a) first, then regress z onto the frozen tag model's 32-d ctx embedding.
(d) Deployment: `gen_planner.RiskModel` rebuilds the legacy `Net` and expects `cnn.*`/`head.*` keys (`scripts/gen_planner.py:154-166`); `n2_arch_train` checkpoints store `front.cnn.*`/`heads.h.*` and a stale `layers=2` (`:305-306`; verified on `stageC/crm_gru_vel_lr0.002_holdout_s0.pt`) - the trainer docstring's "RiskModel-compatible checkpoints" claim is false. A `model_kind`-aware loader is needed, and `score(X, ctx5)` (`:170-184`) plus `planner_arms.member_logits` (`scripts/planner_arms.py:153-162`) need a `hist` argument encoded once and expanded over candidates; `geom_ctx` (`gen_planner.py:149-151`) stays.

## 6. Training cost from the night-2 logs

- RTX 5090, twin (10,822 fit rows, 1,260 steps): CNN-GRU 12 s/seed uncontended (`stageA/crm_gru_*.json`), 36-46 s when sharing the GPU; tx96_2 16 s, tx128_4 50-56 s, tx96_4 70 s. Reanchor (41-42k fit rows, ~4,900 steps): GRU 99-175 s/seed, tx96_2 183-214 s (`stageC/*.json`, contended).
- MI350X (job 425457, 4 ensembles per GPU, `cluster_train/n2_train.sbatch`): twin GRU 22-215 s/seed (mean 29-72), tx96_2 13-51 s, tx128_4 18-65 s; reanchor tx96_2 45-118 s/seed; ~4 min per 5-seed ensemble (`LOG.md:20`). `crm_train.py` (host-pinned batches) 75 s/seed at 13.8k rows.
- Estimate for the shared model: mixed twin + history GRU ~25-40 s/seed on an idle 5090 (~3 min per ensemble); mixed reanchor 4-7 min/seed.

## Gaps / unknowns

- The twin npz builder is not in `scripts/` (built ad hoc on 09-18 from station_ds_crm_v1 + station_ds_all + `n2_energy_targets.py` outputs); a mixed builder must regenerate from those sources.
- Whether 2 s windows carry domain information at moving anchors is untested; the only motion signal seen so far is the survivorship-biased re-anchored vx (`REPORT.md:85-90`).
- `production_v2` (11,412 night-1 rigid episodes) has no CRM twin; including it as unpaired rigid data is a design choice.
- Startup rows (k = 0, all-masked history) are 26 % of reanchor rows and 100 % of twin rows; the startup/established balance must be set deliberately.
- MI350 timings are from 4-way GPU sharing; single-job numbers were not logged.

## What must change for the plan

- Step 3 is already satisfied by the existing files (paired groups, episode-following anchors, eval missions absent); the new work is the history builder, not re-splitting.
- The "17-D state + 3-D action" history in the task text contradicts the plan's deployability rule: drop the tyre-force columns 7-10 (13 + 3 inputs), or declare them privileged-teacher-only.
- Add a loader for `n2_arch_train`-style checkpoints in `gen_planner.RiskModel` before any closed-loop test; nothing trained by the night-2 trainer is loadable by the planner today.
- Per-domain metric masks (or `group|domain` cells) are mandatory in a mixed held-out set, otherwise the within-group AUC is meaningless.
- Runners that plan once at frame 0 always see an all-padding history; only `scripts/nav_runner.py` plans while moving, so the "established history" milestone needs that path or plan step 4's moving-prefix branches.
