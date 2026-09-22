# Implementation plan v2: shared rigid/CRM risk model (A) and NRD-trained tracker (B)

v1 written 2026-09-21 14:05; v2 at 15:20 after the three-lens adversarial review (`scout/plan_review.md`, 38 findings,
5 blocking). Every accepted fix is marked [Rn] with the review finding number. Source: the handoff
(`PLAN_handoff.md`) and the scout maps (`scout/*.md`, `scout/critic.md`). Branch `generalist_v1` (from aa32bd8d3).
Artefact root K = `artifacts/traverse/generalist_20260921/`. Cluster roots: G = `/work1/dannegrut/harry/experiments/
generalist_20260921` (new code with `source_manifest.json`, all new-mode drives in both worlds, training, results);
C = `/work1/dannegrut/harry/experiments/crm_f104_20260916` (CRM_ROOT for unmodified-collector drives only);
R = `/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909/gen_v1` (unmodified rigid drives only, `gen_array.sbatch`).
Local machine luffy: RTX 5090, conda `nedm` (torch 2.12, pychrono incl. FSI, rsl_rl). Nothing under `crm_night2_v1`,
`crm_f104_v1` or `fdm_f104_50h_20260909` is modified.

## Budget in billed node-hours [R31]

Billing weights per raw node-hour: mi2101x 0.1, mi3501x 0.125, mi2104x 0.4, mi2508x 0.8, mi3008x 1.0, mi3508x 1.2.
Balance 489.9/1500 at start; hard stop for new launches at 590 (cap 100 for this effort). CRM drives cost ~0.4 billed
per simulated hour on the mixed partitions (night 1: 91.5 sim-h for ~37 billed); rigid drives ~0.02 billed per
simulated hour on mi2101x/mi2104x. Walltime caps: mi3501x/mi3001x 4 h, mi2101x/mi2508x/mi3008x/mi3508x 12 h,
mi2104x 24 h. Queue cap 50 tasks per user: every wave merges its CRM rows into ONE tasks file per world and rigid arrays
use <= 6 shards on mi2104x (128 cores) [R20]. `slurm_balance2.py` (on PATH) before every launch, logged in `LOG.md`.

## Conventions and contracts (binding for every module)

- New scripts only: `scripts/ga_*.py` (A), `scripts/gb_*.py` (B), `scripts/gc_control.py` (shared controller helpers),
  `scripts/crm_collect_ext.py` (CRM collector with new modes; the name keeps the `crm_collect` substring so
  `crm_worker.py` forwards `--crm-config`) [R2], `scripts/gen_collect_ext.py` (rigid). Specialist scripts and checkpoints
  are imported, never edited. `crm_collect_ext.py` asserts `--crm-config` is given and the first smoke's
  `outcome.json` must show `physics_dt_s == 0.001` before any cluster launch [R2].
- Rigid new-mode drives run from G: `G/source` (rsync of src/, scripts/, assets/ + `source_manifest.json`), G copies
  `gen_array_g.sbatch` / `gen_runner_g.py` with `GEN_ROOT=$G` and a `GEN_COLLECTOR` variable, FDM_RUNTIME_FINGERPRINT
  as in `gen_array.sbatch`; one `gen_collect.py --check-only` against G/source before the first rigid launch [R3].
- One geometry source per purpose. Risk-model corridors (A): OptiX static depth map (`crm_f104_v1/map_root`) in BOTH
  worlds (both specialists were trained on it); the rigid closed-loop harness therefore scores with the depth map too
  and night-2 heightmap numbers are history. Dynamics crops (B): the Chrono-frame v2 grid
  (`crm_f104_v1/grids/arena_f104_50h_v1/grid.npz`, rmse 5 mm), sampled on device at any pose by one module
  `gb_crop.py` (numpy + torch `grid_sample`, 8x8 at +-6 m and 16x16 at +-4 m, heights relative to the map height at the
  vehicle centre, /2 m, plus a validity flag); crops are never stored in a cache [R23].
- Deployable history channels: state columns 0-6 (vx, vy, roll, pitch, roll rate, pitch rate, yaw rate), 11-14
  (spindle omegas), 15 (engine speed) = 12, plus the 3 applied actions = 15 channels at 50 ms; columns 7-10 and 16 are
  teacher-only. Window T = 40 frames (2 s), causal, with a validity mask; startup rows have an all-masked window.
- Recorded action timing: `action[k]` is the follower output computed at the last substep of interval k-1 and applied
  from substep 0 of interval k; the follower refreshes inside the interval. External-control modes clip the command ONCE
  per frame (steer within +-0.1 of the previous frame) and write the identical triple at every substep; the substep
  audit must show per-interval max-min = 0 for held drives [R18]. For dynamics training, recorded PID transitions carry
  `hold_ok = |action[k+1]-action[k]| <= 0.1 per channel and no throttle/brake flip`; retention is reported per regime
  (stalled / moving / brake-onset) and brake response is validated on hold-mode data [R22].
- Physics: CRM 1 ms / 50 substeps (`configs/crm_main.json`); rigid 2 ms / 25 substeps.
- Splits [R1, R12]: the twin group split is the only split. 1,089 train groups fit models; the 56 val groups are used
  for every gate, checkpoint and variant selection; the 55 test groups are sealed for the final offline report and the
  B0 suite. Planner suites (`f104_crm_eval_group_*`, `f104_g1_test_group_*`, `f104_pair_group_*`) are blacklisted by
  id and group in every builder; branch rows keep their group's split; the tracker fragment bank and the NRD train set
  take the split from the cache manifest and assert no held-out group is present.
- Checkpoint contract for shared risk models (`ga_train.py` writes, `ga_planner.py` reads): dict with `model_kind
  ='ga_train'`, `cond`, `state`, `cin`, `nctx` (5 geometry + 2 tag if cond=tag), `zdim`, `hist_cols` (12 state + 3
  action indices), `hist_T`, `norm` (corridor mu/sd), `ctx_mu/ctx_sd` (geometry 5), `hist_mu/hist_sd` (15),
  `train_rows`, `split_hash`. Forward: `model(X (B,6,96,32), ctx (B,nctx), hist (B,T,15), hmask (B,T)) -> {'haz':(B,96)}`.
- Cache contract for B (`gb_build_cache.py` writes, `gb_train_nrd.py` / `gb_tracker_env.py` read): per-episode npz with
  `z1 (T,17) f32`, `act (T,3)`, `pose (T,3)`, `power (T,1)`, `stalled (T,) bool` (|vx| < 0.3 and throttle > 0.3, run
  >= 20 frames), `hold_ok (T,) bool`, `desired_speed (T,)`, route arrays, `domain` (0 rigid, 1 crm), `group`, `status`;
  `cache_manifest.json` with `episodes`, `domain_of`, `group_of`, `split_of`, `status_of`, `schema: 3`; episodes padded
  only up to `--max-frames 1200` (longer episodes are cut, count reported) [R23, R35].
- Task rows for the workers: `{id, group, case, route, run, tier, extra: [...]}`; CRM rows carry absolute paths in
  `extra` and are launched through `crm_worker.py` with `CRM_COLLECTOR=$G/source/scripts/crm_collect_ext.py`; rigid rows
  carry `arena, shard` and are launched through `gen_array_g.sbatch`.
- Every closed-loop comparison: picks hashed before driving; all arms of a group on one node for rigid; groups with all
  arms present; the decision statistic is the ONE-SIDED 95th percentile of the group-bootstrap distribution named in
  each milestone; the 9-cluster terrain-feature CI is a robustness report only [R7].
- Local GPU work is serialised: CRM smokes and the substep audit hold `flock /tmp/luffy_crm.lock`; the CRM smoke
  report records peak GPU memory; A2 training runs after the audit [R34].
- No torch inside any collector process: policies are exported to npz (obs normaliser, MLP, tanh, action affine) and
  evaluated in numpy inside `gc_control.py`; the numpy actor is checked against torch offline (max |da| < 1e-5) [R19].

## Milestone A: one adaptive risk model

**Decision rule [R7, R8, R26].** Primary arm H = `hist_aux` at startup (A3). Per domain: the one-sided 95th
percentile of the group-bootstrap distribution of (specialist-B goal-reached minus H-B goal-reached) on the 600 fresh
`f104_pair_group` groups is < 3.0 points. Rigid secondary with headroom: the 95th percentile of the paired ratio of
median elapsed time H/S_rigid is < 1.10, and a pick-agreement statistic (H agrees with its own-domain specialist more
often than with the other specialist, per domain). Established-history H (A5) is a secondary with its own bound. The
200 reused `f104_crm_eval_group_*` groups are reported as a separate stratum. Cross-specialist controls (S_rigid on
CRM, S_crm on rigid) are reported as the gap to close in goal-reached and time; if the rigid time gap between the two
specialists is < 2x the margin, the report states that rigid does not test adaptation.

**A0. Frozen paired suite.** 600 fresh cases `f104_pair_group_0000-0599` (`gen_cases.py --strata all`, seed
20260921104, margin 2 m against every existing case set) + the 200 reused CRM eval groups, identical in both worlds,
speed-free, CEM 4x64 (arm B). Reference arms in both worlds: S_crm = `CRM_N2_s0-4`, S_rigid = `N2_s0-4`, on depth-map
corridors. `ga_suite.py` writes the case set and the reference-arm picks/tasks per world (planner_arms formats).
Drives: CRM 800 groups x 2 models (existing 200 x S_crm reused), rigid 800 x 2 (~4 billed).

**A1. Mixed datasets [R30, R11].** `ga_build_mixed.py`: reanchor_crm + reanchor_rigid (its k = 0 rows are the twin
rows; twins are not concatenated; uniqueness of (episode, anchor_frame, domain) asserted) -> `mixed_reanchor.npz`
with `domain`, ids suffixed `@crm/@rigid`, groups unchanged, `hist (n,40,15) f16` + `hmask (n,40)` cut from the raw
local episodes (`state[k-39..k]` on the 12 columns paired with `action[k-40..k-1]`), plus a per-row `privileged (n,8)`
teacher context (mean over the window of tyre Fz x4, torque, and for CRM slip/sinkage from `crm_extra.npz`, zeros
elsewhere) [R29]. Blacklist assertion on ids and groups.

**A1b. Identifiability probe [R28].** `ga_domain_probe.py`: established probe on the (episode, frame) pairs present
in both worlds (same anchors, no selection confound) and, later, on the A4 branch rows; startup probe = per-column and
joint logistic AUC of the frame-0 12-column state; the all-masked window is 0.5 by definition and is not reported as a
finding.

**A2. Shared trainer [R9, R25, R29, R38].** `ga_train.py` (CNN-GRU only): `--cond none|tag|hist|hist_aux|hist_rma`,
`--domain-filter crm|rigid|both`, `--hist-drop 0.2` (random full-window masking during training so a masked history
at a moving anchor is in-distribution), `--split-eval val|test`. `hist_rma`: teacher = tag + privileged context
embedding; the history encoder is fitted to the teacher's embedding, then jointly fine-tuned. Metrics per domain x
{startup, established}: pooled and within-group AUC (cells keyed `group|domain`), lowest-risk pick failure, Brier and
10-bin ECE against `unsafe` (the fitted event; fail-calibration secondary with the unsafe-not-fail rate). 5 seeds,
lr 2e-3, 30 epochs; holdout arms on the 5090 after the audit (~6-8 h total), deploy ensembles on mi3501x.

**A3. Planner integration + startup closed loop [R36].** `ga_planner.py`: loader for `model_kind='ga_train'`,
scorer with z computed once per decision (history all-masked at startup), CEM 4x64 unchanged, picks/routes/tasks in the
planner_arms formats, both worlds from the depth map. Deploy ensembles: T (tag, oracle), H (hist_aux), P (pooled). Suite
drives: T and H on all 800 groups in both worlds; P on the 200 reused groups only.

**A4. Moving-prefix branch collection [R11, R13, R14, R15, R16, R33].** Anchors: 700 from train groups, 100 from
held-out groups (56 val / 55 test kept in their split). Per anchor: clean-moving (recorded episode cut at 2, 4 or 6 s,
60 %) or low-progress (onset = first 20 consecutive frames with |vx| < 0.3 and throttle > 0.3; cut in [onset-1 s,
onset+0.5 s]; CRM anchors pre-screened with `crm_extra.npz` so the sinkage increase at the cut is < 0.1 m; survivors and
onset-to-termination margins reported before launch, 40 %). Continuations: 3 per anchor from the night-2 route family
from the pose at F to the goal (speeds 2-6 m/s). CRM (amended 16:05 after the determinism check: 10/10 byte-identical between two collectors re-driven today on
MI350X, 9/10 identical to the original recordings, one blockage episode drifted 0.22 m): two-pass on MI350X partitions
only (mi3501x/mi3508x): pass 1 replays each prefix with horizon F as its own task row and records pose/history at F;
continuations are planned from the replayed state; pass 2 drives prefix+branch; anchors whose replay changes class
(stalled/moving) or pose (> 0.5 m) versus the recording are flagged and reported. The own-remainder drive is dropped. Rigid: in-job
two-pass on one node (`gen_collect_ext.py --mode branch_auto`: replay the prefix with horizon F in a subprocess,
keep the anchor only if the replayed class (stalled/moving) and pose (within 0.5 m) match the recording, sample the
3 continuations from the replayed pose, drive prefix+branch in 3 subprocesses; drop count reported). Driver swap:
rebuild the follower without `Initialize()` (points, Bezier, gains, look-ahead), constructor does `Reset()`, steering
continuity through the existing per-substep clamp on the first branch substep. The collector writes `branch_frame`,
`branch_route_sha256`, the branch pose and the prefix history into `outcome.json`/`trajectory.npz`.
Labels: `ga_branch_dataset.py` slices at `branch_frame`, projects onto the branch route only, starts the event clock
after a grace window (first frame >= 1 m from the branch pose or 3 s), labels goal-reached-from-branch as primary and
rollback as secondary, and reports label discordance among continuations of the same anchor before training.
Cost: 2,400 CRM drives ~37 GPU-h ~15 billed; rigid ~1.

**A5. Established-history closed loop [R9, R10].** Pass 1: a full route (straight start-goal line if it validates,
else `route_00`; heading = the lower-grade of the two) with `--horizon-s 3` in both worlds; analysis set = groups whose
frame-60 state is moving in both worlds (vx > 1 m/s, CRM sinkage increase < 0.1 m; excluded count reported). Arms plan
offline from the recorded pose at frame 60 (CEM 4x64), picks locked, pass 2 drives approach + branch. Arms (all trained
on the same rows = reanchor + branch, deploy mode): S'_crm and S'_rigid (`--domain-filter`, geometry ctx), H with the
measured history, H with the history masked (in-distribution via `--hist-drop`), P, T. CRM_N2/N2 remain secondary.
~6 arms x ~500 groups x 2 worlds (~12 billed CRM).

Read-out: per domain, startup and established, goal-reached with the one-sided bounds, elapsed time, the offline
held-out (val for selection, test for the final report) ranking AUC and calibration as the primary ranking read-out
[R27], suite-drive numbers labelled 'on selected picks', decision latency, cross-specialist controls.

## Milestone B: NRD-trained tracker

**Decision rule [R4, R5, R17].** Suite = all designed routes of the 55 test groups that exist in both worlds (~420),
stratified in advance: feasible stratum = recorded PID reached the goal in both worlds; infeasible = the rest.
Primary metric = station-based cross-track: for every reference station the distance to the nearest trajectory point,
unreached stations capped at 6 m (the off-route abort), per-route Winsorised mean; primary statistic = the one-sided
95th percentile of the paired-bootstrap distribution of the ratio policy/native-PID of the mean over the feasible
stratum < 0.90 (median reported too). Regression bounds (one-sided 95 %): paired completion difference > -3 points
(both strata); paired 'any unsafe event' (rollover, soil breakthrough, prolonged blockage, off-route) rate difference
<= +1 point; speed-error ratio < 1.10. Arms per route, same job/node, identical stop rules (the native StopPolicy plus
a 40 s any-throttle near-stop rule that fires after the native rule for PID): native PID (unmodified follower, the
milestone reference), held PID (callback = shadow follower output, actuator-matched control), policy.

**B0.** `ga_suite.py --tracking` writes the route list and strata; `gb_track_analyze.py` computes the metrics from
`trajectory.npz` + `command_reference.npz` and the decision statistics. The suite and its PID drives are frozen
before the first policy round; no policy drive on it feeds any training [R1].

**B1. Cache [R23, R35].** `gb_build_cache.py` (contract above) from CRM `collect_v1` and rigid `production_v3+v4`,
`--max-frames 1200`; `gb_crop.py` provides the on-device crops. Retention per regime under `hold_ok` reported.

**B2. Timing audit (local, before dynamics training) [R18, R34].** `gb_substep_audit.py`: 20 rigid + 10 CRM
episodes through the UNMODIFIED collectors with a substep hook (rigid `frame_observer.on_substep`; CRM `make_driver`
proxy logging `GetInputs`), under the CRM lock; reports the intra-interval path and the share of intervals whose mean
applied action differs from `action[k]` by > 0.05 / > 0.1, and the flip rate. Also the local determinism check: two
unmodified CRM runs of the same episode byte-compared, plus 3 recorded cluster episodes re-driven locally and compared
(status, elapsed) to learn whether local CRM smokes are representative.

**B3. External-control collectors + perturbed collection [R2, R3, R18, R19, R21].** `crm_collect_ext.py` and
`gen_collect_ext.py` with modes `native` (byte-identical to the unmodified collector), `branch` / `branch_auto`,
`pid_perturbed` (shadow follower output + bounded Ornstein-Uhlenbeck perturbation: steer +-0.15, throttle +-0.25,
brake taps p 0.05 lasting 0.5-1 s, 0.5 s correlation, clamped, one clip per frame), `pid_held` (shadow follower output
held), `policy` (numpy actor from an npz; observation padded at frame 0 with the rest state and the settle action
(0, 0, 1)). The recorded `action` is the held triple. Perturbed collection: 1,500 episodes per world on train-group
routes, sized so >= 2,000 brake onsets exist per world (~5 billed CRM).

**B4. Mixed NRD [R22, R23, R32, R37].** `gb_train_nrd.py` (fork of `traverse_wp2_train_map.py`): raw-crop token via
`gb_crop.py` at every step (training, rollout loss, evaluation, imagination), per-token domain one-hot (arms `tag`,
`notag`), group split from the manifest, per-domain batch fractions 0.5/0.5, domain-rebalanced channel weights,
`hold_ok`-weighted targets, resume (optimizer, scheduler, step) for the 4 h cap. Validation on the 56 val groups at 60
steps: signed displacement, vx and yaw-rate response, stalled vs moving windows (60 consecutive stalled frames),
brake-onset windows from B3 hold-mode data, teacher-forced vs fed-back, per domain. Gate: fed-back rollouts predict
escape (|displacement| > 1 m) in < 20 % of val stalled windows; moving-window displacement error < 25 %; brake-onset
vx-response error < 25 %. PPO uses the `tag` NRD (the env knows each fragment's domain); the NRD hash is recorded in
`policy_meta.json`.

**B5. PPO + Chrono evaluation [R6, R21, R24].** `gb_tracker_env.py` / `gb_train_tracker.py` (fork of `tracker_env.py`
/ `traverse_wp3_train_tracker.py`): one static v2 grid replaces the per-env map bank; observation = the 38-d route/state
block + the last 8 actions + the last 8 observable states; action squash spanning the full box (steer centre 0 scale
1, throttle and brake centre 0.5 scale 0.5) written to `policy_meta.json` and asserted equal across arms; fragments
start anywhere in [0, active_end - len] with frame-0 padding; PID-imitation warm start on observations normalised by
the runner's own normaliser (fitted on the imitation set first), actor-vs-PID MSE logged at PPO iterations 0 and 10;
reward as WP3 plus a progress term; rsl_rl resume from an argument. Chrono evaluation on the B0 suite through the ext
collectors (both worlds, three arms). Improvement round: policy failures harvested on ~1,000 TRAIN-group routes per
world only, added to the cache, NRD refit, PPO retrained, B0 re-driven [R1]. Hybrid residual (PID + NN) only as a
labelled fallback.

## Order

Wave 1 (now): A0 suite + reference picks + drives (C and R, unmodified collectors); A1, A1b (local CPU/GPU-light);
B2 audit and CRM determinism/representativeness checks (local, lock); the two ext collectors with local smokes in
both worlds; `gb_crop.py`, `gb_build_cache.py`; A2 and B4/B5 forks written and self-tested on small data; G/source
shipped with the manifest and the rigid `--check-only`.
Wave 2: cluster CRM determinism check (10 episodes) -> A4 launch (one tasks file per world); B3 perturbed
collection (same tasks file as A4 for CRM); A2 holdout arms locally, deploy ensembles on mi3501x; A3 picks + drives.
Wave 3: A4 labels -> hist retrain -> A5; B1 cache + B3 data -> B4 training and gate -> B5 PPO -> B0 evaluation.
Wave 4: B improvement round; REPORT.md, figures, commit, push.

Budget estimate (billed): A0 4, A3 5, A4 16, A5 13, A training 2 = 40; B3 5, B4 6 (three 4 h mi3501x runs = 1.5 raw
h-equivalents each... counted at 0.125/h: ~2), B5 PPO 2, B0 evals 8, round two 10 = ~27; total ~67 of the 100 cap.

## Stop rules and fallbacks

- Balance reaches 590: no further launches; report what exists.
- Determinism check fails on the cluster (CRM prefix not byte-identical): A4 CRM switches to the rigid two-pass design
  (pass 1 per anchor as separate task rows, continuations planned from the replayed pose), cost +10 %.
- A1b established AUC < 0.7 on coinciding pairs: continue, report; if H is not within margin after A4, report the gap
  with T as the ceiling.
- B4 gate fails after two variants (`tag` with 8x8 and 16x16 crops): no PPO; report the validation numbers.
- Any ext collector whose `native` mode is not byte-identical to the unmodified collector on the local check fails
  review and is not shipped.
