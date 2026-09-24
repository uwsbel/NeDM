# Module note: `scripts/ci_planner.py` (stage S1, speed- and heading-continuous candidates at a moving branch)

2026-09-22 08:30-09:05. New files: `scripts/ci_planner.py`, `s1/run_s1_picks.sh`, `s1/compare_s1_picks.py`, the three
S1 pick sets under `s1/`, self-test outputs under `s1/selftest/`. No existing repo file was edited; K1
(`generalist_20260921`) was read only. No Chrono, no cluster jobs. GPU use under 1 GB.

## What it is

A wrapper around `scripts/ga_planner.py` (imported, not copied or edited). The command line is ga_planner's plus four
options; the loaders, the scorer (history encoded once per decision), the `--poses` decision states, the arms (A =
one-shot 256, B = CEM 4x64; tags and seeds from `planner_arms.arm_specs`) and every output file are ga_planner's.
During one run the wrapper swaps five module attributes and restores them on exit: `ga_planner.decision_for` (to get the
vehicle speed of each decision), `ga_planner.Ensemble` / `ga_planner.Scorer` (to add the new trainer's checkpoints),
`f104_n2_iter.plan_iter` (which the one-shot arm also goes through) and `planner_arms.route_json` (extra route-file
fields, only for the new families). The planning loop itself is a line-by-line copy of `f104_n2_iter.plan_iter` with
two hooks: every built candidate goes through the family transform, and every validity test goes through the family's
acceptance test.

Families (`--family`):

- `free`: the current candidates, unchanged. Same picks as ga_planner, bit for bit (T1, T1b).
- `cont`: every candidate's speed profile (the 9 designed anchors, every prior draw, every CEM draw, every CEM mean) is
  replaced by `v(s) = min(max(v_sampled(s), sqrt(max(v0^2 - 2*2.0*s, 0))), sqrt(v0^2 + 2*1.5*s), 6.0)`, s = distance
  from the route start. The route starts at exactly min(v0, 6), never asks for more than the 1.5 m/s^2 ramp up from v0
  and never for less than the 2 m/s^2 ramp down from v0. The element-wise max/min of profiles that each respect the
  accel/decel limits respects them too, and the sampled profile still ends in its stop-at-goal cone, so the route
  still stops at the goal. Each transformed candidate is checked again by the planner's validator; failures are dropped
  and drawing continues, as plan_iter already does. On 6,000 identical prior draws (20 decisions) the transform never
  changed the validator's verdict (4,498 valid both ways, 1,502 invalid both ways, all for curvature or arena).
- `cont_head`: `cont`, plus rejection of candidates whose first tangent differs from the vehicle yaw by more than
  `--max-head-deg` (default 20): `|wrap(headings[0] - yaw)|`. The anchors' side offset (off * sin^2) has zero slope at
  the start, so the anchors always pass. The base route's own start heading is 0.03-0.06 deg off the yaw.

Vehicle speed v0 at the decision (forward speed, state column 0), first rule that applies:
1. `--v0` (one decision only; refused for several) or a `v0` key in the `--poses` entry -> `explicit`;
2. the `--poses` entry's (or `--history`) npz window: column 0 of its last valid row -> `history_window`. This is what
   the S1 sets use. For the K1 A5 poses file it is the frame-60 state and equals `pass1_state.json` `vx` exactly on all
   800 CRM groups (max |difference| 0.0; also checked per group at plan time against the pass-1 trajectory's
   `terminal_state`, recorded as `v0_check` in every pick file);
3. a recorded run (`run` key / `--from-run`): `state[frame, 0]`, or `terminal_state[0]` when frame = number of rows (the
   pass-1 approach drives stop after 60 rows and keep the frame-60 state as `terminal_state`). ga_planner's own
   window-from-run builder stops at row n-1 in that case, one frame early; rule 3 does not;
4. the entry's `pass1_run` (the masked-history poses files have only this) -> as rule 3;
5. the case start pose with no override -> 0 (`standing_start`). Any other pose override with no speed is an error.
Negative v0 is set to 0. `--v0-min` (default 0) sets a lower limit on the speed the ramps start from; the recorded v0
is kept separately.

Output additions: every `picks/<g>.json` gets a `family` block (v0, where it came from, the cross-check, and per arm
the start speed, speed step, start heading, acceleration range and candidate counts built / rejected by the validator /
rejected by heading / accepted); `summary.json` gets a `family` block with the distributions; for `cont` / `cont_head`
the route file `meta` also carries `family`, `v0_mps`, `start_speed_mps`, `start_heading_err_deg`. Free-family route
files are byte-identical to ga_planner's.

Python API (for S3 data collection; the module imports torch through ga_planner, so build routes offline and ship
them - do not import it in a Chrono collector):
- `candidates(pose, goal, v0, n, seed, family='cont_head', max_head_deg=20.0, *, anchors=False, tries_factor=64,
  base=None, v0_min=0.0)` -> n routes in collector format (numpy `waypoints`, `speeds`, `stations`, `headings` +
  `meta` with family, v0, start speed, speed step, start heading, `route_sha256`, seed, draw count, branch pose, goal).
  These are the planner's own prior draws from `base_route(pose, goal)` with `default_rng(seed)` (with `free` they are
  exactly the planner's round-0 draws, T3b); `anchors=True` puts the valid designed anchors first. Write them with
  `gc_control.route_to_json`. Seeds must differ per decision state. Unlike `gc_control.sample_continuations`, this
  family has no 2 m/s floor mid-route (it is the planner's family, speeds down to 0.5 m/s).
- `plan_decision(ens, pose, goal, v0, hist, hmask, group, arm='B', family='cont_head', ...)` -> one pick with the
  command line's seeds (checked: equals the command-line picks for free and cont_head on 2 groups).

New trainer's checkpoints (model_kind `ci_train`): `scripts/ci_train.py` appeared at 08:34 (within the 45 min poll).
`CIEnsemble` loads them with `ci_train.load_ci_model(path, device)`; `CIScorer` encodes each member's history once per
decision (`ci_train.encode_history`), takes [vx, yaw rate] of the newest valid frame once (`ci_train.vel_from_history`,
for `geom_vel` members), and scores each batch with `ci_train.score(model, ck, X, geom5, z=..., vel=...,
domain_onehot=...)`. The planner still loads 40-frame windows; `ci_train` keeps each member's newest `hist_T` frames.
Ensembles that mix `ci_train` with other checkpoint kinds are refused. With no `ci_train` member, the ga_planner
classes run unchanged.

## Commands

    PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts OMP_NUM_THREADS=6
    $PY scripts/ci_planner.py --selftest artifacts/traverse/crm_improve_20260922/s1/selftest      # T1-T5, 61 s
    $PY scripts/ci_planner.py --selftest-ci artifacts/traverse/crm_improve_20260922/s1/selftest   # T5 only, 8 s
    bash artifacts/traverse/crm_improve_20260922/s1/run_s1_picks.sh                               # the 3 S1 sets, 20 min
    $PY artifacts/traverse/crm_improve_20260922/s1/compare_s1_picks.py                            # comparison below

One S1 set, as in `run_s1_picks.sh`:

    K1=artifacts/traverse/generalist_20260921/A_adapt; K2=artifacts/traverse/crm_improve_20260922
    $PY scripts/ci_planner.py --family cont_head --cases $K1/suite/cases --map-root artifacts/traverse/crm_f104_v1/map_root \
       --models "$K1/train/deploy_v1/H_deploy_s*.pt" --world crm --domain crm --arms B --poses $K1/a5/poses_crm.json \
       --out $K2/s1/picks_crm_H_conthead --task-root $K2 --verify 0

## Self-tests (`s1/selftest/RESULTS.json`, T5 also in `RESULTS_T5.json`; all passed)

- T1 `--family free`, first 3 groups of the K1 moving-start suite, models H, CRM world: picks equal to
  `K1/a5/picks_crm_H_named/picks` (route sha256, z_mean and pool index; z -8.4466 / -11.7026 / -10.8745), and the
  route files byte-identical to `K1/a5/picks_crm_H/routes/<g>__B.json` (same set, unrenamed; same PICKS_LOCKED hash).
- T1b in the same process, the copied loop with the free family against the original `f104_n2_iter.plan_iter` /
  `oneshot`: same pick and identical pool scores for 3 groups x arms A and B.
- T2 `cont`, 20 groups (every 40th of the 800), arms A and B, every one of the 10,260 scored candidates: first speed
  minus v0 = 0.0 (bound 1e-6); acceleration within [-2.0000000000000107, 1.5000000000000153] m/s^2; never above the
  ramp up from v0 or below the ramp down from v0 (margins 0.0); 0 invalid; every pool full (256 / 257); 22 % of the
  built candidates failed the validator (curvature), none because of the transform.
- T3 `cont_head`, same 20 decisions: all of the above, and start heading at most 19.998 deg; 47 % of built candidates
  accepted (9,678 rejected for heading, 1,934 by the validator); pools still full.
- T3b API: `candidates()` for 10 decisions x 3 families holds the same limits and is deterministic; with `free` its
  routes equal the planner's own prior draws (3/3 decisions). Draws needed for 6 routes: 6-12 (free and cont, identical
  because the transform never changes validity), 8-29 (cont_head).
- T4 standing start (case start pose, no history, 3 groups, cont_head, arms A and B): v0 = 0 from `standing_start`,
  first speed 0.0, start heading at most 19.7 deg, acceleration at most 1.5; with `--v0-min 0.5` the first speed is 0.5.
- T5 `ci_train` checkpoints (4 members from ci_train's own self-test, trained on synthetic data, copied into
  `s1/selftest/ci_ckpt`: gru/gru `geom_vel`, gru/transformer-history `geom`, tx96_2/gru `geom`, txjoint `geom_vel`):
  the command line plans 3 groups with cont_head (arms A and B; speed step 0, heading at most 19.4 deg); scorer
  logits vs `ci_train.score` with the raw window encoded inside: max |difference| 7.2e-7; history encoded exactly 4
  times per decision (once per member) after two scoring calls.
- Reproducibility of the built sets with the final file (the ci_train branch was rewritten while the H sets were being
  built): 20 groups per set replanned into /tmp, 20/20 identical route sha256 and z in each set.

## S1 pick sets (800 groups each, CEM 4x64, CRM world, frame-60 states of K1 A5 pass 1)

`s1/picks_crm_H_cont`, `s1/picks_crm_H_conthead` (models `H_deploy_s*`), `s1/picks_crm_Spcrm_conthead` (models
`Sp_crm_deploy_s*`). Layout as `ga_a5_pass2_tasks.py` expects: `picks_crm_<ARM>/tasks.json` + `routes/<g>__B.json`;
`tasks.json` paths relative to K2 (`s1/picks_crm_<ARM>/routes/...`; that script only uses the file name). Wall
394 / 406 / 403 s (0.49-0.51 s per group). Every group got a full pool (257 scored routes). Candidates accepted: 85.8 %
(H cont), 61.2 % (H cont_head), 61.0 % (S'crm cont_head). Locks (`PICKS_LOCKED.sha256`, routes by name + content):

    H_cont          31c22e7137cb210307a8cd002fab658d0ee78c983805ffe536eea848d070feeb
    H_conthead      c6f4dbe4d5b0ea6726da49af02e983e82aeefd06dd360e1286d1fdb3a903aadc
    Spcrm_conthead  0f7da72bbbfa155ec690ebfd0e4b7075fc3c5aa15fdd64d6274a75ced6cdb87a

Dry run of the pass-2 task builder into /tmp (`ga_a5_pass2_tasks.py --world crm --arms H_cont,H_conthead,Spcrm_conthead
--picks-dir K2/s1 --cluster-case-prefix generalist/suite/cases --cluster-approach-prefix generalist/a5/approach
--cluster-route-prefix crm_improve/s1/pass2/routes_crm`): 2,400 rows, 2,369 new drives (H_conthead repeats H_cont's
route in 14 groups; S'crm/cont_head repeats a route of the two H sets in 17). Nothing was written to K2 or the cluster.

### Speed step and start heading of the picks vs the K1 picks of the same models (`s1/compare_s1_picks.json`)

Speed step = the picked route's first speed minus the vehicle speed at frame 60; start heading = |route first tangent
minus vehicle yaw|. Numbers are counts of 800 (percent).

| set | step < -1.5 | -1.5..-0.5 | -0.5..0.5 | 0.5..1.5 | > 1.5 | abs step p95 / max | heading 0-5 | 5-15 | 15-30 | > 30 | heading median / p95 / max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| K1 H free | 47 (5.9) | 182 (22.8) | 226 (28.2) | 142 (17.8) | 203 (25.4) | 3.08 / 4.53 | 119 (14.9) | 154 (19.2) | 275 (34.4) | 252 (31.5) | 21.8 / 44.6 / 51.4 |
| S1 H cont | 0 | 0 | 800 (100) | 0 | 0 | 0 / 0 | 109 (13.6) | 170 (21.2) | 261 (32.6) | 260 (32.5) | 21.7 / 45.2 / 51.9 |
| S1 H cont_head | 0 | 0 | 800 (100) | 0 | 0 | 0 / 0 | 208 (26.0) | 382 (47.8) | 210 (26.2) | 0 | 9.7 / 19.3 / 20.0 |
| K1 S'crm free | 26 (3.2) | 127 (15.9) | 254 (31.8) | 174 (21.8) | 219 (27.4) | 3.20 / 4.53 | 151 (18.9) | 154 (19.2) | 257 (32.1) | 238 (29.8) | 21.0 / 45.2 / 52.0 |
| S1 S'crm cont_head | 0 | 0 | 800 (100) | 0 | 0 | 0 / 0 | 241 (30.1) | 338 (42.2) | 221 (27.6) | 0 | 9.7 / 19.2 / 20.0 |

The model's own risk of its pick (log-hazard z, P = probability of failure; pick time only, no driving):

| set | z mean | P median | groups with P > 0.1 | P > 0.5 | mean route time s | mean speed m/s | mean max side offset m |
|---|---|---|---|---|---|---|---|
| K1 H free | -9.38 | 2.0e-5 | 53 | 29 | 16.6 | 3.19 | 5.14 |
| S1 H cont | -9.15 | 2.0e-5 | 72 | 45 | 16.9 | 3.11 | 5.22 |
| S1 H cont_head | -8.10 | 2.8e-5 | 145 | 115 | 16.7 | 3.14 | 4.41 |
| K1 S'crm free | -10.15 | 8.8e-6 | 41 | 17 | 17.5 | 3.28 | 5.18 |
| S1 S'crm cont_head | -8.54 | 1.4e-5 | 146 | 113 | 17.9 | 3.16 | 4.24 |

Paired per group: H cont vs K1 H: median z change +0.013, higher in 56.5 % of groups; H cont_head vs K1 H: median
+0.20, higher in 69 %; the rise sits in the groups whose K1 pick started more than 20 deg off the yaw (430 groups:
median +0.64; the other 370: +0.05) and whose K1 pick started more than 0.5 m/s above the vehicle speed (345 groups:
median +1.00). S'crm cont_head vs K1 S'crm: median +0.30, higher in 67 %. No pick is the same route as its K1
counterpart (the speed profile changes in every group).

Reading: the continuous speed profile costs the models almost nothing (medians unchanged) except in the ~16 extra H
groups that now look dangerous to the model (P > 0.5: 29 -> 45); the 20 deg start-heading limit is expensive by the
models' own judgement: groups with predicted failure above 0.5 go from 29 to 115 (H) and from 17 to 113 (S'crm). The
limit removes about half of the large early side offsets (mean max offset 5.1 -> 4.3-4.4 m), because with the
three-sine side offset a zero start slope needs the higher sines, and those have 4x / 9x smaller curvature caps. The
models never saw the handover effect (hypothesis H1), so this prediction is not evidence against cont_head; the
closed-loop pass decides. It is a reason to keep H/cont in the drive list as the no-heading-limit reference.

## Known limits / things a reviewer must know

- Standing starts: with v0 = 0 the cont / cont_head routes start at 0 m/s. The frozen follower commands the speed of
  the nearest waypoint (`traverse_fdm_rgbd_diverse_chrono.py:218`, same line in `crm_collect_ext.py:220`), so a vehicle
  at rest on waypoint 0 would be told 0 m/s and not move (read from the code, not driven). Standing-start runs must use
  `--family free` or `--v0-min > 0` (e.g. 0.5; T4). Relevant for S4's standing-start regression check.
- Matching the vehicle speed does not match the throttle: at the branch a new follower starts with no memory of the
  approach, so with a route starting exactly at the vehicle speed its speed error, and therefore its throttle, starts
  near 0. During the last 0.5 s of the 800 pass-1 approaches the old follower held throttle median 0.24 (p5 0.13, p95
  0.55) to keep ~2.85 m/s on soil. Expect a brief coast after the branch instead of the old throttle / brake bursts; the
  first-second throttle and speed of the S1 drives should be checked (D1-style) before crediting or blaming the family.
- The v0 cap: a vehicle faster than 6 m/s gets a route starting at 6 (not at v0); nothing in the suite comes close
  (frame-60 vx 1.20-3.53 m/s).
- `cont_head` rejects draws; it does not shape them. The pools stayed full (61 % accepted at CEM 4x64, 38 % in the
  one-shot pool of T3), but a tighter `--max-head-deg` or short routes may run out of draws (`tries_factor` 8 per
  round, as the planner). A family whose side offset has zero slope at the start by design would keep more lateral
  freedom; not done here (it would change the free family).
- The swap of module attributes is scoped to one command-line run (restored on exit). Importing `ci_planner` alone does
  not change ga_planner.
- `ci_train` support was tested on synthetic-data checkpoints only (T5); real S2/S3 checkpoints should get one T5-style
  check (`--selftest-ci` with `ci_ckpt` replaced, or the scorer comparison in `selftest_ci`).
- The comparison is pick time only. No route of S1 has been driven.
