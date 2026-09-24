# Plan: raise soil (CRM) goal-reaching of the label-free shared planner from a moving start

Written 2026-09-22 08:30. Branch `crm_improve_v1` (from generalist_v1 db7a9e864). Artefact root K2 =
`artifacts/traverse/crm_improve_20260922/`. Previous effort: `artifacts/traverse/generalist_20260921/` (K1; REPORT.md,
PLAN.md, LOG.md). Cluster: CRM drives under CRM_ROOT = C = `/work1/dannegrut/harry/experiments/crm_f104_20260916`
(sub-tree `crm_improve/`), new code, rigid drives and training under G2 = `/work1/dannegrut/harry/experiments/
crm_improve_20260922` (source + `source_manifest.json`). Budget: 100 billed node-hours of my own jobs (account at 560.2
of 1500 at start); `K1/tools/my_spend.sh` style accounting from 2026-09-22 08:00.

## Starting point (measured 08:20)

Moving-anchor protocol of K1 (A5): straight 3 s approach at 3 m/s from each suite start, plan from the recorded
frame-60 state with CEM 4x64, drive. 800 paired suite pairs (600 fresh `f104_pair_group` + 200 reused). Soil:
same-row CRM specialist 83.0 %, shared history model H 83.9 %, oracle tag 84.4 %, rigid specialist 63.3 %. The same
CRM specialist family reaches 95.8 % from a standing start.

- 86/800 groups (10.8 %) fail under every soil-competent arm; 68 of those 86 were completed from a standing start.
  Oracle-of-arms 89.2 %; 13.1 % of groups are decided by the arm's choice.
- Branch-point speed step (route start speed minus vehicle speed at frame 60), 4,652 soil pass-2 drives:
  < -1.5 m/s: 16.7 % fail (n 132); -1.5..-0.5: 6.3 %; -0.5..0.5: 7.1 % (n 1,346); 0.5..1.5: 19.7 %; > 1.5: 37.1 %
  (n 1,319). Vehicle already slowed at frame 60 (vx < 2.5 m/s, uphill approaches): 36-43 % fail.
- Start-heading step of the picked route: weak and non-monotone (21 / 17 / 15 / 24 % for 0-5 / 5-15 / 15-30 / > 30 deg).
- Rigid/CRM balance is not the first suspect: the CRM-only same-row specialist (83.0 %) does no better than the
  shared models (83.5-84.4 %); the mixed training file is already row-balanced (58k rigid / 57k CRM).

## Hypotheses (each gets a direct test)

H1 Planner/controller handover: candidate routes that ask for a large speed change at the branch make the follower
   saturate throttle (or brake) on soil; the risk models never saw such states (training rows follow their own route
   profile), and the geometry-only specialists do not even see the vehicle speed.
H2 Approach damage: a long fixed approach (3 s, up to 6-8 m, 37 % of starts face > 12 deg in the first 12 m) spends
   soil margin before any decision. The user's proposal: a shorter window (0.5-1 s) may identify the domain early
   enough while keeping the vehicle on the flat start pad.
H3 Distribution shift in the risk model: decision states after an approach (moving, possibly on a slope, with a speed
   step) are absent from training (re-anchored rows are survivors that follow their route; branch rows are few).
   Fix = on-distribution data: approach + many continuations from the decision state, labelled in Chrono.
H4 Data balance / quantity between worlds (user question): test with batch-fraction and per-domain subsampling curves.
H5 Architecture (user question): a transformer over time (history tokens) and over stations may use the context better
   than the CNN-GRU with a GRU history encoder.

## Stages

S0 Diagnosis, offline, local (no cluster):
- D1 failure anatomy of the 3 s soil drives: all-fail vs choice-dependent groups; frame-60 state (vx, pitch, grade
  ahead, sinkage), first second after the branch (throttle, wheel slip from `crm_extra.npz`, sinkage growth), failure
  onset time; standing-start counterpart; recoverability.
- D2 cross-scoring: every existing model scores every driven pass-2 route of the same group from the same frame-60
  state (up to 6 per group, identical start state) -> within-group ranking AUC and calibration per model, and whether a
  speed-step term explains the misses. Diagnostic only: the suite is never used to select new variants.
- W window probe: domain-identification AUC and risk-model AUC as a function of history window length
  {0.25, 0.5, 1, 2 s} and decision time {0.5, 1, 2, 3 s} after a standing start, on identical recorded prefixes in both
  worlds (raw episodes, twin routes), val/test groups only for read-out.
- B/A offline arms (after S0 infrastructure): balance (CRM batch fraction 0.5 / 0.75; per-domain data fraction 0.25 /
  0.5 / 1.0) and architecture (CNN-GRU vs tokenised transformer vs joint time-and-station transformer; GRU vs
  transformer history encoder), 3 seeds, val groups.

S1 Planner handover fix (closed loop, 3 s protocol, frozen frame-60 states from K1 pass 1): candidate family options
   `free` (current), `cont` (speed profile floored at the deceleration ramp from the vehicle speed and capped at the
   acceleration ramp from it, so the route starts at the vehicle speed), `cont_head` (+ start heading within 20 deg of
   the vehicle yaw). Arms: H/cont, H/cont_head, S'_crm/cont_head. Only pass 2 is driven (800 each).

S2 Short approach (closed loop): approach lengths 1 s and 0.5 s (same straight 3 m/s approach route, `--horizon-s 1` /
   `0.5`) on the 800 suite pairs in both worlds; models retrained with short-anchor rows (decision frames 10 / 20) and
   the matching history window; arms: H_short/cont_head (and /free for attribution), S'_crm_short/cont_head.

S3 On-distribution data + retrain: 1,200 twin training groups (1,089 train / 56 val / 55 test, the only split) driven
   with the 3 s and the chosen short approach in both worlds, then 6 continuations per decision state (2 `free`, 2
   `cont_head`, 1 CEM pick of H/free, 1 CEM pick of H/cont_head), labelled with `ga_branch_dataset.py`. Retrain the
   winners of S0 B/A on re-anchored + short-anchor + branch + new rows; select on val groups; closed loop on the suite.

S4 Final: the best label-free arm under each protocol on the 800 suite pairs in both worlds (rigid must stay >= 99 %),
   plus the standing-start protocol for no regression (K1 A3: 93.8 % H, 95.8 % S_crm); REPORT.md; commit + push.

## Decision rules (predeclared)

- Primary: for each protocol, soil goal-reached of the new arm vs the K1 3 s baseline H (83.9 %) on the same 800 groups:
  one-sided 95 % lower bound of the paired group-bootstrap improvement > 0 on the 600 fresh groups; the 200 reused
  groups reported separately; terrain-clustered CI as robustness.
- A short-approach protocol counts as "adaptive" only if the history window it uses identifies the domain with val AUC
  >= 0.95 (W probe) and the rigid world stays >= 99 % goal reached under it.
- Secondary: gap to the standing-start soil specialist (95.8 %) and to the oracle-of-arms ceiling.

## Conventions carried over from K1

Suites and splits as K1 (twin group split; suite groups blacklisted in every builder; val groups for selection, test
groups sealed). CRM physics 1 ms (`configs/crm_main.json`); new collectors keep the `crm_collect` substring; torch never
inside collectors; CRM prefix replays are byte-identical across GPU types (K1 determinism check); rigid arms of a group
share one array task. Every closed-loop comparison hashes picks before driving. Plain language in all notes.
