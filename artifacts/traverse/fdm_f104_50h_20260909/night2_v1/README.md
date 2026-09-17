# Night 2 (2026-09-12) — more data, a transformer test, a state-input test, and a wider/faster MPPI proposal

Read in this order: PLAN.md (pre-registration, written before any result), LOG.md (every step and number),
night2_summary.png (figures), REPORT.md (the summary sent to the user).

## What changed on disk
cases_night2/        1,200 fresh start/goal groups, 12 designed routes each  (wave A1, training data)
cases_night2_onpolicy/  9,600 routes drawn from the planner's own proposal  (wave A2, training data)
cases_test_final/    the 184 frozen test groups (never collected in training, >= 6 m from every training group)
fresh_test_groups.json   the frozen test list and the rule that produced it
station_ds_fix.npz   night-1 routes with the corrected throttle column
station_ds_A1/A2.npz new waves (built on the cluster by f104_n2_dataset.py)
station_ds_all.npz   the merged training set
final/N2A_s*.pt      deployment ensemble trained on designed data only (old + A1)
final/N2_s*.pt       deployment ensemble trained on everything (old + A1 + A2)
testcand/            cached candidate sets for the test groups: night1 / night2 / fixed2 proposals
closed/              test 1: 184 held-out start/goals, speed free, 8 arms, 1,102 episodes
closed_ext/          test 2: 523 held-out start/goals at a fixed 2 m/s, model comparison, 929 episodes
closed_haz/          test 3: 300 hill/crater start/goals, 6 arms, 1,523 episodes  <- the decisive one
ext_test_groups.json / haz_test_groups.json   the frozen group lists for tests 2 and 3
night2_tests.png     the three tests side by side, on both safety metrics
scaling.json         data-scaling curve;  sweep.json  architecture x state-input sweep

## Scripts (scripts/)
f104_n2_cases.py     generate fresh groups from the frozen campaign generator's gates
f104_n2_sampler.py   the night-2 proposal (curvature-safe sine basis, free end speed, designed-route anchors)
f104_n2_onpolicy.py  wave A2 route generation
f104_n2_dataset.py   labels + per-station tensors, runnable on the cluster (correct throttle column)
f104_n2_merge.py     merge night-1 and night-2 tensors
f104_n2_train.py     GRU / transformer / no-sequence architectures x full / chassis / no state context
f104_n2_final.py     data-scaling curve;  f104_n2_deploy.py  deployment ensemble
f104_n2_cand.py / f104_n2_pick.py / f104_n2_submit.sh / f104_n2_analyze.py   the closed-loop test
f104_n2_figure.py    night2_summary.png

Cluster: campaign /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909; waves in production_v3 (A1) and
production_v4 (A2); closed loop in night2_closed_v1.
