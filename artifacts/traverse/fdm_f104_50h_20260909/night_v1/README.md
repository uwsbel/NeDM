# Night session 2026-09-10/11 — terrain-aware risk model for the f104 MPPI planner

Start here: `night_summary.png` (results), `LOG.md` (every step, number and correction, in order).

## Deployed model
`final/H1_full_s{0..4}.pt` — 5-seed ensemble of the per-station hazard model (`scripts/f104_night_train.py::HazardNet`).
Score a route with `scripts/f104_night_eval.py::score_ckpt` (route logit = log cumulative hazard; P = 1-exp(-exp(z))).
It is calibrated as P(unsafe = failed OR slid backwards), NOT as P(fail).

## Pipeline (scripts/)
f104_night_index.py      label every driven route; locate first danger event on the route; data-quality audit
f104_night_dataset.py    per-station tensors (elevation, along-path grade, cross-slope, commanded speed, mask)
f104_night_train.py      old + hazard architectures, all regularisers, dev-fold metrics
f104_night_sweep.py      run a list of configs in parallel
f104_night_eval.py / f104_night_test.py / f104_night_paired_boot.py   held-out evaluation, bootstrap CIs
f104_night_ablate.py     terrain-swap / route-length shortcut ablation
f104_night_brittle.py    lateral-sweep brittleness (spurious low-risk wells)
f104_night_candfeat.py, f104_night_pick3.py, f104_night_paired3_analyze.py   closed-loop 3-arm Chrono test (fixed speed)
f104_night_speedcand.py, f104_night_pick_speed.py, f104_night_speed_analyze.py  same, speed+geometry MPPI
f104_night_decomp.py     speed/terrain variance split — RETRACTED claim, kept for provenance (see docstring)

## Result folders
runs/        30 dev-sweep models (10 configs x 3 seeds) + dev metrics
paired3/     closed loop, fixed speed: OLD / A1 / NEW picks, Chrono outcomes, results.json
speedmppi/   closed loop, speed+geometry: same
examples/     5 good / 3 surprise / 4 failure picks from the speed+geometry test (examples.png, table.json)
test_eval.json, decomp.txt, speed_fallback.txt, audit.txt
episodes.json = labels with the correct throttle column (09-11 fix); episodes_v0_steering_col.json = what the deployed models were trained on

Everything is one fixed arena: held-out start/goal groups, not held-out terrain.
