# W: how fast does the measured history identify the world, and short-anchor training rows

Written 2026-09-22 (task W of PLAN S0). Read-out on the 56 validation groups only; the sealed test groups were not loaded
(no `--report-test` run). The 800 suite groups were not touched. No Chrono runs, no cluster jobs.

## Answer in one paragraph

Across the whole requested grid (decision time 0.5 / 1 / 1.5 / 2 / 3 / 4 s after a standing start x window 0.25 / 0.5 / 1 / 2 s)
the small history network separates rigid ground from soil with validation AUC 0.997-1.000 (mean of 3 seeds; worst single
seed 0.9963). The smallest cell in the grid, decision at 0.5 s with the last 0.25 s of history (k = 10, L = 5), already passes both bars:
mean 0.9994, worst seed 0.9992, seed-ensemble 0.9996 with 95 % group-bootstrap interval [0.9988, 0.9999]. A plain logistic
on window means and standard deviations gets 0.995 in the same cell. **Smallest window / decision time with val AUC >= 0.95:
0.25 s window at 0.5 s. With val AUC >= 0.99: the same, 0.25 s at 0.5 s.** The grid does not find a limit. An extra check
below the grid finds 0.991 (lower bound 0.982) at 0.1 s after the start with two frames of history, which is **before the
vehicle has moved** (median speed 0.001 m/s rigid, -0.003 m/s soil). At that point the engine speed alone separates the
worlds (0.99): 26.8 vs 28.0 rad/s at 0.05 s, and the first brake command is 0.034 vs 0.011. These are differences between
the two simulator setups at the moment the vehicle is released, not the soil acting on a moving vehicle. So the planned
gate ("the window identifies the domain with val AUC >= 0.95") passes for every approach length we could try, 0.5 s and
1 s included, but it does not show that the window carries soil information that matters for risk. Windows that leave out
the start (decision at 2-4 s, last 0.25 s only) still score >= 0.999, so the response while driving also identifies the
world on its own.

## Part 1: short-anchor training rows (`scripts/ci_short_anchors.py`)

What it does: for every episode behind `generalist_20260921/A_adapt/datasets/mixed_reanchor.npz` (15,024 twin episodes
per world; raw runs: soil `crm_f104_v1/collect_v1/runs`, rigid `fdm_f104_50h_20260909/production_v3/runs` +
`production_v4/runs`) and each decision frame k = 10 / 20 / 30 (0.5 / 1.0 / 1.5 s) it builds one row if the anchor
passes the same admission rules as `n2_reanchor_dataset.py`: before the labelled event, at least 12 m of route left, off
the route by less than 1 m, not parked, at least 3 waypoints left. The row is built the same way (remaining route from
the projection point, corridor from `f104_n2_dataset.station_tensor` on `crm_f104_v1/map_root`, 22-value context, event on
the remaining route, energy and time targets). History, mask and privileged context come from `ga_build_mixed.cut_episode`
itself (imported, not copied): 40 x 15 window cut at k, mask = action frame >= 0, so k = 10 has 10 valid steps. Ids are
`<episode>@<k>@<world>`. Keys, dtypes and trailing shapes are asserted against the headers of `mixed_reanchor.npz`;
uniqueness, the suite blacklist, split/group agreement with the reference file, the mask count and "last window step =
state in the context" are asserted too.

Commands (repo root, `PYTHONPATH=src:scripts`, `OMP_NUM_THREADS=6`, python = `/home/harry/miniconda3/envs/nedm/bin/python`):

    python scripts/ci_short_anchors.py --verify-mixed 400 --out artifacts/traverse/crm_improve_20260922/datasets/verify_mixed.json
    python scripts/ci_short_anchors.py --ks 10 20 30 --out artifacts/traverse/crm_improve_20260922/datasets/short_anchor.npz
    python scripts/ci_short_anchors.py --ks 40 60 80 --compare-mixed --out artifacts/traverse/crm_improve_20260922/datasets/anchor_k40_60_80.npz

Check that the construction is the same: (a) 400 random episodes per world rebuilt at the anchors the old builder picks:
the anchor sets match for all 800 episodes and all 3,078 rows are byte-identical to the stored rows in every key
(corridor, context, energy/time, labels, history, mask, privileged). (b) The k = 40 / 80 rows built by the new script
that also exist in the reference file: 38,997 rows, zero mismatches in any key.

Output `datasets/short_anchor.npz`: 89,998 rows, 1.58 GB, 83 s with 8 workers, peak memory 3.3 GB (build report
`datasets/short_anchor_build.md` / `.json`, log `short_anchor_build.log`).

| world | k | train | val | test | goal-not-reached rate train/val/test | unsafe rate train/val/test |
|---|---|---|---|---|---|---|
| rigid | 10 | 13,629 | 702 | 693 | 0.195 / 0.174 / 0.198 | 0.362 / 0.345 / 0.374 |
| rigid | 20 | 13,629 | 702 | 693 | 0.195 / 0.174 / 0.198 | 0.362 / 0.345 / 0.374 |
| rigid | 30 | 13,567 | 701 | 692 | 0.194 / 0.174 / 0.197 | 0.361 / 0.345 / 0.373 |
| soil | 10 | 13,629 | 702 | 693 | 0.672 / 0.724 / 0.736 | 0.673 / 0.724 / 0.736 |
| soil | 20 | 13,629 | 702 | 693 | 0.672 / 0.724 / 0.736 | 0.673 / 0.724 / 0.736 |
| soil | 30 | 13,551 | 701 | 690 | 0.672 / 0.723 / 0.735 | 0.672 / 0.723 / 0.735 |

Anchors turned away: none at k = 10 and 20; at k = 30, 64 rigid and 82 soil episodes (0.4 / 0.5 %), all for being more
than 1 m off the route. The goal/unsafe labels belong to the episode, so they barely change with k. Pairs present in both
worlds: 15,024 at k = 10 and 20, 14,924 at k = 30. Anchor speed p10/p50/p90: rigid 0.33/1.26/1.59 m/s at 0.5 s,
1.00/2.32/3.58 at 1 s, 0.91/2.15/4.38 at 1.5 s; soil 0.19/0.88/1.00, 0.71/1.66/2.10, 0.72/2.19/3.13.

Extra file `datasets/anchor_k40_60_80.npz` (88,797 rows, 1.61 GB; report `anchor_k40_60_80_build.md`): needed because the
reference file has anchors only at multiples of 40 frames, so there is no k = 60 (3 s, the current approach) row at
all, and its k = 80 rows are a selected subset (the old anchor choice keeps k = 80 mainly for episodes with few
admissible anchors: 3,010 twin pairs in train+val against 13,810 when every admitted anchor is kept). Same builder, same
rules; turned away 0.4-0.7 % rigid, 2.0-3.3 % soil (mostly off the route, 166 soil episodes had their event before 4 s).

Twin check (all 15,024 pairs): the commanded route is identical in both worlds for every episode; the settled start pose
differs by 3 cm (median; max 12 cm); the applied commands in the first 10 frames differ by up to 0.08 (median of the
per-episode maximum), because the follower reacts to the vehicle.

## Part 2: window probe (`scripts/ci_window_probe.py`)

Rows: (episode, k) pairs present in both worlds. Sources: k = 10/20/30 short_anchor.npz; k = 40 and 80 the k > 0 rows of
mixed_reanchor.npz (as briefed); k = 60 anchor_k40_60_80.npz. Window = last L frames of the stored window. Model per
cell: GRU 15 -> 32 with a linear read-out at the last step (the K1 probe network), 3 seeds, fixed 20 epochs, no early
stopping, normalisation from the training rows; trained on the training groups, AUC on the validation groups; test rows
dropped at load time. Baselines: ridge logistic on window means and standard deviations (30 features), and on the last
step alone.

    python scripts/ci_window_probe.py --full-coverage --common --ablate 10:5 10:10 20:20 60:40 --out artifacts/traverse/crm_improve_20260922/scout/W_window

584 s on the RTX 5090 (peak 0.95 GB GPU memory allocated, 2.3 GB RAM). Full tables incl. the variants:
`W_window_tables.md`; every number: `W_window.json`; run log `W_window_run.log`.

GRU validation AUC, mean of 3 seeds (in brackets: worst seed; * = window longer than the time since the start, so only
k frames are real):

| decision time (k) | twin pairs, train + val (val) | last 0.25 s | last 0.5 s | last 1 s | last 2 s |
|---|---|---|---|---|---|
| 0.5 s (10) | 14,331 (702) | 0.9994 (0.9992) | 0.9997 (0.9995) | 0.9999* | 0.9999* |
| 1.0 s (20) | 14,331 (702) | 0.9997 (0.9995) | 0.9998 (0.9998) | 0.9997 (0.9996) | 0.9996* |
| 1.5 s (30) | 14,234 (700) | 0.9987 (0.9984) | 0.9984 (0.9976) | 0.9998 (0.9997) | 0.9998* |
| 2.0 s (40) | 14,044 (693) | 0.9996 (0.9994) | 0.9991 (0.9989) | 0.9990 (0.9979) | 0.9995 (0.9993) |
| 3.0 s (60) | 14,039 (688) | 0.9997 (0.9996) | 1.0000 | 1.0000 | 1.0000 |
| 4.0 s (80) | 3,010 (151) | 0.9969 (0.9963) | 0.9983 (0.9969) | 0.9999 (0.9999) | 0.9983 (0.9974) |

Lowest 95 % group-bootstrap bound of the seed ensemble anywhere in the grid: 0.9936 (4 s, 0.25 s window, the small
selected k = 80 set); 0.9961 at 1.5 s / 0.25 s; 0.9988 at 0.5 s / 0.25 s.

Logistic on window means + standard deviations (validation AUC):

| decision time | last 0.25 s | last 0.5 s | last 1 s | last 2 s |
|---|---|---|---|---|
| 0.5 s | 0.995 | 0.995 | 0.995* | 0.995* |
| 1.0 s | 0.992 | 0.993 | 0.994 | 0.994* |
| 1.5 s | 0.980 | 0.987 | 0.992 | 0.992* |
| 2.0 s | 0.987 | 0.988 | 0.993 | 0.994 |
| 3.0 s | 0.953 | 0.974 | 0.988 | 0.997 |
| 4.0 s | 0.928 | 0.945 | 0.978 | 0.997 |

The last step alone (state and command at the decision frame, no history): 0.903 / 0.985 / 0.964 / 0.977 / 0.930 / 0.888
at 0.5 / 1 / 1.5 / 2 / 3 / 4 s. (K1 measured 0.64-0.67 for the settled state at rest.)

Checks on the population:
- Every admitted anchor at 2 / 3 / 4 s instead of the reference file's selection (`--full-coverage`): 0.9988-1.0000; at
  4 s with 13,810 pairs 0.9992 / 0.9999 / 1.0000 / 1.0000. The selected k = 80 subset is not what drives the result.
- Same 13,653 episodes at every decision time (`--common`): 0.9981-1.0000 everywhere.
- By anchor speed (validation, 0.25 s window): the slowest bin (< 1 m/s) is the hardest, 0.981-0.999; 1-3 m/s and > 3 m/s
  0.995-1.000.

Below the grid (not briefed; rows at k = 2 / 4 / 6 / 8 built with the same builder into a scratch file, commands in
`W_window.json` under `below_grid_supplement`): GRU mean 0.991 / 0.998 / 0.999 / 1.000 at 0.1 / 0.2 / 0.3 / 0.4 s using
all frames since the start (95 % bootstrap lower bounds 0.982 / 0.995 / 0.999 / 0.9998);
logistic on means/stds 0.998 at 0.1 s.

What carries the signal (GRU on channel subsets, 1 seed, validation AUC):

| channels kept | 0.1 s, 2 steps | 0.2 s, 4 steps | 0.5 s, last 0.25 s | 0.5 s, all 0.5 s | 1 s, all 1 s | 3 s, last 2 s |
|---|---|---|---|---|---|---|
| all 15 | 0.991 | 0.998 | 0.9994 | 0.9997 | 0.9997 | 1.000 |
| engine speed only | 0.990 | 0.840 | 0.986 | 0.918 | 0.991 | 0.925 |
| wheel spin rates only | 0.942 | 0.973 | 0.990 | 0.998 | 0.994 | 0.945 |
| body rates only | 0.903 | 0.947 | 0.971 | 0.990 | 0.999 | 0.998 |
| forward speed only | 0.743 | 0.802 | 0.915 | 0.931 | 0.980 | 0.891 |
| speed, pitch, throttle, brake | 0.839 | 0.856 | 0.961 | 0.977 | 0.995 | 0.998 |
| commands only | 0.815 | 0.781 | 0.733 | 0.852 | 0.960 | 0.965 |
| everything but engine speed and brake | 0.993 | 0.998 | 0.9994 | 0.9998 | 0.9991 | 1.000 |
| motion only (no commands) | 0.989 | 0.998 | 0.9994 | 0.9995 | 0.9986 | 0.9998 |

Reading: the signal is spread over many channels, so no single artefact can be removed to make the task hard. In the
first 0.1 s the vehicle has not moved yet, and engine speed (26.8 vs 28.0 rad/s at 0.05 s, 56.7 vs 58.0 at 0.1 s, rank AUC
0.84 alone at a single frame), the first brake command (0.034 vs 0.011, 0.78 alone) and tiny wheel and body motions
(rigid rolls back slightly more at release: -0.013 vs -0.002 m/s) already separate the worlds. Some of this can be the
soil holding the sunk wheels; some is certainly the two setups (1 ms soil physics vs 2 ms rigid physics, different
settling). From 0.25 s on, the physical launch response is large: at 1 s the wheels turn 2.4 m/s faster than the vehicle
moves on soil against 0.5 m/s on rigid ground, and the speed per engine speed is about half (0.0061 vs 0.0116).

## What this means for the plan

- The predeclared gate for a short approach (history window identifies the world with val AUC >= 0.95) is met by every
  window we could build, including 0.25 s windows at 0.5 s. It passes for both S2 approach lengths (0.5 s and 1 s), but
  it cannot choose between them: the choice should rest on the closed-loop numbers and on the risk-model accuracy by
  window, which PLAN W also names and this task did not measure.
- Because the worlds are told apart before the vehicle moves, a history-driven model trained on short-anchor rows can
  learn the simulator difference instead of the soil. Inside these two simulators that is harmless (the same difference
  is present when the planner is driven), but it is not evidence that the model senses soil. A control that removes the
  setup differences (for example the rigid world run with the soil world's 1 ms step and the same settling, or soil made
  very stiff) would need Chrono and was not run.
- Training use: `short_anchor.npz` can be merged with `mixed_reanchor.npz` directly (same keys, dtypes and shapes, no id
  collisions: new ids carry k = 10/20/30). `anchor_k40_60_80.npz` holds k = 60 rows (the 3 s decision frame of the current
  protocol, missing from the reference file) in the same format; its k = 40 / 80 rows repeat 38,997 reference rows
  exactly, so drop those ids before merging.

## Files

- `scripts/ci_short_anchors.py`, `scripts/ci_window_probe.py`
- `datasets/short_anchor.npz` + `_build.md/.json/.log`; `datasets/anchor_k40_60_80.npz` + `_build.md/.json/.log`;
  `datasets/verify_mixed.json`
- `scout/W_window.md` (this note), `scout/W_window.json`, `scout/W_window_tables.md`, `scout/W_window_run.log`
