# Module note: gb_data_audit (dynamics cache B1 + substep timing audit B2)

Files: `scripts/gb_build_cache.py` (265 lines), `scripts/gb_substep_audit.py` (591 lines). Written 2026-09-21 for
milestone B (PLAN.md B1, B2; conventions "Recorded action timing", "Cache contract for B", "Splits"). The first attempt
was interrupted by a usage limit after the cache build and the two hook smoke tests; the first resume finished the audit
(two small fixes, items 1-2 under "Fixes on resume"); the second resume (22:30, after the adversarial verification in
`VERIFY_gb_data_audit.md`) fixed its five minor findings (items 3-7), rebuilt the cache, re-ran the analysis stages and
both hook smokes, and corrected this note. Artefacts:

- cache: `B_tracker/cache_v1/` (39,235 npz + `cache_manifest.json` + `build_report.json` + `build.log`)
- audit: `B_tracker/audit/` (`selection.json`, `routes/`, `rigid/<id>/`, `crm/<id>/`, `determinism/`, `repr/`,
  `substep_audit.json`)
- self-tests: `B_tracker/selftest/cache/` (a `--limit 40` build, 120 episodes), `B_tracker/selftest/audit/` (the hook
  smoke runs of the first attempt, their re-runs `rigid_smoke_resume` / `crm_smoke_resume`, the stage logs, and the
  pre-fix copies `substep_audit_before_resume.json` / `representativeness_before_resume.json`)

No existing repo file was edited; no cluster job was submitted; nothing under `crm_night2_v1`, `crm_f104_v1` or
`fdm_f104_50h_20260909` was modified.

## Part 1: the cache (`scripts/gb_build_cache.py`)

### What it does

Reads every run directory of the CRM collection (`crm_f104_v1/collect_v1/runs`, domain 1) and of the two rigid
productions (`fdm_f104_50h_20260909/production_v3/runs` + `production_v4/runs`, domain 0) and writes one npz per
episode, keyed `<episode id>@crm` / `<episode id>@rigid` (the same id exists in both worlds), plus the manifest.

Per-episode npz (schema 3, contract in PLAN.md):
`z1 (T,17) f32` (the recorded `state`, preset `tire_normal_force_omega_pt`), `act (T,3) f32` [steer, throttle, brake],
`pose (T,3) f32`, `power (T,1) f32` kW, `desired_speed (T,) f32` (per-interval commanded speed from
`command_reference.npz`), `parked (T,) bool`, `stalled (T,) bool`, `hold_ok (T,) bool`, `route_waypoints (N,2)`,
`route_speeds (N,)`, `route_headings (N,)`, `route_stations (N,)`, `domain` (0 rigid / 1 crm), `group`, `status`,
`episode_id`, `source_dir`, `n_recorded` (frames before the cut), `cut` (bool).

- `stalled[k]`: frame k lies inside a run of at least 20 consecutive frames with |vx| < 0.3 m/s and throttle > 0.3.
- `hold_ok[k]`: the transition k -> k+1 has |act[k+1] - act[k]| <= 0.1 on all three channels (a 1e-7 float tolerance
  is added) and no throttle/brake flip (throttle > 0 at k and brake > 0 at k+1, or the reverse). The last KEPT frame is
  always False, also for cut episodes (whose recording continues past the cut): `hold_ok[k]` True implies that frame
  k+1 is in the cache, so a consumer pairing `hold_ok[k]` with `act[k+1]` / `z1[k+1]` never needs a special case.
- Both masks are computed on the full recording, then the arrays are cut to `--max-frames` (1200 = 60 s); the report
  counts the cut episodes per status and the dropped frames (`cut_by_status`, `frames_dropped_by_cut`,
  `stalled_frames_dropped_by_cut`).

`cache_manifest.json`: `schema: 3`, `episodes` (sorted keys), `domain_of`, `group_of`, `split_of`, `status_of`,
`n_frames_of`, `domain_names`, `z1_preset`, `act_columns`, `dt_s`, `blacklist`, `definitions` (the mask definitions in
words), `split_source`, and the whole `report` (also written separately as `build_report.json`).

Split: the twin group split only (`crm_night2_v1/datasets/twin_crm.npz`, arrays `group` and `split`: 1,089 train,
56 val, 55 test groups). A group absent from the twin file is accepted as `train` only with `--case-split-fallback`
(the default, because the module brief asks for it) and only if it is an `f104_v2_group_*` group whose `case.json`
declares `train`; every episode accepted this way is counted and its group listed in the report
(`case_split_fallback`) and the builder prints a warning, so this deviation from "the twin split is the only split" can
never be silent; with `--no-case-split-fallback` (or any other group) the episode is excluded and counted
(`excluded_counts.unknown_group`). In this build the fallback fired for 0 episodes: all 1,200 groups of both sources
are in the twin file, so the cache IS the twin split. Planner-suite
groups/ids (`f104_crm_eval_group_*`, `f104_g1_test_group_*`, `f104_pair_group_*`) are asserted absent, first on the
queued ids and again on the written set (by id and by group). CRM ids flagged in `collect_v1/qa.json` are excluded
(one id is flagged there, `f104_v2_group_0211_route_11`, but it is not in the local sync, so 0 were excluded).

### How to run

```
cd /home/harry/NeDM-traverse_mppi
PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python scripts/gb_build_cache.py \
    --out artifacts/traverse/generalist_20260921/B_tracker/cache_v1 --max-frames 1200 --workers 8
```
Defaults cover the sources, the twin split file and the qa file; `--limit N` takes the first N ids per source
(self-test); `--no-case-split-fallback` makes the twin file the only accepted split source. Consumers: `manifest["split_of"][key]` gives the split; readers must assert no `val`/`test` group is used
for fitting (the tracker fragment bank and the NRD train set, per PLAN.md).

### Build result (`cache_v1/build_report.json`; built 14:39 in 29.4 s, rebuilt 22:32 in 23.3 s with 8 workers, 0 errors)

The rebuild (second resume) only changed `hold_ok[-1]` of the cut episodes and added report keys; every number below
is identical in both builds (verified before the directory swap, see self-test 3).

| | episodes | frames kept | frames recorded | cut at 1200 | groups | val / test groups present | size |
|---|---|---|---|---|---|---|---|
| all | 39,235 | 19,617,117 | 21,200,751 | 3,065 | 1,200 | 56 / 55 | 1.80 GiB |
| crm | 15,235 | 6,534,757 | 6,588,724 | 192 | 1,200 | 56 / 55 | |
| rigid | 24,000 | 13,082,360 | 14,612,027 | 2,873 | 1,200 | 56 / 55 | |

Episodes per split: all 35,601 train / 1,833 val / 1,801 test (crm 13,821 / 713 / 701; rigid 21,780 / 1,120 / 1,100).
Statuses: crm goal_reached 4,867, soil_breakthrough_terminated 8,372, prolonged_blockage_terminated 1,985,
timeout 9, rollover 2; rigid goal_reached 19,230, prolonged_blockage_terminated 4,120, timeout 529,
terrain_bounds_exit 85, rollover 36. Blacklisted planner-suite episodes found in the sources: 0 (assertion passed).

`hold_ok` retention (hold_ok transitions / transitions with a successor frame, per regime):

| regime | crm | rigid | all |
|---|---|---|---|
| all transitions | 0.9810 | 0.9591 | 0.9664 |
| stalled frames | 0.99996 | 0.9985 | 0.9993 |
| moving frames | 0.9719 | 0.9535 | 0.9587 |
| brake-onset frames (k = first braking frame, transition k -> k+1) | 0.7892 | 0.7474 | 0.7534 |
| the transition INTO a brake onset (k-1 -> k) | 0.0 | 0.0 | 0.0 |

Recorded throttle/brake flip rate per transition: crm 0.0087, rigid 0.0204. Stalled frames: crm 2,119,933 (32 % of
kept CRM frames; 94 of them while parked), rigid 1,608,384 (12 %).

Reading: the transition that switches the brake on is never `hold_ok`, by construction (the follower's exclusive
throttle/brake switch makes every brake onset a flip). So recorded PID data contain no hold-transitions for the
brake-onset event itself; the brake response must come from hold-mode collection (PLAN B3), exactly as R22 anticipated.
Stalled frames are almost all `hold_ok` (the PID output barely moves while the vehicle is stuck), moving frames 95-97 %.

Cut at 1,200 frames (stored in the report since the rebuild): 3,065 episodes lose 1,583,634 frames (7.5 % of the
recorded frames; 476,748 = 30.1 % of the dropped frames are stalled): by status 1,349 prolonged blockage, 1,056 goal
reached, 538 timeouts, 83 breakthroughs, 30 bounds exits, 9 rollovers; rigid 2,873 episodes (1,299 blockage, 1,006
goal reached, 529 timeouts, 39 others; 30 % of the dropped frames stalled), CRM 192 episodes (83 breakthroughs, 50
blockages, 50 goal reached, 9 timeouts; 33 % stalled).

### Self-tests

1. `--limit 40 --out B_tracker/selftest/cache` (the first 40 ids of each of the three source roots = 40 CRM + 80 rigid
   = 120 episodes, 20 of them cut at 1,200; 0.2 s): ran clean, retention all 0.9729 / stalled 0.9997 / moving 0.9671 /
   brake onset 0.7515, flip rate 0.0114, fallback 0, held-out groups present val 0 / test 1 (the first resume's note
   said `--limit 5`, 75 episodes; that was wrong, the directory held a `--limit 40` build; re-run on the second resume
   with the fixed script).
2. Full build, then spot checks (first resume): four random episodes reloaded and compared with their sources: `z1`,
   `act`, `power`, `desired_speed`, `route_waypoints` bit-identical to `trajectory.npz` / `command_reference.npz`,
   `pose` equal to float32 precision, `status` equal to `outcome.json`, `group` equal to the manifest; `stalled` and
   `hold_ok` of `f104_v2_group_0000_op_00@crm` equal to a brute-force recomputation; `split_of` of all 39,235
   episodes equal to the twin file's split of their group (0 mismatches); 39,235 npz files on disk = manifest length.
   The verifier repeated this on 150 random episodes with an independent brute-force mask implementation (all equal).
3. Rebuild check (second resume, before the directory swap): the rebuilt cache against the previous one: `episodes`,
   `domain_of`, `group_of`, `split_of`, `status_of`, `n_frames_of` identical; every value of the `all/crm/rigid` report
   blocks identical; 400 random episodes identical on every array except `hold_ok[-1]`; all 3,065 cut episodes have
   `hold_ok[-1]` False (3,036 had it True before); npz count on disk = manifest length (39,235).

### Known limits

- Episodes longer than 60 s are cut; the tail of long blockage / timeout episodes is where the cut bites (numbers above).
- `hold_ok` is defined on the recorded PID transitions; the substep audit below quantifies what "held" means for them.
- The manifest's sha256 changes with every rebuild (the report carries the build time): it is now
  `21faa5129e2baa84...`; the tracker-environment self-test artefacts under `selftest/tracker/` and any NRD self-test
  checkpoint pinned the previous hash (`656052ebd8feb4cc...`) as provenance / resume guard, so resuming from those
  self-test checkpoints must start fresh (the real B4/B5 runs have not started).
- The manifest is one 10 MB JSON (fast enough to load; not designed for incremental updates: rebuild with the script).
- No terrain crops or map tokens are stored (by contract; `gb_crop.py` samples them on device).

## Part 2: the substep timing audit (`scripts/gb_substep_audit.py`)

### What it does

Drives episodes through the UNMODIFIED collectors with a read-only hook that records the clamped [steer, throttle,
brake] triple handed to `hmmwv.Synchronize` at every physics substep, then compares, per 50 ms interval k, the
applied path with the recorded `action[k]` (which is the follower output at substep 0 of interval k).

- Rigid: the frozen runner `traverse_fdm_rgbd_diverse_chrono.run_chrono` is imported through
  `gen_collect.import_runner` + `gen_collect.adapted_function` (the production stop policy, 3 additive hooks) and
  given a `frame_observer` whose `on_substep(scene, frame, sub, dt, action3)` stores the triple (25 per frame at
  2 ms). The manifest / runtime-fingerprint gates of `gen_collect.main` are bypassed exactly as
  `scripts/rigid_moving_collect.py` does; physics, driver and stop policy are untouched. Episodes: 20 of
  `production_v3` (one per group; 9 goal-reached in 12-24 s, 9 early prolonged blockages, 2 timeouts), routes taken
  from the recording's `command_reference.npz`, `--horizon-s 30`, 4 concurrent processes, OMP threads 1.
- CRM: `scripts/crm_collect.py` is imported unmodified; the frozen module it calls `make_driver` on is pre-imported
  and `make_driver` is swapped for one that wraps the real `ChPathFollowerDriver` in a proxy. The loop mutates the
  `DriverInputs` copy returned by `GetInputs()` (steering clamp) before `hmmwv.Synchronize`, so the proxy reads that
  copy back at the next call: this is the applied triple (50 per frame at 1 ms, plus the raw PID output and the sim
  time). Episodes: 10 of `collect_v1` (4 goal-reached, 3 soil breakthroughs with >= 1 s of stall, 3 early blockages),
  `--horizon-s 20`, `--crm-config crm_f104_v1/configs/crm_main.json` (1 ms, verified `physics_dt_s == 0.001`), each
  subprocess wrapped in `flock /tmp/luffy_crm.lock`, OMP threads 4, peak GPU memory sampled every 2 s from
  `nvidia-smi --query-compute-apps` for the subprocess.
- Determinism: one CRM episode (`f104_v2_group_0000_route_02`) driven twice with the unmodified `crm_collect.py`
  (subprocess, same arguments), `trajectory.npz`, `crm_extra.npz`, `command_reference.npz` compared byte for byte;
  the proxied audit run of the same episode is compared with run A as well (the hook must not change the physics).
- Representativeness: 3 recorded `collect_v1` episodes (the median-elapsed goal-reached, breakthrough and blockage
  episode among unused groups) re-driven locally at the full 120 s horizon with the unmodified collector, compared
  with the cluster recording on status, elapsed, goal_reached, max |xy diff| and the first frame the xy difference
  exceeds 1 cm / 10 cm / 1 m. The ten 20 s audit prefixes are compared with their recordings the same way.

Definitions in `substep_audit.json`: `range` = max - min over the substeps of one interval per channel;
`mean_minus_recorded` = mean over the substeps minus `action[k]`; `flip_within_interval` = throttle > 0 at one
substep and brake > 0 at another of the same interval; regimes `stalled` / `moving` / `brake_onset` (recorded brake
> 0 at k, <= 0 at k-1) / `pre_brake_onset` (the interval before) / `hold_ok` / `not_hold_ok` use the cache's mask
functions (`gb_build_cache.stalled_mask`, `hold_ok_mask`, `brake_onset_mask`) on the audited run's own recording.

### How to run

```
cd /home/harry/NeDM-traverse_mppi
export PYTHONPATH=src:scripts
PY=/home/harry/miniconda3/envs/nedm/bin/python
OUT=artifacts/traverse/generalist_20260921/B_tracker/audit
CFG=artifacts/traverse/crm_f104_v1/configs/crm_main.json
$PY scripts/gb_substep_audit.py --stage select --out $OUT --crm-config $CFG              # selection.json + routes/
OMP_NUM_THREADS=1 $PY scripts/gb_substep_audit.py --stage rigid --out $OUT --crm-config $CFG --rigid-horizon-s 30 --rigid-jobs 4
for st in crm determinism repr; do $PY scripts/gb_substep_audit.py --stage $st --out $OUT --crm-config $CFG --crm-horizon-s 20; done
$PY scripts/gb_substep_audit.py --stage analyze --out $OUT --crm-config $CFG              # substep_audit.json
```
`--stage all` runs everything in that order; the rigid and CRM stages may run concurrently (CPU vs GPU); finished
episodes are skipped on a re-run. The script also serves as its own worker (`--worker rigid|crm`, internal).

### Results (`audit/substep_audit.json`; stage logs in `selftest/audit/stage_*.log`)

Sanity (both worlds): the substep-0 triple equals the recorded `action[k]` exactly (max |diff| 0.0), the rigid
`on_frame` action equals the recorded one, CRM applied throttle/brake equal the raw PID output (the clamp touches
steering only), settle steering is 0. CRM expected `GetInputs` calls = (16 settle + n) x 50 in every run.
Intervals audited: rigid 9,829 (20 episodes: 11 goal-reached locally, 9 timeouts at 30 s; 1,796 stalled intervals,
58 brake onsets), CRM 3,137 (10 episodes: 4 goal-reached, 2 breakthroughs, 4 timeouts at 20 s; 744 stalled, 13
brake onsets).

Per channel, all intervals (range p50 / p90 / p99 / max; share of intervals with |mean applied - action[k]| > 0.05
and > 0.1):

| world | channel | range p50 / p90 / p99 / max | \|mean - rec\| p50 / p90 / p99 | > 0.05 | > 0.1 |
|---|---|---|---|---|---|
| rigid | steer | 0.004 / 0.048 / 0.096 / 0.096 | 0.002 / 0.024 / 0.048 | 0.0000 | 0.0000 |
| rigid | throttle | 0.000 / 0.036 / 0.133 / 1.000 | 0.000 / 0.016 / 0.071 | 0.0191 | 0.0051 |
| rigid | brake | 0.000 / 0.012 / 0.060 / 0.194 | 0.000 / 0.001 / 0.029 | 0.0015 | 0.0004 |
| crm | steer | 0.007 / 0.039 / 0.098 / 0.098 | 0.003 / 0.020 / 0.049 | 0.0000 | 0.0000 |
| crm | throttle | 0.006 / 0.047 / 0.115 / 1.000 | 0.002 / 0.024 / 0.064 | 0.0201 | 0.0061 |
| crm | brake | 0.000 / 0.000 / 0.055 / 0.217 | 0.000 / 0.000 / 0.027 | 0.0051 | 0.0026 |

Any channel and flips, per regime (share of intervals):

| world | regime | n | \|mean - rec\| > 0.05 | > 0.1 | range > 0.1 (any ch.) | range = 0 (all ch.) | flip within interval |
|---|---|---|---|---|---|---|---|
| rigid | all | 9,829 | 0.0206 | 0.0055 | 0.0193 | 0.156 | 0.0167 |
| rigid | stalled | 1,796 | 0.0000 | 0.0000 | 0.0000 | 0.287 | 0.0000 |
| rigid | moving | 8,033 | 0.0251 | 0.0067 | 0.0237 | 0.126 | 0.0204 |
| rigid | brake onset | 58 | 0.0690 | 0.0000 | 0.0517 | 0.000 | 0.3448 |
| rigid | pre brake onset | 58 | 0.0862 | 0.0000 | 0.0862 | 0.000 | 0.9655 |
| rigid | hold_ok (97.1 % of 9,809 transitions) | 9,521 | 0.0040 | 0.0004 | 0.0013 | 0.160 | 0.0050 |
| rigid | not hold_ok | 288 | 0.5660 | 0.1701 | 0.6146 | 0.000 | 0.3993 |
| crm | all | 3,137 | 0.0252 | 0.0086 | 0.0214 | 0.049 | 0.0105 |
| crm | stalled | 744 | 0.0000 | 0.0000 | 0.0000 | 0.167 | 0.0000 |
| crm | moving | 2,393 | 0.0330 | 0.0113 | 0.0280 | 0.013 | 0.0138 |
| crm | brake onset | 13 | 0.1538 | 0.0769 | 0.1538 | 0.000 | 0.0769 |
| crm | pre brake onset | 13 | 0.3846 | 0.3077 | 0.3077 | 0.000 | 1.0000 |
| crm | hold_ok (97.5 % of 3,127 transitions) | 3,048 | 0.0059 | 0.0000 | 0.0023 | 0.051 | 0.0013 |
| crm | not hold_ok | 79 | 0.7468 | 0.3291 | 0.7468 | 0.000 | 0.3544 |

The `hold_ok` / `not hold_ok` rows, the hold_ok share and the recorded flip rate cover only the intervals with a recorded
successor (`n_transitions` = intervals minus one per episode: 9,809 rigid, 3,127 CRM), exactly as the cache's retention
does; the other regimes cover all intervals (their statistics do not need the successor). The first resume had counted
the 20 / 10 successor-less last intervals as "not hold_ok" (rigid 308 intervals at 0.533 / 0.162 / 0.377, CRM 89 at
0.685 / 0.303 / 0.326); corrected on the second resume, no conclusion moved.

Recorded transition flip rate in the audited runs: rigid 0.0122, CRM 0.0086 (cache-wide 0.0204 / 0.0087).
Mean signed throttle deviation (applied minus recorded) over all intervals: rigid +0.0016, CRM +0.0022.

Reading:
1. The recorded action is a good 50 ms hold in 97-98 % of intervals: the mean applied triple is within 0.05 of
   `action[k]` on every channel for 97.9 % (rigid) / 97.5 % (CRM) of intervals and within 0.1 for 99.4 % / 99.1 %.
   Steering never deviates by more than 0.05 in the mean (the 2/s clamp bounds the drift per interval at 0.096-0.098,
   which is the observed maximum); throttle is the channel that moves (max range 1.0 in both worlds: full-throttle
   steps and throttle/brake switches inside an interval).
2. Stalled intervals are exact holds in both worlds: zero intervals with a deviation > 0.05, zero flips, throttle
   range 0 at the 90th percentile. For the NRD's stalled-window training the recorded transitions are
   hold-transitions.
3. The brake-onset event is where the record misrepresents the applied signal: the interval BEFORE the recorded
   onset already contains the brake at some substep (mid-interval onsets in 56/58 rigid and 13/13 CRM pre-onset
   intervals, a flip inside the interval in 97 % / 100 % of them; the brake is on for a median 40 % of that interval's
   substeps in both worlds, mean 42 % rigid / 55 % CRM, range 4-98 %; `mid_interval_brake_substep_share` in the JSON),
   and the onset interval itself often switches back to throttle (rigid 34 %). The brake response in recorded PID data therefore starts up to one interval earlier than
   the record says, with a partial-interval brake; it must be learned from hold-mode collection (PLAN B3, R22), which
   is also what `hold_ok` already excludes (retention 0 on the into-onset transition, Part 1).
4. `hold_ok` isolates the bad intervals well: among hold_ok intervals only 0.4 % (rigid) / 0.6 % (CRM) deviate by more
   than 0.05 and 0.04 % / 0 % by more than 0.1; among the 3 % it discards, 57-75 % deviate by more than 0.05 and
   17-33 % by more than 0.1, with a within-interval flip in 35-40 %.
5. Intra-interval refresh is the norm, not the exception: the applied triple is constant over the whole interval in
   only 15.6 % (rigid) / 4.9 % (CRM) of intervals, mostly because steering is refreshed every substep (CRM steering
   clamp active at 17 % of substeps). This is the baseline the external-control collectors must beat: their held
   drives must show range 0 on all channels (PLAN R18); the analysis functions of this script (`load_intervals`,
   `channel_stats`) accept any run directory that carries a `substep_actions.npz` in the rigid layout (`frame`,
   `sub`, `applied`) or the CRM layout (`raw`, `applied`, `substeps`, `settle_frames`).

Determinism (local, RTX 5090, CUDA): the two unmodified runs of `f104_v2_group_0000_route_02` (20 s horizon, goal
reached at 10.8 s) are byte-identical on `trajectory.npz`, `crm_extra.npz` and `command_reference.npz`, with equal
status / elapsed / frames / goal_reached; the proxied audit run of the same episode is byte-identical to run A
(state, action, pose max |diff| 0.0), so the `GetInputs` hook does not change the physics. Run B ran while another
process held ~5.7 GB of the GPU (total 7.4 GB sampled vs 3.4 GB during run A) and took 45 s instead of 26 s; still
byte-identical. Determinism holds locally on this GPU; the cluster check (LOG 16:05) holds on MI350X.

Representativeness (local re-drives of cluster recordings, unmodified collector, full 120 s horizon):

| episode | cluster status / elapsed | local status / elapsed | xy diff > 1 cm / > 10 cm / > 1 m at frame | max xy diff |
|---|---|---|---|---|
| f104_v2_group_0044_route_03 | goal_reached 13.00 s | goal_reached 13.25 s | 3 / 62 / 95 | 2.26 m |
| f104_v2_group_0009_route_00 | soil breakthrough 18.25 s | soil breakthrough 30.00 s | 0 / 38 / 254 | 1.32 m |
| f104_v2_group_0043_route_00 | prolonged blockage 34.0 s | soil breakthrough 12.3 s | 0 / 41 / never | 0.58 m |

The ten 20 s audit prefixes tell the same story: the xy difference passes 1 cm within the first 0-18 frames and
10 cm within 19-81 frames (1-4 s) in every episode; the goal-reached episodes still reach the goal (4/4, elapsed
within 2.2 s), breakthroughs 2/3 (the third had not broken through at 20 s), and the three blockages cannot be
compared at a 20 s horizon (the blockage rule needs 34 s; all three were timeouts locally with 0.5-3.7 m
divergence). Conclusion: local CRM runs are deterministic among themselves but are a different realisation from the
cluster's (CUDA vs HIP arithmetic; the divergence is visible from the first frames and reaches the metre scale
within 5 s on soft soil). Local CRM smokes are representative of the collector's behaviour class (same outcome kind
for clear goal-reached and clear breakthrough episodes, similar elapsed, same rtf class) but NOT of individual
recorded trajectories: prefix replays, branch anchors and byte-parity checks against cluster recordings must be run
on the cluster (which is what PLAN A4's two-pass design and the 16:05 decision already do), and local CRM
comparisons between arms must drive both arms locally. For rigid the same holds with smaller amplitude: the 20 local
rigid drives match the cluster recordings' status for all 9 goal-reached episodes (max xy diff 0.03-0.62 m; the
first > 1 cm difference between frame 0 and 90) and for the 2 timeouts, while 2 of the 9 recorded blockages were
driven around locally (21-24 m divergence) and 7 timed out at the 30 s horizon (the blockage rule cannot fire before
34 s). The local-vs-cluster difference is present at frame 0 already (`xy_diff_at_frame0_m`: the pose after the 0.8 s
settle differs by 0.6-1.4 cm for the three full CRM re-drives, 0.5-5.6 cm for the ten CRM prefixes and 0.1-8.3 cm for
the 20 rigid drives), i.e. the settle itself is machine-dependent; this supports the conclusion above.

Wall times and memory: rigid stage 242 s for 20 episodes at 4 concurrent (33-63 s per process including ~25 s of
start-up and scene build; 923 s summed, 1.88 wall-s per simulated s per process); CRM audit stage 400 s for 10
episodes (26-54 s each; rtf 0.44-0.54 excluding the 3-5 s soil build, 2.46 wall-s per simulated s including start-up;
one episode ran at rtf 0.28 while the GPU was shared); determinism 72 s; representativeness 123 s (28-63 s per
episode); analysis < 1 s. Peak GPU memory of a CRM collector process: 1,678-1,682 MiB (nvidia-smi per-process, 4.0 M
SPH particles); the machine's total went to 3.4 GB with the display and up to 7.4 GB when another process used the
GPU concurrently. Total for the first resume about 15 min of machine time (rigid and CRM overlapped). Second resume:
self-test build 0.2 s, full rebuild 23.3 s (8 workers) + 3 s verification, `--stage repr` 0.2 s, `--stage analyze`
0.3 s, rigid smoke 18 s, CRM smoke 324 s under a saturated GPU (would be ~15 s on a free one); about 14 min of session
time in total.

### Self-tests of the hooks (`selftest/audit/`, first attempt)

`rigid_smoke` (5 s horizon): 2,500 substep rows = 100 frames x 25, substep-0 triple vs recorded action max
|diff| 2e-8 (float32 rounding of the observer's float64 copy), `on_frame` action vs recorded 0.0.
`crm_smoke` (2 s horizon): 2,800 `GetInputs` calls = (16 + 40) x 50, substep-0 triple vs recorded 0.0,
`physics_dt_s` 0.001, 4,008,004 SPH particles. (The rigid substep-0 figure is 0.0 when compared in float32 as
`load_intervals` does; the 2e-8 quoted by the first resume was a float64 comparison.)

Re-run on the second resume with the fixed script (same case, route and horizon; `rigid_smoke_resume`,
`crm_smoke_resume`): rigid 18 s wall, `trajectory.npz` and `substep_actions.npz` byte-identical to the first attempt's
smoke, layout 100 x 25 rows, 2 ms substeps, steering step <= 0.004 per substep, sanity 0.0 / 0.0. CRM:
2,800 `GetInputs` calls = (16 + 40) x 50, layout 40 x 50, sanity 0.0 on every check, `physics_dt_s` 0.001, 4,008,004
SPH particles, and `trajectory.npz`, `crm_extra.npz`, `command_reference.npz` AND `substep_actions.npz` byte-identical to
the first attempt's smoke from eight hours earlier, although this run shared the GPU with two risk-model training jobs
at 100 % utilisation (real-time factor 0.009 instead of 0.36, 324 s wall instead of 15 s, 325 s including the lock
wait; peak process memory 1,678 MiB as before, GPU total 19.9 GB with the two training jobs). This is a fourth local
determinism data point (after run A / run B / the proxied run): the collector's result on this machine does not depend
on GPU contention or on the time of day.

### Fixes on resume

1. `GpuSampler` (the nvidia-smi polling thread) stored its stop flag in `self._stop`, which shadows
   `threading.Thread._stop()`; `join()` then raised `TypeError: 'bool' object is not callable` after the first CRM
   subprocess had finished (the subprocess itself was fine). Renamed to `_halt`; the CRM, determinism and
   representativeness stages were re-run from clean directories (the one affected audit episode was deleted and
   re-driven).
2. `analyze_world` indexed `peak_process_mib` on a missing-run placeholder entry; now `.get`.
3. (second resume, `VERIFY_gb_data_audit.md` items 3-6) `hold_ok` of the last kept frame is False also for cut
   episodes; the cache was rebuilt into a staging directory, verified against the previous build and swapped in.
4. `--case-split-fallback` / `--no-case-split-fallback` flag; the fallback is counted and its groups listed in the
   report (`case_split_fallback`), with a printed warning; the manifest carries `definitions` and `split_source`.
5. The report stores the cut breakdown (`cut_by_status`, `frames_dropped_by_cut`, `stalled_frames_dropped_by_cut`).
6. `analyze_world` restricts the `hold_ok` / `not_hold_ok` regimes, `hold_ok_share` and the recorded flip rate to
   intervals with a recorded successor (`n_transitions`); `channel_stats` stores `mid_interval_brake_substep_share`;
   `compare_with_recording` and the rigid block store `xy_diff_at_frame0_m`. `--stage repr` and `--stage analyze` were
   re-run (no drives; the pre-fix JSONs are kept under `selftest/audit/`); 84 leaves changed, all in the
   `hold_ok_share` / `recorded_transition_flip_rate` / `not_hold_ok` / `definitions` subtrees.
7. This note: self-test line (`--limit 40`, 120 episodes), the pre-onset wording, the frame-0 sentence.

### Known limits

- The rigid "substep-0 triple equals the recorded action" sanity is tautological (the recorder and the trajectory
  writer read the same inputs object at substep 0); the informative rigid checks are the layout assertions (25 rows
  per frame, 2 ms, the 2/s steering clamp respected at every substep) and the byte-identical local re-drive. For CRM
  the informative check is the clamp reconstruction the verifier did (applied steering = clip of the logged raw
  steering, throttle/brake equal at the same call index, so the proxy reads back a by-value copy of the inputs).
- The CRM proxy logs what the path follower returns; it is the applied triple only because the unmodified loop
  applies exactly that (after its steering clamp). For the external-control collectors (held / perturbed / policy
  modes) the applied triple is built by the loop, so their audit must hook the applied inputs (their own
  `substep_actions.npz` in one of the two layouts above), not the follower.
- Rigid audit runs are local drives, not the cluster's (rigid is deterministic per node only); the regime masks are
  computed on the local recording. The rigid brake-onset regime has 58 intervals and the CRM one 13: the onset
  rows are informative but their shares carry wide uncertainty.
- The representativeness set is 3 full episodes + 10 prefixes; enough to establish that local CRM is a different
  realisation, not enough to bound the outcome-class agreement rate precisely.
- The CRM stage runs strictly sequentially under the lock; another process using the GPU slows it (rtf 0.28 vs 0.53
  observed) but did not affect determinism in the one such run.
