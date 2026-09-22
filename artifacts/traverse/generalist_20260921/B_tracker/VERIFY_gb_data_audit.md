# Verification of module gb_data_audit (B1 cache + B2 substep timing audit), second pass

Adversarial re-check, 2026-09-21 ~22:45-23:15, of the state after the second resume (cache rebuilt 22:33, audit JSON
re-generated 22:33, note corrected 22:40), against `PLAN.md` (conventions, B1, B2), `NOTES_gb_data_audit.md`,
`scripts/gb_build_cache.py` (265 lines) and `scripts/gb_substep_audit.py` (591 lines). This supersedes the 18:05 pass
(whose six items 1-6 the resume addressed; see "Status of the earlier findings"). Nothing under the module's artefacts
was modified; every re-run went to `/tmp/gbv2/`. Rules respected: no cluster interaction; one rigid re-drive (of the two
allowed); one CRM re-drive of the hook smoke queued under `flock /tmp/luffy_crm.lock` (the one allowed; peak process
memory 1.7 GB, well under 8 GB); the cache rebuilds are CPU only.

Verdict: **pass with minor issues**. Every number I could recompute reproduces exactly, including a fully independent
recomputation of the whole cache report from the 39,235 source recordings with my own mask implementation; the
rebuild is deterministic; the hooks log the triple handed to `hmmwv.Synchronize`; the collectors are untouched; the
split is exactly the twin split with no held-out group leaked into `train`. The remaining items are hygiene / latent
assertions / wording, none of which changes a number or a conclusion.

## What was reproduced

### Cache (`cache_v1`, schema 3)

- Full rebuild with the shipped script into `/tmp/gbv2/cache_full` (21.9 s, 8 workers, 0 errors): the printed summary
  is identical to `cache_v1/build.log`; the manifest maps `episodes / domain_of / group_of / split_of / status_of /
  n_frames_of` and every report key except `built` / `wall_s` are identical; 3,000 random episodes compared array by
  array (all 21 arrays incl. `hold_ok`, strings included): 0 mismatches. `--limit 40` rebuild into `/tmp/gbv2/cache_l40`:
  identical to `selftest/cache` (120 episodes, all arrays, all report values). The builder is deterministic.
- Independent recomputation from the SOURCES (`/tmp/gbv2/recompute.py`: own run-length detection via edge indices,
  own transition/flip rule, `gb_build_cache` not imported) over all 39,235 run directories: episodes, frames kept
  (19,617,117), frames recorded (21,200,751), cut (3,065; by status 1,056 / 1,349 / 9 / 83 / 30 / 538), frames dropped
  (1,583,634; 476,748 stalled), stalled frames (3,728,317; 94 parked), flip transitions (323,282) and every retention
  numerator/denominator per regime and per domain (e.g. all 18,919,430 / 19,577,882; CRM stalled 2,109,718 /
  2,109,800; rigid brake-onset 93,895 / 125,635; pre-onset 0 / 147,690) are equal to `build_report.json` to the last
  digit. The `0.9810 / 0.9591 / 0.9664`, `0.99996 / 0.9985`, `0.9719 / 0.9535`, `0.7892 / 0.7474`, flip `0.0087 /
  0.0204` figures in the note follow.
- Manifest consistency: 39,235 npz on disk = `episodes` (unique, sorted); all five maps keyed on the same set; group
  parser consistent with the key; `domain` consistent with the `@crm/@rigid` suffix; `n_frames_of` in [104, 1200].
- Split rules: 0 mismatches between `split_of` and `twin_crm.npz`; the cache's val set of groups == the twin's 56 val
  groups and its test set == the twin's 55 test groups (set equality, not just counts); the twin file covers all 1,200
  groups (so the case.json fallback fired 0 times, `case_split_fallback.episodes = 0`, `split_source` says so); for
  every group, `case.json`'s declared split equals the twin split (1,089 / 56 / 55), so no consumer can be confused by
  the two sources disagreeing. Blacklist: 0 matches by id or group (asserted twice in the script). `qa.json`: the one
  flagged id `f104_v2_group_0211_route_11` is indeed absent from the local sync (group 0211 has 12 other episodes).
- Contract on 1,500 random npz: `z1 (T,17) f32`, `act (T,3) f32`, `pose (T,3)`, `power (T,1)`, `desired_speed (T,)`,
  `parked`, `stalled` bool, `hold_ok` bool, route arrays, `domain` in {0,1}, `group` / `status` equal to the manifest,
  `cut == (n_recorded > 1200)`, `T == n_frames_of`; `hold_ok[-1]` is False in all 1,500 (125 of them cut).
- Timing / axis conventions: both frozen loops record `action[k]` at `sub == 0` of interval k from the `inputs`
  object AFTER the loop's steering clamp and immediately before/at `hmmwv.Synchronize` (rigid runner lines 220-235,
  CRM collector lines 232-244), columns `[m_steering, m_throttle, m_braking]`; `state_fields[0] == vel_body_x_mps`
  and the 17-field preset is asserted; `dt_s == 0.05` asserted. The rigid `on_substep` hook (runner line 282-284) fires
  at every substep with `frame >= 0` after `hmmwv.Synchronize` with the same clamped triple, i.e. the applied action.
- No source tree was touched: `git status` clean for `scripts/crm_collect.py`, `scripts/gen_collect.py`,
  `scripts/traverse_fdm_rgbd_diverse_chrono.py`, `scripts/rigid_moving_collect.py`, `src/nedm`,
  `crm_f104_v1/configs`; no file under `crm_f104_v1`, `crm_night2_v1`, `production_v3`, `production_v4` has an mtime
  after 14:00 today. The five tracked modifications in `git status` are nav_v1 files from the 09-16 commit's follow-up,
  unrelated to this module.

### Substep audit (`audit/substep_audit.json`, generated 22:33:58)

- `--stage analyze` re-run on a copy (`/tmp/gbv2/audit_copy`): the only differing leaf is `generated`.
- Independent recomputation straight from the 30 `substep_actions.npz` + `trajectory.npz` files (own masks, script
  not imported): every regime row of the note's second table (n, > 0.05, > 0.1, flip within interval) for `all /
  stalled / moving / brake_onset / pre_brake_onset / hold_ok / not_hold_ok` in both worlds, `hold_ok_share` 0.9706 /
  0.9747, `n_transitions` 9,809 / 3,127, recorded flip 0.0122 / 0.0086, the pre-onset mid-interval brake counts 56/58
  and 13/13 with median substep share 0.40 / 0.40 (mean 0.416 / 0.551): all equal. Per-channel table (range and
  |mean - rec| percentiles, > 0.05 / > 0.1 shares, `range = 0` shares 0.155 / 0.049, signed throttle deviation
  +0.0016 / +0.0022): equal (the note's rigid throttle range p90 "0.036" is 0.03547 rounded half-up).
- Hook layout: rigid 25 rows per frame, frames 0..n-1 only, `t` step exactly 2 ms, max steering step 0.004 per
  substep (= 2 /s x 2 ms, never exceeded), `frame_action == action` exactly, substep-0 triple == `action[k]` in
  float32 exactly; CRM (16 + n) x 50 calls in all 10 runs, `t[800] = 0.8` (settle boundary), 1 ms steps, substep-0
  triple == `action[k]`; no substep in either world has throttle > 0 and brake > 0 simultaneously.
- CRM proxy semantics (the load-bearing claim that `applied` is what the loop handed to `hmmwv.Synchronize`): I
  reconstructed the loop's clamp from the logged `raw` steering (0 during the settle, then clip to +-2 dt per call) and
  it equals `applied[:,0]` with max |diff| 0.0 on all 10 runs; applied throttle/brake equal raw at the same call index
  (max |diff| 0.0). So the proxy reads back the by-value copy the loop mutated, not the driver's live state.
- Rigid re-drive: the 5 s hook smoke (`f104_v2_group_0000_route_00`, OMP 1 thread, 19.5 s wall) into
  `/tmp/gbv2/rigid_smoke` is byte-identical to `selftest/audit/rigid_smoke_resume` on `trajectory.npz` and
  `substep_actions.npz` (sha256 `423a1106...` / `98818d34...`), which are in turn byte-identical to the 14:44 first
  attempt's `rigid_smoke`. Rigid is reproducible on this machine across the day.
- CRM smoke artefacts: `crm_smoke` (14:44) and `crm_smoke_resume` (22:40) have identical sha256 on all four npz
  (`trajectory 70e0276e...`, `substep_actions 72807401...`, `crm_extra 6c78f081...`, `command_reference 82c0d29f...`);
  `worker.json` of the resume: 2,800 = expected calls, `physics_dt_s` 0.001, 4,008,004 SPH, 323.8 s wall, rtf 0.0088;
  `crm_smoke_resume.gpu.txt`: peak process 1,678 MiB, GPU total 19,913 MiB. Determinism block: run A / run B
  byte-identical (sha `983fe6ed...`, `b5415993...`, `98bf257f...`), proxied run vs run A max |diff| 0.0 on every array;
  run B under 7,430 MiB total GPU use vs 3,394 MiB for run A (the "5.7 GB foreign use" sentence), 45 vs 26 s.
- My own CRM re-drive of the same 2 s hook smoke (`/tmp/gbv2/crm_smoke`, queued 22:50 under the lock behind another
  session's `crm_collect_ext.py` smoke): see "CRM re-drive result" at the end.
- Representativeness block vs the note: three full re-drives (goal 13.0 -> 13.25 s, max xy 2.26 m, 1 cm / 10 cm / 1 m at
  frames 3 / 62 / 95; breakthrough 18.25 -> 30.0 s, 1.32 m, 0 / 38 / 254; blockage 34.0 s -> breakthrough 12.3 s, 0.58 m,
  0 / 41 / never), frame-0 differences 0.64 / 1.19 / 1.43 cm; the ten prefixes: 1 cm at frames 0-18, 10 cm at 19-81,
  frame-0 0.46-5.58 cm, goal 4/4 (elapsed within 2.2 s), breakthrough 2/3, blockages 3 timeouts with 0.48-3.69 m
  divergence; rigid: 9/9 goal statuses match (max xy 0.035-0.62 m, first > 1 cm between frame 0 and 90), 2/9 blockages
  driven around (24.4 m, 20.9 m), 7 timeouts, 2/2 timeouts match, frame-0 0.09-8.3 cm. All as written.
- Selection composition: rigid 9 goal-reached (12.6-23.8 s), 9 early blockages, 2 timeouts, one per group; CRM 4
  goal, 3 breakthroughs with 3.75-6.95 s of stall, 3 early blockages; determinism episode = first CRM pick; repr =
  median goal / breakthrough / blockage among unused groups. Wall times: rigid stage 241.5 s, CRM stage 400.4 s, CRM
  per-episode 26-54 s at rtf 0.44-0.54 (one at 0.28), determinism 26 + 45 s, repr 30 + 63 + 28 = 121 s, peak process
  1,678-1,682 MiB. Stage logs: one clean JSON line each, no warnings.
- The pre-fix vs post-fix JSON diff (`selftest/audit/substep_audit_before_resume.json` vs the stored file): 85 leaves
  present in both differ (84 + `generated`), all under `hold_ok_share`, `recorded_transition_flip_rate`,
  `regimes/not_hold_ok` and `definitions/hold_ok`; in addition 81 leaves are NEW (`n_transitions`,
  `mid_interval_brake_substep_share` x 8 regimes, `xy_diff_at_frame0_m` x 33 comparisons, two definitions), none
  removed. So the note's "84 changed leaves" is right for changed values but silent about the added ones.

### Status of the earlier findings (18:05 pass)

1 (self-test line) fixed in the note; 2 (pre-onset wording) fixed; 3 (successor-less intervals in `hold_ok_share` /
`not_hold_ok`) fixed in `analyze_world` (`has_succ` mask; numbers now equal my recomputation); 4 (case.json split
fallback) made explicit (`--case-split-fallback` default on / `--no-case-split-fallback`, counted, listed, warned;
fired 0 times); 5 (`hold_ok[n-1]` on cut episodes) fixed in the data (`hold_kept[-1] = False`) and documented in the
manifest `definitions`; 6 (frame-0 divergence, tautological rigid sanity) documented in the note. The 18:05 pass's
queued CRM re-drive never ran (`/tmp/gbv/crm_rerun` does not exist; its log is empty).

## Problems (none blocking)

1. Minor, hygiene / documentation: the audit's local re-drives include held-out groups: rigid
   `f104_v2_group_0001_route_09` (test), `f104_v2_group_0038_route_05` (val), `f104_v2_group_0040_route_04` (val); CRM
   `f104_v2_group_0001_op_00` (test). `select_episodes` scans the first 800 sorted ids without consulting the twin
   split. These are unmodified-PID timing drives with a read-only hook; nothing is fitted, selected or tuned on them,
   so this is not a statistical leak, but PLAN seals the 55 test groups for the B0 suite and the note does not say
   that four of the 33 audited episodes belong to sealed groups. Smallest fix: one sentence in the note; for any future
   selection, filter `scan_runs` rows to twin-`train` groups (no re-drive needed now, the timing conclusions do not
   depend on which groups were used).

2. Minor, latent assertions in `gb_build_cache.py` (non-triggering in this build): (a) keys are not asserted unique
   across the source roots; if `production_v3` and `production_v4` ever shared an id, `convert_one` would write the same
   npz twice from two workers and `episodes` would carry a duplicate (here v3 14,400 + v4 9,600 = 24,000 distinct,
   verified); (b) the npz count on disk is not asserted equal to `len(episodes)` and stale npz from an earlier build in
   the same `--out` are not removed, so a consumer that globs the directory instead of reading the manifest could pick
   up orphans. Smallest fix: `assert len({j["key"] for j in jobs}) == len(jobs)` after queuing, and
   `assert sum(1 for p in out.glob("*.npz")) == len(episodes)` before writing the manifest.

3. Minor, accepted contract deviation still in place: `--case-split-fallback` defaults to ON, so the builder can still
   admit a non-twin `f104_v2_group_*` group as `train` from `case.json` (counted, listed, warned, 0 uses now). PLAN says
   the twin split is the only split; the implementer keeps the default on because the module brief asks for it. Since
   consumers assert "no held-out group present" but cannot detect a group that is in neither twin list, the safer
   default is OFF (`--case-split-fallback` opt-in). One-token change; no data changes.

4. Trivial, wording: "84 leaves changed, all in ..." should read "84 values changed (all in ...) and 81 leaves added
   (`n_transitions`, `mid_interval_brake_substep_share`, `xy_diff_at_frame0_m`, two definitions)".

## Contract check (PLAN.md)

- Cache contract [R23, R35]: per-episode arrays, dtypes, `stalled` (|vx| < 0.3, throttle > 0.3, run >= 20) and `hold_ok`
  (<= 0.1 per channel + 1e-7, no throttle/brake flip, successor in the cache) definitions, `domain` 0 rigid / 1 crm,
  `group`, `status`, manifest keys, `schema: 3`, cut at `--max-frames 1200` with counts: satisfied. Retention reported
  per regime (stalled / moving / brake-onset, plus the into-onset transition) [R22]: satisfied.
- Timing convention [R18]: `action[k]` = follower output after the `Advance` at the last substep of k-1, read at
  substep 0 of k after the steering clamp, refreshed every substep; the audit measures exactly this refresh and reports
  the > 0.05 / > 0.1 shares, the intra-interval range and the flip rate; the "held drives must show range 0" baseline
  is stated for the ext collectors.
- Splits / blacklist [R1, R12]: twin split only (item 3 for the dormant exception), planner suites asserted absent by
  id and by group, val/test group sets equal to the twin's.
- Physics: CRM `physics_dt_s` 0.001 / 50 substeps, rigid 2 ms / 25 substeps confirmed in the logged data.
- Lock / GPU [R34]: every CRM subprocess under `flock`, peak process memory sampled and recorded (1.68 GB).
- Geometry source: not applicable (no crops in the cache, by contract). Task rows / checkpoints: not produced here.
- B2 scope: 20 rigid + 10 CRM through the UNMODIFIED collectors with read-only hooks (rigid: `gen_collect.
  adapted_function` with the production StopPolicy 24 / 2 / 8 s and production flags, only the manifest / fingerprint
  gates bypassed as `rigid_moving_collect.py` does; CRM: `crm_collect.main` itself with `frozen.make_driver` swapped,
  `proxies == 1` asserted), determinism (two unmodified runs + proxied run) and three re-driven cluster episodes:
  satisfied.

## CRM re-drive result

The 2 s hook smoke of `f104_v2_group_0000_route_02` (`--worker crm`, `crm_main.json`, OMP 4, same case / route / horizon
as the implementer's) ran at 22:55-22:56 under `flock /tmp/luffy_crm.lock` (350 s including ~260 s of lock wait behind
another session's `crm_collect_ext.py` smoke; 88 s of collector wall at rtf 0.033 with a foreign 8.2 GB / 99 % training
job on the GPU; peak process memory 1,678 MiB, 42 samples). Result: rc 0, 2,800 `GetInputs` calls = expected,
`physics_dt_s` 0.001, 4,008,004 SPH, layout (40, 50, 3), substep-0 triple vs recorded action max |diff| 0.0, and
`trajectory.npz`, `substep_actions.npz`, `crm_extra.npz`, `command_reference.npz` byte-identical (same sha256) to BOTH
`selftest/audit/crm_smoke` (14:44) and `selftest/audit/crm_smoke_resume` (22:40). That is a fifth independent local
determinism data point, taken from a different process tree under different GPU contention. The 18:05 pass's missing
CRM check is thereby closed.
