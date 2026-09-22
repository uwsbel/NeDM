# Verification of module gen_ext (`scripts/gen_collect_ext.py`, `scripts/gen_runner_g.py`, `scripts/gen_array_g.sbatch`), round 2

Adversarial verifier, 2026-09-21 late evening (round 2, after the fix round and the resumed implementer round).
Files under test: `gen_collect_ext.py` sha256 `b9f36a024570...` (804 lines), `gen_runner_g.py` `b47c9fd53c33...`,
`gen_array_g.sbatch` `e3bcb4d8d742...`; imported `gc_control.py` `683f24ad3992...`, `gen_collect.py` `b6ba062260aa...`,
`f104_n2_sampler.py` `c80184491ba5...` (all re-hashed here, equal to the report). Read: `PLAN.md`,
`NOTES_gen_collect_ext.md`, the previous `VERIFY_gen_ext.md`, the three module files, `gen_collect.py`, the frozen
loop (`run_chrono`, `make_driver`), `gc_control.py` (hold_clip, OUPerturb, sample_continuations, make_follower,
history_blocks, PolicyObs), `crm_collect_ext.py` (near-stop rule, branch convention, flags), `ga_build_mixed.py`
(window convention), `gb_tracker_env.py` (obs_layout), `gen_runner.py`/`gen_array.sbatch` (diff), the
`selftest/rigid_final` scripts and analysis JSONs. Budget used: 2 rigid Chrono processes (one replay-only
`branch_auto`, one `pid_perturbed`), 0 CRM runs, no GPU, no cluster submission. Temporary outputs under `/tmp/vg2/`.

Verdict: **pass with issues**. Every number in the implementer's report reproduces; the ship rule of PLAN (native
mode byte-identical to the unmodified collector) holds. One contract point that the previous round left "by
argument" (P5) is now contradicted by data and should be fixed before the B0 PID drives are frozen; the rest is
robustness and documentation.

## Reproduced

1. `--check-only --mode native --local`: hook counts `{insertions: 6, replacements: 1}`; original `run_chrono`
   `f060168debf4...`, adapted `8cf1bd508877...`, gen_collect's adapted `162c594c03a4...`, `f104_n2_sampler_sha256`
   `c80184491ba5...`; no output directory created. Rejected before Chrono loads (each with the expected exception):
   no `--local` (missing `source_manifest.json`), `branch` without `--branch-route`, `pid_perturbed` without a seed,
   `--near-stop-s 10`, branch frame 2400 at horizon 120, `branch_auto` with a missing `--recorded`, `policy`
   without `--actor`.
2. Gates against a temporary root `/tmp/vg2/groot` (symlinked `src/ scripts/ assets/` + a manifest listing the 9
   frozen files): without `--local` both gates `checked` and the manifest sha recorded; `--source-manifest-sha256
   deadbeef` -> `Source manifest mismatch`; one corrupted file hash -> `Frozen source file mismatch`; a real run
   without `FDM_RUNTIME_FINGERPRINT` and without `--local` -> fails at the fingerprint gate before any output.
   (The implementer's `S9_gates/` directories are empty, so this claim was not reproducible from the artefacts; it
   is reproduced here.)
3. Torch-free: after `import gen_collect_ext` and one `sample_continuations` call, neither `torch` nor `pychrono`
   is in `sys.modules`.
4. Byte identity (offline, `filecmp`): `S1_ref` (unmodified path) vs `S1_native` and `S1_native` vs `S2_native` on
   all six files; `outcome.json` differs only in `wall_s`; the runner's native run (S8) `trajectory.npz` identical
   to S1; the S5 replay's arrays identical to native `[:60]`. First-version artefacts (`selftest/rigid/`) equal the
   final-file re-runs for S1/S3/S4/S7 (state and action arrays); S6 differs as expected from the P1 fix.
5. `prefix_history` unit test at F = 0, 1, 20, 39, 40, 60 against the `ga_build_mixed.cut_episode` formula: hist
   equal, `hmask.sum() == min(F, 40)`, masked rows zero, newest row = `[state[F][cols], action[F-1]]`. Same code
   as `crm_collect_ext.prefix_history`; both collectors swap at the top of frame F with the prefix length asserted
   equal to F (`crm_collect_ext.py:200-202`), so `branch_frame` means the same in both worlds.
6. Policy observation contract on the S6 artefacts: the online `policy_obs` past-actions, past-states and
   last-action blocks equal `gc_control.history_blocks` (the offline twin) on 114/114 frames; frame 0 is padded with
   8 x (0, 0, 1) and 8 x `state[0][cols]`; `vx/10` and yaw rate slices equal state columns 0 and 6
   (`vel_body_x_mps`, `yaw_rate_radps` in the 17-field preset). `gb_tracker_env.obs_layout()` builds the same
   `PolicyObs` without `state_mean/state_std`.
7. Run 2 (mine, `/tmp/vg2/run2_perturbed`): `--mode pid_perturbed --episode-seed 0 --substep-log`, chosen offline
   because seed 0 taps before frame 150 (S7's seed 7 never tapped, so the tap path had not been driven). 388 frames,
   goal_reached 19.4 s; one tap at frame 3 lasting 19 frames (0.95 s, level 0.324): throttle 0.0 and brake >= 0.324
   in every tap frame; 0 frames with throttle and brake both > 0; substep max-min `[0, 0, 0]`; `applied[:, 0] ==
   action`; `held == action` to 2.8e-8; max steering step 0.1. An offline `OUPerturb(seed=0)` replay driven by the
   logged shadow actions reproduces the logged `perturbation` columns and, through `hold_clip`, the held triples
   exactly (the stream is follower-independent and deterministic as claimed). 17 brake onsets in the recorded
   action (mostly the follower's own braking), 105 braked frames.
8. Run 1 (mine, `/tmp/vg2/run1_stalled`): `branch_auto --replay-pose-tol-m 0` (replay only, nothing else driven) on
   the cluster recording `production_v3/runs/f104_v2_group_0002_route_04` (prolonged blockage; first 20-frame
   stall run at frames 143-176) with F = 173 so the recorded class is `stalled` (20/20 frames). Replay 173 frames,
   `timeout`, 24 s wall; pose 0.078 m and yaw 0.0001 rad from the recording (prefix max 0.27 m), but the replayed
   vx at F is -0.90 m/s (rolling back) versus -0.14 recorded: replayed class `moving` (10/20 stalled frames) ->
   `replay_mismatch ... class stalled vs moving`, `skipped.json` + `episode_complete.json {skipped: true}`, exit 0,
   no `__c*` directory. So the class check is not vacuous: at a stalled anchor the pose passes the 0.5 m rule while
   the velocity state differs by 0.75 m/s, and the module skips it correctly.
9. `bash -n gen_array_g.sbatch` passes; `diff` against `gen_array.sbatch` = comment block, four `GEN_*` defaults,
   runner path. `diff gen_runner.py gen_runner_g.py` read in full: env parameterisation, `_abs`, `mode`/`extra`,
   branch_auto timeout factor, `TimeoutExpired` handling, sibling-dir telemetry deletion.

## Problems

### P1 (major) B0 stop rules are not identical across arms, and the "native rule fires first" argument is false

PLAN B: "identical stop rules (the native StopPolicy plus a 40 s any-throttle near-stop rule that fires after the
native rule for PID): native PID ..., held PID ..., policy". `check_near_stop` returns `None` unless
`self.external`, so the native PID arm never has the rule; the implementer kept it that way (previous P5) on the
argument that the follower saturates throttle above 0.3 when stalled, so the native prolonged-blockage rule fires
first. Scanning all 14,400 unmodified `production_v3` recordings for an 800-frame run of |vx| < 0.3 and not parked
finds 28 episodes (0.19 %), all of which continued past the 800th frame to a 120 s `timeout`. Mechanism (checked on
three): the nearest-waypoint index ran to the end of the route while the vehicle was 4-19 m from the goal, so the
follower's desired speed was 0 and it braked with zero throttle for the rest of the episode; the native rule needs
throttle > 0.3 in every interval and never fires, the any-throttle rule would have cut them at 40 s. In the B0
suite these episodes end `timeout` (native arm) versus `prolonged_blockage_terminated` (held/policy arms), i.e.
different status, elapsed time and 'unsafe event' counts on the same route: a small (~0.2 point) but systematic
bias against the external arms in the +1 point unsafe-event bound, and a violation of the stated contract. The CRM
sibling already exposes `--near-stop-all-modes` (`crm_collect_ext.py:175, 524`); the rigid collector has no
equivalent, so the two worlds' native arms cannot be driven under the same rules. Smallest fix: add
`--near-stop-all-modes` (default off, so native stays byte-identical when unset), set
`self.near_stop_active = self.external or args.near_stop_all_modes` in `GenExt.__init__`, test it in
`check_near_stop`/`finish` instead of `self.external`, record it in `ext.near_stop_rule.enabled_by` as the CRM
collector does, re-run S1 once without the flag (byte identity) and drive the B0 native arm with the flag on.

### P2 (minor) `policy_state_check` will fail every policy episode of an actor with state normalisation

`policy_state_check` de-normalises the float32-stored observation (`cur*std+mean`) and asserts `<= 1e-6` against
the recorded state. On the S6 data with a plausible `state_mean/std` (per-column mean and sd of the episode) the
de-normalised engine-speed column differs by 9.6e-6 (float32 storage of an O(1) normalised value times a sd of
~50 rad/s), so the assert would raise after the whole drive, write `collection_failure.json` and exit 1. Dead path
today (`gb_tracker_env.obs_layout()` never sets `state_mean`), but `PolicyObs.from_meta` supports it and
`gc_control`'s self-test exercises it. Smallest fix: compare in the normalised space (normalise `rec` with the same
mean/std when `state_mean` is set) or use `atol = 1e-6 * max(1, |rec|)` per column.

### P3 (minor) Runner: a failed or killed `branch_auto` row cannot be resumed; timeouts orphan its subprocesses

`branch_auto` refuses to run when `branch_auto.json` or `skipped.json` exists (line 672), and a continuation whose
subprocess was killed leaves `collection_request.json` behind, which `run_episode` refuses to overwrite. After one
rc=1 (or a runner timeout) the row fails again on every re-launch until someone deletes files by hand, although
the replay and the completed continuations are already cache-aware. Also `subprocess.run(timeout=...)` kills only
the direct child: a runner timeout on a `branch_auto` row leaves its continuation drives running on the node.
Smallest fix: in `branch_auto`, when `episode_complete.json` is absent rename a stale `branch_auto.json` to
`branch_auto.failed.json` and continue (the per-continuation `episode_complete.json` cache does the rest); in
`gen_runner_g.py` use `start_new_session=True` and kill the process group on `TimeoutExpired`.

### P4 (minor) Undeclared deviation: brake-tap probability 0.0037 per frame instead of the plan's "p 0.05"

PLAN B3 writes "brake taps p 0.05 lasting 0.5-1 s"; the collector's `--brake-p` default is
`brake_p_for_rate(1.4/20)` = 0.0037 (0.05 per frame would mean ~43 % braked time). The choice is sensible and is
documented in `gc_control` and the note, but it is absent from the report's `contract_deviations`. Offline over
2,000 seeds: 1.41 taps per 20 s, P(no tap in 20 s) = 21 %. The plan's sizing (>= 2,000 brake onsets per world) is
met by the recorded action anyway (run 2: 17 onsets in 19 s, S7: 21 % braked frames with 0 taps), so the B3 builder
must count onsets from the recorded action, as `gc_control` says, and state the tap rate. Note also that the two
collectors use different flag names (`--ou-*`/`--brake-p`/`--perturb-seed` here, `--perturb-*`/`--episode-seed`
in the CRM sibling); the B3 task builder must emit per-world `extra` lists.

### P5 (minor) Runner hygiene: `--local` is not refused, and sibling-dir deletion globs by id prefix

`gen_runner_g.py` appends `extra` verbatim, so a task file that still carries `["--local"]` from the local tests
would run cluster episodes with the manifest and fingerprint gates bypassed (recorded as BYPASSED, so auditable,
but silent at launch). `glob(d + '__*')` deletes rich telemetry from every directory whose name starts with
`<id>__`, which includes other rows' directories when ids are prefixes of each other (`X` and `X__native`, as in
the S8 task file); the only hazard is a race with a sibling that has just written `rich_telemetry.npz` and is
about to hash its artefacts. Smallest fix: refuse `--local` in `extra` unless `GEN_ALLOW_LOCAL=1`; restrict the glob
to `<id>__replay_F*` and `<id>__c[0-9]*`; the A4/B3 builders should not create ids that are `__`-prefixes of others.

### P6 (minor) Documentation

`S9_gates/` holds no logs (see Reproduced 2). The note's "How to run"/"What was tested" sections still describe the
first-version artefacts (the implementer says so). `validate_mode_args` requires `--branch-frame >= 1` while the
CRM collector accepts 0; harmless (the A4 cuts are >= 40 frames) but worth one sentence.

## Observations for the A4 builder (not module defects)

- Run 1 shows what to expect for the 40 % low-progress anchors: pose within 0.08 m but class mismatch (rollback
  versus stall) when the recording came from another node. Drop counts must be reported per anchor type before
  labels are trusted, as the plan requires; if they are high, re-recording the anchor episodes on the A4 node (the
  replay dir already is such a recording) and choosing the cut on the replay would remove the cross-node dependence.
- With the plan's cut window [onset-1 s, onset+0.5 s] and onset = the first frame of the 20-frame run, the recorded
  class at F is always `moving` by `stall_class`'s definition (the 20 frames before F cannot all be stalled); the
  class check then only guards against the replay stalling where the recording did not. If the builder intends
  `stalled` anchors it must define the onset as the end of the run or cut later.
- Nothing in the module enforces distinct seeds across rows (implementer's own note); `--cont-seed` must be
  anchor-specific.

## Contract checks (PLAN) - no violation beyond P1

- Timing: hook C reads the shadow follower at the frame top (its last `Advance` of interval k-1), one `hold_clip`
  per frame, the identical `DriverInputs` at all 25 substeps, `previous_steer` = held steering (verified in run 2
  and S3/S6/S7); recorded `action[k]` = held triple (asserted in code, re-checked).
- Branch: swap at the top of frame F before the nearest-waypoint search, `wp = 0`, `state[F]`/`pose[F]` identical
  to native, `action[F]` first differing (S4); `branch_pose == pose[F]`, `branch_hist` = mixed-dataset convention;
  `command_reference.npz` carries the branch arrays; the case goal is kept and the branch route must end within
  0.25 m of it.
- Geometry: continuations from `gc_control.sample_continuations` (night-2 family, planner validator, v0 floor and
  15 deg heading acceptance = the gc_control verification's outcome); follower points via `tmap.height` as
  `make_driver`; no depth map or grid in this module.
- Gates: `--local` bypasses only manifest and fingerprint; arena allowlist and case checks stay; recorded in three
  places. Split copied from the case; blacklist/held-out handling is the builders' job.
- Runner/sbatch: rows filtered by `shard`/`run`, `mode` -> `--mode`, `extra` last (a row's `--horizon-s` overrides
  the runner's 120, S8), branch_auto timeout x(1+n_cont), env recipe / fingerprint / chrono data path unchanged,
  `GEN_ROOT` passed explicitly in the launch example.
- No torch in the collector process; float32 everywhere in this module (no float16).

## Not verified here

Cluster-side behaviour (node determinism of replay vs pass-2 prefix under 126 workers, `--check-only` against
`G/source`, rsync/manifest, the arena directories under `G/source/assets`), the CRM sibling, and a full `branch_auto`
with continuations on a stalled anchor (run 1 stopped at the replay by design).
