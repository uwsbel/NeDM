# Verification of fix round 1: `scripts/gc_control.py` and `scripts/gen_collect_ext.py`

Adversarial verifier, 2026-09-21 night. Checked the fix report against `VERIFY_gc_control.md` problems 1-4 and
`VERIFY_gen_ext.md` P1-P4. Files under test: `scripts/gc_control.py` sha256 `683f24ad3992...` and
`scripts/gen_collect_ext.py` sha256 `b9f36a024570...` (both match the report). Budget used: 3 rigid Chrono runs
(CPU only, horizon 12 s, never more than 3 at once), 0 CRM runs, no cluster jobs, no GPU. Scratch outputs under
`/tmp/vfix1/`. Nothing in the repo was edited except this file; the three protected artefact directories have no
file modified tonight (checked by modification time).

Verdict: **pass**. Every number I was asked to re-check reproduces, and the two ship-blocking properties hold:
the policy sees exactly the recorded state, and native mode is still byte-identical to the unmodified collector.

## What was re-run and what came out

1. **gc_control self-test** (`python scripts/gc_control.py --selftest --out /tmp/vfix1/gc_control`, 4.1 s, exit 0,
   "gc_control self-test OK"). The JSON is identical to the implementer's `selftest/fix1/gc_control/gc_control_selftest.json`
   apart from `runtime_s` and the npz path; both actor npz files are byte-identical (`cmp`). The numbers quoted in the
   report are all there: OU at brake_p 0.05 594 taps / 0.5286 braked, default brake_p -> 1.425 taps per 400 frames,
   actor 3.28e-6 / 5.07e-6 / 2.02e-7, 64 base routes identical, 38 fallbacks, 12 contract agreements, follower
   96 -> 24 points, PolicyObs diff 0.0, torch-free import; demo anchors 12/45 below v0 and 27/45 over 15 deg before,
   45/45 and 45/45 after; A4 subset 9/45 and 27/45 before, 45/45 and 45/45 after, max draws 100 (budget 192).

2. **15-anchor continuation check, rebuilt independently** (`/tmp/vfix1/anchors15.py`: own wrap/heading maths,
   the 5 local CRM demo episodes cut at 2/4/6 s, the same md5("run:F") seeds). Before the fix (tolerance off, no v0):
   12/45 continuations start below the vehicle speed, 27/45 kink more than 15 deg, 3-5 draws per anchor; after
   (v0 given, default 15 deg): 0/45 below v0, 0/45 over 15 deg, 4-15 draws. Stronger than the report's first-point
   claim: on all 45 routes the WHOLE speed profile stays at or above the 2 m/s^2 deceleration ramp from v0, the
   accel/decel limits on v^2 along station hold (A_ACC 1.5, A_DEC 2.0 read from `f104_n2_sampler`), speeds stay in
   [0, 6], the planner validator and the reference contract pass, and the `start_heading_err_deg` stored in `meta`
   equals my own recomputation. Bit-identity claim: with tolerance off and no v0, the draw counts and route lengths
   equal the OLD self-test's recorded values (`/tmp/gc_verify2/gc_control_selftest.json`) on 15/15 anchors.
   Note (by design, not a defect): the acceleration cap is opt-in and off by default, so 30/45 start above the vehicle speed (median gap 2.48 m/s, max 4.68 m/s)
   on these 15 anchors; the task asked for the floor only.

3. **P1, policy observation = recorded state.** S6 re-run locally
   (`--mode policy --actor selftest/selftest_actor.npz`, horizon 12, group_0000 route_00): 114 frames,
   `terrain_bounds_exit` at 5.7 s. Recomputed from the raw files (`ext_control.npz` `policy_obs[:, -12:]` vs
   `trajectory.npz` `state[:, (0-6, 11-14, 15)]`): per-column max |diff| = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
   114/114 frames exactly equal; `outcome.json` `ext.policy.current_state_row_check` says the same. The pre-fix
   `selftest/rigid/S6_policy` recomputed the same way shows the earlier finding (column 15 max 1.008 rad/s, 0/114
   frames equal), so the check would have caught it. My S6 files (`trajectory`, `ext_control`, `anchor_state`,
   `command_reference`, `contact_events`) are byte-identical to the implementer's `fix1/rigid/S6_policy`.

4. **P2, no silent seed.** `--check-only --mode pid_perturbed` without a seed: exit 1,
   `ValueError: pid_perturbed needs --perturb-seed or --episode-seed (no default seed: every episode must have its own)`,
   no output directory created. With `--episode-seed 7`: exit 0, contract `perturb_seed 7`, `episode_seed 7`, no
   `seed_source` field. With `--perturb-seed 3` only: exit 0, `perturb_seed 3`.

5. **Native mode byte-identical to the unmodified collector (S1).** Because `S1_ref` was made at horizon 120 s and my
   budget is 12 s, I rebuilt the reference at 12 s with the unmodified path (`gen_collect.import_runner` +
   `gen_collect.adapted_function` + `gen_collect.StopPolicy` + `gen_collect.make_observer`, rich telemetry on,
   `/tmp/vfix1/s1_ref12.py`) and ran `gen_collect_ext.py --mode native` at 12 s: `trajectory.npz`, `anchor_state.npz`,
   `rich_telemetry.npz`, `rich_intervals.npz`, `rich_telemetry.json`, `contact_events.json` all byte-identical (`cmp`);
   `outcome.json` identical apart from `wall_s` (both `timeout`, 240 frames). In addition the 12 s run's first 240
   rows of state/action/pose/power_kw/contact_n equal the 120 s `S1_ref` exactly, so the horizon does not change
   the physics. My native contract records wrapper `b9f36a024570`, gc_control `683f24ad3992`, original `run_chrono`
   `f060168debf4`, adapted `8cf1bd508877` (the frozen-loop hooks are unchanged, as reported).

## Read, not re-run

- P3 docstring (lines 61-67) and P4 code (lines 684-722): the replay directory is `<out>__replay_F<F>`, a cached
  replay must carry `requested_horizon_s == F/20`, and the replay is accepted only with `status == "timeout"` and
  `frames == F`, else a skip with `replay_ended_before_branch_frame`; `branch_auto` calls `sample_continuations`
  with `v0 = replayed vx at F`, the heading tolerance (default 15) and the opt-in cap (lines 733-746). Matches the
  report. The S5/S10/S11/S12 branch artefacts under `selftest/fix1/rigid/` were not re-run (outside the checks
  requested); the 800-anchor sweep (`/tmp/a4_all.py`) was not re-run either.
- `brake_p_for_rate` is module-level, `DEFAULT_BRAKE_P = brake_p_for_rate(1.4/20)` (lines 103-124); the self-test
  value 0.003694 reproduces.

## Problems

None that block. Two notes for the row builder, carried over unchanged from the earlier verification (the fix
report says so too): the continuation seed must be anchor-specific and distinct across rows, and anchors the sampler
cannot solve within 192 draws (the report names 2 of 800) must be dropped, not retried with another seed.
