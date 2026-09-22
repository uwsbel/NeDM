# Verification round 2: gb_nrd_ppo (gb_nrd_common.py, gb_train_nrd.py, gb_tracker_env.py, gb_train_tracker.py)

Verifier run 2026-09-21 (evening) on luffy (RTX 5090, conda `nedm`), after the implementer's fix round 1 and the
resumed session (NOTES sections 5 and 6). Every self-test command of the implementer's report was re-run into
`/tmp/verify_gb2/` from the repo root with `PYTHONPATH=src:scripts`; nothing under `artifacts/` or `scripts/` was
modified (this file excepted); no cluster submission (the cluster sbatch and the failed job's `sacct` line were only
READ over ssh); no Chrono/CRM process was started (none of the four scripts touches Chrono, so no lock was needed).
The GPU was shared with foreign processes at 100 % utilisation / 19.9 GB; my peaks: 0.16 GiB (toy NRD), 0.04-0.07 GiB
(tracker), 3.77 GiB (full-size NRD dry run), all under the 8 GB budget.

Round-1 findings (`VERIFY_gb_nrd_ppo.md` of 18:05: 1 blocking, 1 major, 5 minor) are all closed by fix round 1 and
re-verified below (items N1-N5 and the consistency re-run). Round-1 verdict was "fail until one line is fixed".

**Verdict: pass with minor issues.** Every claimed number reproduces bit-exactly (toy NRD, stop/resume, PPO warm start,
PPO resume, exports, check-actor, independent rebuild, full-size dry run on the real cache); the env / trainer /
deployment conventions agree to float precision; no binding contract of PLAN.md is violated. Three non-blocking
issues with one-line fixes are listed in section 3.

## 1. Reproduced (my numbers vs the implementer's)

| self-test (implementer's command, my `--out` under `/tmp/verify_gb2/`) | result |
|---|---|
| `gb_nrd_common.py --make-synthetic-cache --selftest` | manifest sha256 `8db58091…` identical; all 40 episode npz byte-identical (sha256 of the sha list `5af871e2…` on both); shapes (5,16,64)/(5,16,17)/(5,16,1) tag and notag; tag changes output True; round trip 0.0 |
| NRD 200 steps (toy) | loss 1.3770 (1) -> 0.1138 (100) -> 0.0810 (200); sel 0.9350 -> 0.6519; val_loss 0.0344; **481 numeric metrics equal to `selftest/nrd/run200/metrics.json` with max abs diff 0**, 0 non-finite; `ckpt_best.pt` weights max abs diff 0.0; gate cells rigid/crm/all n_cells 3 (stalled n=8, moving 63/61, brake 4); GPU peak 0.158 GiB |
| stop at 100 + resume | `stop_at_step 100` -> "resumed … at step 100" -> done 200/200; 481 numeric metrics equal to the straight run (max abs diff 0, `wall_s`/`train_sps` excluded); losses at 100/101/200 = 0.1138/0.1093/0.0810; `config.json` has `resumed_from`, `stop_at_step 0` |
| cluster command line on the real cache (`--max-episodes 200 --steps 4 …`) | "1200 cache groups agree with the twin split … train 1089 / val 56 / test 55"; 200 train (117 rigid / 83 crm, 186 groups) / 200 val (54 groups) loaded in 0.6 s; 4,900,946 params; all gate cells present (crm brake n=9); GPU peak 3.7706 GiB; `finished: true`; 0 non-finite |
| tracker smoke (64 envs, toy NRD) | PolicyObs agreement 1.6151e-7 base / 0 / 0; pure pursuit cross-track 0.031728 m, reward 0.647626, 0 failures; random reward -0.030807; **every float identical** to `selftest/tracker/smoke/smoke.json` except throughput (2.3k env steps/s under the foreign load) |
| PPO 3 iterations, warm start, DEFAULT logger | exit 0, tensorboard events written; imitation 2,048 samples (986 rigid / 1,062 crm), epoch MSE [2.2905, 1.9227]; pre-tanh MSE 2.2905 -> 1.6061, physical MAE [0.0494, 0.2378, 0.4157] -> [0.0573, 0.1548, 0.2655]; after 2 PPO iterations 1.3603 / [0.0354, 0.0917, 0.2142]; **identical to the implementer's `imitation.json`**; exports init/0/1/2 on 164 rows: 9.28e-8, 7.85e-8, 1.12e-7, 1.39e-7; `actor_2.npz` byte-equal to `actor.npz`, `policy_meta_2.json` equal to `policy_meta.json`; meta: squash (0, 0.5, 0.5)/(1, 0.5, 0.5), box [-1,0,0]..[1,1,1], rate 0.1, settle (0,0,1), `nrd.cond = tag`, 158-D layout |
| PPO resume from `model_2.pt`, `--max-iterations 5` | "3 iterations completed, continuing at iteration 3"; iterations 3, 4; run_state 5/5; `actor_3.npz` / `actor_4.npz` (9.24e-8, 1.93e-7) |
| `--check-actor model_2.pt --check-n 100` | 164 observations, max abs da 7.23e-8 (pre-squash 2.03e-7), exit 0, `actor_check_2.json` with the npz sha256 |
| `check_actor_independent.py model_2.pt actor_2.npz 100` (rsl_rl classes only) | n=100, max abs da 7.15e-8, exit 0 (its json lands next to the run dir, not in the repo) |
| `py_compile` on the four scripts; `git status` | compile; no file of the four changed by this round (tree as in the implementer's status) |

Throughput claims (11.6 optimizer steps/s, 99-110k env steps/s at 2048 envs) again could NOT be verified: the GPU
ran three foreign processes at 100 % during every run (toy NRD 775 windows/s, tracker 2.2-3.0k env steps/s at 64 envs
here). Not a defect; the implementer's idle numbers are plausible given the bit-identical results.

## 2. Adversarial checks of this round (scratch in `/tmp/verify_gb2/`, nothing in the repo)

N1. `--check-actor` with `W0 * 1.001` in the npz: max abs da 1.83e-4, ok=False, **exit 1**. N2. `actor_init.npz` copied
    over `actor_2.npz`: 2.44e-1, **exit 1**. So the collectors' artefact check fails loudly.
N3. NRD `--resume` with `--batch 16`: `argument mismatch on resume … {"batch": {"checkpoint": 64, "args": 16}}`,
    exit 1; resume with only `--val-batches 2` and another `--out`: accepted, exit 0 (round-1 issue 2 closed).
N4. A `cond='notag'` toy NRD (2 steps): the tracker exits 1 with the `--allow-notag` message; with the flag the env
    builds (`NRD cond=notag`) (round-1 issue 4 closed).
N5. On the REAL manifest (39,235 episodes): a WHOLE val group relabelled `train` passes `check_group_split_consistency`
    (as before) but `check_split_against_twin` raises ("groups with a different split"); an unknown group raises; an
    explicit non-existent `--twin-split` path raises instead of falling back (round-1 issue 7 closed).
N6. `PolicyObs.from_meta(route, policy_meta.json).layout() == policy_meta["obs_layout"]` (True); the npz's `meta_json`
    carries the same `obs_layout` (True). The npz meta lacks `numpy_actor_check` (added after export; cosmetic).
N7. Re-run of the round-1 consistency script (`/tmp/verify_gb/consistency.py`) on the current code: env reset context
    vs `Batcher.window_batch` 0.0; env `_nn_step` vs the `rollout_loss` path under recorded actions 0.0 over 24
    steps (z1 and pose); frame-0 padding at s=0/5 exact (token 6e-8); streamed observation vs `PolicyObs` over 40
    steps: base 1.8e-7, past actions 0, past states 0; env steering clamp vs `hold_clip` 6.6e-8; `find_windows`
    0 definition violations (16 stalled / 124 moving / 8 brake); imitation pairing target = act[s], obs last = act[s-1].
N8. Real cache, tracker `--smoke` with the 300-step full-size NRD (`selftest/nrd/timing_real/ckpt_best.pt`), 300
    train episodes (174 rigid / 126 crm, 148k frames): twin split agrees; PolicyObs agreement on real routes 1.84e-7
    base / 0 / 0; rewards finite; GPU peak 0.07 GiB.
N9. Stalled-escape gate: the code's `escape_frac` uses the ALONG-TRACK projection of the 60-step displacement
    (`rollout_windows`, `disp = c0 (x - x0) + s0 (y - y0)`), PLAN B4 says "|displacement| > 1 m". With the planar norm
    instead (scratch `/tmp/verify_gb2/escape_metric.py`, same windows, same fed-back rollout): synthetic val (toy NRD)
    identical (0.875 / 0.5); real val split, 256 stalled windows per domain, 300-step model: rigid 0.461 (code) vs
    0.473 (planar), crm 0.348 vs 0.355, i.e. ~1 point of predicted escapes are purely lateral (mean predicted
    lateral drift 0.21 m; the recorded stalled windows drift 0.04-0.05 m planar, 0 escapes). See issue 1.
N10. Cluster: `G/train/nrd_tag.sbatch` READ over ssh at 22:4x already carries `set -o pipefail` (no `-u`; file mtime
    22:36, after the 22:25 failure of job 430699 = FAILED 00:00:01 exit 1:0 on k007-005-v3 per `sacct`), and
    `G/train/nrd_tag/` exists (22:41). Someone has already applied the implementer's fix and relaunched; nothing
    submitted or changed by me. The argument set matches the local dry run (explicit `--grid`, `--twin-split`,
    `--max-minutes 225`, resume via `[ -f $OUT/ckpt_last.pt ]`), which the trainer's strict resume check accepts
    because the second job repeats the identical command line.

## 3. Problems (none blocking)

### Issue 1 (minor, definition): the stalled-escape gate under-counts lateral escapes
`scripts/gb_train_nrd.py` `rollout_windows` -> `cell_metrics`: `escape_frac = mean(|disp_pred| > 1)` with `disp` the
projection on the initial heading; a predicted sideways or diagonal drift of > 1 m with |along-track| <= 1 m is not an
escape. PLAN B4: "predict escape (|displacement| > 1 m)". Measured effect ~1 point on the real val split (N9), 0 on
synthetic. Smallest fix: in `rollout_windows` add `add("disp_planar_pred", (pose[:, :2] - pose0[:, :2]).norm(dim=1))`
and in `cell_metrics` `"escape_frac": f(r["disp_planar_pred"] > 1.0)` (keep the along-track signed metrics as they are).

### Issue 2 (minor, doc + robustness): a `--max-minutes` stop skips the final evaluation
`gb_train_nrd.py` main loop: the `max_minutes` check `break`s before the `eval_every` branch; after the loop only
`ckpt_last.pt` and the PREVIOUS evaluation's metrics are written, so `ckpt_best.pt` and `metrics.json` can be up to
`eval_every - 1` (1,999) steps stale and the stopped weights are never scored. NOTES section 2 ("`--max-minutes 225`
leaves 15 min for the final evaluation and checkpoint") and section 5 item 6 describe an evaluation that does not
happen. Harmless when a continuation job finishes the schedule (it evaluates at `step == steps`), wrong for a run that
ends on the time cap. Smallest fix: after the loop, `if stop_reason != "done" and step not in evaluated_steps:` run the
same evaluate/save/write block once (and fix the note).

### Issue 3 (minor, hardening): the fragment bank asserts held-out absence only for `--split train`
`gb_tracker_env.py` `FragmentBank.__init__`: `if cfg["split"] == "train": assert not bad`; `gb_train_tracker.py` has
`--split` (default train) with no guard, so `--split val` or `test` trains PPO on held-out groups with no message
(only `policy_meta.json` `bank_split` records it). PLAN: "the tracker fragment bank … assert no held-out group is
present". Smallest fix: in `gb_train_tracker.py main()` after the env is built,
`if args.split != "train" and not args.smoke: raise SystemExit("PPO bank must be the train split (PLAN); --split "
f"{args.split} is for --smoke only")`.

Nits (no action needed): `find_windows` brake: `range(max(10, context - 1), n - K - 1)` excludes the last admissible
onset frame `f = n - K - 1` (conservative by one window). The bank keeps every status (34 train rollover episodes,
post-rollover frames can be imitation targets; the env terminates them at the first step). rsl_rl's adaptive-KL
learning rate restarts at `--learning-rate` on `--resume` (rsl_rl behaviour, not the fork's).

## 4. Contract check against PLAN.md (no violations found)

- Cache contract (schema 3; `z1 (T,17)`, `act`, `pose`, `power`, `stalled`, `hold_ok`, `desired_speed`, routes,
  `domain` 0/1, `group`, `status`; manifest keys; `--max-frames 1200`): read as specified in both readers; file domain
  cross-checked with the manifest; z1 asserted 17-D. The builder's `hold_ok[k]` is the k -> k+1 transition
  (`gb_build_cache.py:12-13, 73-74`), which is exactly the index the trainer weights (`step_loss` `hold[:, :context]`,
  `rollout_loss` `hold[:, context-1+step]`), and `stalled` is the contract definition (`STALL_VX/THR/RUN` 0.3/0.3/20).
- Splits: train = manifest `train`, val = `val`, test never loaded (`test 1801 episodes untouched` printed); held-out
  groups asserted absent in the NRD train set and the (train) fragment bank; group/split consistency asserted; the
  twin group split (`crm_night2_v1/datasets/twin_crm.npz`, the manifest's `report.twin_split`) cross-checked (N5).
  Blacklist is the builder's job (verified there), not re-checked by these readers.
- Timing convention: transition z1[k] -> z1[k+1] under act[k] with weight `hold_ok[k]` in the step loss, the rollout
  loss and the env (N7); the observation at frame k sees state[k], pose[k], act[k-1] (settle at k = 0) and its output
  becomes act[k]; steering-rate clamp == `hold_clip`.
- Geometry source: crops from the v2 grid via `gb_crop.EgoCrop` at every pose (recorded in the step loss, dead-reckoned
  in the rollout / validation / env), sha256-checked at load; never cached.
- Checkpoint: `model_kind='gb_nrd'`, `config`, `normalization`, `delta_scale`, `grid_path` + `grid_sha256`,
  `domain_vocab`, `train_args`, `cache_manifest_sha256`, `train_state` in `ckpt_last.pt` only; `policy_meta.json`
  records `nrd_sha256`, `nrd.cond`, `cache_manifest_sha256`, the squash and the 158-D `obs_layout`; numpy actor
  check < 1e-5 enforced at every export and by `--check-actor` (max 1.9e-7 observed; corrupted npz -> exit 1).
- Observation = `gc_control.PolicyObs` (38 + 24 + 96), deployable columns 0-6, 11-14, 15, frame-0 padding: N7, N8.
- Action squash centre (0, 0.5, 0.5) scale (1, 0.5, 0.5) over the full box; asserting equality across arms is the
  collectors' step.
- B4: per-token domain one-hot (`tag`/`notag`), batch fractions 0.5/0.5 per domain, equal-domain delta-std channel
  weights (`--delta-scale`, passed by the sbatch), hold_ok weighting, resume of optimizer/schedule/step/RNG (bit-exact),
  validation kinds and gate thresholds (fed-back, all three cells required for `pass`); PPO refuses a `notag` NRD.
- B5: one static grid; fragments from [0, active_end - len] with padding; warm start on the runner's own normaliser
  fitted first; MSE logged at iteration 0 and `--mse-at` (10 by default); WP3 reward + progress; rsl_rl resume.
- No float16 anywhere; grid f32; poses f32 (4e-6 m at +-40 m); determinism confirmed by the bit-identical re-runs.
- Documented deviations (hold-bad down-weight 0.25, window definitions, domain-balanced PPO resets, imitation clip
  0.99, MSE at iteration 2 in the smoke) contradict no binding contract.

## 5. Files

- Code: `/home/harry/NeDM-traverse_mppi/scripts/gb_nrd_common.py`, `gb_train_nrd.py`, `gb_tracker_env.py`,
  `gb_train_tracker.py` (unchanged by this round).
- Implementer's artefacts: `/home/harry/NeDM-traverse_mppi/artifacts/traverse/generalist_20260921/B_tracker/selftest/`.
- Verifier's re-runs and scratch: `/tmp/verify_gb2/` (`nrd/{cache,run200,run_stop100,cluster_cmd_dryrun}`,
  `tracker/{smoke,ppo3,ppo3_resumed,ppo3_check,real_smoke}`, `neg/neg.log`, `consistency_rerun.log`,
  `escape_metric.py`, `escape_synth.log`, `escape_real.log`); not part of the repo.
