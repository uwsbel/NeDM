# Verification of module ga_data (PLAN A1 `ga_build_mixed.py`, A1b `ga_domain_probe.py`)

Adversarial check run 2026-09-21 on luffy. No cluster submissions, no Chrono/CRM runs; GPU use limited to the probe re-runs
(< 1 GB). No repository file was modified; scratch outputs went to `/tmp/verify_ga_data/`.

## Verdict: pass with minor issues

Everything the implementer claimed reproduces exactly (build byte-for-byte, every probe number to 4-5 decimals), the
contract keys/shapes/rules of PLAN A1 are met, and my independent recompute of the windows and privileged context from
the raw episodes agrees. The issues below are documentation/naming and process points, none changes a number.

## What I re-ran and checked

| check | method | result |
|---|---|---|
| Small build determinism | re-ran `--limit-episodes 50 --workers 4` to `/tmp`, compared all 25 arrays (NaN-aware) with `selftest/mixed_small.npz` | identical, all arrays, 9 s |
| Windows / masks / privileged by hand | my own code (step-by-step loop, not the vectorised indexing of the builder) on 160 random rows of the full file: 120 uniform + 20 at k = 40 + 20 at k >= 800; every window step (not only the last) against `trajectory.npz`, CRM slip/spindle against `crm_extra.npz`; `ctx[:, :17] == state[k]` | all equal; max relative f16 error 4.87e-4; privileged max relative error 5.9e-11; hmask equal |
| Global invariants | whole file: hist finite (max abs per channel: vx 8.4, omegas 176, engine 275 rad/s, far below f16 65504), k = 0 rows all-masked, k > 0 rows all-valid, masked steps zero, id suffix per domain, id base == `episode@k`, `(episode, anchor_frame, domain)` unique | all hold |
| Splits | 1,200 groups, one split each, identical across domains; equal to `twin_rigid.npz` group splits and to `case.json` (`split`, `id`) on 30 sampled rows; counts 1,089 / 56 / 55 | hold |
| Preservation of `X` on ALL rows | bitwise (`uint16` view) comparison of the 3.56 GB corridor tensor against both source files (the implementer sampled 2,000 rows) | identical on all 58,424 rigid + 57,444 CRM rows |
| Counts | 15,024 episodes per domain, same episode set in both; rigid 9,135 in `production_v3` + 5,889 in `production_v4`; 29,966 coinciding established pairs; probe set 54,408 / 2,794 / 2,730 | as reported |
| Probe reproducibility | re-ran `--seeds 3 --epochs 20 --ablate` to `/tmp` | every number identical to `probe/probe.json` (GRU per seed to 5 decimals, all bins, hand logistic, 13 ablations, startup joint and per-column); 42 s here vs 80 s reported |
| Timing / pairing convention | read both collectors: `state[k]` is measured at substep 0 of interval k after `driver.GetInputs()` + steering clamp and `hmmwv.Synchronize`, before `Advance`; `action[k]` is that clamped input. The dataset pairs `state[s]` with `action[s-1]`, exactly the PLAN's `state[k-39..k]` / `action[k-40..k-1]` and the same pairing as the deployment-side `build_history` in `src/nedm/traverse/fdm_data.py` ("state at t plus action from [t-dt, t)") | consistent, no off-by-one |
| Cross-domain channel conventions | omega_mean vs vx/R: median ratio 1.04 rigid / 1.50 CRM (more slip on soil, same sign/units); engine/omega 25-28 in both worlds; state field names identical in both trajectories; steer values show no domain-specific quantisation (6-7 % multiples of 0.002 or 0.004 in both) | no unit/sign mismatch that would poison a shared encoder |
| Speed-command confound in the probe | 12 random twin episodes: waypoints, reference speeds and `desired_speed_mps` identical in both worlds | absent |
| Consumer compatibility | `ga_train.py` appends the 3 action columns when `hist_cols` has 12 entries (line 256), reads `hmask` as True = valid and zeroes masked steps; its self-test ran on `mixed_reanchor.npz` | compatible |
| Supplementary probe | removed the k = 40 rows (49 % of the coinciding established pairs; their window is the launch transient frames 1-40), 1 seed 10 epochs | GRU 1.000 / 1.000 in every speed bin; hand logistic 0.757 / 0.776. The AUC = 1 reading is not a launch-window artefact |

## Problems found (evidence and smallest fix)

1. **Minor - the CRM "sinkage" column is not sinkage; it is spindle height above the ground at the vehicle centre.**
   `crm_extra.npz` stores `bmp_ground_z_m` = `tmap.height(pos_x, pos_y)` at the chassis position (crm_collect.py, the
   `record_extra` line), while the collector's own sinkage (`crm_collect.py:285`) is `R - (spindle_z - ground under that
   spindle)`. The builder's column is `mean_w(spindle_z_w) - ground_at_centre`. On the CRM rows, a regression on
   (roll, pitch, |roll|, |pitch|) at the anchor explains 43 % of its variance; at rest (k = 0) it is 0.44 +- 0.05 m, while
   on moving rows it spreads 0.36-0.79 m (p1-p99), i.e. terrain curvature/attitude dominates, not wheel sinkage. It is
   still a valid teacher feature (the same quantity in every row) and matches the notes' wording, but calling it
   "sinkage" in the report would mislead. Smallest fix: keep the array, rename `priv_names[6]` to something like
   `spindle_above_centre_ground_m` and say in NOTES that a true per-wheel sinkage needs per-spindle ground height (or
   spindle x, y) which the current `crm_extra.npz` does not contain; the ext collector for A4/B3 could add it.
2. **Minor (process) - the probe reports the sealed test split by default.** PLAN [R1, R12] seals the 55 test groups for
   the final offline report; `ga_domain_probe.py` prints and stores test AUC for every probe and ablation. No selection
   is done on it, so nothing is contaminated, but the read-out habit is against the rule. Smallest fix: report val only
   unless `--report-test` is given (one flag, one conditional in the table/json).
3. **Minor - `privileged` at k = 0 is a single-frame value, not a window mean.** Documented by the implementer; note for the
   trainer (`hist_rma`): the k = 0 slip ratio is ill-defined at rest (p1 / p50 / p99 = -0.99 / -0.09 / 0.62 vs 0.13 /
   0.81 / 7.95 when moving), so the teacher's privileged embedding sees a different statistic for startup rows. Either
   zero the two CRM columns for all-masked rows or let the trainer standardise startup and established rows separately.
4. **Minor - notes should state the composition of the probe set.** 49 % of the coinciding established pairs are k = 40
   (window = frames 1-40, the launch), median k = 80, p90 = 200. The supplementary run above shows the conclusion holds
   without them, so this is a reporting point only.
5. Already flagged by the implementer and confirmed: the CRM slip-ratio column is heavy-tailed (244 rows with |mean| > 10,
   max 75.9) and stored raw; the trainer must clip or robust-scale it.

## Things I looked for and did not find

- Off-by-one at the anchor (checked first, middle and last window step by hand; `hist[-1] == state[k]`, `hist[0] ==
  state[k-39]` with `action[k-40]`).
- Mask polarity or masked-step leakage (masked steps are exactly zero; k = 0 rows all-masked; `hmask.sum() == min(k, 40)`).
- float16 overflow or NaN in `hist` (none; largest channel 275 rad/s).
- Non-determinism of the pool build (byte-identical re-build) or of the CUDA GRU probe (identical to 5 decimals).
- Held-out leakage in the probe (train-split rows only fit the GRU and the logistic standardisation).
- Blacklisted planner-suite ids/groups (none; only `f104_v2_group_*` names exist in the run dirs) and duplicated twins.
- A per-world unit, sign or quantisation fingerprint that would make the AUC = 1 trivial (none found; the last-step
  logistic is 0.87, window mean+std 0.98, and the signal is redundant across channel groups).
