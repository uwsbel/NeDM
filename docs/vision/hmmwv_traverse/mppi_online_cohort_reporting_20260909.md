# Complete online-cohort accounting and validation energy selection

The new [`report_traverse_fdm_diverse_online_cohort.py`](../../../scripts/report_traverse_fdm_diverse_online_cohort.py) reports every declared trial. Missing results, failed processes, invalid artifacts, timeouts and planner abstentions stay in the unconditional denominators. A process returning zero is distinct from a physically successful traversal.

The reporter verifies the complete task-file hash and task list, unique declared/result IDs, declared and finished counts, batch failure list, per-run result agreement, log hashes, mandatory output hashes and immutable source-manifest/runner hashes. For each usable trial it checks the declared energy coefficient, exact model/observation/case/source hashes, settled launch equality, N intervals and N+1 real endpoints, actual timestamps, solver-rate work/risk coverage, positive-work sums, contact/rollover/blockage outcomes and complete planning/abstention summaries. Raw arrays remain unchanged. Protected test task paths are rejected.

Time/work fractions are available only when both compared runs reach the supplied full goal with verified `schema_safe_goal_reached=true`, using matched checkpoint, observation, source/runtime, native controller and non-energy planner settings. Failed-run work is listed as consumption, not an efficiency improvement. Each unavailable pair retains its reason.

## Preregistered rule

The [selection specification](../../../artifacts/traverse/fdm_diverse_v1_20260909/online_tasks/energy_selection_spec_v1.json) was written before validation-grid outcomes. Its SHA is bound by the grid task file, whose full hash is recorded by the batch runner. The rule is:

- Exactly six validation scenes and four coefficients: **0, 0.02, 0.2, 0.5 s/kJ**, giving 24 declared trials.
- At least **three of six** scenes with paired verified safe full goals.
- For each paired scene, work saving is `1 - candidate_work / baseline_work`; time increase is `candidate_goal_time / baseline_goal_time - 1`. Aggregate these fractions with an unweighted arithmetic mean over scenes.
- Require mean work saving **at least 5%**, mean time increase **at most 25%**, and candidate unconditional safe-success count at least the zero-weight baseline count.
- Choose the **smallest qualifying weight**. One or two safe pairs are descriptive case evidence only.
- A complete valid grid with no qualifying weight retains **0.02** with `gate_passed=false` and no optimum/improvement claim.
- An incomplete or invalid grid returns **no selected weight** and blocks selection/freeze until fixed.

Before selection, the common model must have a 12 s forecast horizon, training seed 11 and declared 5000-update budget. Its training `status.json` must say complete at step 5000. The chosen file is `best.pt`, and its step must match the completed run's `best_step`; the validation-best checkpoint may precede step 5000. Completion-file/checkpoint hashes and the relevant metadata are recorded. Pilot reports do not invoke this selection mode.

[Independent CPU checks](../../../scripts/check_traverse_fdm_online_energy_selection.py) cover all-failure denominators, forbidden failure tradeoffs, the three-pair minimum, inclusive numerical thresholds, equal-scene averaging, conditional-survival bias, smallest-weight selection, explicit default retention and invalid-grid freeze blocking. No model training or physical simulation is performed by these checks.

## Pilot v6 results

[The audited pilot report](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/online_pilot_long_v6_cohort_01/report.md) contains all four trials from AMD job **412083**. All four processes completed and all artifacts passed audit; **zero of four reached a safe full goal**. Every run timed out at 180 s and had bounded blockage. Both ridge-pass runs registered contact. The mixed-obstacle runs made 53.04 m and 65.93 m goal progress; the ridge runs made approximately 31.42 m. Each run had 104–117 planner abstentions among 180 decisions.

There are **no eligible paired safe goals**, so no energy-efficiency fraction or weight selection is reported. This is an implementation/behavior diagnostic for the pilot model, not a generalization result. The later candidate-retention policy and full-model validation grid must be reported as separate declared experiments.

Run the pilot report with a new output directory:

```bash
/home/harry/miniconda3/envs/nedm/bin/python scripts/report_traverse_fdm_diverse_online_cohort.py \
  --batch artifacts/traverse/fdm_diverse_v1_20260909/online_pilot_long_v6 \
  --tasks artifacts/traverse/fdm_diverse_v1_20260909/online_tasks/pilot_long_v6.json \
  --code-root artifacts/traverse/fdm_diverse_v1_20260909/snapshots/online_v6 \
  --path-map /work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909=/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_diverse_v1_20260909 \
  --out /tmp/fdm_pilot_cohort_new
```

Add `--select-validation-energy` only for the complete predeclared 24-trial validation grid and its exact source snapshot.

## Invalid first full validation grid

The [complete accounting for grid v2](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/online_full_validation_grid_v2_cohort_02/report.md) retains all 24 trials from AMD job **412098** (6 min 28 s, process exit 1). Fifteen trials crashed during candidate generation because a fresh reference triggered the contract error `Reference contains a cusp or sharp segment-heading reversal`. They did not write terminal trajectory, rich telemetry or outcome artifacts. The [failure evidence](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/full_validation_grid_v2_failures/failure_evidence.json) records each declared trial, log/result checksums, last saved decision and missing terminal files. Failed runs are not converted into physical timeouts or successes.

The remaining nine complete rollouts all passed artifact/telemetry audit and timed out at 180 s; none reached a full goal. Thus there are **zero verified safe full goals among 24 declared trials**, with 15 runtime failures and nine measured timeouts. The numbers of complete artifacts for weights 0, 0.02, 0.2 and 0.5 are 2, 0, 3 and 4. There are no eligible safe pairs and **selection/freeze is blocked**, with no selected coefficient and no default-retention or efficiency claim. Missing provenance from crashed trials is reported as unavailable, not proof that they used a different model or runtime.

Read-only audit job **412099** failed before analysis because its bare Python environment lacked NumPy. Retry **412100** used `env.sh` plus `nrd_pychrono`, wrote JSON/CSV/Markdown, then failed at the optional Matplotlib import. Preserved retry **412102** used the explicit `--no-plots` option, completed the audit in 3 s and returned the intended exit code **2** for an invalid grid. Its report is `cohort_02`; the local CPU plot was generated solely from the compact audited report, and its [provenance](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/online_full_validation_grid_v2_cohort_02/plot_provenance.json) verifies that JSON/CSV/Markdown remained byte-identical. No simulation or training was repeated by the reporter.

For later AMD audits, source `/work1/dannegrut/harry/nrd/env.sh`, call `nrd_pychrono`, and add `--no-plots`; the required stack contains NumPy and Torch but not Matplotlib. The reporter verifies the complete candidate-policy object and source hashes, so extra geometry-rejection provenance fields remain covered by artifact hashing and matched-setting checks.

## Candidate rejection and checkpoint comparison support

For corrected-source cohorts, `--verify-decision-rejections` cross-checks every saved decision's rejected-reference list against the checksum-verified planning-summary counter. It checks unique fresh-family origins, geometry-hash format, contract-failure reason, anchor/index coverage and exact counts, and records a digest over all decision files. CSV/JSON preserve per-trial rejected-proposal totals, decisions with rejections and the maximum per decision; arm totals state how many trials have verified counter evidence. These counters describe invalid geometry proposals and do not measure terrain risk.

Later 24-trial declarations may set `model_declaration.checkpoint_kind` to `best` or `last`; omission retains the original `best` rule. `best.pt` must match `status.best_step`, while `last.pt` must match the completed training budget. Both require the exact frozen checkpoint/step, completed status checksum and training-data provenance. `frozen_training.checkpoint_kind`, when provided by the task file, must agree with the selection spec. The report states the chosen kind and step without changing another grid's primary-model declaration.

The selection spec may declare top-level `planning_mode` as `receding` or `plan_once`; omission retains `receding`. Each trial's protocol must match its declared `--planning-mode`, and the full grid must match the spec. Planning mode is part of the paired non-energy settings signature. A 48-trial comparison across two checkpoints must be reported as two separately declared 24-trial grids for energy selection; mixing checkpoints or planning modes into one grid blocks selection. CPU checks cover both checkpoint kinds, incorrect steps/filenames/incomplete status, mixed grids and corrupted rejection counters.

## Completed validation and frozen protected reporting

Corrected receding-grid job **412103** completed all 24 trials in 6 min 21 s; audit **412106** completed in 44 s with zero errors. Safe-goal counts for weights 0, 0.02, 0.2 and 0.5 were **4, 4, 2 and 1 out of six**. Every saved rejection list agrees with its hashed summary: 588 invalid fresh geometry proposals were rejected across 66 decisions. No energy coefficient qualified; the 0.02 comparison had three safe pairs but 6.677% greater mean per-scene work. The default was retained with a failed gate.

The two plan-once grids each retained all 24 trials. Best-checkpoint counts were **3, 2, 1 and 0**; fixed-final counts were **5, 4, 2 and 1**. Neither energy gate passed. The [predeclared final ranking result](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/final_validation_selection_v1/selection.md) selected fixed-final plan-once: its default-weight success count tied receding-best at 4/6, and its zero-weight count won the declared tie break, 5/6 versus 4/6. This selection did not open test data or write the concrete freeze.

The parent subsequently authorized the concrete protected freeze SHA `29684ce59cdd68e9b1f7310dcd2e63c55d021f6ff6712b83f99d286722efeb33`. The separate [`report_traverse_fdm_diverse_protected_cohort.py`](../../../scripts/report_traverse_fdm_diverse_protected_cohort.py) requires this explicit caller-supplied hash and verifies frozen tasks, source/case bytes, checkpoints, costs, observation bytes, collection seal, validation selection and exact six-scene/five-arm membership before decoding protected outcomes. It verifies that execution started after the freeze, checks matched checkpoint modality/budget/pack, and never selects parameters from test results. The shared validation CLI rejects protected tasks before reading batch outcomes; direct per-trial reads require the authorized scene set supplied by the separate gate. [Fake-file tests](../../../scripts/check_traverse_fdm_protected_freeze_gate.py) cover authorization and membership/byte tampering without actual protected data.

Protected physics job **412122** completed in 6 min 48 s. Guarded audit **412124** completed in 16 s with zero errors and retained all 30 trials. The five safe-goal counts were **4/6 RGB-D time, 4/6 RGB-D energy, 3/6 blank time, 3/6 blank energy and 4/6 original-best receding time**. Every remaining trial timed out; failures retain their contact/stall evidence. RGB-D wins three scenes and loses two relative to blank. Across four safe RGB-D pairs, the energy term saves 0.3256% mean per-scene work and reduces time by 0.4782%; these are descriptive protected measurements, not a successful energy gate or a new selection.

The [protected audit](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/online_protected_test_v1_cohort_01/report.md) includes each modality's paired comparisons separately. The [final overview](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/protected_test_visuals_v2/all_30_trials.png) includes every trial plus absolute time/work and fractions for the four safe RGB-D pairs. Its provenance records one exact source/runtime signature across all 30 trials. A [separate checkpoint audit](../../../artifacts/traverse/fdm_diverse_v1_20260909/reports/protected_matched_training_controls_v1.json) confirms identical architecture except modality, normalization, training arguments except modality/output directory, source/data provenance except hostname, and sample/augmentation digests. All report and visualization adapters are independently hash-bound to the concrete freeze before execution; raw simulation and model artifacts remain unchanged.
