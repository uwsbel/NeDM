# FDM / MPPI failure investigation

This is a post-hoc diagnosis of existing experiments, not a new protected benchmark or a new model selection. Original frozen outputs and the main checkout remain unchanged. No training is currently planned.

Questions:
1. Is the claimed success drop comparable across the small development demo and the diverse scene-disjoint benchmark?
2. Do errors arise from hazards outside the finite forecast, in-horizon false accepts, inappropriate cost extrapolation, or unchecked MPPI output?
3. Does MPPI refinement turn a physically safe base family into a failed trajectory?
4. With the same final 5,000-update checkpoint, how does1 s online replanning change outcomes relative to plan-once?
5. Are actual hazards represented in the measured RGB-D and sampled candidate patches, and are fresh failure states adequately exposed during training?

Matched online experiment: all six validation and six already opened test scenes, weights 0 and 0.02, fixed final 5,000-update RGBD seed11, unchanged online_v9 source and runtime; compare with existing plan-once trials. Test additions are explicitly diagnostic.

Decision audit: all four failed RGB-D trials plus two successful cross-slope controls. Recompute causal forecasts of the executed reference from measured anchors, join exact future telemetry, apply the deployed event and goal masks, and examine parent/refined and alternative scores. Existing outcomes for different commands are not valid later-state counterfactuals.

Refinement counterfactual: replay each of the four original failed selected references as an exact physical parity control, and execute its exact selected pre-refinement parent family using the same launch state and native online PID adapter. Eight headless AMD trials; only the launch reference is intervened on. Frozen simulation files are imported unchanged; a separately hashed diagnostic wrapper substitutes the recorded launch decision. All physical arrays must reproduce in each replay control before interpreting the paired parent outcome.

Perception/data audit: inspect actual RGB-D projection, candidate patch coverage and fresh pre-failure training support. Distinguish observed defects from untested capacity, exposure and distribution-shift hypotheses. A selected small demonstration is not an unbiased success-rate baseline.

Deliverables: reports with measured timelines, matched results, counterfactual checks and ranked conclusions. No model/cost tuning, no edits to original test freeze, and no new claim of held-out generalization from these diagnostic runs.
