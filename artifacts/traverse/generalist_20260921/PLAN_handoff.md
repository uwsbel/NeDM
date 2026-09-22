# Shared rigid/CRM planner and learned tracker

Date: 2026-09-21. Planning handoff only; no training or simulation launched.

## Objective and scope

Accept the existing rigid and CRM specialist planner results as the baseline. Build two independently testable advances: (A) one risk model/planner that adapts to either domain from observation history; (B) one neural low-level tracker trained through NRD that beats the current PID in Chrono. Preserve the current planners/checkpoints and work in an isolated experimental checkout. Run substantial collection/training on AMD; local work is for bounded integration checks.

Start with the familiar arena and fixed initial depth map in each domain. Keep CEM and its route family fixed while testing model conditioning. Retain terrain geometry throughout: domain identity describes mechanics, not hills or craters. Success on these two configurations establishes cross-domain sharing; it does not by itself establish generalization to unseen soil parameters or mixed-terrain routes.

## A. One adaptive risk model

### Proposed interface

`past observable states + past applied controls -> history encoder -> context z`

`candidate terrain corridor + proposed speed profile + current state + z -> station hazards -> route risk`

Compute history context once at the actual planning decision and share it across all CEM candidates. The candidate-specific corridor and speed profile still differ. Begin with a small GRU/temporal CNN and approximately 1-2 seconds of causal history. Include body motion/attitude, wheel speeds and applied steering/throttle/brake where available. Simulator-only forces, future slip, future outcomes and hidden soil state must not enter the deployable history branch. Mask missing startup history.

The deployed specialist risk networks currently use corridor/speed and geometry context; they are not already history-conditioned physical dynamics models.

### Implementation order

1. Build one shared risk model on domain-balanced rigid/CRM batches, first with a known two-class domain tag. This oracle-context reference tests whether a common predictor can preserve the specialists' behavior. Also retain a pooled model without the tag as a cheap diagnostic.
2. Replace privileged context with a history encoder. Compare explicit adaptation (history estimates a soft domain probability or small task-relevant latent learned by the privileged teacher) against direct history conditioning trained with the risk loss. Keep data, splits, backbone and search budget comparable. Domain classification is a useful auxiliary target; do not force all history information through a hard 0/1 decision.
3. Use recorded trajectories to construct initial causal windows and remaining-route labels. Keep entire episodes, overlapping windows and paired rigid/CRM route groups in the same split. Do not train on the held-out planner evaluation missions.
4. Add a bounded set of moving-prefix branches: from the same observed history, drive alternative route/speed continuations. The existing re-anchored data showed survivorship bias, so a moving vehicle in old recordings is not proof that increasing speed causes safety. Include low-progress cases as well as clean moving prefixes. In CRM, reproduce the prefix from fresh soil or restore a full simulator checkpoint including soil; moving only the vehicle onto fresh terrain does not reproduce the same anchor.
5. Test deployment without domain labels. At startup, allow an unknown/uncertain context and a conservative choice or a common short approach segment; do not demand identification before informative motion. History identifies recently contacted ground, so arbitrary unseen soil ahead is outside the first milestone.

### Milestone A

One shared model with no domain label at deployment preserves specialist goal-reaching performance on each domain under the same controller and CEM budget. Predeclare the acceptable gap (suggestion: 2 percentage points), use a frozen paired held-out suite, and report uncertainty rather than claiming non-inferiority from an underpowered comparison. Report each domain separately, startup versus established history, route-ranking quality, calibration and total decision latency. Classification accuracy alone is not success.

## B. NRD-trained neural tracking policy

### Reuse the closest existing code

- Original mixed-domain dynamics and PPO: `/home/harry/NeDM/scripts/training/train_hmmwv_dynamics.py`, `scripts/training/train_hmmwv_rl_tracking.py`, `scripts/evaluation/eval_hmmwv_rl_chrono_tracking.py`, and `docs/rl_tracking.md` in that checkout.
- Arena adaptation already present here: `src/nedm/traverse/nrd_model.py`, `nrd_data.py`, `tracker_env.py`; `scripts/traverse_wp2_train_map.py`, `traverse_wp3_train_tracker.py`, and `traverse_wp3_chrono_eval.py`.
- Prefer the arena route-following environment and borrow the original domain conditioning/mixed-data training. Do not rebuild the old flat-ground timed-reference interface unnecessarily.

### Implementation order

1. Convert arena recordings into an action-conditioned dynamics dataset, preserving the local depth-derived terrain crop, pose, state history, applied controls, terminal endpoints and domain metadata. Use desired waypoints/speeds from `command_reference.npz` for tracking targets; recorded stalled PID motion is a dynamics example, not a desired trajectory.
2. Audit timing before treating old recordings as valid action transitions. The original generic model is 15-D at 10 ms; arena recordings are 17-D at 50 ms. Current collectors update PID inside physics substeps but log its action at the beginning of each 50 ms interval. Verify the actual applied-action convention, and quantify or correct the mismatch. A new 50 ms zero-order-held controller needs matching dynamics data; do not invent unrecorded substep actions. Keep 20 Hz policy control (action-repeat 1 for a 50 ms NRD), correct physical/normalized delta units, and matched actuator limits.
3. Train a mixed rigid/CRM NRD with a domain tag first. Keep terrain geometry as a separate input. Use the available recordings, then add targeted data with bounded perturbations of steering/throttle/brake around PID and early learned policies. Cover turns, slopes, braking, stalls, rollback and feasible recovery. Collection must preserve CRM coupling and the agreed action-hold convention.
4. Before expensive policy training, validate free multi-step rollout on held-out recorded actions, including signed displacement, speed/yaw response and stalled versus moving cases. Compare measured-state input with recurrent predicted-state feedback. Prior NRD work falsely predicted escape from stalls; good one-step loss alone is insufficient. Validate the model on the rollout duration actually used for RL.
5. Reuse PPO in the frozen NRD. Begin with short 1-3 second imagined fragments, initialized from real contexts, and use only horizons supported by the model audit. A PID-imitation warm start is reasonable; keep a standalone neural policy as the target. Policy inputs should contain observable state/history, route preview, target speed and previous actions. An oracle-tag policy may be a diagnostic, with history-conditioned deployment evaluated separately. Periodically collect learned-policy Chrono failures, update the dynamics dataset, refit NRD, and retrain the policy.

### Milestone B

The neural tracker beats the current PID in actual rigid AND CRM Chrono rollouts on identical held-out desired routes/speeds, physically consistent starts, terrain resets, sensing information and actuator limits. Judge fixed-route tracking before changing planner choices.

Predeclare tracking improvement (suggestion: at least 10% lower paired mean cross-track RMSE in each domain), with no material regression in completion, speed tracking, stalls or safety. Report paired uncertainty, tail errors, heading error, action smoothness and work. Do not drop failed runs to make average tracking error look better. Imagined reward or imitation accuracy is not the milestone. If residual PID+NN control is tried as a fallback, label it as a hybrid intermediate result rather than full PID replacement.

## Integration dependency

Run A initially with PID fixed; run B with reference routes fixed. Existing risk labels describe execution under PID. Once the learned tracker passes B, collect matched route outcomes under that tracker and recalibrate/retrain the shared risk model before claiming success for the combined stack. A stronger controller may change which routes are traversable. The history encoder can later be shared, but tying both learners together is not required for the first milestones.

## Established precedents and limits

- [UP-OSI, RSS 2017](https://faculty.cc.gatech.edu/~turk/paper_pages/2017_learning_universal_policy/index.html): state/action history estimates physical parameters for a universal policy; precedent for explicit identification.
- [RMA, RSS 2021](https://arxiv.org/abs/2107.04034): privileged environment factors train a useful latent, then history estimates that latent; precedent for teacher-to-history adaptation, demonstrated in legged control rather than route-risk prediction.
- [Miki et al., Science Robotics 2022](https://leggedrobotics.github.io/rl-perceptiveloco/): recurrent belief fuses proprioception and terrain perception; supports direct implicit history conditioning.
- [Levy et al., RSS 2025](https://arxiv.org/html/2504.16923v1): online dynamics adaptation feeding MPPI on a full-scale off-road vehicle. Closest vehicle precedent; it adapts dynamics rather than this project's route-risk scorer.
- [MBPO, NeurIPS 2019](https://arxiv.org/abs/1906.08253): short model rollouts branched from real data limit model-bias exposure. This motivates the proposed validation/training pattern; reusing PPO is not an implementation of MBPO.

Existing local evidence: `artifacts/traverse/crm_night2_v1/REPORT.md` section 2 documents moving-anchor bias; `/home/harry/NeDM/docs/vision/hmmwv_traverse/stall_independent_audit_20260908.md` documents recurrent false recovery. These motivate targeted checks, not another reproduction of the established specialist planner runs.

