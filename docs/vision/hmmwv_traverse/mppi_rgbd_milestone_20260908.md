# RGB-D FDM and MPPI route-choice milestone

The required outcome is a trained RGB-D-conditioned predictor guiding MPPI to a
route that reaches the goal quickly, makes sustained progress, and avoids
collision and stall in Chrono. The previous geometry-only pilot does not meet
this milestone. It remains an explicitly privileged baseline.

## Observation and architecture

- Actual current RGB and depth from the existing overhead camera, with its
  measured calibration. Depth-to-elevation conversion uses sensor data and
  camera geometry only. It must not sample the simulator heightmap.
- Measured recent vehicle/controller history and localization; proposed
  reference geometry and speed. A candidate's achieved future is a target,
  never an input. Goal location remains an external planning input.
- A trained spatial RGB-D encoder, history GRU, candidate command encoder and
  forward GRU over the horizon, following the reference's information flow.
  Motion, contact and low-progress forecasts feed the MPPI cost. This predicts
  candidate outcomes from current perception; it does not predict future images.
- The reference's original quadruped commands are adapted to HMMWV reference
  paths and speeds under a fixed Chrono PID driver. No pretrained quadruped
  weights are represented as HMMWV-trained weights.

The learned scorer and route generator must not receive authored obstacle
locations or BMP elevations. Kinematic checks may reject impossible steering,
speed or arena-boundary proposals; they do not establish obstacle clearance.
For the first matched-driver experiment only, the standard PID's original
terrain-height lookup constructs its 3D tracking curve inside execution. That
controller privilege is reported explicitly and is unavailable to the scorer.

## Evidence required

1. **Input and training gate:** actual RGB-D is loaded and causally aligned;
   both RGB and depth have a differentiable path to predictions; source,
   calibration, split and runtime are recorded. All optimizer updates run on
   AMD. RGB-only, depth-only and blank-image arms are declared diagnostic
   controls, not substitutes for the required RGB-D model.
2. **Prediction gate:** evaluate future contact and low progress separately
   from recognizing existing contact/stall. Report independent episode counts,
   false accepts, predicted versus actual progress and the effect of shuffled
   or blank images. Pooled AUC alone does not pass the gate.
3. **Route-choice gate:** predeclare development scenes and geometric candidate
   families before examining model scores. Include a visible obstacle crossing
   versus detour and a difficult terrain crossing versus detour. From the same
   initial context, physically execute the RGB-D-selected route and matched
   alternatives with the same controller. Report goal completion, elapsed
   time, goal progress, contact and sustained low progress, plus selection
   regret among measured alternatives. An unsafe fast route cannot count as
   an improvement. Abstention remains in the denominator.
4. **Demonstration gate:** save actual rendered observations and executed
   trajectories with predictions and route-choice rationale. A lower predicted
   cost is not evidence of a better physical outcome. A successful small
   development demonstration is distinguished from terrain generalization
   and continuous online replanning.

The first optimizer uses several geometric route families and refines the best
distinct families with MPPI. Its cost combines estimated time to goal, predicted
goal progress, contact risk and low-progress risk. Remaining travel time beyond
the trained horizon is a declared heuristic, not a trained full-route outcome.
Risk weights are fixed before physical evaluation and are not tuned on its
outcomes. Short-horizon evaluation cases must place the decision-relevant
hazard within the model's actual launch reach.

Initial declared costs use 60 s penalties for contact and low progress, plus
estimated time and a 0.1 s/m terminal-progress term. Candidates exceeding 0.35
predicted contact probability or 0.5 predicted low-progress probability are
rejected; if all candidates exceed the limits, planning abstains. These are
fixed operating choices, not established probability-calibration guarantees.

## Data and simulator boundaries

Reuse the frozen train/validation identities from the existing pilot for the
first matched observation ablation. Preserve all protected test identities.
The full raw stores contain per-frame RGB-D. Some newer failure banks contain
only frame-zero images; those cannot supply current images for later anchors.
Collect focused standard-PID RGB-D failures and sibling alternatives where
existing recordings cannot support prospective failure/ranking evaluation.

Legacy OptiX depth has a measured ray scale around 1.2. The AMD Vulkan renderer
uses a different camera implementation; calibration and visual domain transfer
must be verified and recorded, not assumed. Simulator truth may supply labels
and evaluation outcomes but must not enter perceptive planning inputs.

All source changes and outputs stay in the `traverse_mppi` worktree and its
isolated AMD experiment. Existing main-checkout jobs and files remain untouched.
