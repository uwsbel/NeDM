# Larger-terrain RGB-D FDM / MPPI campaign

The goal is measured long-distance traversal across diverse terrain, with time,
collision/stall risk and mechanical energy in the planning objective. The
accepted finite-horizon FDM design remains: a measured RGB-D map and causal
vehicle history condition candidate-command forecasts. No future-image model is
introduced. The narrow four-route demonstration is retained as development
evidence, not counted as generalization success.

## Staged milestones

1. **Collection contract and throughput.** Verify flat launches, heightfield
   orientation, global camera coverage/metric depth, shared settled anchors,
   passive telemetry, and rendered/headless physics parity on a small pilot.
   Benchmark before selecting worker count. No full cohort is sealed with
   missing/failed runs.
2. **Diverse data.** Generate 240 m square heightmaps with six terrain families,
   24 training layouts, six validation layouts and six protected test layouts.
   Each has a flat launch pad and a separate flat destination pad. Run 15
   geometric route/speed siblings per layout, including 2/4/6 m/s commands,
   with up to 180 s per execution. Keep failures and sustained attempts.
3. **Train and calibrate.** Train on AMD only, with two seeds and matched RGB-D
   and blank-image controls. Compare the existing four-second formulation with
   a twelve-second finite forecast where compute allows. Select checkpoints,
   thresholds and cost weights using validation only. Report scene-level risk
   calibration, forecast errors and energy/attitude errors, plus positive-label
   support and performance before failures begin.
4. **Closed-loop planning.** Replan from measured causal state/history while
   reusing the initial global RGB-D snapshot. Execute selected reference
   segments through native Chrono PID. Compare time/risk-only and energy-aware
   costs using matched starts, limits, model and candidate budgets; record
   stall, collision, goal time, positive mechanical work and roll/pitch exposure.
5. **Protected evaluation and demonstration.** Freeze the chosen model,
   thresholds, candidate generator and cost configurations before test scoring.
   Report all test arenas, including abstentions/timeouts. Produce at least one
   actual long-traversal video and a route/cost comparison; do not select only
   successful arenas for aggregate metrics.

These are sequential gates, not presumed successes. Dataset size and training
budget can be expanded after measured coverage/throughput checks, with new
versioned cohorts rather than changing a sealed split.

## Observation and physical data contract

One 1024-square RGB-D observation is captured before driving from a calibrated
nadir camera 400 m above the origin (47-degree horizontal FOV, 600 m ray range).
The proposed model input is 512-square registered RGB plus measured elevation
with a fixed 40 m scale. Candidate patches preserve local observed detail;
global context may use a smaller image representation. Raw float metric depth
is retained, avoiding the legacy uint16 camera's limited depth window.

The vehicle may remain visible at its initial location in the fixed snapshot;
it is not a current vehicle detector. Current vehicle pose/history is supplied
by localization/proprioception. This is a prior global sensing assumption, not
onboard exploration or visibility of the full route from the driver's view.
Map capture and headless physics share case hashes, terrain/assets, runtime and
settled state. Preparation rejects unmatched anchors. Future images and authored
heightmaps/obstacle lists are excluded from model inputs and candidate scoring.
Terrain truth remains permitted for simulator construction, PID path altitude
and physical validation, as in the accepted initial implementation.

At 20 Hz, record full pose, velocity/acceleration, roll/pitch and rates,
controller effort/desired speed, gear/powertrain state, per-wheel spin, native
slip, tire forces/moments/load proxies, deflection and available suspension
measurements. At physics substeps, integrate signed/positive mechanical work
at explicitly named engine-interface, engine-rotor and driveshaft locations;
also retain interval contact and roll/pitch maxima. Missing APIs yield explicit
validity/capability metadata, not plausible zero measurements.

Mechanical work is not fuel consumption. RigidTerrain with TMEASY tires exposes
traction, grade response, wheel slip and chassis dynamics. It does not simulate
soil sinkage/ruts; a deformable-soil cohort would be a separately identified
domain. The first cohort keeps known friction fixed so hidden material changes
are not falsely presented as inferable from identical RGB-D.

## Reuse, isolation and fairness

The AMD audit found 6,821 richer prior traversal episodes across 13 arenas and
the original focused FDM packs. The prior records have different controller,
runtime and sensor contracts and lack some new substep quantities. Reuse is
conditional on those contracts; never silently mix their protected tests or
invent missing labels. The focused pack remains a regression reference. New
large-terrain collection uses the existing verified Chrono runtime and native
PID implementation.

All edits, outputs and job source snapshots are isolated from the busy main
checkout under `traverse_mppi`. Existing experiments and checkpoints are
preserved. Physics, rendering and all optimizer updates run on AMD. Job arrays
or bounded process pools parallelize independent trajectories and model arms;
no compute-intensive work runs on login nodes.

Numerical acceptance criteria and risk/cost settings will be declared after
pilot data support is measured and before protected test evaluation. Faster
arrival is compared only among safe completed routes; energy comparisons also
require matched goals and completion, so a stuck vehicle cannot win by using
less energy. Report the time/energy tradeoff, not a presumed universal optimum.
