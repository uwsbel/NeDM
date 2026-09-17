**Post-hoc diagnostic of frozen RGB-D FDM / MPPI decisions**

The four failed protected trials contain two different problems. Rolling-hills contact occurs about 30 seconds after launch, beyond the 12-second forecast; the initial finite forecast cannot establish safety of the full reference. Rough mosaic also has a genuine near-term false acceptance: the time arm assigns 11.6% contact risk despite measured chassis contact 11.70 seconds later. The energy arm accepts again at 2 seconds while its later collision is inside the horizon.

Later evaluations of the unchanged reference often reject it before contact. These judgments were never executed in plan-once mode, and they do not prove that a particular replanning or stopping policy would succeed. The earliest rolling-hills rejection is actually too early for the measured contact label: its 12-second future contains no contact or bounded-motion event.

| Scene / arm | First contact | Launch contact / bounded probability | Contact within launch 12 s | First audited rejection | First audited rejection with actual hazard in 12 s |
|---|---:|---:|---|---:|---:|
| rolling_hills / time | 29.70 s | 12.7% / 1.7% | False | 11.45 s | 17.75 s |
| rolling_hills / time + work | 30.45 s | 12.8% / 1.7% | False | 11.45 s | 18.50 s |
| rough_mosaic / time | 11.70 s | 11.6% / 1.4% | True | 3.80 s | 3.80 s |
| rough_mosaic / time + work | 12.50 s | 12.9% / 1.4% | False | 0.50 s | 4.00 s |
| cross_slopes / time | — | 8.9% / 1.3% | False | — | — |
| cross_slopes / time + work | — | 9.0% / 1.4% | False | — | — |

[Causal risk curves](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/decision_audit/causal_risk.png) · [Position and reference-tracking errors](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/decision_audit/forecast_and_tracking_error.png) · [Complete numerical audit](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/decision_audit/audit.json)

The rough-mosaic energy arm has a transient rejection at 0.50 seconds, then accepts at 2 seconds and rejects again later. A single early rejection is therefore not a monotonic or calibrated warning. Successful cross-slope controls remain accepted at every audited anchor through 32 seconds. All supported risk heads are present. The deployed caps are contact 0.35 and bounded motion 0.50; rollout/attitude masks and limits come from each frozen protocol.

**What MPPI changed.** Final weighted means are re-scored through the same frozen eligibility gate; these failures are not caused by an unchecked mean. The reported final-reference forecasts are distinct from saved pre-refinement family scores.

| Scene / arm | Parent model cost | Refined model cost | Best sibling with a similar previously safe fixed route | Maximum waypoint / speed change |
|---|---:|---:|---:|---:|
| rolling_hills / time | 45.555 | 45.477 | 45.836 | 0.774 m / 0.021 m/s |
| rolling_hills / time + work | 59.726 | 59.627 | 65.762 | 0.773 m / 0.011 m/s |
| rough_mosaic / time | 48.721 | 46.475 | 47.608 | 1.031 m / 0.013 m/s |
| rough_mosaic / time + work | 66.388 | 62.394 | 68.385 | 1.149 m / 0.167 m/s |
| cross_slopes / time | 53.531 | 53.363 | 53.531 | 0.601 m / 0.000 m/s |
| cross_slopes / time + work | 69.799 | 69.530 | 69.799 | 0.894 m / 0.013 m/s |

Matched physical replays now establish one avoidable ranking failure in rough-mosaic/time. The exact original best unrefined family 12 (−44 m at 6 m/s; model cost 47.608) safely reaches the goal in 41.30 s. MPPI instead refines parent family 6 (cost 48.721) into a route with lower predicted cost 46.475, making it the winner; that route contacts within 12 s, blocks and times out at 180 s. The best-unrefined intervention was selected by the original model-cost argmin, before its physical outcome was known. This is a harmful ranking change in this case, not evidence that removing MPPI improves general performance.

All nine controlled trials passed artifact/source/runtime/input checks, and all four selected-reference controls matched every one of 241 stored physical arrays exactly. All four selected parent references were already unsafe: the rough-mosaic parent reaches the goal at 120.25 s after contact and blockage, while its refinements time out; rolling-hills parents and refinements both time out after contact/blockage. Thus refinement worsens the rough-mosaic outcome but does not create its initial unsafe classification, and small reference changes are not necessary for every failure. These interventions change the executed reference, including geometry and speed; they do not isolate lateral deformation alone.

[Verified physical counterfactual report](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/refinement_summary/report.md) · [Counterfactual cost and route figure](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/refinement_summary/ranking_counterfactual.png) · [Physical checks and results](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/refinement_summary/report.json)

Previously collected safe routes exist at both ±44 m on rolling hills,−44 m on rough mosaic, and+44 m on cross slopes. They are contextual feasibility evidence: their waypoints differ from the actual online siblings by roughly millimeters, and their controller/collector provenance differs. No existing sibling outcome is labeled an exact counterfactual for a later measured state. The audit retains exact array comparisons, launch differences, prior source hashes and a separate 2 m/15-degree proximity flag for conditional sibling queries.

[Every launch candidate and cost component](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/decision_audit/launch_cost_decomposition.csv) · [Compact table](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/decision_audit/summary.csv)

**Reconstruction and limits.** We use the unchanged protected LAST 5000 checkpoint and all archived online_v9 source bytes. Frame-zero measured pose/state and reconstructed history match the recorded decision exactly; replayed saved base-family scores agree within 0.001 (the observed differences are recorded). CPU runtime versions are pinned in provenance. The decision re-evaluation performs no training, model selection or physics stepping. The separately linked counterfactuals add controlled physical measurements without changing the original protected evaluation.

Inputs are the same single pre-drive RGB-D snapshot, measured pose and causal 17-state/action history, supplied goal, and frozen references. Outputs are compared only with actual future measurements along the unchanged plan-once reference. Every terminal/censored mask is retained. Bounded-motion supervision requires a full 2-second window, so outputs before 2 seconds are ineligible. The exact frozen scorer also removes predicted post-goal parking. Raw early probabilities are saved only as diagnostics.

Anchor selection is post-hoc: 0, 2, 4, … seconds, measured hazard-minus 12 s and onset boundaries±0.05 s, then extra measured frames around the first audited accept/reject transition. It is not an exhaustive 20Hz alarm scan; “first audited” must not be read as the earliest possible crossing. Future outcomes select diagnostic sample times and labels, but never enter model features. Later sibling queries are conditional predictions, not new physical counterfactuals.

Native-spline inspection isolates interpolation from actual vehicle deviation. Exact frozen 3-D spline knots and native ChBezierCurve were reconstructed in memory with no vehicle or simulation step. Rough-mosaic/time stays within 0.079 m of this curve before contact (0.005 m at contact), whereas rough-mosaic/energy reaches 0.902 m at first contact. Spline-to-command-polyline differences are only 0.0043 m and 0.0029 m. The planner uses a 1.3 m footprint half-width; small centerline tracking error does not establish footprint clearance. Rolling-hills trajectories deviate about 4.38 m/4.16 m before returning within 0.034 m of the reference at contact. Cross controls stay within 0.168 m. The native tracking measurements are separate from decoder endpoint error, which can be tens of meters.

[Native spline and tracking audit](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/decision_audit/native_curve_audit.json) · [Read-only native geometry source](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/decision_audit/audit_native_curve.py)

Each trial folder contains anchor_records.json, launch_candidates.json, summary.json and causal_forecasts.npz. The NPZ retains all forecast/target masks and selected commands, nominal_pose and global_features for independent perception/support auditing. Forecast errors can grow well before contact while physical reference tracking remains comparatively close; the plotted waypoint-polyline distance is not a full tire-terrain or PID-path diagnosis.

[Reconstruction source](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/decision_audit/audit_decisions.py) · [Reporting source](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/decision_audit/report_decisions.py) · [Source, checkpoint and runtime provenance](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/decision_audit/provenance.json) · [Downloaded raw artifact hashes](/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_failure_investigation_20260909/decision_audit/inputs/download_manifest.json)
