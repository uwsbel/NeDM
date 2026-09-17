# FDM / MPPI failure investigation

The completed post-hoc diagnosis separates finite-horizon coverage, incorrect hazard prediction, optimization-induced ranking errors, and scene-specific online replanning effects. Original protected results remain unchanged; no new training was performed.

- [Main findings and recommended controlled changes](../../../docs/vision/hmmwv_traverse/fdm_failure_investigation_20260909.md)
- [Physical counterfactual: safe original candidate versus failed MPPI winner](refinement_summary/ranking_counterfactual.png)
- [All nine reference interventions and exact replay controls](refinement_summary/report.md)
- [Same-checkpoint online versus plan-once comparison](matched_modes_v1/findings.md)
- [Causal forecast and controller-tracking audit](decision_audit/report.md)
- [Sensor visibility, candidate patches and exact training exposure](perception_support/report.md)
- [Risk timelines](decision_audit/causal_risk.png)
- [Investigation protocol](protocol.md)
- [Machine-readable progress](progress.json)

All 24 new online trials and 24 existing controls are retained. Their launch decisions and first-second physical prefixes match exactly. All four selected-reference replay controls reproduce all 241 physical arrays exactly; parent and best-original-reference interventions use the same native Chrono PID and scene runtime.

Working files are isolated in /home/harry/NeDM-traverse_mppi on branch traverse_mppi. Physical runs used AMD under /work1/dannegrut/harry/experiments/fdm_failure_investigation_20260909. Reopened test scenes are explicitly diagnostic, not a new protected benchmark or configuration selection.
