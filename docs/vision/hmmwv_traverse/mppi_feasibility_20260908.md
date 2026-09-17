# FDM-style HMMWV trajectory prediction and MPPI: feasibility assessment

This is the initial assessment. The subsequent [RGB-D implementation and
measured route-choice results](mppi_rgbd_progress_20260908.md) demonstrate the
initial proposal and record its remaining limits.

**Recommendation: proceed with a separate, planner-oriented NRD variant.** Learn the short-horizon response to a proposed reference path and speed profile under a fixed Chrono PID driver. Predict actual motion and failure risk, then score progress toward the goal analytically. This is a credible way to reduce the long physical-state rollout failure observed in the current system. Improvement remains an experiment, not an established result.

This assessment inspected the local paper, FDM implementation, current NeDM implementation/results, actual data payloads, and AMD availability on 2026-09-08. No training, simulation collection, or production code changes were made. The isolated worktree is `/home/harry/NeDM-traverse_mppi`, branch `traverse_mppi`, based on `4b6f2a1`. It does not automatically include the active checkout's uncommitted state-ablation and simulator changes. Source hashes and observation provenance accompany this note in `mppi_feasibility_evidence_20260908.json`.

**What the paper actually does.** The principal FDM is a finite-horizon, command-conditioned predictor, not a learned scalar reward or critic. Current terrain geometry and recent proprioception are encoded, combined with a candidate sequence of body velocity commands, and processed by a forward GRU. The heads predict corrected velocities and failure probabilities. Corrected velocities are integrated into future SE(2) poses. A goal-distance and risk cost is computed outside the model for MPPI.

The default horizon is ten commands at 0.5 s, or 5 s; history is ten observations at 0.05 s. The main multi-step implementation repeats the observed context at each command step and jointly decodes the sequence of GRU features. It does not feed predicted physical state or future perception back into each prediction step. Thus, **internal neural recurrence remains, while physical-state autoregression is avoided**. The paper's prose about initializing the GRU differs slightly from the actual context-concatenation implementation; its appendix gives the implemented dimensions. Energy/friction modules also exist in the code, but the default energy loss weight is zero.

Supervision combines pose/heading errors, failure classification and a stopping loss. This supports learning a platform-and-controller response. It does not establish that scalar reward regression alone is sufficient. The code's default simulation MPPI uses 1,024 correlated, bounded samples and a weighted action mean; the paper reports 2,048 deployment candidates. Neither their inference speed nor their controller frequencies should be assumed for HMMWV.

Sources: local PDF, physical pp. 4–7 and Appendix C; [official paper](https://arxiv.org/abs/2504.19322), [project](https://leggedrobotics.github.io/fdm.github.io/); local FDM commit `06f5d8033bafed4f6838662af99ba31cbd4f5ce0`. Implementation: [model forward](/home/harry/NeuralReducedDynamic-LiteratureReview/off-road-nav/fdm/exts/fdm/fdm/model/fdm_model.py:936), [horizon/config](/home/harry/NeuralReducedDynamic-LiteratureReview/off-road-nav/fdm/exts/fdm/fdm/model/fdm_model_cfg.py:129), [planner cost configuration](/home/harry/NeuralReducedDynamic-LiteratureReview/off-road-nav/fdm/exts/fdm/fdm/planner/sampling_planner/planner_cfg.py:24), [MPPI update](/home/harry/NeuralReducedDynamic-LiteratureReview/off-road-nav/fdm/exts/fdm/fdm/planner/sampling_planner/trajectory_optimizer_mbrl.py:1112).

**Why it is worth trying here.** Current NRD already scores route-and-speed candidates through controller/model rollouts and uses CEM refinement. Its deployed visual representation is sampled from a fixed observed map at predicted poses; it is not a recursively predicted visual latent. The remaining recurrence includes the physical state and pose-to-map feedback.

The Sept8 audit found that, over 48 already-stalled four-second windows, actual mean displacement was 0.315 m while frozen NRD predicted 4.426 m. False completion also occurs with native PID, including BMP/registered-map diagnostic arms. Removing the learned tracker or swapping CEM for MPPI alone therefore does not address the demonstrated failure mechanism. A horizon predictor has a plausible advantage because it can learn displacement and failure directly without needing accurate recurrent wheel-force, wheel-speed and engine-state predictions. It can still be wrong, especially under missing counterfactual controls.

There is already a useful precedent in this repository: [WP7 direct predictor](/home/harry/NeDM/scripts/traverse_wp7_cheap_predictor.py:43) and [WP9 direct predictor](/home/harry/NeDM/scripts/traverse_wp9_arm_cheap.py:81) score proposed routes from terrain profiles, reference speeds and initial state.

| Existing evidence | Direct terrain/profile predictor | Head on imagined NRD trajectory |
|---|---|---|
| Historical f108–f111 stall AUC, three seeds | 0.814 / 0.815 / 0.811 | 0.757 / 0.759 / 0.752 |
| Successful selections among 187 layouts with a feasible alternative | 155 / 158 / 155 | 152 / 152 / 152 |

The fastest-speed heuristic achieved 153/187. These counts were recomputed from saved predictions and labels. They support direct prediction on this terrain family, but use privileged true terrain and are not proof of camera-only navigation or MPPI performance. These arenas are now development data. [Result discussion](/home/harry/NeDM/docs/vision/hmmwv_traverse/wp4_implementation_notes.md:2034), [saved scores](/home/harry/NeDM/artifacts/traverse/wp8_head/results_sealed2.json).

Energy evidence is more limited: the analytic scorer beat NRD, while direct learned scoring and NRD were approximately tied on cost. On 385 layouts with true compliant candidates provided, selected positive work averaged 209.20 kJ analytic, 212.77 direct learned, 212.88 frozen NRD and 210.85 fine-tuned NRD. These oracle-filtered results do not establish end-to-end planner success. The direct model's reported 67.3% full-bank compliance is 342/508 selections with 39 abstentions among 547 layouts; it is not a same-denominator success advantage over analytic scoring's 343/547. Keep the analytic baseline. [A1 readout](/home/harry/NeDM/artifacts/traverse/wp9_energy/a1_table.txt:8).

**Proposed model/controller contract.**

```text
Observed terrain/obstacle map + measured recent vehicle/controller history
             + candidate reference path and speed profile
                              |
                    short-horizon NRD-FDM
                              |
       predicted XY/yaw/progress + failure probabilities + optional work
                              |
              goal progress + risk + control-smoothness cost
                              |
        constrained MPPI -> execute a short reference interval
                              |
              fixed Chrono PID path-following driver
                              |
                  new measurements -> replan
```

Start with the standard `ChPathFollowerDriver` to reuse its broad route data, keeping its gains, 3D reference-path construction, initialization, speed updates and physics-substep advancement. The newer `NativePID` wrapper is a second, distinct controller domain: it uses a planar frame and holds outputs at 20 Hz. Equal nominal gains do not make the two implementations interchangeable. Use the same controller in training-data interpretation and validation, or explicitly model and evaluate controller identity.

Represent candidates as vehicle-feasible reference geometry plus speed, not ANYmal's independent lateral-velocity and yaw-rate commands. The network predicts where the HMMWV will actually travel under that reference. The commanded path is known input; the achieved future path is a target. The goal can remain outside the network so the same predicted trajectory can be scored against different goals.

Proposed starting settings are a 0.8 s measured history, a 2–4 s prediction horizon with 0.2 s output spacing, and execution of only 0.2–0.5 s before replanning. These are pilot choices to benchmark, not validated real-time settings. Preserve the chosen driver's existing physics/control timing. Include the driver's persistent state when available, or measured causal history as a partial-observation approximation; resetting/replacing paths online needs its own collection coverage.

At HMMWV speeds, the model must observe the candidate corridor over the full horizon. For example, 4 s at 8 m/s spans 32 m before footprint/margin allowance. An initial ±5 m terrain crop is insufficient. Encode map features along the *proposed* path or use a suitably sized spatial map; do not read true future vehicle poses to choose input crops. Start with BMP geometry as an explicitly privileged dynamics baseline, then compare correctly registered RGB-D. BMP elevation alone does not observe placed rocks/trees: retain a separate obstacle/footprint check and report asset-contact cases separately until the observation arm supports them.

**Reuse the data before recollecting.**

| Data | Verified usable content | Initial role and limitation |
|---|---|---|
| `full_v1`–`full_v4` raw stores | 12,000 unique episodes, 4,797,627 recorded 20 Hz frames; 9,600 nominal path/speed episodes and 2,400 actuator-meander episodes; rich 105-field telemetry in inspected payloads | Broad standard-driver route/history/motion training on one arena. The 2,335 `full_v4_partial` records duplicate `full_v4`; exclude them. |
| `wp7_cache_v1`, `wp7_cache_sealed`, `wp8_cache_sealed2` | 5,745 episodes across 11 arenas; state17, actions, world pose, power, route geometry/speeds and outcome/contact summaries | Multi-terrain direct-scoring baseline and recorded-action diagnostics. Much of this is learned-tracker data, not interchangeable standard-PID response data. |
| Sept8 state-ablation bank | 6,821 records with 56 selected states; its separate original-Chrono10 repeat additionally exposes 129 raw numeric fields | Separate failure-rich matched-controller study; native-PID train 616 episodes/154 layouts, validation 48/12. PID cases are straight crossings with v4/v6 and speed-switch/retry siblings. |

The older broad `complete` status means the recording survived its 20 s interval, **not goal arrival**. Build horizon displacement/yaw/work from telemetry; reconstruct task success from route-end geometry and poses. Inspect event timestamps and masks for contact labels. The newer 5,745-cache endpoint outcome is also not a per-step collision annotation. Wheel unloading alone is not asset collision or an absorbing failure.

For low progress, use actual net displacement and commanded effort over a declared interval; deliberate stopping must not become a failure target. Keep rollback motion and distinguish sustained low progress from terminal contact/rollover. For censored horizons, mask unavailable future pose/work targets and unknown future event negatives, while retaining positive supervision for failures already observed before truncation. Do not copy FDM's absorbing collision-pose convention onto reversible stalls. Positive shaft work must be integrated from positive power, not the legacy signed-energy target.

Split by arena/layout/episode before extracting overlapping windows; keep sibling controls together. The legacy WP2 manifest has 9,518 entries split 6,662/1,427/1,429 across training/validation/test, including `full_v4_partial` aliases and a pilot store. Preserve its existing layout assignments when adding remaining `full_v4` records: rerunning the current permutation-based splitter on a larger inventory would move old layouts between splits. Canonicalize partial/full aliases first. Never reconstruct proposed future reference speeds from the *realized future* station or use recorded reactive throttle as if it were a planned high-level command. At an arbitrary history anchor, provide the original known path/profile and causal current progress. Test controller domains separately.

Data implementation references: [standard-driver collector](/home/harry/NeDM/scripts/traverse_collect.py:96), [collection timing/labels](/home/harry/NeDM/scripts/traverse_collect.py:126), [raw example metadata](/home/harry/NeDM/artifacts/traverse/full_v4/ep_0000_spline/meta.json), [legacy split manifest](/home/harry/NeDM/artifacts/traverse/wp2_z2_cache_v6/cache_manifest.json), [WP7 cache contract](/home/harry/NeDM/scripts/traverse_wp7_build_cache.py:149), [new PID task construction](/home/harry/NeDM/scripts/traverse_state_prepare.py:63), [new PID implementation](/home/harry/NeDM/src/nedm/traverse/state_controller.py:10), [expanded-state manifest](/home/harry/NeDM/artifacts/traverse/state_ablation_20260908/data/manifest.json).

No bulk recollection is needed to establish a direct-motion baseline. Targeted new data will be needed for repeated path changes, curved failure/recovery maneuvers and MPPI-induced controls beyond the logged support. Train on random feasible perturbations first, then collect bounded planner proposals under the same driver. Preserve failed attempts and multiple controls from comparable starts. Freeze one Chrono runtime for each comparison; the newer AMD build changes some borderline outcomes, while an original Chrono10 CPU runtime is also available on AMD.

**Smallest informative experiment.** First validate labels and evaluate identical real starting histories and recorded candidate prefixes over the same 2–4 s horizon. Compare a horizon-trained route/profile baseline, analytic cost, current recurrent NRD truncated to that horizon and a history-conditioned finite-horizon pose/risk predictor using the same observation/controller domains. Report horizon failure acceptance at matched feasible-prefix retention, pose/progress error on pre-stall and moving controls, calibration and interval work. Do not assign a later whole-route timeout to an earlier successful prefix. Single logged continuations support prediction tests, not counterfactual ranking; prefix ranking requires sibling trials from comparable starting contexts, or new branched collection. Whole-route success, abstention-inclusive selections and realized route regret belong to later closed-loop Chrono evaluation or a separately trained full-route outcome head. Existing NRD checkpoints are useful inherited baselines; a claim about architecture itself additionally requires matching data, objectives and compute.

For a fast direct-predictor sweep, use four arms with seeds 11 and 29: a horizon-trained route-profile outcome baseline; finite-horizon poses/risk without past history but retaining the same current physical/controller-visible state; the same with measured history; and the history model without terrain. Fix samples, splits, optimizer budget and output-label definitions. First run 100-update cluster smoke jobs, then benchmark approximately 600 updates before selecting a bounded pilot budget. Eight independent variants can occupy one eight-GPU AMD node, with separate device assignments, logs and outputs. This is a proposed sweep; no new trainer or job was launched in this assessment.

Only after the scorer passes the fixed-bank test, compare CEM and MPPI **with the same scorer**, candidate representation and model-evaluation budget. Receding-horizon closed-loop Chrono evaluation is a separate gate. MPPI is a local sampling optimizer, not a guarantee of the globally optimal route. Keep global route/waypoint guidance for obstacle topology and long-distance goals. Do not average distinct left/right obstacle routes together; optimize within route families. Apply the same curvature, speed and footprint checks to the reconstructed weighted-mean route before model rescoring, and retain a validated sampled fallback. Bounded steering/speed does not establish that the joint terrain/state/control query is in training support.

Final confirmation needs newly reserved layouts: f111 and the f114/f115 state-ablation tests have already been examined. Do not repurpose active experiments' test banks as unseen confirmation for iterative MPPI design.

**Execution isolation and AMD plan.** The existing jobs observed during the audit were `409318` (`state_train`, running) and `409356` (`state_full_audit`, dependency pending). They were not changed. The AMD cluster was reachable, the three WP7/WP8 caches were present, and the scheduler reported two idle MI350 eight-GPU nodes at that observation; capacity can change. Use a new remote root such as `/work1/dannegrut/harry/experiments/traverse_mppi_20260908`, with frozen source snapshots and references to immutable data. Do not rsync this branch over `/work1/dannegrut/harry/nedm` or the active state-ablation directory.

The remote inventory also confirmed `full_v1`, `full_v2`, `full_v3` and `full_v4_partial` under `/work1/dannegrut/harry/nedm/artifacts/traverse`; the complete `full_v4` directory was not present at that path. Begin with the verified existing data or sync only the missing metadata/state payloads into the isolated experiment, after checking manifests and sizes. There is no need to transfer every compressed camera frame for a BMP/state pilot. Directory presence was checked; full remote payload completeness/checksums remain a pre-training verification step.

All training belongs on AMD. Use the established PyTorch 2.10.0 module, Python3.12 and `/work1/dannegrut/harry/venvs/nedm`, preserving the module's `PYTHONPATH`. New CPU collection can be sharded into disjoint per-worker outputs on AMD; the current Chrono CPU infrastructure makes Euler unnecessary for the initial reuse study. No new collection, training submission, local training, production model replacement or MPPI performance claim is part of this completed feasibility assessment.
