# Nearby-assets twelve-second diagnostic

Three validation-arena starts were chosen using geometry alone and frozen in scene_design/scene_spec_v2.json. Existing arenas, model, costs, and controller remain unchanged. Every start uses a fresh global RGB-D observation and the same settled native HMMWV state for its four references. There are no optimizer updates.

The model predicts 60 endpoints at 0.2 s. All candidates and selections are saved before physical execution. Select two lowest finite-cost references and two highest normalized-risk references, with 2 m maximum separation within each pair. Risk is relative if no candidate crosses the frozen rejection gate. If the model cannot supply two accepted geometries, retain and explicitly label rejected lowest-cost fallbacks. No physical result changes the selection.

Execute each chosen reference with native Chrono PID for 12 s, without replanning, so its physical future corresponds to the single saved forecast. Existing early goal/rollover termination is retained and reported as censoring if encountered; no fabricated continuation. The goal is 84 m ahead to avoid ordinary early arrival.

Report all twelve outcomes, horizon-aligned risk labels, predicted/measured progress, pose error, work and roll/pitch. These selected validation diagnostics do not establish a population success rate. Render the primary scene's four executions with the existing passive observer and verify timestamps and same-start equality. Main checkout remains read-only; physics runs on AMD.
