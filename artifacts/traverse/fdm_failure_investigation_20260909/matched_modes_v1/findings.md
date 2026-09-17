# Matched planning-mode diagnosis

This is a post-hoc diagnostic on six validation scenes and six previously opened test scenes. It does not update protected scores, choose a configuration, or justify tuning on these test outcomes. The original freeze and reports are preserved.

All 24 new receding trials and 24 existing plan-once controls passed the raw-telemetry and checksum audit. They use the same final-update-5000 RGB-D checkpoint, source, runtime, input observation, costs, seed and controller. All 24 launch decisions, including candidate scores and selected route, match exactly; state, action and pose match through 1 s, and work matches over the first second. Only the planning mode changes.

| Scenes | Energy weight | Plan once safe | Receding safe | Lost safe | Gained safe |
|---|---:|---:|---:|---:|---:|
| Validation | 0 | 5/6 | 4/6 | 1 | 0 |
| Validation | 0.02 | 4/6 | 4/6 | 0 | 0 |
| Opened test, diagnostic | 0 | 4/6 | 4/6 | 2 | 2 |
| Opened test, diagnostic | 0.02 | 4/6 | 5/6 | 0 | 1 |

The data do not support a uniform same-checkpoint success loss from replanning. They show scene-specific gains and losses. At zero energy weight, validation ridge loses safety; on opened test, mixed obstacles and valley lose safety, while rolling hills and rough ground gain it. At weight 0.02, opened-test rough ground gains safety; rolling hills still has contact despite eventually reaching the goal.

The validation ridge/time loss has contact first detected at 6.95 s, before the first pause at 10 s. Its first bounded-motion window is 14.05–16.05 s and contains zero commanded pause time. Selected references change sides of the fixed launch-to-goal midplane at 2, 3 and 4 s; there are 86 geometry updates after launch. These observations differ from the previous validation-best-checkpoint stop-only ridge diagnosis and should not be merged with it. They show a replanning-associated physical failure, without proving which individual update caused it.

The opened-test mixed/time loss has contact first detected at 118.55 s and 24 s of commanded pause over the run. Valley/time times out without the declared contact, rollover or bounded-motion event, with 28 s paused. The opened-test rolling/energy trial arrives after contact first detected at 30.90 s; it remains an unsuccessful safe-traversal trial.

Pause and failure are not interchangeable. For example, validation rough/time has 148 s of commanded pause and its first bounded-motion window contains 1.45 s paused; the ridge/time bounded window contains none. The full report records both window start/end and paused duration. Contact timestamps identify the first positive 50 ms interval endpoint, not an exact 2 ms substep onset. Reference-side changes use geometric crossings of a fixed midplane; dynamic candidate-family indices are not treated as stable route identities.

Among paired safe completions, receding changes mean per-scene time/work relative to plan once as follows. These conditional comparisons exclude failed pairs; they do not override unconditional safe-completion counts.

| Scenes | Weight | Safe pairs | Time change | Positive mechanical work change |
|---|---:|---:|---:|---:|
| Validation | 0 | 4 | +3.13% | +85.05% |
| Validation | 0.02 | 4 | +7.21% | +119.64% |
| Opened test, diagnostic | 0 | 2 | −1.21% | +3.12% |
| Opened test, diagnostic | 0.02 | 4 | −0.45% | +2.32% |

Work remains positive engine-interface mechanical work, not fuel consumption. The large validation work increases deserve a separate controller/command investigation; this mode comparison does not isolate their physical cause.

- [All outcomes and comparison table](report_v1/report.md)
- [Complete audit, onset, pause and decision evidence](report_v1/report.json)
- [All 48 new/control rows](report_v1/all_trials.csv)
- [Prelaunch diagnostic protocol](protocol.json)
- [Exact task declaration](tasks.json)
- [Analysis code binding](analysis_binding_v1.json)

AMD physics job 412215 completed with exit 0 in 6 min 14 s, 2026-09-10 01:42:15–01:48:29 UTC. Audit job 412220 completed with exit 0 in 1 min 05 s, 01:48:55–01:50:00 UTC. Physics used 24 workers with 128 requested CPUs. No rendering, optimizer updates or source/model/cost tuning occurred. Large raw outputs remain at `/work1/dannegrut/harry/experiments/fdm_failure_investigation_20260909/matched_modes_v1/raw`.
