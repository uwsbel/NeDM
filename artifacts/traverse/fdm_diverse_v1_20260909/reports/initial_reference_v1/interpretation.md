# Initial-reference and closed-loop validation diagnosis

All 90 references on all six validation scenes were retained. This is a diagnostic of a fixed reference library, not an MPPI result. The primary model remains RGBD seed 11 best step 4000; the last-step result is supplemental.

## Three measured limits

1. **Forecast horizon:** 78/90 routes are safe through 12 s, but only 21/90 finish safely. Even substituting measured first-12-second consequences into the frozen scoring formula selects a full-route safe goal on only 3/6 scenes. Its valley, rough and mixed winners first contact at 12.60, 12.65 and 15.60 s. Accurate short forecasts therefore cannot guarantee full-route feasibility.
2. **Risk calibration:** the primary model misses all 12 first-horizon contact positives at the declared 0.35 rejection threshold, despite contact AUROC 0.840. Every initial candidate passes its risk gates. Its fixed-reference winners finish safely on 3/6 scenes for time and 2/6 for energy, versus 0/6 for matched blank. The last step reaches 5/6 and 4/6, but this diagnostic does not change checkpoint selection.
3. **Replanning and stopping:** the completed v7 ridge trial starts on a small refinement of physically safe original route 14 (+44 m at 6 m/s). At 1 s it switches to a fresh -44 m family while both the retained route and original route 14 remain admissible. Their costs are 63.344/63.383 s versus 60.280 s for the new family. Subsequent refinements accumulate deviation: original-route distance first exceeds 2 m at 3.85 s and the original reference disappears at 4 s. This supports a bounded route-commitment ablation; it does not prove that commitment alone will recover success.

## The ridge bounded event includes planner-induced stopping

The ridge trial spends 95.0/180 s paused by the planner and times out. All **301** measured bounded-motion windows overlap a planner pause or zero desired-speed request: 297 fully and four partly; none occur entirely under a positive speed request. The first window, 24.65–26.65 s, is fully paused. These remain positive under the declared motion/effort schema, which uses applied throttle and measured displacement, but they are **not evidence of spontaneous hill stall**. Primary success and event labels were left unchanged; command-intent overlap is supplemental.

The rough trial spends 80.998/180 s paused and also times out, with no contact, bounded-motion window or sustained near-stop. Its initial fixed winner was already physically unsafe, so this case does not represent loss of a successful fixed reference. It changes family repeatedly while avoiding contact but does not finish.

The source/controller settings match collection: native Chrono path follower, identical gains and terrain-height lookup. Pack-versus-deployment features match exactly apart from documented float16 image caching: primary maximum resulting pose difference is 0.041 m; GPU-versus-CPU differences are about 0.000012 m. These small numeric differences do not explain the observed route and risk failures.

Evidence: `diagnostic.json`, `event_interpretation.json`, `v7_switch_audit.json`, and `v7_pause_window_audit.json`; read-only scripts are retained beside the trace audits. No test outcomes were opened.
