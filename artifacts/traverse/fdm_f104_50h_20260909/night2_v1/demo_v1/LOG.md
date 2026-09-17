# Demo scenarios (2026-09-14) — five start/goals, three routes each, filmed in Chrono

Purpose: show, on real Chrono runs, whether the deployed risk model's ordering of candidate routes matches
what the simulator does. Not a new experiment — a read-out of the night-2 planner on already-frozen test groups.

## Design (fixed before any run)
For each start/goal, take the cached 256-candidate night-2 proposal pool and score it with the deployed
5-seed ensemble `night2_v1/final/N2_s*.pt`. Drive three of them:
  optimal    = argmin risk (what the planner drives)
  suboptimal = nearest candidate to 10% predicted risk, required to differ from the optimal by >3 m somewhere
  risky      = argmax risk (what the planner refuses)
Same model, same pool, three ranks. Nothing else differs between the three videos of a scenario.

## Steps
1. `scripts/f104_demo_pick.py` over the 28 hazard groups where the night-1 planner was unsafe and the
   night-2 planner was clean (from `closed_haz/results.json`) — terrain known to be able to trap the truck.
   84 routes staged.
2. Cluster screening, job 419374, `demo_night2_v1`, 28 array tasks on mi2101x, 84 episodes, frozen collector,
   all three routes of a group in one task (Chrono is node-deterministic only). 26/28 groups returned in time.
3. `scripts/f104_demo_select.py`: keep groups where the trio tells three different stories (optimal clean,
   suboptimal reaches the goal but is slower or leans harder, risky stalls or slides), then thin to five
   scenarios at least 12 m apart in start position.
4. Cluster re-run with Chrono's Blender exporter chained onto the collector's own observer, job 419411,
   `demo_blend_v1`, 15 array tasks. **All 15 reproduced the screening run exactly** (status and elapsed to 0.05 s).
5. Local Blender render (`scripts/f104_demo_render_blender.py`, EEVEE, 1280x720, 20 fps, planned route drawn
   as a ground ribbon, goal disc at 2.5 m) and ffmpeg captions (`scripts/f104_demo_videos.py`).
   Clips of stalled runs cut at 35 s, stated on screen.

## Result
Every scenario came out as predicted: the model's pick reached the goal, the middling candidate reached it
slower or with more body tilt, and the model's worst candidate stalled or ran out of time. Predicted risks
were 0.01-0.12% / ~10% / 100% respectively.

## Notes and limits
- `rank` inside `picks.json` is the count of candidates scoring lower. For the worst-ranked route the rank is
  not meaningful because many candidates tie at 100%; the captions say "worst of 256" instead.
- These are held-out start/goal pairs on a single arena; every stretch of terrain in them was driven during
  training by some other episode.
- The model ranks well but is badly calibrated (median predicted 0.008% vs realised 0.33% over the full
  300-group hazard test). The videos show ordering, not probability.
