# nav_v1 videos — four planners, two missions

Every video is a real continuous Chrono rollout: the vehicle is never reset, and at each planning decision the
simulator renders one overhead depth frame **with the HMMWV in it**, the vehicle's own footprint is masked out of
every candidate corridor, 256 candidates are scored by the frozen depth ensemble, and the winner goes to the path
follower. The four planners differ **only in when the planner is allowed to think**:

| tag | planner |
|---|---|
| `1_waypoint_only` | one decision per waypoint (the previous behaviour, made sensor-driven) |
| `2_replan_2s` | a decision every 2 s of simulated time |
| `3_replan_1s` | a decision every 1 s of simulated time |
| `4_replan_1s_latency` | every 1 s, with the measured planning delay charged back to the simulation — the vehicle keeps following the old route until the new one is ready |

Same mission file, same frozen checkpoints, same controller, same node.

### Reading a single-planner video
* **left** — the height map back-projected from *that decision's own depth frame*. White is terrain the sensor did
  not see. This is the planner's entire input.
* **middle** — a zoom on the vehicle. The dashed white box is its footprint, the black box the footprint plus the
  1.5 m margin; everything inside is marked unobserved in every candidate corridor, so the planner never scores the
  vehicle's own body as terrain.
* **right** — grey: every route committed so far; red: the route being followed; blue: the track actually driven;
  black x: sliding backwards.
* **bottom** — forward speed, with a red tick at every planning decision. Negative speed is the vehicle sliding
  back down a slope.

### Reading a comparison video
All four planners on the same mission, synchronised on **simulated time**. A panel freezes and fades when that
planner's run has ended, and its title says why. The strip at the bottom overlays all four speed traces with a
time cursor.

---

## Set 1 — `f104_nav_002`: all four planners do the job
7 waypoints, 195 m of legs. Every planner reaches all 7 waypoints, within 5 s of each other, with no backward
sliding anywhere. This is what the loop looks like when nothing goes wrong, and it is the answer to "does the
continuous sensor-driven pipeline actually work in all four modes".

| file | outcome |
|---|---|
| `set1_all_work__1_waypoint_only.mp4` | 7/7, 76.2 s, 7 decisions |
| `set1_all_work__2_replan_2s.mp4` | 7/7, 73.8 s, 35 decisions |
| `set1_all_work__3_replan_1s.mp4` | 7/7, 74.8 s, 69 decisions |
| `set1_all_work__4_replan_1s_latency.mp4` | 7/7, 71.2 s, 50 decisions |
| `set1_all_work__compare_2x2.mp4` | all four side by side |

Watch the red route in the replanning panels: it is re-drawn from the vehicle's actual pose every 1-2 s, so the
grey fan of committed routes thickens continuously, while the waypoint-only panel commits seven routes in total.

## Set 2 — `f104_nav_001`: planning once per waypoint loses the mission, replanning saves it
7 waypoints, 212 m of legs. Same mission, same model, same controller — and the waypoint-only planner does not
finish it.

| file | outcome |
|---|---|
| `set2_contrast__1_waypoint_only.mp4` | **5/7 — STUCK.** On leg 6 the committed route cannot be climbed; the vehicle slides backwards for 10.7 s (speed trace goes negative and stays there) and the stall rule ends the run. It has no way to change its mind. |
| `set2_contrast__2_replan_2s.mp4` | **7/7 in 63 s.** Re-drawing the route every 2 s takes it round the same terrain without ever sliding. |
| `set2_contrast__3_replan_1s.mp4` | **7/7 in 81.5 s**, with 7.5 s of sliding on the way — it gets into the same trouble and drives out of it, which the waypoint-only planner cannot do. |
| `set2_contrast__4_replan_1s_latency.mp4` | **7/7 in 74.0 s** — see the note below; the campaign roll of this same arm on this same mission **left the arena at 2/7**. |
| `set2_contrast__compare_2x2.mp4` | all four side by side — the clearest single view |

### The one arm that is not reproducible
`4_replan_1s_latency` charges each decision's **measured wall-clock** planning time back to the simulation, and
that measurement depends on how loaded the machine is. Re-running this mission for the videos charged
1.35/1.45/1.80/1.90 s where the campaign run charged 1.45/1.55/1.70/1.80 s — a few hundredths of a second each
time, enough to activate the plans at different simulated frames, and the two rolls end differently: the campaign
roll drove off the arena at 2/7, this one finished all 7 waypoints. Every other arm reproduces bit-for-bit on the
same partition (the waypoint-only run above is identical to its campaign run to the last decimal). If that arm is
ever to be compared like for like, it needs a fixed latency budget (`--latency-s 1.5`) rather than the measured one.

> **Correction (2026-09-16).** The next paragraph is withdrawn. Every arena exit in the campaign, including the
> 2/7 campaign roll of `4_replan_1s_latency` on this mission, followed a rescue route that doubled back on itself —
> a runner bug, fixed since (see `../../REPORT.md`, correction at the top). The eight rollouts in these videos never
> used such a route (no route in them turns more than 5.1 deg between consecutive points), so the videos themselves
> are unaffected.

Across the whole campaign the two failure modes are: planning once per waypoint **stalls** (3 of 30 missions, never
leaves the terrain), and replanning **drives off the map** (4-5 of 30 missions per arm, never happens to the
plan-once arm). Set 2 shows the first of those; the second is visible in `main/runs/*__R2` on `g223_nav_002` and
`g204_nav_001`, and can be rendered the same way on request.
