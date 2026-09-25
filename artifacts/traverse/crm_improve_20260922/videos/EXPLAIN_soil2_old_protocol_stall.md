# Why the old protocol stalls in soil case f104_pair_group_0500 ("soil2")

Written 2026-09-24. Scope: the recorded cluster drive of the CNN-GRU planner that decides after a 3 s straight
approach (manifest arm `H3s`). It is compared with the five short-approach arms of the same group and with the
seven other planners that also decided from the same 3 s state. Read-only analysis. Every number below comes from
recorded files, except where a line says "reconstructed" or "local re-drive".

## Short answer

The vehicle does not stall because the ground under it is steep, because the soil swallowed it, or because the
controller stopped asking for speed. It stalls because **at the 3 s decision point it is already 1.5 m (front-left
wheel) or 3.6 m (chassis reference point) from the crater's west wall.** The chosen route turns 33 degrees right,
away from the crater, and starts 1.5 m/s slower than the vehicle is moving. The vehicle cannot make that turn in the
space it has left. It brakes to below 1 m/s, reaches full right lock, and still turns only 9 degrees. Its front-left
wheel runs onto the crater wall, so the four contact points end up out of plane by 0.25 m. The rear-right wheel
loses all load, and the front-left wheel carries only about 1.4 kN. The two wheels on the other diagonal carry
12 kN each. The differentials are open, so the drive torque escapes through the unloaded rear-right wheel: it spins
at 174 rad/s while the other three wheels stand still, at full throttle, for 29 s.

The "nearly flat, 1.8 degrees" description holds only at the chassis reference point. The front axle straddles a
13-degree cross-slope, and the front-left wheel sits 0.43 m lower than the front-right.

The limiting factor is traction, specifically load. Throttle is not the limit: throttle is 1.0 without a break from
5.20 s to 34 s, and the engine turns at 272 rad/s while delivering only 2.9 N m.

## Sources

- Recorded drive: `artifacts/traverse/generalist_20260921/A_adapt/a5/crm_pass2_runs/f104_pair_group_0500__H_B/`,
  containing `trajectory.npz`, `outcome.json` and `episode_complete.json`. The trajectory hash matches
  `episode_complete.json`.
- Soil extras: the run directory has no `crm_extra.npz`. The copy used here is
  `artifacts/traverse/crm_improve_20260922/scout/d1_cache/extra/f104_pair_group_0500__H_B/crm_extra.npz`. Its
  sha256 (78da40b0...44ff1) equals the `crm_extra.npz` hash recorded in the run's `episode_complete.json`, so it is
  the file from this drive.
  - Keys: `pos_z_m`, `bmp_ground_z_m`, `quat`, `spindle_z_m` (n,4), `slip_ratio` (n,4),
    `fsi_force_wheel_fx_n` (n,4), `wheel_order` (fl fr rl rr) and `tire_radius_m` (0.467 m).
  - `bmp_ground_z_m` is the ground height **at the chassis reference point only**
    (`scripts/crm_collect_ext.py:412`).
- State layout, from `trajectory.npz` `state_fields` (the `tire_normal_force_omega_pt` preset in
  `src/nedm/training/constants.py`):

  | Columns | Content |
  |---|---|
  | 0 | vx |
  | 1 | vy |
  | 2 | roll |
  | 3 | pitch |
  | 4 | roll rate |
  | 5 | pitch rate |
  | 6 | yaw rate |
  | 7-10 | tyre normal force fl, fr, rl, rr: the soil (FSI) force on each spindle, projected on world up |
  | 11-14 | spindle angular speed fl, fr, rl, rr (rad/s) |
  | 15 | engine speed (rad/s) |
  | 16 | engine output-shaft torque (N m) |

  Actions are [steer, throttle, brake]. Steer -1 means full right lock.
- Routes:
  - Branch route: `.../a5/picks_crm_H_named/routes/f104_pair_group_0500__H_B.json`. Its sha256 equals
    `branch_route_sha256` in `outcome.json`.
  - Approach route: `.../a5/approach/f104_pair_group_0500.json`, a straight line at 3.0 m/s, heading -1.3 degrees.
- Terrain: `TerrainMap.from_dir(assets/traverse/arena_f104_50h_v1)`.
- Per-wheel ground and sinkage: `ga_approach.wheel_sinkage`, which uses the collector's formula with spindle xy
  rebuilt from pose, quaternion and the HMMWV spindle offsets.
- "Twist" is defined here as ((ground under fl + ground under rr) - (ground under fr + ground under rl)) / 2 on the
  undisturbed height map: how far the four contact points are from a common plane.
- Comparison drives: the five short-approach arms listed in `manifest.json`. Also the other 3 s planners for the
  same group, in `.../a5/crm_pass2_runs/f104_pair_group_0500__{Hmask,P,Spcrm,Sprigid,T}_B` and
  `crm_improve_20260922/e2/runs/f104_pair_group_0500__{Hcont,Hbr}_B`.
- Local re-drive of this arm (`videos/chase_work/soil2_H3s/run`, within 11 cm of the recorded path, same outcome).
  It is used only for its logged commanded speed.

## Timeline of the recorded drive (50 ms frames, t = frame x 0.05 s)

The four numbers in the load and spin columns are for the fl, fr, rl and rr wheels.

| t (s) | vx (m/s) | steer / throttle / brake | wheel load (kN) | wheel spin (rad/s) | twist (m) | what happens |
|---|---|---|---|---|---|---|
| 2.0-3.0 (mean) | 2.94 | 0 / 0.19 / 0 | 7.2 6.1 5.8 7.3 | 8.0 8.0 8.0 8.2 | -0.00 | straight approach at cruise, all wheels loaded |
| 3.00 | 2.85 | 0.16 / 0 / 0 | 4.7 10.5 3.4 1.3 | – | -0.03 | **decision**: route v0 1.37 m/s, route heading -34.6 deg vs vehicle -1.4 deg |
| 3.05 | 2.81 | 0.06 / 0 / **0.87** | | | | brake burst (5 frames with brake > 0.3, 3.05-3.25 s), throttle 0 until 3.45 s |
| 3.65 | 1.00 | **-1.00** / 0.31 / 0 | 2.1 8.9 9.2 0.6 | 4.8 2.5 2.2 3.5 | -0.10 | full right lock reached (steering rate limit 2/s); stays at -1.00 until 34 s |
| 4.0-5.0 (mean) | 0.62 | -1 / 0.66 / 0 | 2.0 10.4 12.2 0.6 | 9.5 1.6 1.8 24.0 | -0.20 | front-left wheel onto the crater wall; lightly loaded wheels begin to spin |
| 5.00 | 0.29 | -1 / 0.92 / 0 | 1.5 10.4 12.6 0.2 | 16.6 0.5 1.0 44.6 | -0.24 | **stall onset**: 20 frames in a row with vx under 0.3 m/s and throttle over 0.3 start here |
| 5.0-7.25 (mean) | 0.03 | -1 / 0.99 / 0 | 0.5 12.0 13.1 -0.2 | 24.5 0.0 0.1 51.2 | -0.25 | fl and rr spin with no forward motion; fl digs in, roll -5 to -8 deg |
| 7.25-12 (mean) | 0.00 | -1 / 1.00 / 0 | 1.5 11.4 12.3 0.0 | 1.6 0.0 0.0 135.7 | -0.25 | rr loses contact (load exactly 0 in 89 of 95 frames); fl stops; all drive torque escapes through rr |
| 12-34 (mean) | 0.00 | -1 / 1.00 / 0 | 1.4 11.8 12.0 0.0 | 0.0 0.0 0.0 174.2 | -0.25 | engine 272 rad/s at 2.9 N m, 0.8 kW; rr load exactly 0 in 429 of 440 frames (at most 0.47 kN) |
| 34.00 | | | | | | no-progress rule fires at the earliest time allowed (24 s minimum + 2 s confirmation + 8 s tail) |

From 5.0 s to 34 s the vehicle never moves more than 7.8 cm from its 5.0 s position. The farthest distance between
any two positions in that window is also 7.8 cm. From 5.3 s on it stays within 4.3 cm of its 5.3 s position.

For scale: with the vehicle rolling at 2-3 s the total wheel load is 26.4 kN, about 6.6 kN per wheel.

## Findings

1. **The decision is made too close to the crater wall** (`trajectory.npz` frame 60, t = 3.00 s; TerrainMap).
   - The vehicle is at (-5.94, -25.20), heading -1.4 degrees, moving at 2.85 m/s.
   - A crater lies ahead and to the left: centre (2.44, -23.25), sigma 3.3 m, 1.75 m deep.
   - Straight ahead, the first ground cell steeper than 15 degrees is 3.6 m away for the chassis reference point,
     1.5 m for the front-left wheel and 2.2 m for the front-right wheel. Slopes use the anatomy study's definition:
     central difference over +-2 cells.
   - The arms that decide at 0.5 s or 1 s stand at x = -11.75 or -11.19. For them the same distances are 9.3 or
     8.8 m (chassis) and 7.3 or 6.8 m (front-left wheel).
   - Fix: this is what the new protocol already fixes by deciding on the start pad. For any planner that decides
     while moving, check that the route's first metres can be reached with the vehicle's turning radius before the
     nearest steep cell.

2. **The route asks for a turn the vehicle cannot make in that space, and for a lower speed. It does not ask for
   zero speed, a turn into a slope, or a reverse** (branch route JSON; `outcome.json` `branch`).
   - Heading: the route's first point is the vehicle position, but its first heading is -34.6 degrees, a
     33.2-degree step to the right at station 0.
   - Speed: the speed profile starts at 1.37 m/s against the vehicle's 2.85 m/s, a step of -1.48 m/s. It then rises
     to 1.50, 1.85, 2.35 and 3.39 m/s at 1, 2, 3 and 5 m.
   - Commanded speed: reconstructed from the recorded poses it is 1.37 → 1.54 → 1.74 m/s. In the local re-drive's
     command log it is 1.37 → 1.74 → 2.00 m/s. It is never near zero.
   - Terrain along the route: tracked as drawn, the route crosses gentle ground. Over the first 12 m the grade is at
     most +6.8 degrees, the cross-slope at most 7.5 degrees, the slope at most 6.5 degrees, and the twist of a
     vehicle placed on it at most 0.10 m. The short-approach routes give 0.09-0.12 m. The route bends away from the
     crater and around its south side.
   - Tracking: the vehicle never gets onto the route. At 5.0 s it is 1.0 m left of it (at station 1.76), heading
     23 degrees left of the route heading. Straight ahead of the stalled vehicle, the slope is 8 degrees at 1 m,
     18.8 degrees at 2 m, and 25-27 degrees from 3 to 8 m, down into the crater.
   - Fix: make routes built from a moving state start at the vehicle's heading with a curvature the vehicle can
     follow. `gc_control.sample_continuations` already has a 15-degree start-heading cap and a start-speed floor.
     The route family used by this planner (CEM 4x64 in `ga_planner.py`) evidently had neither: this pick starts
     33 degrees off the vehicle's heading and 1.5 m/s below its speed.

3. **The turn physically cannot clear the wall from this point** (recorded poses and actions).
   - Measured turning radius at full lock: 7.4 m, both in this drive (3.6-6.45 s, mean 0.37 m/s) and in the 1 s
     CNN-GRU drive (1.5-2.45 s, mean 2.64 m/s).
   - Best case: the tightest possible turn, starting instantly from the 3 s pose (a 6.8-8 m arc). The front-left
     wheel reaches a cell steeper than 15 degrees after 1.7 m, with the heading changed by only 12-14 degrees. The
     static twist reaches 0.22 m after 2 m and 0.30-0.33 m after 3 m.
   - What actually happened: the steering needed 0.6 s to move from +0.16 to -1.0 because of the rate limit. The
     vehicle covered 2.1 m by 5.0 s and turned 9.2 degrees.
   - Fix: the same as finding 1. Once the vehicle stands at the 3 s point on this line, no right turn keeps the
     front-left wheel off the wall.

4. **Stall mechanism: one wheel unloaded plus open differentials** (`trajectory.npz` state columns 7-16).
   - At the stall the ground under the wheels (fl, fr, rl, rr) is at -0.33, +0.10, -0.05 and -0.12 m. The front
     axle is on a 13.2-degree cross-slope (left side low), the rear axle on 2.4 degrees the other way. Twist:
     -0.25 m.
   - The terrain slope at the chassis reference point is 1.8 degrees, which is where the "flat ground" reading
     comes from.
   - Twist grows from -0.03 m (3.0 s) to -0.15 m (4.0 s) to -0.24 m (5.0 s) and stays at -0.25 m. The successful
     short-approach drives never exceed 0.13 m; in them some wheel is below 200 N in only 4-7 % of frames, against
     86 % here.
   - Loads: the fr-rl diagonal carries 11.8 + 12.0 kN (about 94 % of the 25.2 kN total), fl carries 1.4 kN, and rr
     carries 0. The vehicle set-up is in `src/nedm/traverse/scene.py` `build_config`: AWD drive, SHAFTS engine,
     AUTOMATIC_SHAFTS transmission. No differential lock is set anywhere in `create_hmmwv`
     (`src/nedm/hmmwv_data.py`).
   - The data shows the open-differential signature: rr spins at 174.2 rad/s (tyre surface 81 m/s; slip ratio p95
     814) while fl, fr and rl stay at or below 0.36 rad/s. At throttle 1.0 the engine turns at 272 rad/s but
     delivers only 2.9 N m (0.8 kW), because all it drives is one unloaded wheel. The two wheels that carry the
     weight get no torque: an open differential passes each output only as much torque as the output that resists
     least, here the airborne rr wheel.
   - Before rr lost contact completely (5.0-7.25 s), fl spun at 16-31 rad/s and dug in. Its reconstructed sinkage
     goes from -0.09 m at 5.0 s to +0.03 m at 7.25 s and +0.06 m at the end. Roll went from -5 to -8.5 degrees,
     which lifted rr clear: from 7.25 s its load is exactly 0 in 97 % of frames, with brief touches of at most
     1.1 kN.
   - Fix beyond the planner: a locking or limited-slip differential would change this outcome. That was not
     simulated and it changes the vehicle contract. Alternatively, add a footprint-twist term to the risk labels or
     the planner's objective.

5. **Braking at the handover contributes but does not decide the outcome in this group** (`outcome.json` of all
   eight 3 s planners).
   - The brake burst takes vx from 2.85 to 1.03 m/s within 0.6 s and to 0.76 m/s by 4.0 s, so the vehicle reaches
     the wall with no momentum.
   - All eight planners that decided from the same 3 s state fail. Seven of them end with one wheel at zero load
     spinning at 168-174 rad/s at throttle 1.0. The eighth (the rigid-trained specialist, which turned left) bogs
     down in the crater at 11.15 s with two unloaded wheels spinning at 36-44 rad/s.
     - Planners whose routes start slowly (v0 1.16-1.37 m/s) stall at the same spot, (-3.8, -25.45), at
       5.0-5.2 s.
     - Two planners' routes start at the vehicle's speed (v0 2.85-2.86 m/s, heading step -33 and -35 degrees):
       - The speed-continuous route family (1 brake frame) stalls at 5.0 s, 0.7 m further along the same rim, with
         twist 0.35 m.
       - The soil specialist (no braking) gets 2.9 m further, onto the crater wall, and stalls at 5.95 s with roll
         -24 degrees.
     - Planners whose routes start faster (v0 3.4-3.6 m/s) cross the rim and stall inside the crater at 7-8 s,
       with roll about -19 degrees and pitch -15 to -16 degrees.
   - The speed step therefore sets where the stall happens, not whether it happens. This matches the population
     handover test in the report: matching the vehicle's speed gave 81.8 % goal reached against 83.9 %.
   - Fix: none needed beyond findings 1-2. A start-speed floor on its own would not have rescued this group.

6. **The soil is not what stops it** (`outcome.json` `crm`; reconstructed per-wheel sinkage).
   - The deepest wheel is 0.175 m below the surface (collector value), against a 0.30 m limit for being bogged.
     The loaded fr and rl wheels end at 0.18 and 0.17 m.
   - In this model the chassis does not touch the soil, so the vehicle cannot be hung up on its belly.
   - A naive reading of `crm_extra` gives a misleading number. Tyre radius minus (spindle height minus
     `bmp_ground_z_m`) puts the fl wheel 0.37 m deep, past the limit. That is because the reference ground height is
     taken under the chassis, while fl stands over the crater slope. At the wheel's own position the value is
     0.06 m.
   - Fix: when reading `crm_extra`, use per-wheel ground heights (`ga_approach.wheel_sinkage`), not
     `bmp_ground_z_m`.

7. **How the other arms stand at 5 s** (manifest run dirs; `compare_soil2.mp4` at 5.0 s shows the same).

   | arm | decides at | decision x | route start speed vs vehicle speed | heading step | at 5.0 s | outcome |
   |---|---|---|---|---|---|---|
   | CNN-GRU, 3 s (old) | 3.0 s | -5.94 | 1.37 vs 2.85 | -33 deg | (-3.83, -25.44), 0.29 m/s, full lock | stuck, ended 34.0 s |
   | CNN-GRU, 1 s | 1.0 s | -11.19 | 2.85 vs 1.45 | -37 deg | (-1.08, -29.22), 3.37 m/s | goal 9.55 s |
   | Transformer, 1 s | 1.0 s | -11.19 | 3.03 vs 1.45 | -26 deg | (1.57, -30.29), 5.14 m/s | goal 8.05 s |
   | CNN-GRU, 0.5 s | 0.5 s | -11.75 | 2.28 vs 0.82 | -32 deg | (-1.38, -29.14), 3.47 m/s | goal 9.45 s |
   | Transformer, 0.5 s | 0.5 s | -11.75 | 5.25 vs 0.82 | -29 deg | (-0.64, -29.47), 2.71 m/s | goal 9.95 s |
   | CNN-GRU + gradient, 0.5 s | 0.5 s | -11.75 | 4.01 vs 0.82 | -7 deg | (4.52, -29.41), 5.06 m/s | goal 7.45 s |

   - The short-approach arms make the same kind of right turn, but on the start pad. By 3 s they already head -23
     to -36 degrees at y = -26.8 to -27.5.
   - They pass 2.1-2.7 m south of the old arm's stall point at 3.0-4.7 m/s.
   - Their routes start at or above the vehicle's speed, so they never brake (maximum brake after the decision is
     at most 0.13).

## What is established

- **Established:** the stall is traction-limited, not throttle-limited. Throttle is 1.0 from 5.20 s to 34 s, the
  engine delivers only 2.9 N m, the three wheels that touch the soil stand still, and the one airborne wheel spins
  at 174 rad/s.
- **Established:** the traction loss comes from wheel unloading on a twisted footprint (rr load 0, fl 1.4 kN)
  combined with open differentials. It does not come from sinkage beyond the soil depth.
- **Established:** the twist comes from the front-left wheel reaching the crater's west wall. That happens because
  the vehicle decides 1.5 m (front-left wheel) from the wall and cannot make the route's 33-degree immediate right
  turn. The measured 7.4 m full-lock radius and the steering-rate limit make it geometrically impossible.
- **Established:** commanded speed never drops toward zero (1.37-2.0 m/s). The route does not turn into a slope, and
  it does not ask for a reverse.
- **Established for this group, within the eight planners tried:** deciding at the 3 s point commits the vehicle to
  crossing the crater rim. All eight 3 s planners fail, with the same wheel-in-the-air signature, and all five
  short-approach arms succeed.

## What is not established

- **Not established:** that no route at all could succeed from the 3 s state. Only eight planner picks were driven,
  none of them a stop-and-reverse, which the follower cannot do anyway.
- **Not established:** that a locking or limited-slip differential would have freed the vehicle. It is not simulated.
  The numbers suggest it strongly: 24 kN rests on two wheels that receive no torque.
- **Not established:** the exact share of the handover brake burst. The matched-speed counterpart is a different
  route pick, so this is not a controlled single-variable test.
- **Not established:** why the risk model rated this route at 5.1 % failure (P = 0.051 in the pick file). The
  model's internals were not examined.
- **Approximations:**
  - Twist and slopes use the undisturbed height map (0.156 m cells), not the deformed soil surface.
  - Per-wheel sinkage is a reconstruction that ignores suspension travel. On the start pad it already shows a
    spread of about 0.1 m between wheels, so trust changes over time more than absolute depths.
  - The soil model's 0.08 m particle size is the usual resolution caveat.
- **Not comparable:** the rigid-ground drive of this group. It decided at a different point (x = -4.90), took a
  route that turned left through the crater, and reached the goal in 10.45 s.
