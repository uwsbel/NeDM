# sensor_v2 log — corrected depth->world pipeline and matched height/depth training

Started from the independent review artifacts/reviews/sensor_input_20260915/README.md (two findings: v1 sampling
assumes flat ground; the v1 depth arm dropped the route-start range). Existing checkpoints, datasets and results are
untouched; everything here is a separate v2 path.

2026-09-15
- scripts/sensor_map_v2.py: depth -> 3D points (camera intrinsics) -> metric world grid (0.15625 m, +-40 m) with
  z, absolute range, ray secant, colour and pixel-count-per-cell. No authored-height fallback. Grids for all six
  captured arenas: sensor_v2/grids/<arena>/.
- scripts/sensor_dataset_v2.py: 12-channel corridors sampled from that grid; label path verified identical to the
  existing labeller on 40 routes. Datasets on the cluster for f104 (36,199 routes) and the five sibling arenas.
- scripts/sensor_train_v2.py: matched training, identical rows/seeds/budget, split by whole arena
  (train f104+g228+g203+g217, held out g216+g231), primary metric = route choice at matched speed.
- Diagnostics (3 agents + 2 verifiers, saved data only, sensor_v2/diag/):
  * v2 height error vs the simulator reference 0.0071-0.0087 m MAE (v1 flat sampling 0.027-0.044 m), 1.2-1.4x the
    8-bit heightmap's own quantisation floor; orientation/intrinsics/registration/aggregation all pass adversarial tests.
  * The review's algebra holds, but the practical information loss from the dropped route-start range is ~4 mm RMS
    (the route-start secant is in the same tensor). The raw-depth arm's real handicaps are scale/conditioning (the
    perspective term is 1.9-2.5x the height signal), locality, and cross-arena shift (absolute range + ray direction
    is the worst representation measured for transfer: two-sample AUC 0.893 vs 0.563 for height).
  * Replay of the sensor_v1 test-2 pools reproduces the shipped test bit-for-bit; under v2 corridors the driven route
    changes in 19.5-25.2% of groups (35.3-42.8% at fixed 2 m/s), flips track the flat-ground displacement rather than
    slope, and on routes with known outcomes v2 orders unsafe-vs-safe pairs better for the height models
    (fixed 2 m/s 0.665 -> 0.738, 10 vs 29 discordant, p = 0.0034) but not for the depth arm.
  * Defects fixed in sensor_map_v2.py: arena-size and depth-convention guards, honest coverage percentiles, grid
    index-order metadata. Docstring corrected: v2's case is placement/locality/conditioning, NOT restored information.
- Matched offline result (3-seed ensembles, held-out arenas, 1,200 same-speed choices):
  H 14.50% unsafe picks, H0 14.50% (+0.00 [-0.75, +0.75]), Dabs 14.92% (+0.42 [-0.42, +1.25]),
  Drel 15.58% (+1.08 [+0.25, +2.00], 20 vs 7 cells). Random pick 29.72%, unavoidable 12.42%.
- Chrono pilot on 200 fresh held-out-arena start/goals (1,066 drives): arms H, Dabs, Drel (speed free and fixed
  2 m/s) + the 6 m/s straight line.
- Verifiers (2, adversarial) corrected three things and found two real bugs:
  * TerrainMap (the repo's privileged height oracle) is NOT the frame Chrono simulates: Chrono's RigidTerrain puts the
    512 BMP samples at the patch edges (80/511) while TerrainMap puts them at cell centres (80/512), i.e.
    x_terrainmap = (511/512) x_chrono, up to 0.078 m horizontally at the arena edge. Against the terrain Chrono
    actually simulates, v2 recovers height to 0.00075-0.00087 m MAE per pixel (9-12x better than v1), so my
    "at the 8-bit quantisation floor" statement was wrong in the conservative direction. This affects every previous
    comparison in the project that used TerrainMap as ground truth.
  * The replay's label-carrying pair test treated within-group pairs as independent (169 pairs from 75 groups; 260
    from 149): the quoted p = 0.0034 is inflated, and an optimiser's-curse placebo was missing. Under a placebo the
    depth arm also gains (+0.024, p = 0.045), so "v2 does not help raw depth" is too strong.
  * v1 -> v2 also changes grid resolution (0.1868 -> 0.15625 m), which the replay's own control shows accounts for
    ~10-15% of the movement and penalises the depth arm slightly more.
  * Bugs fixed in scripts/sensor_dataset_v2.py: NaN could leak through the bilinear sample of an empty cell; the
    route-start range reference was read after masking (an invalid start silently became the camera height). Neither
    ever fired on these captures (coverage is 100%), so no built dataset changes.
- Chrono pilot, 200 fresh held-out-arena start/goals, 1,066 drives, 0 failures:
  fixed 2 m/s unsafe H 6.0%, Dabs 5.0% (-1.0 [-4.5, +2.5], 5 vs 7, p = 0.77), Drel 7.5% (+1.5 [-1.5, +5.0], p = 0.55);
  speed free unsafe 0.5 / 0.0 / 0.5%, failures 0 everywhere, median time 12 s for all three, tilt>30 3.0 / 5.5 / 7.0%.
  The 6 m/s straight line: 6.0% unsafe, 21% tilt>30, 8 s. Underpowered for 1-1.5 point differences.
2026-09-16T00:46:02Z REPORT.md written
2026-09-16T01:51:00Z vehicle-included pilot complete: VEHICLE_PILOT.md
