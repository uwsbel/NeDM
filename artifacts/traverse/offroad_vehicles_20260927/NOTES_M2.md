# NOTES M2: the M113 tracked vehicle in the soil collector (2026-09-28 00:00-00:30 CDT)

**Verdict: GO.** Admit `m113` and `m113_g4` to the smoke, with one declared model change (the brake, item 1).
All four gates pass.

## What I built (new files only)
- `scripts/ov_m113.py`: the vehicle factory plus the run-time swaps.
  - Interface: `install(ctx)`, exactly as M1's dispatcher `ov_crm_collect.py` calls it. It returns None for any other
    vehicle name (nothing is patched), else a dict with `scene_hook`, `frame_hook`, `argv`.
  - Contents: 127 shoes coupled to the soil as one flat pad box each; the per-quarter mapping of the 17-column state
    and the tyre fields; the breakthrough-rule equivalent (lowest pad bottom per quarter, radius 0); hull belly points
    (972); the `vehicle` block.
  - It refuses a soil step other than 0.5 ms and the `crm_collect_ext` base.
  - Two diagnostic flags are removed from the argv before the collector sees them: `--m113-pad`, `--m113-brake`.
  - Final sha256 `a61f29ad…`. The cluster checks ran on `94177c4e…`; the only difference is one provenance field
    (declared gear ratios instead of a null).
- `configs/crm_m113.json`: a byte copy of the production `crm_f104_v1/configs/crm_main.json` (`90cd049e…`, the file
  the K3 soil jobs used) with only `step_s` changed from 0.001 to 0.0005 (sha `0bba40c0…`). The inherited `name`
  field still says "step1ms".
- `assets/traverse/vehicles/ov_m113/`:
  - `M113_AutomaticTransmissionShafts_g4.json`: the stock C++ transmission written as JSON, every gear ratio / 4;
  - an `x1_check` copy with the stock ratios, for checks only;
  - a README.
- `scripts/ov_m113_check.py`, with the sub-commands `build`, `rigid`, `tilt` and `qa`.
- Cluster job scripts and results: `scratch/M2/` (`m2_job_{a,b}.sbatch`, `cluster/`, `local/`). Cluster outputs are
  in `G4/checks/M2/out/`.

## Checks and results
- **Rigid, CPU.**
  - My construction and the `veh.M113` wrapper give identical poses (max difference 0.0 m).
  - The JSON transmission with the stock ratios gives identical poses to the C++ transmission (0.0 m).
- **Main finding: the backward slide found by the scouts comes from the brake, not from the pads.**
  - Chrono's M113 simple brake is a friction torque that flips sign every step and never locks.
  - With it, the braked M113 rolls back even on RIGID ground: 0.89 m/s at 10° and 1.0 m/s at 15°, with the sprocket
    turning 6.7 rad.
  - `SetBrakeType(SHAFTS)` is silently ignored for single-pin shoes (the source always builds the simple brake).
  - Fix: Chrono's own shafts-brake construction (a clutch per sprocket, 10,000 N·m), added from Python. The stock
    brakes then get zero input.
- **Gate (a), build on a compute node: PASS.** MI350X node k007-005-v3, `chrono-build-fsi`. Both arms build at
  11,343 kg with 63 + 64 shoes, and every binding needed is present (also checked on the login node).
- **Gate (b), braked hold on tilted soil (production soil settings, 0.5 ms step, 14 x 6 m box): PASS with the brake
  fix.** "Holds" = mean speed over the last 1 s of a 2 s hold is 0.10 m/s or less.

  | set-up | 10° | 15° |
  |---|---|---|
  | stock brake, flat pads | slides at -0.62 m/s: **fails** | slides at -0.67 m/s: **fails** |
  | shafts brake, flat pads | -0.009 m/s: holds | -0.028 m/s: holds |
  | shafts brake, grouser pads | -0.029 m/s: holds | -0.035 m/s: holds |
  | Gator (reference) | -0.018 m/s | -0.036 m/s |

  - Grousers are not needed. Thick pads were not re-tried (the scouts had found no gain).
  - Climbing 10° from rest at full throttle for 3 s: `m113` reaches 1.6 m/s, `m113_g4` 2.5 m/s, the Gator 2.7 m/s.
- **Gate (c), f104 episodes through the new dispatcher: PASS.** All 4 cluster episodes and 2 short local ones pass
  crm_qa, with a finite (n, 17) state, a passed launch check and the vehicle block.
  - `m113` stalled on both routes it drove: g0005 route_01 and g0000 route_02. It was stopped by the blockage rule at
    34 s, on about 19-20° of pitch, with the engine pinned at 163 rad/s. That is the torque-converter stall speed, so
    the vehicle is short of drive torque, not of grip.
  - `m113_g4` **reached the goal** on g0005 route_01 in 22.3 s, where both the stock M113 and the Gator stalled.
  - The quarter soil loads sum to the weight: 110.5 of 111 kN.
  - Lowest belly clearance: 0.15-0.38 m.
- **Gate (d), cost on MI350X: the M113 costs 4.1 times the Gator.** Wall seconds per simulated second, same node,
  one process at a time: M113 7.78 (both arms), 7.90 on the second route; Gator 1.89. The scouts had projected 7.7
  times.

## Worker timeout for M113 rows
The worst case is a 120 s horizon, which takes about 950 s on MI350X. On MI210 it should take about 1,550 s,
projected with the usual 1.6 factor. The worker's 2,400 s limit is enough, but set `CRM_EPISODE_TIMEOUT_S=3600`, as
PLAN 1.3 says, and stop claiming at least 1,800 s before the job limit on MI210.

## Not done
- No sample-A drives.
- No grouser or thick-pad f104 episode.
- No MI210 timing.
- No HMMWV/Gator bit-identity run through the dispatcher (that is M1's).

## Budget used
- Local soil: 4.6 min, all under the lock.
- Cluster: jobs 441588 (8 min 48 s) and 441589 (5 min 46 s), both mi3501x, 1 node. About 0.25 node-hours.

## For the orchestrator to decide
1. **PLAN amendment needed** (the plan names `veh.M113` and blames the pads):
   - Build path: Chrono's M113 C++ class `M113_Vehicle_SinglePin`, built exactly as `veh.M113` builds it (identical
     by test).
   - Brake: replaced by the shafts-brake construction. This is the only physics change.
   - Pads: flat.
2. **M113 rows need:**
   - `"config": "configs/crm_m113.json"` in each row;
   - that file staged in G4/configs;
   - `extra` `['--vehicle', 'm113']` or `['--vehicle', 'm113_g4']`;
   - the timeout above.
3. **Keep as caveats:**
   - The reference point is the front sprocket: goals count about 2 m early.
   - 0.5 ms step for the M113 against 1 ms for the Gator.
   - The chassis is not coupled to the soil.
   - On rigid ground the M113 still sleds at 0.24 m/s on 10°, because Chrono's contact model has no static friction.
     So no rigid M113 rows.
4. **Cost is 4.1 times the Gator,** within the plan's 8x quick-look gate.
