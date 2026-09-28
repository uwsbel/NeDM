# NOTES M1: Polaris vehicle + new soil dispatcher (2026-09-28 00:10 CDT)

Module M1 of PLAN section 6. Nothing existing was edited; nothing is frozen yet (the orchestrator freezes at staging).
`NEDM_VEHICLE` was never set. Local soil wall time used: about 6 min of the 10 min allowed (all under
`flock /tmp/luffy_crm.lock`). One cluster job: 441578 (mi3501x, 1 node, 45 min limit), outputs under
`G4/checks/M1/out_441578/`.

## What I built

| file | what it is |
|---|---|
| `scripts/ov_vehicle.py` | vehicle switch and factory (copy of the `ag_vehicle.py` pattern) for `hmmwv, gator, polaris, polaris_pc, polaris_4wd, polaris_w08, m113, m113_g4` |
| `scripts/ov_crm_collect.py` | the new soil dispatcher (name contains `crm_collect`): copy of `ag_crm_collect.py`'s logic calling `ov_vehicle` |
| `scripts/ov_polaris_check.py` | checks: `make-data`, `rigid`, `flat-soil` (settle / straight + brake / turn), `episode-check`, `compare` |
| `assets/traverse/vehicles/ov_polaris/` | private vehicle data: `Polaris_ov/` (re-framed chassis + 3 top-level vehicle files + power-corrected driveline and gearbox), `ov_polaris_belly.json` (431 belly points), `README.md`, `MANIFEST.json` (sha256 of every file) |
| `K4/checks/M1/` | `ov_m1_check.sbatch` (the cluster job), `worker_tasks.json`, `local/` (local runs and results) |

How each vehicle name is run by the dispatcher:
- `hmmwv` (or no `--vehicle`): only the JSON writer is wrapped, to add a `vehicle` block (`name: hmmwv`). The vehicle,
  soil wheels, spawn and every array are the frozen collector's own.
- `gator`: the same swaps `ag_crm_collect.py` installs, built from `ag_vehicle`'s own functions. Only the recorded
  wrapper path and its hash differ in the vehicle block.
- `polaris*`: Chrono's JSON Polaris behind a small adapter, with:
  - the chassis reference moved to mid-wheelbase (every chassis-frame location + (1.35763, 0, -0.42) m);
  - spawn at ground + 0.40 m;
  - soil wheels = one cylinder per wheel, r 0.25 m (`polaris_w08`: 0.33 m), width 0.2121 m;
  - driveline: stock as shipped (`polaris`, `polaris_w08`), power-corrected (`polaris_pc`: conical ratios 1.0, gear
    ratios x 0.25), or Chrono's shafts 4WD (`polaris_4wd`).
- `m113*`: `import ov_m113` (M2), then `ov_m113.install(ctx)`. `ctx` = `types.SimpleNamespace(vehicle, base_module,
  crm_collect, argv, wrapper, source_root, chrono_data, ov_vehicle)`. It may return None, `(scene_hook, frame_hook)`
  or a dict with `scene_hook`, `frame_hook`, `argv`. If `ov_m113.py` is missing, the dispatcher stops with a clear
  error. M2's `ov_m113.py` (already written) uses exactly this interface.

The vehicle data path points at a temporary folder with two links only while the Polaris is being built:
- `Polaris_ov/` links to the asset folder;
- `Polaris/` links to the Chrono build's own stock folder.

The path is restored straight after the build. Before any build, every file the loader will open is checked to exist:
a missing file would otherwise abort Chrono with no Python error. The asset folder is also checked against its
MANIFEST.

The Polaris vehicle block (outcome.json, collection_request.json, f104_episode.json) records:
- name, the JSON files and their sha256, the private-folder manifest and file hashes;
- frame shift, spawn, soil wheel geometry;
- driveline and the defect note;
- measured mass and spindle positions;
- the belly summary (outcome.json only).

`vehicle_extra.npz` holds the per-frame belly clearance.

Versions tested: `ov_vehicle.py` sha256 `3bc549ee8714acd8…`, `ov_crm_collect.py` `4cc116ec1ee10d58…`, asset MANIFEST
`82cf309e08575ec5…`. The files in the worktree are byte-identical to what cluster job 441578 ran. `ov_polaris_check.py`
changed afterwards in one respect only: it now deletes its temporary flat-arena folder at exit.

## Checks and results

**1. Build and settle on rigid ground (local conda build and cluster chrono-build-fsi on MI350X): pass, all four arms.**
- Mass 1,378.334 kg.
- Spindles at (+-1.358, +-0.616, -0.023 front / -0.015 rear) m from the reference point.
- Launch height 0.362 m (window 0.30-0.45); vertical speed 0.0035 m/s at 0.8 s.
- Identical numbers on both builds.

**2. HMMWV and Gator rows through the new dispatcher: identical to the frozen dispatchers.**
- Local (5090, 3 s horizon, route `f104_v2_group_0005_route_01`):
  - `crm_collect.py` vs `ov_crm_collect.py --vehicle hmmwv`: all 4 array files identical.
  - `ag_crm_collect.py` vs `ov_crm_collect.py --vehicle gator`: all 5 array files identical.
  - The Gator vehicle block differs only in `wrapper` and `wrapper_sha256`.
- Cluster (same GPU, back to back, full 120 s horizon): identical again.
  - HMMWV: goal at 13.05 s.
  - Gator: stalled, blockage stop at 34.0 s.
- Against the stored runs (other nodes, other dates):
  - HMMWV (`collect_v1`, node k007-003): same outcome and time (goal 13.05 s); arrays differ slightly (cross-node drift, as S3 found).
  - Gator (`G3/soil_v1`, node k003-007): same outcome, and every array identical.

**3. Polaris on f104 soil through the dispatcher: pass.** Passing here means crm_qa ok, launch check passed, launch
height 0.30-0.45 m, a finite 247 x 17 state, the vehicle block in all three JSON files, and `vehicle_extra.npz`
present.
- Local: `0005_route_01` reached the goal at 12.40 s. Launch height 0.385 m; lowest belly point 0.21 m above the
  original surface.
- Cluster:

  | arm | `0005_route_01` result | launch height |
  |---|---|---|
  | `polaris` | goal at 12.35 s | 0.385 m |
  | `polaris_pc` | goal at 12.30 s | |
  | `polaris_4wd` | goal at 12.55 s | |
  | `polaris_w08` | **stalled** (blockage stop at 34.0 s, valid episode) | |

- The HMMWV reached the goal on this route at 13.05 s; the Gator stalled.
- One row through the unchanged `crm_worker.py`: `polaris__f104_v2_group_0000_route_02`, with the extra
  `['--vehicle','polaris']` and `--crm-config` forwarded. Goal at 7.55 s; all checks pass.
- Real-time factor on the MI350X: 0.43 for every vehicle, the same as HMMWV and Gator.

**4. Flat soil, 80 x 80 m, production soil settings, on the cluster.**

Settle (from the straight runs):

| vehicle | markers per wheel | launch height (m) | vertical speed at 0.8 s (m/s) |
|---|---|---|---|
| Polaris (r 0.25 m) | 152 | 0.370 | 0.004 |
| Polaris w08 (r 0.33 m) | 244 | 0.399 | |
| HMMWV | 207 | 0.568 | |
| Gator | 88 / 170 | 0.303 | |

Every vehicle is below 0.05 m/s at 0.8 s.

Straight drive: speed steps of 5 s each, speed judged over the last second of each step, then a full brake from 6 m/s.

| vehicle | mean speed at 2 / 4 / 6 m/s | stop time, distance | result |
|---|---|---|---|
| polaris | 1.89 / 3.88 / 5.90 | 0.90 s, 1.9 m | pass |
| polaris_4wd | 1.95 / 3.89 / 5.83 | | pass |
| polaris_w08 | 1.92 / 3.90 / 5.91 | | pass |
| HMMWV | 1.89 / 3.85 / 5.94 | 0.80 s, 2.5 m | pass |
| polaris_pc | means within 7 % | | fails by a hair: one frame at 1.796 m/s against the 1.80 limit |
| Gator | 1.87 / 3.23 / 4.01 | | fails: cannot hold 4 or 6 m/s on soil |

Also measured:
- Sideways drift 0.02 m or less for every vehicle.
- Every vehicle rolls back briefly after the stop, at up to 0.36-0.39 m/s, the HMMWV included. This is the body
  pitching back, and it sits just inside the 0.4 m/s limit.
- Gears used: stock Polaris first gear only (the defect); `pc` gears 1-2; `4wd` gears 1-3.

Half circle, 10 m radius at 3 m/s:
- Every vehicle completed it.
- **Every vehicle fails the 0.5 m cross-track rule**, the HMMWV included:

  | vehicle | RMS cross-track error |
  |---|---|
  | HMMWV | 0.77 m |
  | Polaris arms | 0.79-0.81 m |
  | Gator | 0.95 m |

- Cause: the frozen follower's 5 m look-ahead cuts a 10 m-radius curve by about 0.8 m for any vehicle.

## Deviations, not done, open

- The straight drive uses 5 s per speed step, not S3's 8 s, so that it fits the production-size 80 m patch. The pass
  rule ("within 10 % after 4 s") is unchanged.
- The turn rule (cross-track < 0.5 m, PLAN 2.1 via S3 2.1) cannot be met by the frozen follower even with the HMMWV.
  **I think PLAN 2.1 is wrong here.** The orchestrator should read it relative to the HMMWV (Polaris 0.80 vs HMMWV
  0.77 m), or change the rule by amendment.
- Not built: a rigid-ground wrapper (soil only is in scope).
- Not tested:
  - `--base crm_collect_ext` with the Polaris.
  - The `m113*` path of the dispatcher. It needs M2's module; M2 already implements `install(ctx)` and returns the
    dict form.
- HMMWV rows through the new dispatcher now carry a vehicle block `{"name": "hmmwv", ...}`. The stored
  `collect_v1` rows have none. M4's `vehicle_ok` and `ov_eval_index` accept both.
- The dispatcher refuses to run if `NEDM_VEHICLE` is set at all.
- Staging must copy `assets/traverse/vehicles/ov_polaris/` into `G4/source/assets/traverse/vehicles/`. `ov_stage.sh`
  (M3) already does. If the folder is missing or changed, the Polaris refuses to build, with a Python error and a
  `collection_failure.json`.
- `polaris_w08` stalled on the one route where the other three arms reach the goal. It is a single route, but it
  suggests the wheel-size arm matters (as for the Gator).
- Cost:
  - job 441578 ran 9.6 min on one mi3501x node;
  - local soil wall time about 6 min;
  - the check job left 12 tiny temporary arena folders in `/tmp` on node k007-005-v2.
- Outputs:
  - `K4/checks/M1/cluster_441578/` (copy of `G4/checks/M1/out_441578`);
  - `K4/checks/M1/local/` (rigid, identity, episode check, flat-soil smoke).
