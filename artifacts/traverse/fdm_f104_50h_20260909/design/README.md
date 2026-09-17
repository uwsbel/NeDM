# F104 collection design

The generated initial cohort contains **6,000 PID-reference episodes in 500 start/goal groups**, using an exact copy of the specified F104 BMP. This is a prepared collection, not 50 hours already recorded. The runner must accumulate at least **180,000 actual recorded task seconds**, excluding settling and failed jobs, and report training duration separately.

The copied terrain is `assets/traverse/arena_f104_50h_v1`: 80 × 80 m, 512² pixels, height range −1.8 to 3.9 m. BMP SHA-256 is `5d5bc683b6a8d9e99fd323c4112752e0cad214b986319ea1b2396631104ee8ed`. No assets or terrain changes were added. The external source checkout was read only.

Each group pairs offsets 0/−4/+4 m with four speed profiles: constant 2, 4 or 6 m/s, and a smooth spatial 2→6→2 m/s profile. All twelve siblings share a split. The nominal 90/5/5 hash allocation produces **5,376 training, 360 validation and 264 test episodes**. These evaluate new start/goal/command groups on one fixed terrain; they do not establish unseen-terrain generalization.

The initial 6 × 3 m footprint passes explicit BMP geometry gates: fitted grade ≤6°, maximum sampled grade ≤12°, height range ≤0.65 m, and plane residual ≤0.20 m. These are gently sloped initialization candidates; the AMD native-height and settled-state gates must validate them. No physically safe route labels were assigned.

All 6,500 case/route file hashes passed the [integrity check](geometry_integrity.json). The 96-route pilot is preserved byte-for-byte. Reference lengths range from 28.23 to 83.55 m and headings cover all eight angular bins. Route geometry stays at least 5.58 m from the arena boundary. The collection curvature cap is 0.10 m⁻¹; this intentionally covers a broader command distribution than the frozen online planner's 0.025 m⁻¹ policy.

The [raster feature audit](raster_feature_audit.json) confirms the transformed feature coordinates against actual BMP heights: all five hills have positive local relief and all five craters negative relief. The generator uses `TerrainMap.features`, which mirrors authoring y coordinates into the loaded map frame. Raw metadata centers would miss most features. Native TerrainMap/Chrono agreement is a separate AMD audit owned by the parent task.

Nine features have central targeted groups. The boundary crater at approximately (−24.87, +29.02) m has no accepted central group under the unchanged initialization and complete-sibling geometry gates, but 24 general-route references enter its one-sigma neighborhood, approaching within 1.55 m. This does not establish physical infeasibility. The first generator attempt stopped on this stratum; its source and failure record are preserved in `generator_attempt_v1.*`. The completed generator advances to the next stratum after 1,000 unsuccessful geometry attempts without relaxing the gates.

The 120 s episode caps provide **200 hours of nominal capacity**, but idealized prescribed-speed traversal sums to only **24.98 hours** before acceleration, terrain and blockage effects. Those are neither measured nor predicted Chrono durations. Reaching 50 hours with 6,000 episodes requires a 30 s actual average; reserve groups are likely needed. The runner retains recovery attempts with its declared minimum-time/blockage-tail rule and counts only valid completion markers. Goal completion, blockage and boundary exit must remain distinct outcomes.

Generator command:

```bash
/home/harry/miniconda3/envs/nedm/bin/python scripts/generate_traverse_f104_collection.py \
  --groups 500 --pilot-groups 8
```

Outputs are `../cases/cases.json`, `../cases/pilot_manifest.json`, standalone cases and standard route JSON files. Each record includes filenames and hashes. Existing incompatible outputs are refused. `--groups` is configurable; to extend the cohort, generate a larger deterministic cohort in a new output directory and dispatch only previously unseen group IDs. Preserve this initial manifest and avoid recounting the prefix.

The production collector is the independently owned `scripts/collect_traverse_f104.py`; it accepts the standard case/route files. Its `episode_complete.json` contains actual elapsed seconds and artifact hashes. The parent task owns the single static terrain-only RGB-D render and deferred observation joins, so the headless cohort does not wait for per-group rendering.

## Reserve prepared

`../cases_reserve_v1/cases.json` contains only **groups 0500–1499**, adding 12,000 episodes. The combined inventory is 18,000 episodes, with 16,104 training, 1,020 validation and 876 test episodes. [Reserve integrity proof](reserve_integrity_v1.json) verifies all 6,500 reproduced initial-prefix files and all 13,000 new case/route files. The original 500-group manifest is unchanged.

The reserve adds one targeted group for boundary crater 9 and four references passing within 1 m of its center, closest 0.170 m. These are commanded-reference geometry measurements, not vehicle outcomes. All ten authored features now have targeted group coverage. The combined maximum capacity is 600 hours; the actual global collection goal remains 50 hours and unused reserve cases need not run.

Sync and dispatch `cases_reserve_v1` directly. `design/reserve_generation_v1` retains the full deterministic reproduction used to verify the prefix, including duplicate original groups, and is not a separate collection cohort.
