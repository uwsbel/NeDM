# AMD collection and reuse audit — 2026-09-09

The existing AMD installation can run this expansion. Use the same current
Chrono build for physics and RGB-D, with a headless first stage and cached
static-map rendering as a separate stage. No new build or Newton fallback is
needed to start a benchmark. This audit was read-only on AMD and the main NeDM
checkout; it did not launch, cancel, or alter jobs.

## Verified runtimes

`ssh amd` reaches `login1.hpcfund`; remote `HOME=/home1/harry` and
`WORK=/work1/dannegrut/harry`. The following installations were inspected in
their CMake caches and filesystem:

| Build | Vehicle | Sensor | FSI | Use |
|---|---|---|---|---|
| `/work1/dannegrut/harry/nrd/chrono-build` | Yes | Vulkan RT | No | Current FDM domain; recommended for both stages |
| `/work1/dannegrut/harry/nrd/chrono-build-fsi` | Yes | Vulkan RT | Yes | Existing expanded-state collection domain; separate provenance |
| `/home1/harry/chrono/build` | Yes | No | Yes | Available headless build; current `scene.py` imports sensor globally, so not a drop-in |
| `/home1/harry/chrono-pratik/build` | Enabled in cache | No | Yes | Build tree present; runtime import not validated |
| `/work1/dannegrut/harry/nrd/chrono10_cpu` | Yes | Legacy package | — | Original binary compatibility probes; do not silently mix with current build |

Core, vehicle and sensor imports succeeded from the recommended runtime on the
login node. No simulation was performed there. Its generated `chrono/ChVersion.h`
says `10.0.0`; that string alone does not identify the runtime. The source tree
`/home1/harry/chrono` is at `f54254fa9f81cfe00f4f9581607d0c11594d2cbd`, with
uncommitted sensor/SWIG interface changes. The recommended build is Release,
GCC 12.2.0, Python 3.12, and its CMake source root is that tree. The alternate
`chrono-pratik` source is `098a0d53f6ddd391cb514c7d1bd26a13e8d03f98`.

Binary SHA-256 at this audit:

```text
lib/libChrono_core.so
cb1015234ea5621426773bfb05aa9aba448bd0bc9d4a19258a5bf53bb87e6d01
lib/libChrono_vehicle.so
09ee59c80ec9f831062bff5aebf10f2e970cc01718ba88e13d86fb475d5190a8
bin/pychrono/_sensor.so
8313ff155966a8d86d68d26afbd3b2b1c072557c64c16ca518bdf8cdc7d554bd
```

Run inside a newly allocated compute job, with a new immutable `CODE_ROOT` and
new disjoint output directories:

```bash
source /work1/dannegrut/harry/nrd/env.sh
nrd_pychrono
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/scripts:$PYTHONPATH"
cd "$CODE_ROOT"
"$NRD_PYTHON" your_new_collector.py --chrono-data "$CHRONO_BUILD/data"
```

`NRD_PYTHON` resolves to `/work1/dannegrut/harry/venvs/nedm/bin/python`.
`nrd_pychrono` adds the selected build's `bin` and the existing PyTorch 2.10.0
Python stack. For rendering also call `nrd_use_lavapipe`; this uses
`/work1/dannegrut/harry/nrd/toolchain/lvp/lvp_icd.json` and the shared Mesa shader
cache. The AMD CDNA installation uses CPU lavapipe for Vulkan ray tracing, not
the training accelerators. Keep the shared environment and existing builds
unchanged; record their hashes in each new collection manifest.

## Staged collection is practical

`src/nedm/traverse/scene.py::build_scene(..., render=None)` already creates the
vehicle, collision heightfield and assets without a sensor manager. It still
imports the sensor module and constructs visual shapes; the recommended build
supports that import. The current FDM runner hardcodes a render specification
and initial/terminal sensor captures. Setting `--record-rgbd-stride 0` alone
therefore does **not** make it headless. A new runner should make the existing
headless scene option explicit and avoid every manager/tap call during physics.

For static rigid terrains and fixed rocks/trees, render one vehicle-free global
RGB-D map for each immutable arena/layout/camera combination. Join that measured
image to all trajectories using hashes and a map identifier. Pose/history remain
separate model inputs. This is a declared static-map observation contract; it
must not be mislabeled as the old current image, which contained the vehicle.
The vehicle silhouette can be omitted consistently from both training and
deployment. Where mutable soil, moving assets, or state-dependent occlusion is
part of the intended observation, a single static image cannot substitute for
current sensing.

Long maps also require a new camera scale and pixel-to-world transform. The old
100 m high, 47-degree, 256-pixel camera spans only about 87 m at datum. Do not
stretch an 80 m image over a hundreds-of-metres arena or let the model see scene
truth behind the cached image. Preserve RGB/depth calibration, range versus
axial-depth conventions, and true observed map coverage. A high-resolution map
can be cached once and queried by candidate patches without one full-resolution
image copy per training window.

Before expanding workers, run a paired headless/rendered collection benchmark
with identical initialization, controls and physics. Compare state, pose,
action, work, collision timing and endpoint arrays. The smooth-hill revision
already established an analogous passive-observer equality check, but a new
scene/runner split needs its own check.

Historical measured costs, not forecasts for the larger maps:

| Existing job | Work | Recorded wall time |
|---|---|---:|
| Focused FDM `409545` | 120 short episodes, 24 workers, sparse RGB-D | 290.0 s |
| Terrain FDM `409568` | 72 short episodes, 24 workers, sparse RGB-D | 160.3 s |
| Expanded-state shard collections | 904–905 episodes per shard, headless | 536–561 s per shard |

The old state collection Slurm script requested 128 CPUs and 96 fresh-process
workers. Use a smaller 8/16/32-worker sweep first for larger terrain meshes;
rendering workers need different CPU budgets because lavapipe itself is threaded.
Avoid every worker launching a full unbounded OpenMP/BLAS/Mesa thread pool.

## Existing data found and inspected

The exact-current FDM packs are under
`/work1/dannegrut/harry/experiments/traverse_mppi_20260908/data`:

- `fdm_rgbd_focused_pack_v2`: 230 MB; the train payload has 1,341 windows,
  20-step trajectories/work/events/commands and 16-by-24 histories. This is the
  closest replay/regression source for the current driver/runtime.
- `fdm_rgbd_focused_pack_v1`: 146 MB.
- `fdm_rgbd_fast_v1`: 3.2 GB; `fdm_fast_data_v2`: 148 MB.
- Actual focused records, cases and frame anchors remain under the sibling
  `rgbd_sim_v1/artifacts/traverse/` tree.

The expanded-state effort has **6,821** actual NPZ records under
`/work1/dannegrut/harry/experiments/state_ablation_20260908/artifacts/traverse/state_ablation_20260908/records`,
across 13 arenas. The manifest preserves these splits:

| Split | All episodes | NativePID episodes | Learned-tracker episodes |
|---|---:|---:|---:|
| train | 5,161 | 616 | 4,545 |
| val | 770 | 48 | 722 |
| diagnostic | 510 | 32 | 478 |
| test | 380 | 40 | 340 |

All 6,821 files were checked via their ZIP members: they have the 56-dimensional
state contract and **none has `all_state`**. The 56 fields include roll/pitch and
rates, world-vertical tire forces, spindle speeds, wheel velocities, deflection,
effective radius, horizontal tire-force components, suspension state, gear and
torque-converter state. They also retain actions, poses, power and exact command
references. They do not provide the full native tire-slip/force-vector/substep
work/contact-event contract now desired. Six separate `chrono10_probe/records`
files do have a 129-field `all_state` superset, but belong to the original-runtime
probe domain. Missing fields must remain missing rather than receiving invented
zero labels.

`/work1/dannegrut/harry/nedm/artifacts/traverse/wp7_cache_v1` contains 2,472
NPZs; a sampled file has 17-D states, actions, pose, power, route commands and a
64-by-64-by-64 latent map. `wp7_cache_sealed`, `wp8_cache_sealed2`, and
`full_v1`, `full_v2`, `full_v3`, `full_v4_partial` also exist. Keep sealed
test splits out of new training. The raw old RGB-D may have the known OptiX
registration defect; latent maps are not interchangeable with corrected
Vulkan RGB-D.

Reuse these records for explicit auxiliary/domain-transfer arms and regression
checks. NativePID, learned tracker and the standard substep `ChPathFollowerDriver`
are different closed-loop dynamics targets; feeding them as one untagged domain
would obscure the new model's error. New exact-driver episodes on genuinely new
long terrains are still needed.

## Power and terramechanics measurement boundary

The existing product
`engine.GetOutputMotorshaftTorque() * transmission.GetOutputMotorshaftSpeed()`
is engine-interface mechanical power. Despite the name, the transmission getter
returns motorshaft speed passed **to the engine**, not wheel-side speed. This
was checked in the actual AMD source:
`ChTransmission.h:73-75`, `ChAutomaticTransmissionShafts.cpp:159-160`, and
`ChPowertrainAssembly.cpp:30-35`. An initial concern based only on the getter's
name was corrected after this inspection. Save the raw torque/speed factors,
signed work and positive work separately. This does not measure fuel chemical
energy or consumption without an additional engine-efficiency/fuel model.

RigidTerrain + TMEASY can support grade-dependent traction, slip, tire loads,
vehicle attitude, suspension and mechanical-energy demonstrations. It does not
simulate deformable-soil rutting or sinkage. The FSI-enabled build exists for a
later explicit soil-domain experiment, but mixing soil models now would add a
second generalization problem and invalidate a static-map assumption where the
surface changes.

## Resource snapshot and next action

At 2026-09-09 20:56 UTC, `squeue -u harry` was empty. Scheduler availability is
transient: `mi2104x` had 10 idle 128-CPU nodes; `mi3508x` had all four nodes
allocated; `mi3501x` had seven idle 24-CPU instances; `mi3258x` had one idle
256-CPU node. `/work1` reported about 1.5 TB available. Slurm advertises unusual
`RealMemory=1` values here, so follow the existing scripts' CPU/partition requests
rather than introducing a large `--mem` request that may be unschedulable.

Use `mi2104x` for independent CPU physics shards and separate short image-cache
jobs; use the established PyTorch module on AMD GPUs for training. Independent
one-GPU training jobs on idle `mi3501x` can return feedback sooner than waiting
for an eight-GPU node; the current eight-model suite hardcodes exactly eight
devices, so a one-model wrapper is required. Preserve all current remote trees
and start a new isolated experiment root rather than syncing over either
`/work1/dannegrut/harry/nedm` or the prior FDM simulation workspace.
