# Batch rendering for NRD rollouts

`src/nedm/render` draws every world of a vectorized NRD rollout at every step: one RGB
and one depth image per world, as tensors on the same GPU the NRD model runs on. It is a
renderer only. The NRD model stays the dynamics, and nothing is simulated to make a picture.

The scene, the poses and the cameras are plain torch and numpy. The drawing is done by a
backend, chosen by name, and the two backends draw the same scene from the same inputs:

| Backend | What it is | Runs on | License of the renderer |
|---|---|---|---|
| `madrona` | Madrona's batch ray tracer, driven directly from PyTorch | NVIDIA | MIT |
| `newton` | Newton's Warp ray tracer, with no solver created | NVIDIA, AMD Instinct, CPU | Apache-2.0 |

Status: both verified with the Go2 study's NRD model and policy, 1024 worlds. Madrona on an
RTX 3090. Newton on an RTX 3090 and an AMD Instinct MI210. Not yet used to train anything.

## Install

Madrona backend (NVIDIA only):

```bash
scripts/render/madrona/build.sh ~/madrona      # about 5 minutes
source ~/madrona/env.sh
```

It builds the C++ engine inside `madrona_mjx` at a pinned commit. The JAX and MuJoCo MJX
wrapper that project ships is not installed and not used.

Newton backend, NVIDIA or CPU:

```bash
pip install -r requirements-render.txt
```

Newton backend, AMD Instinct (written for the AMD HPC Fund cluster, ROCm 7.2.0):

```bash
sbatch scripts/render/hpcfund/build_warp_rocm.sbatch     # about 5 minutes, including the tests
```

Newton 1.2.0 and Warp 1.13 are pinned as a pair, because AMD's port is Warp 1.13.0 and
Newton 1.2.0 is the release built against it.

## Use

```python
from nedm.render import BatchRenderer, Scene, cameras

scene = Scene.from_urdf("go2_description.urdf").add_ground()
renderer = BatchRenderer(scene, num_worlds=1024, width=128, height=128, backend="madrona")

frames = renderer.render(cameras.follow(xy), joint_q=joint_q)   # joint_q: (1024, 19) torch, on the GPU
frames.depth_torch()    # (1024, 1, 128, 128) float32, metres along the ray, 0 where nothing was hit
frames.rgb_torch()      # (1024, 1, 128, 128, 3) uint8
```

Torch tensors on the render device are read in place and the outputs are views, so a policy
can consume them without a copy. `frames.rgb` and `frames.depth` are the numpy versions.

### Plugging a model in

A model needs two things: geometry and a way to turn its reduced state into poses.

| The model gives you | Scene | Poses | Worked example |
|---|---|---|---|
| Joint angles of a robot with a URDF | `Scene.from_urdf(path)` | `renderer.joint_map(state_fields, {...})`, then `render(joint_q=...)`. The forward kinematics is batched torch (`nedm/render/urdf.py`). | `scripts/render/go2_nrd_rollout.py` |
| Link poses from its own kinematics | `Scene.from_meshes([MeshBody(...)])` | `transform_from_matrix(kin.link_transforms(q))`, then `render(body_q=...)` | `scripts/render/arm_fk_demo.py` |
| A planar pose and a few angles (a vehicle) | `Scene.from_meshes(...)` | `base_transform(xy_yaw, z, roll, pitch)` for the chassis, `compose` for parts fixed to it | none yet |

The reduced states here carry no x, y or yaw. `PlanarPose` integrates them from the predicted
body velocities and yaw rate, with the same update the environments use. An environment that
already keeps a pose (for example `HMMWVNeuralTrackingEnv.current_pose()`) can pass it to
`base_transform` directly.

Joints are matched by name, with a sign and an offset where conventions differ:

```python
joint_map = renderer.joint_map(state_fields, {"FL_hip_joint": ("joint_fl_hip_pos_rad", -1.0), ...})
joint_q = joint_map.joint_q(state, base=base_transform(pose.xy_yaw, z, roll, pitch))
```

### Cameras

A camera is a transform per world, looking along its own -Z with +Y up.

- `cameras.follow(xy)`: third person, at a fixed world-frame offset from each robot.
- `cameras.look_at(eye, target)`: anything fixed or scripted.
- `cameras.mounted(renderer.body_q()[:, renderer.body_index["base"]], local)`: a camera
  attached to a link. Call `renderer.set_joint_q(...)` first so the link pose is current.

Several cameras per world: `BatchRenderer(..., cameras=4)` and pass transforms of shape
`(worlds, 4, 7)`. All cameras share one resolution and field of view.

## What it costs

Go2 with its full visual meshes (about 400,000 triangles), 1024 worlds, one camera each, on an
RTX 3090. Milliseconds per frame of 1024 views, and views per second. Madrona measured
2026-10-07, Newton 2026-10-03 and 2026-10-07.

| | Madrona | Newton |
|---|---|---|
| 128x128 with shadows | 36 ms (28,100) | 270 ms (3,800) |
| 128x128 without shadows | 28 ms (36,100) | 130 ms (7,900) |
| 64x64 with shadows | 15 ms (69,700) | 66 ms (15,400) |
| 64x64 without shadows | 12 ms (83,600) | 34 ms (29,900) |
| 256x256 with shadows | 108 ms (9,500) | not measured |

For scale, the NRD model itself takes 231 ms per control step at 1024 worlds on the same GPU.
With Madrona, rendering every control step adds about 16% to a rollout. With Newton it more
than doubles it.

On the AMD Instinct MI210 the Newton backend takes 341 ms per frame at 128x128 with shadows
(3,000 views per second), and the NRD model 175 ms per control step. Measured 2026-10-03.

The meshes have not been decimated, which is the obvious next lever for both.

## Checks

```bash
PYTHONPATH=src python -m unittest discover -s tests/render -v
```

- The renderer tests work their expected depths out by hand (a box at a known height under a
  camera at a known height), and the same tests run against each backend that is installed.
- The forward kinematics is checked by hand and against Newton's on a branching test robot.
- Madrona allows one renderer per process, so its two test scenes need two runs. The second
  is `NEDM_MADRONA_TEST=urdf`.

The two backends are independent implementations, so agreement between them is evidence that
both are right. On the same Go2 trajectory and cameras, over 7.7 million pixels:

- within 3 m of the camera (the robot and the ground around it), depth agrees to 0.001 mm at
  the 99th percentile
- 0.03% of those pixels differ by more than 1 cm. They are silhouette edges, where one
  renderer's ray clips the robot and the other's passes it.
- farther out, where rays graze the ground, depth agrees to 1.3 mm at the 99th percentile

RGB differs because their shading does.

To check a new machine, render one saved trajectory on it and on a machine you trust, then
compare `sample_frames.npz`:

```bash
PYTHONPATH=src python scripts/render/render_trajectory.py --traj traj.npz --urdf go2_description.urdf --out out/here
```

With the Newton backend, the 3090 and the MI210 agree on 99.992% of RGB pixels for the same
trajectory, and on depth to 0.3 mm at the 99.9th percentile.

## Madrona notes

- NVIDIA only. Its GPU code is CUDA, compiled at start-up by NVIDIA's runtime compiler. The
  first start takes minutes. `env.sh` sets the two cache variables that make later starts fast.
- One renderer per process. A second one fails when Madrona sets its GPU heap size again.
- A world with exactly one object crashes the engine. The backend adds an invisible second
  object in that case.
- The upstream project, `madrona_mjx`, is no longer maintained. The build is pinned to the
  commit that was tested, and needs a CUDA 12.5 toolkit, which `build.sh` installs into its own
  conda environment.
- Licenses: Madrona and `madrona_mjx` are MIT. Madrona bundles third-party code under other
  licenses, including Apache-2.0 (NVIDIA's CUDA core libraries and SPIRV-Reflect). Check
  `external/madrona/external` before relying on "MIT only".

## AMD notes

Only the Newton backend runs on AMD. Everything for it is in `scripts/render/hpcfund/`. The
three things that cost time:

- One file of AMD's Warp port (`reduce.cu`) does not finish compiling at full optimization.
  Their build script has an option that applies to that file only, and the build script here
  passes it.
- The CPU kernel compiler of that port does not link on the cluster, so the build is GPU only.
- torch bundles the ROCm 7.1 runtime and Warp is built against 7.2. Without one shared
  runtime, whichever is imported second finds no GPU. `env.sh` preloads the system runtime.

`env.sh` also sets two library paths that make torch work on the single-GPU partitions,
where it otherwise fails its first matrix multiply.

Only MI210 (gfx90a) was run. The build also targets MI300X (gfx942), untested.

## Limits

- Flat colors only. The ground checker is geometry and meshes are not textured.
- The scene is the same in every world, apart from the poses. Per-world terrain is not written.
- Anything the reduced state does not carry cannot be drawn. For the Go2 on soil that means no
  ruts or footprints: the ground is a plane at the soil surface.
- A policy trained on these images and validated in Chrono would see Chrono::Sensor images
  there. Depth is close to renderer-independent. RGB is not, and the two backends here already
  shade differently from each other.
- With the Newton backend, render time rose about 20% over a 5 s rollout. The likely cause is
  that its acceleration structure is refit each frame and never rebuilt. That has not been
  tested. Madrona's stayed flat.
