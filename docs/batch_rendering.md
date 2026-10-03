# Batch rendering for NRD rollouts

`src/nedm/render` draws every world of a vectorized NRD rollout at every step: one RGB
and one depth image per world, as tensors on the same GPU the NRD model runs on. It is a
renderer only. The NRD model stays the dynamics, and nothing is simulated to make a picture.

It is built on Newton's Warp ray tracer. Newton holds the scene and the cameras, and no
solver is ever created. Because the renderer is pure Warp kernels, it runs wherever Warp
does: NVIDIA, AMD Instinct (through AMD's ROCm port of Warp) and CPU.

Status: verified on an RTX 3090 and on an AMD Instinct MI210, with the Go2 study's NRD model
and policy. Not yet used to train anything.

## Install

NVIDIA or CPU:

```bash
pip install -r requirements-render.txt
```

AMD Instinct (written for the AMD HPC Fund cluster, ROCm 7.2.0):

```bash
sbatch scripts/render/hpcfund/build_warp_rocm.sbatch     # about 5 minutes, including the tests
```

Newton 1.2.0 and Warp 1.13 are pinned as a pair, because AMD's port is Warp 1.13.0 and
Newton 1.2.0 is the release built against it. The same renderer code then runs on both.

## Use

```python
from nedm.render import BatchRenderer, Scene, cameras

scene = Scene.from_urdf("go2_description.urdf").add_ground()
renderer = BatchRenderer(scene, num_worlds=1024, width=128, height=128)

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
| Joint angles of a robot with a URDF | `Scene.from_urdf(path)` | `renderer.joint_map(state_fields, {...})`, then `render(joint_q=...)`. Newton runs the kinematics. | `scripts/render/go2_nrd_rollout.py` |
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

Go2 with its full visual meshes (about 400,000 triangles), 1024 worlds, one camera each,
5 s rollout of the fine-tuned policy in the NRD model. Measured 2026-10-03.

| | RTX 3090 | AMD Instinct MI210 |
|---|---|---|
| NRD model, per control step | 231 ms | 175 ms |
| Render, 128x128 with shadows, per frame | 264 ms | 341 ms |
| Views per second | 3,880 | 3,003 |
| Render, 128x128 without shadows | 130 ms | not measured |
| Render, 64x64 without shadows | 34 ms (about 30,000 views per second) | not measured |

Cost is close to linear in pixels, and shadows double it. The meshes have not been
decimated, which is the obvious next lever.

## Checks

```bash
PYTHONPATH=src python -m unittest discover -s tests/render -v
```

The renderer tests work their expected depths out by hand (a box at a known height under a
camera at a known height), so a pass means poses reached the right world and the camera
convention is the documented one. They are skipped when Newton is not installed.

To check a new machine, render one saved trajectory on it and on a machine you trust, then
compare `sample_frames.npz`:

```bash
PYTHONPATH=src python scripts/render/render_trajectory.py --traj traj.npz --urdf go2_description.urdf --out out/here
```

The 3090 and the MI210 agree on 99.992% of RGB pixels for the same trajectory, and on depth
to 0.3 mm at the 99.9th percentile. The pixels that differ are on silhouette edges.

## AMD notes

Everything here is in `scripts/render/hpcfund/`. The three things that cost time:

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

- Flat colors only. Hardware textures are the one Warp feature AMD's port does not run, so the
  ground checker is geometry and meshes are not textured.
- The scene is the same in every world, apart from the poses. Per-world terrain is not written.
- Anything the reduced state does not carry cannot be drawn. For the Go2 on soil that means no
  ruts or footprints: the ground is a plane at the soil surface.
- A policy trained on these images and validated in Chrono would see Chrono::Sensor images
  there. Depth is close to renderer-independent. RGB is not.
- Render time rose about 20% over a 5 s rollout (218 to 264 ms per frame on the 3090). The
  likely cause is that the acceleration structure is refit each frame and never rebuilt as
  the robots move away from where it was built. That has not been tested.
