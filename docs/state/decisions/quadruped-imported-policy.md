# The imported policy walks and turns

**2026-09-03. This obtains the command channel without retraining.**

Source: `wty-yy/go2_rl_gym`, `go2_cts_150k.pt`. MIT over inherited BSD-3.
Not vendored — the 1.9 MB checkpoint sits outside the repo pending a decision.

## Result

**Rigid, torque plant, PD at 500 Hz:**

| | travel | fell | max tilt |
|---|---|---|---|
| **imported** | **2.1506 m** | **no** | **2.6°** |
| `model_2999`, position plant *(old baseline)* | 3.1100 m | no | 6.6° |
| `model_2999`, torque plant | −0.9165 m | **yes, 1.57 s** | 179.9° |

About 63% of commanded speed after the ramp and settle are removed — a reasonable
cross-simulator transfer — and **steadier than `model_2999` ever managed on its
own plant.**

**Yaw sweep — the point of the exercise:**

| commanded | achieved | tracking |
|---|---|---|
| −57.3 °/s | −52.77 | 92% |
| −28.6 | −26.38 | 92% |
| 0.0 | +0.81 | — |
| +28.6 | +27.09 | 95% |
| +57.3 | +51.50 | 90% |

Correct sign, 90–95% magnitude, monotonic, symmetric about zero. Nothing fell;
max tilt 2.6° throughout. **`+1.0 rad/s` traces an anticlockwise circle of ~0.45 m
radius and returns to its start**; `−1.0` mirrors it.

**`forward_travel_m` is near zero on the turning runs and that is correct** — it
measures x displacement and a closed circle ends where it began. Do not read it as
failure.

## The sign convention is a Chrono URDF property, settled at last

Holding *their* default pose under the servo:

| sign | base z | tilt | feet |
|---|---|---|---|
| **−1** | 0.1779 | 33.6° | 0.061–0.088 — **below the base** |
| +1 | 0.1033 | 179.8° | 0.402–0.411 — **0.4 m above the base** |

**This settles a question open all day.** The negation applies to a policy that
has never touched Genesis *or* Chrono, so it is a **Chrono URDF joint-sense
convention**, not a Genesis quirk — and should hold for any policy trained against
the standard Unitree URDF. First actual evidence for why the harness negates.

## Interface, verified rather than assumed

Observation block order read line by line from `deploy/deploy_mujoco/deploy_go2.py`
and identical to ours: `[ang_vel×0.25, gravity, cmd×cmd_scale,
(q−default)×1.0, dq×0.05, prev_actions]`, previous actions **unscaled** and
permuted into model order. Their PD is `kp(target−q) + kd(0−dq)` — ours exactly.

**The policy is stateful.** TorchScript, and internally teacher-student: a 5-step
observation history feeds a 32-dim latent that the actor consumes alongside the
current observation. Call once per control step in order, and **reload per
episode**. Cross-run contamination is impossible in practice because each run is a
separate process.

**Joint order is FL, FR, RL, RR** — *not* the Genesis FR, FL, RR, RL we inherited.
Reusing `CHRONO_TO_POLICY` would silently swap left and right legs.

## A camera that tracks the subject cannot show the subject's path

The first turning clips used the overhead camera, which **tracks the robot in x
and y** — a feature built deliberately, because a static frame loses a robot that
walks 2.5 m.

It re-centres the robot every frame. Heading rotation is visible; **the curved
path is not** — which is precisely the claim being recorded. The numbers would
have looked right and the video would have quietly failed to show the thing it
existed to show.

Redone with a **fixed** overhead at (0, 0, 3.2), verified to keep the robot in
frame through the final frame so nothing is cropped on the hard turn.

**The general form: match the camera to the claim, not to the subject.** Tracking
is right for "does it walk"; fixed is right for "where does it go."
