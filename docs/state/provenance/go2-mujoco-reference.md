# MuJoCo reference points (drift-free policy check)

CPU, deterministic, no GPU and no SPH. Re-running these separates "the policy's
computation changed" from "the physics changed" -- the control that was missing when the
CRM verdict moved 21.8% between 2026-09-10 and 2026-09-15 and could not be diagnosed
after the fact. See `../decisions/go2-crm-verdict-drift.md`.

Harness `scripts/evaluation/eval_go2_mujoco.py`, scene
`sbel-artifacts/mujoco_menagerie/unitree_go2/scene.xml`, command vx 0.6 / vy 0 / wz 0,
8 s, mujoco 3.13.0, recorded 2026-09-15 on kyle-sbel.

| policy | sha256[:16] | mae_vx | height | fell |
|---|---|---|---|---|
| `go2_cts_150k` (BASE) | `3e102d0d5e69aebd` | 0.3437 | 0.300 | no |
| `w_h15r0` (CRM fine-tuned) | `20c084424570341b` | 0.4134 | 0.278 | no |

Two observations, both independent of Chrono and of the surrogate:

- The CRM fine-tune is **20% worse on rigid ground**, confirming the specialisation
  result in a third simulator sharing no code with either.
- It stands **22 mm lower**, consistent with a gait adapted to sinking into soil.

Single command, single episode: a reference point, not an evaluation.

## Provenance of the base policy

`go2_cts_150k.pt` comes from https://github.com/wty-yy/go2_rl_gym
(`deploy/pre_train/go2/go2_cts_150k.pt`), a legged_gym-family repo. Its README names
**Isaac Gym** for training and **MuJoCo** for deployment, confirmed by Kyle 2026-09-15.
Isaac Gym is archived by NVIDIA and needs a developer account, Python 3.8 and an old
torch, so MuJoCo -- the authors' own sim-to-sim validation target, whose deploy config we
hold verbatim at `checkpoints/go2_mujoco.yaml` -- is the reachable third domain.

The policy therefore trained on **rigid** terrain only: the repo's deploy scenes are
flat, stairs, cross_slope, cross_stairs and race_track. CRM deformable soil is
out-of-distribution for it by construction, which is why fine-tuning has room to help.
