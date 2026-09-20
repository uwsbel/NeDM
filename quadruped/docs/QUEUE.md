# Queue

Ordered. Top item is next. Move an item to STATE.md when done, with its result.

## Now

1. **Policy adoption.** Fetch rl_sar `robot_lab/policy.pt`. Assert the load gates:
   Identity normaliser, `observations_history == []`, no encoder/estimator keys in the
   state dict, obs and action dimensions as built. Then establish the joint sign
   convention and channel ordering by round-trip test against Chrono -- never inherited,
   since the old policy's was carried on faith and a sign flip followed.
2. **Remaining params.** `excitation.yaml` (OU sigma ladder, probe fraction, push schedule
   and sphere sampling, initial-state bounds), `presets.yaml` (channel sets, moved out of
   `dataset.py` module globals), `training.yaml`, `transforms.py` (quaternion projected
   gravity, world-to-body rotation).

## Done

- **`params/machines.yaml`** -- fleet registry, every value measured.
- **`doctor.py`** -- preflight, tested on euler, sbel, a3 and d33 across pass and fail
  paths. Refuses the conda trap hash, refuses an action a host cannot do, and verifies
  torch with a real GEMM rather than `is_available()`.
- **Fleet standardised** -- pinned source build and env name `nedm` on all four desktops;
  d33 given a working ROCm torch.

## Next

4. **`collect.py`.** Excitation layer, failure gating with truncation, episode splitting
   at pushes, manifest emission.
5. **Calibration sweep.** OU sigma against truncation rate and action identifiability.
   Sets the operating range by measurement rather than guess.
6. **`train.py`**, then **`finetune.py`** (`--method {analytic,ppo}`), then `evaluate.py`.

## Gates before any corpus is accepted

- Gate 1 action identifiability: residual action variance and effective rank
- Gate 2 Jacobian: model `d s'/d a` vs Chrono finite differences -- NEVER RUN BEFORE
- Gate 3 rollout horizon: err/dist vs horizon, no-motion floor at 1.0
- Gate 4 coverage: state occupancy and command balance

## Deferred, carried from the old tree

- Why gradients beat PPO. Sample budget is refuted; the in-distribution hypothesis is
  untested. Cheap: truncate PPO to 15 steps from recorded starts.
- Dose-ladder `val_loss` recomputation under the fixed sampler. The rho = -0.80 result has
  a confound aligned with its own independent variable.
