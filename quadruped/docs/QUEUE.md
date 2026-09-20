# Queue

Ordered. Top item is next. Move an item to STATE.md when done, with its result.

## Now

1. **Params layer.** `machines.yaml` (collection hosts, GPU memory, Chrono build hash),
   `excitation.yaml`, `presets.yaml`, `training.yaml`, `transforms.py`.
2. **`doctor.py`.** Preflight: host permitted for the action requested, Chrono build hash
   matches the manifest, dataset integrity, preset/checkpoint agreement, GPU memory.
3. **Policy adoption.** Fetch `robot_lab/policy.pt`, assert the load gates, establish sign
   and observation layout by round-trip test against Chrono.

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
