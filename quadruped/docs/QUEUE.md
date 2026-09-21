# Queue

Ordered. Top item is next. Move an item to STATE.md when done, with its result.

## Now

1. **`collect.py`.** Everything it needs now exists: a verified policy, a scene recipe
   that stands and walks on both terrains, the excitation layer, validity gating and the
   manifest schema. It wires them together, splits episodes at pushes, and emits a corpus
   with its provenance.
2. **Calibration sweep.** OU sigma against truncation rate and action identifiability,
   which sets the operating range by measurement rather than guess.
3. **`train.py`**, then **`finetune.py`** (`--method {analytic,ppo}`), then `evaluate.py`.

## Done

- **`params/machines.yaml`** -- fleet registry, every value measured.
- **`doctor.py`** -- preflight, tested on euler, sbel, a3 and d33 across pass and fail
  paths. Refuses the conda trap hash, refuses an action a host cannot do, and verifies
  torch with a real GEMM rather than `is_available()`.
- **Fleet standardised** -- pinned source build and env name `nedm` on all four desktops;
  d33 given a working ROCm torch.
- **Policy adopted and verified.** rl_sar `robot_lab/policy.pt`: plain MLP, 8 state_dict
  entries, statelessness confirmed empirically. Load gates refuse an encoder, a wrong
  entry count, a non-empty history, or a dimension mismatch.
- **Sign convention established at -1 by physics**, not inherited, with the evidence
  written into `params/policy.yaml`. A command-scaling bug was found in the same run.
- **`lib/excitation.py`** -- OU injection, sphere-uniform pushes, chirp probes, push
  scheduling that refuses infeasible schedules. 30-check self-test.
- **The robot walks**, rigid 95% and CRM 64% tracking.

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
