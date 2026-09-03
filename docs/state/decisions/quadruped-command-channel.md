# The Go2 checkpoint has no command channel

**Established 2026-09-03. This gates the quadruped case study.**

## The finding

`model_2999.pt` cannot be commanded. Not "was trained on one command" — **there
is no command input to write to.** Three independent mechanisms, any one of which
alone would prevent trajectory following:

**1. The observation slot is a literal, rebuilt every call.**
`chrono_crmenv.py:503`, inside `_compute_observations`:

```python
scaled_commands = torch.tensor([[0.5, 0.0, 0.0]], device=self.device) \
                       .repeat(self.num_envs, 1) * self.lin_vel_scale
```

It reads no buffer and no attribute. Our `self.command` is **dead code** —
verified empirically, not by reading: same state, yaw 0.0 vs yaw +0.5, and the
resulting observations are **bit-identical**.

**2. The training target has no yaw component.** `env_cfg` in full:
`target_lin_vel: [0.5, 0.0]` — two elements. No `commands` block, no
`command_ranges`, no `lin_vel_x`/`ang_vel_yaw`. Grepping the harness for
`resample`, `randomi[sz]e`, `uniform`, `command_range` returns **zero hits**, and
`target_lin_vel` is written once at `__init__` and never touched.

**3. Turning was actively penalised.** `chrono_crmenv.py:527-530`:

```python
target_ang_vel = 0.0                     # "assume target is 0"
ang_vel_error  = torch.square(target_ang_vel - self.base_ang_vel[:, 2])
ang_vel_reward = torch.exp(-ang_vel_error / 0.25)
```

Weighted 0.2 in the reward sum. Plus termination at 0.2 rad roll/pitch, tight
enough to kill most turning transients during training.

**The checkpoint is single-command by construction.** It is a *terrain-robustness*
policy — walk forward at 0.5 m/s over varied ground — not a locomotion-control
policy.

## Why this gates the case study

NRD learns a **command-conditioned** reduced model: given state and command,
predict the evolution. With no command input there is nothing to condition on,
and the model degenerates to "continue walking forward."

**A pure-pursuit outer loop is not reachable at any amount of integration work.**
It would emit yaw rates into a slot the network never reads. The architectural
parallel to the HMMWV's `ChPathFollowerDriver` — planner → waypoints → controller
— cannot be built on this checkpoint.

**This is a training-side problem, not a deployment-side one.**

## CORRECTION 2026-09-03: the yaw scale, and the probe's units

**Our command scaling is wrong for any nonzero command.** We apply
`* LIN_VEL_SCALE` to the whole three-vector. The legged_gym convention is
`cmd_scale = [2.0, 2.0, 0.25]` — **yaw is scaled by `ang_vel_scale`, not
`lin_vel_scale`.**

It is unobservable today because the yaw command is identically zero and
`0 × 2.0 == 0 × 0.25`. It is wrong the moment a command is driven, which is
exactly the regime an imported or retrained policy enters. **Any imported policy
must use `[2.0, 2.0, 0.25]`.**

**And it changes the units on the obs[8] probe.** The probe swept the raw
observation slot, so the *finding* stands — the network responds to that channel
incoherently. But the rad/s labels divided by 2.0 where they should have divided
by 0.25, so **every quoted command was 8× too small**: `obs[8] = 1.0` was
**4.0 rad/s**, not 0.5.

That reframes the result. The network was being driven at **four times the widest
yaw range any comparable policy is trained over** (±1.0 rad/s). "Responds
incoherently to a channel it never saw vary" becomes "responds incoherently when
driven far outside any trained range" — still consistent with our checkpoint
needing a command channel, and a materially weaker statement about the network.

## What was ruled out along the way

**Our command scaling is correct.** The harness multiplies the whole three-vector
by `lin_vel_scale`, yaw included, and our code reproduces that exactly. The yaw
channel was never mis-scaled; it was constant.

Confirming that *first* is what prevented a false conclusion: runs with yaw
commands would have produced paths identical to the baseline, and the report
would have read "the policy ignores the yaw channel" when the yaw never reached
the policy.

## Options

1. **Retrain with command randomisation and a yaw target.** The harness is an RL
   environment and supports training. Needs a `commands` block, resampling, a yaw
   term in the reward, and relaxed termination. Cost: unmeasured — establish the
   wall-clock for 3000 iterations before committing.
2. **Import a command-conditioned policy** from another framework. Those are
   normally trained with command randomisation as standard. This was the plan's
   rejected option 3 (sim-to-sim transfer risk), which the present finding makes
   more attractive than it was.
3. **Scripted parametric gait.** Direction becomes an explicit input. Less
   realistic, no learning, but a command space by construction.
