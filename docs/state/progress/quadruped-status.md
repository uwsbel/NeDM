# Study 4 — quadruped on CRM: where it stands

**Updated:** 2026-09-03. **Status: collecting.**

The contribution is **contact-mode conditioning**, a methodological extension the
manuscript names as future work — *not* "a fourth vehicle." Full reasoning and the
staged plan: [`../decisions/quadruped-case-study-plan.md`](../decisions/quadruped-case-study-plan.md).

## What changed today, in the order it mattered

**1. The plant had no actuator model.** Joints were `ChLinkMotorRotationAngle` —
kinematic constraints, tracking commands to 10⁻¹⁵ rad with unbounded torque.
Switched to torque motors with a PD law at kp 20 / kd 0.5 against the URDF's real
effort limits (23.7 N·m hip and thigh, 45.43 N·m calf), PD at 500 Hz under a
50 Hz policy. Both plants remain selectable; the old one reproduces every historical
number bit-for-bit. See [`../decisions/quadruped-actuation.md`](../decisions/quadruped-actuation.md).

**2. The in-house policy does not survive that plant.** `model_2999` was trained
against infinite joint stiffness and has never produced a torque. On the torque
plant it inverts at 1.57 s. Expected, and not a fault to fix.

**3. So we imported one, and it works better than ours ever did.**
`wty-yy/go2_rl_gym` / `go2_cts_150k.pt`, MIT over BSD-3. Observation is the same
45 channels, trained against exactly kp 20 / kd 0.5, and — the point —
**commands randomised over ±0.5 m/s and ±1.0 rad/s yaw.**

| | rigid | CRM |
|---|---|---|
| straight travel | 2.1506 m | **2.2218 m** |
| max tilt | 2.6° | 5.4° |
| yaw tracking | 90% | 88.5% |
| fell | no | no |

**No meaningful degradation on soil.** Details:
[`../decisions/quadruped-imported-policy.md`](../decisions/quadruped-imported-policy.md).

**4. The command channel is therefore obtained without retraining.** The 35 h
retrain is off the table, and the "finetune from a command-conditioned base"
fallback is unnecessary — there is nothing to finetune away.

## The finding that justifies the case study's structure

| commanded forward velocity | achieved, rigid | achieved, CRM |
|---|---|---|
| 0.30 m/s | 0.030 | **0.145** — nearly 5× |
| 0.50 m/s | 0.337 | 0.347 — agree to 3% |

Same policy, same command, same plant. **The manuscript asserts terrain
conditioning from tire physics; this measures it on the quadruped**, and shows the
effect is regime-dependent — terrain matters most at low command, which a context
input represents and an unconditioned model cannot.

A port bug would affect both terrains alike, so it cannot be one. Mechanism
(compliance-assisted translation vs foot slip) is open; the logged slip channel
would settle it.

## The dataset

100 Hz, matching the HMMWV. 16 s episodes, 1475 rows after discarding the 1.25 s
ramp and settle. **Eight** command families mirroring the paper's maneuver-family
practice rather than uniform sampling, with **amplitudes stratified per episode**.

*(The first collection had eleven, but four of them — two march-in-place variants,
`constant_high` and `reverse` — were four fixed POINTS on one shape. Once amplitude
is randomised they are one experiment, so they collapsed into `constant`. Keeping
them apart would have quadruple-weighted constant-velocity motion in the family
balance.)*

| | episodes | transitions | status |
|---|---|---|---|
| rigid | **968** (121 × 8) | ~1.4 M | 968/968 clean, **0 fell** |
| CRM, `kyle-sbel` | 152 (19 × 8) | in progress | seed offset 0 |
| CRM, `kyle-N7-B650E` | 152 (19 × 8) | in progress | seed offset 1,000,000 |

Two machines collect CRM in parallel — **38 episodes per family pooled**. Verified
equivalent first: rigid identical to four decimals across `sm_86` and `sm_120`, CRM
within 0.88%. Every episode stamps `machine`, `gpu_name`, `gpu_arch`, `seed_offset`
and `git_commit` so origin is recoverable if a cross-box artefact ever appears.

Scale chosen from the paper's own data-scaling appendix: **20% of their data
already gives single-digit rollout error**, and the curve is flat past 80%.

Both action candidates are logged — 12 joint targets *and* the 3-D command — so
the choice is made by ablation rather than assertion. The command is the likelier
one: the imported policy is part of the plant, exactly as the HMMWV's powertrain
is, which makes level-3 an outer loop issuing commands rather than a policy
learning to walk inside a learned model.

## What is not done

The processed dataset, the training config, and any training run. Level 3
(closed-loop transfer) is reachable now that commands exist, but is not started.

## Landmines, all measured

- **The foot is smaller than the SPH kernel support** (22 mm against 40 mm), so it
  floats 22.9 mm above true contact and never penetrates. Everything previously
  read as sinkage was surface deflection.
- **Contact cannot be detected by a force threshold on soil** — no bimodality, no
  plateau; a plain threshold invents 66% of its transitions. Use hysteresis.
- **The imported policy is stateful** — a 5-step history behind a 45-wide
  interface. Reload per episode.
- **The bed extent has bitten three times**: spawn edge, turning-circle margin,
  and a far-edge GPU crash. Now a logged termination status.
