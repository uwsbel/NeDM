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

### There are TWO datasets and they are not interchangeable

| | families | commands | what it is for |
|---|---|---|---|
| **`go2_stratified`** | 8 | amplitudes **drawn per episode** | **training.** The one below. |
| `go2_discrete` | 11 | eleven **fixed** operating points, repeated | evaluation of repeatability at known commands |

`go2_discrete` is the first collection. It is not a training set and cannot
demonstrate command conditioning — eleven repeated points are not a sampled space.
It is kept because exactly-repeated conditions measure something a randomised set
cannot: how much of the variance is command versus seed.

**The eight families are a deliberate collapse of the eleven.** Four of them —
both march-in-place variants, `constant_high` and `reverse` — were four fixed
POINTS on one shape (constant `vx` at 0.15, 0.30, 0.50, −0.40). Once amplitude is
randomised per episode they are one experiment under four names, and keeping them
apart would have quadruple-weighted constant-velocity motion in the family balance.

**Any per-family figure must say which dataset it describes.** 44 × 11 and 14 × 11
are `go2_discrete`; 121 × 8 and 19 × 8 are `go2_stratified`.

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

The processed dataset and any training run. **The training config exists and has
been checked against the HMMWV anchor** — `configs/go2_transformer_v01_contact_mix25_onehot.json`
is key-for-key identical to
`configs/hmmwv_transformer_v07_..._mix25_rebal_rollout_onehot.json` except the
dataset paths, so the 75/25 mix, the Huber loss with per-domain channel
rebalancing, the rollout-selected checkpoint, and the 80x2000 compute budget all
match. That is what parity with the case study means, and it is now verified
rather than intended.

The policy environment is planned and not written; see
[quadruped-policy-env.md](../decisions/quadruped-policy-env.md). The headline is
that it is roughly a 150-line subclass of `hmmwv_tracking_env.py` rather than an
871-line parallel implementation, because the two studies share `DEFAULT_STATE_FIELDS`
verbatim and both actions are 3-D. It waits on levels 1 and 2 — writing it against
a reduced state not yet shown to propagate would be building on the thing under
test.

## Six metadata defects, all found by consuming the data

None was found by reading the collector. Three surfaced from writing the code that
consumes the data and two from a second machine reproducing a result. Full
taxonomy in [experiment-design.md](../lessons/experiment-design.md); the operational
residue is `scripts/collection/validate_go2_dataset.py`, ten gates that run between
the repair pass and preprocess, on a box with no torch.

Two are worth naming here because they reached model selection:

- **`scenario_family` was a constant** — one value per terrain against eight
  command families. It has **three** consumers (`trainer.py:796`,
  `build_combined_flat_crm_rl_references.py:135`, `references.py:111`) that each
  degrade differently, and the third is the worst because a seeded shuffle over one
  bucket returns the same plausible, unstratified reference bank every run.
- **`terminated_near_boundary` was never written to the Go2 index at all**, so a
  default-on exclusion in the reference builder silently never fired. The HMMWV
  collector writes it; ours diverged from the schema it was meant to match.

Both are fixed in the data rather than patched in the consumers, which is the only
fix that is correct in all three places at once.

## Open, and needs Kyle

1. **`n_layer` 6 or 8.** Our config anchors at 6, matching
   `configs/ablation_ofat/manifest.json`, whose `anchor_spec` is `L6_H8_E256_ctx128`
   with `train: false` because v07 already ran it. But `sections/appendix_data_scaling.tex`
   says the data-scaling study holds "the 8-layer backbone" fixed and calls its
   100% point "the same run used throughout the paper". **Those cannot both
   describe the deployed HMMWV model.** We match the repo, which is the thing that
   actually ran. Worth resolving before the parity claim goes in writing.
2. **dorm-pc's two local commits** (encoder draft, analysis scripts) are still
   unpushed, pending your word.
3. **Whether to vendor `go2_cts_150k.pt`** into the repo, and its citation and hash.
4. **The upstream Chrono report** — six defects found, none filed.

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

## Provenance: what is correct and what is not

**The `policy` field is correct in the datasets.** All 1,606 existing episodes —
968 stratified rigid and 638 discrete — name the imported checkpoint, verified by
count rather than inferred. The collection script has recorded it conditionally
since `db62c53`, the first commit that loads an imported checkpoint at all, so no
episode was ever written with the stale value.

*(A report that 968 episodes needed a metadata pass was wrong, and came from
checking the single-episode GATE script rather than the collection script. Both
contain the same `model_2999.pt` literal, which is what made the wrong one look
like confirmation.)*

**The gate script's field IS wrong** and unconditional — so gate and verify
summaries name the wrong policy. Not a dataset defect, but those are the summaries
anyone reads to check a reference number, **including the 2.1506 / 2.6 pair the
cross-machine equivalence decision rested on.** Fixed on `kyle-N7-B650E`.

**The real dataset defect is the checkpoint PATH.** Episodes record
`/tmp/go2_import/go2_cts_150k.pt`, which no longer exists since the checkpoint
moved off `/tmp`. That is worse than a wrong name in one respect: **a wrong name is
obviously wrong on inspection, and a right name at a dead path looks fine until
someone tries to use it.** Folded into the same metadata pass as the five origin
keys, deliberately **after** the CRM run rather than mid-run — four episodes
previously ran on a different actuator plant because a file changed under a live
batch.
