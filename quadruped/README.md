# Study 4: a Go2 quadruped on CRM granular terrain

A Unitree Go2 walks on Chrono's CRM terrain (SPH granular soil). Full SPH simulation is
far too slow to train a controller against, so this study learns a neural reduced model
(NN-ROM) of the robot on that soil from recorded walking, fine-tunes the robot's existing
locomotion policy inside that model with PPO, and verifies every result back in full
Chrono, paired episode by episode against the unmodified policy.

- **Live write-up, with every table and figure:**
  <https://claude.ai/artifact/G8PHCRfk7M8Mu7b8rW2PbU>
- **All data and checkpoints:** the Hugging Face dataset
  [`ksha23/nedm-study4-go2-crm`](https://huggingface.co/datasets/ksha23/nedm-study4-go2-crm) (layout below). Nothing
  here depends on any machine of ours: the code is in this repository, everything else is
  in that dataset.
- **Branch:** `kyle/quadruped-pipeline` on uwsbel/NeDM. Author of this study: Kyle Sha.

## Result

The recipe below, stopped at PPO iteration 1000, in ten independently trained
surrogates, scored in Chrono CRM on 40 held-out command paths from ten command families
(37 usable; three are too wide for the largest soil bed and are skipped in every arm):

| | forward speed error | sideways | yaw rate |
|---|---|---|---|
| held-out CRM paths | **-52%** (-43% to -57%) | -28% (-9% to -34%) | **-67%** (-52% to -71%) |
| straight walking on CRM | -62% | | -75% |
| the same paths on rigid ground | -27% | | -71% |

Every surrogate improves every axis (one exception: seed 6 sideways, within noise).
Nothing fell. On rigid ground the fine-tuned policies are better too, so tuning in a
soft-soil model does not cost hard-ground walking.

## The loop

```
collect (Chrono CRM, 100 Hz) -> train NN-ROM -> PPO inside the NN-ROM -> verify in Chrono
```

1. **Corpus** (`collect.py`, `merge_corpus.py`): the base policy walks commanded paths on
   CRM with injected action noise, occasional pushes and randomised starts. `go2_crm_v2`:
   1,150 episodes (920 train, 230 validation, split by episode), 2,041,752 rows, about
   5.7 hours of robot time. Push windows are cut out; the recoveries are kept.
2. **Surrogate** (`train.py`): a transformer (6 layers, 8 heads, width 256, context 128
   rows = 1.28 s, 4.85 M parameters) maps a 36-D state and 12-D action to the next-state
   change over 10 ms. The 36-D state is the closure of the control interface: body
   velocities, attitude and rates, 12 joint positions and velocities, the gravity vector in
   the body frame, height and vertical velocity. Trained in two stages: one-step
   prediction (80 x 2,000 steps), then fine-tuned on its own 1 s rollouts (8 x 1,000
   steps). The second stage is what makes multi-second rollouts trustworthy.
3. **Fine-tune** (`finetune.py --method ppo`): PPO inside the frozen surrogate, 1024
   rollouts of 2 s per update, branches starting from recorded states, reward = command
   tracking error plus an uprightness term, stopped at about 1000 iterations. A guard
   watches the training log for signs of the policy exploiting model error.
4. **Verify** (`evaluate.py`, `paired_eval.py`): the base policy and the fine-tuned one on
   the same 40 held-out paths, same spawns, same Chrono binary; the paired change per
   axis with its standard error. Arms from different Chrono builds are refused.

## What each study showed

Each line links a question to its answer; the numbers, tables and caveats are on the
site and in `docs/STATE.md`.

| Question | Answer |
|---|---|
| Does the gain need the soil model? | About half of it. Surrogates trained on RIGID-ground data reach -28% forward on CRM against -52% for CRM-trained ones; on rigid ground the ordering reverses (-60% against -32%). Yaw is largely terrain-independent: that part is a flaw of the base policy. |
| What should the reduced model carry? | The 36-D control interface. Adding foot contact forces, foot sinkage, both, or 3-D foot forces improves neither accuracy nor transfer (mean -45% to -53%). |
| How big a model, how much data? | 6x256 on the normal corpus. Smaller (3x128) makes a weak run likely; larger (12x512) buys nothing at 6x the fine-tuning cost; doubling the data changes nothing, alone or with the larger model; less data makes a weak run likely. Every surrogate plateaus near 0.47 rollout error at 2 s, so the limit is the model's formulation, not its size, data or soil state. |
| Why 2 s branches? | 0.30 s branches make forward tracking worse on held-out paths in every surrogate tried, however good the surrogate; the effect of an action on forward speed over soil plays out over several gait cycles. |
| Why rollout training? | One-step models are usable to about 2 s. 1 s rollout training beats predicting no motion out to 10 s in 10 of 10 seeds (0.5 s training: 2 of 10). |
| Two training stages? | A schedule, not a requirement: one stage from scratch matches at 2.5x the compute, and is less reliable at the same compute. |
| Why PPO and not gradients through the model? | Analytic gradients work only at 0.30 s branches with a very small weight budget (best -36%); longer horizons make them noise, larger budgets steer into model error. PPO reaches -51% in the same surrogate. This is also why the first attempt found the opposite ordering: its one-step surrogates confined everything to 0.30 s. |
| How many parallel rollouts? | 1024. Forward gain grows from 64 to 1024 and flattens; 2048 is a few points better, inside the seed spread, at twice the cost (measured at the old stop). |
| When to stop? | About iteration 1000. Tracking roughly doubles the old weight-budget stop's gain by then and plateaus by 1500; robustness to 300 N pushes holds at the base policy's level only through 1000. |
| Does it make the robot more robust to pushes? | No, and it cannot here: the corpus tops out at 140 N pushes, so the surrogate never saw a recovery from the 240-300 N test. Training with disturbances in the surrogate does not help. The claim is scoped to tracking. |
| How often does it fail? | One collapse in about sixty runs at the recipe's size (a policy that tumbles, in one exploitable surrogate). The guard catches collapses that leave the data but misses some weak runs: it is an early warning, not a certificate, so every policy is scored in Chrono. |
| Why is Chrono the bottleneck? | 95.5% of a CRM step is the SPH soil. One robot runs 5.1x slower than real time on an MI210; the surrogate runs 17x faster for one robot and about 600x in throughput at 1024 rollouts. A 1000-iteration fine-tune is 4.5 GPU-hours against about four months of Chrono. |

## Where everything is

**Hugging Face: [`ksha23/nedm-study4-go2-crm`](https://huggingface.co/datasets/ksha23/nedm-study4-go2-crm)**, named by meaning rather than
by run id. Its `MANIFEST.tsv` lists every file with its original run name, source machine
and sha256; its dataset card describes each folder.

```bash
pip install -U huggingface_hub
hf download ksha23/nedm-study4-go2-crm --repo-type dataset --local-dir study4
cd study4/corpora/normal_corpus && tar -xzf go2_crm_v2.tgz          # the recipe's corpus
```

```
corpora/normal_corpus/           go2_crm_v2    the recipe's corpus
corpora/extended_corpus/         go2_crm_v23   v2 + 1,174 training-only episodes
corpora/rigid_corpus/            go2_rigid_v2  the same 1,200 episodes on rigid ground
base_policy/                     the base policy (rl_sar robot_lab, Apache-2.0)
assets/robot/go2_irrvis/         the Go2 robot model (unitree_ros, BSD-3-Clause)
surrogates/standard/             the ten recipe surrogates
surrogates/<variant>/            one-step-only, half-second rollout, small, large,
                                 extended corpus, quarter/half corpus, four channel
                                 sets, rigid ground, one-stage 16k/40k
policies/<study>/                every fine-tuned policy behind a result on the site
results/<machine>_<build>/       the raw paired-scoring records
```

In this repository: code under `quadruped/`, the exact job scripts that produced the
results under `quadruped/jobs/`, small result files under `quadruped/results/`, and the
channel presets in `quadruped/params/presets.yaml`.

## Reproducing

Environment and Chrono build: `docs/STANDARD.md` (env `nedm`, the pinned NeDM Chrono
source with its two patches; CRM needs FSI-SPH with GPU support). Then, from `quadruped/`:

```bash
# To skip steps 1-3, use the dataset's corpus (DATA=study4/corpora/normal_corpus after
# unpacking), a surrogate from study4/surrogates/standard/, or a policy from
# study4/policies/recipe_iteration1000/.
# 1. corpus: 24 shards x 50 episodes, then merge (all shards on ONE Chrono build)
U=study4/assets/robot/go2_irrvis/urdf/go2_description.urdf; B=study4/base_policy/policy.pt
python collect.py --corpus go2_crm_v2_s0 --out DATA --policy $B --urdf $U \
  --episodes 50 --terrain crm --pushes 2 --long-fraction 0.25 --seed 20260921
python merge_corpus.py --shards DATA/go2_crm_v2_s* --out DATA --name go2_crm_v2

# 2. surrogate: one-step, then 1 s rollout training
python train.py --corpus DATA/go2_crm_v2 --out M/s8 --seed 8
python train.py --corpus DATA/go2_crm_v2 --out M/s8_ms100 --seed 8 --init-from M/s8/best.pt \
  --rollout-loss-steps 100 --rollout-loss-batch 4 --epochs 8 --steps-per-epoch 1000 \
  --lr 1e-4 --min-lr 1e-5 --warmup-steps 200 --select-window 3

# 3. fine-tune: the recipe
python finetune.py --model M/s8_ms100/best.pt --policy $B --corpus DATA/go2_crm_v2 \
  --out FT/s8 --method ppo --seed 0 --steps 100 --branches 1024 --iters 1000 \
  --target-dw 1e9 --snapshot-every 500

# 4. verify in Chrono, base and arm on the same machine and build, then pair
python evaluate.py --policy $B --urdf $U --corpus DATA/go2_crm_v2 \
  --out EV/base --label base --terrain crm --paths 4 --seconds 15 --spawn-spread 1.0
python evaluate.py --policy FT/s8/policy_ft.pt ...same flags... --out EV/s8 --label s8
python paired_eval.py --base EV/base --arms s8=EV/s8
```

`quadruped/jobs/` has the SLURM and shell scripts actually used on each machine
(hpcfund MI210, euler A100/H100, the lab desktops), with their resource settings.
`diagnostics/` holds the measurement tools (`horizon_sweep.py` for surrogate accuracy
by horizon, `cost_profile.py`, `probe_ood.py`, `guard_rules.py`, and others).

## Limits and open directions

- Robustness beyond the corpus: teaching recovery from harder pushes needs a corpus with
  harder pushes, and probably the disturbance as a model input (the push-force preset
  exists but needs a corpus that keeps push windows).
- The accuracy plateau (~0.47 at 2 s in every surrogate): try a longer context or a
  different objective; size, data and soil channels do not move it.
- The guard: re-fit its rule against Chrono-scored labels for every logged run.
- `finetune.py` cannot resume mid-run; on preemptible partitions that costs whole runs.
- Not measured: leave-one-out ablations in the same surrogates; 2048 rollouts at the new
  stop; the Chrono GPU fault's root cause (contained: about 1 episode in 720, resumed).

## Document map

| File | What it is |
|---|---|
| `docs/STATE.md` | The lab notebook: every result as it landed, with job ids, newest sections first. Long. |
| `docs/EVALUATION.md` | How the metrics must be measured, and why (pairing, noise floor). |
| `docs/COST.md` | What Chrono CRM costs and where the time goes. |
| `docs/SOIL.md` | This study's soil against the HMMWV study's. |
| `docs/STANDARD.md` | Environment and the pinned Chrono build; also the lab's machines. |
| `docs/ARTIFACTS.md` | Manifest schema and the published dataset. |
| `docs/LESSONS.md` | What cost time, and the rule each produced. |
| `docs/RETRACTIONS.md` | Claims withdrawn, and what replaced them. |
| `docs/QUEUE.md` | Future work. |
| `docs/PLAN.md`, `docs/INVENTORY.md` | Historical: the rebuild plan and the survey of the old tree. |
