# Go2 on sand: a velocity policy fine-tuned with PPO inside a learned sand NRD

A Unitree Go2 velocity policy stalled when it walked sideways on sand. We fine-tuned it with PPO only inside a
neural reduced dynamics model (NRD) of the robot on sand and on rigid ground. The NRD learned from Project Chrono CRM
sand episodes and MuJoCo rigid-ground episodes of the unchanged base policy. In a held-out Chrono sand test, the
released policy V2n s81 decreased the lateral stuck time to 0.214 of the base policy value. It did not pass the
pre-registered confirmation gates. It fell more often on medium uneven sand, moved the wrong way on soft sand and used
1.66 times the base power. This is follow-on work, separate from the NeDM paper.

The study author ran V2n s81 on a real Go2, on sand and on a rigid floor, on 2026-11-10 and 2026-11-11. The author
reported that it performed well: a higher foot lift and stronger lateral motion. The author also saw a small jitter at
some stops after lateral motion. The 2026-11-11 record also says "base still gets stuck". There is no robot log
([Real robot](#real-robot)).

NeDM keeps the results, the limits and the pinned release links here. The data, the NRD, the policy, the code and the
evaluation records are in the `go2_sand_finetune/` folder of the
[Hugging Face dataset](https://huggingface.co/datasets/harryzhang1018/NeDM/tree/aacf0f7c49217f1538bf3e16608784047e783ace/go2_sand_finetune).

## Names

| Name | Meaning |
|---|---|
| Base policy | The `unitree_rl_mjlab` Go2 flat-ground velocity policy (checkpoint `model_10000.pt`, ONNX `c58a3494`). It made all NRD training data. PPO started from it. |
| NRD | The learned dynamics model of the Go2 on rigid ground and on CRM sand. PPO trains only inside it. File names call it `FRESH-b`. |
| V2n s81 | The released policy: PPO recipe V2 in the NRD, PPO seed 81, iteration 299 (ONNX `1419d972`). |
| Chrono | Project Chrono CRM, a particle model of deformable sand. It is the judge of all sand results. |
| Stuck time | Lateral hold time in 1 s windows where the lateral speed is less than 0.4 of the command. A fall or an early stop counts the rest of the hold as stuck. |
| Stuck-time ratio | The stuck time of a policy divided by the stuck time of the base policy in the same jobs. 1.00 is the base policy. Lower is better. |
| Soils | Soft 100 to 120 Pa cohesion, friction 0.60. Medium 450 to 500 Pa, 0.68 to 0.70. Firm 1,800 to 2,000 Pa, 0.78 to 0.80. These are study levels, not measured sand. |
| Realistic cells | Medium flat, medium uneven and firm uneven sand. Soft sand is a stress soil. |

## Result

All results are simulation results. The confirmation suite is a held-out Chrono suite with new seeds. Its gates were
registered before any candidate ran on it.

| Test | V2n s81 | Base policy | Gate (pre-registered) |
|---|---|---|---|
| Chrono held-out suite: stuck-time ratio, realistic cells (blocks better, of 9) | **0.214** (9) | 1.00 | 0.75 or less, 8 or more of 9 blocks better, no cell worse: pass |
| Same suite: stuck-time ratio toward +y / -y | 0.31 / 0.11 | 1.00 | 1.0 or less: pass |
| Same suite: lateral progress / command, +y / -y | 0.67 / 0.94 | 0.17 / 0.21 | - |
| Same suite: early stops in 72 realistic episodes (all are falls) | 5 (all on medium uneven sand) | 0 | base + 2: **not satisfactory** |
| Same suite: wrong-way segments on soft sand toward -y, of 16 | 8 | 0 | base + 1: **not satisfactory** |
| Same suite: power / action rate, times the base value | 1.66 / 1.46 | 1.00 | 1.25 or less: **not satisfactory** |
| Label | NOT_CONFIRMED (G2 wrong way, G3 falls, S1 cost) | - | - |
| MuJoCo rigid ground, 120 development paths: tracking loss ratio (95 % range) | 0.854 (0.836 to 0.874), 0 falls | 1.00, 0 falls | upper bound 1.05 or less: pass |
| C++ deploy stack in MuJoCo, flat ground, 2 profiles | completed, 0 falls | completed, 0 falls | pass |

- A block is one cell and one repeat. The suite has 3 realistic cells and 3 repeats.
- No policy of the study got the label CONFIRMED. V2n s81 was released only as an experimental package
  ([`legged_rl_deploy` PR #3](https://github.com/harryzhang1018/legged_rl_deploy/pull/3)).
- The Chrono robot model lets only the feet and calves touch the particles.

## The problem: the lateral stall on sand

| Value (lateral +y 0.3 m/s, leading feet) | Base policy, rigid ground | Base policy, Chrono medium sand |
|---|---|---|
| Step rate of each foot | 1.68 Hz | 1.67 Hz |
| Sideways foot travel in each step | 10.2 cm | 1.8 cm |
| Steps that land within 5 cm of the last footprint | 0.00 | 0.81 |
| Next step stays in the own footprint, after a step into it | - | 0.95 |
| Lateral speed / command, first 2 s to after 6 s | - | 0.48 to 0.08 |

- The gait clock of the base policy fixes the step rate. Thus the lateral speed depends only on the step length.
- On sand the feet land in their own craters, and the robot stalls in 2 s to 5 s.
- On the held-out suite, the base policy was stuck for 0.849 of the planned lateral time in the realistic cells.
- The earlier NRD fine-tune of a prior study (ONNX `6f26f6fd`) got a stuck-time ratio of 0.891 there. It kept the
  base gait.
- A higher foot lift alone did not help. In the new CRM data, it changed the stuck share by +0.4 points (95 % range
  -6.4 to +6.7).
- In simulation, a longer sideways stride helped on medium and firm sand, but not on soft sand. Recipe V2 rewards a
  high swing and penalizes foot drag and lateral shortfall.

## Pipeline: data → NRD → V2n s81

Times are US Central local times. Hashes are the first 8 hex digits of the SHA-256. Code capsule and snapshot names
carry their own digests. `release_manifest.json` has the full values.

| Step | What | Size | SHA-256 | Job and hardware | Time |
|---|---|---|---|---|---|
| 1 Base policy | `unitree_rl_mjlab` Go2 flat-ground PPO, seed 42, 10,001 iterations, 4,096 environments, upstream commit `1425b15f` | 4.7 MB | `7ee303f9`, ONNX `c58a3494` | Euler job 95472, euler17 (NVIDIA A100) | 2026-10-22, 02:14 to 05:11 (2 h 57 min) |
| 2 Rigid data | MuJoCo Warp (mjlab) rigid plane. 8,700 planned episodes of the base policy with pushes, drops and landings | 10,443 pieces, 9.06 h, 2.9 GB | `b02c0c50` | Local workstation, RTX 5090 | Raw file times 2026-10-29, 22:27 to 22:28. Export 2026-10-30, 01:27 |
| 3 CRM data | Chrono CRM, 2,000 episodes of the base policy with target noise and pushes | 3,489 pieces, 5.94 h, 1.1 GB | `35b73ce2` | AMD array job 487049, MI210 | Start 2026-10-30, 01:57. Export 06:03 |
| 4 Lateral data | Chrono CRM, 480 episodes of the base policy. Slow lateral holds of 10 s or more, uneven beds, soft soils, swing-lift injection in 144 of 384 lateral episodes | AMD HIP build: 730 pieces, 1.84 h, 0.27 GB. Euler CUDA build: 149 pieces, 0.38 h, 54 MB | `2e78115a`, `f1079976` | AMD job 509760 (8 nodes with 4 MI210 each, 16.6 GPU-h, with the test episodes). Euler job 113924 (euler07, 5 RTX A4500, 3.27 GPU-h). Exports: AMD 509836, Euler 113929 | 2026-11-08, 22:57 to 23:48 |
| 4a Held-out lateral test | 60 episodes with a new seed. Never used for training or normalization | 105 pieces, 0.27 h, 38 MB | `e1f18d48` | AMD job 509760, HIP build | 2026-11-08 |
| 5 Split manifest | The parent split (rigid 3,081 and CRM 1,563 TRAIN pieces) plus all lateral TRAIN pieces (657 HIP, 135 CUDA) | 7.6 MB | `9ba6dce9` (parent `f1d264dd`) | Local | 2026-11-08, 23:47 |
| 6 NRD | Three Transformers, trained from random weights, seed 62, 13,500 updates in one job, no resume | Checkpoint 10 MB | `last.pt` `b41561c3`, loader export `501c9b8d` | AMD job 509956, one MI350X | 2026-11-09, 00:52 to 03:15 (job 2 h 24 min, training 8,602 s) |
| 7 PPO | Recipe V2, seed 81, 300 iterations, 1,024 NRD environments, 24 steps of 20 ms in each iteration. Start from the base checkpoint | 4.7 MB | `model_299.pt` `cdb54fb5` | Euler job 114010, euler08 (RTX A4500) | 2026-11-09, 05:21 to 05:31 (job 10 min 11 s, training 601 s) |
| 8 Deploy export | CPU re-export of `model_299.pt`: ONNX opset 18, static `[1, 47]` to `[1, 12]`, 4,000-row parity check | 0.76 MB | `1419d972` | Local CPU | 2026-11-09, 05:34 |
| 9 Deploy package | `legged_rl_deploy` PR #3, merged 2026-11-10. The ONNX bytes are unchanged | - | `1419d972` | - | 2026-11-09, 23:44 |
| 10 Real robot | Test by the study author on sand and on a rigid floor | - | Not recorded | Real Go2 | 2026-11-10 to 2026-11-11 |

**Data.**

- The base policy made all NRD training data. No episode of a fine-tuned policy went into the NRD.
- A piece is a record of up to 1,001 states at 10 ms (10 s). The robot state has 37 values: base position, quaternion,
  12 joint angles, base velocity, base angular velocity and 12 joint rates. The action is 12 joint position targets.
- The ordinary and rollout training draws of the NRD came from rigid data (50 %), CRM data (25 %), lateral HIP data
  (20.7 %) and lateral CUDA data (4.3 %). The TRAIN pieces held 2.68 h, 2.68 h, 1.65 h and 0.34 h.
- In the new lateral data, the base policy was stuck in 72.6 % of the low-noise lateral windows without lift (95 % range
  69.0 to 76.1). In the older CRM data with the same script it was stuck in 19.3 %. The two plans differ (slower
  commands, softer soils, uneven beds and long holds in the new plan). At matched conditions the shares were 65.8 % and
  57.1 %.

**NRD.**

| Network | Architecture | Input | Output |
|---|---|---|---|
| Core, collision and contact network | Each a Transformer encoder: 4 layers, width 128, 4 heads, feed-forward 512. 0.81 to 0.85 M parameters each | The last 4 robot states (20 ms step, heading frame) and the 12 joint targets. The collision and contact networks also get 3 ground features of the episode (rigid or CRM, cohesion, friction) | The change of the robot speeds in one 20 ms step. The collision and contact networks run for four foot-ground pairs |
| Policy actor (V2n s81) | MLP 512-256-128, ELU, embedded observation normalization | 47 observations, with the gait phase | 12 joint actions at 50 Hz. A 200 Hz low-level loop runs the joint targets |

- One job ran a staged schedule. The first 12,000 updates were core only (1,000), collision only (1,000), joint
  one-step (8,000) and joint with rollouts of 10 to 50 steps (2,000).
- Then 1,000 updates used 50-step rollouts with short lean and stall terms. The last 500 updates used only starts
  with 100 future steps.
- The trainer fitted the normalization once on TRAIN pieces, before update 1.
- On the held-out lateral test, with the base policy in the loop, the NRD predicted 33.2 % stalled 4 s windows. Chrono
  gave 54.2 %, and the NRD of the prior study 6.5 %.
- The main NRD gate gap was 0.210 (95 % range 0.10 to 0.33, limit 0.239). The NRD still missed 2 rigid accuracy rules
  ([Limits](#limits)).
- The export is the last checkpoint with loader metadata added: the two heading-frame fields, the run config and an
  export record. The weights are unchanged.

**PPO.**

- The actor, the critic and both normalizers started as bit-for-bit copies of the base checkpoint. The normalizers
  stayed frozen.
- The first 25 iterations trained only the critic. The action standard deviation was capped at 0.2.
- Recipe V2 adds 50 % lateral and 15 % diagonal commands. It starts 30 % of the CRM branches from stuck states and
  15 % from soft-soil states.
- Recipe V2 decreases the weight of the base-policy anchor term to 0.2 of its value for lateral commands. It adds
  swing-peak, foot-drag and lateral-shortfall reward terms.
- The NRD weights had the same digest before and after PPO (`ca01f9e1`).
- The released ONNX is not the run's own `policy.onnx` (`6e82a553`). It is a separate re-export of `model_299.pt`
  with the same weights. The ONNX-to-PyTorch difference was 1.91e-06 at most, on 4,000 recorded observations.

## Chrono and MuJoCo results

### Held-out Chrono suite

AMD MI210, 28 jobs, 13.2 node-h. 72 realistic episodes of lateral holds at 0.3 to 0.45 m/s. Base policy = the same
base policy in the same jobs.

| Cell (mean of 3 repeats) | Base policy stuck time (s of 136 s) | V2n s81 stuck time (s of 136 s) | Stuck-time ratio |
|---|---|---|---|
| Medium sand, flat | 120.2 | 12.9 | 0.11 |
| Medium sand, uneven | 130.7 | 45.9 | 0.35 |
| Firm sand, uneven | 94.9 | 15.1 | 0.16 |

- V2n s81 had less stuck time than the base policy in 9 of 9 blocks (one-sided sign test, p = 0.002).
- All 5 falls were on medium uneven sand: 5 of 24 episodes (20.8 %, 95 % range 9.2 to 40.5).
- The falls are roll-overs onto the trailing side while the robot slides backward (32 of 32 V2n s81 fall events in
  the fall audit). The base policy falls toward the commanded side, almost only on soft sand.
- On the soft stress soil, V2n s81 stopped early in 21 of 32 episodes (base policy 16).

### Development checks

These checks also guided choices during the study. Their numbers can be optimistic.

| Check | V2n s81 | Base policy |
|---|---|---|
| Chrono selection suite, stuck-time ratio (48 realistic episodes for each job set) | 0.2207 (6 blocks). 0.197 to 0.221 over 5 job sets | 1.00 |
| Same suite, deploy torque clip on (clip off) | 0.197 (0.199) | 1.00 |
| Same suite, medium-sand power, times the base value, clip on (clip off) | 1.56 (1.57) | 1.00 |
| Same suite, soft sand: stuck time, times the base value, over the job sets | 0.95 to 1.27 | 1.00 |
| MuJoCo rigid, 120 paths: lateral-path loss ratio / forward-speed loss ratio | 0.62 / 0.99 | 1.00 / 1.00 |
| MuJoCo rigid, 120 paths: power, times the base value (95 % range) | 1.22 (1.20 to 1.24) | 1.00 |
| Chrono video, medium flat sand, +y 0.45 m/s, 20 s: lateral distance / stuck time | 8.04 m / 0.0 s | 1.65 m / 17.4 s |
| Chrono video, medium uneven sand, -y 0.45 m/s, 20 s: lateral distance / stuck time | 6.70 m / 4.2 s | 0.86 m / 18.4 s |

- The two video cases were chosen after the held-out results were known. They show the higher power, but none of the
  falls or wrong-way slides.

### C++ deploy stack in MuJoCo

The packaged ONNX ran through the C++ deploy stack (`legged_rl_deploy` and `unitree_mujoco`) on flat rigid ground. One
run for each profile.

| Policy, profile | Speed error vx / vy / wz (m/s, m/s, rad/s) | Mean power (W) | Maximum tilt (deg) |
|---|---|---|---|
| Base policy, demo (31 s) | 0.083 / 0.059 / 0.125 | 18.4 | 3.81 |
| V2n s81, demo (31 s) | 0.071 / 0.041 / 0.119 | 22.5 | 5.53 |
| Base policy, stress (30 s) | 0.096 / 0.073 / 0.143 | 29.8 | 6.29 |
| V2n s81, stress (30 s) | 0.078 / 0.052 / 0.135 | 34.1 | 6.98 |

- Lateral speed / command on the demo profile: +0.5 m/s 0.70 (base policy 0.54), -0.5 m/s 0.84 (base policy 0.56).
- Both policies completed both profiles with 0 falls and 0 non-finite outputs. These are single-run values on rigid
  ground.

## Real robot

| Item | Record |
|---|---|
| Test | The study author ran V2n s81 on a real Go2 on sand and on a rigid floor, 2026-11-10 to 2026-11-11 |
| What the author reported on 2026-11-11 | Higher foot lift and stronger lateral motion. The record says "base still gets stuck" (it does not say if "base" is the robot body or the base policy). After a move to the robot's right (-y), the robot stops, jitters a little, then recovers. On the rigid floor, stops after lateral motion sometimes jitter, then recover |
| What the author said on 2026-11-17 | V2n s81 performed well on the robot |
| Robot logs, videos, trial counts | None |
| Which file ran | Not recorded on the robot. The author names ONNX `1419d972`. The package `provenance.json` still says `hardware_tested: false` |

- Treat the robot result as a qualitative report of the author, not as a measured result.
- A later simulation study, not in this release, examined the stop jitter. It showed the jitter mainly on rigid ground
  that slopes down toward the robot's right (-y), and weakly on 2 cm to 4 cm steps. Nominal flat rigid ground and sand
  did not show it. With added control delay or sensor noise, flat rigid ground showed it in 40.4 % of the V2n s81
  stops, and in 83.1 % of the base policy stops.
- On held-out slopes of 3 to 8 deg, V2n s81 jittered in 14.3 % of the stops after -y holds (95 % range 13.2 to 15.5,
  2,555 stops). The base policy jittered in 0.5 %.

## Selection history

1. The pool had 21 PPO runs: 8 combinations of recipe and NRD, in 2 NRDs.
2. A pre-registered rule picked the lowest stuck-time ratios on the Chrono selection suite. V2n s81 was rank 1 in the
   new NRD (0.2207). Ranks 2 and 3 got 0.2333 and 0.2346. This margin is less than the spread between job sets.
3. The held-out suite then gave NOT_CONFIRMED for all 4 picks, also for V2n s81.
4. The later selection rules of phases 4 and 5 did not pick V2n s81. Under the phase-4 rule, V2n s81 is excluded for
   early stops and soft-sand wrong-way holds. The phase-5 rule picked two ensemble policies. They are also
   NOT_CONFIRMED. Their policy files are not in this release. Some of their evaluation records and videos are. One
   of them, E1 s131, is also in `legged_rl_deploy` PR #3 as an experimental package.
5. The study packaged V2n s81 outside the rules, as an experimental candidate. Its README lists the failures.
6. The study author ran V2n s81 on the real robot.

## Limits

| Limit | Effect |
|---|---|
| NRD gates not passed | The NRD missed 2 rigid accuracy rules: velocity error 0.0468 m/s (limit 0.0444), roll and pitch error 0.0248 rad (limit 0.0241). A written waiver of the study coordinator, made during the study, let PPO use it |
| NRD seed | The main NRD gate is the gap between the NRD and Chrono in the share of stalled 4 s windows. Seeds 61 to 64 of the same recipe gave a mean gap of 0.290 (95 % range 0.176 to 0.404). Only seed 62 met the limit of 0.239. The NRD is the best seed, not a typical seed |
| Control NRD | A later control NRD (same data and seed, no stall term) passed all gates, by 0.006 on the main gate. It was made after the FRESH-b test result was known, so its pass uses the test data a second time. Its validation score (0.265) is above the validation limit (0.257). It came after V2n s81, which does not use it |
| PPO seed | Recipe V2 in the NRD, seeds 81 / 82 / 83: selection-suite ratio 0.22 / 0.28 / 0.23 |
| Falls | 5 of 72 realistic held-out episodes (6.9 %, 95 % range 3.0 to 15.2). The NRD predicts almost no falls: 0.000 to 0.004 of its 4 s rollouts |
| Power | 1.66 times the base value on the held-out suite, 1.22 times on rigid ground |
| NRD gain bias | The NRD over-predicts PPO gains by +0.29 (soft sand +0.49). Chrono must stay the judge |
| Soil | The soil levels are study levels. Nobody measured the real sand |
| Suite repeats | The 3 repeats of a Chrono suite shift only the policy phase (0 to 15 ms). They are not new soil draws. Thus the 95 % ranges of the fall rates are too narrow |
| Reproduction | Chrono CRM is not bit-deterministic on MI210. NRD training is not bitwise across GPU types. PPO set no determinism flags. A new run can only agree statistically |
| Code history | The study code was not committed to Git. The release keeps the exact code as archives with SHA-256 lists |

## Relation to the NeDM contact NRD

- The NRD uses the design of the NeDM [contact NRD](../contact_nrd/README.md): a core, a collision and a contact
  Transformer, 4 layers, width 128, 4 heads, feed-forward 512, 4-state history, 20 ms step.
- Its model code is a copy of `src/nedm/contact_nrd/` at NeDM main commit `6d3f4396d`, with additions. The additions
  are a ground code from the soil features of each episode, a fused-heading state frame and a staged training driver.
- One NRD covers rigid ground and CRM sand. Each NeDM contact NRD case has one fixed ground.
- We did not load this NRD with the unchanged NeDM main code. Use the loader in the release code archives.

## Release and reproduction

[`release_manifest.json`](release_manifest.json) pins the Hub revision and the SHA-256 of the full Hub manifest. The
Hub manifest gives the size and the SHA-256 of each file. Run this block from the NeDM repository root. It downloads
the policy, the NRD, the base policy and the code (less than 0.3 GB), and it examines each file:

```bash
pip install huggingface_hub
python - <<'PY'
import hashlib, json
from pathlib import Path
from huggingface_hub import hf_hub_download

pin = json.loads(Path("go2_sand_finetune/release_manifest.json").read_text())

def fetch(path, revision, size, sha256):
    p = Path(hf_hub_download(pin["repo_id"], path, repo_type=pin["repo_type"], revision=revision))
    assert p.stat().st_size == size, path
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    assert h.hexdigest() == sha256, path
    return p

hub = pin["hub_manifest"]
full = json.loads(fetch(hub["path"], hub["hf_revision"], hub["bytes"], hub["sha256"]).read_text())
files = full["files"]
if isinstance(files, list):
    files = {f["repo_path"]: f for f in files}
keep = ("go2_sand_finetune/policy/", "go2_sand_finetune/nrd/fresh_b/", "go2_sand_finetune/teacher/",
        "go2_sand_finetune/code/")   # add "go2_sand_finetune/data/" for the 5.0 GB of data
for path in sorted(files):
    if path.startswith(keep):
        print(fetch(path, pin["hf_revision"], files[path]["bytes"], files[path]["sha256"]))
PY
```

- Data revision: `ee97b73e395e690659a50b45db8a32bda97a9575`.
  Manifest revision: `aacf0f7c49217f1538bf3e16608784047e783ace`.
- Except the generated files (code archives, log archives, the compressed training log of the base policy, job records,
  file inventories and release documents), the files are byte-exact copies of the study files. They keep
  the paths of the workstation and of the two clusters. Change these paths to your own paths.
- `code/REPRODUCE.md` on the Hub tells how to rebuild each step from the archives: collection, NRD training, PPO and the
  deploy export. `code/source_manifest.json` gives the SHA-256 of each archived source file.

| Step | Exact code in the release | What a new run gives |
|---|---|---|
| NRD training | Code capsule `c0e07a22e57434cd` (frozen model source, driver, samplers, configs) and the split manifest | A new NRD that agrees statistically. Use the released NRD for exact work |
| PPO | Code snapshot `ppo_nrdv3_8e08e2544f0d`. The configuration is the base config `5ceaa0bc` plus recipe `162b4883` plus overlay `17d563ae` | A new policy that agrees statistically. A run needs a new output folder |
| Deploy export | Exporter and the base ONNX (for the metadata) | The same bytes. A new export with the archived code gave `1419d972` again (examined on 2026-11-17) |
| Chrono evaluation | Suite definitions, run and analysis scripts and results in `eval/` | The Chrono builds are not in the release |
