# Scoring CRM policies on the AMD HPC Fund cluster

Everything lives in `/work1/dannegrut/kyle/nedm/`. Nothing here writes outside
`/work1/dannegrut/kyle/`.

## Score one policy, or all of them

```bash
ssh hpcfund
cd /work1/dannegrut/kyle/nedm

# all 45 arms, 24 nodes at a time (24 is the mi2101x node count)
sbatch --array=0-44%24 score_policy.sbatch

# just the base policy (line 1 of policies.txt is index 0)
sbatch --array=0-0 score_policy.sbatch

# override the per-node episode concurrency
sbatch --array=0-44%24 --export=ALL,CONC=12 score_policy.sbatch
```

Each array task scores one policy on all 80 val episodes and writes
`out/crmtrack_<arm>_hpcfund.json`. Logs land in `/work1/dannegrut/kyle/logs/`.
Every task prints its own charged node-hours at the end, so the sweep's real
cost is read off the logs rather than estimated.

`policies.txt` is the task index: line N+1 is array task N. Line 1 is the base
policy `go2_cts_150k.pt`, lines 2-45 are the fine-tuned arms.

## Validate against the NVIDIA fleet

```bash
python compare_to_fleet.py \
  --cluster   out/crmtrack_go2_cts_150k_hpcfund.json \
  --reference stage/reference_BASE_nvidia.json
```

This pairs by `episode_id` and reports the aggregate, the per-episode delta
distribution, a sign test for one-directional bias, and the episode flip rate
against the fleet's own 0.93% baseline across five pychrono builds. It flags
both failure directions: a systematic offset, and a suspiciously exact match
(CUDA and HIP reorder SPH reductions, so a genuine re-simulation cannot agree
to machine precision).

## Measure, do not assume

```bash
sbatch bench_concurrency.sbatch     # episodes per charged node-hour at conc 4/8/12/16
sbatch smoke_devel.sbatch           # one episode on devel, 0.1x, for any code change
```

Run `smoke_devel.sbatch` after any change to the repo or the environment before
spending `mi2101x` time on it.

## Measured cost and throughput

All measured on `mi2101x` (1 MI210 + 16 cores, 0.1x charge), not estimated.

| | |
|---|---|
| One policy, 80 episodes, conc 8 | **01:29:27** wall (job 414064) |
| **Charged per policy** | **0.149 node-hours** |
| All 45 arms | **~6.7 node-hours**, ~3.0 h wall on 24 nodes |
| One episode, conc 1 | 61 s (job 414034) |
| Concurrency 4 / 8 / 12 / 16 | 522.8 / 515.3 / 513.2 / 514.3 episodes per charged node-hour |
| Node utilization during the run | 16 cores at 8.6% mean, memory 13.8% |

Against the desktop fleet at ~90 min per policy per box across four boxes: the
fleet does 45 arms in about 17 h, the cluster in about 3 h on 24 nodes, a ~5.6x
wall-clock speedup for 6.7 of the ~1144 remaining node-hours. Per-policy wall
time is nearly identical (89 min vs 90), so the gain is entirely concurrency,
not a faster part. The MI210 is GPU-saturated by ~4 concurrent episodes, which
is why one policy per node is the right unit and why `CONC` above 8 is pointless.

Whole-project cost to build and validate this workflow: **0.318 node-hours**,
including 0.083 wasted on one run killed by a `-t` set too tight (users cannot
raise `TimeLimit` on their own running job, so a short `-t` means resubmitting
from scratch; size it from a measurement).

## Validation against the NVIDIA fleet

Base policy, same 80 episodes, same seeds, ids paired one-to-one:

| | cluster (MI210, HIP) | fleet (`kyle-sbel`, CUDA) |
|---|---|---|
| episodes scored | 77 of 80 | 77 of 80 |
| mean `mae_vx` | **0.1607** | **0.1601** |

- paired mean delta **+0.0006 m/s (+0.35%)**, inside the fleet's own 0.155-0.162 band
- per-episode: median abs delta 0.0090, p90 0.0281, max 0.0481, sd 0.0160
- across-episode correlation **r = 0.9841**
- **no systematic bias**: cluster higher on 37, lower on 40; sign test p 0.73;
  paired t p 0.76; the offset is 0.6% of the metric's own episode-to-episode
  sd (0.0885), far too small to reorder arms
- **episode flip rate 0.00%** (0 of 80 scored on one machine only), against the
  fleet's own 0.93% across five pychrono builds. The same three episodes fall
  below `--min-rows` on both machines
- deltas are nonzero, as a genuine re-simulation on a reordered-reduction
  backend must be. The command signal is reproduced bit-exactly (`cmd_vx`
  matches to 16 digits), so the divergence is physics, not episode setup

Conclusion: cluster scores are interchangeable with fleet scores for this metric.

## Why one policy per node

Nodes are exclusive and billed whole, and an `mi2101x` node is exactly 1 MI210 +
16 cores. The charged cost of a policy is
`(80 / concurrency) x per-episode-wall x 0.1`, and that product does not change
when the 80 episodes are spread over K nodes: splitting divides the wall time by
K and multiplies the billed nodes by K. What splitting does change is overhead,
and only for the worse, since every node repays the fixed startup (module load,
pychrono import, torch import, SLURM dispatch) and the shards finish raggedly
because per-episode cost varies about 10x across command families.

Splitting only buys latency on a single policy. For that case,
`mkindex.py --shard K --of M` writes a round-robin shard (round-robin, not
contiguous, because the index is grouped by command family and a contiguous
split would hand one shard all the pivots and another all the weaves).

## What had to change, and why

**The repo patch** (`213b5e1` on `kyle/locomotion`). The CRM guard tested
`/dev/nvidiactl` and reported a missing NVIDIA driver as "no CUDA device",
which is a false negative on this cluster. It is now
`nedm.quadruped.gpu.require_gpu_backend()`, which tests `/dev/nvidiactl` then
`/dev/kfd`. The order is load-bearing: `sbel-ubuntu` has an AMD integrated GPU
with the ROCm driver loaded alongside the NVIDIA card its Chrono is built
against, so it presents both nodes, and testing `/dev/kfd` first would label the
whole desktop fleet `hip`.

**A second Chrono tree**, `$KWORK/chrono-build-parsers`. Scoring needs
`pychrono.parsers`, because the Go2 is loaded from a URDF through
`ChParserURDF`, and the original `$KWORK/chrono-build` was configured with
`CH_ENABLE_MODULE_PARSERS=OFF`. That tree is the only validated HIP pychrono we
have and the only one with a working Vulkan sensor backend, so it was left
byte-identical rather than reconfigured; the build job fingerprints it before
and after to prove that. Same source, same module set, same fat
`gfx90a;gfx942;gfx950` compile, plus PARSERS.

**Four dependencies** into `$TOOLCHAIN/urdf`, none of which exist on this
system: `console_bridge` 1.0.2, `urdfdom_headers` 1.1.1, `tinyxml2` 10.0.0 and
`urdfdom` **4.0.1**. The 4.x series matters: urdfdom 3.x still parses with
tinyxml v1, while Chrono does `find_package(tinyxml2 REQUIRED)` and
`ChParserURDF.h` includes `<tinyxml2.h>`. They install to `lib64`, so
`env-scoring.sh` puts both `lib` and `lib64` on `LD_LIBRARY_PATH`.

**No PyTorch was installed, deliberately.** `/share/sw/ai/pytorch/2.10.0`
already provides torch 2.10.0+rocm7.1 together with numpy 2.4.3, which is the
exact numpy the pychrono bindings were compiled against. A pip CPU-torch would
have pulled its own numpy and broken that ABI. The policy never touches the GPU
regardless: `imported_policy.py` loads TorchScript with `map_location="cpu"`.

**Episode metadata only, not the corpus.** The scorer never opens an episode
CSV; it reads the `.json` sidecar and `.config.json` beside it and uses
`csv_path` only as a key. So the staged set is 3.19 MB of sidecars rather than
the ~200 GB corpus. `csv_path` was rewritten to cluster paths preserving the
last four components, because `episode_id` is derived from them, which keeps the
ids byte-identical to the fleet's and makes the pairing exact.
