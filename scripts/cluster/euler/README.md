# Scoring CRM policies on UW-Madison Euler

Everything lives in `/srv/home/kasha2/nedm/`. Nothing here writes outside
`/srv/home/kasha2/`, and in particular nothing writes to
`/srv/home/groups/sbel`, which is shared with the whole lab and has no quota.

Read [`docs/state/machines/euler.md`](../../../docs/state/machines/euler.md)
first if you have not: the login is passcode + Duo and an agent cannot perform
it, the default partition contains no nodes, and there is no accounting of any
kind to make a runaway job or a runaway writer self-limiting.

## Score one policy

**Prefer the per-GPU script.** A single CRM episode saturates one GPU, so the
unit of parallelism here is a GPU, not a thread.

```bash
ssh euler
cd /srv/home/kasha2/nedm

# one policy across 3 of euler19's 4 A100s, one episode per GPU
sbatch --export=ALL,ARM=go2_cts_150k,M=3 \
       -p sbel -w euler19 --gres=gpu:3 -c 6 -t 03:00:00 \
       score_policy_pergpu.sbatch

# the single-GPU form, for a one-GPU node or an unattended sweep
sbatch --array=0-0 score_policy.sbatch
sbatch --array=0-$(( $(wc -l < policies.txt) - 1 ))%2 score_policy.sbatch
```

Both write `out/crmtrack_<arm>_euler.json` in the same record schema and with
the same `episode_id` keys the desktop fleet writes, so the files pair directly.
Logs land in `/srv/home/kasha2/logs/`.

`policies.txt` is the task index for the array form: line N+1 is array task N,
and line 1 is the base policy `go2_cts_150k.pt`.

## Validate against the fleet

```bash
python NeDM/scripts/cluster/compare_to_fleet.py \
  --cluster   out/crmtrack_go2_cts_150k_euler.json \
  --reference stage/reference_BASE_nvidia.json
```

This pairs by `episode_id` and reports the aggregate, the per-episode delta
distribution, a sign test for one-directional bias, and the episode flip rate
against the fleet's own 0.93% baseline across five pychrono builds. It flags
both failure directions: a systematic offset, and a suspiciously exact match.

`stage/reference_BASE_hpcfund.json` is the AMD cluster's run of the same policy,
for a second, independent comparison.

## Smoke first

```bash
sbatch smoke.sbatch                                   # one episode on an A100
sbatch -p sbel -w euler16 --gres=gpu:1 -c 4 smoke.sbatch   # ... on the 2080 Ti
```

Run this after any change to the repo, the Chrono tree, or the environment,
before spending a node on a full pass. It prints which node, which GPU, which
pychrono and which parsers module it actually got, because every failure it
catches is one of those not being what the script assumed.

## Concurrency: the knob is GPUs, not threads

Measured on euler19, same 80-episode val index, same base policy, episodes taken
in the same index order:

| | |
|---|---|
| `--concurrency 6`, 1 A100 | 10 episodes in **44 min 39 s** (job 51573) |
| `--concurrency 1`, 2 A100s | 8 episodes in **8 min 06 s** (job 51744) |

A single episode already saturates the device: `nvidia-smi` read 100%
utilization with 6.0 GB in use across six concurrent episodes, about 1 GB each.
Stacking them onto one GPU overlaps nothing. It time-slices work that was
already serial and pays six CUDA contexts of switching for it, and the measured
result is roughly **half** the per-GPU throughput of running one at a time.

hpcfund found the same curve *flat* from concurrency 4 to 16 on an MI210 and
concluded the GPU was saturated; on the A100 the same saturation shows up as an
actual loss. Either way the conclusion is the same and the fix is different,
because an hpcfund node is one GPU billed whole while Euler's dense nodes carry
eight.

## Wall time, measured

| | |
|---|---|
| One episode, 1 A100, alone | **65 s** (job 51571) |
| One episode, 1 RTX 2080 Ti, alone | **112 s** (job 51743) |
| **One policy, 80 episodes, 3 A100s, one per GPU** | **56 min 44 s** (job 51751) |
| Chrono rebuild, euler16 at `-c 32` | **17 min 52 s** (job 51527) |

Against the desktop fleet at about 90 min per policy per box, and against
hpcfund at 89 min per policy on one MI210. Per-GPU throughput is comparable to
both; the gain here is that a single job can hold several GPUs.

## Validation against the fleet

Base policy `go2_cts_150k`, the same 80 val episodes, same seeds, ids paired
one-to-one. Full output in the run log for job 51751.

| | Euler (A100, CUDA) | fleet (`kyle-sbel`) | hpcfund (MI210, HIP) |
|---|---|---|---|
| episodes scored | 77 of 80 | 77 of 80 | 77 of 80 |
| mean `mae_vx` | **0.16363** | **0.16009** | **0.16065** |

Euler vs the fleet:

- paired mean delta **+0.0035 m/s (+2.21%)**, correlation **r = 0.9832**
- **episode flip rate 0.00%** against the fleet's own 0.93% across five pychrono
  builds. The same three episodes fall below `--min-rows` on both machines
- **no one-directional bias**: higher on 42, lower on 35, sign test p 0.425.
  The paired t is p 0.064, which is not significance but is not nothing either
- **the offset is a tail, not a shift.** Drop the largest 5 of 77 per-episode
  deltas and it falls from +2.21% to **+0.41%**; drop 10 and it turns negative
  (-0.44%). A uniform bias would not do that
- by command family, it is carried by `vel_step` (+0.0150 over 10 episodes) and
  `pivot` (+0.0070); `yaw_step`, `arc` and `lateral` are at or below zero
- **the episode setup is bit-identical.** `cmd_vx` matches to every digit on all
  **43 of 43** episodes whose row count also matches. It differs only where the
  episode terminated at the bed boundary at a different step, which changes the
  scored window and therefore the mean command over it. The single largest
  contributor, `vel_step_011`, exited at 765 rows here against 935 on the fleet

Against hpcfund the picture is the same shape: +1.86%, r = 0.9820, flip rate
0.00%, sign split 46/31 (p 0.087).

**Read:** 0.1636 puts Euler at the **top of the fleet's own BASE band**
(kyle-sbel 0.16009, a3 0.16060, hpcfund 0.16065, north-windows 0.16322) --
essentially where north-windows sits, the box already known to run about 2%
high. Euler scores are usable for cross-arm comparison, and arms should be
compared against arms scored on the same machine rather than mixed across the
band. That is not a new rule; it is the rule the fleet already has, and Euler
does not widen the band.

## GPU architectures: the trap that costs you most of the cluster

`~/chrono-build` **cannot run on most of Euler's GPUs**, and nothing tells you
so until a kernel launch fails. Chrono's `cmake/ChronoGPUDetect.cmake` ends with

```cmake
set(CMAKE_CUDA_ARCHITECTURES ${CHRONO_CUDA_ARCHITECTURES}
    CACHE STRING "CUDA architectures" FORCE)
```

so `-DCMAKE_CUDA_ARCHITECTURES=...` is accepted and then silently overwritten,
and the fallback when `CHRONO_CUDA_ARCHITECTURES` is empty is
`all-major` = `60;70;80;90;100;120`. That covers the A100 and the H100 and
nothing else here: the RTX 2080 Ti (sm_75), RTX A4500 (sm_86) and RTX 4000 Ada
(sm_89) get no cubin, and the only PTX in the fat binary is `compute_120`, which
cannot JIT backwards. Those three are most of Euler's ~98 GPUs, and one of them
is euler16 -- half the `sbel` tier.

`build_parsers.sbatch` sets the variable Chrono actually reads:

```
-DCHRONO_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real;90"
```

Verified on both ends of that range: the same episode scores
`0.19542514273658756` on euler16's 2080 Ti and on euler19's A100, to every digit.

## Rendering (Chrono::Sensor on OptiX)

Unlike hpcfund, where Sensor runs on lavapipe and ray tracing is a CPU cost,
Euler renders on the GPU. `render_probe.sbatch` measures it:

```bash
CAMS=chase,side,front_3q,low_close,topdown,trail,fixed_wide,fixed_ground \
  sbatch --export=ALL -p sbel -w euler16 --gres=gpu:1 -c 8 render_probe.sbatch
```

Measured on euler16's RTX 2080 Ti, a 2 s clip with 886,611 SPH sprites:

| cameras | frames | total wall | of which shader compile |
|---|---|---|---|
| 1 | 48 | 65 s | 21 s |
| 8 | 384 | 74 s | ~21 s |

**Extra cameras are nearly free**: eight of them cost 9 s more than one. What
costs is the one-time OptiX shader compile and the CRM step itself, which the
simulation would pay anyway. Extrapolating the sim to an 11 s clip puts a full
eight-camera render at roughly **5 minutes** on a 2080 Ti.

Caveat before trusting that against the desktop 3090's ~17 min: this probe used
the script's defaults (960x540, `--render-ray-recursions 2`) and a 2 s clip, and
the desktop figure was not re-measured with the same command line. Treat it as
"comfortably faster, worth re-timing on a real clip," not as a 3x claim.

**Render on an RTX node, not an A100.** GA100 has no RT cores, so OptiX there
ray-traces on the SMs. Euler's RT hardware is euler16 (2080 Ti, Turing),
euler05-08 (A4500, Ampere) and euler01-04 / 09-10 (RTX 4000 Ada) -- all three of
which the default CUDA arch list excludes, so this only works against
`~/chrono-build-parsers`.

## Rebuilding the Chrono tree

```bash
sbatch build_parsers.sbatch     # ~18 min on euler16 at -c 32
```

This builds a SECOND tree, `~/chrono-build-parsers`, and leaves `~/chrono-build`
alone -- it fingerprints the original before and after and prints both so the
claim is checked rather than asserted. Scoring needs `pychrono.parsers` because
`src/nedm/quadruped/robot.py` loads the Go2 through `ChParserURDF` and has no
alternative path, and the original tree was configured with
`CH_ENABLE_MODULE_PARSERS=OFF`.

Three dependencies are built into `~/toolchain/urdf`: console_bridge 1.0.2,
urdfdom_headers 1.1.1, urdfdom **4.0.1**. Not four -- **Euler already ships
tinyxml2 10.0.0** with a CMake config in `/usr/lib64/cmake/tinyxml2`, which is
the exact version hpcfund had to build from source. urdfdom must be 4.x: the 3.x
series still parses with tinyxml v1, while `chrono_parsers/CMakeLists.txt` does
`find_package(tinyxml2 REQUIRED)`.

## What is staged, and why it is 200 MB and not 200 GB

**The scorer never opens an episode CSV.** It reads `csv_path[:-4] + ".json"`
and `csv_path[:-4] + ".config.json"` beside it and uses `csv_path` itself only
as a dictionary key. So `stage/episodes` is 3.1 MB of sidecars rather than the
~200 GB corpus.

The directory structure above them still has to be reproduced exactly, because
`episode_id` is built from the last four path components and an id that differs
by one component pairs with nothing -- silently, as a clean-looking file that
shares zero episodes with the reference. `scripts/cluster/restage_index.py` does
the rewrite and refuses to emit an index pointing at a sidecar that is not
there.

```
stage/episodes     3.1 MB   160 sidecars, 80 episodes
stage/assets       100 MB   Go2 URDF + meshes + the base RL model
stage/checkpoints   93 MB   52 policies
```

## Things that are Euler facts, not preferences

- **`-p` is mandatory.** The default partition has no nodes.
- **`TMPDIR=/tmp`**, which is node-local NVMe. The scorer makes and deletes a
  temp dir per episode and once leaked 16,352 of them / 67 GB; doing that on
  `/srv/home` would be doing it to the whole lab's unquota'd pool.
- **gcc 12.2.0 via module**, not the 14.3.1 default, which is too new for nvcc.
  Lmod warns it "was built for an older platform"; the binaries run.
- **numpy is the system 1.26.4** and the bindings are compiled against it. Do
  not pip a different one in. (hpcfund builds against 2.4.3. Neither moves.)
- **torch is pip-installed CPU-only** (2.6.0+cpu) because Euler has no system
  torch. That is not a compromise: the policy is a ~0.5M-parameter TorchScript
  at 50 Hz that `imported_policy.py` loads with `map_location="cpu"` on every
  machine.
