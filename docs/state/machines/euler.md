# Euler cluster (UW-Madison)

**Verified:** 2026-09-11, live, by running the CRM scorer on it end to end ·
**Account:** `kasha2` · **Host:** `euler.engr.wisc.edu`

> This file used to live in [`reference/`](reference/) and open with "NO ACCESS."
> That is no longer true: Kyle has an account, and the CRM policy-scoring
> pipeline runs here and has been validated against the desktop fleet
> per-episode. It also used to say "CPU array" and "no GPU renderer," which was
> wrong in the other direction -- Euler has 98 GPUs and Chrono::Sensor is built
> here on **OptiX**, against real RT hardware.

## Access: Kyle authenticates, Claude reuses the socket

Login is **passcode + Duo push**, which an agent cannot complete. The working
pattern is that Kyle opens a master once per session in a real terminal:

```bash
ssh -fN euler          # Kyle: passcode + Duo, backgrounds, good for 8h
ssh -O check euler     # anyone: is the master alive?
```

and every subsequent call carries `-o BatchMode=yes`, so a dead socket fails
fast instead of hanging on an invisible password prompt. A key does not bypass
Duo. Repeated attempts after a refusal are push fatigue, not a workaround.

Because the socket can die on a network change or suspend, **anything long runs
under `sbatch` and is read back later**, never held open on an interactive
connection.

## What makes this machine different from hpcfund

| | **hpcfund** (AMD) | **Euler** (NVIDIA) |
|---|---|---|
| GPU stack | ROCm / HIP | CUDA |
| Chrono::Sensor | Vulkan RT, but on **CPU** (lavapipe; CDNA has no RT hardware) | **OptiX, on the GPU** |
| Node allocation | exclusive, billed whole | **shared**, `OverSubscribe=YES:4`, `ExclusiveUser=NO` |
| Budget | 1500 shared node-hours, metered | **none, and no accounting, QOS, or fairshare** |
| Storage quota | 2 TB shared `$WORK` | **none**, on a 941 TB pool shared with the whole lab |

The missing meter is the hazard. On hpcfund a runaway sweep is self-limiting
because it burns a visible budget. Here nothing stops it, and a runaway writer
fills shared SBEL storage for everyone. **Restraint is the only limit**, so
request the cores actually used, throttle job arrays, and check `squeue` before
launching something large.

Treat Euler as **unbacked** until snapshots are confirmed. Do not let it hold
the only copy of anything.

## Partitions

**The default partition is `none` and contains no nodes. `sbatch` without `-p`
goes nowhere.** Always pass it.

| Partition | Nodes | Priority tier | Notes |
|---|---|---|---|
| `sbel` | euler16, euler19 | **400** | Kyle is in `euler-sbel`. Not preemptible in practice. |
| `research` | all | 200 | Preempted by `sbel`, **with requeue**. |
| `interactive` | all | 100 | Preempted first. |

Preemption is by tier with requeue, so anything long outside `sbel` must
checkpoint or it silently restarts from zero.

## GPUs, and which ones Chrono can actually use

Surveyed 2026-09-11 from `sinfo`:

| Nodes | GPU | SM | Count | Partitions |
|---|---|---|---|---|
| euler01-04, 09-10 | RTX 4000 Ada | sm_89 | 8 each | research, interactive |
| euler05-08 | RTX A4500 | sm_86 | 8 each | research, interactive |
| euler16 | RTX 2080 Ti | sm_75 | 2 | **sbel**, research, interactive |
| euler17, euler19 | A100-SXM4-40GB | sm_80 | 4 each | euler19 in **sbel** |
| euler29-30 | H100 | sm_90 | 4 each | research, interactive |

Useful `--constraint` features: `turing`, `ampere`, `ada`, `hopper`,
`quadro_gpu`, `ent_gpu`.

### The architecture trap: `CHRONO_CUDA_ARCHITECTURES`, not `CMAKE_…`

`cmake/ChronoGPUDetect.cmake` ends with

```cmake
set(CMAKE_CUDA_ARCHITECTURES ${CHRONO_CUDA_ARCHITECTURES}
    CACHE STRING "CUDA architectures" FORCE)
```

so **`-DCMAKE_CUDA_ARCHITECTURES=...` is accepted and then silently
overwritten.** When `CHRONO_CUDA_ARCHITECTURES` is empty Chrono falls back to
`CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR`, which under CUDA 12.9 is
`60;70;80;90;100;120` -- major architectures only.

On Euler that default is quietly wrong. It covers the A100 and the H100 and
nothing else the cluster owns: sm_75, sm_86 and sm_89 get no cubin, and the only
PTX in the fat binary is `compute_120`, which cannot JIT backwards. That is
**16 of Euler's ~98 GPUs**, and it excludes euler16, which is half the `sbel`
tier.

`~/chrono-build` has exactly this problem. Its configure script passes
`-DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90"` and its `CMakeCache.txt` reads
`60-real;70-real;80-real;90-real;100-real;120`. It runs on the A100 and H100
only. `~/chrono-build-parsers` sets the variable Chrono actually reads and
covers all five.

## The two Chrono trees

```
source  ~/chrono-src               pinned 6982828 + patches 0001, 0002 (detached)
venv    ~/venvs/nedm               python 3.12, --system-site-packages
optix   ~/optix                    OptiX 9.0.0
env     ~/env.sh                   modules gcc/12.2.0 + nvidia/cuda/12.9.1
```

| | `~/chrono-build` | `~/chrono-build-parsers` |
|---|---|---|
| Parsers (URDF) | **OFF** | **ON** |
| CUDA archs | `all-major` (A100, H100 only) | `75-real;80-real;86-real;89-real;90` |
| Sensor | OptiX | OptiX |
| Role | the original, left untouched | **what scoring and collection use** |

Scoring needs `pychrono.parsers`: `src/nedm/quadruped/robot.py` loads the Go2
through `ChParserURDF` and has no alternative path. The second tree exists
rather than a reconfigure of the first because the first is the only tree whose
OptiX sensor backend is known to work, and a configure that goes wrong costs the
whole artifact. The build job fingerprints the original before and after to
prove it was not touched.

Use **gcc 12.2.0 via module**, not Euler's 14.3.1 default, which is too new for
nvcc. Lmod warns that gcc 12.2.0 "was built for an older platform"; it is a
warning, and the resulting binaries run.

### URDF dependencies

`Chrono_parsers` needs `urdfdom`, `urdfdom_headers`, `console_bridge` and
`tinyxml2`, and Chrono vendors none of them. **Euler already ships tinyxml2
10.0.0** with a CMake config in `/usr/lib64/cmake/tinyxml2` -- the same version
hpcfund had to build from source -- so only three are built here, into
`~/toolchain/urdf` (they install to `lib64`, not `lib`).

`urdfdom` must be **4.x**. The 3.x series still parses with tinyxml v1, while
`chrono_parsers/CMakeLists.txt` does `find_package(tinyxml2 REQUIRED)`.
Versions used: console_bridge 1.0.2, urdfdom_headers 1.1.1, urdfdom 4.0.1.

## Python stack

numpy is the **system 1.26.4** (`/usr/lib64/python3.12/site-packages`), and the
pychrono bindings are compiled against exactly that. Do not pip a different one
into the venv. Note the AMD cluster builds against numpy 2.4.3 -- that is a
per-machine fact, not a discrepancy to reconcile, and neither should be moved.

Euler has **no system torch**. `torch 2.6.0+cpu` is pip-installed into
`~/venvs/nedm`; it pulls no numpy, so the ABI above is preserved. CPU is not a
compromise -- the policy is a ~0.5M-parameter TorchScript at 50 Hz and
`imported_policy.py` loads it with `map_location="cpu"` on every machine. The
GPU is Chrono's SPH alone.

## Running the CRM scorer here

See [`scripts/cluster/euler/README.md`](../../../scripts/cluster/euler/README.md)
for the commands, the staging layout and the measured numbers.

Validated 2026-09-11 on the base policy, 80 val episodes, ids paired one-to-one
with the fleet: mean `mae_vx` **0.16363** here against **0.16009** on
`kyle-sbel`, r = 0.9832, episode flip rate **0.00%** against the fleet's own
0.93%, and no one-directional bias (higher on 42, lower on 35, sign test p
0.425). The +2.2% offset is a tail rather than a shift -- dropping the five
largest per-episode deltas leaves +0.41% -- and it puts Euler at the top of the
fleet's own BASE band, next to north-windows. One policy takes **57 min** on
three A100s.

### The knob is GPUs, not threads

A single CRM episode saturates one GPU. Measured on euler19, same episodes in
the same order: `--concurrency 6` on one A100 did 10 episodes in 44 min 39 s,
while `--concurrency 1` on two A100s did 8 in 8 min 06 s. Stacking episodes onto
one GPU time-slices work that was already serial and costs about half the
per-GPU throughput. Use `score_policy_pergpu.sbatch`, which pins one episode per
GPU with `CUDA_VISIBLE_DEVICES`.

## Never do these

- `ssh`/`rsync`/`scp` without `-o BatchMode=yes`.
- Drive the Duo prompt from inside an agent, or retry after a refusal.
- `sbatch`/`srun` without `-p`.
- Compute on the login node.
- Writes to `/srv/home/groups/sbel` outside Kyle's own subdirectory. It is shared
  with the whole lab and has no quota.
- Per-episode scratch on CephFS. Use `/tmp`, which is node-local NVMe.
- Treat Euler as the only copy of a result.
