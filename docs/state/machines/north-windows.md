# north-windows

**Verified:** 2026-09-10 · **Owner:** Kyle · **Role:** Fifth CRM scoring node.
WSL2 Ubuntu on a Windows host, cloned from `sliger-ubuntu`.

| | |
|---|---|
| GPU | NVIDIA RTX 5060 Ti, **8 GB** (Blackwell, sm_120), Windows driver 576.88 = CUDA 12.9 |
| CPU / RAM | Intel i5-14600KF, 20 threads / 32 GB host, **WSL sees 15.5 GB** |
| Disk (free) | 955 G free inside the distro (1.7 T free on `C:`) |
| Repo path | `/home/kyle/sbel/NeDM`, branch `kyle/locomotion` |
| Interpreter | `/home/kyle/miniconda3/envs/nedm/bin/python` (3.12.14, torch 2.12.0+cu130) |
| Chrono | `/home/kyle/chrono-build`, **bit-identical to `sliger-ubuntu`'s build** |
| Reachable from | the coordinator Mac as `kyle@north-windows` (Tailscale, key auth) |
| OS | WSL2 Ubuntu 26.04 LTS, glibc 2.43, hostname `Intel-PC` |

## What this machine is for

CRM (SPH deformable terrain) policy scoring, as a fifth box alongside
`sbel`, `north`, `a3` and `sliger`. It is a scoring node, not a dev box.

## Validation against the fleet

BASE (`go2_cts_150k.pt`) scored on all 80 episodes of
`go2_crm_merged/score_subset_index.json`, 2026-09-11, and paired per episode
against the three boxes that hold a BASE file.

Structural agreement is exact. All four boxes score **77 of 80**, fail on the
**same three** episodes (`weave_000`, `lateral_006`, `arc_002`), and disagree on
the `completed` flag for **zero** episodes.

| | mean `mae_vx` |
|---|---|
| `kyle-sbel` | 0.160094 |
| `kyle-B650M-D3HP` (a3) | 0.160604 |
| `kyle-N7-B650E` (north) | 0.163222 |
| **north-windows** | **0.164000** |

Per-episode paired differences, and the sign test on the direction:

| pair | n | B higher | mean shift | p |
|---|---|---|---|---|
| a3 -> **north-windows** | 77 | 48 (62%) | +0.0034 (+2.1%) | 0.040 |
| sbel -> **north-windows** | 77 | 49 (64%) | +0.0039 (+2.4%) | 0.022 |
| north -> **north-windows** | 77 | 41 (53%) | +0.0008 (+0.5%) | 0.65 |
| a3 -> north | 77 | 44 (57%) | +0.0026 (+1.6%) | 0.25 |
| sbel -> north | 77 | 46 (60%) | +0.0031 (+2.0%) | 0.11 |
| a3 -> sbel | 15 | 4 | -0.0005 | 0.12 |

**Report this honestly: there is a one-directional offset.** north-windows reads
~2% *high* against `a3` and `sbel`, on 62-64% of episodes, and it reaches nominal
significance. Two things keep it from being a verdict against the box. First,
`kyle-N7-B650E` — an accepted node — shows the *same direction* at nearly the
same size (1.6-2.0%) and only misses p<0.05; north-windows is statistically
indistinguishable from it (p=0.65). Second, `a3` and `sbel` are not independent
samples of "the fleet": they agree on **62 of 77 episodes bit-for-bit**, so they
are one build, not two. The fleet is really two clusters — {a3, sbel} and
{north, north-windows} — and the offset tracks the *build*, not the box.

The spread is small against what the project actually measures. On a3, real
policy effects run -0.041 (`w_r0`) to +0.58 (`fix_s2`) — 10x to 150x this offset.
The exception that matters: `RAND11` differs from BASE by **-0.0030**, which is
the same size as the box offset. **Any comparison at the 0.003 level must be
scored on one box.** Large effects cross boxes safely; null-sized ones do not.

**The missing control.** north-windows runs `sliger`'s Chrono binary
(md5-identical), but `sliger` has no BASE file, so the one comparison that would
settle this — same build, different GPU — has never been run. If that offset
ever needs a verdict, score BASE on `sliger`: agreement there attributes the ~2%
to sliger's build (benign, and shared by this node by construction), while
disagreement points at the PTX JIT of gotcha 2, the only thing here that is not
byte-for-byte sliger.

## What it must not do

1. **Do not run at `--concurrency 8`.** That is the scorer default and it leaves
   only ~700 MiB of VRAM spare here — see gotcha 3. Pass `--concurrency 6`.
2. **Do not transplant binaries from `a3`, `sbel` or `north`.** All three are
   AMD Zen 4/5 with AVX-512; this is a Raptor Lake host with none. See gotcha 1.
3. **Do not edit files here by hand.** Same rule as everywhere: commit and push.

## Launch recipe

```bash
source /root/nedm_env.sh     # sets the four variables below
```

which is:

```bash
export NEDM_ROOT=/home/kyle/sbel/NeDM
export NEDM_PY=/home/kyle/miniconda3/envs/nedm/bin/python
export NEDM_ANALYSIS_PY=/home/kyle/miniconda3/envs/nedm/bin/python
export PYTHONPATH=/home/kyle/chrono-build/bin:$NEDM_ROOT/src
export NEDM_GO2_ASSETS=/home/kyle/Documents/sbel-reproducibility/2025/multi-terrain-RL
```

`PYTHONPATH` must start with `chrono-build/bin`: the conda env also carries an
unrelated `pychrono` in `site-packages`, and the scorer reads `PYTHONPATH[0]`
as the Chrono root.

Scoring a policy:

```bash
source /root/nedm_env.sh && cd "$NEDM_ROOT"
"$NEDM_PY" scripts/evaluation/score_crm_tracking.py \
  --policy /home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt \
  --index  /home/kyle/sbel-artifacts/datasets/go2_crm_merged/score_subset_index.json \
  --out    /home/kyle/sbel-artifacts/crmtrack_<tag>_north-windows.json \
  --concurrency 6
```

Reaching the box at all — inline quoting through `ssh -> cmd.exe -> wsl` gets
mangled, so ship a script and run it:

```bash
scp job.sh kyle@north-windows:C:/Users/kyle/job.sh
ssh kyle@north-windows "wsl -d Ubuntu -u root bash /mnt/c/Users/kyle/job.sh"
```

Long jobs go in a `tmux` session **and** need a keepalive holding a `wsl.exe`
client open for the whole run — see gotcha 0, which is the one that will cost
you a night's work if you skip it.

## Gotchas

0. **WSL terminates the distro ~40 s after the last `wsl.exe` client detaches,
   and it does not care that you left work running.** A `tmux` session running
   `sleep 900` was killed 42 seconds after the launching command returned;
   `systemd` restarts and every process dies. This silently destroyed three
   concurrency probes before it was identified — the symptom is a job that
   vanishes with a truncated log and no error, which reads exactly like a crash.
   Diagnose it by `systemctl show -p ActiveEnterTimestamp --value systemd-journald`:
   if that timestamp moved, the distro restarted. Note `/proc/uptime` does *not*
   move, because the shared WSL utility VM keeps running — only the distro's
   init is replaced, so `uptime` will lie to you here.

   `vmIdleTimeout=-1` in `C:\Users\kyle\.wslconfig` does **not** fix it (it
   governs the VM, which was never the thing shutting down). What works is
   keeping a client attached for the duration of the job:

   ```bash
   # on the coordinator, for the life of the run
   while true; do
     ssh kyle@north-windows "wsl -d Ubuntu -u root sleep 300"; sleep 2
   done &
   ```

1. **AVX-512 is the reason the donor is `sliger` and not `a3`.** The obvious
   plan — clone `a3`, which has the identical RTX 5060 Ti — fails immediately:
   `import pychrono` dies with SIGILL. `a3` is a Ryzen 9600X (Zen 5) and its
   `libChrono_core.so` carries 54,048 AVX-512 instructions; the i5-14600KF has
   no AVX-512 at all. `sliger` is a Ryzen 5900X (Zen 3, no AVX-512), so its
   build is the only one on the fleet that runs here. The libraries here are
   md5-identical to `sliger`'s.

2. **The GPU-side code is JIT'd, not native.** `sliger` built for
   `CUDA_ARCHITECTURES=75` (its RTX 2060), so the shipped cubins are sm_75 and
   this Blackwell card has no matching binary. It works because CMake also
   embedded `compute_75` PTX, which the driver JITs to sm_120 on first use.
   This is the one place where north-windows is not running the same machine
   code as `sliger`, and therefore the first place to look if its physics ever
   drifts. First run of a process pays the JIT; it is cached in `~/.nv`.

3. **8 GB of VRAM is shared with the Windows desktop, so the fleet's default
   `--concurrency 8` is too tight here.** Measured, on an idle desktop:

   | | VRAM |
   |---|---|
   | desktop idle, no episodes | 731 MiB |
   | 6 episodes, steady state | 4778 MiB |
   | **6 episodes, worst peak over a full 2h40m pass** | **7215 MiB of 8151 (88%)** |
   | 8 episodes, steady state | 5506 MiB |
   | 8 episodes, startup transient | **7452 MiB of 8151 (91%)** |

   Peaks are short transients while processes build terrain — construction costs
   ~840 MiB per episode against ~600 MiB for a running sim — so the peak depends
   on how many happen to overlap, not on the steady footprint. Note the 6-way
   peak of 7215 MiB: measured over a full pass it came within 940 MiB of the
   card, so the margin at 8 is thinner than its startup number suggests. **Use
   6**, and do not assume the steady-state figure is the one that matters.

   That safety is not free, and the measured cost is bigger than the ~91% GPU
   utilisation figure suggests:

   | concurrency | aggregate | full 80-episode BASE pass |
   |---|---|---|
   | 1 | ~5.0 rows/s | — |
   | 6 | 6.2 rows/s | **2 h 40 m** (measured, full 80-episode pass) |
   | 8 | 9.3 rows/s | ~1 h 55 m (extrapolated) |

   So 8 is roughly 50% faster than 6, not the few percent you would guess from
   the GPU sitting at 91% either way. The trade is throughput against the risk
   of losing a three-hour run to a transient VRAM spike. 6 is the right default
   for unattended work; 8 is defensible when someone is watching the box and
   nothing else is using the GPU. Either way this node is slower than `a3`,
   which does the same pass in ~90 minutes.

   Do **not** conclude from a vanished run that you ran out of VRAM. Every
   apparent OOM seen while provisioning this box was actually gotcha 0.

4. **CUDA is detected through `/dev/dxg`, not `/dev/nvidiactl`.** WSL2 never
   creates `/dev/nvidiactl`, so the collector's fail-fast CUDA guard rejected
   this box until it was taught about `/dev/dxg`.

5. **torch reports `cuda_available False` and that is fine.** torch here is
   `2.12.0+cu130`, which wants driver >= 580; the Windows driver is 576.88
   (CUDA 12.9). It does not matter: `nedm.quadruped.policy` runs the policy on
   CPU (`a.device = "cpu"`, `map_location="cpu"`). Only Chrono uses the GPU, and
   Chrono here links CUDA 12.8, which 576.88 supports. The warning printed on
   every episode is noise. Updating the Windows driver to 580+ would silence it
   and is the prerequisite if anything ever needs torch on the GPU here.

6. **RAM is not the constraint; WSL sees only half the host.** 15.5 GB of the
   host's 32 GB, and peak observed during scoring was ~5.8 GB — so the memory
   default was left alone. If it ever needs raising, add `memory=24GB` under the
   `[wsl2]` section of `C:\Users\kyle\.wslconfig` and run `wsl --shutdown`.

   That file exists and was created during this provisioning. It holds only
   `vmIdleTimeout=-1`, which — see gotcha 0 — **did not work**. It is inert and
   can be deleted; it is left in place solely so the next person does not try the
   same setting expecting a different result.