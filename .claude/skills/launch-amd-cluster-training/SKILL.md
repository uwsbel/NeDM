---
name: launch-amd-cluster-training
description: Launch NeDM training jobs on the AMD HPC Fund cluster (ssh amd) — env recipe, sbatch templates, data sync, monitoring, result sync-back. Use when asked to run/launch/scale a training job, GPU experiment, or ablation sweep. Default policy — validation/smoke checks run locally, real training runs on the cluster.
---

# Launch training on the AMD cluster

**Division of labor (user policy):** the local 5090 workstation is for quick
validation only (a few hundred steps to prove the config runs); every real
training job goes to the cluster. Don't queue multi-hour runs locally.

## Cluster facts

- `ssh amd` → `login1.hpcfund` (AMD HPC Fund, SLURM, Rocky 9). Login node has
  2× MI210 usable for quick interactive sanity checks.
- Work area: `$WORK` = `/work1/dannegrut/harry`. **`$WORK` is not set in
  non-interactive ssh/rsync** — always use the absolute path from the local side.
- Repo mirror: `$WORK/nedm/` (`src/`, `scripts/`, `assets/`,
  `artifacts/traverse/` with the WP1 stores, `slurm/`, `logs/`).
- Venv: `$WORK/venvs/nedm` (python3.12, `--system-site-packages`,
  `zstandard` pip-installed). Billing account `dannegrut`; check balance in
  the sbatch filter output on submit.

## The env recipe (every job script)

```bash
source /etc/profile
module load pytorch/2.10.0        # torch 2.10.0+rocm7.1 via PYTHONPATH
source /work1/dannegrut/harry/venvs/nedm/bin/activate
cd /work1/dannegrut/harry/nedm
PYTHONPATH=src:$PYTHONPATH python3.12 scripts/<train>.py ...
```

Three rules that were each learned from a real failure:

1. **`python3.12`, never bare `python3`** — system python3 is 3.9 and crashes
   importing the torch 2.10 build (`TypeError ... 'type' and 'NoneType'`).
2. **Append to PYTHONPATH, never assign** — torch lives on the module's
   PYTHONPATH (`/share/sw/ai/pytorch/2.10.0`); `PYTHONPATH=src` alone clobbers it.
3. **`zstandard` comes from the venv** — episode stores are zstd-compressed;
   without the venv active, DataLoader workers die with
   "episode is zstd-compressed but zstandard is not importable".

## Partition choice

| Partition | Node | Use for | Walltime |
|---|---|---|---|
| `mi3501x` | 1× MI350, 24 CPU | **default** for single-GPU training; usually has idle nodes | **4 h cap** (submit filter overrides sinfo's 4-day claim) |
| `mi3508x`/`mi2508x`/`mi3008x` | 8× GPU, whole-node | only when packing ~8 single-GPU variants on one node (`HIP_VISIBLE_DEVICES` per process); billed whole-node | 4-day sinfo, verify filter |
| `devel` | 1× MI210, 16 CPU | nothing — see MI210 gotcha | 30 min |

- **MI210 compute nodes are a trap**: pytorch/2.10.0+rocm7.1 dies on the first
  matmul (`HIP error: file not found`) even though all libs bundle gfx90a, and
  the rocm/6.3.1 module that pytorch/2.7.1 needs is missing from the compute-node
  module DB. Don't burn time there; use MI350.
- **The 4 h cap is the sizing constraint**: WP1 trainer does ~200 samples/s on
  MI350 with `--workers 20` (batch 48 → ~4.2 steps/s), so a 30k+8k-step run is
  ~3 h — fits. Anything projected >3.5 h: shorten, split, or add resume support
  first. Benchmark unknown configs with a ~600-step job before committing
  (sps in `train_log.jsonl` ramps for a few hundred steps; read the last line).

## Launch procedure

1. **Local validation** (only if code changed): ~100-step run on the
   workstation or a config parse; do not run full training locally.
2. **Sync code** (fast, do it every launch — cheap insurance):
   ```bash
   W=/work1/dannegrut/harry
   rsync -a src scripts assets amd:$W/nedm/
   ```
3. **Sync any new data roots / manifests** (`rsync -a artifacts/traverse/<store> amd:$W/nedm/artifacts/traverse/`;
   ~50 MB/s, so ~20 min per 60 GB). Existing stores are already there — check
   with `du -sh` before re-sending.
4. **Write the sbatch** into `$W/nedm/slurm/` (copy an existing one, e.g.
   `wp1_v7b.sbatch`): `-p mi3501x -N 1 -t 04:00:00`,
   `-o /work1/dannegrut/harry/nedm/logs/%x_%j.out`, then the env recipe block.
   Give each run a distinct `--out artifacts/traverse/<name>` — suffix `_amd`
   if the same experiment name could ever exist locally.
5. **Cluster smoke first for new code paths**: same sbatch with
   `--steps 100 --probe-steps 50 --val-batches 5` and `-t 00:25:00`, confirm
   exit 0, then submit the real job.
6. **Submit and verify it actually trains**: `sbatch`, then after ~2 min check
   the log shows step lines, not a traceback.
7. **Monitor**: arm a Monitor polling the job log for
   `exit: [0-9]|Traceback|AcceleratorError|TIME LIMIT|CANCELLED` every ~5 min.
   Silence is not success — match failure states too.
8. **Sync results back** when done:
   ```bash
   rsync -a amd:$W/nedm/artifacts/traverse/<out>/ artifacts/traverse/<out>/
   ```
   and record the run in the relevant notes doc.

## Debugging on the cluster

- Interactive shell on a compute node:
  `srun -p mi3501x -N1 -t 30 --pty bash -l` (the submit filter limits
  concurrent interactive jobs to 1).
- Quick GPU sanity: `$W/nedm/gpu_test.py` (matmul + conv + bmm/softmax).
- `sacct -j <id> --format=JobID,State,Elapsed,ExitCode` for post-mortems.
