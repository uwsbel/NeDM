---
name: storage
description: Storage and compute division of labor for this 1TB / RTX 5090 workstation. Use before planning data collection, launching anything that writes large artifacts, transferring datasets between machines, or deciding where a job should run.
---

# Storage policy: this workstation trains, other machines collect

## Machines and roles

| Machine | GPU | Disk | Role |
|---|---|---|---|
| **This desktop** | RTX 5090, 32 GB | 1 TB total (~700 G free, 2026-09-01 after cleanup) | Training, eval, figures. **Never raw data collection.** |
| **newton** (`ssh newton`, repo at `~/NeDM`) | RTX 4090, 24 GB | 1.8 TB (~630 G free; `artifacts/` already 542 G) | Chrono data collection, including Chrono::Sensor RGB-D rendering. Home of the raw frame stores. |
| **Euler cluster** | CPU array | cluster storage | State-only CPU-scale collection via SLURM arrays — see the `create-euler-script` skill. |

## Rules

1. **No dataset collection on this machine.** Chrono episode collectors, RGB-D
   rendering runs, and anything that writes raw per-frame data run on newton
   (needs GPU rendering) or Euler (state-only). If asked to "collect data",
   set the job up for one of those machines, not here.
2. **Only trainable assets come here:** compressed/processed training datasets
   (the episode-chunked stores the loaders actually read), layout/camera
   manifests, model checkpoints, small eval summaries. Raw frame dumps,
   per-frame PNG dirs, uncompressed memmaps, and video archives stay on newton.
3. **Check sizes before transferring.** `ssh newton "du -sh ~/NeDM/artifacts/<name>"`
   and `df -h /` locally first. With ~700 G free (2026-09-01), local
   `artifacts/` can hold up to ~400 G of training-ready stores; keep ≥250 G
   headroom for checkpoints, caches, and system growth.
4. **Transfer pattern:**
   ```bash
   rsync -avP newton:NeDM/artifacts/<name> /home/harry/NeDM/artifacts/
   ```
   Checkpoints/results produced here that newton needs go back the same way.
5. **Study 3 (traverse) budget context:** per plan §6.1, raw RGB-D frames are
   ~25–49 GiB at pilot tier and **122–488 GiB at full tier**. Full-tier raw
   must never land on this disk — only the compressed, training-ready store.
   The compressed store IS the trainable asset; pulling it here for 5090
   training is the intended workflow.

## Running jobs on newton

- Python with pychrono + torch: `~/anaconda3/envs/nedm/bin/python` (conda env
  `nedm`); repo scripts need `PYTHONPATH=src` from `~/NeDM`. 32 CPU cores;
  headless Chrono HMMWV episodes run ~15–25× slower than realtime, so batch
  with ~12 workers and expect ~4 min per 10 s episode.
- Detached launch (plain `ssh newton 'nohup ... &'` hangs the session):
  ```bash
  ssh newton 'cd ~/NeDM && nohup env PYTHONPATH=src ~/anaconda3/envs/nedm/bin/python -u <script> > /tmp/<name>.log 2>&1 < /dev/null & echo launched'
  ```
- Deliver code by commit + push from here, `git pull --ff-only` on newton —
  never edit files on newton directly.
