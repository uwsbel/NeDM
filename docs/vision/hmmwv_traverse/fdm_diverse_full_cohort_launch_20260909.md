# Full diverse cohort launch plan — 2026-09-09

Use **one 128-CPU `mi2104x` node**, with 48 headless physics workers and eight
observation workers in sequential stages. Request 90 minutes. The measured
pilot gives an ideal estimate of about 42 minutes; that is a projection, not a
measured full-cohort runtime. Launch only after the root's `campaign_v2` rich
getter and renderer/observer parity checks pass. This audit did not submit a job.

The full manifest contains 24 training arenas and six validation arenas:
**30 one-time RGB-D captures and 450 fixed-reference episodes**, each capped at
180 simulated seconds. All 15 declared references per arena are retained,
including collisions, stall, rollover, timeout and successful outcomes. The six
test arenas and their 90 references are excluded. No pilot outcome determines
which train/validation episode gets collected.

## Measured pilot and capacity

AMD job `412053` completed all four observations and 12 episodes without a
failure. Slurm reports 232 seconds; the batch ledger reports **231.345 seconds**.
Two observation workers averaged **34.137 seconds per map**. Eight headless
workers averaged **79.891 seconds per episode**, mostly at the 60-second physics
cap. The 12 episodes total 701.35 simulated seconds and 36.87 MB of saved raw
artifacts. The four cached observations occupy 13.34 MB.

The larger plan allows at most **22.5 simulated vehicle-hours**. Scaling the
pilot's per-episode cost by three gives ten waves of 48 physics workers at
roughly four minutes per wave, plus four short waves of map rendering. Changes
to v2 getter coverage, CPU/cache pressure, filesystem contention or longer-lived
telemetry buffers can increase this cost; the 90-minute request leaves room for
that uncertainty. It is a limit, not an estimate of charged runtime.

At the resource check, four `mi2104x` nodes were idle. Example node `k005-007`
reported 128 CPUs and 493,202 MiB free memory, roughly 482 GiB. Pilot Slurm
accounting did not report `MaxRSS`, so per-worker memory has **not** been measured.
A provisional four-GiB budget per physics worker would use 192 GiB for 48
workers; verify actual usage during collection rather than treating this budget
as a measurement. Rendering has only eight simultaneous sensor managers and
does not overlap the physics stage.

Extrapolating recorded bytes per simulated second gives about **4.3 GB raw
episodes plus 100 MB cached observations**, assuming every route runs the full
180 seconds. Reserve approximately 20 GB for raw data, packs and diagnostics;
training checkpoints are additional. `/work1` reported about 1.5 TB available.

The partition uses exclusive node allocation and advertises
`TRESBillingWeights=NODE=0.04`. Two nodes with 24 workers each have the same total
of 48 physics workers and similar ideal runtime, but approximately twice the
allocated node-time cost. The scheduler does not provide a currency price here.
Use two nodes only if actual memory or contention makes the single-node option
inefficient. Do not request 48 workers on *each* of two nodes while describing
that as a matched resource comparison.

## Exact command, after v2 validation

Run from an AMD login shell. The log directory already exists from the pilot.
The command deliberately clears pilot route/scene filters and disables parity
rendering for the main headless cohort:

```bash
env CODE_ROOT=/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/snapshots/campaign_v2 \
    MANIFEST=/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/snapshots/campaign_v2/artifacts/traverse/fdm_diverse_v1_20260909/cases/cases.json \
    OUT_ROOT=/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/cohort_v2_w48 \
    STAGE=all SPLITS="train val" HORIZON_S=180 \
    WORKERS=48 OBSERVATION_WORKERS=8 LP_NUM_THREADS=4 \
    ROUTE_INDICES= SCENE_IDS= RENDER_PARITY=0 RICH_TELEMETRY=1 EVALUATION_ONLY=0 \
    sbatch --export=ALL --time=01:30:00 \
    /work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909/snapshots/campaign_v2/slurm/traverse_fdm_rgbd_diverse_collect.sbatch
```

Do not edit `campaign_v2` after launch. Its source/runtime fingerprints must
remain identical for observations and physics. The output root is new; the
pilot's v1 records are retained as diagnostic evidence rather than blended into
this cohort. If execution is interrupted, rerun the same command: the batch
reuses only completed outputs with matching input and output hashes. Failed or
interrupted attempt directories remain available for inspection. A stale task
lock requires checking that its process/job is gone before removing that lock;
completed records are never overwritten.

After collection, check each observation's actual Chrono height/orientation and
camera geometry, prepare the matched train/validation packs, then launch the
four-arm AMD training smoke before full training in a separate output root.
Neither full training nor held-out MPPI results are established by this resource
plan.

The timestamped machine-readable plan is
[`full_cohort_launch_plan.json`](../../../artifacts/traverse/fdm_diverse_v1_20260909/full_cohort_launch_plan.json).
It records pilot measurements, explicit projections, source/manifest hashes and
pending collection/training/evaluation gates; it is a launch-time snapshot, not
a live status file.
