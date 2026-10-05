# Ball NRD data and launch-optimization cost

The current19-model campaign reused5,400 training and900 validation episodes,
and collected900 new test episodes. Its7,200 used episodes contain4.05949 hours
of retained trajectory motion (average2.02975 s/episode). The simulator actually
integrated4.10940 hours for these episodes, including the short lookahead used
to detect the second ground impact and truncate training trajectories.

| Split | Episodes | Retained simulation hours |
|---|---:|---:|
| Training |5,400|3.04458|
| Validation |900|0.50746|
| Fresh test |900|0.50746|
| Total used |7,200|4.05949|

Collection wall time is much lower because episodes run in parallel. The
original7,200-episode bulk collection, job445723, took245 s (4 min5 s,
0.06806 wall hours). This original bulk included900 older test episodes; the
current comparison keeps its6,300 train/validation episodes and replaces test.
The new900-episode test collection and merge, job447352, took84 s (1 min24 s,
0.02333 wall hours). Thus the two collection jobs consumed329 s of wall time
in total, while collecting8,100 episodes including the older test cohort.
These are Slurm execution times including collection/validation/packing,
excluding queue waiting; they are not sums of individual worker runtimes.

The19 training jobs ran in parallel over a first-start-to-last-finish span of
0.66167 wall hours (39 min42 s). Their allocated GPU-job durations sum to
1.81722 hours. This excludes collection, smoke, certification, plotting,
optimization, code preparation and idle research time. It is not a measure
of continuous GPU utilization. All fitting used the same training split;
retraining19 variants does not multiply the unique dataset size.

## Differentiable five-target experiment

Both new designs solved the exact same five reachable validation benchmarks
at1.7 s shown in the earlier precision comparison. The optimizer changes
`[vx0,vz0]` using PyTorch autograd through the frozen NRD rollout, minimizing
`L = (x_NRD(1.7)-target_x)^2 + (z_NRD(1.7)-target_z)^2`.
Every optimization and backtracking rollout uses NRD. Chrono supplies only
independent physical replay afterward, with no optimizer feedback.

| Model | Global batched rounds | Five-target optimization wall time | Chrono target misses |
|---|---:|---:|---:|
| Scalar contact MLP/shared bounce |45|25.0317 s|0.262–1.792 mm|
| Transformer/two bounce NNs |45|46.1396 s|0.208–0.662 mm|
| Earlier analytical precision reference |45|5.9418 s|0.101–1.166 mm|

All five final launches pass the required two-bounce physical check for each
model. The45 rounds count a batch timeline, including held converged cases,
not45 accepted action changes for every individual target. The reported
solve timers exclude model/data loading, subsequent gradient checks, physical
replay, plotting and queue waiting.

## Measured cost of one gradient-descent update

AMD compute job447491 ran a fresh synchronized timing benchmark on an
AMD Instinct MI350X, PyTorch2.10.0+ROCm7.1, float64. Each operation used
3 warmups and10 measured repetitions, at the common launch `[5.5,-9.75]`.
The table reports medians. Host timers synchronize the GPU before and after
each call, so they include Python/kernel-launch overhead.

| Model | Rollout steps to1.7 s | One-target forward+backward gradient | One-target complete update | Five-target complete update |
|---|---:|---:|---:|---:|
| Scalar contact MLP/shared bounce |170 at10 ms|331.4 ms|496.8 ms|505.7 ms|
| Transformer/two bounce NNs |170 at10 ms|610.0 ms|897.0 ms|921.8 ms|
| Earlier analytical precision reference |34 at50 ms|77.9 ms|105.5 ms|106.9 ms|

A complete update includes gradient computation, velocity bounds, the0.25 m/s
step cap, evaluation of16 parallel backtracking candidates and selection of an
acceptable action update. This timing does not include Chrono, I/O, model
loading or convergence of an entire target solve. The single-target result
was measured directly; it is not the five-target time divided by five.
The older reference has a different architecture and a coarser native step,
so its speed difference does not isolate the network architecture alone.
Hard contact gates give piecewise gradients; physical success was verified
for the five recorded benchmarks, not every possible target.

Evidence:

- `artifacts/analysis/bouncing_ball_timing/benchmark.json`: all benchmark
  repetitions' summary statistics, hardware/software, checkpoint paths and
  operation definitions.
- `artifacts/analysis/bouncing_ball_timing/data_and_run_costs.json`: split
  sums, observed optimization timers and collection/training wall-time scope.
- `artifacts/analysis/bouncing_ball_timing/training_slurm_records.txt`: all19
  completed training jobs, with starts, ends and elapsed seconds.
- `artifacts/analysis/bouncing_ball_timing/timing.out`: AMD benchmark output.
- Existing frozen targeting records are under the final Transformer review
  bundle's `targeting/` directories.

No new collection or training was performed for this audit. Only the latency
benchmark ran, entirely on AMD compute. The active traversal checkout was
preserved.
