# Actual demo rendering throughput audit

The running demos are **not limited to four Lavapipe threads**. Their process environment leaves `LP_NUM_THREADS` unset; each job has 16 available vCPUs. The failure process has 16 named LLVMpipe raster workers observed idle and 16 unnamed active workers consistent with the compute-render pool. A two-second sample used 9.67 CPU cores, with no measured steal time or I/O wait. `OMP_NUM_THREADS=4`, `OPENBLAS_NUM_THREADS=1` and `MKL_NUM_THREADS=4` remain independent physics/model settings. Slurm's `sstat` CPU field overflowed and was not used.

The installed `nrd_use_lavapipe` function only sets Vulkan library/ICD paths. Mesa documents the default `LP_NUM_THREADS` as the number of CPU cores: [official environment documentation](https://docs.mesa3d.org/envvars.html#envvar-LP_NUM_THREADS). Setting it to 16 on these jobs would merely make the existing behavior explicit. Setting it to 32 on a 16-vCPU node could oversubscribe the allocation; it is not a supported speed-up claim.

| Replay | Job | Saved frames at sample | Expected frames | Mean of last 20 frame intervals |
|---|---:|---:|---:|---:|
| Cross slopes, time | 412134 | 125 | 208 | 2.811 s |
| Cross slopes, energy | 412133 | 116 | 207 | 3.026 s |
| Rolling-hill failure | 412135 | 116 | 901 | 2.932 s |

At that sample, the failure render projected about 38.4 minutes remaining and 46–47 minutes total, including initialization, if cadence persisted. This is an estimate rather than a guarantee. Measurements above are snapshots, not current job status.

A controlled future graphics benchmark would use a **32-CPU allocation on the same node**, compare `LP_NUM_THREADS=16` and `32`, and retain `OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4`. Both arms would execute the exact same seven-second prefix at 5 Hz into new outputs. Only measured wall time and render settings would differ; all physical arrays/work/risk telemetry would require strict prefix parity. No benchmark was launched and no speedup is claimed.

The parent chose a media-only alternative: keep the two success videos at 5 Hz and have the rich-telemetry agent launch a separate full 180-second failure replay at 1 Hz. Inspection of `ChVulkanRTEngine::UpdateSensors` confirms one render per due sensor per manager update, without a catch-up loop for all missed high-rate sensor periods. Therefore fewer capture updates reduce render calls. The failure video must retain actual timestamps and full physical telemetry, and pass complete replay parity. This audit did not change, cancel or launch any jobs.
