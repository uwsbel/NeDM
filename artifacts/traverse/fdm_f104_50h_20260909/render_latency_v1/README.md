# Why the overhead depth frame takes seconds on AMD (render_latency_v1, 2026-09-16)

Scene: exactly the navigation runner's frame — 80 m arena heightmap + HMMWV, one overhead depth camera 110 m up,
47 deg FOV, 180 m range. Harness: `scripts/render_latency_bench.py` (same script on every backend). Raw results:
`amd/`, `optix_local/`, `vulkan_local/`; code reading: `source_analysis/`; every agent result and all 38 skeptic
verdicts: `workflow_result.json`.

## 1024x1024 depth frame, steady state

| where | what renders | per frame |
|---|---|---|
| AMD mi2101x, 16 CPUs | Chrono Vulkan-RT on lavapipe (Mesa 24.2.8, CPU software) | **2.70-2.76 s** |
| AMD mi2104x, 32+ threads | same (Mesa caps its workers at 32) | 1.43 s |
| local Ryzen 5800X, AMD's own lavapipe copied over | same | 2.45 s (depth bit-identical to AMD) |
| local, system Mesa 25.2.8 lavapipe | same renderer, newer Mesa | 1.12-1.21 s |
| local RTX 5090, NVIDIA Vulkan driver | same Chrono Vulkan-RT renderer on the GPU | 0.69-0.81 s at every size 256-2048 |
| local RTX 5090, **OptiX** (fork with the FOV fix #819) | OptiX renderer | **6.9 ms** (144 fps over 300 frames; p99 8.3 ms) |

## Where AMD's 2.7 s goes
1. **~1.9 s — ray tracing on the CPU.** lavapipe spends ~27 us of CPU per pixel: 1 M pixels = ~28 CPU-s, split over
   16 threads. Scales exactly with pixel count (0.12 / 0.49 / 1.9 / 7.7 s at 256 / 512 / 1024 / 2048) and almost
   perfectly with threads, up to Mesa 24.2.8's cap of 32 workers.
2. **~0.8 s — Chrono rebuilds the entire scene before every frame, on one thread.** The Vulkan scene fingerprints
   every drawn body's pose and velocity to 1e-8; any change (a parked vehicle after one physics step is enough)
   re-copies all ~797k triangles (522k terrain that never moves + 274k HMMWV) into staging records, flattens and
   re-uploads them and rebuilds the ray-tracing acceleration structure from scratch — no refit, no split between
   static and moving geometry (`vulkan/ChVulkanRTScene.cpp` 481-801, `vulkan/ChFilterVulkanRTRender.cpp`
   2029-3061). Parked and driving frames cost the same; only a frame in which nothing drawn moved skips it.
   OptiX, by contrast, builds each mesh's structure once and per frame only updates ~10 instance transforms.
- Copying the frame into Python is negligible on this path (<1 ms depth-only).
- The colour camera adds ~3.9 s per frame on AMD (its tracing costs twice the depth camera's).
- First frame of each scene: +1-2 s with a warm Mesa shader cache; 63 s pipeline compile with a cold one.

## Correctness of the fixed OptiX depth
Against the AMD frame of the same case, on 721,194 terrain pixels (6 m around the vehicle excluded): median
7.6e-6 m, p99 2.3e-5 m, max 4.6e-5 m; identical valid-pixel mask (70.02%); best-fit image scale 1.0000005 (the old
FOV bug would give 0.833 — applying it to the AMD frame as a control gives a 1.75 m median depth error). Colour and
depth register exactly (arena edge on the same pixels). After back-projection to the planner's 512x512 grid the two
agree to a median of 5.7e-6 m. Vulkan on the 5090 and Mesa 25.2.8 also match AMD to <0.04 mm.

## Once the render is 7 ms
The back-projection to the planner's height grid (`sensor_map_v2.grid_from_arrays`) takes 54-56 ms on this CPU —
~88% of the sensing step. ~10 ms of it averages an all-zero colour image the depth-only runner passes in, and
~11-13 ms recomputes the per-pixel ray grid, which never changes.

## Corrections to earlier statements
- The "4.3-4.6 s when four runs share a node" figure quoted during nav_v1 was not a render time. The campaign's
  render medians with four arms on one mi2104x node were 2.35-2.59 s; 4.2-4.4 s was the whole decision (render +
  planning), and the 4.3-4.6 s renders were first frames. Several renders on one 16-CPU node were never measured.
- The workstation is 8 cores / 16 threads (Ryzen 7 5800X), not 16 cores.

## Open items
- A Vulkan-only build of the fork's current source has no Python depth camera: the bindings that expose it exist
  only as 7 uncommitted SWIG files in the AMD source tree (/home1/harry/chrono on the cluster). The local Vulkan build
  used a mirror of that tree (/home/harry/chrono_src_amd_mirror).
- While driving, the vehicle's silhouette in the OptiX frame sits ~0.16 m behind its reported chassis position
  (~0.10 m parked, mostly the reference-point offset); not resolved, harmless for the 1.5 m mask margin.
- Terrain colour differs between machines (texture found locally, fallback colour on AMD); depth is unaffected.
- Files left on disk by the local Vulkan build: /home/harry/chrono_src_amd_mirror (1.3 GB),
  /home/harry/chrono_build_vulkan_bench, /home/harry/tools/{vulkan_sdk,glslang,lvp_amd}.
