# Module note: gb_crop (ego-aligned elevation crops from the v2 grid)

File: `scripts/gb_crop.py` (183 lines, numpy + torch only). Self-test artefacts: `B_tracker/selftest/crop/selftest.json`.
Written 2026-09-21 for milestone B (dynamics crops; PLAN.md conventions "Dynamics crops (B)"). Revision 2 (same day,
after `VERIFY_gb_crop.md`): the three robustness problems the verifier found (P1-P3) and one more found on re-check
are fixed; the public API and every numeric result are unchanged (section "Revision 2" below).

## What it does

One sampler for the local terrain around the vehicle, read from the Chrono-frame v2 grid
`artifacts/traverse/crm_f104_v1/grids/arena_f104_50h_v1/grid.npz` (sha256 `dfd3cfb0deff4fc1...`, recorded in
`selftest.json`), at any pose, on CPU (numpy) or on the GPU (torch, differentiable in the pose). Nothing is cached to
disk; the crop is computed on the fly.

- `load_grid(path)` -> `dict(z (512,512) float32, mpp 0.15625, half_extent 40.0, n 512, path)`. `path` may be the
  npz or its directory; `grid.json` next to it supplies `mpp` and `half_extent_m`. Grid rows increase with +y (row 0 is
  the -y edge), columns with +x, cell centres at `-40 + (i + 0.5) * mpp`. The grid is a measurement of Chrono's own mesh,
  so poses are used as recorded (no 511/512 scale). The loader refuses grids with empty (NaN) cells.
- `sample_height(grid, xy)` (numpy) and `EgoCrop.sample_height(xy)` (torch): bilinear map height at world (x, y),
  border-clamped outside the arena; `xy` is `(..., 2)`.
- `ego_crop_np(grid, pose, k=8, half_m=6.0)` -> `heights (..., k, k) float32`, `valid (...,) bool`. `pose` is
  `(..., 3)` = world x, y, yaw (recorded `trajectory.npz['pose']` can be passed as is). Sample points are
  `linspace(-half_m, half_m, k)` along the vehicle's forward axis (crop axis 0, index 0 = behind) and left axis (crop
  axis 1, index 0 = right); heights are the map height at the point minus the map height at the vehicle centre, divided
  by 2 m (identical convention to `traverse_wp2_add_terrain.terrain_patch`, i.e. the NRD cache's `terrain` field).
  `valid` is true when all k*k points lie inside +-40 m.
- `EgoCrop(grid, device)`: an `nn.Module` holding the grid as a non-persistent buffer (it never lands in a checkpoint;
  consumers rebuild it from the grid path). `forward(pose, k=8, half_m=6.0)` accepts `(B, 3)` or `(B, L, 3)` (any leading
  shape, any float dtype; the computation is float32) and returns the same two outputs as the numpy version, using
  `grid_sample` (bilinear, border padding, `align_corners=False`, which maps the normalised coordinate `x / 40` to exactly
  the cell-centre index `(x + 40) / mpp - 0.5` used by numpy). Gradients flow to the pose (x, y and yaw). The module may
  be built on one device and moved with `.to()`, used first under `torch.inference_mode()` and later with autograd, and
  converted with `.half()` / `.bfloat16()` (directly or through a parent): the grid stays float32 in all cases.

The two crop sizes named in the plan are `k=8, half_m=6.0` (1.71 m spacing) and `k=16, half_m=4.0` (0.53 m spacing).

## How to run

```
cd /home/harry/NeDM-traverse_mppi
PYTHONPATH=src:scripts /home/harry/miniconda3/envs/nedm/bin/python scripts/gb_crop.py --selftest \
    --out artifacts/traverse/generalist_20260921/B_tracker/selftest/crop
```
Library use:
```python
import gb_crop
grid = gb_crop.load_grid("artifacts/traverse/crm_f104_v1/grids/arena_f104_50h_v1/grid.npz")
h, ok = gb_crop.ego_crop_np(grid, pose_T3)                 # numpy, (T,8,8), (T,)
crop = gb_crop.EgoCrop(grid, "cuda")
h, ok = crop(pose_BL3, k=16, half_m=4.0)                    # torch, (B,L,16,16), (B,L), differentiable in pose
```
`scripts/gb_nrd_common.py` (`CropTokenizer`) already imports it this way.

## Tests (all from the self-test command above, RTX 5090, torch 2.12.0+cu130; numbers in selftest.json, re-run after revision 2)

1. Sign conventions on synthetic ramp grids (z = x, z = y), pose (3, -5, yaw): with yaw 0 the x-ramp rises along crop
   axis 0 by 6.0 (= 12 m / 2 m) and is flat along axis 1; the y-ramp rises along axis 1 (left = +y); with yaw 90 deg the
   x-ramp falls along axis 1 (left = -x) by 6.0. So axis 0 = forward, axis 1 = left, rows = +y is right.
2. Height sampler vs the existing v2 corridor sampler `sensor_dataset_v2.sample` on 20,000 points in +-42 m:
   max |diff| = 0.0 m (bit-identical). Torch vs numpy sampler: max |diff| 2.9e-6 m.
3. Numpy vs torch crops on 10,000 random poses with centres in +-46 m (about half the crops touch or cross the arena
   edge, so border clamping is exercised): k=8/6 m max |diff| 2.7e-6, k=16/4 m max |diff| 2.3e-6 (in /2 m units;
   requirement < 1e-4). `valid` flags identical (valid fraction 0.489 for 8x8/6 m, 0.573 for 16x16/4 m at this pose
   spread); `(B, L, 3)` input gives exactly the same tensors as the flattened `(B, 3)` input. Also checked ad hoc:
   a single `(3,)` float64 pose gives `(8, 8)` + scalar bool in both versions (diff 5.7e-7), and `state_dict()` is empty
   before and after `.to("cuda")` / `.half()` (the grid buffer is not saved; `load_state_dict({}, strict=True)` passes;
   `copy.deepcopy` keeps the buffer on its device).
4. Differentiability: autograd gradient of sum(crop) w.r.t. (x, y, yaw) vs central finite differences of the numpy
   version on 200 interior poses: median |diff| (7e-4, 1.8e-3, 4.0e-3) against median |gradient| (2.24, 2.20, 1.54),
   all finite. The residual is float32 plus the piecewise-linear kinks of bilinear interpolation at cell edges.
5. Robustness regressions (`regress_to_device_inference_half` in selftest.json): one module built on CPU, run once at
   k=8, moved to the GPU, first used at k=16 inside `torch.inference_mode()`, then run at k=16 with autograd and
   backward (gradient finite), then converted through a parent `nn.Sequential(...).half()` and fed an fp16 pose: the
   grid buffer is still float32 and the crop matches numpy given the same fp16-rounded pose to 1.0e-6. Ad hoc: direct
   `crop.half()` and `crop.bfloat16()` work and give the CPU float32 result to 2.4e-7; the verifier's exact P1 and P2
   repros pass; `gb_nrd_common.CropTokenizer` built on CPU, run, moved to the GPU and run again gives the same token
   (3.7e-8) and backpropagates to the pose.
6. Reference against `traverse_wp2_add_terrain.terrain_patch` on `TerrainMap` (BMP heightmap of
   `assets/traverse/arena_f104_50h_v1`) for 20 random poses inside +-31 m, k=8, +-6 m, differences in metres:
   - raw (TerrainMap queried at the Chrono pose): rmse 9.3 mm, max 41 mm;
   - TerrainMap queried at the pose scaled by 511/512 (offsets unscaled): rmse 4.7 mm, max 22 mm;
   - TerrainMap queried at every sample point scaled by 511/512: rmse 4.6 mm, max 19 mm.
   The scale removes half the discrepancy; the rest is the depth-derived grid versus the 8-bit BMP (the scout measured
   grid-vs-TerrainMap 3.7 mm rmse per point at that scale; a relative height subtracts two such errors, so ~5 mm is the
   expected floor). Both are far below the 22 mm BMP quantisation step. No orientation or axis error (that would show as
   decimetres).
7. Throughput on the 5090, B=4096 poses, k=16, +-4 m (1.05 M sample points per call), 100 timed calls after warm-up:
   forward 17.3-18.8 M crops/s (0.22-0.24 ms per batch) over three runs; forward + backward to the pose 3.0-3.2 M crops/s
   (1.3 ms per batch). Peak GPU memory of that configuration was 31 MB (verifier's measurement).

## Revision 2 (what changed and why)

`VERIFY_gb_crop.md` passed the module "with issues": the torch class kept its sample offsets in a plain Python dict
filled lazily on first use, so (P1) `.to(device)` after a forward left the offsets on the old device and the next forward
crashed; (P2) a first forward under `torch.inference_mode()` cached inference tensors and a later backward for the same
crop size failed; (P3) `.half()` on a parent module quantised the grid and the coordinates silently (up to 69 mm with
bf16). Re-checking also found that the attribute holding the arena half-extent was named `half`, which shadowed
`nn.Module.half()`, so a direct `crop.half()` raised `TypeError: 'float' object is not callable`.
Fixes, all inside `EgoCrop`: the offset cache is rebuilt whenever its device differs from the grid's and is always filled
inside `torch.inference_mode(False)`; the pose and the normalised coordinates are cast to float32; `_apply` keeps the
original float32 grid tensor across dtype conversions (moving it to the new device only); the attribute is now
`half_extent`. The self-test gained check 5 above and the grid sha256. Numbers in checks 1-4 and 6 are bit-identical to
revision 1; throughput is within timing noise.

## Known limits

- Bilinear interpolation is only C0: the pose gradient is piecewise constant and jumps at cell boundaries (0.156 m).
  Fine for training through the crop; gradient-based planning on the crop alone would see a staircase.
- `valid` uses +-40 m (the arena edge). Border clamping actually begins at the last cell centre (39.92 m), so points in
  the outer 8 cm ring read the edge cell and are still flagged valid; poses far outside the arena get clamped edge values
  with `valid = False`. The torch flag is computed in float32, so a point within ~1e-5 m of the edge could in principle
  be flagged differently from numpy (none of the 10,000 test poses did).
- A NaN pose gives `valid = False` in both versions; the numpy crop is then NaN (with a numpy RuntimeWarning from the
  integer cast) while the torch crop is finite. Consumers must gate on `valid`.
- The loader asserts full coverage; a future sparse capture would need a fill policy (not needed for this arena,
  coverage 1.0).
- Only the static undeformed grid is sampled; CRM sinkage is not in the crop by design (heights are relative to the map
  at the centre, not to the chassis).
- The new crops are not bit-compatible with the old NRD cache `terrain` field (BMP at the unscaled pose; 9.3 mm rmse
  apart, check 6); `gb_train_nrd.py` must not mix the two. Consumers should record the grid path and sha256 in their
  metadata since the grid is never saved in a checkpoint.
- No chart or image output; the check is numeric only.


## Fix round 1 (2026-09-21, after `VERIFY_gb_crop.md` revision 2)

The three robustness issues named for this round (the sample-offset cache not following `.to(device)`; offsets created
under `torch.inference_mode()` breaking a later backward; silent precision loss of the grid and the coordinates under
`.half()`) are the first-pass problems P1-P3 already fixed in revision 2; the revision-2 verifier confirmed them and
left two minor notes. This round re-verified P1-P3 on the current code and fixed both notes:

- N2 (a CPU module aliased the caller's numpy grid): `scripts/gb_crop.py:75` builds the buffer with `torch.tensor(...)`,
  which always copies (+1 MB per CPU module; a CUDA module copied already). Public API and numbers unchanged.
- N1 (a half-converted consumer MLP failed loudly because the crop is always float32): fixed on the consumer side,
  `scripts/gb_nrd_common.py:267` casts the crop features to the token MLP's weight dtype. The crop output itself stays
  float32 by design (documented above); autocast is unaffected.

Tests re-run: `PYTHONPATH=src:scripts python scripts/gb_crop.py --selftest --out .../selftest/crop` (exit 0, GPU peak 39 MB):
37 of 39 leaves identical to the previous `selftest.json` (sign conventions +6.0/0.0, 0.0/+6.0, 0.0/-6.0; sampler vs
`sensor_dataset_v2.sample` 0.0 m; torch vs numpy sampler 2.87e-6 m; k8/6 m crops max |diff| 2.68e-6 with valid
fraction 0.4894, k16/4 m 2.32e-6 / 0.5727, `valid` and (B, L, 3) shape checks true; gradient medians (7.3e-4, 1.8e-3,
4.0e-3) vs (2.24, 2.20, 1.54), finite; `regress_to_device_inference_half` grad finite, grid float32, fp16-pose diff
1.04e-6; reference rmse 9.3 / 4.7 / 4.6 mm; grid sha256 unchanged); only the two throughput numbers moved (16.6 M vs
17.3 M crops/s forward, 3.4 M vs 3.0 M forward+backward, timing noise on a GPU shared with a foreign training job).
Explicit repros on 256 random interior poses of the real grid (scratch, not in the repo):
- P1: built on CPU, forward, `.to('cuda')`, forward: max |diff| vs numpy 1.34e-6 / 1.49e-6, offsets on cuda:0.
- P2: first use of k=16 under `inference_mode`, then autograd forward + backward at k=16: gradient finite (median |grad|
  4.7), crop vs numpy 1.83e-6, cached offsets are not inference tensors.
- P3: `crop.half()`, `crop.bfloat16()`, `nn.Sequential(crop).half()[0]`: grid float32, output float32, max |diff| vs
  numpy 1.49e-6 in all three cases; `state_dict()` still empty after `.to()`/`.half()`.
- N2: after `grid['z'] += 5` in place, a CPU module's output changes by 0.0 and shares no memory with the array.
- N1: `CropTokenizer(...).to('cuda').half()` now returns a float16 token (3.4e-4 from the float32 token, the MLP's
  fp16 rounding); under `autocast(bfloat16)` the crop inside stays float32 and the token is bfloat16 (2.5e-3 from
  float32, as before); a tokenizer built on CPU and moved to the GPU gives the same token to 5.8e-7.
