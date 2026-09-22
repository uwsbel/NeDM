# Verification: gb_crop (ego-aligned elevation crops), revision 2 + fix round 1

Third verifier pass, 2026-09-21 (luffy, RTX 5090, conda `nedm`, torch 2.12.0+cu130), on the current
`scripts/gb_crop.py` (184 lines by `wc -l`, untracked; the report says 183, either way under the 200-line limit).
Reviewed the code, `NOTES_gb_crop.md` (including the "Revision 2" and "Fix round 1" sections), the shipped
`selftest/crop/selftest.json` and the consumer `scripts/gb_nrd_common.py` (`CropTokenizer`, `save_nrd`/`load_nrd`)
against PLAN.md "Dynamics crops (B)" and the checkpoint / cache contracts. Rules respected: no cluster submission, no
CRM or rigid run (the module has no Chrono dependency), no lock taken; GPU peak of my probe script 75 MB (self-test
configuration alone 31-39 MB). My re-run wrote only to `/tmp/verify_gb_crop_r3`; the only artefact file I wrote is this
one. `grid.npz` / `grid.json` mtimes are still 2026-09-16 23:36; `git status` shows only the five pre-existing
modifications from before this workflow (nav_v1 LOG/REPORT, docs/progress.md, nav_local_batch.py, nav_video_rgb.py)
plus untracked new scripts.

Verdict: **pass**. Every claimed number reproduces bit-for-bit (throughput within timing noise), the earlier problems
P1-P3 and notes N1-N2 are fixed and their repros pass on the current code, the grid orientation is confirmed against
recorded pitch/roll in both worlds with all three mirrored hypotheses rejected, and none of my attempts to break the
module found a defect. Three minor notes are recorded in section 4; none blocks use.

## 1. Reproduction of the self-test

`PYTHONPATH=src:scripts python scripts/gb_crop.py --selftest --out /tmp/verify_gb_crop_r3` (exit 0). Leaf-by-leaf diff
of the produced `selftest.json` against the shipped file: **41 of 43 leaves identical**; only the two throughput
numbers differ (forward 19.36 M vs 16.57 M crops/s, forward+backward 3.23 M vs 3.40 M; the GPU is shared with a 1.25 GB
foreign process). Identical: sign conventions (+6.0/0.0, 0.0/+6.0, 0.0/-6.0), sampler vs `sensor_dataset_v2.sample`
0.0 m, torch vs numpy sampler 2.866e-6 m, k8/6 m crop max diff 2.682e-6 (valid fraction 0.4894), k16/4 m 2.325e-6
(0.5727), `valid_agree` and `BL_shape_equal` true, gradient medians (7.26e-4, 1.83e-3, 3.99e-3) vs (2.237, 2.201,
1.538) finite, regression check (grad finite, z float32, fp16-pose diff 1.04e-6), reference rmse 9.29 / 4.74 / 4.60 mm
with max 41.1 / 21.8 / 19.2 mm, grid sha256 `dfd3cfb0deff4fc1e8418e9b073cc261ec1e6868e14c59e8ebc19dfe9617b23c`
(matches `sha256sum` of the file).

## 2. Contract checks against PLAN.md

- Geometry source: default `--grid` is `crm_f104_v1/grids/arena_f104_50h_v1/grid.npz` (the Chrono-frame v2 grid);
  `load_grid` reads `mpp` = 0.15625 and `half_extent_m` = 40 from `grid.json`, asserts `mpp * n == 2 * half` and full
  coverage (this grid: coverage 1.0). Poses are used unscaled, correct for a grid measured from Chrono's mesh (the
  reference test shows the BMP `TerrainMap` needs the 511/512 scale, the grid does not).
- Crop definition: 8x8 at +-6 m and 16x16 at +-4 m, `linspace` offsets forward (axis 0) x left (axis 1), heights minus
  the map height at the vehicle centre, /2 m, plus `valid`; numpy and torch twins agree to < 1e-4 (2.7e-6 measured).
  The offset ordering (`meshgrid(..., indexing='ij')`, raveled, reshaped to (k, k)) is the same as
  `traverse_wp2_add_terrain.terrain_patch`.
- Nothing cached to disk: the module writes only `selftest.json`; the B cache episodes (`cache_v1`) carry `z1, act,
  pose, power, desired_speed, parked, stalled, hold_ok, route_*` and no crop field [R23].
- Checkpoint contract: the grid is a non-persistent buffer; `state_dict()` is empty and `_non_persistent_buffers_set ==
  {'z'}` after `.half()`, `.bfloat16()`, `.double()`, `.to('cuda', float16)`, a parent `.half()`, CPU->CUDA and
  CUDA->CPU moves; `load_state_dict({}, strict=True)` passes; `copy.deepcopy` keeps cuda:0 and does not share the offset
  cache. `gb_nrd_common.save_nrd` stores `grid_path` + `grid_sha256`, `load_nrd` refuses a grid with a different
  sha256, and `gb_train_nrd.py` records `grid_sha256` in its config and checks it on resume (lines 463-474).
- Task-row, observation, timing, split and blacklist contracts do not apply (no episodes, groups, anchors or time
  axes are handled by this module).

## 3. Independent checks beyond the implementer's tests (real grid unless stated)

1. **Cell-centre identity**: at all 36 (row, col) combinations of {0, 1, 255, 256, 510, 511}, numpy and torch samplers
   return `z[row, col]` with error 0.0; the transposed hypothesis `z[col, row]` is off by up to 0.94 m.
2. **Grid orientation against recorded vehicle attitude, both domains** (12 rigid + 12 CRM cache episodes chosen at
   random, frames with |vx| > 1 m/s: 3,705 / 3,225 valid frames). Correlation of chassis pitch (z1 col 3) with the
   crop's central forward slope and of roll (col 2) with the central left slope:
   as-is grid rigid (-0.945, +0.944), CRM (-0.940, +0.967) (negative pitch sign = Chrono's nose-down-positive ISO
   convention); rows flipped rigid (0.195, 0.569), CRM (0.024, 0.055); columns flipped (-0.333, 0.058) / (-0.245,
   0.098); transposed (0.113, -0.121) / (-0.197, -0.074). Axis 0 = forward, axis 1 = left, rows = +y, columns = +x on
   real data; every mirrored alternative is rejected.
3. **Fix-round repros on the current code**: P1 (built on CPU, forward, `.to('cuda')`, forward): 1.7e-6 to numpy;
   P2 (first use of k=16 under `inference_mode`, then autograd forward + backward): gradient finite, cached offsets are
   not inference tensors; P3 (`.half()`, `.bfloat16()`, `.double()`, `.to('cuda', float16)`, parent `.half()`): grid
   float32, output float32, 1.6e-6 to numpy in all cases; N2 (in-place `grid['z'] += 5` after building a CPU module):
   output unchanged, `np.shares_memory` False; N1 (consumer `CropTokenizer(...).half()`): float16 token, 6.5e-5 from
   the float32 token.
4. **CPU vs CUDA `grid_sample`** on 10,000 poses in +-46 m: k8 2.4e-6, k16 2.7e-6; `valid` identical across CPU, CUDA
   and numpy.
5. **Shapes and inputs**: `(0, 3)` -> `(0, 8, 8)` + `(0,)`; `(3,)` -> `(8, 8)` + scalar (numpy the same); a
   non-contiguous `state[:, :3]` slice gives exactly the contiguous result (0.0); yaw + 4 pi changes the crop by 8e-7;
   a float64 pose with `requires_grad` returns a finite float64 gradient; under `torch.autocast('cuda', bfloat16)` the
   crop stays float32 and matches numpy to 1.4e-6.
6. **Bad poses**: (nan, 0, 0), (inf, 0, 0), (0, 0, nan), (1e9, 0, 0) all give `valid = False` in both versions; the
   torch crop is finite, the numpy crop is NaN for the NaN cases (as documented).
7. **Border**: `valid` flips at exactly the arena edge in both versions (x = 33.999 / 34.0 / 34.001 with a +6 m forward
   sample: True / False / False).
8. **Backward determinism**: five backward passes at k=16 on the same 64 poses are bitwise identical.
9. **Consumer path**: `CropTokenizer(grid, 16, 4.0, device='cpu')` on a (1, 2, 3) pose, `.to('cuda')`, forward: token
   (1, 2, 64), CPU vs CUDA 3.0e-8; `state_dict` keys under `mlp` only; backward to the pose finite; `.half()` gives a
   float16 token (the line-267 cast the fix round added); `no_grad` forward fine.

## 4. Problems found

None blocking or major.

### N1 (minor, documentation): an fp16 pose is accepted but silently quantised to ~3 cm at the arena edge

`forward` upcasts the pose with `pose.float()`, so a float16 pose is used at its fp16-rounded position (resolution
0.031 m for |x| in [32, 64) m, 0.016 m in [16, 32)). The docstring and note say "any float dtype; the computation is
float32", and the self-test compares against numpy evaluated at the same rounded pose, so the check cannot see this.
No planned caller passes fp16 poses (the cache stores `pose` as float32; autocast leaves inputs alone), so this is a
documentation risk only. Smallest fix: state in the docstring that poses should be float32/float64 (fp16 poses lose
centimetres), or `assert pose.dtype in (torch.float32, torch.float64)`.

### N2 (minor, note): the offset cache is a plain Python dict, not a buffer

`_off` is rebuilt when its device differs from the grid's, which handles `.to()`. Under `nn.DataParallel` the replicas
would share the same dict object and race on the rebuild; the plan is single-GPU and no consumer uses DataParallel or
DDP with this module, so no change needed. `torch.compile` is not used by any gb_* consumer (grep), so the Python
side effect in `forward` causes no graph breaks in practice.

### N3 (trivial): line count in the report

The report says 183 lines; `wc -l` gives 184 (final newline). Under the 200-line limit either way.

## 5. Notes (no action required)

- The known limits in the note are accurate: `valid` at +-40 m while clamping starts at the last cell centre (8 cm
  ring), NaN numpy crop vs finite torch crop, static undeformed grid only (no CRM sinkage), and the 9.3 mm rmse gap to
  the old NRD cache `terrain` field (BMP at the unscaled pose), which `gb_train_nrd.py` must not mix with the new crops.
- The `_apply` override relies on the private `nn.Module._apply(fn, recurse=True)` signature (torch 2.12.0); it
  forwards `*args/**kwargs`. Worth a glance on a torch upgrade, as the implementer says. A consequence of the override:
  `.to(device)` copies the grid twice (super's copy is discarded, then the kept float32 tensor is moved); 1 MB, harmless.
- The sampler-vs-`sensor_dataset_v2.sample` check (0.0 m) is a copy-consistency check; the independent orientation
  evidence is section 3.1-3.2 here (cell-centre identity and pitch/roll correlations with mirrored hypotheses) and the
  earlier passes' dihedral test against the BMP.
- Throughput reproduced within 15 % (19.4 M forward, 3.2 M forward+backward crops/s at B=4096, k=16); as the note says,
  order-of-magnitude figures on a shared GPU.
