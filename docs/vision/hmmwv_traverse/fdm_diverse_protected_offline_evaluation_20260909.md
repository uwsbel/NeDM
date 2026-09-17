# Protected offline evaluation protocol

The final model and planning protocol were frozen at 2026-09-09 22:56:01 UTC before protected outcomes were opened. Core freeze `protected_test_freeze_v1.json` SHA256 is `29684ce59cdd68e9b1f7310dcd2e63c55d021f6ff6712b83f99d286722efeb33`. The parent explicitly authorized unsealing and inference afterward. The core freeze was never modified.

## Immutable analysis

`evaluate_traverse_fdm_diverse_sealed_test.py` requires the core freeze SHA and a separate analysis supplement SHA before it opens test data. It rejects sealed declarations before reading supplement/test paths. It verifies source, checkpoint, task and training-pack hashes, then uses the original preparation/evaluation/report functions. No optimizer is created, no checkpoint is selected from test, and no success-based filtering is applied.

Source snapshot: `snapshots/test_offline_v1/source_manifest.json`, SHA256 `385aa09092a28801a8f68f092d9aee0330358f0ca92226b1054d3ce5177c999d`. All model/trainer/helper hashes agree with the original training provenance. The reporter functions are unchanged from reporting_v1; preparation runs the frozen packing_v3 code against campaign_v2 physical data.

- RGBD: seed11, LAST5000, checkpoint SHA `0251cb87ddd25b6dfd8c628680f3470983ca4ca36ef65a05bb966021a4691a20`.
- Matched blank: seed11, LAST5000, checkpoint SHA `14cc3302ede9e514feee08285bcef3e12d554fc7db4d9a7770bf13c8d59e0a1a`.
- Controls: original image, scene-deranged image, blank image; same 20260908 shuffle seed.
- Forecast metrics: 4/8/12s, all windows, anchor0, causal pre/prior failure groups, and per-scene/equal-scene summaries. Fixed contact/rollover/bounded-motion thresholds are .35/.35/.5.
- Labels include strict asset OR chassis contact, solver attitude peaks, bounded-motion and separate near-stop stall, positive engine-interface mechanical work and signed endpoint attitude. Work is not fuel energy.
- Fixed baseline: lateral offset -44m, cruise6m/s; exact route metadata must agree with index11. All15 measured references per scene remain in support tables. Hindsight best references describe feasibility only.

The native test filenames remain `test.npz` and `test_rgbd.npy`; the test split is never renamed validation. Preparation copies the frozen training-pack normalization, while model inference uses its original checkpoint normalization. The test image cache uses the same float16 encoding as training, promoted to float32 for model input. This is explicitly distinct from deployment's original float32 map cache.

## Jobs and files

CPU job412125 prepared all90 references on6 scenes, yielding12901 uniform1s-anchor windows. Pack: `packs/protected_test_h60_v1`; manifest SHA `708612fa1da0aadc6b1d20f8fed064084d09faa6a7d495c6bfda9c67eb209cfd`. CPU inference was then stopped to move the six full evaluation passes to AMD GPU; its draft directory is preserved.

GPU job412126 uses the identical evaluator with `--stage evaluate --device cuda:0`, MI3501x,24CPUs,30min cap. The immutable device-transition supplement is `offline_evaluation_freeze_v2.json`, SHA `c444305339fc87f90c66320cf480433ef27f0d0e4f4c79df297a623a8d743dec`; report destination `reports/protected_test_offline_gpu_v1`. Original supplementv1 remains intact.

Example command (after an authorized freeze, using a fresh report destination):

```bash
campaign=/work1/dannegrut/harry/experiments/fdm_diverse_v1_20260909
sbatch --partition=mi3501x --cpus-per-task=24 --time=00:30:00 \
  --export=ALL,EVAL_SNAPSHOT=$campaign/snapshots/test_offline_v1,FINAL_FREEZE=$campaign/protected_test_freeze_v1.json,FINAL_FREEZE_SHA256=29684ce59cdd68e9b1f7310dcd2e63c55d021f6ff6712b83f99d286722efeb33,OFFLINE_FREEZE=$campaign/offline_evaluation_freeze_v2.json,OFFLINE_FREEZE_SHA256=c444305339fc87f90c66320cf480433ef27f0d0e4f4c79df297a623a8d743dec,TEST_PACK_OUT=$campaign/packs/protected_test_h60_v1,TEST_REPORT_OUT=$campaign/reports/protected_test_offline_gpu_v1,EVAL_STAGE=evaluate,EVAL_DEVICE=cuda:0 \
  "$campaign/snapshots/test_offline_v1/slurm/traverse_fdm_rgbd_diverse_sealed_test.sbatch"
```

Offline accuracy and fixed-reference feasibility do not establish receding-horizon MPPI success. The separately frozen online trials provide that evidence. The computed results below are tied to the frozen producer exports; finalization independently verifies their hashes, recalculates every summary, and preserves small cross-CPU angular rounding differences in the audit.


## Measured protected results

All 90 reference routes remain in the cohort. One early-censored episode has no complete 12s trajectory/contact/bounded negative label, so those anchor-zero metrics use 89 known endpoints; all90 episodes remain available with masks. Rollover support is known for all90.

| At anchor0, 12s horizon | RGBD LAST5000 | Matched blank LAST5000 |
|---|---:|---:|
| Endpoint error | 5.069m | 7.118m |
| Positive mechanical work MAE | 65.73kJ | 112.68kJ |
| Contact AUROC | .9347 | .7528 |
| Contact positives rejected at .35 | 6/9 | 1/9 |
| Contact false rejections | 4/80 | 2/80 |
| Bounded-motion positives rejected at .5 | 0/5 | 0/5 |
| Rollover positives rejected at .35 | 0/1 | 0/1 |

At the same90 starting anchors, substituting another scene map increases RGBD endpoint error from5.069m to7.974m; blanking the map increases it to21.468m. The independently trained blank arm is a separate, capacity-matched model and has7.118m error. These controls support use of the actual scene image.

All-window contact/bounded recall is95.5%/89.0%, but that aggregate includes many anchors after failure was already observed. The 4,551 pre-first-failure windows give73.8%/41.6% for RGBD versus51.5%/19.1% for matched blank. Report these causal figures alongside the easier aggregate; neither eliminates the missed starting-state stall and rollover cases.

Measured safe reference counts are cross3, mixed3, ridge3, rolling6, rough3, valley3:21/90 total, with at least one feasible control on every scene. The predeclared -44m at6m/s baseline safely finishes mixed, rolling and rough (3/6). Its successful runs take41.35,41.35,41.30s and use275.15,351.90,263.12kJ respectively. Selecting hindsight references per scene is a feasibility diagnostic, not a deployed baseline.

## Export finalization audit

GPU job412126 completed all six passes in30.25s of inference, then hit a provenance-only error while trying to hash PyTorch's nonexistent synthetic `_ops.py`. All prediction arrays, metrics and physical support were preserved. Export manifest `protected_test_prediction_export_v1.json` SHA `7449340dc1c24be24630b4a06355787183e9b30bdbc46e3aee9fd9ea702f5ec8` pins all nine output files and the original source, supplement, pack and job-log hashes.

A CPU finalizer verifies those exact prediction hashes and independently recomputes every horizon/causal/per-scene summary plus all90 raw outcomes. The first strict JSON comparison detected float32 angular rounding across CPU instruction sets (largest RGBD-normal difference0.0000122degrees), so the corrected finalizer preserves the producer values and logs numeric differences. Counts, schemas, keys and booleans remain exact; float tolerances are explicit in the report. It skips only the explicitly named nonexistent synthetic external `torch.ops/_ops.py` and `torch.classes/_classes.py`; every pinned first-party file remains hash-required.

Finalizer source `snapshots/test_offline_finalize_v2/source_manifest.json` SHA `1acd3299fd6862104a07bde2e3e0a78642df5d7a4f37f026b14e27374666e913`; supplementv4 SHA `ad65cbeda742edab906e9b4c683062dec883e9ac4aa789cc31e6b6427c88072d`. Job412139, CPU-only, writes a fresh `reports/protected_test_offline_final_v2`. Inference is not repeated; earlier outputs and failed audit logs remain intact.


Finalization **completed successfully** in job412139 (1m24s), at2026-09-09 23:13:15UTC. Local small report and support files are in `artifacts/traverse/fdm_diverse_v1_20260909/reports/protected_test_offline_final_v2/`, with download hashes verified. Report SHA `e799eed0189741f3b788b8560c9ad5ec3a45e74ffdcb6fb5253e9c4c250585c1`; support SHA `2902cd5fc4d03bc562e522405e46116a1b7f1571e242eb80037ec22ffcda9dcb`.

All six summaries passed: every non-angular metric, count, key and discrete value recomputed exactly. Only angular summaries differed, by at most0.0000275degrees across the six controls, within the recorded floating-point tolerance. The independent90-route physical support recomputation was completely exact, including all continuous work/time/attitude values. Only the two explicitly named synthetic PyTorch modules were skipped; all real first-party source hashes passed. No forecast was recomputed and no optimizer ran during finalization.
