# Static F104 RGB-D audit

**All available checks passed.** This audit used saved AMD job 412395 outputs only; no physics, native-height query or sensor render was rerun.

- All four downloaded file hashes match AMD; the BMP, metadata, capture script and frozen source-manifest hashes match.
- Against all 4,096 saved native terrain points, measured ray-range error is **1.03 mm median / 8.83 mm p95**, including image interpolation. Reconstructed height error p95 is **8.56 mm**.
- Mirroring image y gives **2.73 m p95** error, versus 0.0088 m for the saved orientation. Other flip/transpose controls are likewise worse. Native/BMP p95 recomputes exactly at **20.49 mm**.
- RGB terrain/sky segmentation and valid-depth support agree at **every pixel** (IoU 1.0). All arena corners fit inside the frame. This checks the outer boundary; no vehicle marker exists to prove every internal RGB correspondence.
- The measured 1024² RGB-D encodes exactly to the saved 4×512×512 model input. No anchor state or native truth arrays occur in that input file.

Join the static pixels with **each episode's own measured anchor pose/state/history, goal and reference**. For later windows, use that episode's causal trajectory prefix. The map must never supply a shared vehicle anchor. Keep native_height_audit.npz and authored terrain metadata outside model inputs.

The deferred pack/model must use this map's **110 m camera height, 47° FOV and 10 m elevation scale**. Existing diverse defaults are 400 m / 40 m and would misproject local patches if reused silently. Pin camera, BMP/meta, image, material and runtime hashes. Material values match the physics source: friction 0.9, restitution 0.01, Young's modulus 2×10⁷ Pa. This audit records a canonical material hash because the capture metadata does not store one explicitly.

The map is suitable for deferred joining under those requirements. Episode-specific joins are not claimed complete by this audit. [Machine-readable checks and provenance](audit.json).
