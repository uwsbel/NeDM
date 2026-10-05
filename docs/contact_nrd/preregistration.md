# Unified contact NRD: pre-registration (written 2026-10-02 ~11:20 CDT, before any version-2 contact training)

## Arms

**Unified model, version 2.** One shared Transformer core per system; one
collision network and one contact network shared over all pairs; pair frame;
per-pair-type conditioning (code: `src/nedm/contact_nrd/model_v2.py`,
`train_v2.py`). The core is trained once per system (`v2_pool_core`,
`v2_ball_core`) and shared by the contact-stage seeds.
- **Pool variants:** sphere-pair frame `centerline` or `approach`; seeds 61, 62, 63.
- **Ball variants:** contact band `band` (±30 ms), `gap2` (band without windows that end within 2 ms before an impact) or `noband`; seeds 61, 62, 63.
- **Variant choice per system:** the lowest validation score among the seed-61 runs. Seeds 62 and 63 are then run for the chosen variant only. Exception: if time allows, every variant runs all seeds, and the choice uses the mean validation score.
- **Deployment checkpoint per system:** the validation-selected seed of the chosen variant.
- **Selection scores (each system's own, unchanged):**
  - pool: target-worthy p95 of B at 2.0 s + 0.5 x B path p95 + 0.05 x event failures;
  - ball: trajectory p95 + 0.5 x endpoint p95 + 10 x order/count failures.

**Unified model, version 1 (earlier recipe), world and pair frame.** Reported as
controls (`runs/{pool,ball}_{world,pair}`).

**References** (frozen earlier; not retrained here; one seed each, which is a stated limitation):
- Pool:
  - per-cushion model `pool_ball_20261002/frozen/structured4_refcore_rr10` (SHA a9d01435...), the comparison reference;
  - shared-cushion model `structured_pb_v4_refcore_rr10`;
  - literal port `scalar_joint_v4`;
  - Transformer only `none_joint`.
- Ball:
  - two-switch K8 `ball_transformer_v2.../frozen/trained_ap_k8_w128_l4`, its validation pick, the comparison reference;
  - one-switch shared response `direct_binary_mlp_shared5`;
  - Transformer with analytic contact timing `nrd_v2_certified`;
  - analytic flight/timing + MLP `ball_precision_20261001/runs/certified_v2`.
  - The last two are physics-prior references and are labelled as such.

## Fresh cohorts (collected after the freeze)

1. Freeze manifest: every unified checkpoint (all variants and seeds) and every
   reference, with SHA256. It is written to `frozen/MANIFEST.txt` on AMD.
2. Cohort seed = `int(sha256(manifest_bytes + system_name)[:8], 16)`.
3. **Pool:** 4,800 shots, 4 per cell of the 30x40 speed/aim grid, same
   physics config and collector snapshot as training.
4. **Ball:** 1,800 shots, 4 per cell of the 30x15 launch grid, same config and
   collector.
5. The cohorts collected this morning (seeds 202610031 and 202610032) are not
   used for any headline number.

## Metrics

- **Pool**, all fresh shots:
  - B's error at t = 2.0 s (median, p95);
  - target-worthy p95;
  - B path error to t;
  - shots over 10 mm;
  - events right.
- **Ball:**
  - trajectory RMSE on the 10 ms grid over the valid horizon (median, p95);
  - endpoint p95;
  - contact order right;
  - shots with endpoint error over 10 mm.
- Every arm runs through the same evaluator family as its references, with the same episodes.

## Verdict rules (unified vs the comparison reference, per system)

- For median and p95: r = unified / reference, with a paired bootstrap 95 %
  interval (10,000 resamples of episodes, same resample for both arms).
  - **beats:** upper bound < 1.0;
  - **matches:** upper bound ≤ 1.25;
  - **worse:** lower bound > 1.25;
  - otherwise **inconclusive**.
- Counts (events wrong; over 10 mm): a paired exact McNemar test at 5 %.
- Any non-finite rollout fails that system.
- **Headline = the weakest cell**, never an average across systems. Downgrades are named explicitly:
  - "needs per-system settings": anchoring, augmentation, loss scales and refinement schedule are per system;
  - "one seed of the reference";
  - a pass on median only.

## Targeting (same protocol on both systems)

- **Pool:** 100 targets from the fresh cohort with the pool study's filter (half with no B cushion hit before t, half with one).
- **Ball:** 50 seeded random targets at 1.7 s from the fresh cohort.
- **Optimisers:** Levenberg-Marquardt (LM) is primary, fixed now; gradient descent is secondary.
- **Comparison and replay:**
  - data-only baselines (nearest training shot; local linear fit);
  - every chosen launch, every converged branch and each target's own source launch replayed in Chrono in one job per system.
- **Report:** misses within 1 cm (pool bar) and within 3 mm (the ball study's tolerance).

## Exploratory (only if time allows; labelled outside the matching claim)

- **Pool, mirrored:** 400 shots with A at (+0.635, 0) launched toward -x. The -x cushion module of the per-cushion model was never trained.
- **Ball, moved wall:** 200 shots with the wall at x = 4.5 m. The unified token carries the new plane; the references cannot receive it.
