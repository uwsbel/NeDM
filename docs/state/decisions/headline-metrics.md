# Headline metrics, with provenance

Every number this case study puts in front of an audience, with the artifact behind it and
the command that regenerates it. Nothing enters this table that cannot be pointed at.

This file exists because several figures reached a slide with no record behind them. Two
were withdrawn on 2026-09-20 after an audit: a throughput pair that had only ever been
printed to a terminal, and a horizon figure with no measurement at all. A third, the
corpus collection cost, turned out to be a hardcoded literal in the script that appeared
to compute it. The failure mode is always the same, a number that is true-sounding,
repeated, and unattached, so the rule is that a presentable number lives here or it is
not presentable.

Paths are relative to the euler working root `/srv/home/kasha2/nedm` unless marked REPO,
which means `/srv/home/kasha2/nedm/NeDM`.

## Terrain and corpus

| quantity | value | source |
|---|---|---|
| SPH particles, Go2 CRM terrain | 886,611 | REPO `docs/state/decisions/quadruped-bootstrapping.md:146`; runtime truth is `GetNumSPHParticles()` at REPO `scripts/quadruped_go2_crm.py:263` |
| CRM cost vs real time | 7.5x slower | REPO `docs/state/decisions/go2-crm-pilot.md:18`, measured over 14 pilot episodes on this terrain |
| CRM real-time factor | RTF 0.1136 | REPO `docs/state/decisions/go2-p-controller-baseline-result.md:64`, 152 episodes |
| baseline corpus | 795 episodes / 555,851 transitions | `training_datasets/go2_crm_baseline/metadata.json`, key `splits` |
| split | 635 train / 447,372 and 160 val / 108,479 | same file. **447,372 is TRAIN ONLY and must not be paired with 795** |
| command families | 8 | same dataset, verified by enumerating episode ids |

## Model

| quantity | value | source |
|---|---|---|
| architecture | 6 layers, n_embd 256, 8 heads, block 128 | `training_runs/go2_crm_sel_baseline/checkpoints/best_val.pt`, key `config["model"]` |
| state | 36-D, preset `quadruped_crm_baseline` | `training_datasets/go2_crm_baseline/metadata.json`; preset at REPO `src/nedm/training/constants.py:258` |
| action | 12-D, four legs x hip/thigh/calf target | same metadata, `action_fields` |

## Delivered policy, the headline

Regenerate:

```
cd /srv/home/kasha2/nedm/NeDM && python3 scripts/evaluation/crm_verdict.py \
  ../out/crmtrack_go2_cts_150k_euler.json ../out/crmtrack_sel_baseline_euler.json
```

| quantity | value |
|---|---|
| mean abs tracking error, imported policy | vx 0.1551 / vy 0.0599 / wz 0.1007 |
| mean abs tracking error, after fine-tuning | vx 0.0917 / vy 0.0459 / wz 0.0753 |
| change | **-40.9% / -23.4% / -25.2%** |
| paired episodes | n = 74 |
| completion | base 77/80, fine-tuned 74/80, 3 base-only |
| falls, either arm | 0/80 |

Reproduced digit for digit on 2026-09-20. The percentages are transcribed in
`rollout-selection.md`; the absolute means exist only as this command's output, which is
why the command is given rather than a citation.

**Carry the completion disagreement whenever the percentage is quoted.** The verdict tool
calls it "a result in its own right": n=74 is the intersection, not the sample size.

## Cost

| quantity | value | source |
|---|---|---|
| NN-ROM throughput | 11,075 transitions/s | REPO `docs/state/decisions/throughput-and-amortisation.md:12`, measured on north-ubuntu |
| Chrono CRM throughput | 4.17 transitions/s **per worker** | same, mean over 525 episodes, range 0.25-13.39 |
| per-transition speedup | 2,657x **against single-worker CRM** | derived, 11,075 / 4.17 |
| one fine-tune, wall clock | about 55 s, "about a minute on a 3090" | REPO `docs/state/decisions/go2-crm-finetune-RECIPE.md:69` |
| same transitions in Chrono | 6.0 h | derived, 90,240 / 4.17 |
| corpus collection | 37.0 h single-worker Chrono | derived, 555,851 / 4.17; script `scripts/throughput/measure_finetune_throughput.py` |
| break-even | 6.2 fine-tunes | derived, 37.0 / 6.0 |
| Chrono verdict, one policy | 25 min on 8 sharded GPUs, 57 min on 3 A100s, 70-90 min on one box | REPO `docs/state/machines/euler.md`, `go2-crm-experiment-ledger.md:949` |

The speedup is an order-of-magnitude claim: 4.17 is a mean over a 53x range, and
re-aggregating per collection gives 2.92 / 6.62 / 1.96. **Never quote it without naming
single-worker CRM as the denominator** -- a ~2,700x figure was already retracted once for
overstating exactly this.

## Rollout horizon

Regenerate: `scripts/evaluation/horizon_sweep.py sel_baseline valw256e wcov_roll
--episodes 32 --seed 0`. Artifact `out/horizon_sweep.json`, detail in REPO
`docs/state/decisions/rollout-horizon.md`.

| quantity | value |
|---|---|
| err/dist at the 0.30 s branch | 0.034 |
| stays below the predict-no-motion floor to | about 3 s |
| disturbed-corpus model, same measure at 0.30 s | 0.166, crosses the floor before 1 s |

`errdist` is planar error over distance travelled, so 1.0 is what predicting no motion
scores by construction. It is an analytic floor, not a fitted baseline.

## PPO comparison

| quantity | value | source |
|---|---|---|
| quadruped PPO budget | 2,048 envs x 64 steps x 2,000 iters = 262,144,000 | REPO `docs/state/decisions/go2-merged-model-comparison.md:36`, corroborated by `go2-p-controller-baseline-result.md:63` |
| HMMWV PPO budget | 2,048 x 64 x 1,000 = 131,072,000 | REPO `docs/progress.md:105` |
| ratio | exactly 2.0x | derived |
| best of seven PPO configurations | -6.5% | `out/crmtrack_ppolr_1e4_euler.json` via `crm_verdict.py` |

Catastrophic PPO figures recorded earlier (+217.5%, +438.6%) were **withdrawn** in
`de348b75` as an export defect. Do not quote them.

## Not presentable

Recorded here so they are not reached for by accident.

- **The abstraction ladder as a monotone trend.** The arms are not an information ladder:
  the 40-D contact flags are thresholded `foot_*_force_fz_n`, which the 48-D arm already
  carries in full. Ordered by information added the result is non-monotonic. What IS
  supportable is that no enrichment beat the 36-D baseline.
- **The capacity peak.** Two internally consistent ladders disagree on whether w512 beats
  w256, and the euler ladder is incomplete.
- **Any `val_loss` comparison spanning `6c0cb18d` (2026-09-17)**, which changed the
  validation subset from a prefix to a random sample. That includes the dose-ladder
  correlation, whose prefix bias scales with corpus size, the same axis it varies.
- **"RL needs ~10^8 steps".** Asserted in two places, both marked as superseded framing,
  neither citing a measurement. Use the study's own budget instead: 262,144,000 steps,
  which `go2-p-controller-baseline-result.md:65` puts at 3.66 years single-stream.
