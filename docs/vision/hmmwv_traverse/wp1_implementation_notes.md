# WP1 implementation notes — perception pilot (iterations v1–v7)

**Date:** 2026-09-02/03 · **Modules:** `nedm/traverse/perception.py`
**Scripts:** `scripts/traverse_wp1_train.py`, `scripts/traverse_wp1_probe_res.py`
**Artifacts:** `artifacts/traverse/wp1_v{1..4}/` (config, train_log.jsonl, ckpts, g1_readout.json), `wp1_v3/probe_res_readout*.json`

## Setup (plan §5)

Encoder: 256² RGB-D → 16²×256 spatial map → attention-pooled global latent z₂
(the v1 spine). Warm-up shapes the encoder with the four mandatory auxiliary
heads decoding **from z₂** (class masks, vehicle-center heatmap + yaw + pose,
foreground-weighted RGB recon, elevation recon) plus a BEV-occupancy head;
labels are analytic (rasterized on demand, never stored). After warm-up,
frozen-encoder probes decode BEV occupancy + vehicle pose from z₂ AND from
the pre-pooling spatial map — the gap is the measured cost of global pooling.
Layout-level 70/15/15 split; all metrics on held-out layouts; test untouched.

## Iteration log

| Run | Change | Held-out result | Lesson |
|---|---|---|---|
| v1 (20k steps, 1199 layouts) | first build | BEV z₂ 0.06 vs spatial 0.47; center 21 m (random) | two defects: ckpt selected on noise-level rock IoU (froze an early ckpt); heatmap loss ~2000× under-weighted (head stayed at zero) |
| v2 (30k) | attention pooling, direct pose head, heat ×2000, composite ckpt score | center 2.1 m / yaw 7.5°; house IoU 0.54; BEV z₂ 0.26 vs spatial 0.48 | localization + big objects fixed; **rock/tree still ~0** — encoder only gets gradients through z₂, so pooling-rejected content is never perceived |
| v3 (30k, 2199 layouts) | **deep supervision on the spatial map** (seg+BEV from 16² map), rock/tree class weights 8/4, probes 8k, train-vs-val diagnostic | spatial path: rock IoU **0.67** / recall **0.96**, tree 0.91, BEV **0.83**, center **0.90 m**, yaw 4.4°. z₂ path: BEV **0.72 train vs 0.23 val** | architecture + pipeline validated; z₂'s decodable layout content is **memorized**, not generalizable |
| v4 (30k) | z₂ capacity test: 512-D, 16 queries | BEV z₂ 0.243 (v3: 0.231); spatial unchanged 0.836 | capacity is **not** the constraint |
| v5 (30k, **7188 layouts**) | data-scaling test, v3 config (run by the day session) | BEV z₂ **0.327** / spatial **0.878**; rock recall 0.992; center 0.80 m spatial; z₂ memorization gap collapsed (0.44 train / 0.33 val) | z₂ **does** scale with data — but toward an apparent ~0.4 ceiling, not spatial's 0.88 |
| v6 (30k, **9518 layouts**) | third data-scaling point, v3 config | BEV z₂ **0.390** / spatial 0.883; rock recall 0.994; center 1.35 m z₂ / 0.78 m spatial | train/val pinch: z₂ train BEV 0.72→0.44→**0.42** while val rose 0.23→0.33→**0.39** — the curves converge at ~0.4: a representational ceiling, not overfitting |
| v7 (30k, same 9518) | **positional embedding before attention pooling** (`--pos-enc`), pinned to v6's episode manifest | *queued behind v6* | tests whether the ~0.4 ceiling is position-blindness rather than capacity or data |

The train-vs-val diagnostic (eval on training layouts alongside held-out) is
what separated "can't represent the task" from "can't generalize it": the
spatial path generalizes (0.97 train / 0.83 val) while z₂ mostly memorizes
(0.72 / 0.23).

## v7 hypothesis: the pooling stage is position-blind (2026-09-03)

`AttnPool` (`perception.py:209`) computes `pooled = att @ f` — a convex
combination of feature vectors over spatial positions. Nothing in the query
path encodes *where* a feature sat, and the conv backbone is
translation-equivariant, so absolute position only leaks in weakly through
zero-padding boundary effects. **z₂ is close to position-blind by
construction**, which is the one hypothesis consistent with every result so
far:

- vehicle pose survives (0.8–2.1 m) — one large salient object, the weak
  padding-based position leak suffices;
- multi-object layout fails (0.33) — 8–15 rocks at 3–6 px need real positional
  binding;
- z-dim 256→512 and n_q 8→16 changed nothing (v4) — more capacity for
  information that never entered the pooling;
- data scaling saturates near ~0.4 (v5) — the encoder learns a better *census*
  of the scene, not its arrangement.

Note the decoder is **not** the suspect: `LatentProbe` is already a full
conv-transpose decoder (z₂ → 4×4 seed → 64×64 pyramid → 128² BEV), and v3's
0.72 train / 0.23 val says it was memorizing layouts, not failing to fit them.
Widening it cannot help.

**v7** (`--pos-enc`) adds a learned per-cell embedding before pooling; **v7b**
(`--slot-z`) additionally keeps z₂ as `n_q` unmixed slots instead of the
default `Linear(n_q·c_map → z_dim)`, which flattens away the slot structure v4
was trying to test. Both default off, so v1–v6 configs reproduce bit-exactly
(v5 `ckpt_warmup.pt` still loads `strict=True`). Run pos-enc alone first —
one variable.

If v7 lifts held-out z₂ BEV toward the spatial path's 0.88, the §5 spatial-map
fallback becomes unnecessary and the planner can read z₂ directly.

## Dataset pinning

`split_episodes` permutes the **whole** enumerated set, so appending episodes
to a growing store (`full_v4_partial`) reshuffles every split and silently
breaks run-to-run comparison. Runs now write `episode_manifest.json` and accept
`--episode-manifest` to pin a predecessor's exact episode set. v6's set was
reconstructed and verified against its `config.json` (identical 6662/1427/1429
counts and identical test list) into
`artifacts/traverse/wp1_v6_episode_manifest.json`; v7 is pinned to it.

## Fallback sizing (frozen v3 encoder, BEV probes on compressed maps)

Held-out BEV IoU by planner-facing representation (`probe_res_readout*.json`):

| Representation | floats/frame | BEV IoU |
|---|---|---|
| 16×16×256 (full map) | 65 536 | 0.835 |
| 16×16×32 | 8 192 | 0.831 |
| **16×16×16** | **4 096** | **0.824** |
| 16×16×8 | 2 048 | 0.805 |
| 8×8×256 / ×128 / ×64 | 16 384 / 8 192 / 4 096 | 0.695 / 0.691 / 0.676 |
| 4×4×256 / ×64 | 4 096 / 1 024 | 0.357 / 0.328 |
| z₂ 256-D vector | 256 | 0.231 |

**Resolution is the load-bearing axis; channels are nearly free.** At the same
4 096-float budget, 16×16×16 gives 0.824 where 8×8×64 gives 0.676. A 2 048-
float 16×16×8 map — only 8× the size of z₂ — recovers 0.81 vs z₂'s 0.23.

## G1 recommendation

Adopt the plan §5 pre-declared fallback: **planner-facing representation =
16×16 low-channel (8–16 ch) projection of the spatial map; z₂ remains the
dynamics token** (its vehicle-pose content is real and generalizes:
1.6–2.1 m / 3.3–4.4°). The claim is bounded on three axes:

- **Capacity:** 512-D / 16-query z₂ changed nothing (v4).
- **Data:** layouts 2.2k → 7.2k lifted z₂ val BEV 0.231 → 0.327 and collapsed
  its memorization gap — z₂ scales, but toward an apparent ~0.4 ceiling while
  the spatial path sits at 0.88 (v5; v6 adds a third scaling point at 9518
  layouts with full_v4 partially landed).
- **Resolution frontier:** the failure is spatial arrangement, not bits — a
  2 048-float 16×16×8 map recovers 0.81 where the 256-float z₂ gets 0.23.

Provisional bars, all met by the spatial path on held-out layouts at v5:
occupancy IoU ≥ 0.8 (achieved 0.878), rock recall ≥ 0.95 (0.992), center
error ≤ 1.5 m mean (0.80), yaw ≤ 8° mean (3.3–4.0).

## Data inventory (schema v1 stores, 20 s / 400 frames / 256² episodes)

| Store | Episodes | Seeds | Disk | Notes |
|---|---|---|---|---|
| smoke_v1 | 10 | 20260910+ | 44 MB | gate iteration set |
| pilot_v1 | 200 | 20261000+ | 873 MB | 0 abnormal |
| full_v1 | 1000 | 20270000+ | 4.3 GB | 1 rollover |
| full_v2 | 1000 | 20280000+ | 4.3 GB | 1 rollover |
| full_v3 | 5000 | 20290000+ | 22 GB | 10 rollovers; see incident below |
| full_v4 | 5000 | 20300000+ | 21.6 GB | complete 09-03 15:50 (4986 + 14 rollovers, 823 contact eps); 40-episode read-back verified; mirrored to luffy; `full_v4_partial` (2335-ep interim sync used by v6) deleted after the full mirror |

Next free seed block: **20310000+**.

## full_v3 incident (fixed in `ebfa0a3`)

Collection workers leak ~2–3 MB/episode (incomplete Chrono/OptiX teardown).
On the 5000-episode run, workers grew to 9–14 GB, the OOM killer took them
near episode ~4600, the pool respawned replacements that ballooned again, and
the main process hung waiting for 6 results lost with killed workers —
4996/5000 episodes were complete on disk. Recovery: killed the tree,
reconstructed `episodes.jsonl` from per-episode metas, re-collected the 4
missing episodes via `--indices`, verified 44 episodes with full read-backs
(0 errors). Fix: `maxtasksperchild=50` recycles workers (~1 GB peak).
Operational note: long-run watchers need ≥3-strike ssh-failure tolerance —
two transient drops false-declared this run finished. Two sessions raced on
this repair in parallel; the deterministic per-episode seeds made the results
identical and the merged store passed 44-episode full read-back verification.

**full_v4 addendum:** recycling fixed the fleet-wide leak, but a rarer mode
remains — on the 5000-episode full_v4 run, two workers wedged (never hit
their 50-task recycle, grew to 6.6/3.7 GB, survived pool exit as orphans)
and halved throughput mid-run (~4.8 eps/min around 12:00) before the pool
routed around them; final wall 6.8 h. All 5000 results were delivered; the
orphans were killed post-run. Suspected Chrono/OptiX teardown deadlock at
worker exit — if it recurs, add a watchdog that SIGKILLs workers whose
current task exceeds ~5 min.
