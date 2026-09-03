# Enforcing the planner's search margin: what it does and does not do

**Measured 2026-09-03 on `kyle-N7-B650E`.** 100 episodes per arm, seeds
20260901-20260999, 24 procs, pychrono 10.0.0 (conda), NeDM commit `103ad0b`.
Unenforced arm 2927.8 s, enforced arm 2975.6 s.

> **Environment caveat.** Both arms ran the **conda 10.0.0 API**. Results from the
> source build cross an API change, not a version bump. See
> [`../../state/machines/chrono-build.md`](../../state/machines/chrono-build.md).

## What was tested

`_shortcut` preserves the 2.60 m search inflation (`inflation_m` 2.0 +
`tracker_p95_margin_m` 0.6), and Chaikin smoothing then erodes it, because
`validate_candidate` accepts on footprint overlap (centreline ≥ 1.3 m) rather
than on the search bound. `NEDM_ENFORCE_INFLATION=1` makes `clearance_ok` test
the search bound instead. The question was what that costs and what it buys.

## The mechanism is selection. Enforcement improves no path it accepts.

**This is structural, and the structure is the evidence.** `validate_candidate`
can only *accept* or *reject*. It cannot re-smooth, re-shortcut, or repair. For a
given layout it either passes the same path through unchanged, or rejects it and
`sample_episode` draws a new layout. **There is no channel by which it improves a
path.**

The arms therefore do not share a population: **40 of 100 layouts differ**, and
layout resampling rises from **33 to 56 of 100**. The entire improvement below
comes from changing which episodes run.

On the 60 episodes whose layout is identical in both arms, driven clearance is
bit-identical, zero of 60 improved. **This is a consistency check, not
independent confirmation** — equal `layout_attempt` means enforcement was
inactive by construction, so the paired zero is entailed by the structural
argument rather than evidence for it. What it *does* establish independently is
that enforcement has **no side channel**: it consumes no RNG and perturbs nothing
when it accepts. Had it done either, the shared-layout paths would have diverged.

**The substitution is invisible in the summary statistics a reader would check.**
Assets per episode 19.92 enforced against 19.78 unenforced. Mean plan length
43.65 m against 45.10 m, and **maximum** plan length *falls*, 63.44 → 59.79 m.
Nothing suggests "it picked sparser maps," which is what makes it dangerous.

## The distribution shift, which measures the selection

Driven-trajectory `min_asset_clearance_m`:

| | min | p05 | p10 | p25 | median |
|---|---|---|---|---|---|
| enforced | 2.086 | 2.555 | 2.612 | 2.924 | 3.925 |
| unenforced | 1.073 | 1.348 | 1.711 | 2.249 | 2.710 |

Mann-Whitney U=6895, z=4.630, two-sided **p=3.65e-06**. Cliff's δ **+0.379**.
Bootstrap median difference **+1.216 m**, 95% CI [+0.396, +1.712]; p05 **+1.207 m**
[+0.725, +1.434]; p10 **+0.901 m** [+0.706, +1.265].

Fraction finishing below the 2.60 m bound the search was run under:

| | below 2.60 m | 95% CI |
|---|---|---|
| enforced | **9/100 = 9.0%** | [4.8%, 16.2%] |
| unenforced | **45/100 = 45.0%** | [35.6%, 54.8%] |
| difference | **−36.0 pp** | [−47.0, −25.0] |

**Consequence for the gate.** `gate_G0a_pass` is `true` under enforcement and
`false` without. That is a true statement about **two different episode
populations**. Wherever the enforced pass is quoted it needs the qualifier: it
passes partly because 56% of layouts were rerolled until they admitted a
compliant plan. *A gate that discards the episodes it finds hard does not measure
what one that keeps them measures.*

## Two fixes are needed, and neither is sufficient alone

1. **Enforcement is the wrong lever for path quality.** Its only action is
   refusal. Making smoothing clearance-aware — re-running `_shortcut` against
   `_segment_valid` after Chaikin, or repairing the violated span — is what would
   change delivered paths on the layouts we want to evaluate, and what would make
   the arms comparable.
2. **The guarantee is stated in the wrong space.** 9 of 100 enforced episodes
   finish below 2.60 m *driven* despite plan compliance, as do 5 of the 60
   shared-layout episodes. 2.60 m is a planning-space quantity evaluated in
   driven space; tracking error erodes it afterwards and no plan-side enforcement
   reconciles that.

**Whether the guarantee should hold on the plan or on the driven trajectory is a
design decision, not an implementation one, and it needs deciding before either
fix is written.**

## On `inflation_m` = 2.0

45% of driven episodes violate a bound the arena can satisfy only by rerolling
half its layouts. That is evidence 2.60 m is more corridor than this arena
reliably provides, and makes lowering `inflation_m` a data-supported option
rather than a matter of taste.

## Method notes

**Pre-registered before the enforced arm existed.** Mann-Whitney at n=100/arm
detects a **0.40 m** shift at 80% power (2000 sims/point resampling the observed
distribution; the zero-shift row returned 0.05, confirming calibration). Observed
shift +1.216 m, well clear.

**The collision count is a footnote only** — 0/100 enforced against 1/100
unenforced — because it cannot answer the question: at a 1/100 baseline,
P(0/100 | enforcement does nothing) = 0.366, so a clean zero is the single most
likely null outcome. *Consistent with enforcement working, and underpowered to
demonstrate it.* Episodes below the 1.10 m hull half-width: 0 against 1.

**Statistics implemented directly** (no scipy in `nedm`) and validated by scoring
the unenforced arm against itself: U=5000, z=0.000, p=1, δ=0.000, bootstrap CI
[−0.524, +0.529] — the degenerate answer, which also gives the instrument's
resolution.

**`min_asset_clearance_m` verified against its implementation**
(`scripts/traverse_wp0a_gate.py:141-146`), since the whole result rests on it. It
is a per-step minimum over the **driven** trajectory of
`hypot(pos − asset) − footprint_radius_m`, sampled every step rather than every
`record_every`, so it is a true trajectory minimum. **It is measured from the
chassis reference point, not the hull boundary** — centre-to-edge. That is the
correct comparison against 2.60 m, which is an inflation applied around a planned
centreline, and it makes the 1.10 m hull half-width row the right test for actual
overlap.
