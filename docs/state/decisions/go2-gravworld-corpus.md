# go2_gravworld: the corpus collected to make ground tilt observable

**Collected:** 2026-09-07 · **Episodes:** 3115 usable of 3520 attempted · **Status:** in use

## Why it exists

Ground tilt in this simulator is implemented by **rotating gravity on flat ground**
(`collect_go2_smoke.py`). Every state preset in use derived its gravity channel
from the stored quaternion as `R^T . [0,0,-1]` — the body-frame direction of world
-Z, which equals gravity only on level ground. On a tilted episode it asserts the
ground is level.

Tilt was randomised per episode across the previous training corpus, so it entered
the surrogate as an **unobserved latent folded into the residual**. Measured on
1,762 episodes of that corpus: the applied pitch is not recoverable from the
logged attitude, `corr = -0.030`.

`grav_world_*_mps2` — the gravity vector *as set* — was added to the logger to fix
this and **no training preset ever consumed it**. The previous corpus
(`go2_comprehensive_merged/flat`, 2026-09-04) predates the columns entirely, so
the state definition could not be built from it. Hence a new collection.

## What differs from the previous corpus

| | previous | this one |
|---|---|---|
| `grav_world_*_mps2` | absent | **logged** |
| ground pitch range | ±1.5° | **±3.0°** |
| ground roll range | ±3.0° | ±3.0° |
| `scenario_family` | `constant_command` for every episode | the commanded family |

**The pitch cap was lifted because its reason was removed.** It existed because
tilt was unlogged and therefore unlearnable; once tilt is a state channel it is an
input rather than a latent. The cap also removed the region where policies differ
most — measured on the previous corpora as 113 discordant episodes above +1.0°,
every one favouring the base controller.

## FOUR SHARDS, FOUR PHYSICS BINARIES

Collected in parallel on four machines with disjoint seed offsets:

| shard | machine | `pychrono/_core.so` md5 |
|---|---|---|
| `off5000000` | a3 | `cfbf8af6` |
| `off6000000` | sliger | `60457362` |
| `off7000000` | sbel | `3b0bd530` |
| `off8000000` | north | `d1d0bd0a` |

**Within-shard comparisons are build-matched. Across-shard ones are not.** Anyone
computing a per-band rate will pool all four, and this note is the only thing that
will say so. The one measurement of the build effect available — same seed, same
checkpoints, two binaries — put it at **one episode in 320** on a coarse
divergence screen, which bounds it as small and does not establish it is zero for
anything sensitive to fine trajectory detail.

## Splits and coverage

Assigned at collection, so every preset preprocessing this corpus inherits the
identical split. **Measured on the merged index, not projected:**

| pitch band | total | val | val % |
|---|---:|---:|---:|
| −3.0 to −2.0 | 342 | 68 | 19.9% |
| −2.0 to −1.0 | 414 | 94 | 22.7% |
| −1.0 to 0.0 | 587 | 119 | 20.3% |
| 0.0 to +1.0 | 610 | **124** | 20.3% |
| +1.0 to +2.0 | 599 | 118 | 19.7% |
| +2.0 to +3.0 | 563 | 115 | 20.4% |

Eight command families, 380–402 episodes each.

## Two things to know before using it

**405 of 3520 attempted episodes produced no index entry.** They wrote a CSV with
0–17 rows: the robot fell at spawn. Genuinely unusable rather than filtered on
outcome, but the exclusion is still outcome-correlated and the corpus is therefore
conditioned on the robot having stood up.

**`scenario_family` was collapsed in the raw collection** and is recovered in the
merged index from the scenario directory, which is authoritative. The source was
fixed at 430702e, *after* this corpus was collected — so the raw shards still
carry the wrong value and only the merged index is correct.

## Merged index

`go2_gravworld_merged/dataset_index.json` references the shard CSVs in place;
nothing is copied. Episode ids are namespaced by scenario directory **and** shard,
because ids collide both across shards and within them — every family's directory
holds a `go2_flat_000`, and namespacing by shard alone silently dropped 679 of 789
episodes on the first attempt.
