# Go2 excitation collection — design, and the two constraints that shaped it

Purpose: the existing dataset is policy data, so at any state the action is ~96%
determined — measured conditional/unconditional action variance 0.079, effective
rank 3.89 of 12. A model trained on it never had to learn how actions affect the
world. This collection samples actions independently of the state.

**It works:** conditional variance 0.079 -> **0.675**, effective rank 3.89 -> **12.00
of 12**. Full rank. That was the question the collection exists to answer.

## Two constraints the design had to work around

**1. State injection is unavailable.** Joint angles and velocities cannot be written
on a `ChLinkMotorRotation` — the angle is a consequence of body placement and the
motor link exposes no setter. So "sample a recorded walking state and put the robot
in it" is not implementable. **The only route onto the walking manifold is through
the dynamics**: run the policy until the robot is walking, then branch to random
targets. Everything else about the design follows from this.

**2. The controller is stateful, so recovery is incomplete.** Between perturbation
bursts the policy pulls the robot back, but joint containment recovers to **0.64–0.69
against a walking ceiling of 0.812 — about 0.79x, not 1.0.** The burst pollutes the
policy's 5-step observation history, so it resumes from a history that never occurred.
This is stable rather than cumulative — bursts 1, 2 and 3 recover to 0.651, 0.639,
0.685 with no downward trend — but it is a real property of exciting a stateful
controller and it will surprise whoever looks next.

## What the numbers mean: measure the ceiling before quoting a containment

With 33 channels at p1–p99, walking data scores **0.812 against itself** on held-out
episodes. So an excitation score of 0.03 is **0.04x the ceiling**, not "3% of some
absolute standard". Quoting the raw figure without the ceiling invites over-reaction.

| population | joint containment | vs ceiling |
|---|---|---|
| walking, held out | 0.812 | 1.00x |
| excitation, first 10 rows of a burst | 0.111 | 0.14x |
| first 20 rows | 0.055 | 0.07x |
| first 40 rows | 0.030 | 0.04x |

## Parameters, derived rather than inherited

| | NeRD (ANYmal) | ours | why |
|---|---|---|---|
| action scale | 0.5 | **0.3** | 1.6 was derived from our own p99.9 and drove joints past the URDF limit in **30 of 30** windows |
| gains | Kp U[30,200], Kd U[0,1] | **fixed 20 / 0.5** | with fixed gains torque is a bijection of the target given the state, so randomising buys an action-space change for nothing |
| action | resulting torque | **12 joint targets** | follows from fixed gains |
| window | 1.67 s | **mixed, 10 and 40 rows** | marked in `phase`/`burst`; the training run decides |
| falls | discarded | **kept** | a surrogate that has never seen a fall cannot penalise one |

**The action scale is the cautionary one.** 1.6 was chosen to cover the policy's own
|target − stand| at p99.9 = 1.53 rad. But the policy's twelve offsets are strongly
correlated and independent draws are not: at 1.6 every window left the physical joint
range. **Matching a correlated distribution's marginals does not match its joint
behaviour** — the same error appears again in containment below.

## Runaway is detected, not inferred from length

In the existing collection, diverged episodes are excluded only because they are too
short to yield a training window — a filter that works for the wrong reason, and one
that stops working here, where falls are kept and divergences run longer. Three tiers,
each counted and reported by reason rather than silently dropped, because the
divergence rate per action scale is the measurement that sizes the sweep:

- **primary** `|q|` beyond 1.5x the URDF range — physical, not a tuned constant
- **secondary** torque pinned at the effort limit for 8 consecutive rows with `|q|` growing
- **backstop** NaN, Inf, `|state| > 1e5` — NeRD's own rule

At scale 0.3 the rejection rate is 0; at 1.6 it is 100% on the primary tier.

## Final arm plan (2026-09-06)

`L` = burst length in rows, `N` = bursts per episode, both arms 340 rows/episode.

| arm | L | N | perturbed rows/ep | rejected | role |
|---|---|---|---|---|---|
| A | 40 | 4 | 160 | 0.0% | **bulk** |
| B | 10 | 16 | 160 | 13.7% | **dropped** |
| C | 10 | 4 | 40 | 0.0% | burst-length comparison |

The original A/B pair held `L x N` constant, so `L` and `N` were perfectly
anti-correlated and completely confounded by construction. **A-versus-C isolates
burst length at matched handover count (N=4)**, which is the comparison that pair
could not make.

Arm B is dropped for **survivorship, not for its rejection rate**: its rejections
are concentrated at the policy handover, so excluding them biases the survivors
toward easy handovers — a survivorship filter operating on the mechanism under
study. Arm A gives the same 160 perturbed rows at the same ~1.03 s/episode with no
rejection and no selection. See [go2-policy-history-provenance.md](go2-policy-history-provenance.md),
which records the handover result as a finding about the study rather than about
the collector.

Runs (`--action-scale 0.3 --branch-from-policy`):

```
go2_exc_b40      seed 21    900 eps   arm A   (complete, 306,000 rows)
go2_exc_b40_c2   seed 31  6,453 eps   arm A
go2_exc_b40_c3   seed 33  3,680 eps   arm A
go2_exc_b40_c4   seed 34  3,680 eps   arm A
go2_exc_c10_c2   seed 35  2,500 eps   arm C   (190,000 rows, comparison only)
```

Arm A total 14,713 episodes = **5.00M rows**. Arm C needs only enough volume for
the comparison, not for the bulk; effective rank saturates by ~2,000 rows and arm C
supplies 100,000 perturbed rows.

Superseded: `go2_exc_b10` (seed 22, arm B, kept) and `go2_exc_b10_c2` (seed 32,
killed ~30 s in, marked `ABANDONED.txt`).

### The A-versus-C comparison must be run at matched perturbed-row counts

Arm A supplies ~2.35M perturbed rows and arm C ~100K. **Subsample A down to C's
100K before comparing.** Run at native volumes the comparison confounds burst
length with training volume — the same error, one axis over, as the original
A/B pair that held `L x N` constant.

This is stated here because the shard sizes make the mismatch invisible to
anyone reading the datasets rather than the plan: nothing in `go2_exc_b40_c*`
or `go2_exc_c10_c2` signals that their row counts are not comparable.
