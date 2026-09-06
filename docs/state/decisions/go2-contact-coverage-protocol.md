# Contact-coverage collection protocol

**Designed 2026-09-05, BEFORE the mode histogram was available, and deliberately so.**
A protocol written after seeing which of the 16 contact modes came back sparse is
fitted to that sample: it would over-represent exactly the modes we happened to miss
and under-represent the ones we happened to catch, and its coverage claim would then
be circular. **I have not computed the mode histogram on this half.** dorm-pc's
measurement over the merged 3,503 decides whether any of this runs at all.

Purpose: the surrogate is trustworthy to ~0.1 s against the HMMWV's 10 s, and the
diagnosed cause is that a walking quadruped is a **contact-mode-switching** system —
four feet making and breaking contact ~4x per gait cycle, so one continuous
transition model must fit a function with repeated discontinuities and can only
average across them. Conditioning the transition on contact mode requires the dataset
to contain the modes.

## Two measurements that constrain the design

### Torque perturbation has NEVER been exercised

`perturb_torque_x/y/z_nm` are recorded and are **identically zero in 210 of 210
episodes sampled**. The collector applies `[mag*cos(th), mag*sin(th), mag*U(-0.3,0.3),
0, 0, 0]` — a horizontal push with a small vertical component and **no torque at
all**. So tool 2 is not an adjustment to an existing axis; it is an axis that has
never been used, which is why it is ranked first below.

### The perturbation ceiling is NOT where we are

Loss rate against commanded peak, over all 2,000 attempted episodes:

```
    0 N  10.4%      72 N  14.3%
   24 N  13.1%      96 N  13.4%
   48 N  11.0%     120 N   9.1%      slope -0.0025 %-loss per N
```

**Flat, and 120 N loses less than 0 N.** The ~11% baseline loss is solver divergence
unrelated to pushing: it is a floor, not a perturbation cost. **Raising perturbation
strength is free in yield up to at least 120 N**, and the "12-13% loss at current
settings" figure should not be read as evidence we are near a ceiling. Where the knee
actually is remains unmeasured — see the bracketing probe below.

## The four tools, and what each is FOR

A trot alternates two diagonal pairs, FL+RR and FR+RL. Everything below is described
by which contact configurations it reaches **that a trot does not**.

### 1. Stronger linear perturbations — lateral pairs and flight

A horizontal impulse accelerates the body sideways; the legs must step out to catch
it. **Target configurations:** the two LATERAL pairs (FL+RL, FR+RR), which a straight
trot never uses, plus extended single-support (1 foot) while stepping out, and at
large magnitude a transient 0-foot flight phase as the body is thrown.
**Vary:** peak magnitude past 120 N, and direction, with lateral pushes weighted
because they are what force a lateral catch.
**Check it worked:** lateral-pair and 1-foot dwell time per episode rises with
magnitude. If it does not, the pushes are being absorbed without changing footfall
and the tool is not doing its job.

### 2. Torque perturbations — the front and rear pairs, reachable no other way

A pure torque rotates the body without translating the centre of mass. A **pitch**
impulse unloads the front or rear pair together, producing FL+FR or RL+RR — **the
configurations a trot structurally never visits**, because it is always diagonal. A
**roll** impulse unloads one entire side, producing lateral pairs and single-foot
support on the loaded side.
**This is the highest-information tool**: it is unexercised, and it is the only one
that reaches front/rear pairs by design rather than by accident.
**Vary:** axis (roll, pitch, yaw), magnitude, sign.
**Check it worked:** nonzero occupancy of FL+FR and RL+RR, which should be near-absent
without it. That is a clean before/after with a reachable failure.

### 3. Phase-timed perturbations — the same mode entered differently

Applied at a known point in the gait cycle rather than at a random instant. A push
during swing extends single support and can force an early touchdown at an unusual
foot position; the same push during double support loads both feet and can cause a
slip instead.
**This one is not primarily about new modes.** It is about the same mode entered from
different phases, which is exactly what a mode-conditioned transition model needs in
order to distinguish "3 feet down, descending into stance" from "3 feet down, rising
out of it". Marginal mode coverage cannot show that; transition-conditioned coverage
can.
**Feasible now:** `foot_*_in_contact` gives the gait phase directly, so the trigger
needs no new instrumentation.
**Check it worked:** for a fixed mode, the distribution of successor modes differs
between swing-triggered and stance-triggered episodes. If the successor distributions
are identical, the timing is not buying anything.

### 4. Terrain roughness — sustained rather than transient asymmetry

Uneven ground makes feet touch down at different heights, so asymmetric support is
the steady state rather than a recovery transient.
**The distinction that matters:** perturbation-induced modes are TRANSIENT — the
robot re-enters its trot within a step or two — so they are sampled briefly and
always in a recovery context. Terrain-induced modes are SUSTAINED, giving dwell times
and mode-sequence statistics that impulses cannot produce at any magnitude.
**Check it worked:** mean dwell time in non-trot modes rises, distinguishing it from
tools 1-3 which should raise occupancy while leaving dwell short.

## Ceiling: bracket it before committing a full run

Yield is flat to 120 N, so the knee is unmeasured and above. Escalating probe, one
family, ~24 episodes per level:

```
  120 N (control, expect ~10%)  ->  200  ->  320  ->  500  ->  800
  stop at the first level whose loss exceeds 40%
```

Cost is ~120 episodes, well under an hour at 16.5 s each with 8 workers. **A protocol
that loses 80% of its episodes is not cheaper than a longer one losing 15%**, and the
same probe applies to torque magnitude, for which no prior exists at all — that one
must be bracketed from zero rather than extrapolated from the force numbers, since
the units and the failure mechanism both differ.

## Ordering, if the histogram says collection is needed

1. **Torque bracketing probe**, because the axis is unused and its safe range is
   entirely unknown.
2. **Torque collection** — highest information per episode, reaches modes nothing
   else reaches.
3. **Phase-timed**, which needs no new mechanism and buys transition-conditioned
   coverage rather than marginal coverage.
4. **Stronger linear**, cheap but partly redundant with what 120 N already produces.
5. **Terrain roughness** last: it changes the plant rather than the excitation, so
   episodes from it are not drawn from the same distribution as the existing 3,503
   and cannot simply be pooled with them.

That last point is a constraint rather than a preference. Tools 1-3 disturb the same
plant and their episodes pool with the existing set. Tool 4 produces a **different
plant**, and mixing it in silently would repeat the rigid/soil error of treating two
distributions as one.

## Measured: torque implemented and bracketed, 2026-09-05

The channel did not exist. `perturb_torque_*` was recorded but never driven — the
collector applied `[Fx, Fy, Fz, 0, 0, 0]`. Added `--perturb-torque-peak-nm`, wired to
`AccumulateTorque`, direction drawn in the roll-pitch plane because that is where the
target modes live; yaw gets a small share only, since spinning the trunk does not
change which feet carry load.

Torque only, linear push disabled to isolate it, 8 episodes per level:

| torque | kept | fell | FRONT | REAR | LEFT | RIGHT |
|---|---|---|---|---|---|---|
| 0 N·m | 8/8 | 0 | 0.00% | 0.00% | 0.00% | 0.00% |
| 5 | 8/8 | 0 | 0.00% | 0.00% | 0.00% | 0.00% |
| 10 | 8/8 | 0 | 0.00% | 0.00% | 0.00% | 0.00% |
| 20 | 8/8 | 0 | 0.01% | 0.00% | 0.05% | 0.00% |
| **40** | 8/8 | 0 | **0.30%** | **0.10%** | **0.30%** | **0.55%** |
| existing 3,503 | — | — | 0.91% | 0.06% | 0.38% | 0.09% |

Yield is 100% to 40 N·m — no divergence cost — and a single 80 N·m test toppled the
robot at 5.6 s, so the usable ceiling is between 40 and 80.

**It does not fill all four, and the protocol should not claim it does.** RIGHT gains
6.1x and REAR 1.7x — the two thinnest modes — LEFT is comparable, and **FRONT comes
out WORSE than the existing dataset**. Torque is the right tool for modes 10 and 12
specifically.

Caveats kept in view: this probe is constant-family backward-only against a 3,503 set
spanning eight families, so the enrichment factors are indicative rather than matched;
and REAR at 0.10% of ~30k frames is about 30 frames per level.

### What the coverage buys, stated so it is not credited with the wrong effect

The eight common modes cover 97.3% of transitions, so **the rollout horizon is limited
by prediction error in modes we already have, not by the rare ones.** Filling the
lateral modes does not extend the horizon. What it does is close exploitation holes: a
model that has never seen a configuration produces unconstrained predictions there,
and that is exactly where a policy finds exploits — three fine-tunes have already
exploited this surrogate. A policy stumbling into a lateral-pair contact currently
meets a model with 1,991 samples and no reason to be right.

### A limitation of the histogram that justified this collection

**A marginal mode histogram cannot show phase structure.** It counts occupancy, not
how a mode was entered, so it cannot distinguish "three feet down, descending into
stance" from "three feet down, rising out of it" — which is precisely what a
mode-conditioned transition model needs. Tool 3 targets that, and no marginal count
can measure whether it is needed.


## The FRONT deficit was command-direction, not a limit of torque

FRONT pair means the REAR feet are up, i.e. a nose-down attitude, and the bracket
above was run backward-only. Re-running the 40 N·m level with forward commands, 8
episodes, everything else identical:

| command | kept | fell | FRONT | REAR | LEFT | RIGHT |
|---|---|---|---|---|---|---|
| backward 40 N·m | 8/8 | 0 | 0.30% | **0.10%** | 0.30% | 0.55% |
| forward 40 N·m | 8/8 | 0 | **1.15%** | 0.07% | 0.44% | 0.50% |
| existing 3,503 | | | 0.91% | 0.06% | 0.38% | 0.09% |

**FRONT rises 3.8x with forward commands and overtakes the existing dataset.** So the
earlier "torque does not help FRONT" was an artifact of a backward-only probe, and the
protocol needs BOTH command directions rather than a different tool.

The asymmetry is not symmetric, which is the useful part:

- **FRONT** wants forward commands (1.15% vs 0.30%)
- **REAR** wants backward (0.10% vs 0.07%) and is the ONLY mode that prefers it
- **LEFT and RIGHT** are largely direction-independent (0.44/0.30 and 0.50/0.55)

So a run should be split across both directions, weighted toward forward, with a
backward share retained specifically for REAR.

**REAR remains the hard case and torque does not solve it.** Its best observed rate is
0.10% against an existing 0.06% — 1.7x, the weakest enrichment of the four, and it is
the thinnest mode in the dataset. Nose-up means rearing, which the policy actively
resists. If REAR coverage matters, it likely needs a different mechanism than a trunk
torque impulse, and that should be established rather than assumed to fall out of a
longer run.

## REAR is structurally near-absent during locomotion, not under-collected

Applying "a caveat you can measure is not a caveat" to my own claim that REAR
"probably needs a different mechanism". The candidate was a gravity tilt: a SUSTAINED
rearward weight shift rather than an impulse the policy can resist and recover from.
Tested rather than assumed.

**Steep tilt reaches REAR, but only by toppling the robot.** Forward commands, 40 N·m
torque, 4 episodes per level:

| pitch | kept | med rows | REAR | air (mode 0) | fall time |
|---|---|---|---|---|---|
| −20° | 2/4 | 545 | 3.17% | 55.6% | 0.6 s |
| −10° | 2/4 | 148 | 4.40% | 56.1% | 0.8 s |
| +10° | 0/4 | 115 | — | — | 0.9 s |
| +20° | 3/4 | 1905 | 8.84% | 26.7% | 0.6 s |

**Every episode at every level fell within 0.6–0.9 s.** So the 50–150x REAR
enrichment is the collapse trajectory, not locomotion with weight shifted rearward.

**And there is no window.** Bracketing the walkable range, 6 episodes per level:

| pitch | kept | med rows | REAR |
|---|---|---|---|
| **+3°** | 5/6 | **3755** (full episodes) | **0.03%** |
| +5° | 0/6 | 53 | — |
| +7° | 1/6 | 41 | — |
| +9° | 0/6 | 56 | — |

At the steepest slope the policy can actually walk, REAR is **0.03%** — below the flat
control's 0.07%. The gait collapses between 3° and 5°, sharply. There is no tilt at
which the robot both walks and rears.

**Conclusion, stated as the finding it is rather than the gap it is not: for this
policy on this plant, the REAR configuration does not occur during locomotion.** It
occurs while falling. Torque gets 1.7x, tilt gets nothing without destroying the gait,
and 1,991 frames out of 3.39M is what the system produces, not what we failed to
collect. A contact-conditioned model will not have this mode, and that is correct
rather than a deficiency to close.

Caveats: 6 episodes per level; pitch only, not combined with roll; and the +7° row's
apparent 12.46% FRONT comes from a single 41-row fall and is noise, not signal.

**What this changes:** REAR drops out of the collection target. The episodes-over-
magnitude arithmetic that made RIGHT worth pursuing at 0.55% does not apply to a mode
that does not occur, so no episode count reaches it.

## Lateral pairs are on the path INTO a fall — and most "falls" are not falls

Measured on existing data, no collection. For each fall episode, contact modes in the
final 1 s before `fell_at_s` against all of that same episode's history before
`fell_at_s − 2 s` — a within-episode control, so command, spawn and terrain are held.

| mode | steady (>2 s before) | final 1 s | ratio |
|---|---|---|---|
| FRONT | 1.00% | 4.90% | 4.9x |
| **REAR** | 0.01% | 5.90% | **664x** |
| LEFT | 0.16% | 0.60% | 3.9x |
| **RIGHT** | 0.16% | 2.30% | **14x** |
| **all lateral** | **1.33%** | **13.70%** | **10.3x** |
| all four down | 66.04% | 27.30% | 0.4x |
| diagonal fl+rr | 4.25% | 0.70% | 0.2x |

**Losing balance is mediated by lateral contact.** The diagonals collapse, four-foot
support halves, and the lateral pairs rise ten-fold in the second before the fall.
That answers the question the histogram could not: these modes are not merely rare,
they sit on the trajectory into the failure every fine-tune has exhibited.

It also confirms the REAR result from the other direction — REAR occurs at 664x its
steady rate while falling, and essentially never otherwise.

### The 428 fall episodes are overwhelmingly failures to STAND, not falls

Only 10 episodes were usable, and the reason is the finding:

```
  fell_at_s   min 1.20   median 1.39   p90 1.78   max 40.63
  falls after t=2s:  26/428  (6%)
  falls after t=3s:  12/428  (3%)
  falls after t=5s:  10/428  (2%)
```

**94% of falls occur before t = 2 s**, against a recording start of
`warmup_s = pose_ramp + settle + prewalk` = 1.25–4.25 s. Those episodes are robots
that failed to stand during the ramp, not robots that lost balance while walking. It
explains dorm-pc's finding that fall episodes are 89% mode 0: a robot that collapsed
during the ramp lies still for the whole recorded episode.

**So n = 10 is not a 2% subsample of 428 — it is nearly the complete population of
genuine locomotion falls** (10 of the 12 that fall after 3 s). Small in absolute
terms, but not a biased slice, which is a different and much weaker caveat.

**Consequence:** the hope that 435 fall episodes already supply contact diversity is
wrong. They supply stand-up failures. Genuine loss-of-balance data is ~10 episodes in
1,762, and that — not the marginal histogram — is the argument for collecting RIGHT.

## The 24% stand-up failure rate is ground PITCH, and it is fixable

428 of 1,762 episodes carry `fell: True` with a median `fell_at_s` of 1.39 s, i.e.
0.14 s after the ramp+settle ends. Diagnosed rather than assumed.

**Not the perturbation.** Fall rate against commanded peak is flat and the
zero-perturbation episodes fall MOST: 28.2% at 0 N against 24.8% at 120 N, slope
+0.007 %/N. (The gate `t > warmup_s * 0.5` does let a push land mid-ramp and should
still be tightened, but it is not the cause.)

**It is ground tilt, and specifically PITCH.** Isolated by experiment, then confirmed
on the full collection:

```
  no tilt, otherwise identical    0/16 fell
  random +/-3 deg roll and pitch  3/16 fell (19%), median fell_at_s 1.39
  collection at 0 N               28.2% fell,      median fell_at_s 1.37
```

| band | \|pitch\| fell | \|roll\| fell |
|---|---|---|
| 0.00–0.75° | **2%** | 21% |
| 0.75–1.50° | 17% | 21% |
| 1.50–2.25° | 38% | 29% |
| 2.25–3.00° | **48%** | 26% |

`corr(|pitch|, fell) = +0.427`; `corr(|roll|, fell) = +0.057`. **Pitch is monotone
and steep; roll is flat.** The driver draws both independently at U(−3,3), so the
combined magnitude reaches 4.24° — inside the 3–5° band where the walkable-slope
bracket above found the gait collapses. Two measurements taken for unrelated reasons
agree.

### The fix

| pitch cap | episodes kept | fall rate |
|---|---|---|
| none (current) | 100% | 24% |
| \|p\| ≤ 2.0° | 70% | 15% |
| **\|p\| ≤ 1.5°** | 55% | **9%** |
| \|p\| ≤ 1.0° | 38% | 3% |

**Cap pitch at 1.5°, leave roll at ±3°.** Capping the COMBINED magnitude — the
obvious first instinct — would sacrifice roll range for no benefit, since roll does
not drive falls. One line in `drive_go2_collection.py`.

**Not applied.** Changing the tilt range alters what a future collection samples and
is a deliberate decision, not something to do while diagnosing.

### Separate live bug, found while looking

`src/nedm/quadruped/policy.py:175` computes `action * 0.25 + POLICY_DEFAULTS`, and
`POLICY_DEFAULTS` has all four hips at **0.0** where upstream has FL/RL +0.1 and
FR/RR −0.1. The collector is unaffected — it uses `imported_policy.py`, whose
`IMPORTED_DEFAULTS` are correct — but anything on the `policy.py` path applies a
0.1 rad hip offset on all four legs.

### Applied: pitch capped at 1.5°, roll unchanged — and what it costs

`drive_go2_collection.py` now draws `pitch ~ U(−1.5, 1.5)` with `roll ~ U(−3, 3)`
unchanged.

**The 11.9% "solver-divergence floor" was not a floor.** It has been quoted in this
project as irreducible and unrelated to perturbation. The first half is true and the
second is misleading: it is unrelated to *pushing* but strongly related to *tilt*.

| \|pitch\| | episodes | lost entirely |
|---|---|---|
| 0.00–0.75° | 516 | **0.6%** |
| 0.75–1.50° | 504 | 7.9% |
| 1.50–2.25° | 469 | 17.3% |
| 2.25–3.00° | 511 | **22.3%** |

`corr(|pitch|, lost) = +0.265`. Under the cap, loss falls from **11.9% to 4.2%**.

**Combined effect on usable yield**, where "usable" means an episode that was kept AND
actually walked:

```
  before   2000 x 0.881 kept x 0.76 walking = 1339   67%
  after    2000 x 0.958 kept x 0.91 walking = 1743   87%
```

**The diversity cost, stated rather than left to be inferred from a changed
constant.** The cap removes the 1.5–3.0° pitch band from future collections
entirely. Most of what is lost is failures — that band runs 38–48% falls and
17–22% total loss — but not all of it, and this is a real narrowing of the terrain
distribution in exchange for yield. Future data will contain no episode on a pitch
steeper than 1.5°, and any claim about slope robustness must say so.

**Why 1.5 rather than 2.0:** at the cap the fall rate is 9%, approaching what remains
after tilt is accounted for, so further capping buys progressively less while costing
diversity linearly. 2.0° would leave 15%.

**Existing data is unaffected** — the 3,503 already collected keep their original
distribution, and merging future episodes with them mixes two tilt distributions.
That must be recorded wherever the two are pooled, for the same reason rigid and soil
cannot be pooled silently.

### Correction: the tilt is NOT observable in the state

An earlier reading held that mixing two tilt distributions is milder than the
rigid/soil case, because tilt is in the state and the model can condition on it.
**Measured, it is not:**

```
  corr(applied roll,  mean recorded roll_rad)  = +0.294   weak
  corr(applied pitch, mean recorded pitch_rad) = -0.030   nothing
  gravity channels in this half: NONE
```

Applied **pitch** — the axis just capped — is not recoverable from the recorded body
attitude. The cause is the implementation: tilt is applied by **rotating gravity** on
flat ground, while `roll_rad` and `pitch_rad` are measured against world z. The robot
leans to balance, but that lean is confounded with gait phase, command and
acceleration, so it does not track the applied slope.

**So this is the same shape as rigid/soil, not a coverage caveat.** Two episodes can
carry identical recorded state and different accelerations, because the gravity
direction that produced them is absent from the state.

**And it propagates to gravity channels derived post-hoc from the quaternion.** That
computes `R^T · [0,0,−1]` — the world-z axis in body frame — not the true gravity
direction, which under tilted gravity is rotated away from world-z:

| applied pitch | derived vs true gravity | body-x component |
|---|---|---|
| 1.5° | differ by 1.5° | +0.026 vs +0.000 assumed |
| 3.0° | differ by 3.0° | +0.052 vs +0.000 assumed |

On every tilted episode the derived channel is wrong by exactly the tilt, and wrong in
the direction of asserting the ground is level. A 3° tilt is **0.51 m/s²** of
horizontal acceleration the state does not carry, and roughly half the existing 3,503
have |pitch| above 1.5°.

**Proposed, not done: log the applied gravity vector as three columns.** The collector
sets it, so this records a known quantity rather than estimating one. It would make
tilt observable, make future pooling genuinely a coverage question, and make the
derived channels unnecessary — removing the need to filter the existing 3,503 down to
match. It changes the schema, so both machines agree the column set before either
collects again.

## STOOD DOWN 2026-09-05 — and the mechanism is UNTESTED, not falsified

The targeted collection is not running. Contact conditioning worked decisively
without it: correlation at 0.5 s went from 0.181 to 0.876, off the declared
trade-off curve, the first surrogate to pass the gate at that horizon.

**The stated reason for standing down was that the mechanism is falsified. It is
not, and the distinction matters because one of these is a finding.** dorm-pc
measured the CONDITIONED model at corr 0.859 on windows containing a contact
transition against 0.934 on windows without — real, and distinguishable from equal
(Fisher z = +2.19, two-sided p = 0.029). But that is one model's absolute
performance across two window types. The mechanism is a claim about the
**difference between two models** in each window:

```
  measured   conditioned model: 0.859 with a transition, 0.934 without
  mechanism  conditioning should help MORE at transitions than away from them
```

A conditioned model worse at transitions is expected whether or not conditioning
helped there, because transition windows are intrinsically harder — which is what
the discontinuity story asserts in the first place. If the UNCONDITIONED model
scored 0.10 at transitions and 0.90 without, conditioning improved transitions by
0.76 and non-transitions by 0.03, the mechanism would be strongly supported, and
the reported numbers would be unchanged. The split cannot separate those worlds.

**So the collection is stood down on cost against expected value — a defensible
call — and NOT on a falsified mechanism.**

Testing it needs the unconditioned model on the same two window sets. The pre-fall
test (`prefall_prediction_test.py`) is exactly that comparison, a ratio of two
models' errors per window, and was designed that way before this arose.

### What the collection work produced

| | |
|---|---|
| pitch-tilt diagnosis and cap | +20 points of usable yield on every future collection |
| torque perturbation | a recorded channel that had never been driven, now working and bracketed |
| REAR characterised | structurally unreachable during locomotion; not a coverage gap |
| the "11% floor" corrected | a number quoted four times, wrong in kind |
| lateral modes on the fall path | 1.33% -> 13.70% in the final second — **this measurement stands** |

**The last row is not retracted by the stand-down.** Lateral contact does lie on the
path into a fall. What is open is whether conditioning helps *by handling those
transitions*. Both can be true: the modes precede falls, and the model's improvement
comes from somewhere else.

### The pre-fall test measures the axis conditioning got WORSE on

dorm-pc reports the conditioned surrogate is roughly **2x worse at absolute open-loop
prediction** while far better at action-response correlation — more accurate about how
the state responds to actions, less accurate about the state itself. The gate scores
the first, which is why it passes.

`prefall_prediction_test.py` scores **absolute open-loop error** on body velocity. So
as declared it measures the axis on which conditioning regressed, and would show
conditioning "helping less" in both windows for a reason unrelated to the mechanism.

**This is a defect in the declared instrument, and it is not being fixed quietly.**
Changing a declared bar after seeing related results is what pre-registration exists
to prevent. The two honest options are:

1. **Amend the declaration explicitly and now**, before any result, stating that the
   metric changes because of what was learned about the MODEL rather than about the
   ANSWER, with the correlation-based bar and its threshold declared before it runs.
2. Run it as declared and report that it tests the wrong axis.

Preference is (1), recorded as an amendment rather than a silent substitution. Either
way the change and its reason are on the record before the result exists.

## AMENDMENT 1 to the pre-fall test — declared 2026-09-05, before any result

**The metric changes from absolute open-loop error to action-response correlation.**
Recorded as an amendment rather than substituted silently, with the old bar, the new
bar, and the reason, so it can be judged rather than taken on trust.

**Old bar.** Per episode, `gap = r_steady − r_prefall` where `r = err_conditioned /
err_unconditioned` on body velocity, absolute open-loop error. Bar: median gap > 0.10
with an exact 95% CI excluding 0.

**Why it is invalid.** The conditioned surrogate is roughly **2x worse at absolute
open-loop prediction** while far better at action-response correlation — more accurate
about how state responds to actions, less about the state itself. The declared metric
therefore scores the axis the treatment regressed on, and would report conditioning
"helping less" in both windows for a reason unrelated to the mechanism.

**Why this is a legitimate amendment and not fitting the instrument to the answer.**
The prompting fact is about the MODEL'S BEHAVIOUR, reported by dorm-pc as a caution
against its own result, and learned independently of what the pre-fall test would
say. No pre-fall number exists. The distinguishing test is what the new information is
about: a fact about the system can justify an amendment, a fact about the outcome
cannot. "The metric was wrong" is also what someone says after seeing a result they
dislike, which is why this is written down before rather than after.

**New bar.** Per episode, with `improvement_w = corr_conditioned(w) −
corr_unconditioned(w)` in window `w`:

```
  gap = improvement_prefall − improvement_steady
  BAR: median gap > 0.10 with an exact 95% order-statistic CI excluding 0
```

Same 0.10, now in correlation units. The five verdict branches are unchanged —
SUPPORTED, PRESENT BUT BELOW BAR, CONTRADICTED, UNSUPPORTED (uniform), INCONCLUSIVE —
and the INCONCLUSIVE branch still exists because at n = 6 val the interval spans the
data range.

**Note the structural change.** The old bar used a RATIO of errors; the new one uses a
DIFFERENCE of correlations, because correlations are already on a bounded common scale
and a ratio of them is not meaningful near 1. The gap is still a difference of
differences, so the within-episode design is unchanged.

**Blocked on one thing:** the exact `corr` computation must match the gate's, or this
repeats the specification-mismatch class six times over. Requested from dorm-pc rather
than guessed; the bar above is definition-independent in form and will be implemented
against its stated computation.

## AMENDMENT 2 to the pre-fall test — declared 2026-09-05, still before any result

**The interval construction changes from order statistics to a percentile bootstrap
over episodes.** Forced by Amendment 1, found before implementation, and recorded
separately rather than folded into it.

**Why.** The original metric produced one number PER EPISODE — an error ratio computed
within each episode — so a median across 21 episodes with an order-statistic CI was
well defined. **A correlation is computed ACROSS a set of windows, not within an
episode**, so the amended quantity yields exactly one number per window-set:

```
  improvement_prefall = corr_cond(prefall) - corr_uncond(prefall)   ONE number
  improvement_steady  = corr_cond(steady)  - corr_uncond(steady)    ONE number
  gap                 = their difference                            ONE number
```

There is no sample to take a median of and no order statistics to build an interval
from. The median threshold and the k-th order-statistic CI both assumed a per-episode
sample the amended quantity does not produce.

**The fix.** Resample the episodes with replacement, recompute all four correlations
and the gap on each draw, and take a percentile interval from the bootstrap
distribution.

| | |
|---|---|
| kept | the difference-of-differences structure |
| kept | the 0.10 threshold and all five verdict branches |
| kept | the within-episode window design and the excluded stand-up collapses |
| changed | interval construction only — percentile bootstrap, not order statistics |

**The INCONCLUSIVE branch is what this most has to preserve.** At 21 episodes the
bootstrap interval will be wide, and a wide interval containing zero must still report
INCONCLUSIVE rather than uniform.

**Degenerate draws are counted and reported.** With 21 episodes, some resamples will
produce too few windows of one type or a near-constant series, and a correlation on
those is undefined or meaningless. Those draws are discarded and **the count and
reason are printed**. A bootstrap that silently drops a third of its draws is not the
interval it claims to be — the same vacuity trap in a new place.

**An alternative was available and rejected.** Each episode contributes ~20 pre-fall
anchors, enough to compute a per-episode correlation and keep the original
order-statistic machinery intact. That was not taken because a per-episode correlation
is a DIFFERENT QUANTITY from the pooled-across-windows correlation the gate scores,
and matching the gate's quantity is the entire reason for Amendment 1. Preserving the
old statistics would have quietly undone the amendment's purpose.

### Calibration of the bootstrap: what 40 nulls do and do not establish

Forty independent synthetic nulls, gap constructed to be exactly zero:

```
  CI excluded 0 in 1 of 40   point estimates mean +0.0034, sd 0.0385
```

**This rules out gross miscalibration. It does NOT establish that the test is
conservative.** The Wilson 95% interval on 1/40 runs from **0.4% to 12.9%**, so the
nominal 5% sits comfortably inside it — as do 2.5% and 10%. Forty draws cannot
separate those, and "calibrated, slightly conservative" claims a precision this
measurement does not have. Distinguishing 2.5% from 5% at any useful power needs
hundreds of draws, not forty.

**One of the forty came back at +0.093 with a CI excluding zero, on a true null.**
That is what a low false-positive rate looks like from the inside, and it is the
argument for running forty rather than trusting the first — a single clean-looking
run is exactly what a correctly-calibrated test produces most of the time and an
overfitted one produces always.

## The pre-fall test is NOT RUN. The question is answered by composition instead.

**Decided 2026-09-05, before any pre-fall result and with none ever computed.**

### Why it cannot run

The gate's metric is a **paired counterfactual**: `d_chrono = SB − SA`, Chrono run
twice from the same state under two action sequences, correlated against the model's
`d_model = MB − MA`. It measures whether the model predicts the state change *caused
by changing the actions*.

`prefall_prediction_test.py` does **open-loop prediction against recorded truth** and
cannot produce SA and SB. Computing the gate's metric on pre-fall windows would need
Chrono branched from an arbitrary recorded row — injecting base pose, base velocity,
joint positions and joint velocities — and no such capability exists in this repo.

**Amendment 1 committed an open-loop instrument to computing a paired-counterfactual
metric, and neither machine noticed they were incompatible.** Found by asking dorm-pc
for the exact computation instead of implementing something that resembled it.

### What answers the question instead

```
  1. pre-fall windows are transition-rich    lateral modes 1.33% -> 13.70% in the
                                             final second before a fall (sbel-pc)
  2. conditioning helps at transitions       +0.793 with a transition, -0.015
                                             without, n=200 windows (dorm-pc)
  => conditioning helps pre-fall             BY COMPOSITION
```

**This is a derivation from two direct measurements, not a direct measurement, and it
must be stated that way.** The composition carries an assumption neither measurement
tests: that transitions in the second before a fall behave like transitions
generally. They may not — a fall may involve several feet changing at once, or
transitions into modes that are rare elsewhere. Nothing here rules that out.

### Options rejected, with why

**(b) run it on dorm-pc's half with its apparatus** — n=11 episodes, 3 val. At that n
the bootstrap interval spans most of the plausible range and INCONCLUSIVE was the
near-certain outcome before any of this. It buys an unresolvable version of the
question, not the question.

**(a) build the branching apparatus here — AVAILABLE AND UNSCHEDULED.** Chrono state
injection at an arbitrary row, **validated against a replay proving the branch
reproduces the recorded continuation before any difference is trusted**. Days, not
hours, and it duplicates apparatus dorm-pc already has. If the fall-specific question
later matters enough, this is the honest way to get it and should be scheduled rather
than improvised.

Taking (c) is not choosing less evidence over more: the cheap version cannot resolve
and the expensive version duplicates existing apparatus.

## The composition's gap has a mechanism, and it is the collection's argument again

The derivation above assumes pre-fall transitions behave like transitions generally.
They measurably do not — REAR occurs at 664x its steady rate before a fall and
essentially never otherwise. **The mechanism that makes this matter:**

```
  conditioning helps at transitions    because the model can condition on the mode
  it helps LESS at RARE modes          because it has barely seen them
                                       REAR is 1,991 frames of 3.39M; all four
                                       lateral pairs are under 1%
  pre-fall windows are rare-mode-rich  that is where those modes occur
  => the composition is weakest exactly where it matters most
```

**That is the collection's justification arriving from a different direction** — not
"fill the histogram" but "the mechanism we confirmed may not extend to the regime we
care about, because the data does not." The two arguments are one and should be read
together.

**It does not change the ordering.** v6 runs first. But the prediction is now specific
rather than generic: **if v6 fails, check whether the failures involve rare contact
modes.** If they do, the collection becomes a diagnosed fix aimed at modes a policy
was watched to exploit, rather than insurance against modes we predicted it might.

### The rare-mode subset check is feasible — measured, not assumed

Sampling one 50-row (0.5 s) window per episode across 200 episodes of this half:

```
  windows sampled                       153
  containing any contact transition     118  (77%)
  of those, visiting a lateral mode      49  (41.5%)
  lateral frames within such a window    median 6% of the 50
```

Scaled to dorm-pc's 153 transition windows, **expect ~64 to visit a lateral mode** —
enough to split. So the question "does conditioning still help when the transition
involves a rare mode" is answerable on data already computed, and does not need
collection to answer it.

One caveat on what the subset means: "visits a lateral mode" is a brief event, a median
of 3 rows in 50. The subset is windows that *touch* a rare mode, not windows dominated
by one.


## CORRECTED 2026-09-05: the transition effect is 3x smaller than first reported

dorm-pc's channel-selection defect reached its own headline. Re-run with the shared
two-channel set named explicitly:

| | with transition (n=153) | without (n=47) |
|---|---|---|
| unconditioned 34ch | +0.121 | +0.958 |
| conditioned 40ch | **+0.366** | +0.889 |
| **improvement** | **+0.245** | **−0.069** |

Previously reported as +0.793 at transitions. **The direction holds and the mechanism
is still supported — improvement at transitions, slightly negative away from them —
but the effect is three times smaller.** The extra `vel_body_z_mps` channel in the
conditioned model's selection was worth about 0.55 of the original 0.79.

**This changes what the rare-mode split is for.** At +0.793 the rare-mode caveat could
not plausibly invert the conclusion; the margin was enormous. At +0.245 it can. If
conditioning helps materially less on transitions involving modes the model has barely
seen — and pre-fall windows are exactly where those modes occur — the composed claim
"conditioning helps before a fall" could shrink toward zero on the very population it
was meant to cover. **The split now tests whether the derivation survives, rather than
by how much it holds.** A number three times too large was hiding how much the caveat
mattered.

### Conditioning is a trade, not a uniform improvement

Two facts that "0.181 to 0.876" conceals:

- the conditioned model is **worse away from transitions**, 0.889 against 0.958
- it is roughly **2x worse at absolute open-loop prediction**

It buys action-sensitivity across contact changes and pays in general accuracy. That
is still a good trade for a surrogate a policy optimises inside, because
action-sensitivity is what a policy exploits — but the headline reads as a uniform
improvement and the behaviour underneath is not one.

**The corrected headline must come from the gate's own pipeline.** dorm-pc's script
gives +0.423 pooled for the unconditioned model where the gate gives +0.181, on the
same model and the same channels, so the episode selection differs too. Substituting
one pipeline's number for the other's would be the same error one level up.
