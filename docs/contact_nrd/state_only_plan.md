# State-only collision and contact networks: plan (written 2026-10-03 ~18:45 CDT, before any training)

## Question

The user's rule: the collision network and the contact network receive only the
NRD state, i.e. the same information the core gets. That is each moving body's 9
numbers (position, velocity, spin), optionally a history of them. No gravity, plane
geometry, relative positions, radii or pair frames. Networks are trained per system.

Is the state, with or without history, enough to match the deployed pair-feature
model?

## Fixed

- **Core:** frozen deployed cores (`runs_v2/v2_{ball,pool}_core_b`), so arms differ only in the two contact networks.
- **Recipe:** exact labels, the deployed revised recipe (160k Adam updates, cosine 1e-3 to 1e-5), 90 % contact windows and 10 % far windows with one pair forced on, the same rollout refinement, and the same validation selection score.
- **Code:** `src/nedm/contact_nrd/model_v3.py`, `train_v3.py`; configs in `configs/contact_nrd/v3/`.

## Arms (seed 61 first)

| Arm | What a pair's networks see | History | Networks |
|---|---|---|---|
| 1 `routed_k1` | the current state of the pair's own bodies (A-B: both balls, in both orders; ball-cushion: that ball) | 1 | collision MLP 256-256-32-1; contact MLP 3 x 1024 tanh with a per-partner scale and shift + linear path |
| 2 `routed_k8_mlp` | same, over 8 steps (current state + 7 differences) | 8 | same MLPs |
| 3 `routed_k8_tf` | same as 2 | 8 | causal Transformers over the 8 time steps (collision width 64 x 2 blocks; contact width 256 x 3 blocks) |
| 4 `joint_k1` (pool only) | the whole current state of both balls, for every pair; the pair is a learned code | 1 | same MLPs as arm 1, scale and shift per pair |

- For the ball, arm 4 is the same model as arm 1, so arm 4 is not run on the ball.
- A learned group/pair code, per-group scaling and per-group output heads are architecture, not information. The pair list fixes only the output structure, i.e. which bodies a pair can change.

## Decision rule (validation only; pool decides)

1. **Futility.** A pool arm with an end-of-Adam score above 0.020 skips refinement.
2. **Ball veto (gross failure only).** Path p95 above 2.2 mm, wrong events above 2 % of shots, or any non-finite rollout.
3. **Winner.** The lowest pool validation score.
   - Arms within 20 % of the lowest count as tied (the deployed pool seeds spread 27 %).
   - Ties go to the simplest arm, in this order: 4, 1, 2, 3.
4. **Seeds.** The winner gets seeds 62 and 63 on both systems.
5. **Matching on validation** requires:
   - pool: the best seed at most 0.0110 (1.25 x the deployed 0.0088), and events right at least 98.8 %;
   - ball: the best seed with 0 wrong events and path p95 at most 1.35 mm.
6. **Fresh test.**
   - Freeze the winner's seeds and the deployed seeds in a manifest.
   - Collect new Chrono cohorts seeded from the manifest hash (ball 1,800, pool 4,800 shots).
   - Paired bootstrap of the median and p95 ratios, using each system's own evaluator. Pool: B at 2 s. Ball: path error.
   - Verdicts:
     - **matches:** both upper bounds are at most 1.25, and failures are not significantly worse;
     - **worse:** either lower bound is above 1.25;
     - **inconclusive:** otherwise.

## What the answer will say

- **"The state is enough":** a history-1 arm wins and matches.
- **"Enough with history":** a history-8 arm wins and matches, and history-1 missed the bar.
- **"Not enough at this budget":** no arm passes, or the fresh-test verdict is worse. In that case, diagnostics (per-pair on/off errors) name where the gap is.

## Change before the counted runs (18:52 CDT)

A code review (three reviewers, each finding independently checked) confirmed one
defect. The first launch at 18:46 was stopped after 5 minutes, before any
contact training.

**Defect.** In routed pool rows, an A row and a B row of the same group (for example
"ball with the +x cushion") share inputs of the same kind. The output was
an 18-number vector masked to the affected ball, so the starting linear fit
and the output scaling mixed A's real changes with zeros. They came out at about
half strength.

**Fix.** A routed row now outputs one 9-number change for the ball it affects,
placed into that ball's slot. A and B therefore share one cushion response (and
one A-B response), which was the intent. The fit uses each row's own ball. Joint
rows (arm 4) keep the 18-number output masked to the pair's members.

The other seven reported issues were refuted:
- the test-split labels are only counted for a log line;
- memory use is fine;
- retargeting is unused here;
- the futility rule matches the plan.

## Second change before the counted runs (relaunched 19:25 CDT): input maps fitted on contact data

**Seen:** the first counted launch (18:52), stopped at 19:21:
- pool scores at 10k to 90k contact updates were 0.14 to 0.73, against the deployed model's 0.020 to 0.096 at the same stages;
- training loss was about 100 times the deployed one;
- the ball learned normally (path p95 30 mm at 10k, deployed 87 mm).

**Cause.** The deployed contact network scales its inputs with statistics of
contact windows. In version 3, rows were scaled by their spread over the whole table
(about 0.4 m). So the centimetres that decide a cushion contact, and the A-B
separation (a small difference of two large positions), were squeezed into a
sliver of the input range.

**Change.** Each network now has its own per-group data-fitted affine map of its
rows:
- the collision network's map is fitted on that group's near-contact rows;
- the contact network's map is fitted on that group's single-contact rows, as in version 2;
- the map standardises, then whitens (principal axes rescaled, eigenvalue floor
  1e-4, so no direction is magnified more than 100 times).

The map is linear in the state and adds no information. Same arms, same seeds,
same decision rule.

## Third change (relaunched ~20:00 CDT): no whitening in the main arms

**Seen** (validation, second launch):
- Whitening made the teacher-forced fit as good as the deployed model's: pool joint arm training loss 50 at 140k updates (deployed about 35-50), and collision false alarms 2-8 per 2,200 contacts.
- Rollouts were worse. Pool joint arm score 0.36 with 77 % of events right. Ball routed arm 167 mm path p95 at 10k updates, against 30 mm without whitening. The history MLP arms diverged in both systems.

**Cause.** Whitening magnifies directions that are nearly constant in contact steps (for example spin locked to velocity while a ball rolls) by up to 100 times. A small drift in a rollout then becomes a huge input. The earlier pool study put per-channel floors in place for exactly this reason.

**Change.**
- Main arms: per-group, per-channel standardisation with floors. Contact network: fitted on that group's contact rows. Collision network: fitted on near-contact rows. This is the version-2 and pool-study practice.
- One extra pool arm `a4w` (joint, whitening with eigenvalue floor 1e-2, so at most 10x magnification) as a check.
- The stopped runs (second launch) are kept under `runs_v3` (renamed `*_whiten_attempt`) for the record.

## Fourth change (relaunched 20:20 CDT): a fixed soft limit on the scaled inputs

**Seen** (third launch, 10k to 60k updates):
- collision networks good (pool 0-8 false alarms, 0-4 misses per ~2,200 contacts per pair);
- contact stage 5-20x worse than the deployed model (pool routed 0.49 at 20k, deployed 0.030), with training losses in the tens of thousands.

**Cause.** The contact maps are fitted on contact rows, where a ball at a cushion
sits within millimetres of one position. The 10 % far windows (one pair forced on,
target ~ no change) then give inputs of about -100 to -400 standard units. The
data-fitted linear path turns these into nonsense changes, and the loss explodes.
The deployed model bounded its distance input with s*tanh(x/s).

**Change.** Every scaled input passes through a fixed c*tanh(z/c), with c = 6, in
both networks. It is about the identity inside +-3 and bounded far from the
contact data. It adds no information. All arms are relaunched, including `a4w`.

## Exploratory arms added after seeing the joint arm (launched 20:58 CDT): mirror augmentation (pool)

**Seen.** Joint arm `a4`, seed 61, end of contact Adam:
- validation score 0.0297 (target-worthy B p95 at 2 s 21.9 mm, median 2.3 mm, events 98.0 %);
- refinement skipped by the futility rule;
- its training loss ended at the deployed model's level (27-37).

So the networks fit training contacts as well as the deployed model. The gap is in
generalisation. With the state only, each cushion is learned from its own
contacts, while the pair frame let all four cushions share one response.

**Arms `a1m` and `a4m`.** As `a1` and `a4`, plus training-only copies of every
training episode mirrored in x, in y and in both:
- state signs flipped, with spin treated as an axial vector;
- pair labels permuted (+x cushion <-> -x cushion, etc.);
- the networks' inputs are unchanged (the state only). The symmetry of the table
  is used to make data, not as an input.
- This also gives the -x cushion (never hit in training) its first data.

These arms are exploratory: added after seeing validation results. They are
reported as such, and judged by the same validation score.

## More exploratory pool arms (launched 21:35 CDT)

**Seen** (validation, one step with the true switches, final joint arm against the deployed model):
- A-B contacts: position error 0.096 against 0.037 mm, velocity error 4.9 against 1.7 mm/s, about 3x worse;
- cushion contacts: 1.3-2.4x worse.

The gap is mainly the A-B response, which the pair frame made independent of where
and in which direction the collision happens.

**Arms.**
- `a4r`: `a4` plus training-only copies of single A-B contact steps under a random rigid
  motion of the pair on the table. The pair is rotated about the vertical through its
  midpoint, and the midpoint is moved to a random point within 1.1 x 0.5 m. This keeps
  separation, speeds, spin sizes and rolling. The inputs are still the state only.
- `a4f`: `a4` with the futility rule off, to measure what refinement alone adds.

Both arms are exploratory, with no futility rule.

## Follow-up requested by the user (10-04): contact network trained on contact steps only

The user's point: with an accurate collision network, the contact network only ever
acts on contact steps. The 10 % far steps (one pair forced on, target "no change")
are only insurance against false alarms. They are also what blew up state-only
training.

**Runs, seed 61, everything else as the counted arms:**
- `a4nf`: pool joint, contact mix 100 % contact steps, soft limit kept. Baseline: `a4` seed 61.
- `a4nfc`: as `a4nf` without the soft limit. Tests whether the limit was needed only because of the far steps.
- `a1nf`: ball routed history 1, contact steps only. Baseline: `a1` seed 61.
