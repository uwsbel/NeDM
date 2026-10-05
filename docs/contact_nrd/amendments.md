# Unified contact NRD: amendments to the pre-registration

Each entry is written before the data it could influence exists.

**Correction to the pre-registration header.** Its header says "written
~11:20 CDT"; the file on the cluster is timestamped 10:47:12 CDT (SHA recorded
at 10:47:14). The file itself is left unchanged so its SHA still matches.

## 1. Keep the best core phase (2026-10-02, rerun submitted 10:56 CDT, before any version-2 contact training)

**What I saw.** In the first ball core run (`v2_ball_core`), Adam then L-BFGS
brought the free-flight training loss to 8.5e-7. The final Adam phase (lr
1e-4) then pushed it to 0.03. The recipe saves the state after the last
phase, so a worse core would have been kept.

**Change.** Every core phase (and the closed-form readout before them) is
scored on 100,000 held-out validation free-flight windows per moving body,
using the training loss (squared error in free-scale units). The best phase
is kept. Nothing else changes.

**Runs.** Both cores are retrained with this rule: `v2_pool_core_b` and
`v2_ball_core_b`, code snapshot `code_train_v4`. Drawing the held-out
windows first changes the random sampling order, so the reruns are new draws,
not copies of the first runs. The contact stage uses the rerun cores; the
first runs are superseded (the first pool core was stopped at 10 minutes).

## 2. Wall-clock guards inside the 4-hour job limit (2026-10-02 ~10:58 CDT, before any version-2 contact training)

The ball contact recipe asks for 3,000 L-BFGS steps on 200,000 windows, which
may not fit in one 4-hour cluster job. The trainer now stops the contact
L-BFGS polish at the first validation check after 2.5 hours of the run, and
stops rollout refinement at the first check after 3.75 hours. The checkpoint
is still the validation-best one. Any run that hits a guard is reported with
the step it reached. Code snapshot `code_train_v5`.

## 3. Exploratory out-of-range cohorts collected early (2026-10-02, submitted 11:00 CDT)

The two exploratory cohorts (pool mirrored, 400 shots; ball wall at 4.5 m,
200 shots) are collected now, during training, with seeds fixed here (ball
20261005, pool 20261006). They are not read until every checkpoint is frozen,
and they are not used for any choice. They run on the CPU-only MI210 nodes
(16 cores) to keep the MI350 nodes for training; the Chrono build and
collector are the same.

## 4. A fourth ball variant, exact collision labels (2026-10-02, submitted 11:37 CDT)

**Seen before writing this:** the first 20,000 of 80,000 contact-network
updates of the three ball seed-61 runs (validation path p95 0.3 to 7 m, i.e.
not yet trained).

**Reason (physics, not numbers).** The ball's contacts are Chrono NSC
impulses: the velocity flips at one instant. A 10 ms window that ends just
before the impulse and one that just contains it have almost the same start
state but very different next states. With collision labels widened by one
step, the on/off network fires a step early, and the contact network must
output "no change" on one side of that jump and the full bounce on the other.
A smooth network cannot fit that well. With exact labels, the jump sits in the
on/off decision, which can be sharp, and the contact network only sees windows
that contain the impulse.

**Change.** Variant `exact`: collision labels not widened, contact network
trained on contact windows (90 %) and far windows with a forced-on pair (10 %),
like `noband`. Seeds 61, 62, 63 configured. It joins the seed-61 comparison
under the same rule (lowest validation score).

## 5. Stopping dominated ball variants; one revised recipe for both systems (2026-10-02, stopped and submitted 11:49 CDT)

**Seen before writing this** (validation only, seed 61):

| Ball variant | Best score so far | Path p95 at best | Note |
|---|---|---|---|
| band | 0.596 | 255 mm | after 80k Adam; L-BFGS checks 0.967 and 0.669 loss, no gain |
| gap2 | 2.04 | 866 mm | L-BFGS loss flat at 0.026 |
| noband | 14.7 | 6.7 m | L-BFGS loss flat at 0.038 |
| exact | 0.073 | 38.7 mm | at 40k of 80k Adam, every contact event right |

**Stopped:** band, gap2 and noband (seed 61) during their L-BFGS polish.
Their best scores are 8 to 200 times exact's score at half its training,
and the polish loss was flat in all three. They are reported as stopped.

**Two recipe faults, the same in both systems:**
1. The contact network's learning rate never decays: its floor equals the
   start value (1e-3).
2. The contact L-BFGS polish makes no progress (flat loss over 50 steps, about
   5 s per step on the ball). It would take 2.5 hours per ball run.

**Revised recipe `exactdecay`, identical in both systems:**
- exact collision labels;
- contact network trained on contact windows (90 %) and far windows with a
  forced-on pair (10 %);
- cosine learning-rate decay from 1e-3 to 1e-5 over 160,000 updates;
- no contact L-BFGS;
- each system's own rollout refinement, unchanged.

**Runs.**
- Ball: `v2_ball_exactdecay_s61/62/63`.
- Pool: `v2_pool_{centerline,approach}_exactdecay_s61`, then seeds 62/63 of the better frame.
- The registered recipe keeps running: ball `exact` (seed 61) and pool `centerline`/`approach` (seed 61).
- Choice per system: lowest validation score over every completed seed-61 run, registered or revised. The deployment checkpoint is the validation-best seed of the chosen variant.

Note (11:54 CDT): ball `exact` seed 61 was stopped during its L-BFGS
polish (loss 0.0374 after Adam, 0.0383 after 25 L-BFGS steps). Its
validation-best checkpoint (end of Adam, update 80,000, score 0.0021, path p95
1.08 mm) is kept and stays a candidate.

## 6. Pool approach frame stopped in the revised recipe (2026-10-02 12:41 CDT)

**Seen:** validation scores at matching checkpoints, seed 61:

| Recipe | Check | Centerline score (B p95 at 2 s) | Approach score (B p95 at 2 s) |
|---|---|---|---|
| registered | 50k | 0.028 (21.8 mm) | 0.067 (49.9 mm) |
| revised | 20k | 0.030 (22.4 mm) | 0.075 (55.0 mm) |

The approach frame trails at every check of both recipes. The revised
approach run shared a GPU with the revised centerline run and would have
pushed the centerline run's refinement past the time guard. It was stopped at
20k updates; its validation-best checkpoint stays on record. The registered
approach run continues. Pool seeds 62 and 63 use the centerline frame.

## 7. Ball variant choice (2026-10-02 12:55 CDT)

Seed-61 validation scores (ball study's own score, lower is better):
- `exact` (registered recipe, exact labels): **0.002090** (update 80k, path p95 1.083 mm, end p95 2.015 mm, all events right);
- `exactdecay`: 0.002149 (refinement update 400, path p95 1.116 mm, end p95 2.066 mm, all events right);
- `band` 0.596, `gap2` 2.04, `noband` 14.7 (stopped, amendment 5).

By the rule the ball variant is `exact`. The two exact-label recipes are within
3 % of each other. Seeds 62 and 63 of `exact` run without the L-BFGS polish:
it made no progress in any of the four ball runs that reached it, and the
checkpoint is chosen on validation anyway. Their rollout refinement (500
updates, 0.5 s) runs as configured. Seed 61 of `exact` had no refinement
(stopped in the polish).

## 8. One freeze per system (2026-10-02 12:57 CDT)

The ball checkpoints are complete about three hours before the pool ones. So
each system is frozen on its own (`frozen_ball/`, `frozen_pool/`), each with
its own manifest. The cohort seed rule is unchanged:
`int(sha256(manifest_bytes + system_name)[:8], 16)`, using that system's
manifest. Each manifest lists every unified checkpoint of its system (all
variants and seeds, stopped runs included, earlier-recipe controls included)
and that system's references. The ball cohort is collected after the ball
freeze; the pool cohort after the pool freeze. No pool choice can depend on
the ball cohort: the pool rules are fixed, and the ball cohort is not pool data.

## 9. Exploratory, after the ball freeze: a core that does not see position (2026-10-02 13:45 CDT)

**Seen:** the ball's fresh-cohort and moved-wall results. On the moved wall,
the unified model is 36 mm off (median path) after the bounce.

**Hypothesis.** The core sees absolute position, which free motion does not
depend on, in either system. After the wall bounce the ball passes through
positions the core never saw.

**Run.** Ball core and `exact` contact stage (seed 61) with the core's
position inputs held at the training mean (`core_position_free`):
`runs_explore/`, code `code_explore`. This is a post-hoc exploratory run, not
a candidate for any pre-registered claim. It is scored on the fresh ball
cohort and the moved-wall cohort and reported as exploratory.

## 10. Pool variant choice (2026-10-02 15:15 CDT)

Seed-61 validation scores (the pool study's own score, lower is better; B's
error at 2 s, p95, on target-worthy shots in brackets):
- `centerline_exactdecay` (revised recipe): **0.00880** (6.40 mm, events 99.1 %);
- `centerline` (registered): 0.01657 (11.31 mm);
- `approach` (registered): 0.04011 (28.98 mm);
- `approach_exactdecay`: 0.07509 (stopped at 20k, amendment 6);
- earlier-recipe controls: pair frame 0.01542, world frame 0.06213.

Chosen variant: `centerline_exactdecay`. Seeds 62 and 63 were started at
12:04 as a hedge (amendment 5's plan). The deployment checkpoint is the
validation-best of seeds 61, 62 and 63. Seeds 62 and 63 share one GPU and
reach the refinement time guard; their stopping point is reported.
