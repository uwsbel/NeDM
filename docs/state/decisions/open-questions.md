# Open questions

Live and unresolved. Each says what would settle it. Delete an entry when it is
answered, and move the answer into
[`architecture.md`](architecture.md) or a `../lessons/` file.

**Updated:** 2026-09-02.

## Does a global pooled `z2` survive at 256²?

Study 1's 64-D global latent worked on a 128² scene where the object covered 3%
of pixels. Study 3 has a 256² arena where the vehicle is ~15×7 px and small rocks
are 3–6 px. **Settles it:** the WP1 perception pilot's occupancy and localization
probes, run from both the global latent *and* the encoder's pre-pooling spatial
feature map, quantifying what pooling destroys. A fallback is pre-declared —
keep a low-res spatial map (or factored `z_layout`/`z_vehicle`) as the
planner-facing representation while the global `z2` continues to serve the
dynamics token.

## Can the vehicle marker be detected reliably enough for the localization probe?

6/10 layouts at 256². **Settles it:** either a larger marker footprint or
detection tuned on collection-light (55°) frames rather than probe-light (80°)
frames. Blocking for G1.

## Is the 16-token context long enough for a gaited system?

Only relevant if the quadruped case study proceeds. A gait cycle is ~0.3–0.5 s;
16 tokens at 50 Hz is 0.32 s. If phase is not inferable from context, the model
cannot predict touchdown. **Settles it:** decide before collection — feed the
controller phase/clock into the token, or lengthen the context. See
[`../progress/future-case-studies.md`](../progress/future-case-studies.md).

**Measured 2026-09-02, and it does not transfer.**
`scripts/quadruped_wp0_gait.py cycle` reports RoboSimian's shipped
`walking_cycle.txt` as 19,164 rows at dt 1 ms, 32 joints, **19.16 s per cycle**.
That is the period by construction, since `RS_Driver` replays the file on a
loop; the autocorrelation cross-check scatters from 5.9 to 18.9 s because one
period barely fits the record, so do not quote it.

19.16 s against a 0.32 s context is **60x short**, and covering it would need
~960 tokens. For RoboSimian, feeding the controller clock into the token is
therefore the only option.

**But RoboSimian is a slow statically-stable walker and the study targets a Go2
dynamic trot.** At the plan's 0.3-0.5 s that is 15 to 25 tokens at 50 Hz, so
lengthening the context stays entirely viable there, and 16 tokens is already
within a few of sufficient. The prototype measurement bounds the machinery, not
the answer. **Still open for Go2**, and it needs a Go2 gait period. That was blocked while
both boxes ran pychrono 9.0.0, where `ChParserURDF` fails to load. Under the
`nedm` environment (pychrono 10.0.0) the parser is exposed and verified, so
**this is now answerable**: import a Go2 URDF, run a trot, and measure the
period the same way `scripts/quadruped_wp0_gait.py cycle` does for RoboSimian.

## Where does the quadruped's seed controller come from?

You cannot train locomotion in Chrono + CRM (PPO needs ~10⁸ steps; CRM runs
below realtime) and random actions produce only collapse data. Three candidate
paths are ranked in
[`../progress/future-case-studies.md`](../progress/future-case-studies.md); the
recommendation is a scripted gait as a WP0-style vertical slice, with
RoboSimian's in-tree `walking_cycle.txt` as the prototype. **Settles it:** a
privileged scripted gait walking on rigid ground, then CRM, with zero learning.

## Does the `z1` gap close with more data, or is there a floor?

Study 1 measured `error ≈ data^-0.6` at 0.5 s while the matched state-only model
was saturated, and extrapolated that another ~5–6× data (~15 min of collection)
would bring NRD near the state-only floor. It was never run. Residual factors
named in the notes: multi-task loss competition and latent-drift feedback.
**Settles it:** one collection run and one training run — cheap, and it would
turn an extrapolation into a result.

## Is the figure and bibliography pipeline reproducible?

The manuscript's `\graphicspath` points at directories that are gitignored and
absent, filenames differ from what `scripts/figures/` emits
(`hmmwv_rl_reward.pdf` vs `hmmwv_rl_reward_L8.pdf`), and the shared `BibFiles/`
repo is not referenced anywhere. **Settles it:** check in the copy/rename step
and record where `BibFiles/` comes from. See
[`../machines/manuscript.md`](../machines/manuscript.md).
