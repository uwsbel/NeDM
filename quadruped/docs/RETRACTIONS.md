# Retractions

Claims withdrawn after being recorded or presented. Listed so they are not resurrected by
someone reading an older document. Each says what replaced it.

| withdrawn | status | replacement |
|---|---|---|
| "3.1% error at 0.3 s, usable to 1-1.5 s" | unsourced, no measurement existed | measured: 0.034 at 0.30 s, under the no-motion floor to ~3 s |
| "HMMWV/arm get ~2,100x more samples than our PPO" | number exists nowhere; the ratio is backwards | our PPO ran 262,144,000 steps against HMMWV's 131,072,000 -- **2x more**, not fewer |
| "PPO needs longer" | refuted, not merely unsupported | PPO ran twice the budget of the case study where it works and still degraded the policy |
| "one fine-tune is 25 s" | wrong | 8.1 s of transition compute; ~55 s of job wall-clock |
| "corpus collection is 24.1 h, break-even ~4 fine-tunes" | hardcoded literal, never computed | 37.0 h derived from the corpus manifest; break-even 6.2 |
| "training is not machine-invariant (~30% of val_loss)" | wrong diagnosis | the `val_loss` definition changed on 2026-09-17; two same-host runs match to 0.25% |
| "adding channels makes the policy monotonically worse" | mis-stated | the arms are not an information ladder; contact flags are thresholded forces the 48-D arm already has. Supportable: no enrichment beat 36-D |
| "the coverage corpus was contaminated and the reference was clean" | wrong scope | both are ~3% push-active; the defect is common-mode and cancels in that comparison |
| "a wider, more disturbed corpus delivers a worse policy" | confounded, open again | the mechanism offered was distribution mismatch; the contamination explanation turned out common-mode. Unexplained |
| PPO figures of +217.5% and +438.6% | export defect | terminal iterate of a declining run was exported instead of best-in-budget |
| "~20x slower than real time" for CRM | unsourced | 7.5x measured directly on this terrain; RTF 0.1136 over 152 episodes |
| "795 episodes / 447,372 transitions" | mis-paired | 447,372 is the train split; 795 episodes is 555,851 transitions |

## Standing cautions

- **"RL needs ~10^8 steps"** is asserted in two places, both of which mark themselves as
  superseded framing, and neither cites a measurement. Use the study's own budget instead:
  262,144,000 steps, which is 3.66 years single-stream in Chrono.
- **Any `val_loss` compared across 2026-09-17** is invalid, on any host.
- **Any speed-up quoted without its denominator** is not a number. 2,657x is against
  single-worker Chrono CRM; a ~2,700x figure was already retracted once for overstating
  exactly this.
