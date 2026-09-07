# Pre-registration: the tracking-capable cell (cell5)

Written 2026-09-07 **while the cell5 verdict is running and before any of its numbers
have been read**. Progress at the time of writing is recorded in the commit message.

## Why this run exists

Every verdict so far ran in `|vx| in [0.02, 0.18]`, which
`go2-command-realisation-deadband.md` shows lies inside the forward-velocity deadband:
the robot realises 4-5% of command below 0.1 m/s.

```
   cell3   ratio 0.14      cell4   ratio 0.30      cell5   ratio 0.56
```

cell5 is the same corpus design and the same disturbance distribution as cell4; only the
command band moves, to `|vx| in [0.4, 0.8]`. The band was chosen by measurement, not by
the documented trained range: `imported_policy.py` documents vx +-0.5, and a six-episode
sweep found tracking still improving beyond it (0.79 at 0.6, 0.90 at 0.8).

## The two-branch reading, registered in advance

The deadband is **common-mode**: it affects both arms on the same episodes and cancels
to first order in a paired difference. It therefore costs **sensitivity, not validity** —
the treatment can only act on the residual, while the -0.020 criterion was calibrated
against the total. cell5 decides which of these was happening:

```
   effect APPEARS at ratio 0.56   -> a POWER problem. The instrument could not see a
                                     real effect inside the deadband, and every prior
                                     verdict is uninformative rather than negative.

   effect STAYS NULL at ratio 0.56 -> the deadband was never the reason. The prior
                                     verdicts were measuring a real absence, and the
                                     null stands with four times the headroom.
```

"Effect appears" means the registered primary: median paired difference reaching
-0.020 m/s with an interval excluding zero. Both branches are informative and neither is
the desired one; recording that here so that whichever lands is not narrated as the
expected outcome.

**What this run cannot decide.** cell5 differs from cell4 in commanded speed, so a
difference between them is not attributable to tracking headroom alone — faster gaits
differ in contact schedule, disturbance rejection and margin. A change is a lead about
the operating point, not a demonstration about the deadband specifically.

## Also to be reported

The registered E1/E2/E3 endpoints, unchanged, and the five-way breakdown, still
unregistered. The family-split prereg's criteria were written for the original cell; they
are computed here for comparability, and E2's -0.020 threshold was calibrated in the
deadband cell, so it is not the same test of the same quantity.
