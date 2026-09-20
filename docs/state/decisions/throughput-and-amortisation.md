# Throughput and amortisation: what the surrogate actually saves

**Status:** measured 2026-09-19 on north-ubuntu. Recorded here because it was NOT
recorded anywhere, which an audit of the cost claims correctly reported as the numbers
not existing. They exist; they had only ever been printed to a terminal. A measurement
that lives in scrollback is not a result, and quoting one as though it were citeable is
the defect this file closes.

## The measurement

```
  surrogate     11,075 transitions/s   64 branches x 15 steps, 36-D state, cuda
  Chrono CRM         4.17 transitions/s   PER WORKER, mean over 525 episodes,
                                          range 0.25 - 13.39
  per-transition speedup                  2,657x
```

Applied to the three displacement budgets actually used:

```
  arm            updates   transitions   surrogate   Chrono    speedup
  analytic dW 1.0      94        90,240       8.1 s    6.0 h    2,657x
  analytic dW 2.0     205       196,800      17.8 s   13.1 h    2,657x
  analytic dW 4.0   1,147     1,101,120      99.4 s   73.4 h    2,657x
```

## How to quote it, and how not to

This repository already carries a standing rule that a speed-up without its baseline is
not a number, and it already retracted a ~2,700x figure that overstated the soil cost by
3x. So state the denominator every time: **2,657x is against single-worker Chrono CRM**.

Three caveats the single number hides.

**The CRM figure is a mean over a 53x range.** 0.25 to 13.39 transitions/s across 525
episodes. Per-episode wall-clock is contaminated by whatever else shared the box, and
aggregating the per-episode sidecars of different collections gives 2.92, 6.62 and 1.96
transitions/s. Any ratio built on 4.17 inherits that spread, so it is an order-of-
magnitude claim, not a three-significant-figure one.

**Per worker is the honest denominator, but only because CRM does not parallelise.** One
CRM episode saturates the GPU, and concurrency measured NEGATIVE on this workload: two
concurrent episodes took 54.1 s against 36 s sequential. A fleet does scale, and against
a fleet of N boxes the ratio is 2,657/N.

**8.1 s is transition compute, not wall-clock.** The fine-tune job around it is about 55
seconds, or "about a minute on a 3090", once data loading, periodic validation and the
policy export are counted. Quote 8.1 s only as the cost of the transitions themselves.

## Amortisation is the real answer, and it is a break-even, not a speed-up

The per-transition ratio flatters the method, because the corpus is not free. Collection
has to be paid before any fine-tune can run:

```
  corpus collection to amortise    24.1 h of Chrono wall-clock, paid ONCE
  one fine-tune saves               6.0 h of Chrono wall-clock (dW 1.0)
  break-even                        ~4 fine-tunes on one corpus
```

Under 4 fine-tunes the surrogate LOSES to running the same work directly. Past 4 it wins,
and it keeps winning at 6.0 h per additional fine-tune against 8.1 s. This study has run
dozens of arms against a single corpus, so the amortisation is real here rather than
hypothetical, but a project that needed exactly one fine-tune should not build a
surrogate for it.

## The part that does NOT amortise, and it is the bottleneck

Every fine-tuned policy still needs a Chrono verdict, and the verdict is 25 minutes on
eight sharded GPUs, 57 minutes on three A100s, or 70-90 minutes on one box. Against a
fine-tune of 8.1 s of compute, **scoring is three orders of magnitude more expensive than
the optimisation it evaluates.**

So the surrogate makes optimisation nearly free and does not make SELECTION free at all,
and selection is where the Chrono cost lives. The obvious escape, ranking candidates by
the surrogate's own metric and scoring only the winner, is exactly what the abstraction
ladder closed off: rollout fidelity ranks data volumes at rho +0.90 and ranks abstractions
BACKWARDS, so on that axis it would have picked the worst of three arms. Within one
abstraction the rule is sound and can triage; across abstractions there is currently no
cheap proxy and the verdicts have to be paid.

The honest summary of what this method buys, then, is: optimisation cost collapses by
three orders of magnitude, evaluation cost is unchanged, and the win is real only when
candidates per verdict is high enough to amortise a 24.1 h corpus.

## Reproducing

`scripts/throughput/measure_finetune_throughput.py`, which is the script this file's
numbers came from, committed alongside it. Surrogate side times 64x15 branch batches on
the target GPU; Chrono side reads `wall_clock_s` and `rows` from the episode sidecars.

## Correction: the corpus cost was an invented constant

The amortisation section above used "24.1 h of Chrono wall-clock, paid once" and derived
a break-even of about four fine-tunes from it. A provenance audit found that 24.1 h was a
hardcoded f-string literal in `measure_finetune_throughput.py` -- nothing computed it, and
no collection log, node-hour tally or sidecar aggregation reproduces it. It was an
invented numerator carrying a derived conclusion, which is the same defect class as the
throughput figures this file was written to fix.

The script now computes it from the corpus manifest on disk at the Chrono rate it
measures in the same run. The corrected figures:

```
  corpus on disk                795 episodes / 555,851 transitions
  at 4.17 transitions/s         37.0 h of single-worker Chrono
  one fine-tune saves (dW 1.0)   6.0 h
  break-even                    6.2 fine-tunes
```

So the corpus is HALF AGAIN more expensive than recorded, and the break-even is 6.2
fine-tunes rather than 4. The direction of the conclusion is unchanged and the margin is
still comfortable, because this study has run dozens of arms against this corpus, but the
numbers to quote are the ones above.

Two caveats travel with them. The 37.0 h is the full corpus including the validation
split; train alone is 447,372 transitions, or 29.8 h, and which to quote depends on
whether the val episodes are counted as collection cost, which they are, since they had
to be simulated. And 4.17 transitions/s is a mean over a 53x range, so this is an
order-of-magnitude figure like every other number resting on that rate.

An independent sanity check from a different direction, recorded in `go2-crm-pilot.md`:
"A 2000-episode corpus at 16 s is roughly 10 hours of FLEET time." That is fleet rather
than single-worker and a different episode length, so it does not contradict 37.0 h
single-worker; it is the same cost divided across concurrent boxes.
