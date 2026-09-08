# Chrono replay across boxes: measured, not assumed

**2026-09-08.** The corpus note says episodes replay bit-identically only on their
collecting machine, which would make any cross-machine arm comparison confound arm
with machine. That is a strong constraint to accept on faith when it costs a 4x
serialisation, so it was measured.

## Method

One policy (`finetune_ep80_A_s1`, md5 `f0c7f9b8`, byte-identical on both boxes) scored
on the same 536 val episodes on two machines, **with different pychrono builds**:
`_core.so` md5 `3b0bd530` on sbel against `d1d0bd0a` on north. So this bounds machine
and build together, and a small result is correspondingly stronger than one from
matched binaries.

## Result

```
  sbel    310 / 536  = 57.8%
  north   311 / 536  = 58.0%

  identical row counts        480 of 536   (89.6%)
  completion verdict flips      5 of 536   ( 0.93%)
```

**Replay is not bit-identical across boxes** -- 10.4% of episodes diverge in
trajectory, and five of those cross the completion threshold. The five flips run in
both directions and are episodes that finish near the threshold, which is what a
trajectory divergence should do.

**But the effect on the quantity being compared is 0.2 percentage points.** Against
between-seed spreads of several points and between-arm effects of tens of points,
that is negligible.

## Decision

**Scoring is distributed across the fleet**, and every output filename carries the
scoring host (`score_ep80_<arm>_<host>.json`) so a machine can never be silently
pooled. The measured 0.93% paired flip rate is the quantified cost, recorded here
rather than assumed away.

**Where this does NOT license distribution:** a paired per-episode McNemar test on a
contrast whose true discordance is of order five episodes. There, five spurious flips
are the whole signal. The seed-level analysis this fleet is feeding uses per-arm rates
with the surrogate run as the replicate, where a 0.2 pp bias is far below the error
term -- but any future per-episode paired test must either be scored on one box or
carry machine as a blocking factor.

**Scope:** rigid terrain, 536 val episodes, two boxes, one policy. It bounds the effect
for this corpus and these two builds; it is not a general claim about Chrono.

**A gap this measurement does NOT cover.** The fleet carries **three** distinct pychrono
builds, not two:

```
  sbel     _core.so  3b0bd530     probed
  north    _core.so  d1d0bd0a     probed
  a3       _core.so  cfbf8af6     NOT probed
```

The 0.93% figure bounds sbel against north. a3 scored two arms in this round and its
build has never been compared against either. That is a stated limitation, not an
oversight to be discovered later: if any a3-scored arm ends up carrying a contrast that
matters, its policy should be re-scored on a probed box before the contrast is reported.
