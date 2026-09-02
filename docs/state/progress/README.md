# Progress

**One file per workstream.** Each answers two questions and nothing else:
*where are we* and *what is the single next action*.

This is deliberately different from [`docs/progress.md`](../../progress.md),
which is the paper's **reproduction record** — a stable archive of how published
numbers were produced. These files are volatile and are expected to be rewritten
often.

| File | Workstream | Status |
|---|---|---|
| [`00-overview.md`](00-overview.md) | All of it, one screen | — |
| [`state-only-paper.md`](state-only-paper.md) | The published state-only NRD | Submitted |
| [`vision-study1-dpend.md`](vision-study1-dpend.md) | Study 1: double pendulum + RGB | Complete |
| [`vision-study3-traverse.md`](vision-study3-traverse.md) | Study 3: HMMWV + overhead RGB-D | WP0 done, WP1 blocked |
| [`future-case-studies.md`](future-case-studies.md) | Candidate case studies 3 and 4 | Under discussion |

## Template

```markdown
# <workstream>

**Status:** <one line> · **Updated:** <date> · **Branch:** <branch>

## Where we are
## What is done (with evidence)
## What is next  — the ONE next action, first
## Blocked on
## Open risks
```

Rules: date everything, cite the artifact or notes file behind every claim, and
when a claim turns out to be wrong **delete it** rather than appending a
correction underneath.
