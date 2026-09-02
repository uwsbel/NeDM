# Reference machines — no access

**Kyle cannot log into any machine in this folder** (verified 2026-09-02).

These files exist for context, not as options. They explain where collected
datasets and published checkpoints physically live, and why the project's
storage and collection rules are written the way they are.

If a plan's next step requires one of these boxes, that step is **blocked** —
surface it as a blocker rather than attempting a workaround.

| File | Machine | Holds |
|---|---|---|
| [`newton.md`](newton.md) | RTX 4090 collection box | Raw frame stores; all Study 3 datasets |
| [`workstation-5090.md`](workstation-5090.md) | Harry's RTX 5090 desktop | Published training and eval runs |
| [`euler.md`](euler.md) | UW Euler cluster | SLURM-scale state-only collections |

Details in each file were transcribed from `.claude/skills/storage/SKILL.md` and
the WP0 implementation notes. None have been verified against the live machines.
