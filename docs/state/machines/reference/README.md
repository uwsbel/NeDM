# Reference machines — no access

**Kyle cannot log into any machine in this folder** (verified 2026-09-02).

Euler used to be listed here. It is not any more: Kyle has an account on it as
of 2026-09-10, and the CRM scorer runs there and has been validated against the
fleet. See [`../euler.md`](../euler.md).

These files exist for context, not as options. They explain where collected
datasets and published checkpoints physically live, and why the project's
storage and collection rules are written the way they are.

If a plan's next step requires one of these boxes, that step is **blocked** —
surface it as a blocker rather than attempting a workaround.

| File | Machine | Holds |
|---|---|---|
| [`newton.md`](newton.md) | RTX 4090 collection box | Raw frame stores; all Study 3 datasets |
| [`workstation-5090.md`](workstation-5090.md) | Harry's RTX 5090 desktop | Published training and eval runs |

Details in each file were transcribed from `.claude/skills/storage/SKILL.md` and
the WP0 implementation notes. None have been verified against the live machines.
