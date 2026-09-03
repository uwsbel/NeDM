# Decisions and open questions

**Why things are the way they are**, so a future agent does not relitigate a
settled call or silently violate a deliberate constraint.

| File | Covers |
|---|---|
| [`architecture.md`](architecture.md) | Standing architectural commitments |
| [`open-questions.md`](open-questions.md) | Live, unresolved, with what would settle each |
| [`reuse-chrono-crmenv.md`](reuse-chrono-crmenv.md) | Why the policy's input contract is inherited, not reimplemented |

**Case study III (quadruped on CRM)** — read in this order; each answers the
question the previous one raises:

| File | The question it settles |
|---|---|
| [`quadruped-case-study-plan.md`](quadruped-case-study-plan.md) | **Start here.** What the contribution is, and the staged plan |
| [`quadruped-command-channel.md`](quadruped-command-channel.md) | Why the robot cannot be steered: there is no command input, by construction |
| [`quadruped-bootstrapping.md`](quadruped-bootstrapping.md) | Why needing a policy to collect data is a new situation for this framework, not a flaw |
| [`quadruped-contact-mode.md`](quadruped-contact-mode.md) | Why contact is temporally but not amplitude separable on soil — and why the foot never penetrates |

The study plans carry their own decision logs and are authoritative for their
own scope:

- `docs/vision/NRD_overall_project_plan.md` §7 — project-level risks and responses
- `docs/vision/hmmwv_traverse/NRD_hmmwv_traversal_study_plan.md` §16–17 —
  decision log and review resolutions (v1.0 → v1.1)

## Template

```markdown
## <decision>

**Decided:** <date> · **By:** <who> · **Status:** standing | superseded

**Choice:** ...
**Alternatives rejected:** ...
**Why:** ...
**What would reopen it:** ...
```

Mark a decision `superseded` rather than deleting it. The reasoning stays useful
after the conclusion changes.
