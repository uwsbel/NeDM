# Decisions and open questions

**Why things are the way they are**, so a future agent does not relitigate a
settled call or silently violate a deliberate constraint.

| File | Covers |
|---|---|
| [`architecture.md`](architecture.md) | Standing architectural commitments |
| [`open-questions.md`](open-questions.md) | Live, unresolved, with what would settle each |

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
