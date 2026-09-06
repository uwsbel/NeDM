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
| [`quadruped-joint-level-plan.md`](quadruped-joint-level-plan.md) | **Start here.** The current plan: joint-level contact, and what changed |
| [`quadruped-case-study-plan.md`](quadruped-case-study-plan.md) | *Superseded.* The original staged plan; still authoritative on collection scale and the ablation design |
| [`go2-finetune-postmortem.md`](go2-finetune-postmortem.md) | Why fine-tuning the imported policy is closed, what is retracted, and the two distinct failures |
| [`quadruped-command-channel.md`](quadruped-command-channel.md) | Why the robot cannot be steered: there is no command input, by construction |
| [`quadruped-bootstrapping.md`](quadruped-bootstrapping.md) | Why needing a policy to collect data is a new situation for this framework, not a flaw |
| [`quadruped-contact-mode.md`](quadruped-contact-mode.md) | Why contact is temporally but not amplitude separable on soil — and why the foot never penetrates |
| [`go2-contact-mode-coverage.md`](go2-contact-mode-coverage.md) | Which of the 16 contact modes the data actually contains: 8 well, 4 barely |

**Results and gates for that study**, declared before their numbers existed:

| File | The question it settles |
|---|---|
| [`go2-action-sensitivity-gate.md`](go2-action-sensitivity-gate.md) | Whether the surrogate responds to action changes the way Chrono does |
| [`go2-finetune-acceptance-criterion.md`](go2-finetune-acceptance-criterion.md) | The two rules a fine-tune had to clear |
| [`go2-finetune-baseline-predicate.md`](go2-finetune-baseline-predicate.md) | What the fine-tune is compared against |
| [`go2-finetune-displacement-result.md`](go2-finetune-displacement-result.md) | What weight displacement buys and costs |
| [`go2-level3-preregistration.md`](go2-level3-preregistration.md) | The level-3 transfer rule, timestamped ahead of the checkpoint |
| [`go2-p-controller-baseline-result.md`](go2-p-controller-baseline-result.md) | The trivial baseline, and that it wins |
| [`go2-scripted-trot-negative-result.md`](go2-scripted-trot-negative-result.md) | Why the scripted gait was parked |
| [`go2-contact-coverage-protocol.md`](go2-contact-coverage-protocol.md) | How contact coverage is measured |

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
