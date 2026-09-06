# Paper specs

Technical extractions of three papers that bear directly on the quadruped joint-level
plan, each read **in full with its released code**, not from the abstract. Written
2026-09-05.

| File | Paper | What it is good for |
|---|---|---|
| [`nerd-spec.md`](nerd-spec.md) | Neural Robot Dynamics, Xu et al., CoRL 2025 | Contact as **input** from an analytic query; base-frame re-anchoring; the ablation table |
| [`dhal-spec.md`](dhal-spec.md) | Discrete-Time Hybrid Automata Learning, Liu et al., 2025 | Routing the **loss** through a hard one-hot to mode-specific weights |
| [`halo-spec.md`](halo-spec.md) | HALO, Werner/Esteban et al., L4DC 2026 | Poincare/event indexing, and why it is gated |

**Read the discrepancy sections.** All three papers disagree with their own released code
in ways that change what you would build:

- **NeRD:** the 1000-step open-loop result is **Cartpole only**; ANYmal has no reported
  open-loop state error at any horizon. Noise injection is dead code with no call site.
  Table 2 widths disagree with the shipped configs.
- **DHAL:** the body text says "transformer", the table and the code say 1-D CNN. The KL
  weight differs from the paper by ~90x. The entropy term the paper credits with
  preventing mode collapse has total collapse as its **global minimum**.
- **HALO:** the section uses **one foot**; the other is discarded with the code comment
  `# just choose the default first one for now`. The paper's prose describing
  foot-strike-to-foot-strike is inaccurate for what was implemented.

The specs tag every claim `[P]` paper, `[C]` code, `[M]` measured off a figure,
`[?]` undetermined. **Prefer them to the papers for anything you intend to implement.**
