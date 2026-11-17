# G1: the 3T and a one transformer trained the same way

> **Record of the original runs.** The numbers and figures below come from the first implementation of this
> ablation, in the NRDLab code base. Scripts, run folders and `nrdlab` commands named in this text are that
> implementation's and are not in this repository. The NeDM implementation is `nedm.contact_nrd.arch_ablation`
> ([`contact_nrd/arch_ablation/README.md`](../README.md)). Result files named here are in `artifacts/contact_nrd/arch_ablation/g1_locomotion/fair_1t/`.

The G1 counterpart of [`../go2`](../go2/README.md). The 3T is the contact NRD of `ContactNRD-G1Locomotion` (the study's `train_g1_locomotion.json`: history 4, core, collision and contact `TokenNet`s, 2 530 941 parameters). The 1T is one causal transformer trained by the same objective code, `train.py`.

| | 3T | 1T |
| --- | --- | --- |
| Network | core, collision, contact: 4 layers, 4 heads, width 128 each (2 530 941 parameters) | one causal transformer, 6 layers, 8 heads, width 256, no bias, one change head (about 4.83 M parameters, counted by hand) |
| Tokens | one per kind of input | one per model step: `[state_t, command_t]` × 4 |
| Commands read | the current one | all 4 |
| Output | `Δcore + Σ gate · Δcontact`, then `assemble` | its own change, then the same `assemble` |
| Contact switches (2 pairs, the feet) | the collision network's logits ≥ 0; they gate the contact network | none (the original 1T's design); reported as all off |
| Integrated channels (position, quaternion, joint angles) | from the new rates | the same |
| Stages | core fit, collision 30 000, co-training 120 000 | co-training 120 000 (the 3T's fits run on an untrained 3T, for the windows and scales) |
| Losses, sampling, history noise, pushes, rollouts, optimiser, checks | the study's | the same code and settings; core term in the 1T's `target_std` |
| Checkpoint selection | the official score (7 terms) | the motion score (the 5 terms without contacts) |
| Seed | 61 | 61 |

The 3T reference is the study's released G1 3T, seed 61 (`contact_nrd/g1_locomotion/seed61`). It was trained by the study's released `train.py`, which NRDLab's objective reproduces bit for bit (NRDLab's `docs/status.md`). Both are scored by the study's test (`nrdlab.tasks.contact_nrd.scoring.test_report`) on the G1 test episodes. The official score is the evaluation block's task score (lower is better). Two of its 7 terms (`event_error_rate`, onset `timing_s`) score the predicted contact switches, which the 1T does not predict. So the comparison uses the **motion score**: the other 5 terms (target position, orientation and joint error at 1 s, fall disagreement, non-finite fraction), for every model. `test.py` also gives the 3T's official score.

## Running

The 1T of this comparison is `--arch 1t --width 256` of the NeDM implementation
(`ONE_ARGS="--width 256" scripts/contact_nrd/arch_ablation/reproduce.sh g1_locomotion 1t test`). The files of the original run are
`test_scores.json` (33 MB), `horizon_errors.json` and `horizon_error.png`;
`scripts/contact_nrd/arch_ablation/g1_locomotion/plot_horizon.py` redraws the figure from `horizon_errors.json`.

## Caveats

- The 1T has about twice the parameters of the 3T and reads the command history. As with Go2, a difference in score cannot be put on one cause.
- The 1T is selected on the motion score, the 3T on the official score. A 3T selected on the motion score might rank its checkpoints differently.
- One seed (61) of each model, so the seed spread is not measured.
- The 3T reference was trained by the study's code, not by an NRDLab run in this folder. NRDLab's own full G1 3T run (test score 4.8837 against the released 4.9513) is not readable from this account.
