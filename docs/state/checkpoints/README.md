# Checkpoint tracking

**What exists, where it physically is, and what it is for.**

[`docs/model_checkpoints.md`](../../model_checkpoints.md) is the authoritative
list for the **published** models and should not be duplicated here. This folder
tracks everything else: vision-line checkpoints, RL runs, and — importantly —
**which machine each copy lives on**.

| File | Covers |
|---|---|
| [`vision-line.md`](vision-line.md) | Study 1 and Study 3 checkpoints |

## Rules

1. **Checkpoints are selected on open-loop rollout error**
   (`checkpoint_metric: rollout_sel`), not one-step validation loss. The file is
   still named `best_val.pt` but it is the rollout-selected epoch. The two
   metrics rank checkpoints differently — for `tracked_transformer_v1`, rollout
   picks epoch 8 where val loss does not.
2. **Record the machine.** Artifacts are per-machine. A run in this registry may
   exist on `newton` and not on the box you are sitting at. Verify with
   `ls artifacts/training_runs artifacts/rl_runs`.
3. **A context bank is tied to the checkpoint that encoded it.** Record the
   pairing; the env enforces it via a `z2_mean` fingerprint.
4. Checkpoints are Git LFS. If `git lfs version` fails, `.pt` files are pointer
   stubs, not weights.

## Template

```markdown
| Run | Path | Shape / config | Selected | Machine | What it is for |
|---|---|---|---|---|---|
```
