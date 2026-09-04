"""Unbiased rollout evaluation of a selected checkpoint, on the FULL val split.

WHY THIS EXISTS. The checkpoint was selected as the minimum over 80 epochs of
rollout_sel, computed on 12 episodes per domain -- and that same 12-episode
figure was then reported as the checkpoint's performance. The metric's series
across epochs ranged 0.0853 to 0.5406, a 6.3x spread, so taking its minimum and
quoting that draw estimates the checkpoint optimistically: some of the number is
the checkpoint being good and some is that epoch being a lucky read on those
twelve episodes. Nothing in the value separates the two.

The anchor does not share the problem -- its published figures are 100-episode
evaluations -- so the 12-episode number is not even the same quantity.

Re-evaluating on every val episode gives an estimate that selection never saw:
  crm    34 val episodes, 12 used in selection -> 22 unseen
  flat  199 val episodes, 12 used in selection -> 187 unseen
The GAP between the two numbers measures the selection bias directly, which is
why both are reported rather than the biased one being quietly replaced.

REUSES THE TRAINER'S OWN ROLLOUT. It drives Trainer.evaluate_rollouts with
num_episodes raised, rather than reimplementing the loop -- a reimplementation
would be a second mirror of code that has already been hand-copied once tonight,
and mirrors drift.

PER-FAMILY IS A DIAGNOSTIC, NOT A SELECTION METRIC. Selection stays rollout_sel
at the 10 s horizon, pooled, because that is the anchor's rule; changing it after
seeing results is how a metric gets chosen for its answer. Per-family exists
because errdist divides by a POOLED mean_dist over families whose path lengths
differ by 2.7x, so a checkpoint excellent on arc and poor on pivot reports the
same pooled number as the reverse.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nedm.training.trainer import HMMWVTrainer as Trainer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--per-family", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text())
    # Large enough that _select_rollout_episodes exhausts every family rather
    # than stopping at a cap. It round-robins until the pool is empty.
    cfg.setdefault("rollout_eval", {})["num_episodes"] = 100000
    cfg["training"]["device"] = cfg["training"].get("device", "cuda")

    trainer = Trainer(cfg)
    trainer.load_checkpoint(Path(a.checkpoint))

    result: dict = {"checkpoint": str(a.checkpoint), "full_val": trainer.evaluate_rollouts()}

    if a.per_family:
        # Restrict the selection to one family at a time, reusing the same
        # rollout loop. The trainer's own selector is replaced only for the
        # duration of each pass and restored afterwards.
        original = trainer._select_rollout_episodes
        per_family: dict = defaultdict(dict)
        families = set()
        for spec in cfg["rollout_eval"]["datasets"]:
            from nedm.training.dataset import load_split_metadata

            meta = load_split_metadata(Path(spec["processed_dataset_dir"]).resolve(),
                                       str(spec.get("split", "val")))
            families |= set(meta["scenario_families"])
        for fam in sorted(families):
            trainer._select_rollout_episodes = (
                lambda eps, n, _f=fam: [e for e in eps if e["scenario_family"] == _f])
            for k, v in trainer.evaluate_rollouts().items():
                if isinstance(v, dict):
                    per_family[fam][k] = v
        trainer._select_rollout_episodes = original
        result["per_family"] = dict(per_family)

    text = json.dumps(result, indent=2)
    if a.out:
        a.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
