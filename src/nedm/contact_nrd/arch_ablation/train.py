"""Train one model of the architecture ablation on a contact NRD data directory.

From the repo root:

    PYTHONPATH=src python -m nedm.contact_nrd.arch_ablation.train --arch 3m --data <case>/train --output <runs>/pool_3m_s61
    PYTHONPATH=src python -m nedm.contact_nrd.arch_ablation.train --arch 1t --data <case>/train --output <runs>/pool_1t_s61
    PYTHONPATH=src python -m nedm.contact_nrd.arch_ablation.train --arch 1m --data <case>/train --output <runs>/pool_1m_s61

``--config`` is the 3T's training config (configs/contact_nrd/train.json; G1: train_g1_locomotion.json). Run the
same command again to resume an interrupted run. Each architecture has its own module:

* **3T** (``arch_3t``): ``python -m nedm.contact_nrd.train``, unchanged. The ablation's 3T is the released
  checkpoint, so this is only for a 3T of one's own.
* **3M** (``arch_3m``): the 3T's own training, stage by stage, with the three trunks replaced by residual MLPs of
  the same size. Nothing else differs.
* **1T** (``arch_1t``) and **1M** (``arch_1m``): the 3T's co-training with one network in place of three
  (``one_train``, which lists the differences). ``--updates``, ``--parameters`` and ``--width`` apply to these two.

Output directory, 1T / 1M: see ``one_train``. 3T / 3M: the files of ``nedm.contact_nrd.train``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from nedm.contact_nrd.arch_ablation.checkpoints import ARCHS
from nedm.contact_nrd.train import REPO_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arch", choices=tuple(ARCHS), required=True)
    parser.add_argument("--data", type=Path, required=True, help="data directory (system.json, unified_data.npz)")
    parser.add_argument("--output", type=Path, required=True, help="run directory (an existing run resumes)")
    parser.add_argument("--seed", type=int, help="overrides the config seed")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/contact_nrd/train.json", help="the 3T's config")
    parser.add_argument("--updates", type=int, help="1T / 1M: co-training updates (default: the config's)")
    parser.add_argument("--parameters", type=int, help="1T / 1M: parameter target (default: the 3T's count)")
    parser.add_argument("--width", type=int, help="1T / 1M: token width (default: solved for the parameter target)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true", help="a few updates of every stage (pipeline check)")
    parser.add_argument("--deterministic", action="store_true", help="repeatable GPU runs on one machine (slower)")
    args = parser.parse_args()
    if args.arch in ("3t", "3m") and (args.updates or args.parameters or args.width):
        parser.error("--updates, --parameters and --width apply to 1t and 1m only")
    ARCHS[args.arch].train(args)


if __name__ == "__main__":
    main()
