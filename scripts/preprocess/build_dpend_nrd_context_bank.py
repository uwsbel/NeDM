"""Encode recorded 16-step [z1, z2, a] windows into a reset bank for NRD-in-the-loop RL.

    PYTHONPATH=src python scripts/preprocess/build_dpend_nrd_context_bank.py \
        --split train --num-contexts 16384 --seed 20260826

The latents come from the frozen encoder of the given NRD checkpoint, so a bank is
tied to that checkpoint (the environment checks the z2 normalization fingerprint).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from nedm.nrd.checkpoint import load_nrd_model  # noqa: E402
from nedm.nrd.context_bank import build_context_bank, save_context_bank  # noqa: E402
from nedm.rl.dpend_nrd_reach_env import DEFAULT_NRD_CHECKPOINT  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nrd-checkpoint", type=Path, default=DEFAULT_NRD_CHECKPOINT)
    parser.add_argument("--processed-dataset-dir", type=Path, default=None,
                        help="defaults to the checkpoint config's processed_dataset_dir")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--num-contexts", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--encode-batch", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None,
                        help="defaults to artifacts/rl_reference_sets/<run>_<split>_contexts_<N>_seed<seed>.npz")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = torch.device(args.device)
    model, payload = load_nrd_model(args.nrd_checkpoint, device)
    processed_root = args.processed_dataset_dir or Path(payload["config"]["processed_dataset_dir"])
    run_name = args.nrd_checkpoint.resolve().parents[1].name
    output = args.output or (
        REPO_ROOT / "artifacts" / "rl_reference_sets" / f"{run_name}_{args.split}_contexts_{args.num_contexts}_seed{args.seed}.npz"
    )
    print(f"checkpoint={args.nrd_checkpoint} (epoch {payload.get('epoch')}) processed_root={processed_root}")
    started = time.time()
    bank = build_context_bank(
        model,
        processed_root,
        args.split,
        args.num_contexts,
        args.seed,
        device,
        nrd_checkpoint=str(args.nrd_checkpoint),
        encode_batch=args.encode_batch,
    )
    path = save_context_bank(bank, output)
    absmax = bank["z2_norm_absmax"]
    p999 = bank["z2_norm_p999"]
    print(
        f"saved {path} ({bank['states'].shape[0]} windows x {bank['states'].shape[1]} steps, "
        f"{time.time() - started:.0f} s); normalized-z2 |max| per dim: median {float(absmax.mean()):.2f} "
        f"(min {float(absmax.min()):.2f}, max {float(absmax.max()):.2f}); p99.9 mean {float(p999.mean()):.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
