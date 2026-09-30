#!/usr/bin/env python3
"""Generate the OFAT architecture-ablation configs around the current best
flat+CRM one-hot generalist (the 6L / 8H / 256-embd / ctx-128 anchor).

One-Factor-At-A-Time (OFAT): every config is a byte-for-byte copy of the base
recipe (data mix, rebalanced combined-std Huber loss, rollout_sel selection,
dual-domain 5s/10s rollout eval, optimizer, 80 epochs x 2000 steps, seed) with
ONLY the four model dims varied and output_dir/head_hidden_dim adjusted. This
keeps the ablation clean: any metric delta is attributable to the single
architecture axis that changed.

The anchor (6L/8H/256/ctx128) is the existing trained run, so it is NOT
re-emitted for training; it appears in the manifest pointing at its existing run
dir (train=false) so the ranking script can read all 14 uniformly.

Memory setting (changed 2026-07-09 after a hard freeze during L4): the sweep runs
with load_dataset_into_memory=FALSE (mmap-stream the ~23 GB cache from disk rather
than a resident RAM copy) + pin_memory=FALSE. The resident-RAM load was the
suspected freeze trigger; false is also the trainer's own default.

Usage:
    python scripts/ablations/gen_configs.py                 # Stage A (1 seed)
    python scripts/ablations/gen_configs.py --seed 2026061802 --suffix _s2   # a Stage B seed
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIG = (
    REPO_ROOT
    / "configs"
    / "hmmwv_transformer_v07_tire_normal_force_omega_300g_crm2000_mix25_rebal_rollout_onehot.json"
)
EXISTING_ANCHOR_RUN = (
    "artifacts/training_runs/"
    "hmmwv_transformer_v07_tire_normal_force_omega_300g_crm2000_mix25_rebal_rollout_onehot"
)
CONFIG_OUT_DIR = REPO_ROOT / "configs" / "ablation_ofat"
RUN_ROOT = "artifacts/training_runs/ablation_ofat"

# Anchor shared across every arm: n_layer=6, n_head=8, n_embd=256, block_size=128.
# Each spec: (arm, n_layer, n_head, n_embd, block_size). n_head is scaled so the
# width sweep keeps head_dim == 32 (128/4, 192/6, 256/8, 384/12, 512/16).
SPECS = [
    ("anchor", 6, 8, 256, 128),
    # --- Depth arm (vary n_layer; H8 / E256 / ctx128) ---
    ("depth", 2, 8, 256, 128),
    ("depth", 4, 8, 256, 128),
    ("depth", 8, 8, 256, 128),
    ("depth", 12, 8, 256, 128),
    # --- Width arm (vary n_embd; n_head scaled to head_dim 32; L6 / ctx128) ---
    ("width", 6, 4, 128, 128),
    ("width", 6, 6, 192, 128),
    ("width", 6, 12, 384, 128),
    ("width", 6, 16, 512, 128),
    # --- Heads arm (vary n_head at E256; L6 / ctx128) ---
    ("heads", 6, 4, 256, 128),
    ("heads", 6, 16, 256, 128),
    # --- Context arm (vary train block_size; L6 / H8 / E256) ---
    ("context", 6, 8, 256, 32),
    ("context", 6, 8, 256, 64),
    ("context", 6, 8, 256, 256),
]


def spec_name(n_layer: int, n_head: int, n_embd: int, block_size: int) -> str:
    return f"L{n_layer}_H{n_head}_E{n_embd}_ctx{block_size}"


def build_config(base: dict, n_layer: int, n_head: int, n_embd: int, block_size: int,
                 name: str, run_dir: str, seed: int) -> dict:
    if n_embd % n_head != 0:
        raise ValueError(f"{name}: n_embd {n_embd} not divisible by n_head {n_head}")
    cfg = copy.deepcopy(base)
    cfg["output_dir"] = run_dir
    cfg["model"] = {
        "block_size": block_size,
        "n_layer": n_layer,
        "n_head": n_head,
        "n_embd": n_embd,
        "dropout": 0.0,
        "bias": False,
        # head hidden scales with width so the readout is not a confound; equals
        # the base's 256 at E256, and grows/shrinks naturally with n_embd.
        "head_hidden_dim": n_embd,
    }
    cfg["training"]["seed"] = seed
    # Memory-light setting: mmap-stream the cache from disk instead of holding a
    # ~23 GB resident copy per run. After a hard freeze during L4 (2026-07-09),
    # the resident-RAM load (load_dataset_into_memory=true) was the suspected
    # trigger, so the sweep runs with false (the trainer default) + pin_memory
    # false. This drops non-reclaimable RAM/run from ~23 GB to roughly a batch.
    cfg["training"]["load_dataset_into_memory"] = False
    cfg["training"]["pin_memory"] = False
    cfg["sweep_recipe"] = {
        "version": f"ofat_{name}",
        "slug": name,
        "arm": None,  # filled by caller
        "notes": (
            "OFAT architecture ablation around the 6L/8H/256/ctx128 anchor of the "
            "flat+CRM one-hot generalist. Identical data mix / rebalanced combined-std "
            "Huber loss / rollout_sel selection / dual-domain rollout eval / optimizer / "
            "80 epochs x 2000 steps as the base config; ONLY the model dims vary. "
            f"head_hidden_dim tied to n_embd={n_embd}. seed={seed}."
        ),
    }
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=None,
                    help="Override training seed (default: base config seed, Stage A).")
    ap.add_argument("--suffix", type=str, default="",
                    help="Suffix appended to config/run names (e.g. _s2 for a Stage B seed).")
    ap.add_argument("--only", nargs="*", default=None,
                    help="Restrict to these spec names (e.g. L12_H8_E256_ctx128); anchor always skipped for training.")
    args = ap.parse_args()

    base = json.loads(BASE_CONFIG.read_text())
    seed = args.seed if args.seed is not None else int(base["training"]["seed"])
    CONFIG_OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "base_config": str(BASE_CONFIG.relative_to(REPO_ROOT)),
        "anchor_spec": spec_name(6, 8, 256, 128),
        "seed": seed,
        "suffix": args.suffix,
        "entries": [],
    }

    for arm, n_layer, n_head, n_embd, block_size in SPECS:
        base_name = spec_name(n_layer, n_head, n_embd, block_size)
        name = base_name + args.suffix
        is_anchor = arm == "anchor"

        if is_anchor and not args.suffix:
            # Stage A anchor == the existing trained run; do not retrain.
            manifest["entries"].append({
                "name": name, "arm": arm, "spec": base_name, "train": False,
                "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd, "block_size": block_size,
                "config": None, "run_dir": EXISTING_ANCHOR_RUN,
            })
            continue

        if args.only and base_name not in args.only:
            continue

        run_dir = f"{RUN_ROOT}/{name}"
        cfg = build_config(base, n_layer, n_head, n_embd, block_size, name, run_dir, seed)
        cfg["sweep_recipe"]["arm"] = arm
        cfg_path = CONFIG_OUT_DIR / f"{name}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        manifest["entries"].append({
            "name": name, "arm": arm, "spec": base_name, "train": True,
            "n_layer": n_layer, "n_head": n_head, "n_embd": n_embd, "block_size": block_size,
            "config": str(cfg_path.relative_to(REPO_ROOT)), "run_dir": run_dir,
        })

    manifest_path = CONFIG_OUT_DIR / (f"manifest{args.suffix}.json" if args.suffix else "manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    n_train = sum(1 for e in manifest["entries"] if e["train"])
    print(f"wrote {n_train} trainable configs (+ {len(manifest['entries']) - n_train} reference) "
          f"to {CONFIG_OUT_DIR.relative_to(REPO_ROOT)}")
    print(f"manifest: {manifest_path.relative_to(REPO_ROOT)}")
    for e in manifest["entries"]:
        tag = "TRAIN" if e["train"] else "REF  "
        print(f"  [{tag}] {e['arm']:8s} {e['spec']:20s} -> {e['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
