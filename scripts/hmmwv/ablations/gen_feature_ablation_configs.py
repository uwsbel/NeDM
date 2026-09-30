#!/usr/bin/env python
"""Generate the two INPUT-FEATURE ablation configs around the L8 Stage-A winner.

Both are deep copies of configs/ablation_ofat/L8_H8_E256_ctx128.json (same arch,
optimizer, loss mode, 75/25 mix, rollout_sel selection, 80x2000 schedule, seed
2026061801, load_dataset_into_memory=false), changing ONLY what each ablation
names:

  no_onehot            terrain_conditioning.enabled true -> false.
                       Input 20-D -> 18-D; the model can no longer tell flat from
                       CRM, so it must infer the terrain from the state history.
                       Isolates the value of the terrain key on an otherwise
                       identical generalist.

  no_tireforce_omega   every processed_dataset_dir -> its body7 derivation
                       (4 tire Fz + 4 spindle omega channels dropped from BOTH
                       state and target; 15-D -> 7-D). Input 20-D -> 12-D and
                       the readout predicts 7 deltas. Isolates the value of the
                       tire-contact feature block.

Kept OUT of manifest.json: Stage-A ranking compares architectures at a fixed
feature set, and no_tireforce_omega's val_loss is over 7 channels rather than 15,
so it is not comparable there. rollout_sel stays comparable across all three
(it is integrated from vx/vy/yaw_rate, which every arm keeps) -- that is the
metric to judge these on.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "ablation_ofat"
BASE_CONFIG = CONFIG_DIR / "L8_H8_E256_ctx128.json"
RUN_ROOT = "artifacts/training_runs/ablation_ofat"

# 15-D cache -> its 7-D (body-only) derivation, built by derive_state_subset_dataset.py.
BODY7_DATASETS = {
    "artifacts/training_datasets/hmmwv_tire_rigid_300g_normal_force_omega_seq_v1":
        "artifacts/training_datasets/hmmwv_tire_rigid_300g_body7_seq_v1",
    "artifacts/training_datasets/hmmwv_crm_2000_normal_force_omega_seq_v1":
        "artifacts/training_datasets/hmmwv_crm_2000_body7_seq_v1",
}

DROPPED_FIELDS = [
    "tire_fl_force_wheel_fz_n", "tire_fr_force_wheel_fz_n",
    "tire_rl_force_wheel_fz_n", "tire_rr_force_wheel_fz_n",
    "tire_fl_spindle_omega_radps", "tire_fr_spindle_omega_radps",
    "tire_rl_spindle_omega_radps", "tire_rr_spindle_omega_radps",
]

BASE_NOTE = (
    "Input-feature ablation around the OFAT Stage-A winner L8_H8_E256_ctx128 "
    "(S=0.0456). Deep copy of that config: identical L8/8H/E256/ctx128 arch, "
    "75/25 flat/CRM mix, equal-domain-combined-std Huber, rollout_sel selection, "
    "AdamW 3e-4->3e-5, 80x2000 steps, seed 2026061801, "
    "load_dataset_into_memory=false. "
)


def remap_dataset_dirs(node: Any, mapping: dict[str, str], hits: list[str]) -> Any:
    """Rewrite every processed_dataset_dir / channel_weight_datasets entry in place."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "processed_dataset_dir" and isinstance(value, str):
                if value not in mapping:
                    raise ValueError(f"no body7 derivation registered for {value!r}")
                node[key] = mapping[value]
                hits.append(value)
            elif key == "channel_weight_datasets" and isinstance(value, list):
                node[key] = [mapping[item] for item in value]
                hits.extend(value)
            else:
                remap_dataset_dirs(value, mapping, hits)
    elif isinstance(node, list):
        for item in node:
            remap_dataset_dirs(item, mapping, hits)
    return node


def build_no_onehot(base: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base)
    slug = "L8_H8_E256_ctx128_no_onehot"
    config["output_dir"] = f"{RUN_ROOT}/{slug}"
    config["terrain_conditioning"]["enabled"] = False
    config["sweep_recipe"] = {
        "version": f"ofat_{slug}",
        "slug": slug,
        "arm": "feature_ablation",
        "notes": BASE_NOTE + (
            "ONLY change: terrain_conditioning.enabled true->false, so the 2-D "
            "[flat,crm] one-hot is dropped from every token (input 20-D -> 18-D). "
            "The terrains list is retained but inert (num_terrains=0 in the model). "
            "Data mix, datasets and 15-D state/target set are untouched. Tests "
            "whether the terrain key earns its keep or the state history already "
            "identifies the terrain."
        ),
    }
    return config


def build_no_tireforce_omega(base: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base)
    slug = "L8_H8_E256_ctx128_no_tireforce_omega"
    hits: list[str] = []
    remap_dataset_dirs(config, BODY7_DATASETS, hits)
    config["output_dir"] = f"{RUN_ROOT}/{slug}"
    config["sweep_recipe"] = {
        "version": f"ofat_{slug}",
        "slug": slug,
        "arm": "feature_ablation",
        "notes": BASE_NOTE + (
            "ONLY change: every dataset dir -> its body7 derivation "
            "(derive_state_subset_dataset.py --state-field-preset default), which "
            "drops the 4 tire normal forces + 4 spindle omegas from BOTH state and "
            "target: 15-D -> 7-D [vx,vy,roll,pitch,roll_rate,pitch_rate,yaw_rate]. "
            "Input 20-D -> 12-D, readout predicts 7 deltas. One-hot conditioning "
            "stays ON. NOTE: val_loss is over 7 channels here so it is NOT "
            "comparable to the 15-D runs; judge on rollout_sel, which is integrated "
            "from vx/vy/yaw_rate and stays comparable. Dropped: "
            + ", ".join(DROPPED_FIELDS) + "."
        ),
    }
    return config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--print-diff", action="store_true", help="Show what changed vs the L8 base")
    args = parser.parse_args(argv)

    base = json.loads(BASE_CONFIG.read_text())
    configs = {
        "L8_H8_E256_ctx128_no_onehot": build_no_onehot(base),
        "L8_H8_E256_ctx128_no_tireforce_omega": build_no_tireforce_omega(base),
    }

    for slug, config in configs.items():
        path = CONFIG_DIR / f"{slug}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"wrote {path.relative_to(REPO_ROOT)}")
        if args.print_diff:
            for key in sorted(set(base) | set(config)):
                if base.get(key) != config.get(key) and key != "sweep_recipe":
                    print(f"    {key}: {json.dumps(base.get(key))}\n      -> {json.dumps(config.get(key))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
