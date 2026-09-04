"""Rollout-family coverage measured on the PROCESSED cache, not predicted from raw.

validate_go2_dataset.py answers this against the raw dataset_index, which is the
right place for it: it fails before the expensive step. But the trainer selects
its rollout episodes from the PROCESSED cache -- dataset.py:119 load_rollout_split
reads split_metadata written by preprocess.py:280 -- so the gate's answer is one
transformation away from the one that decides a checkpoint.

The prediction is exact provided the roots are passed in the same order to both
and --max-episodes-per-split is never used, because preprocess neither sorts nor
shuffles. run_go2_preprocess.sh enforces both. So this script is not a fallback
for an unreliable prediction: it is the measurement that makes a DISAGREEMENT
MEAN SOMETHING. If it differs from the gate, something in the pipeline moved, and
that is worth knowing loudly rather than attributing to raw-versus-processed
drift.

Stdlib plus numpy; no torch, so it runs anywhere the cache does.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def select_rollout_episodes(families: list[str], max_episodes: int) -> list[int]:
    """Mirror of Trainer._select_rollout_episodes over episode INDICES.

    Verified AST-identical to trainer.py:796 as of 2026-09-04. The trainer pops
    from the FRONT within a family, so this is order-sensitive; preserving
    episode order out of preprocess is what makes the mirror meaningful.
    """
    by_family: dict[str, list[int]] = {}
    for index, family in enumerate(families):
        by_family.setdefault(family, []).append(index)
    keys = sorted(by_family)
    selected: list[int] = []
    cursor = 0
    while len(selected) < max_episodes and keys:
        key = keys[cursor % len(keys)]
        if by_family[key]:
            selected.append(by_family[key].pop(0))
        if not by_family[key]:
            keys.remove(key)
            cursor -= 1
        cursor += 1
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+", type=Path)
    ap.add_argument("--split", default="val")
    ap.add_argument("--rollout-episodes", type=int, default=12)
    a = ap.parse_args()

    failed = False
    for cache in a.caches:
        meta_path = cache / f"{a.split}_episodes.json"
        if not meta_path.is_file():
            print(f"[????] {cache.name}: no {meta_path.name} -- cannot measure")
            failed = True
            continue
        meta = json.loads(meta_path.read_text())
        fams = list(meta["scenario_families"])
        if not fams:
            # A gate that passes on nothing is the failure it exists to catch.
            print(f"[????] {cache.name}: {a.split} split is EMPTY -- inconclusive, not pass")
            failed = True
            continue
        picked = select_rollout_episodes(fams, a.rollout_episodes)
        got = Counter(fams[i] for i in picked)
        need = len(set(fams))
        ok = len(got) == need
        failed |= not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {cache.name}: "
              f"{len(picked)} selected spanning {len(got)}/{need} families "
              f"(pool {len(fams)} episodes)")
        for fam, n in sorted(got.items()):
            print(f"          {fam}: {n}")
        missing = set(fams) - set(got)
        if missing:
            print(f"          MISSING: {sorted(missing)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
