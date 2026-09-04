"""Concatenate per-terrain reference sets into one domain-labelled set.

WHY NOT build_combined_flat_crm_rl_references.py. That script builds its CRM
half by reading a RAW hmmwv dataset directory and re-deriving segments, so it is
tied to the HMMWV collection's on-disk layout and cannot see a processed Go2
cache at all. Our two halves already exist, were built by the same
build_reference_set call from the same split with the same segment length, and
differ only in which processed cache they drew from. Concatenating them is the
whole operation; re-deriving would introduce a second code path that could
disagree with the halves already used for evaluation.

WHAT domains IS FOR. hmmwv_tracking_env resolves a per-reference terrain id from
metadata["domains"] (falling back to splitting refs evenly across terrains when
it is absent -- which for a flat-then-crm concatenation would still be right by
accident, and accidents stop being right the moment the counts are unequal).
Writing the labels makes the mapping explicit and order-independent.

REFUSES TO MERGE SETS THAT ARE NOT COMMENSURABLE. states/actions/poses are
concatenated along the reference axis, so every layout field that indexes those
axes must agree exactly -- state_fields, action_fields, rollout_fields, dt_s and
num_steps. A mismatch is a stack of tensors whose columns mean different things
per row, which nothing downstream would detect. Checked, not assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from nedm.rl.references import ReferenceSet, load_reference_set, save_reference_set  # noqa: E402

# Every field whose meaning is positional in states/actions/poses. Equality here
# is what makes a concatenation along axis 0 meaningful.
LAYOUT_FIELDS = ("state_fields", "action_fields", "rollout_fields", "dt_s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", action="append", required=True, metavar="DOMAIN=PATH",
                    help="Repeatable. Domain label and reference .npz, e.g. flat=a.npz")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()

    pairs = []
    for spec in a.input:
        if "=" not in spec:
            raise SystemExit(f"--input must be DOMAIN=PATH, got {spec!r}")
        domain, _, path = spec.partition("=")
        pairs.append((domain.strip(), Path(path).expanduser()))

    sets = [(d, load_reference_set(p), p) for d, p in pairs]
    head_domain, head, head_path = sets[0]

    for domain, ref, path in sets[1:]:
        for field in LAYOUT_FIELDS:
            lhs, rhs = getattr(head, field), getattr(ref, field)
            if lhs != rhs:
                raise SystemExit(
                    f"{field} differs between {head_path.name} and {path.name}:\n"
                    f"  {head_domain}: {lhs}\n  {domain}: {rhs}\n"
                    "These sets are not commensurable and must not be concatenated."
                )
        if ref.num_steps != head.num_steps:
            raise SystemExit(
                f"num_steps differs: {head_domain}={head.num_steps} {domain}={ref.num_steps}"
            )

    domains: list[str] = []
    for domain, ref, _ in sets:
        domains.extend([domain] * ref.num_references)

    merged = ReferenceSet(
        states=np.concatenate([r.states for _, r, _ in sets], axis=0),
        actions=np.concatenate([r.actions for _, r, _ in sets], axis=0),
        poses=np.concatenate([r.poses for _, r, _ in sets], axis=0),
        episode_ids=[e for _, r, _ in sets for e in r.episode_ids],
        scenario_families=[f for _, r, _ in sets for f in r.scenario_families],
        dt_s=head.dt_s,
        state_fields=head.state_fields,
        action_fields=head.action_fields,
        rollout_fields=head.rollout_fields,
        metadata={
            "domains": domains,
            # The halves came from different caches and different seeds may have
            # been used; keeping one source_processed_root would misattribute
            # half the references. Both are recorded per source instead.
            "merged_from": [
                {
                    "domain": domain,
                    "path": str(path.resolve()),
                    "num_references": ref.num_references,
                    "source_processed_root": ref.metadata.get("source_processed_root"),
                    "source_split": ref.metadata.get("source_split"),
                    "seed": ref.metadata.get("seed"),
                    "segment_nn_steps": ref.metadata.get("segment_nn_steps"),
                    "random_segment_start": ref.metadata.get("random_segment_start"),
                }
                for domain, ref, path in sets
            ],
        },
    )

    out = save_reference_set(merged, a.output)
    counts = {d: domains.count(d) for d in dict.fromkeys(domains)}
    print(f"wrote {out}")
    print(f"  {merged.num_references} references x {merged.num_steps} steps  domains={counts}")
    print(f"  families: {len(set(merged.scenario_families))}")

    # Read back rather than trusting the write. A save/load round trip is where
    # the pickle-free npz path would drop a metadata key, and metadata['domains']
    # is the entire reason this file differs from a plain concatenation.
    back = load_reference_set(out)
    assert back.metadata["domains"] == domains, "domains did not survive the round trip"
    assert back.states.shape == merged.states.shape
    assert np.array_equal(back.states, merged.states)
    print("  round trip verified: domains, shape and states all match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
