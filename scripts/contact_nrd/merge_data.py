"""Merge contact NRD data folders of one system into one folder (numpy only).

    python scripts/contact_nrd/merge_data.py OUT BASE EXTRA [EXTRA ...]

BASE gives system.json (bodies, pairs, evaluation block). Each EXTRA folder must have the same bodies, pairs and phases;
families that BASE lacks are appended to the families list (their indices in EXTRA must follow BASE's). The arrays of
unified_data.npz are concatenated along the episode axis; episode_id and group_id of each EXTRA are offset so they stay
unique. The splits are kept as written: a supplementary campaign (collect --campaign extra) has only split 0, so the
validation episodes of the merged folder are those of BASE. Used for the SO101 grasping fix (2026-10-10):
production + campaign extra (--families one_sided near_miss weak_closure push).
"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    out, base, extras = Path(sys.argv[1]), Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
    system = json.loads((base / "system.json").read_text())
    names = [f["name"] for f in system["families"]]
    arrays = dict(np.load(base / "unified_data.npz"))
    for extra in extras:
        other = json.loads((extra / "system.json").read_text())
        for key in ("bodies", "pairs", "phases"):
            if other.get(key) != system.get(key):
                raise SystemExit(f"{extra}: {key} differ from {base}")
        their = [f["name"] for f in other["families"]]
        if their[: len(names)] != names[: len(their)]:
            raise SystemExit(f"{extra}: families must extend {base}'s list in order")
        for f in other["families"][len(names):]:
            system["families"].append(f)
            names.append(f["name"])
        packet = np.load(extra / "unified_data.npz")
        if set(packet.files) != set(arrays):
            raise SystemExit(f"{extra}: arrays differ: {sorted(set(packet.files) ^ set(arrays))}")
        n = len(arrays["lengths"])
        for key in arrays:
            x = packet[key]
            if key in ("episode_id", "group_id"):
                x = x + n
            arrays[key] = np.concatenate((arrays[key], x), 0)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "unified_data.npz", **arrays)
    (out / "system.json").write_text(json.dumps(system, indent=1))
    (out / "manifest.json").write_text(json.dumps({"merged_from": [str(base)] + [str(e) for e in extras],
                                                   "episodes": int(len(arrays["lengths"])),
                                                   "splits": {str(s): int((arrays["splits"] == s).sum()) for s in (0, 1, 2)}},
                                                  indent=1))
    print(f"{out}: {len(arrays['lengths'])} episodes, families {names}")


if __name__ == "__main__":
    main()
