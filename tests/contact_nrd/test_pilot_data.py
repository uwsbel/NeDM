"""The collected pilot data have what the current evaluation blocks read (CPU, stdlib unittest; reads one array).

For each block in configs/contact_nrd/evaluation/ whose pilot directory artifacts/contact_nrd/collect_pilot/<name>
exists: the per-record arrays that the block names (SO101: the controller ``phase`` [N, T] and the system.json list
``phases`` with the values ``close`` and ``grip``; data.read_record_arrays, as load_data checks them at the start of
training) and the bodies, channels and pairs of its outcome flags. The collection contract does not list ``phase``
yet, so this test is the early check on this workstation. Without the pilot data the test is skipped.
CONTACT_NRD_PILOT_ROOT gives another directory of collected data.

    PYTHONPATH=src python -m unittest tests.contact_nrd.test_pilot_data -v
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

import numpy as np

from nedm.contact_nrd.data import compile_flags, read_record_arrays

REPO = Path(__file__).resolve().parents[2]
BLOCKS = REPO / "configs/contact_nrd/evaluation"
ROOT = Path(os.environ.get("CONTACT_NRD_PILOT_ROOT", REPO / "artifacts/contact_nrd/collect_pilot"))


class PilotData(unittest.TestCase):
    def test_blocks_against_pilots(self):
        found = [(path, ROOT / path.stem) for path in sorted(BLOCKS.glob("*.json"))
                 if (ROOT / path.stem / "system.json").exists()]
        if not found:
            self.skipTest(f"no pilot data in {ROOT}")
        for path, root in found:
            with self.subTest(system=path.stem):
                spec = json.loads(path.read_text())
                system = {**json.loads((root / "system.json").read_text()), "evaluation": spec}
                with np.load(root / "unified_data.npz") as packet:   # lazy: reads only the arrays asked for
                    arrays = read_record_arrays(system, packet)
                if "timing_exclude" in spec:
                    self.assertEqual(list(arrays), [spec["timing_exclude"]["array"]])
                meta = system.get("pair_metadata") or [{} for _ in system["pairs"]]
                compile_flags(spec.get("flags"), system, [m.get("name", f"pair{p}") for p, m in enumerate(meta)])


if __name__ == "__main__":
    unittest.main()
