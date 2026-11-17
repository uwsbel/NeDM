"""Pool: the ablation's four networks on the case's system (balls A and B, four cushions; no commands).

    PYTHONPATH=src python -m unittest tests.contact_nrd.arch_ablation.pool.test_networks
"""
from __future__ import annotations

import unittest

from tests.contact_nrd.arch_ablation.checks import CaseChecks, plane


class Pool(CaseChecks, unittest.TestCase):
    SYSTEM = {"name": "pool",
              "bodies": [{"name": "A", "kind": "sphere", "moving": True}, {"name": "B", "kind": "sphere", "moving": True},
                         plane("cushion_xp"), plane("cushion_xm"), plane("cushion_yp"), plane("cushion_ym")],
              "pairs": [[0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [1, 2], [1, 3], [1, 4], [1, 5]]}
    CONFIG = "train.json"
    # 3T: the released pool model's size. 3M, 1T, 1M: the size-matched design of this implementation; the original
    # pool runs used other variants (the case's README).
    PARAMETERS = {"3t": 2_404_916, "3m": 2_404_812, "1t": 2_483_816, "1m": 2_405_160}


if __name__ == "__main__":
    unittest.main()
