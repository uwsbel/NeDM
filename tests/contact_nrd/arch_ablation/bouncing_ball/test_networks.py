"""Bouncing ball: the ablation's four networks on the case's system (one ball, a floor and a wall; no commands).

    PYTHONPATH=src python -m unittest tests.contact_nrd.arch_ablation.bouncing_ball.test_networks
"""
from __future__ import annotations

import unittest

from tests.contact_nrd.arch_ablation.checks import CaseChecks, plane


class BouncingBall(CaseChecks, unittest.TestCase):
    SYSTEM = {"name": "bouncing_ball",
              "bodies": [{"name": "ball", "kind": "sphere", "moving": True}, plane("floor"), plane("wall")],
              "pairs": [[0, 1], [0, 2]]}
    CONFIG = "train.json"
    # The size-matched design of this implementation; the original ball runs used other variants (the case's README).
    PARAMETERS = {"3t": 2_397_853, "3m": 2_398_389, "1t": 2_480_504, "1m": 2_397_752}


if __name__ == "__main__":
    unittest.main()
