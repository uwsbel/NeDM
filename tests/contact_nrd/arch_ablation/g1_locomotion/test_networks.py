"""G1 locomotion: the ablation's four networks on the case's system (the 27-joint robot with commands, the ground,
one contact pair per foot; scored by the task metrics of an ``evaluation`` block).

    PYTHONPATH=src python -m unittest tests.contact_nrd.arch_ablation.g1_locomotion.test_networks
"""
from __future__ import annotations

import unittest

from nedm.contact_nrd.arch_ablation import arch_1t, common
from tests.contact_nrd.arch_ablation.checks import CaseChecks, plane

ROBOT = (["position"] * 3 + ["quaternion"] * 4 + ["angle"] * 27
         + ["velocity"] * 3 + ["angular_velocity"] * 3 + ["angular_rate"] * 27)


class G1Locomotion(CaseChecks, unittest.TestCase):
    SYSTEM = {"name": "g1_locomotion",
              "bodies": [{"name": "robot", "kind": "articulated", "moving": True, "state_dim": 67, "action_dim": 27,
                          "channel_types": ROBOT, "channel_mask": [1] * 67}, plane("ground")],
              "pairs": [[0, 1], [0, 1]],
              "evaluation": {}}   # its presence selects the task metrics; the networks do not read it
    CONFIG = "train_g1_locomotion.json"
    # The sizes of the original runs (contact_nrd/arch_ablation/training_setup.md, section 2).
    PARAMETERS = {"3t": 2_530_941, "3m": 2_531_477, "1t": 2_506_632, "1m": 2_530_376}

    def test_wide_one_transformer(self):
        """``--width 256``, the width of the 1T in the original 3T-against-1T comparison (the case's README). That 1T
        had another backbone code and 4 829 696 parameters; this implementation's has 4 833 024."""
        _, net = self.one_network(arch_1t, width=256)
        self.assertEqual(common.count(net.one), 4_833_024)


if __name__ == "__main__":
    unittest.main()
