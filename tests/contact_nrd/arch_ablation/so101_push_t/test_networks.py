"""SO101 push-T: the ablation's four networks on the case's system (a 5-joint arm with commands, the T, a table).

    PYTHONPATH=src python -m unittest tests.contact_nrd.arch_ablation.so101_push_t.test_networks
"""
from __future__ import annotations

import unittest

from tests.contact_nrd.arch_ablation.checks import CaseChecks

ARM = ["angle"] * 5 + ["angular_rate"] * 5
T = ["position"] * 3 + ["quaternion"] * 4 + ["velocity"] * 3 + ["angular_velocity"] * 3


class SO101PushT(CaseChecks, unittest.TestCase):
    SYSTEM = {"name": "so101_push_t",
              "bodies": [{"name": "arm", "kind": "articulated", "moving": True, "state_dim": 10, "action_dim": 5,
                          "channel_types": ARM},
                         {"name": "tshape", "kind": "rigid", "moving": True, "state_dim": 13, "channel_types": T},
                         {"name": "table", "kind": "fixed", "moving": False}],
              "pairs": [[0, 1]]}
    CONFIG = "train.json"
    # The sizes of the original runs (contact_nrd/arch_ablation/training_setup.md, section 2).
    PARAMETERS = {"3t": 2_420_190, "3m": 2_420_086, "1t": 2_487_496, "1m": 2_421_128}


if __name__ == "__main__":
    unittest.main()
