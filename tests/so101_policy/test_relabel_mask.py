"""vision.viral.relabel_mask: which worlds get the commanded-speed JT label (CPU, < 1 s)."""
import unittest
from types import SimpleNamespace as NS

import torch

from nedm.so101_policy.vision import viral as VI


def fake_env(on, ep_len, held, dxy, cube_xy, lifted, goal_xy):
    t = lambda x, **k: torch.tensor(x, **k)
    F = dict(held=t(held), dxy=t(dxy, dtype=torch.float64),
             c=torch.cat((t(cube_xy, dtype=torch.float64), torch.full((len(on), 1), 0.015, dtype=torch.float64)), 1))
    task = NS(F=F, tr=NS(lifted=t(lifted)), goal_xy=t(goal_xy, dtype=torch.float64))
    return NS(dyn=NS(on=t(on)), ep_len=t(ep_len), task=task)


class TestRelabelMask(unittest.TestCase):
    # world: 0 approach far from the cube, 1 at the cube (not held), 2 held + lifted far from the goal,
    # 3 held + lifted near the goal, 4 nominal world (dyn off), 5 first decision of an episode (at the cube)
    env = fake_env(on=[True, True, True, True, False, True], ep_len=[7, 9, 30, 50, 9, 0],
                   held=[False, False, True, True, False, False], dxy=[0.12, 0.02, 0.0, 0.0, 0.02, 0.02],
                   cube_xy=[[0.2, 0.0], [0.2, 0.0], [0.2, 0.05], [0.2, 0.09], [0.2, 0.0], [0.2, 0.0]],
                   lifted=[False, False, True, True, False, False], goal_xy=[[0.2, 0.1]] * 6)

    def mask(self, **kw):
        return VI.relabel_mask(VI.OnlineConfig(relabel_cmd_speed=True, **kw), self.env).tolist()

    def test_everywhere_skips_nominal_worlds_and_the_first_decision(self):
        self.assertEqual(self.mask(), [True, True, True, True, False, False])

    def test_place_phase_only(self):
        self.assertEqual(self.mask(relabel_near_m=0.05), [False, False, False, True, False, False])

    def test_after_the_lift_anywhere(self):
        self.assertEqual(self.mask(relabel_near_m=1.0), [False, False, True, True, False, False])

    def test_grasp_zone_off(self):
        self.assertEqual(self.mask(relabel_grasp_zone_m=0.04), [True, False, True, True, False, False])

    def test_defaults_are_off(self):
        o = VI.OnlineConfig()
        self.assertFalse(o.relabel_cmd_speed)
        self.assertEqual((o.relabel_near_m, o.relabel_grasp_zone_m), (0.0, 0.0))
        self.assertEqual(VI.OnlineConfig.from_dict(dict(relabel_grasp_zone_m=0.04)).relabel_grasp_zone_m, 0.04)


if __name__ == "__main__":
    unittest.main()
