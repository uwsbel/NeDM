"""State conventions of the contact NRD for the layouts of contact_nrd/collection_contract.md (CPU, stdlib unittest).

- Round trip: data.step_average_rates, then model.assemble with the true step-average rates, reproduces every recorded
  model-step state (G1: one floating articulated body, 67 channels; SO101: arm 12 + cube 13), to 1e-9 with unit
  quaternions. The contract stores float32, whose quaternion norms are 1 +- 4e-8; assemble re-normalises, so the raw
  records are reproduced to that norm error (checked: < 1e-7).
- Quaternion convention: a constant world-frame angular velocity (q(t) = exp(w t) * q0, increment multiplied from the
  left) gives step-average rates equal to w, and assemble integrates it back; a body-frame rotation does not.
- Duplicate pairs: two pairs with the same [i, j] have their own pair code and their contact changes add.
- rollout_from(start(x), a) equals rollout(x, a).

    PYTHONPATH=src python -m unittest tests.contact_nrd.test_conventions -v
"""
from __future__ import annotations

import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from nedm.contact_nrd.data import load_data, step_average_rates
from nedm.contact_nrd.model import ContactNRD, model_config
from tests.contact_nrd import synthetic

KINDS = ["sphere", "rigid", "articulated"]
SMALL = {"width": 32, "heads": 2, "ffn": 64, "layers": 1}


def full_masks(system):
    moving = [b for b in system["bodies"] if b["moving"]]
    smax = max(int(b["state_dim"]) for b in moving)
    return [[1.0] * int(b["state_dim"]) + [0.0] * (smax - int(b["state_dim"])) for b in moving]


def random_model(system, seed=0):
    """A small model with random (not zero) output heads and a collision bias that switches some pairs on."""
    torch.manual_seed(seed)
    model = ContactNRD(model_config(0.02, 4, SMALL, KINDS), system, full_masks(system))
    for net in (model.core, model.collision, model.contact):
        for head in net.heads:
            nn.init.normal_(head.weight, std=0.05)
    return model.eval()


class RoundTrip(unittest.TestCase):
    def worst(self, name, unit):
        with tempfile.TemporaryDirectory() as tmp:
            system = synthetic.write(tmp, name, episodes=10, records=121, seed=3)
            data = load_data(tmp, "cpu", model_step_s=0.02)
        model = ContactNRD(model_config(0.02, 4, SMALL, KINDS), data["system"], full_masks(system))
        if unit:   # float32 records -> unit quaternions in float64
            st = data["states"].clone()
            for d in range(model.n_bodies):
                qc = torch.nonzero(model.quat[d]).flatten()
                st[:, :, d, qc] = st[:, :, d, qc] / st[:, :, d, qc].norm(dim=-1, keepdim=True)
            data["states"] = st
        st, s = step_average_rates(data, set(KINDS))["states"], data["stride"]
        worst = 0.0
        for r in range(0, st.shape[1] - s, s):
            cur, nxt = st[:, r], st[:, r + s]
            out = model.assemble(cur, nxt - cur, torch.zeros_like(cur))
            worst = max(worst, float((out - nxt).abs().max()))
        return worst

    def check(self, name):
        self.assertLess(self.worst(name, unit=True), 1e-9)
        self.assertLess(self.worst(name, unit=False), 1e-7)

    def test_g1_layout(self):
        self.check("g1_locomotion")

    def test_so101_layout(self):
        self.check("so101_grasp")


class QuaternionConvention(unittest.TestCase):
    def states(self, left):
        rng = np.random.default_rng(1)
        w = np.array([0.7, -1.1, 2.3])
        q0 = rng.normal(size=4)
        q0 /= np.linalg.norm(q0)
        q0 *= np.sign(q0[0])
        t = np.arange(41) * 0.01
        inc = synthetic.quat_exp(w * t[:, None])
        q = synthetic.quat_mul(inc, q0) if left else synthetic.quat_mul(q0, inc)
        x = np.zeros((1, len(t), 1, 13))
        x[0, :, 0, 3:7] = q
        x[0, :, 0, 10:13] = w
        system = {"bodies": [dict(name="cube", kind="rigid", moving=True, state_dim=13,
                                  channel_types=["position"] * 3 + ["quaternion"] * 4 + ["velocity"] * 3 + ["angular_velocity"] * 3),
                             dict(name="table", kind="box", moving=False)], "pairs": [[0, 1]]}
        return dict(states=torch.as_tensor(x), stride=2, dt=0.02, system=system), w

    def test_world_frame_left_multiplied(self):
        data, w = self.states(left=True)
        st = step_average_rates(data, {"rigid"})["states"]
        self.assertLess(float((st[0, 2:, 0, 10:13] - torch.as_tensor(w)).abs().max()), 1e-9)
        model = ContactNRD(model_config(0.02, 4, SMALL, KINDS), data["system"], [[1.0] * 13])
        for r in range(0, st.shape[1] - 2, 2):
            out = model.assemble(st[:, r], st[:, r + 2] - st[:, r], torch.zeros_like(st[:, r]))
            self.assertLess(float((out[:, 0, 3:7] - st[:, r + 2, 0, 3:7]).abs().max()), 1e-9)

    def test_body_frame_differs(self):
        data, w = self.states(left=False)
        st = step_average_rates(data, {"rigid"})["states"]
        self.assertGreater(float((st[0, 2:, 0, 10:13] - torch.as_tensor(w)).abs().max()), 0.1)


class DuplicatePairs(unittest.TestCase):
    def test_codes_and_sum(self):
        system = synthetic.g1_system(101)
        self.assertEqual(system["pairs"][0], system["pairs"][1])
        model = random_model(system)
        self.assertTrue(torch.equal(model.pair_pos[0], model.pair_pos[1]))
        self.assertEqual(int(model.pair_fixed[0]), int(model.pair_fixed[1]))
        self.assertFalse(torch.equal(model.contact.e_query.weight[0], model.contact.e_query.weight[1]))
        torch.manual_seed(1)
        b, s, a = 16, model.smax, model.amax
        cur = torch.randn(b, 1, s, dtype=torch.float64)
        diff = torch.randn(b, model.k - 1, 1, s, dtype=torch.float64)
        act = torch.randn(b, 1, a, dtype=torch.float64)
        lag_ok = torch.ones(b, model.k, dtype=torch.bool)
        with torch.no_grad():
            one = [model.contact_sum(cur, diff, act, lag_ok, torch.tensor([[1.0, 0.0], [0.0, 1.0]][p]).repeat(b, 1)) for p in (0, 1)]
            both = model.contact_sum(cur, diff, act, lag_ok, torch.ones(b, 2))
        self.assertGreater(float((one[0] - one[1]).abs().max()), 1e-6)
        self.assertGreater(float(one[0].abs().max()), 1e-6)
        self.assertLess(float((both - one[0] - one[1]).abs().max()), 1e-12)


class RolloutFrom(unittest.TestCase):
    def test_equals_rollout(self):
        system = synthetic.so101_system(101)
        model = random_model(system, seed=2)
        nn.init.normal_(model.collision.heads[0].weight, std=1.0)
        torch.manual_seed(3)
        x = torch.randn(8, model.n_bodies, model.smax, dtype=torch.float64) * model.smask
        acts = torch.randn(8, 12, model.n_bodies, model.amax, dtype=torch.float64) * model.amask
        with torch.no_grad():
            r1, g1 = model.rollout(x, acts, return_gates=True)
            r2, g2 = model.rollout_from(*model.start(x), acts, return_gates=True)
        self.assertTrue(torch.equal(r1, r2))
        self.assertTrue(torch.equal(g1, g2))
        self.assertTrue(0 < float(g1.mean()) < 1)


if __name__ == "__main__":
    unittest.main()
