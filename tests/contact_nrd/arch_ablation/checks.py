"""The checks of the ablation's four networks, for one study case's system (CPU, no data; stdlib unittest).

Each study case has a folder with a ``test_networks.py`` that gives the case's system (the entries of its
``system.json`` that the networks read), its training config and the sizes its models must have, and runs these checks:

    PYTHONPATH=src python -m unittest discover -s tests/contact_nrd/arch_ablation -t .
    PYTHONPATH=src python -m unittest tests.contact_nrd.arch_ablation.pool.test_networks
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from nedm.contact_nrd.arch_ablation import arch_1m, arch_1t, arch_3m, checkpoints, common, one_net
from nedm.contact_nrd.model import ContactNRD, model_config

REPO = Path(__file__).resolve().parents[3]
SPHERE = ["position"] * 3 + ["velocity"] * 3 + ["angular_velocity"] * 3


def plane(name):
    return {"name": name, "kind": "plane", "moving": False}


class CaseChecks:
    """Mix into a ``unittest.TestCase`` with the case's ``SYSTEM``, ``CONFIG`` (file name in configs/contact_nrd)
    and ``PARAMETERS`` ({"3t" | "3m" | "1t" | "1m": count})."""

    SYSTEM: dict
    CONFIG: str
    PARAMETERS: dict

    @classmethod
    def setUpClass(cls):
        cfg = json.loads((REPO / "configs/contact_nrd" / cls.CONFIG).read_text())
        cls.history = int(cfg["history"])
        cls.model_config = model_config(cfg["model_step_s"], cls.history, cfg["network"], cfg["integrate_kinds"])
        moving = [b for b in cls.SYSTEM["bodies"] if b["moving"]]
        width = max(int(b.get("state_dim", 9)) for b in moving)
        # every channel of a body on (the data's channel_mask if it has one), padding off
        cls.masks = [[float(x) for x in b.get("channel_mask", [1] * int(b.get("state_dim", 9)))]
                     + [0.0] * (width - int(b.get("state_dim", 9))) for b in moving]
        torch.manual_seed(0)
        cls.three = cls.build_three()
        cls.three_m = arch_3m.build(cls.model_config, cls.SYSTEM, cls.masks)
        cls.one = {name: cls.one_network(arch) for name, arch in (("1t", arch_1t), ("1m", arch_1m))}   # (sizes, model)

    @classmethod
    def build_three(cls):
        return ContactNRD(cls.model_config, cls.SYSTEM, cls.masks)

    @classmethod
    def one_network(cls, arch, **sizes):
        """(sizes, model) of a 1T (arch_1t) or 1M (arch_1m), by default at the 3T's parameter count."""
        three = cls.build_three()
        dims = (cls.history, three.n_bodies * three.smax, three.amax)
        config = arch.config(*dims, **{"target": common.count(three), **sizes})
        return config, one_net.OneNetNRD(arch.build(config, *dims), three)

    def models(self):
        return {"3t": self.three, "3m": self.three_m, "1t": self.one["1t"][1], "1m": self.one["1m"][1]}

    def test_parameters(self):
        """The four networks have the case's sizes, to the parameter."""
        found = {arch: common.count(model.one if arch.startswith("1") else model) for arch, model in self.models().items()}
        self.assertEqual(found, {**found, **self.PARAMETERS})
        self.assertEqual(set(self.PARAMETERS), set(found), "a size for each architecture")

    def test_three_mlp_matches_each_trunk(self):
        """Every MLP trunk has the parameter count of the Encoder it replaces, to within one hidden unit."""
        sizes = arch_3m.three_mlp_config(self.three)
        for name in common.NETWORKS:
            trunk = getattr(self.three_m, name).trunk
            self.assertIsInstance(trunk, common.ResidualMLP)
            per_unit = common.MLP_BLOCKS * (2 * arch_3m.THREE_MLP_WIDTH + 1)
            self.assertLessEqual(abs(common.count(trunk) - sizes[name]["replaces"]), per_unit)

    def test_one_network_sizes(self):
        """The 1T's width and the 1M's hidden width are solved for the 3T's parameter count."""
        target = common.count(self.three)
        for sizes, net in self.one.values():
            self.assertLess(abs(common.count(net.one) - target) / target, 0.05, sizes)   # the 1T's width moves in steps of 8
        self.assertEqual(self.one["1m"][0]["width"], self.one["1t"][0]["width"])

    def test_rollouts(self):
        """All four give a finite rollout of the same shape; a one-network model reports no contact switches."""
        three = self.three
        masks = torch.tensor(self.masks, dtype=torch.float64)
        state = 0.1 * torch.randn(3, three.n_bodies, three.smax, dtype=torch.float64) * masks
        for n in range(three.n_bodies):   # unit quaternions
            q = three.chan[n, 1][three.chan[n, 1] >= 0]
            if len(q):
                state[:, n, q] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
        actions = torch.randn(3, 5, three.n_bodies, three.amax, dtype=torch.float64) * three.has_action[:, None]
        for arch, model in self.models().items():
            with torch.no_grad():
                out, gates = model.eval().rollout(state, actions, return_gates=True)
            self.assertEqual(out.shape, (3, 6, three.n_bodies, three.smax), arch)
            self.assertEqual(gates.shape, (3, 5, three.n_pairs), arch)
            self.assertTrue(torch.isfinite(out).all(), arch)
            if arch.startswith("1"):
                self.assertFalse(gates.any(), arch)

    def test_one_network_reads_the_command_history(self):
        """An earlier command changes a 1T's step, and only while its history entry is valid."""
        net = self.one["1t"][1].eval()
        three = self.three
        if not three.amax:
            self.skipTest("the case has no commands")
        hist = 0.1 * torch.randn(2, self.history, three.n_bodies, three.smax, dtype=torch.float64) * torch.tensor(self.masks)
        act = torch.zeros(2, three.n_bodies, three.amax, dtype=torch.float64)
        ok = torch.ones(2, self.history, dtype=torch.bool)
        past = torch.zeros(2, self.history - 1, three.amax, dtype=torch.float64)
        with torch.no_grad():
            net.past = past.clone()
            base = net.next_state(hist, ok, act)
            net.past = past + torch.tensor([1.0, 0.0])[:, None, None]   # row 0: other earlier commands
            other = net.next_state(hist, ok, act)
            net.past = None
        self.assertFalse(torch.allclose(base[0], other[0]))
        self.assertTrue(torch.allclose(base[1], other[1]))

    def test_checkpoints(self):
        """load_model rebuilds each architecture from its checkpoint; one without an entry is a 3T."""
        common_ = dict(model_config=self.model_config, system=self.SYSTEM, masks=self.masks)
        packets = {"3t": dict(common_, model_state_dict=self.three.state_dict()),
                   "3m": dict(common_, model_state_dict=self.three_m.state_dict(), runtime={"config": {"arch": "3m"}})}
        for arch, (sizes, model) in self.one.items():
            packets[arch] = dict(common_, arch=arch, one_config=sizes, model_state_dict=model.one.state_dict())
        kinds = {"3t": ContactNRD, "3m": ContactNRD, "1t": one_net.OneNetNRD, "1m": one_net.OneNetNRD}
        with tempfile.TemporaryDirectory() as folder:
            for arch, packet in packets.items():
                path = Path(folder) / f"{arch}.pt"
                torch.save(packet, path)
                model, loaded = checkpoints.load_model(path)
                self.assertEqual(checkpoints.arch_of(loaded), arch)
                self.assertIsInstance(model, kinds[arch])
                self.assertFalse(model.training)

    def test_motion_score(self):
        """The motion score drops the contact-event terms of the case's kind of report."""
        if "evaluation" in self.SYSTEM:   # task metrics
            report = {"score_terms": [{"stat": "windows/1.0/robot/pos_m/p95", "term": 1.5},
                                      {"stat": "windows/contacts/event_error_rate", "term": 4.0}]}
            self.assertAlmostEqual(common.motion_score(report), 1.5)
        else:                             # free rollouts
            report = {"selection_score": 0.03, "event_ok_fraction": 0.8}
            self.assertAlmostEqual(common.motion_score(report), 0.03 - 0.05 * 0.2)
