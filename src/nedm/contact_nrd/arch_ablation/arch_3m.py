"""The 3M: the contact NRD with each ``TokenNet``'s ``Encoder`` replaced by a residual MLP over its flattened tokens
(padded tokens zeroed).

The MLPs keep the ``Encoder``'s conventions (bias, normal(0.02), output layers of a block scaled by 1/sqrt(2L)).
Each MLP's hidden width is solved so that it has as many parameters as the trunk it replaces, so the sizes follow
from the 3T and a checkpoint needs no extra entry.

Training is the 3T's own, ``nedm.contact_nrd.train.main``, stage by stage, with the trunks swapped. Nothing else
differs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from nedm.contact_nrd import train as base
from nedm.contact_nrd.arch_ablation import arch_3t, common
from nedm.contact_nrd.arch_ablation.common import MLP_BLOCKS, NETWORKS, ResidualMLP, count, solve_hidden
from nedm.contact_nrd.data import atomic_json, file_sha256
from nedm.contact_nrd.model import ContactNRD

THREE_MLP_WIDTH = 256


def three_mlp_config(net):
    """The hidden width of each network's MLP, matched to its Encoder's parameters."""
    out = {}
    for name in NETWORKS:
        tn = getattr(net, name)
        d = tn.e_action.numel()
        tokens = 2 + net.k * tn.P + tn.P   # query, fixed partner, K x P state tokens, P action tokens
        hidden = solve_hidden(count(tn.trunk), tokens, d, THREE_MLP_WIDTH, MLP_BLOCKS, True)
        out[name] = {"tokens": tokens, "d": d, "hidden": hidden, "replaces": count(tn.trunk)}
    return out


def use_mlp_trunks(net, config):
    """Replace the three Encoders of a ContactNRD by residual MLPs (in place)."""
    for name in NETWORKS:
        c = config[name]
        getattr(net, name).trunk = ResidualMLP(c["tokens"], c["d"], THREE_MLP_WIDTH, c["hidden"], MLP_BLOCKS,
                                               bias=True, scaled_init=True)
    return net


def build(model_config, system, masks):
    """A ContactNRD with MLP trunks."""
    net = ContactNRD(model_config, system, masks)
    return use_mlp_trunks(net, three_mlp_config(net))


def load(packet):
    """The model of a checkpoint dict."""
    net = build(packet["model_config"], packet["system"], packet["masks"])
    net.load_state_dict(packet["model_state_dict"])
    return net


def train(args):
    """The 3T's trainer with MLP trunks: the config gains ``"arch": "3m"`` and the hashes of this file and
    common.py, which the checkpoints record."""
    from nedm.contact_nrd.arch_ablation import checkpoints   # it imports this module

    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    config = args.output / "train_config.json"
    hashes = {Path(f).name: file_sha256(Path(f)) for f in (__file__, common.__file__)}
    atomic_json(config, {**cfg, "arch": "3m", "arch_ablation_sha256": hashes})
    base.ContactNRD = build
    base.load_model = checkpoints.load_model
    sys.argv = arch_3t.trainer_argv(args, config)
    base.main()
