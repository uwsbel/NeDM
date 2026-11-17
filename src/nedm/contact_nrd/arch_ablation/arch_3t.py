"""The 3T: the contact NRD, unchanged (``nedm.contact_nrd.model.ContactNRD``: three ``TokenNet``s, core, collision
and contact, each with an ``Encoder`` trunk).

The ablation does not train it: its 3T is the released contact NRD checkpoint. ``train`` is
``python -m nedm.contact_nrd.train`` itself, for a 3T of one's own.
"""
from __future__ import annotations

import sys

from nedm.contact_nrd import train as base
from nedm.contact_nrd.model import ContactNRD

build = ContactNRD
"""``build(model_config, system, masks)``."""


def load(packet):
    """The model of a checkpoint dict (a contact NRD checkpoint has no ``arch`` entry)."""
    net = build(packet["model_config"], packet["system"], packet["masks"])
    net.load_state_dict(packet["model_state_dict"])
    return net


def trainer_argv(args, config):
    """The command line of ``nedm.contact_nrd.train`` for the ablation's arguments."""
    return [sys.argv[0], "--data", str(args.data), "--output", str(args.output), "--config", str(config),
            "--device", args.device, *(["--seed", str(args.seed)] if args.seed is not None else []),
            *(["--smoke"] if args.smoke else []), *(["--deterministic"] if args.deterministic else [])]


def train(args):
    sys.argv = trainer_argv(args, args.config)
    base.main()
