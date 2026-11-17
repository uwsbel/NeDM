"""The 1M: the 1T with its attention blocks replaced by the residual MLP over the flattened step tokens.

The same token projection and position embedding, then ``Linear(K*d -> w)``,
L x ``[x + Linear(GELU(Linear(LN(x))))]``, ``Linear(w -> d)``, ``LN``; the 1T's conventions (no bias, normal(0.02)).
The same head and the same training as the 1T (``one_net.OneNet``, ``one_train``).
"""
from __future__ import annotations

from torch import nn

from nedm.contact_nrd.arch_ablation import arch_1t
from nedm.contact_nrd.arch_ablation.common import MLP_BLOCKS, ResidualMLP, count, solve_hidden
from nedm.contact_nrd.arch_ablation.one_net import OneNet, load_one, normal_init, positions

ONE_MLP_WIDTH = 512


class MLPBackbone(nn.Module):
    """The 1T's token projection and position embedding, then a residual MLP in place of the attention blocks.
    Same call as ``arch_1t.CausalBackbone``; returns [N, 1, d] (the newest step's)."""

    def __init__(self, input_dim, context, width, hidden):
        super().__init__()
        self.block_size = context
        self.inp = nn.Linear(input_dim, width, bias=False)
        self.pos = nn.Embedding(context, width)
        self.mlp = ResidualMLP(context, width, ONE_MLP_WIDTH, hidden, MLP_BLOCKS, bias=False, scaled_init=False)
        self.apply(normal_init)

    def forward(self, tokens, valid):
        if tokens.shape[1] != self.block_size:
            raise ValueError(f"the MLP reads exactly {self.block_size} steps, got {tokens.shape[1]}")
        return self.mlp(self.inp(tokens) + self.pos(positions(valid)), valid)


def _net(history, state_dim, action_dim, width, hidden):
    return OneNet(MLPBackbone(state_dim + action_dim, history, width, hidden), state_dim, action_dim, width)


def config(history, state_dim, action_dim, target, width=None):
    """Sizes of a 1M with about ``target`` parameters: the 1T's token width (the 1T's width at the same target, or
    ``width``) and the residual MLP's hidden width solved for the rest."""
    width = width or arch_1t.config(history, state_dim, action_dim, target)["width"]
    fixed = (count(_net(history, state_dim, action_dim, width, 1))
             - count(ResidualMLP(history, width, ONE_MLP_WIDTH, 1, MLP_BLOCKS, False, False)))
    hidden = solve_hidden(target - fixed, history, width, ONE_MLP_WIDTH, MLP_BLOCKS, False)
    return {"trunk": "mlp", "width": width, "mlp_hidden": hidden}


def build(config, history, state_dim, action_dim):
    return _net(history, state_dim, action_dim, config["width"], config["mlp_hidden"])


def load(packet):
    """The model of a checkpoint dict, behind the contact NRD's calls (``one_net.OneNetNRD``)."""
    return load_one(packet, build)


def train(args):
    from nedm.contact_nrd.arch_ablation import one_train   # it imports this module (through checkpoints)

    one_train.train_one(args)
