"""The 1T: one causal transformer over one token per step, ``[state_t of every moving body, command_t]``.

Token projection, position embedding, 6 causal pre-LN blocks of 8 heads, final norm; no linear bias; normal(0.02).
One head on the newest token gives the change of every channel (``one_net.OneNet``). It predicts no contacts.

Training is the 3T's co-training with this one network in place of three (``one_train``).
"""
from __future__ import annotations

import torch
from torch import nn

from nedm.contact_nrd.arch_ablation.common import count
from nedm.contact_nrd.arch_ablation.one_net import OneNet, load_one, normal_init, positions

ONE_LAYERS, ONE_HEADS = 6, 8


class _CausalBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d, bias=False), nn.GELU(), nn.Linear(4 * d, d, bias=False))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class CausalBackbone(nn.Module):
    """Step tokens [N, K, D] and valid [N, K] -> features [N, K, d]. Padded steps are masked out of attention."""

    def __init__(self, input_dim, context, width):
        super().__init__()
        self.block_size, self.n_head = context, ONE_HEADS
        self.inp = nn.Linear(input_dim, width, bias=False)
        self.pos = nn.Embedding(context, width)
        self.blocks = nn.ModuleList([_CausalBlock(width, ONE_HEADS) for _ in range(ONE_LAYERS)])
        self.ln_f = nn.LayerNorm(width)
        self.apply(normal_init)

    def forward(self, tokens, valid):
        length = tokens.shape[1]
        if length > self.block_size:
            raise ValueError(f"history of {length} steps exceeds the model context {self.block_size}")
        x = self.inp(tokens) + self.pos(positions(valid))
        causal = torch.ones(length, length, dtype=torch.bool, device=tokens.device).tril_()
        allowed = (causal[None] & valid[:, None, :]) | torch.eye(length, dtype=torch.bool, device=tokens.device)
        mask = torch.zeros(allowed.shape, device=tokens.device).masked_fill_(~allowed, float("-inf"))
        mask = mask.repeat_interleave(self.n_head, dim=0)
        for block in self.blocks:
            x = block(x, mask)
        return self.ln_f(x)


def _net(history, state_dim, action_dim, width):
    return OneNet(CausalBackbone(state_dim + action_dim, history, width), state_dim, action_dim, width)


def config(history, state_dim, action_dim, target, width=None):
    """Sizes of a 1T with about ``target`` parameters: the width (a multiple of 8) closest to the target, or
    ``width`` if given."""
    if width is None:
        width = min(range(64, 513, 8), key=lambda w: abs(count(_net(history, state_dim, action_dim, w)) - target))
    return {"trunk": "transformer", "width": width, "mlp_hidden": None}


def build(config, history, state_dim, action_dim):
    return _net(history, state_dim, action_dim, config["width"])


def load(packet):
    """The model of a checkpoint dict, behind the contact NRD's calls (``one_net.OneNetNRD``)."""
    return load_one(packet, build)


def train(args):
    from nedm.contact_nrd.arch_ablation import one_train   # it imports this module (through checkpoints)

    one_train.train_one(args)
