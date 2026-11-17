"""What more than one architecture of the ablation uses: the residual MLP (3M and 1M) and the motion score.

The residual MLP replaces a transformer trunk, the part that mixes the tokens. Everything around a trunk, the token
embeddings, the heads, the normalisation, the contact switches and ``assemble``, is the T model's own.
"""
from __future__ import annotations

import math

from torch import nn
from torch.nn import functional as F

NETWORKS = ("core", "collision", "contact")
MLP_BLOCKS = 2


def count(module):
    return sum(p.numel() for p in module.parameters())


# ---- the residual MLP ------------------------------------------------------------------------------------------
class ResidualMLP(nn.Module):
    """Flattened tokens [N, T, d] -> features [N, 1, d].

    ``Linear(T*d -> w)``, ``blocks`` x ``x + Linear(h -> w)(GELU(Linear(w -> h)(LN(x))))``, ``Linear(w -> d)``,
    ``LayerNorm(d)``. Tokens marked invalid are zeroed before flattening.
    """

    def __init__(self, tokens, d, width, hidden, blocks, bias, scaled_init):
        super().__init__()
        self.inp = nn.Linear(tokens * d, width, bias=bias)
        self.blocks = nn.ModuleList(
            nn.ModuleDict(dict(ln=nn.LayerNorm(width), fc1=nn.Linear(width, hidden, bias=bias),
                               fc2=nn.Linear(hidden, width, bias=bias))) for _ in range(blocks))
        self.out = nn.Linear(width, d, bias=bias)
        self.ln = nn.LayerNorm(d)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02 / (math.sqrt(2 * blocks) if scaled_init and name.endswith("fc2") else 1.0)
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, ok=None):
        if ok is not None:
            x = x * ok[..., None].to(x.dtype)
        x = self.inp(x.flatten(1))
        for blk in self.blocks:
            x = x + blk["fc2"](F.gelu(blk["fc1"](blk["ln"](x))))
        return self.ln(self.out(x))[:, None]


def solve_hidden(target, tokens, d, width, blocks, bias):
    """The hidden width giving a ResidualMLP closest to ``target`` parameters."""
    def size(h):
        return count(ResidualMLP(tokens, d, width, h, blocks, bias, False))
    base = size(1)
    per = size(2) - base
    return max(1, round((target - base) / per) + 1)


# ---- the motion score ------------------------------------------------------------------------------------------
FREE_EVENT_WEIGHT = 0.05
"""The weight of ``1 - event_ok_fraction`` in ``data.free_metrics``' score."""


def is_contact_term(term):
    """A score term computed from the predicted contact switches."""
    return "/contacts" in term["stat"]


def motion_score(report):
    """A report's selection score without its contact-event terms: the score every architecture can be compared
    on, since a one-network model predicts no contacts."""
    if "score_terms" in report:
        return sum(t["term"] for t in report["score_terms"] if not is_contact_term(t))
    return report["selection_score"] - FREE_EVENT_WEIGHT * (1 - report["event_ok_fraction"])
