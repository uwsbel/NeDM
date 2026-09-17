"""The deployed risk network, standalone (definition identical to scripts/f104_n2_train.py Net + route_logit).

Station-preserving CNN over a 6x96x32 corridor -> lateral mean+max -> per-station features, concatenated with a
32-d context embedding and the normalised station position -> Conv1d(k=5) -> BiGRU (or transformer / MLP)
-> one hazard logit per station. Route risk = 1 - exp(-sum softplus(hazard)).
"""
import torch, torch.nn as nn, torch.nn.functional as F


def route_logit(haz):
    """log cumulative hazard == cloglog(P(event anywhere on the route))."""
    return torch.log(F.softplus(haz).sum(1) + 1e-6)


class Net(nn.Module):
    def __init__(self, cin, nctx, arch='gru', width=64, layers=2, heads=4):
        super().__init__()
        c = [cin, 32, 64, 64, 96]
        L = []
        for i in range(4):
            L += [nn.Conv2d(c[i], c[i+1], 3, stride=(1, 1 if i == 0 else 2), padding=1),
                  nn.BatchNorm2d(c[i+1]), nn.GELU()]
        self.cnn = nn.Sequential(*L)
        self.lat = nn.Linear(2 * c[-1], 96)
        self.ctx = nn.Sequential(nn.Linear(nctx, 32), nn.GELU())
        self.arch = arch
        k = 1 if arch == 'mlp' else 5
        self.tconv = nn.Sequential(nn.Conv1d(96 + 32 + 1, 96, k, padding=k // 2), nn.GELU(), nn.Dropout(0.1))
        if arch == 'gru':
            self.mix = nn.GRU(96, width, batch_first=True, bidirectional=True)
            self.head = nn.Linear(2 * width, 1)
        elif arch == 'tx':
            self.pos = nn.Parameter(torch.zeros(1, 96, 96)); nn.init.normal_(self.pos, std=0.02)
            enc = nn.TransformerEncoderLayer(96, heads, 2 * 96, dropout=0.1, batch_first=True,
                                             norm_first=True, activation='gelu')
            self.mix = nn.TransformerEncoder(enc, layers)
            self.head = nn.Linear(96, 1)
        else:
            self.mix = nn.Sequential(nn.Linear(96, 2 * width), nn.GELU(), nn.Linear(2 * width, 2 * width), nn.GELU())
            self.head = nn.Linear(2 * width, 1)

    def forward(self, x, ctx):
        f = self.cnn(x)
        f = torch.cat([f.mean(-1), f.amax(-1)], 1).transpose(1, 2)
        f = F.gelu(self.lat(f))
        B, S, _ = f.shape
        pos = torch.linspace(0, 1, S, device=x.device)[None, :, None].expand(B, S, 1)
        c = self.ctx(ctx)[:, None, :].expand(B, S, 32)
        h = torch.cat([f, c, pos], -1).transpose(1, 2)
        h = self.tconv(h).transpose(1, 2)
        if self.arch == 'gru':
            h, _ = self.mix(h)
        elif self.arch == 'tx':
            h = self.mix(h + self.pos)
        else:
            h = self.mix(h)
        return self.head(h).squeeze(-1)
