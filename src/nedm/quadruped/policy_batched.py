"""Batched, explicit-state reimplementation of the exported Go2 policy.

The TorchScript export is BATCH-1 ONLY: `self.history` is (1,5,45) and the forward
does torch.cat([history[:,1:], x.unsqueeze(1)]), which fails for B>1. It also mutates
module state, which makes it awkward to differentiate through cleanly.

This carries the FIFO explicitly instead. Reproduces the export EXACTLY -- verified,
not assumed -- and the weights are the same tensors, so gradients reach the same
parameters.

Forward, read off the export's own code:
    history <- cat([history[:, 1:], obs.unsqueeze(1)], 1)
    latent  <- student_encoder(history.flatten(1))
    out     <- actor(cat([latent, obs], 1))
`normalizer` is the identity (verified from its source), so it is omitted.
"""
import torch

class BatchedGo2Policy(torch.nn.Module):
    def __init__(self, ts_module):
        super().__init__()
        self.student_encoder = ts_module.student_encoder
        self.actor = ts_module.actor
        self.hist_len = ts_module.history.shape[1]
        self.obs_dim = ts_module.history.shape[2]

    def initial_history(self, batch, device):
        return torch.zeros(batch, self.hist_len, self.obs_dim, device=device)

    def forward(self, obs, history):
        history = torch.cat([history[:, 1:], obs.unsqueeze(1)], dim=1)
        latent = self.student_encoder(history.flatten(1))
        return self.actor(torch.cat([latent, obs], dim=1)), history
