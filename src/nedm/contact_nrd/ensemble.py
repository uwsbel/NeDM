"""Average several trained state-only models that share the same frozen core.

Each step: the collision logits of the members are averaged (one switch per pair),
then the contact changes of the members are averaged for the switched-on pairs.
The core change is the members' common core (identical weights, checked).
Same rollout interface as the single models, so free_metrics and the certifier
work unchanged. Score with:  python -m nedm.contact_nrd.ensemble --data D --runs R1 R2 R3 --output O
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch import nn

from nedm.contact_nrd.model_v2 import load_any


class EnsembleNRD(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        first = members[0]
        for m in members[1:]:
            for (k, a), b in zip(first.backbone.state_dict().items(), m.backbone.state_dict().values()):
                if not torch.equal(a, b):
                    raise ValueError(f"members do not share the core ({k})")
        self.dt, self.system, self.moving, self.context = first.dt, first.system, first.moving, first.context
        self.state_mean = first.state_mean
        self.k = max(m.k for m in members)

    def initial_history(self, state):
        n = max(self.k, self.context)
        return state.unsqueeze(-3).expand(*state.shape[:-2], n, *state.shape[-2:]).clone()

    def advance_history(self, history, state):
        return torch.cat((history[..., 1:, :, :], state.unsqueeze(-3)), -3)

    def details(self, state, history=None, gates=None):
        history = self.initial_history(state) if history is None else history
        first = self.members[0]
        d_core = first.core_delta(state, history)
        logits = torch.stack([m.pair_logits(m.scaled_rows(history[..., -max(m.k, m.context):, :, :], "gate"))
                              for m in self.members]).mean(0)
        g = first.gates(logits) if gates is None else gates.to(state.dtype)
        change = torch.stack([m.contact_sum(m.scaled_rows(history[..., -max(m.k, m.context):, :, :], "contact"), g[..., m.row_pair])
                              for m in self.members]).mean(0)
        return first.next_state(state, d_core, change), {"logits": logits, "gates": g, "core": d_core}

    def forward(self, state, history=None):
        return self.details(state, history)[0]

    @torch.no_grad()
    def rollout(self, initial, steps, return_gates=False):
        state = initial.to(self.state_mean.dtype)
        history = self.initial_history(state)
        trajectory, gate_trace = [state], []
        for _ in range(steps):
            state, info = self.details(state, history)
            history = self.advance_history(history, state)
            trajectory.append(state)
            gate_trace.append(info["gates"])
        out = torch.stack(trajectory, -3)
        return (out, torch.stack(gate_trace, -2)) if return_gates else out


class EnsembleV2(nn.Module):
    """The same averaging for pair-feature (version-2) models: mean collision logits, mean contact changes."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        first = members[0]
        self.dt, self.system, self.moving, self.context = first.dt, first.system, first.moving, first.context
        self.state_mean = first.state_mean

    def initial_history(self, state):
        return self.members[0].initial_history(state)

    def advance_history(self, history, state):
        return self.members[0].advance_history(history, state)

    def details(self, state, history=None, gates=None):
        first = self.members[0]
        history = first.initial_history(state) if history is None else history
        d_core = first.core_delta(state, history)
        terms = [m.contact_terms(state) for m in self.members]
        logits = torch.stack([t[0] for t in terms]).mean(0)
        d_first = torch.stack([t[1] for t in terms]).mean(0)
        d_second = torch.stack([t[2] for t in terms]).mean(0) if terms[0][2] is not None else None
        g = first.gates(logits) if gates is None else gates.to(state.dtype)
        nxt = state + d_core + first.scatter(state, g, d_first, d_second, terms[0][3])
        return nxt, {"logits": logits, "gates": g, "core": d_core}

    forward = EnsembleNRD.forward
    rollout = EnsembleNRD.rollout


class EnsembleV5(nn.Module):
    """Averaging for version-5 models (each member has its own core): mean core change, mean collision logits
    (one switch per pair), mean contact changes for the switched-on pairs."""

    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        first = members[0]
        self.dt, self.system, self.moving = first.dt, first.system, first.moving
        self.k = max(m.k for m in members)

    def start(self, state):
        hist = torch.zeros(state.shape[0], self.k, *state.shape[1:], dtype=state.dtype, device=state.device)
        hist[:, -1] = state
        ok = torch.zeros(state.shape[0], self.k, dtype=torch.bool, device=state.device)
        ok[:, -1] = True
        return hist, ok

    def details(self, hist, ok, gates=None):
        coded = [m.code(hist[:, -m.k:], ok[:, -m.k:]) for m in self.members]
        core = torch.stack([m.run_core(*c) for m, c in zip(self.members, coded)]).mean(0)
        logits = torch.stack([m.logits(*c) for m, c in zip(self.members, coded)]).mean(0)
        cur = coded[0][0]
        g = (logits >= 0).to(cur.dtype) if gates is None else gates.to(cur.dtype)
        change = torch.stack([m.contact_sum(*c, g) for m, c in zip(self.members, coded)]).mean(0)
        return cur + core + change, {"logits": logits, "gates": g, "core": core}

    def forward(self, hist, ok):
        return self.details(hist, ok)[0]

    def rollout(self, initial, steps, return_gates=False):
        from nedm.contact_nrd.model_v5 import UnifiedNRD
        state = initial.to(torch.float64)
        hist, ok = self.start(state)
        trajectory, gate_trace = [state], []
        for _ in range(steps):
            state, info = self.details(hist, ok)
            hist, ok = UnifiedNRD.advance(hist, ok, state)
            trajectory.append(state)
            gate_trace.append(info["gates"])
        out = torch.stack(trajectory, -3)
        return (out, torch.stack(gate_trace, -2)) if return_gates else out


def load_ensemble(runs, device="cpu"):
    members = [load_any(Path(r) / "best.pt", device)[0] for r in runs]
    if hasattr(members[0], "pair_pos"):
        return EnsembleV5(members).eval()
    if hasattr(members[0], "row_pair"):
        return EnsembleNRD(members).eval()
    return EnsembleV2(members).eval()


def main():
    from nedm.contact_nrd.evaluate import free_metrics, load_data
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("SLURM_JOB_ID")
    torch.set_num_threads(1)
    data = load_data(args.data, "cuda")
    val = torch.nonzero(data["splits"] == 1).flatten()
    rows = {}
    for label, runs in [(Path(r).name, [r]) for r in args.runs] + [("ensemble", args.runs)]:
        model = load_ensemble(runs, "cuda")
        rep = free_metrics(model, data, val)
        rows[label] = {"score": rep["selection_score"], "eligible_p95_mm": 1e3 * rep["eligible_at_target_m"]["p95"],
                       "at_target_median_mm": 1e3 * rep["at_target_m"]["median"], "rmse_p95_mm": 1e3 * rep["rmse_m"]["p95"],
                       "end_p95_mm": 1e3 * rep["end_m"]["p95"], "events_ok": rep["event_ok_fraction"]}
        print(json.dumps({label: rows[label]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
