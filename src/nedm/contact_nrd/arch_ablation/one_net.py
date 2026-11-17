"""What the 1T and the 1M share: the network around the backbone, and the adapter that puts a one-network model behind
the calls that the contact NRD's training, checks and test make.

The two differ only in the backbone (``arch_1t.CausalBackbone``, ``arch_1m.MLPBackbone``).
"""
from __future__ import annotations

import torch
from torch import nn

from nedm.contact_nrd.arch_ablation.common import is_contact_term, motion_score
from nedm.contact_nrd.model import ContactNRD


def normal_init(module):
    """normal(0, 0.02) for every Linear and Embedding (the 1T's convention)."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)


def positions(valid):
    """Positions count from each row's first valid step (a left-padded history)."""
    return (torch.cumsum(valid.long(), dim=1) - 1).clamp_(min=0)


class OneNet(nn.Module):
    """One network over ``[state_t, command_t]`` step tokens: a backbone (``arch_1t.CausalBackbone`` or
    ``arch_1m.MLPBackbone``, already initialised) and one head. The head on the newest token's features gives the
    change of every padded channel (zero outside the mask). Inputs and the output change are normalised by
    statistics of training windows, held as buffers."""

    def __init__(self, backbone, state_dim, action_dim, width):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(width, width, bias=False), nn.GELU(), nn.Linear(width, state_dim, bias=False))
        self.head.apply(normal_init)
        for name, size in (("state_mean", state_dim), ("state_std", state_dim), ("action_mean", action_dim),
                           ("action_std", action_dim), ("target_mean", state_dim), ("target_std", state_dim),
                           ("smask", state_dim)):
            self.register_buffer(name, torch.full((size,), 0.0 if name.endswith("mean") else 1.0, dtype=torch.float32))

    def set_statistics(self, states, actions, targets, smask):
        """Statistics from training windows: states, actions, one-step changes [N, .]."""
        def std(x):
            return x.std(0).clamp_min(1e-6)
        on = smask > 0
        self.state_mean.copy_(torch.where(on, states.mean(0), 0.0))
        self.state_std.copy_(torch.where(on, std(states), 1.0))
        self.action_mean.copy_(actions.mean(0))
        self.action_std.copy_(std(actions))
        self.target_mean.copy_(torch.where(on, targets.mean(0), 0.0))
        self.target_std.copy_(torch.where(on, std(targets), 1.0))
        self.smask.copy_(smask)

    def features(self, states, actions, valid):
        """The newest token's features [B, d] from states [B, K, S] and actions [B, K, A]."""
        tokens = torch.cat(((states.float() - self.state_mean) / self.state_std * self.smask,
                            (actions.float() - self.action_mean) / self.action_std), -1)
        return self.backbone(tokens, valid)[:, -1]

    def change_of(self, features):
        """The change [B, S] after the newest state."""
        return (self.head(features) * self.target_std + self.target_mean) * self.smask


class OneNetNRD(nn.Module):
    """A 1T or 1M behind the interface that the contact NRD's training, checks and test call.

    Those callers pass only the current command, so the commands before it are held in ``past`` [B, K-1, A]: they
    are set for each batch and moved along in a rollout. Anything not defined here (``dt``, ``k``, ``moving``,
    ``code``, ``advance``, ``start``, ``assemble``, ...) is read from ``three``, a ContactNRD of the same system whose
    three networks never enter a prediction. The D moving bodies of a step are one token; the command is body 0's.
    """

    def __init__(self, one, three):
        super().__init__()
        if bool(three.has_action[1:].any()):
            raise ValueError("the one network reads the commands of moving body 0 only")
        self.one, self.three = one, three
        self.past = None

    THREE_ONLY = ("details", "run_core", "logits", "contact_sum")
    """Calls that would run the untrained three networks: not available on a one-network model."""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in self.THREE_ONLY:
                raise AttributeError(f"a one-network model has no {name}: it predicts no contacts") from None
            return getattr(super().__getattr__("three"), name)

    def set_past(self, data, episode, time):
        """The recorded commands of the K-1 model steps before ``time`` (record rows); invalid history entries are
        masked by the network, so rows before the start are clamped."""
        rows = time[:, None] + torch.arange(1 - self.k, 0, device=time.device) * data["stride"]
        self.past = data["actions"][episode[:, None], rows.clamp_min(0), 0]

    def change(self, hist, ok, act, noise=None):
        """The network's change [B, D, S] from hist [B, K, D, S] and the commands [B, D, A]; the noise goes on the
        history differences and leaves the newest state exact, as in the 3T."""
        cur, diff, _ = self.three.code(hist, ok, noise)
        older = (cur[:, None] - diff.cumsum(1)).flip(1)
        states = torch.cat((older, cur[:, None]), 1) * ok[..., None, None]
        actions = torch.cat((self.past, act[:, :1, :].to(self.past.dtype)), 1)
        change = self.one.change_of(self.one.features(states.flatten(2), actions, ok))
        return change.to(hist.dtype).view(cur.shape)

    def next_state(self, hist, ok, act, noise=None):
        """The 3T's step with this network's change as its whole change: the integrated poses follow the new rates."""
        d = self.change(hist, ok, act, noise)
        return self.three.assemble(hist[:, -1], d, torch.zeros_like(d))

    def step_with_gates(self, hist, ok, act, gates, noise=None, core_grad=False):
        """The contact term's prediction: the switches select windows and are not an input."""
        return self.next_state(hist, ok, act, noise)

    def forward(self, hist, ok, act):
        state = self.next_state(hist, ok, act)
        self.past = torch.cat((self.past[:, 1:], act[:, :1, :].to(self.past.dtype)), 1)
        return state

    def rollout_from(self, hist, ok, actions, return_gates=False):
        if self.past is None or len(self.past) != len(hist):
            raise RuntimeError("the commands before this rollout were not set (set_past)")
        hist = hist.to(torch.float64)
        trajectory = [hist[:, -1]]
        try:
            for t in range(actions.shape[1]):
                state = self(hist, ok, actions[:, t].to(torch.float64))
                hist, ok = self.advance(hist, ok, state)
                trajectory.append(state)
        finally:
            self.past = None   # never reused by a later rollout
        out = torch.stack(trajectory, 1)
        if not return_gates:
            return out
        # No contact predictions: the switches are all off. The motion score does not read them.
        return out, torch.zeros(len(out), actions.shape[1], self.three.n_pairs, device=out.device)

    def rollout(self, initial, actions, return_gates=False):
        """From a first state: no earlier entry is valid, so the earlier commands are not read."""
        initial = initial.to(torch.float64)
        if isinstance(actions, int):
            actions = torch.zeros(len(initial), actions, self.n_bodies, self.amax, dtype=initial.dtype, device=initial.device)
        self.past = torch.zeros(len(initial), self.k - 1, actions.shape[-1], dtype=torch.float64, device=initial.device)
        return self.rollout_from(*self.start(initial), actions, return_gates)


def load_one(packet, build):
    """The model of a 1T / 1M checkpoint dict; ``build`` is the architecture's (``arch_1t.build``, ``arch_1m.build``).
    The ContactNRD behind it is rebuilt from the checkpoint's system and is not loaded: it enters no prediction."""
    three = ContactNRD(packet["model_config"], packet["system"], packet["masks"])
    one = build(packet["one_config"], three.k, three.n_bodies * three.smax, three.amax)
    one.load_state_dict(packet["model_state_dict"])
    return OneNetNRD(one, three)


# ---- the motion score of a one-network model -------------------------------------------------------------------
class _RecordedActions:
    """``data["actions"]`` for ``data.evaluate``: its (episode, time) lookup of a rollout from a recorded history
    also hands the network the commands before the start."""

    def __init__(self, net, data):
        self.net, self.actions, self.stride = net, data["actions"], data["stride"]

    def __getitem__(self, index):
        if isinstance(index, tuple) and len(index) == 2:   # (episode, time grid) of a rollout
            episode, grid = index
            self.net.set_past({"actions": self.actions, "stride": self.stride}, episode[:, 0], grid[:, 0])
        return self.actions[index]

    def __getattr__(self, name):
        return getattr(self.actions, name)


def motion_evaluate(evaluate, net):
    """``data.evaluate`` for a one-network model: scored by the motion terms only, and the earlier commands
    supplied to the network."""
    def wrapped(model, data, indices, **kwargs):
        system = data["system"]
        if "evaluation" in system:
            spec = system["evaluation"]
            system = {**system, "evaluation": {**spec, "score": [t for t in spec["score"] if not is_contact_term(t)]}}
        report = evaluate(model, {**data, "system": system, "actions": _RecordedActions(net, data)}, indices, **kwargs)
        if "evaluation" not in system:   # free rollouts: also the reports of the families, if any
            for part in (report, report.get("selected"), *report.get("families", {}).values()):
                if part is not None:
                    part["selection_score"] = motion_score(part)
        return report
    return wrapped
