from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from nedm.training.model_transformer import ContinuousTransformer, TransformerConfig


class HMMWVDynamicsModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        target_dim: int,
        transformer_cfg: dict[str, Any],
        normalization: dict[str, list[float]],
        num_terrains: int = 0,
        state_fields: list[str] | None = None,
        dt_s: float | None = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.target_dim = target_dim
        # Input-channel ablation: ``blind_state_fields`` lists state channels the
        # network never sees (they are dropped from the token before the input
        # projection) while the state/target layout is left untouched. Used to test
        # e.g. a qd-only arm ROM whose q is propagated kinematically outside the net
        # (see ``integrated_state_fields`` below). Default: see everything.
        blind_fields = list(transformer_cfg.get("blind_state_fields", []) or [])
        if blind_fields:
            if state_fields is None:
                raise ValueError("blind_state_fields requires the state_fields list to resolve channel indices")
            unknown = [field for field in blind_fields if field not in state_fields]
            if unknown:
                raise ValueError(f"blind_state_fields not in state_fields: {unknown}")
        keep = [index for index, field in enumerate(state_fields or [None] * state_dim) if field not in blind_fields]
        # non-persistent: derived from the config, and older checkpoints must keep loading.
        self.register_buffer("input_state_indices", torch.tensor(keep, dtype=torch.long), persistent=False)
        self.blind_state_fields = blind_fields
        # Kinematic propagation: ``integrated_state_fields`` maps a position-like
        # channel to its rate channel ({"q_0": "qd_0", ...}). The network's delta for
        # that channel is replaced by the trapezoid integral of the (predicted) rate in
        # ``predict_delta``/``predict_next_delta``, so every consumer (trainer rollout,
        # RL envs, eval scripts) propagates it the same way. Pair the loss weight of
        # such channels with 0 -- their head output is unused.
        integrated = dict(transformer_cfg.get("integrated_state_fields", {}) or {})
        self.integrated_pairs: list[tuple[int, int]] = []
        if integrated:
            if state_fields is None or dt_s is None:
                raise ValueError("integrated_state_fields requires state_fields and dt_s")
            self.integrated_pairs = [(state_fields.index(pos), state_fields.index(rate)) for pos, rate in integrated.items()]
        self.dt_s = float(dt_s) if dt_s is not None else None
        # Terrain conditioning: when num_terrains > 0 a one-hot terrain code is
        # concatenated to every (state, action) token so a shared backbone can make
        # terrain-specific predictions (rigid: vx≈ωR; CRM: vx<ωR + sinkage). 0 keeps
        # the original unconditioned token and is fully backward-compatible.
        self.num_terrains = int(num_terrains)
        self.backbone = ContinuousTransformer(
            TransformerConfig(
                input_dim=len(keep) + action_dim + self.num_terrains,
                block_size=int(transformer_cfg["block_size"]),
                n_layer=int(transformer_cfg["n_layer"]),
                n_head=int(transformer_cfg["n_head"]),
                n_embd=int(transformer_cfg["n_embd"]),
                dropout=float(transformer_cfg["dropout"]),
                bias=bool(transformer_cfg["bias"]),
            )
        )
        hidden_dim = int(transformer_cfg.get("head_hidden_dim", self.backbone.config.n_embd))
        self.head = nn.Sequential(
            nn.Linear(self.backbone.config.n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, target_dim),
        )
        self.register_buffer("state_mean", torch.tensor(normalization["state_mean"], dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(normalization["state_std"], dtype=torch.float32))
        self.register_buffer("action_mean", torch.tensor(normalization["action_mean"], dtype=torch.float32))
        self.register_buffer("action_std", torch.tensor(normalization["action_std"], dtype=torch.float32))
        self.register_buffer("target_mean", torch.tensor(normalization["target_mean"], dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(normalization["target_std"], dtype=torch.float32))

    def normalize_state(self, states: torch.Tensor) -> torch.Tensor:
        return (states - self.state_mean) / self.state_std

    def normalize_action(self, actions: torch.Tensor) -> torch.Tensor:
        return (actions - self.action_mean) / self.action_std

    def normalize_target(self, targets: torch.Tensor) -> torch.Tensor:
        return (targets - self.target_mean) / self.target_std

    def denormalize_target(self, normalized_targets: torch.Tensor) -> torch.Tensor:
        return normalized_targets * self.target_std + self.target_mean

    def _terrain_one_hot(
        self, terrain: torch.Tensor | int, batch: int, sequence: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Build a (batch, sequence, num_terrains) one-hot from a terrain id.

        Accepts a python int (whole batch shares a terrain), a 0-d tensor, a
        per-sample (batch,) id tensor, or an already per-token (batch, sequence)
        id tensor.
        """
        if isinstance(terrain, int):
            ids = torch.full((batch, sequence), terrain, dtype=torch.long, device=device)
        else:
            ids = terrain.to(device=device, dtype=torch.long)
            if ids.dim() == 0:
                ids = ids.view(1, 1).expand(batch, sequence)
            elif ids.dim() == 1:  # (batch,) per-sample
                ids = ids.view(batch, 1).expand(batch, sequence)
            elif ids.dim() != 2:  # not (batch, sequence)
                raise ValueError(f"terrain id tensor must be 0/1/2-D, got shape {tuple(terrain.shape)}")
        return torch.nn.functional.one_hot(ids, num_classes=self.num_terrains).to(dtype)

    def _build_tokens(
        self, states: torch.Tensor, actions: torch.Tensor, terrain: torch.Tensor | int | None
    ) -> torch.Tensor:
        state_tokens = self.normalize_state(states)
        if self.blind_state_fields:
            state_tokens = state_tokens.index_select(-1, self.input_state_indices)
        tokens = torch.cat([state_tokens, self.normalize_action(actions)], dim=-1)
        if self.num_terrains > 0:
            if terrain is None:
                raise ValueError("this model is terrain-conditioned; a terrain id must be supplied")
            one_hot = self._terrain_one_hot(
                terrain, tokens.shape[0], tokens.shape[1], tokens.device, tokens.dtype
            )
            tokens = torch.cat([tokens, one_hot], dim=-1)
        return tokens

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor, terrain: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        features = self.backbone(self._build_tokens(states, actions, terrain))
        return self.head(features)

    def apply_kinematic_integration(self, states: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """Override integrated channels' deltas with the trapezoid integral of their rate.

        ``states`` is the state the delta is applied to; both share the trailing
        state dimension. No-op unless ``integrated_state_fields`` is configured.
        """
        if not self.integrated_pairs:
            return delta
        delta = delta.clone()
        for pos_index, rate_index in self.integrated_pairs:
            rate_now = states[..., rate_index]
            rate_next = rate_now + delta[..., rate_index]
            delta[..., pos_index] = 0.5 * (rate_now + rate_next) * self.dt_s
        return delta

    def predict_delta(
        self, states: torch.Tensor, actions: torch.Tensor, terrain: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        return self.apply_kinematic_integration(states, self.denormalize_target(self.forward(states, actions, terrain)))

    def predict_next_delta(
        self, states: torch.Tensor, actions: torch.Tensor, terrain: torch.Tensor | int | None = None
    ) -> torch.Tensor:
        features = self.backbone(self._build_tokens(states, actions, terrain))
        delta = self.denormalize_target(self.head(features[:, -1, :]))
        return self.apply_kinematic_integration(states[:, -1, :], delta)
