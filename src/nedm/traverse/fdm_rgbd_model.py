"""RGB-D adaptation of the perceptive FDM's finite-horizon forward GRU.

This is an HMMWV/PID adaptation, not a reproduction of the ANYmal network.
One measured RGB-D observation and causal history condition a forward GRU over
the proposed path/speed commands. The observation stays fixed during prediction;
there is no future-image prediction and no authored terrain/clearance input.

The fourth image channel is registered elevation reconstructed from observed
ray depth and calibrated fixed-camera geometry, as specified by the data
manifest. It is not a ground-truth terrain map. Indexed spatial image features
are flattened rather than globally pooled, preserving where image evidence is.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from nedm.traverse.fdm_model import EVENT_NAMES, fdm_loss


RGBD_ARMS = ("rgbd", "rgb_only", "depth_only", "blank")
PROGRESS_TARGET_VERSIONS = {"net_progress": 1, "sustained_stall": 1, "bounded_motion": 2}


@dataclass
class RGBDFDMConfig:
    arm: str = "rgbd"
    history_dim: int = 24
    command_dim: int = 5
    global_dim: int = 8
    image_size: int = 128
    horizon: int = 20
    dt: float = 0.2
    history_hidden: int = 64
    visual_hidden: int = 128
    command_hidden: int = 32
    forward_hidden: int = 128
    forward_layers: int = 2
    decoder_hidden: int = 256
    cnn_channels: int = 32
    work_unit: str = "kJ"
    velocity_residual_scale: tuple[float, float, float] = (2.0, 1.0, 0.5)
    candidate_patches: bool = False
    patch_size: int = 16
    patch_span_m: float = 8.0
    patch_hidden: int = 64
    patch_depth_iterations: int = 2
    camera_hfov_deg: float = 47.0
    camera_height_m: float = 100.0
    progress_event_definition: str = "net_progress"
    progress_event_version: int | None = None

    def __post_init__(self) -> None:
        if self.arm not in RGBD_ARMS:
            raise ValueError(f"Unknown RGB-D arm {self.arm!r}")
        if self.image_size < 16 or self.image_size % 16:
            raise ValueError("image_size must be a positive multiple of 16")
        if self.horizon < 1 or self.dt <= 0:
            raise ValueError("horizon and dt must be positive")
        if self.patch_size < 4 or self.patch_size % 4 or self.patch_span_m <= 0:
            raise ValueError("patch_size must be a positive multiple of four and patch_span_m positive")
        if self.patch_depth_iterations < 0 or self.camera_height_m <= 10.0 or not 0 < self.camera_hfov_deg < 180:
            raise ValueError("Invalid patch projection calibration")
        if self.progress_event_definition not in PROGRESS_TARGET_VERSIONS:
            raise ValueError("Unknown progress-event target definition")
        expected_version = PROGRESS_TARGET_VERSIONS[self.progress_event_definition]
        if self.progress_event_version is None:
            self.progress_event_version = expected_version
        elif self.progress_event_version != expected_version:
            raise ValueError("Progress-event definition and semantic version disagree")


class RGBDFiniteHorizonFDM(nn.Module):
    """Measured image/history → command-conditioned forward GRU → outcomes.

    Physical input tensors:
      rgbd[B,4,128,128]: RGB 0..1 plus calibrated observed-depth channel;
      history[B,16,24]: causal vehicle state and preceding applied controls;
      commands[B,20,5]: nominal ego x/y, sin/cos heading, desired speed;
      global_features[B,8]: route/driver context and measured anchor pose;
      nominal_pose[B,20,4]: nominal x/y/sin/cos in the anchor vehicle frame.

    Outputs are trajectory[B,20,4], cumulative positive work[B,20,1] in kJ,
    and event_logits[B,20,3]. Class weighting is corrected by the shared loss;
    probabilities still require measured validation calibration.

    All modality arms have identical parameters. Excluded image channels are
    set to zero *after* fixed pixel normalization. Forward controls ``blank``
    and ``current_history`` are evaluation interventions, not extra trained arms.
    """

    def __init__(self, config: RGBDFDMConfig, normalization: dict[str, Any] | None = None):
        super().__init__()
        self.config = config
        norm = normalization or {}
        for key, dim in (("history", config.history_dim), ("commands", config.command_dim),
                         ("global_features", config.global_dim)):
            stats = norm.get(key, {})
            mean = torch.as_tensor(stats.get("mean", [0.0] * dim), dtype=torch.float32)
            std = torch.as_tensor(stats.get("std", [1.0] * dim), dtype=torch.float32)
            if mean.shape != (dim,) or std.shape != (dim,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all():
                raise ValueError(f"Invalid {key} training normalization")
            self.register_buffer(f"{key}_mean", mean)
            self.register_buffer(f"{key}_std", std.clamp_min(1e-6))
        self.register_buffer("pixel_mean", torch.tensor([0.5, 0.5, 0.5, 0.0]).reshape(1, 4, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.5, 0.5, 0.5, 1.0]).reshape(1, 4, 1, 1))
        self.register_buffer("xy_scale", torch.tensor(float(norm.get("xy_scale", 1.0))).clamp_min(0.1))
        self.register_buffer("work_scale", torch.tensor(float(norm.get("work_scale", 1.0))).clamp_min(1.0))
        self.register_buffer("event_pos_weight", torch.as_tensor(norm.get("event_pos_weight", [1.0] * 3), dtype=torch.float32))
        self.register_buffer("supported_events", torch.as_tensor(norm.get("supported_events", [True] * 3), dtype=torch.bool))
        self.register_buffer("velocity_scale", torch.tensor(config.velocity_residual_scale, dtype=torch.float32))
        if self.event_pos_weight.shape != (3,) or self.supported_events.shape != (3,) or (self.event_pos_weight <= 0).any():
            raise ValueError("Three positive event weights and three support flags are required")
        c = config.cnn_channels
        self.image_encoder = nn.Sequential(
            nn.Conv2d(4, c, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(c * 2, c * 2, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(c * 2, c * 2, 3, stride=2, padding=1), nn.SiLU(),
        )
        spatial_dim = (config.image_size // 16) ** 2 * c * 2
        self.image_projection = nn.Sequential(nn.Flatten(), nn.Linear(spatial_dim, config.visual_hidden), nn.SiLU())
        self.history_encoder = nn.GRU(config.history_dim, config.history_hidden, batch_first=True)
        self.command_encoder = nn.Sequential(nn.Linear(config.command_dim, config.command_hidden), nn.SiLU())
        context_dim = config.visual_hidden + config.history_hidden + config.global_dim
        patch_dim = 0
        if config.candidate_patches:
            self.patch_encoder = nn.Sequential(
                nn.Conv2d(4, 16, 3, padding=1), nn.SiLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.SiLU(), nn.Flatten(),
                nn.Linear(32 * (config.patch_size // 4) ** 2, config.patch_hidden), nn.SiLU(),
            )
            offsets = torch.linspace(config.patch_span_m / 2, -config.patch_span_m / 2, config.patch_size)
            forward, left = torch.meshgrid(offsets, offsets, indexing="ij")
            # meshgrid returns overlapping zero-stride views; owned storage is
            # required for state_dict copying when a patch checkpoint reloads.
            self.register_buffer("patch_forward_m", forward.contiguous())
            self.register_buffer("patch_left_m", left.contiguous())
            patch_dim = config.patch_hidden
        self.forward_gru = nn.GRU(context_dim + config.command_hidden + patch_dim, config.forward_hidden,
                                  num_layers=config.forward_layers, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(config.horizon * config.forward_hidden, config.decoder_hidden), nn.SiLU())
        # Per horizon: residual body vx, vy, yaw rate; positive work increment;
        # contact, rollover, low-progress logits. Decoder sees all GRU outputs,
        # as in the reference multi-step architecture.
        self.output_head = nn.Linear(config.decoder_hidden, config.horizon * 7)
        nn.init.normal_(self.output_head.weight, std=0.001)
        nn.init.zeros_(self.output_head.bias)
        with torch.no_grad():
            self.output_head.bias.reshape(config.horizon, 7)[:, 3] = -1.0

    def normalize_pixels(self, rgbd: torch.Tensor, control: str = "normal") -> torch.Tensor:
        """Apply the same modality intervention to global pixels and local crops."""
        if control not in ("normal", "blank", "current_history"):
            raise ValueError(f"Unknown intervention {control!r}")
        pixels = (rgbd.to(self.pixel_mean.dtype) - self.pixel_mean) / self.pixel_std
        if self.config.arm == "rgb_only":
            pixels = torch.cat((pixels[:, :3], torch.zeros_like(pixels[:, 3:])), dim=1)
        elif self.config.arm == "depth_only":
            pixels = torch.cat((torch.zeros_like(pixels[:, :3]), pixels[:, 3:]), dim=1)
        elif self.config.arm == "blank" or control == "blank":
            pixels = torch.zeros_like(pixels)
        # Evaluation blank must override the trained single-modality setting.
        if control == "blank":
            pixels = torch.zeros_like(pixels)
        return pixels

    def encode_image(self, rgbd: torch.Tensor, control: str = "normal") -> torch.Tensor:
        """Encode current pixels once; reusable across all candidate routes."""
        pixels = self.normalize_pixels(rgbd, control)
        return self.image_projection(self.image_encoder(pixels))

    def encode_history(self, history: torch.Tensor, control: str = "normal") -> torch.Tensor:
        """Encode measured history once; candidate commands do not enter here."""
        hist = (history - self.history_mean) / self.history_std
        if control == "current_history":
            hist = hist[:, -1:, :]
        _, state = self.history_encoder(hist)
        return state[-1]

    def context_from_features(self, visual: torch.Tensor, history_feature: torch.Tensor,
                              global_features: torch.Tensor) -> torch.Tensor:
        """Combine cached measurements with each candidate's route context."""
        global_context = (global_features - self.global_features_mean) / self.global_features_std
        return torch.cat((visual, history_feature, global_context), dim=-1)

    def encode_observation(self, rgbd: torch.Tensor, history: torch.Tensor, global_features: torch.Tensor,
                           control: str = "normal") -> torch.Tensor:
        return self.context_from_features(self.encode_image(rgbd, control), self.encode_history(history, control), global_features)

    def candidate_patch_pixels(self, rgbd: torch.Tensor, nominal_pose: torch.Tensor,
                                global_features: torch.Tensor, control: str = "normal") -> torch.Tensor:
        """Route-oriented crops from current camera pixels, with no terrain query.

        The patch top points forward along the candidate heading; its left edge
        is vehicle-left. Projection starts at elevation zero and refines using
        only registered observed depth. RGB-only and blank arms use the camera
        ground-plane approximation so excluded depth cannot leak via crop position.
        Unknown depth and out-of-image samples keep the observed invalid marker.
        """
        if not self.config.candidate_patches:
            raise ValueError("Candidate patches are disabled in this checkpoint")
        batch, horizon = nominal_pose.shape[:2]
        if rgbd.shape[0] == 1 and batch != 1:
            rgbd = rgbd.expand(batch, -1, -1, -1)
        if rgbd.shape[0] != batch:
            raise ValueError("Image batch must be one shared observation or match candidate batch")
        pixels = rgbd.float()
        anchor_xy = global_features[:, 4:6] * 40.0
        anchor_s, anchor_c = global_features[:, 6], global_features[:, 7]
        centre_x = anchor_xy[:, None, 0] + anchor_c[:, None] * nominal_pose[..., 0] - anchor_s[:, None] * nominal_pose[..., 1]
        centre_y = anchor_xy[:, None, 1] + anchor_s[:, None] * nominal_pose[..., 0] + anchor_c[:, None] * nominal_pose[..., 1]
        relative_s, relative_c = nominal_pose[..., 2], nominal_pose[..., 3]
        heading_s = anchor_s[:, None] * relative_c + anchor_c[:, None] * relative_s
        heading_c = anchor_c[:, None] * relative_c - anchor_s[:, None] * relative_s
        world_x = centre_x[..., None, None] + heading_c[..., None, None] * self.patch_forward_m - heading_s[..., None, None] * self.patch_left_m
        world_y = centre_y[..., None, None] + heading_s[..., None, None] * self.patch_forward_m + heading_c[..., None, None] * self.patch_left_m
        height, width = pixels.shape[-2:]
        focal = (width / 2) / math.tan(math.radians(self.config.camera_hfov_deg) / 2)
        elevation = torch.zeros_like(world_x)

        def project(z: torch.Tensor) -> torch.Tensor:
            distance = (self.config.camera_height_m - z).clamp_min(1.0)
            u = (width - 1) / 2 + focal * world_x / distance
            v = (height - 1) / 2 - focal * world_y / distance
            return torch.stack((2 * u / (width - 1) - 1, 2 * v / (height - 1) - 1), dim=-1).reshape(
                batch, horizon * self.config.patch_size, self.config.patch_size, 2)

        observed_valid = (pixels[:, 3:4] > -1.5).float()
        geometry_uses_depth = self.config.arm not in ("rgb_only", "blank") and control != "blank"
        if geometry_uses_depth:
            for _ in range(self.config.patch_depth_iterations):
                grid = project(elevation)
                z = F.grid_sample(pixels[:, 3:4], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
                valid = F.grid_sample(observed_valid, grid, mode="bilinear", padding_mode="zeros", align_corners=True) >= 1 - 1e-5
                elevation = torch.where(valid, z.clamp(-1, 1) * 10.0, torch.zeros_like(z)).reshape_as(world_x)
        grid = project(elevation)
        sampled = F.grid_sample(pixels, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        valid = F.grid_sample(observed_valid, grid, mode="bilinear", padding_mode="zeros", align_corners=True) >= 1 - 1e-5
        sampled = torch.cat((sampled[:, :3], torch.where(valid, sampled[:, 3:4], -2 * torch.ones_like(sampled[:, 3:4]))), dim=1)
        patches = sampled.reshape(batch, 4, horizon, self.config.patch_size, self.config.patch_size).permute(0, 2, 1, 3, 4)
        flat = patches.reshape(batch * horizon, 4, self.config.patch_size, self.config.patch_size)
        return self.normalize_pixels(flat, control).reshape(batch, horizon, 4, self.config.patch_size, self.config.patch_size)

    def encode_candidate_patches(self, rgbd: torch.Tensor, nominal_pose: torch.Tensor,
                                  global_features: torch.Tensor, control: str = "normal") -> torch.Tensor:
        """Per-command image evidence supplied directly to the forward GRU."""
        patches = self.candidate_patch_pixels(rgbd, nominal_pose, global_features, control)
        batch, horizon = patches.shape[:2]
        features = self.patch_encoder(patches.reshape(batch * horizon, 4, self.config.patch_size, self.config.patch_size))
        return features.reshape(batch, horizon, self.config.patch_hidden)

    def predict_from_context(self, context: torch.Tensor, commands: torch.Tensor,
                             nominal_pose: torch.Tensor, patch_features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Score candidates using cached measured context; context may be expanded."""
        encoded = self.command_encoder((commands - self.commands_mean) / self.commands_std)
        repeated_context = context[:, None].expand(-1, self.config.horizon, -1)
        if self.config.candidate_patches:
            if patch_features is None or patch_features.shape != (*commands.shape[:2], self.config.patch_hidden):
                raise ValueError("This checkpoint requires candidate-aligned observed image features")
            encoded = torch.cat((encoded, patch_features), dim=-1)
        hidden, _ = self.forward_gru(torch.cat((encoded, repeated_context), dim=-1))
        raw = self.output_head(self.decoder(hidden.flatten(1))).reshape(-1, self.config.horizon, 7)
        nominal_twist = pose_to_body_twist(nominal_pose, self.config.dt)
        corrected_twist = nominal_twist + raw[..., :3] * self.velocity_scale
        trajectory = integrate_body_twist(corrected_twist, self.config.dt)
        increments = F.softplus(raw[..., 3:4]) * (self.work_scale / self.config.horizon)
        return {"trajectory": trajectory, "work": increments.cumsum(1), "event_logits": raw[..., 4:7]}

    def forward(self, batch: dict[str, torch.Tensor], control: str = "normal") -> dict[str, torch.Tensor]:
        context = self.encode_observation(batch["rgbd"], batch["history"], batch["global_features"], control)
        patches = (self.encode_candidate_patches(batch["rgbd"], batch["nominal_pose"], batch["global_features"], control)
                   if self.config.candidate_patches else None)
        return self.predict_from_context(context, batch["commands"], batch["nominal_pose"], patches)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def pose_to_body_twist(pose: torch.Tensor, dt: float) -> torch.Tensor:
    """Recover nominal body velocities using each interval's starting heading."""
    previous_xy = torch.cat((torch.zeros_like(pose[:, :1, :2]), pose[:, :-1, :2]), dim=1)
    yaw = torch.atan2(pose[..., 2], pose[..., 3])
    previous_yaw = torch.cat((torch.zeros_like(yaw[:, :1]), yaw[:, :-1]), dim=1)
    displacement = (pose[..., :2] - previous_xy) / dt
    c, s = torch.cos(previous_yaw), torch.sin(previous_yaw)
    vx = c * displacement[..., 0] + s * displacement[..., 1]
    vy = -s * displacement[..., 0] + c * displacement[..., 1]
    delta_yaw = yaw - previous_yaw
    yaw_rate = torch.atan2(torch.sin(delta_yaw), torch.cos(delta_yaw)) / dt
    return torch.stack((vx, vy, yaw_rate), dim=-1)


def integrate_body_twist(twist: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrate body twist into anchor-frame x/y/sin(yaw)/cos(yaw)."""
    yaw = (twist[..., 2] * dt).cumsum(1)
    previous_yaw = torch.cat((torch.zeros_like(yaw[:, :1]), yaw[:, :-1]), dim=1)
    c, s = torch.cos(previous_yaw), torch.sin(previous_yaw)
    dx = (c * twist[..., 0] - s * twist[..., 1]) * dt
    dy = (s * twist[..., 0] + c * twist[..., 1]) * dt
    return torch.stack((dx.cumsum(1), dy.cumsum(1), torch.sin(yaw), torch.cos(yaw)), dim=-1)


def rotate_world_batch(batch: dict[str, torch.Tensor], quarter_turns: torch.Tensor) -> dict[str, torch.Tensor]:
    """Rotate the north-up observation and measured world pose together.

    A quarter turn is +90 degrees about the fixed camera/world origin. Ego
    commands, goal-relative context, history, and targets stay unchanged. This
    is a camera-coordinate augmentation, not synthesis of new terrain physics.
    """
    if quarter_turns.shape != (len(batch["rgbd"]),):
        raise ValueError("One quarter-turn index is required per observation")
    result = dict(batch)
    pixels = batch["rgbd"].clone()
    globals_ = batch["global_features"].clone()
    original = batch["global_features"]
    for turns in (1, 2, 3):
        selected = torch.where(quarter_turns.remainder(4) == turns)[0]
        if not len(selected):
            continue
        pixels[selected] = torch.rot90(batch["rgbd"][selected], turns, dims=(-2, -1))
        x, y = original[selected, 4], original[selected, 5]
        sine, cosine = original[selected, 6], original[selected, 7]
        if turns == 1:
            globals_[selected, 4], globals_[selected, 5] = -y, x
            globals_[selected, 6], globals_[selected, 7] = cosine, -sine
        elif turns == 2:
            globals_[selected, 4], globals_[selected, 5] = -x, -y
            globals_[selected, 6], globals_[selected, 7] = -sine, -cosine
        else:
            globals_[selected, 4], globals_[selected, 5] = y, -x
            globals_[selected, 6], globals_[selected, 7] = -cosine, sine
    result["rgbd"], result["global_features"] = pixels, globals_
    return result


def rgbd_model_config_dict(model: RGBDFiniteHorizonFDM) -> dict[str, Any]:
    return asdict(model.config)


def load_rgbd_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[RGBDFiniteHorizonFDM, dict[str, Any]]:
    """Load a trusted local RGB-D checkpoint for inference or route scoring."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    if checkpoint.get("model_family") != "rgbd_reference_gru":
        raise ValueError("Expected an RGB-D forward-GRU checkpoint")
    model = RGBDFiniteHorizonFDM(RGBDFDMConfig(**checkpoint["model_config"]), checkpoint["normalization"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


__all__ = ["RGBD_ARMS", "RGBDFDMConfig", "RGBDFiniteHorizonFDM", "EVENT_NAMES", "fdm_loss",
           "load_rgbd_checkpoint", "rgbd_model_config_dict", "pose_to_body_twist", "integrate_body_twist", "rotate_world_batch", "PROGRESS_TARGET_VERSIONS"]
