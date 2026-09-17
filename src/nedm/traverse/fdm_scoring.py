"""Batched learned costs for one measured context and candidate PID references.

The first pilot uses privileged BMP/asset geometry. These costs are predictions,
not evidence of closed-loop feasibility. Unvalidated rollover forecasts are
excluded from the pilot cost explicitly.
"""
from __future__ import annotations

import numpy as np
import torch

from nedm.traverse.fdm_data import build_candidate_features


class FDMReferenceScorer:
    def __init__(self, model, history, anchor_pose, terrain, layout, goal_xy, *,
                 elapsed_s=0., contact_cost=20., low_progress_cost=10., work_cost_per_kj=0.,
                 batch_size=128):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.history = np.asarray(history, np.float32)
        self.anchor_pose = np.asarray(anchor_pose, np.float64)
        self.terrain, self.layout = terrain, layout
        self.goal_xy = np.asarray(goal_xy, np.float64)
        self.elapsed_s = float(elapsed_s)
        self.contact_cost, self.low_progress_cost = float(contact_cost), float(low_progress_cost)
        self.work_cost_per_kj = float(work_cost_per_kj)
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    @torch.no_grad()
    def predict(self, routes):
        rows = [build_candidate_features(r, self.anchor_pose, self.terrain, self.layout,
                                         station=r.get("meta", {}).get("fdm_station"),
                                         elapsed_s=self.elapsed_s) for r in routes]
        if not rows:
            return {}
        accumulated = {k: [] for k in ("trajectory", "work", "event_probability")}
        for start in range(0, len(rows), self.batch_size):
            subset = rows[start:start+self.batch_size]
            batch = {k: torch.from_numpy(np.stack([r[k] for r in subset])).to(self.device)
                     for k in ("candidate", "global_features", "nominal_pose")}
            batch["history"] = torch.from_numpy(np.repeat(self.history[None], len(subset), axis=0)).to(self.device)
            output = self.model(batch)
            for key in ("trajectory", "work"):
                accumulated[key].append(output[key].cpu().numpy())
            accumulated["event_probability"].append(torch.sigmoid(output["event_logits"]).cpu().numpy())
        result = {k: np.concatenate(v) for k, v in accumulated.items()}
        if any(not np.isfinite(v).all() for v in result.values()):
            raise ValueError("Nonfinite learned prediction")
        return result

    def __call__(self, routes):
        if not routes:
            return np.empty(0)
        output = self.predict(routes)
        xy = output["trajectory"][:, -1, :2]
        c, s = np.cos(self.anchor_pose[2]), np.sin(self.anchor_pose[2])
        world_xy = self.anchor_pose[:2] + np.stack((c*xy[:, 0]-s*xy[:, 1], s*xy[:, 0]+c*xy[:, 1]), axis=1)
        # Goal remains analytic and can change independently of the predictor.
        distance_change = np.linalg.norm(world_xy-self.goal_xy, axis=1) - np.linalg.norm(self.anchor_pose[:2]-self.goal_xy)
        probability = output["event_probability"]
        # Contact is cumulative, but independently decoded horizon probabilities
        # are not forced monotone. Penalize any high-risk prefix. Low progress
        # has no supervision before 2 s, so those logits must never enter costs.
        eligible = (np.arange(probability.shape[1])+1)*self.model.config.dt >= 2. - 1e-6
        contact_risk = probability[..., 0].max(1)
        progress_risk = probability[:, eligible, 2].max(1) if eligible.any() else np.zeros(len(routes))
        return (distance_change + self.contact_cost*contact_risk
                + self.low_progress_cost*progress_risk
                + self.work_cost_per_kj*output["work"][:, -1, 0])
