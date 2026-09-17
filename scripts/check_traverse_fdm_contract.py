#!/usr/bin/env python
"""CPU contract checks for the isolated FDM experiment; no training or simulation.

These checks use synthetic, independently calculable motion and interventions.
They verify information boundaries, physical-label timing, ablation meaning,
and batch inference rather than fitting a model to synthetic data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nedm.traverse.fdm_model import FDMConfig, FiniteHorizonFDM, fdm_loss


class FlatTerrain:
    def height(self, x, y):
        return np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)))

    def gradient(self, x, y):
        shape = np.broadcast_shapes(np.shape(x), np.shape(y))
        return np.zeros(shape), np.zeros(shape)


def data_checks() -> dict:
    from nedm.traverse import fdm_data as D
    from nedm.traverse.nrd_data import split_keys

    # Identity aliases preserve the original split, including its original
    # manifest ordering. Adding aliases with conflicting assignments must fail.
    ids = [f"full_v4__ep_{i:04d}_spline" for i in range(20)]
    expected = {k: s for s, group in zip(("train", "val", "test"), split_keys(ids)) for k in group}
    assert D.frozen_split_assignments({"episodes": ids}) == expected
    aliased = [k.replace("full_v4__", "full_v4_partial__") for k in ids]
    assert D.frozen_split_assignments({"episodes": aliased}) == expected
    assert D.novel_split(ids[0]) == D.novel_split(aliased[0])
    conflict = ids.copy()
    order = np.random.default_rng(20260902).permutation(len(ids))
    conflict[order[14]] = aliased[order[0]]
    try:
        D.frozen_split_assignments({"episodes": conflict})
    except ValueError:
        pass
    else:
        raise AssertionError("Canonical duplicate crosses frozen train/validation boundary")

    # Two m/s straight motion yields 0.4 m at the first 0.2 s target.
    n, anchor = 121, 20
    time = np.arange(n) * .05
    states = np.broadcast_to(np.arange(n)[:, None], (n, 17)).astype(np.float32).copy()
    states[:, 2:4] = 0.
    actions = np.broadcast_to(np.arange(n)[:, None], (n, 3)).astype(np.float32).copy()
    poses = np.column_stack((2.*time, np.zeros(n), np.zeros(n))).astype(np.float32)
    power = np.where(np.arange(n) % 2, -3., 10.).astype(np.float32)
    history = D.build_history(states, actions, poses, anchor)
    assert history.shape == (16, 24)
    np.testing.assert_array_equal(history[:, :17], states[5:21])
    np.testing.assert_array_equal(history[:, 17:20], actions[4:20])
    np.testing.assert_allclose(history[-1, 20:], [0., 0., 0., 1.])
    altered = [a.copy() for a in (states, actions, poses)]
    altered[0][anchor+1:] += 900.
    altered[1][anchor:] += 900.  # Current interval action is not previous-action history.
    altered[2][anchor+1:] += 900.
    np.testing.assert_array_equal(history, D.build_history(*altered, anchor))
    altered[0][anchor, 0] += 1.
    assert not np.array_equal(history, D.build_history(*altered, anchor)), "Current physical state was discarded"

    x = np.arange(101, dtype=float)
    route = {"waypoints": np.column_stack((x, np.zeros_like(x))), "stations": x,
             "headings": np.zeros_like(x), "speeds": np.full_like(x, 2.)}
    prefix = D.tracked_route_indices(route, poses)
    changed_poses = poses.copy()
    changed_poses[anchor+1:, 0] += 80.
    changed_indices = D.tracked_route_indices(route, changed_poses)
    np.testing.assert_array_equal(prefix[:anchor+1], changed_indices[:anchor+1])
    candidate = D.build_candidate_features(route, poses[anchor], FlatTerrain(), {"assets": []},
                                           station=2., elapsed_s=1.)
    assert candidate["candidate"].shape == (64, 13)
    assert candidate["nominal_pose"].shape == (20, 4)
    np.testing.assert_allclose(candidate["nominal_pose"][:, 0], np.arange(1, 21)*.4, atol=2e-6)
    np.testing.assert_allclose(candidate["nominal_pose"][:, 1:3], 0., atol=1e-7)
    np.testing.assert_allclose(candidate["nominal_pose"][:, 3], 1.)
    np.testing.assert_allclose(candidate["candidate"][:, list(D.TERRAIN_INDICES)], 0.)

    base_meta = {"status": "complete", "contact": {"events": []}}
    parked = np.zeros(n, bool)
    target = D.build_targets(poses, states, actions, power, base_meta, anchor, parked)
    np.testing.assert_allclose(target["trajectory"][:, 0], np.arange(1, 21)*.4, atol=2e-6)
    expected_work = [np.maximum(power[anchor:anchor+off], 0.).sum()*.05 for off in D.OUTPUT_OFFSETS]
    np.testing.assert_allclose(target["work"][:, 0], expected_work, atol=2e-6)
    assert target["trajectory_mask"].all()
    assert not target["events"][:, 2].any()

    # Event frame i occurs after state_i. Frame anchor+4 must NOT be
    # credited to a horizon ending at state_anchor+4; frame anchor+3 must.
    for event_frame, expected_first in ((anchor-1, 0.), (anchor+3, 1.), (anchor+4, 0.)):
        meta = {"status": "complete", "contact": {"events": [[event_frame, 0, 10.]]}}
        t = D.build_targets(poses, states, actions, power, meta, anchor, parked)
        assert t["events"][0, 0] == expected_first, (event_frame, t["events"][:, 0])
        if event_frame == anchor+4:
            assert t["events"][1, 0] == 1.
    boundary_states = states.copy()
    boundary_states[anchor+4:, 2] = np.deg2rad(61.)
    t = D.build_targets(poses, boundary_states, actions, power, base_meta, anchor, parked)
    assert t["events"][0, 1] == 1., "Endpoint rollover observation was excluded"

    # A terminal event can be known despite no recorded future endpoint pose.
    short_n = anchor+3
    terminal_meta = {"status": "rollover", "contact": {"events": [[short_n-1, 0, 12.]]}}
    short = D.build_targets(poses[:short_n], states[:short_n], actions[:short_n], power[:short_n],
                            terminal_meta, anchor, parked[:short_n])
    assert not short["trajectory_mask"].any()
    assert (short["events"][:, :2] == 1.).all() and (short["event_mask"][:, :2] == 1.).all()
    assert not short["event_mask"][:, 2].any()
    capped_meta = {"status": "complete", "contact": {"events": [[anchor-1, 0, 10.]]*2000}}
    capped = D.build_targets(poses, states, actions, power, capped_meta, anchor, parked)
    assert not capped["event_mask"][:, 0].any(), "Truncated event list created unknown negatives"
    capped_meta["contact"]["events"][-1] = [anchor+1, 0, 10.]
    capped = D.build_targets(poses, states, actions, power, capped_meta, anchor, parked)
    assert capped["events"][:, 0].all() and capped["event_mask"][:, 0].all()

    stuck_pose = np.zeros_like(poses)
    effort = np.zeros_like(actions)
    effort[:, 1] = .7
    stuck = D.build_targets(stuck_pose, states, effort, power, base_meta, anchor, parked)
    assert not stuck["event_mask"][:9, 2].any() and stuck["events"][9:, 2].all()
    deliberate_park = np.ones(n, bool)
    stopped = D.build_targets(stuck_pose, states, effort, power, base_meta, anchor, deliberate_park)
    assert not stopped["event_mask"][:, 2].any(), "Deliberate route parking was labeled failure"
    effort[:, 1] = 0.
    idle = D.build_targets(stuck_pose, states, effort, power, base_meta, anchor, parked)
    assert not idle["events"][:, 2].any(), "Zero-motion without effort was labeled stall"
    rollback = poses.copy()
    rollback[:, 0] *= -1.
    effort[:, 1] = .7
    rolled = D.build_targets(rollback, states, effort, power, base_meta, anchor, parked)
    assert not rolled["events"][:, 2].any()
    assert (rolled["trajectory"][:, 0] < 0.).all(), "Rollback displacement was rewritten to zero"
    return {"frozen_split_and_aliases": True, "causal_history_and_station": True,
            "nominal_path_uses_known_reference": True, "pose_and_work_interval_timing": True,
            "contact_and_rollover_endpoint_timing": True, "censored_event_positives": True,
            "parking_and_effort_labels": True, "rollback_preserved": True}


def pack_checks(path: Path) -> dict:
    """Read-only integrity and event-support report; no outcome-based selection."""
    manifest = json.loads((path / "manifest.json").read_text())
    identities, result = {}, {}
    for split in ("train", "val"):
        records = json.loads((path / f"{split}_episodes.json").read_text())
        identities[split] = {r["id"] for r in records}
        assert len(identities[split]) == len(records), "Duplicated episode identities"
        with np.load(path / f"{split}.npz") as d:
            assert all(np.isfinite(d[k]).all() for k in d.files)
            positive = (d["events"] > .5) & (d["event_mask"] > .5)
            positive_episodes = [len(np.unique(d["episode_index"][positive[..., j].any(axis=1)])) for j in range(3)]
            positive_windows = positive.any(axis=1).sum(axis=0)
            declared = manifest["splits"][split]["event_positive_windows"]
            assert positive_windows.tolist() == [declared[k] for k in ("contact", "rollover", "low_progress")]
            result[split] = {"episodes": len(records), "windows": len(d["history"]),
                             "positive_episodes": dict(zip(("contact", "rollover", "low_progress"), positive_episodes)),
                             "positive_windows": dict(zip(("contact", "rollover", "low_progress"), positive_windows.tolist()))}
    assert not identities["train"] & identities["val"], "Training/validation episode overlap"
    return result


def trainer_checks() -> dict:
    # Importing the training entry point does not invoke its CLI or optimizer.
    sys.path.insert(0, str(ROOT / "scripts"))
    from traverse_fdm_train import make_normalization, outcome_metrics, evaluate
    rng = np.random.default_rng(818)
    n, horizon = 2, 20
    nominal = np.zeros((n, horizon, 4), np.float32)
    nominal[..., 3] = 1.
    candidate = rng.normal(size=(n, 64, 13)).astype(np.float32)
    candidate[:, :48, 12] = 1.
    candidate[:, 48:] = 0.
    data = {"history": rng.normal(size=(n, 16, 24)).astype(np.float32),
            "candidate": candidate, "global_features": rng.normal(size=(n, 4)).astype(np.float32),
            "nominal_pose": nominal, "trajectory": nominal.copy(),
            "work": np.zeros((n, horizon, 1), np.float32),
            "events": np.zeros((n, horizon, 3), np.float32),
            "trajectory_mask": np.ones((n, horizon, 1), np.float32),
            "event_mask": np.ones((n, horizon, 3), np.float32),
            "episode_index": np.arange(n, dtype=np.int64)}
    data["event_mask"][:, :9, 2] = 0.
    train = {k: v.copy() for k, v in data.items()}
    train["events"][0, -1, (0, 2)] = 1.
    norm = make_normalization(train, max_pos_weight=20.)
    assert norm["candidate"]["mean"][12] == 0. and norm["candidate"]["std"][12] == 1.
    assert norm["supported_events"] == [True, False, True]
    model = FiniteHorizonFDM(FDMConfig(hidden_dim=8), norm).eval()
    batch = {k: torch.from_numpy(v) for k, v in data.items()}
    _, normalized_candidate, _ = model.normalize_inputs(batch)
    assert normalized_candidate[:, 48:].abs().sum() == 0., "Padding acquired artificial normalized feature values"
    assert (normalized_candidate[:, :48, 12] == 1.).all()

    # Structurally ineligible short low-progress horizons must not erase all
    # negative windows or contribute untrained high scores to window risk.
    logits = np.full((n, horizon, 3), -4., np.float32)
    logits[..., 1] = 8.  # Unsupported rollover output must not affect combined risk.
    logits[:, :9, 2] = 8.
    output = {"trajectory": nominal.copy(), "work": data["work"].copy(), "event_logits": logits}
    metrics = outcome_metrics(output, data, supported_events=np.array([True, False, True]), dt=.2)
    for block in (metrics["events"]["low_progress"]["window"], metrics["any_event_window"]):
        assert block["count"] == 2 and block["negative"] == 2
        assert block["brier"] < .001, "Ineligible or unsupported event logits affected risk"
    assert "mae_kj" in metrics["work"] and "mae_j" not in metrics["work"]

    prediction = model(batch)
    loss, _ = fdm_loss(model, prediction, batch)
    gradient = torch.autograd.grad(loss, prediction["event_logits"])[0]
    assert gradient[..., 1].abs().sum() == 0., "Unsupported rollover head received confidence-producing supervision"
    assert gradient[..., 0].abs().sum() > 0.

    # Global masked validation loss must not depend on where batch boundaries
    # happen to divide short/censored episodes.
    data["trajectory_mask"][1, 5:] = 0.
    data["event_mask"][1, 5:] = 0.
    weights = {"xy": 1., "yaw": .5, "work": .2, "events": 1.}
    together, _ = evaluate(model, data, 2, torch.device("cpu"), weights)
    separated, _ = evaluate(model, data, 1, torch.device("cpu"), weights)
    for key in together["loss"]:
        np.testing.assert_allclose(together["loss"][key], separated["loss"][key], rtol=2e-5, atol=2e-6)
    return {"structural_event_masks_keep_negatives": True, "unsupported_event_excluded": True,
            "binary_validity_normalization": True, "padding_zero_after_normalization": True,
            "work_units_kj": True, "masked_evaluation_batch_parity": True}


def scorer_checks() -> dict:
    from nedm.traverse.fdm_mppi import MPPIConfig, deform_reference, validate_reference
    from nedm.traverse.fdm_scoring import FDMReferenceScorer

    class KnownPrediction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))
            self.config = SimpleNamespace(dt=.2)
            self.seen = []

        def forward(self, batch):
            self.seen.append({k: v.detach().clone() for k, v in batch.items()})
            count = len(batch["history"])
            pose = torch.zeros(count, 20, 4)
            pose[..., 0], pose[..., 3] = 2., 1.
            probability = torch.full((count, 20, 3), .1)
            probability[:, 2, 0] = .8  # Early contact risk exceeds final contact risk.
            probability[..., 1] = .999  # Unsupported rollover must not enter pilot cost.
            probability[:, :9, 2] = .99  # Low progress is unsupervised before 2 s.
            probability[:, 9:, 2] = .2
            return {"trajectory": pose, "work": torch.full((count, 20, 1), 5.),
                    "event_logits": torch.logit(probability)}

    x = np.arange(36, dtype=float)
    route = {"waypoints": np.column_stack((x, np.ones_like(x))), "speeds": np.full_like(x, 2.),
             "stations": x, "headings": np.zeros_like(x), "meta": {"fdm_station": 10.}}
    # Vehicle has rolled behind the monotone driver's station. This station
    # must not silently be replaced by the globally nearest point at x=2.
    anchor = np.array([2., 1., np.pi/2])
    model = KnownPrediction()
    history = np.arange(16*24, dtype=np.float32).reshape(16, 24)
    scorer = FDMReferenceScorer(model, history, anchor, FlatTerrain(), {"assets": []}, [2., 11.],
                                work_cost_per_kj=.3, batch_size=1)
    cost = scorer([route, route, route])
    # Two metres goal progress, 0.8 contact, 0.2 eligible low-progress, 5 kJ work.
    np.testing.assert_allclose(cost, -2.+20.*.8+10.*.2+.3*5., rtol=1e-6)
    np.testing.assert_allclose(model.seen[0]["candidate"][0, 0, :2].numpy(), [0., -8.], atol=1e-6)
    np.testing.assert_array_equal(model.seen[0]["history"][0].numpy(), history)
    batched = FDMReferenceScorer(model, history, anchor, FlatTerrain(), {"assets": []}, [2., 11.],
                                 work_cost_per_kj=.3, batch_size=3)
    np.testing.assert_allclose(cost, batched([route, route, route]))
    cfg = MPPIConfig()
    modified = deform_reference(route, np.zeros(6), anchor, cfg)
    assert abs(modified["meta"]["fdm_station"]-10.) < 1e-6
    # An obstacle well behind the current commanded station is not a new
    # future-route collision; an obstacle ahead still invalidates the route.
    assert validate_reference(modified, [(0., 1., .2)], cfg)["valid"]
    assert not validate_reference(modified, [(20., 1., .2)], cfg)["valid"]
    return {"causal_station_retained": True, "early_contact_risk_retained": True,
            "ineligible_low_progress_and_rollover_excluded": True,
            "ego_to_world_goal_cost": True, "scorer_batch_parity": True,
            "future_geometry_check": True}


def model_checks() -> dict:
    torch.set_num_threads(2)
    torch.manual_seed(117)
    batch_size, horizon = 3, 20
    nominal = torch.zeros(batch_size, horizon, 4)
    nominal[..., 0] = torch.arange(1, horizon + 1) * .4
    nominal[..., 3] = 1.
    batch = {
        "history": torch.randn(batch_size, 16, 24),
        "candidate": torch.randn(batch_size, 64, 13),
        "global_features": torch.randn(batch_size, 4),
        "nominal_pose": nominal,
    }
    summaries = {}
    for arm in ("profile", "no_history", "history", "no_terrain"):
        model = FiniteHorizonFDM(FDMConfig(arm=arm, hidden_dim=16)).eval()
        output = model(batch)
        assert output["trajectory"].shape == (batch_size, horizon, 4)
        assert output["work"].shape == (batch_size, horizon, 1)
        assert output["event_logits"].shape == (batch_size, horizon, 3)
        for value in output.values():
            assert torch.isfinite(value).all(), arm
        assert (output["work"] >= 0.).all()
        assert (output["work"].diff(dim=1) >= 0.).all()
        torch.testing.assert_close(output["trajectory"][..., 2:].norm(dim=-1),
                                   torch.ones(batch_size, horizon))

        # Candidate batching and ordering cannot change an individual judgment.
        single = [model({k: v[i:i + 1] for k, v in batch.items()}) for i in range(batch_size)]
        permutation = torch.tensor([2, 0, 1])
        shuffled = model({k: v[permutation] for k, v in batch.items()})
        for key in output:
            torch.testing.assert_close(output[key], torch.cat([o[key] for o in single]), atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(output[key][permutation], shuffled[key], atol=2e-6, rtol=2e-5)

        changed = {k: v.clone() for k, v in batch.items()}
        if arm in ("no_history", "profile"):
            changed["history"][:, :-1] += 100.
            altered = model(changed)
            for key in output:
                torch.testing.assert_close(output[key], altered[key], atol=0., rtol=0.)
        if arm == "no_terrain":
            changed["candidate"][..., list(model.config.terrain_indices)] += 100.
            altered = model(changed)
            for key in output:
                torch.testing.assert_close(output[key], altered[key], atol=0., rtol=0.)

        # Inspect gradients to prove retained inputs are still connected, not
        # merely that excluded-input perturbations happen to change little.
        grad_batch = {k: v.detach().clone().requires_grad_(k != "nominal_pose") for k, v in batch.items()}
        grad_output = model(grad_batch)
        sum(v.square().sum() for v in grad_output.values()).backward()
        history_grad = grad_batch["history"].grad
        candidate_grad = grad_batch["candidate"].grad
        assert history_grad is not None and history_grad[:, -1].abs().sum() > 0, arm
        assert candidate_grad is not None and candidate_grad[..., 11].abs().sum() > 0, arm
        if arm in ("no_history", "profile"):
            assert history_grad[:, :-1].abs().sum() == 0, arm
        if arm == "no_terrain":
            assert candidate_grad[..., list(model.config.terrain_indices)].abs().sum() == 0

        # Censored pose targets must not remove a known positive event.
        loss_batch = {
            **batch,
            "trajectory": nominal.clone(),
            "work": torch.zeros(batch_size, horizon, 1),
            "events": torch.zeros(batch_size, horizon, 3),
            "trajectory_mask": torch.ones(batch_size, horizon, 1),
            "work_mask": torch.ones(batch_size, horizon, 1),
            "event_mask": torch.ones(batch_size, horizon, 3),
        }
        loss_batch["trajectory_mask"][:, 3:] = 0.
        loss_batch["work_mask"][:, 3:] = 0.
        loss_batch["events"][:, 3:, 0] = 1.
        loss, components = fdm_loss(model, output, loss_batch)
        assert torch.isfinite(loss) and all(torch.isfinite(v) for v in components.values())
        altered_targets = {k: v.clone() for k, v in loss_batch.items()}
        altered_targets["trajectory"][:, 3:] += 10000.
        altered_targets["work"][:, 3:] += 10000.
        masked_loss, _ = fdm_loss(model, output, altered_targets)
        torch.testing.assert_close(loss, masked_loss)
        event_gradient = torch.autograd.grad(loss, output["event_logits"], retain_graph=True)[0]
        assert event_gradient[:, 3:, 0].abs().sum() > 0., "Censored motion erased observed event supervision"
        model.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        summaries[arm] = {"parameters": model.parameter_count(), "batch_parity": True,
                          "retained_input_gradients": True, "finite_masked_backward": True}
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--pack", type=Path, help="Optional prepared pack for independent label-support counts")
    parser.add_argument("--out", type=Path, help="Optional JSON result path inside the isolated experiment")
    args = parser.parse_args()
    result = {"model": model_checks(), "training_performed": False}
    if not args.model_only:
        result["data"] = data_checks()
        result["trainer"] = trainer_checks()
        result["scorer"] = scorer_checks()
    if args.pack:
        result["pack"] = pack_checks(args.pack)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
