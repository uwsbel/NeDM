#!/usr/bin/env python3
"""Independent CPU inference-only checks of causal history and online costs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nedm.traverse.fdm_data import build_history
from nedm.traverse.fdm_diverse_data import build_command_features
from nedm.traverse.fdm_diverse_planner import RGBDReferenceScorer, RGBDCostConfig, propose_route_families, check_reference_contract
from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference


def scorer(anchor=(0., 0., 0.), goal=(100., 0.), energy=.02):
    value = RGBDReferenceScorer.__new__(RGBDReferenceScorer)
    value.anchor_pose = np.asarray(anchor, float)
    value.goal_xy = np.asarray(goal, float)
    value.model = SimpleNamespace(config=SimpleNamespace(dt=.2), supported_events=torch.ones(3, dtype=torch.bool))
    value.cost_config = RGBDCostConfig(energy_weight_s_per_kj=energy)
    return value


def forecast(horizon=20, distance=8., work=20.):
    return {'trajectory': np.stack((np.linspace(distance/horizon, distance, horizon), np.zeros(horizon)), -1)[None],
            'work': np.linspace(work/horizon, work, horizon)[None, :, None],
            'event_probability': np.zeros((1, horizon, 3)), 'attitude': np.zeros((1, horizon, 2))}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--out'); args = parser.parse_args()
    rng = np.random.default_rng(408)
    states = rng.normal(size=(50, 17)).astype(np.float32)
    actions = rng.normal(size=(50, 3)).astype(np.float32)
    poses = rng.normal(size=(50, 3)); anchor = 24
    expected = build_history(states, actions, poses, anchor)
    future_state, future_action, future_pose = states.copy(), actions.copy(), poses.copy()
    future_state[anchor+1:] = 1e6; future_action[anchor:] = 1e6; future_pose[anchor+1:] = 1e6
    assert np.array_equal(expected, build_history(future_state, future_action, future_pose, anchor))
    previous_action = actions.copy(); previous_action[anchor-1] += 1.
    assert not np.array_equal(expected, build_history(states, previous_action, poses, anchor))
    startup = build_history(states[:1], np.ones((1, 3)), poses[:1], 0)
    assert np.array_equal(startup[:, 17:20], np.tile([0., 0., 1.], (16, 1)))
    assert np.array_equal(expected, build_history(states[:anchor+1], actions[:anchor+1], poses[:anchor+1], anchor))
    history_report = {'future_state_pose_and_current_future_actions_ignored': True,
                      'previous_applied_action_changes_history': True, 'startup_synthetic_brake_exact': True,
                      'online_prefix_equals_full_recording_history': True}

    value = scorer(); output = forecast()
    base = value.cost_breakdown(output)
    slower = value.cost_breakdown(forecast(distance=4.))
    assert base['cost'][0] < slower['cost'][0]
    high_work = value.cost_breakdown(forecast(work=40.))
    assert high_work['cost'][0] > base['cost'][0]
    no_energy = scorer(energy=0.)
    assert no_energy.cost_breakdown(forecast(work=40.))['cost'][0] == no_energy.cost_breakdown(forecast(work=20.))['cost'][0]
    rotated = scorer(anchor=(3., -4., np.pi/2.), goal=(3., 96.))
    assert np.allclose(rotated.cost_breakdown(output)['cost'], base['cost'])
    rejected = {}
    for channel, name in enumerate(('contact', 'rollover', 'stall')):
        hazardous = forecast(); hazardous['event_probability'][0, -1, channel] = .9
        rejected[name] = bool(np.isinf(value.cost_breakdown(hazardous)['cost'][0]))
        assert rejected[name]
    tilted = forecast(); tilted['attitude'][0, 3, 0] = np.deg2rad(56.)
    assert np.isinf(value.cost_breakdown(tilted)['cost'][0])
    unsupported = scorer(); unsupported.model.supported_events[0] = False
    result = unsupported.cost_breakdown(output)
    assert np.isinf(result['cost'][0]) and np.isnan(result['contact_probability'][0])
    unsupported_rollover = scorer(); unsupported_rollover.model.supported_events[1] = False
    result = unsupported_rollover.cost_breakdown(output)
    assert np.isfinite(result['cost'][0]) and np.isnan(result['rollover_probability'][0])
    arrival = scorer(goal=(4., 0.)); early = forecast(distance=8.)
    # First arrival is at 0.6 s, when predicted x=1.2 enters the 3 m radius.
    early['event_probability'][0, 10:, :] = .99
    got = arrival.cost_breakdown(early)
    assert np.isfinite(got['cost'][0]) and np.isclose(got['estimated_time_to_goal_s'][0], .6)
    assert got['contact_probability'][0] == 0. and got['low_progress_probability'][0] == 0.
    early['event_probability'][0, 2, 0] = .9
    assert np.isinf(arrival.cost_breakdown(early)['cost'][0])
    before_duration = forecast(); before_duration['event_probability'][0, :9, 2] = .9
    assert value.cost_breakdown(before_duration)['low_progress_probability'][0] == 0.
    cost_report = {'progress_improves_cost': True, 'energy_only_affects_enabled_arm': True,
                   'world_ego_rotation_invariant': True, 'risk_gates': rejected,
                   'hard_attitude_gate': True, 'required_head_support_enforced': True,
                   'unsupported_rollover_reported_unavailable': True, 'post_arrival_events_ignored': True,
                   'contact_at_arrival_is_counted': True, 'stall_requires_two_second_future': True}

    pose = np.array([-100., -40., .1]); goal = np.array([100., 40.])
    cfg = MPPIConfig(arena_half_extent_m=120., max_speed_mps=6., path_step_m=.5)
    routes = propose_route_families(pose, goal)
    for route in routes:
        check_reference_contract(route)
        assert np.array_equal(route['waypoints'][0], pose[:2]) and np.allclose(route['waypoints'][-1], goal)
    valid = [route for route in routes if validate_reference(route, [], cfg, pose)['valid']]
    assert len(valid) >= 3
    commands = build_command_features(valid[0], pose, station=0., elapsed_s=1.2, horizon=20)
    assert commands['commands'].shape == (20, 5) and np.isclose(commands['global_features'][3], 1.2)
    report = {'passed': True, 'scope': 'CPU no-optimizer contract tests; no simulated physics, trained accuracy or closed-loop success claim',
              'history': history_report, 'costs': cost_report, 'references': {'valid_family_count': len(valid), 'total_families': len(routes)},
              'sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                         ('src/nedm/traverse/fdm_data.py', 'src/nedm/traverse/fdm_diverse_data.py', 'src/nedm/traverse/fdm_diverse_planner.py', 'src/nedm/traverse/fdm_mppi.py')}}
    if args.out:
        target = Path(args.out); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
