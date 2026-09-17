#!/usr/bin/env python3
"""Exercise the real online decision block with CPU planner/driver stand-ins.

This tests scheduling, call inputs, route retention and abstention persistence.
It does not claim physical parity or test the neural model or native PID.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np

import traverse_fdm_rgbd_diverse_online as online


def decision_code():
    tree = ast.parse(Path(online.__file__).read_text())
    blocks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
              and isinstance(node.test, ast.Call)
              and isinstance(node.test.func, ast.Name)
              and node.test.func.id == "planning_due"]
    assert len(blocks) == 1, "Need the single production decision block"
    return compile(ast.fix_missing_locations(ast.Module(body=blocks, type_ignores=[])),
                   online.__file__, "exec")


def exercise(mode, abstain, code, frames=3600):
    selected = online.geometric_line([-100., 0.], [100., 0.], speed=4.)
    selected["waypoints"][100][1] = 1.25  # Exact retention, not a reconstructed line.
    before = copy.deepcopy(selected)
    initial = online.geometric_line([-100., 0.], [100., 0.], speed=2.)
    pose = np.array([-100., 0., 0.])
    state, action = np.zeros(17, np.float32), np.zeros(3, np.float32)
    calls, updates, records, desired = [], [], [], []

    def planner(model, image, history, actual_pose, goal, **kwargs):
        calls.append({"pose": actual_pose.copy(), "goal": goal.copy(),
                      "history": history.copy(), "image": image.copy(), **kwargs})
        return {"route": None if abstain else selected, "abstained": abstain,
                "model_evaluations": 7, "selected_family_index": None if abstain else 2}

    def update(route, actual_pose):
        updates.append((copy.deepcopy(route), actual_pose.copy()))
        return {"geometry_changed": True}

    def families(actual_pose, goal, active, originals, **kwargs):
        return [active]

    driver = SimpleNamespace(update_route=update, SetDesiredSpeed=desired.append)
    scope = {"np": np, "time": time, "hashlib": hashlib, "DT": online.DT,
        "planning_due": online.planning_due, "replan_stride": 20,
        "args": SimpleNamespace(planning_mode=mode, speeds=[2., 4., 6.],
                                offsets=[0., -22., 22., -44., 44.], seed=11,
                                image_intervention="normal"),
        "at_end": False, "paused": False, "record_state": [], "record_pose": [],
        "record_action": [], "state": state, "pose": pose, "action": action,
        "build_history": lambda states, actions, poses, anchor: np.r_[states[-1], actions[-1]],
        "online_route_families": families, "plan_rgbd_routes": planner,
        "model": object(), "map_observation": {"rgbd": np.arange(16).reshape(4, 2, 2)},
        "goal": np.array([100., 0.]), "route": initial, "original_families": [initial],
        "mppi": object(), "costs": object(), "planning_rows": [], "driver": driver,
        "xy": np.asarray(initial["waypoints"]), "speed": np.asarray(initial["speeds"]),
        "wp": 0, "pos": pose[:2], "case": {"goal_radius_m": 3.}, "dt": .002,
        "out": Path("unused"), "dump": lambda path, value: records.append(copy.deepcopy(value))}
    for frame in range(frames):
        scope["frame"] = frame
        exec(code, scope)
        # The production frame loop applies this persistent pause to PID.
        assert scope["paused"] == abstain
        if mode == "plan_once" and frame > 0:
            assert len(calls) == 1 and len(updates) == (0 if abstain else 1)
    expected = [0] if mode == "plan_once" else list(range(0, frames, 20))
    assert [row["frame"] for row in scope["planning_rows"]] == expected
    assert [call["seed"] for call in calls] == list(range(11, 11 + len(expected)))
    assert [call["elapsed_s"] for call in calls] == [frame * online.DT for frame in expected]
    assert len(records) == len(expected) and len(desired) == len(expected)
    assert all(value == (0. if abstain else 4.) for value in desired)
    assert scope["route"] is (initial if abstain else selected)
    assert selected == before, "Execution may not alter the selected full reference"
    assert len(updates) == (0 if abstain else len(expected))
    return {"mode": mode, "initial_abstention": abstain, "frames": frames,
            "planner_calls": len(calls), "reference_updates": len(updates),
            "later_planner_calls": len(calls)-1,
            "final_paused": scope["paused"]}, calls[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Exhaustive parity with the old condition on every valid frame, including
    # blocked scheduled frames; plan-once never retries after an initial pause.
    comparisons = 0
    for stride in (1, 3, 20, 80):
        for frame in range(3600):
            for at_end in (False, True):
                old = frame % stride == 0 and not at_end
                assert online.planning_due(frame, stride, at_end) == old
                assert online.planning_due(frame, stride, at_end, "receding") == old
                assert online.planning_due(frame, stride, at_end, "plan_once") == (frame == 0 and not at_end)
                comparisons += 1
    for mode in ("receding", "plan_once"):
        assert not any(online.planning_due(frame, 20, False, mode) for frame in range(-16, 0))
    try:
        online.planning_due(0, 20, False, "misspelled")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown planning mode must fail")
    code = decision_code()
    results, first_calls = [], {}
    for abstain in (False, True):
        for mode in ("receding", "plan_once"):
            result, first = exercise(mode, abstain, code)
            results.append(result)
            first_calls[mode, abstain] = first
        a, b = first_calls["receding", abstain], first_calls["plan_once", abstain]
        for key in ("pose", "goal", "history", "image"):
            np.testing.assert_array_equal(a[key], b[key])
        for key in ("elapsed_s", "seed", "families", "control"):
            assert a[key] == b[key]
    # CLI defaults and batch forwarding are also production source contracts.
    tree = ast.parse(Path(online.__file__).read_text())
    mode_args = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and node.args and isinstance(node.args[0], ast.Constant)
                 and node.args[0].value == "--planning-mode"]
    assert len(mode_args) == 1
    keywords = {kw.arg: ast.literal_eval(kw.value) for kw in mode_args[0].keywords}
    assert keywords == {"choices": ("receding", "plan_once"), "default": "receding"}
    batch = Path(online.__file__).with_name("traverse_fdm_rgbd_diverse_online_batch.py")
    assert any(isinstance(node, ast.Set) and any(isinstance(item, ast.Constant)
               and item.value == "planning-mode" for item in node.elts)
               for node in ast.walk(ast.parse(batch.read_text())))
    report = {"passed": True, "scope": "Actual production decision block with CPU stand-ins; no Chrono physical parity claimed",
        "schedule_comparisons": comparisons, "receding_schedule_matches_previous": True,
        "same_frame_zero_inputs_seed_and_family": True, "cases": results,
        "source_sha256": {str(path.relative_to(online.ROOT)): online.sha256(path)
                          for path in (Path(online.__file__), batch, Path(__file__))}}
    online.dump(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
