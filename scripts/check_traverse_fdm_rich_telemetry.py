#!/usr/bin/env python3
"""Check recorded rich-telemetry timing, energy identities, and optional parity.

Use --self-test for a deterministic no-Chrono integration/missing-data contract
check. This is separate from actual Chrono observer-on/off trajectory parity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nedm.traverse.fdm_rich_telemetry import POWER_CHANNELS, RichTelemetry


def verify(directory, *, trajectory=None, require_solver_steps=False):
    directory = Path(directory)
    metadata = json.loads((directory / "rich_telemetry.json").read_text())
    with np.load(directory / "rich_telemetry.npz") as data:
        states = {key: data[key] for key in data.files}
    with np.load(directory / "rich_intervals.npz") as data:
        intervals = {key: data[key] for key in data.files}
    N = metadata["interval_rows"]
    assert all(array.shape == (N+1,) for array in states.values()), "Not all sample fields are N+1"
    assert all(array.shape == (N,) for array in intervals.values()), "Not all interval fields are N"
    np.testing.assert_array_equal(states["frame"], np.arange(N+1))
    np.testing.assert_array_equal(intervals["frame"], np.arange(N))
    np.testing.assert_array_equal(states["is_terminal"], np.r_[np.zeros(N), 1.])
    np.testing.assert_allclose(np.diff(states["chrono_time_s"]), intervals["duration_s"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(states["time_s"][:-1], intervals["start_time_s"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(states["time_s"][1:], intervals["end_time_s"], rtol=0, atol=1e-9)
    assert np.all(intervals["duration_s"] > 0)
    if require_solver_steps:
        assert np.all(intervals["solver_step_work"] == 1), "Coarse frame work found"
        assert np.all(intervals["physics_power_sample_count"] > 0), "Missing substep integration"
        assert np.all(intervals["post_step_risk_sample_count"] == intervals["physics_power_sample_count"]), "Incomplete post-step risk sampling"
    for channel in POWER_CHANNELS:
        signed = intervals[f"{channel}_signed_work_kj"]
        positive = intervals[f"{channel}_positive_work_kj"]
        negative = intervals[f"{channel}_negative_magnitude_work_kj"]
        np.testing.assert_allclose(signed, positive-negative, rtol=1e-10, atol=1e-9, equal_nan=True)
        assert np.all(positive[np.isfinite(positive)] >= 0)
        assert np.all(negative[np.isfinite(negative)] >= 0)
        np.testing.assert_allclose(intervals[f"{channel}_mean_power_kw"]*intervals["duration_s"], signed, rtol=1e-10, atol=1e-9, equal_nan=True)
    for key, field in metadata["fields"].items():
        assert field["finite_count"] == int(np.isfinite(states[key]).sum()), f"Wrong availability: {key}"
        assert field["missing_count"] == int((~np.isfinite(states[key])).sum()), f"Wrong missing count: {key}"
    parity = None
    if trajectory:
        with np.load(trajectory) as recorded:
            assert len(recorded["state"]) == N
            pose = np.vstack([recorded["pose"], recorded["terminal_pose"]])
            actual = np.column_stack([states["pos_world_x_m"], states["pos_world_y_m"], states["yaw_rad"]])
            np.testing.assert_allclose(actual, pose, rtol=0, atol=2e-6)
            action = np.column_stack([states[f"driver_{name}"][:-1] for name in ("steering", "throttle", "braking")])
            np.testing.assert_allclose(action, recorded["action"], rtol=0, atol=1e-7)
            np.testing.assert_allclose(states["engine_interface_power_kw"][:-1], recorded["power_kw"], rtol=1e-7, atol=1e-7)
            if np.all(intervals["solver_step_work"] == 1):
                np.testing.assert_allclose(intervals["engine_interface_positive_work_kj"], recorded["positive_work_kj_per_interval"], rtol=1e-7, atol=1e-7)
            parity = "Pose, applied actions, engine-interface power and solver-step work match trajectory.npz"
    return {"passed": True, "rows": N+1, "intervals": N, "sample_fields": len(states),
            "solver_step_intervals": int(intervals["solver_step_work"].sum()),
            "fields_with_missing_samples": [key for key, array in states.items() if not np.isfinite(array).all()],
            "trajectory_parity": parity}


def self_test():
    """Exercise variable-step quadrature, event timing and missing getter targets."""
    class FakeTelemetry(RichTelemetry):
        def _snapshot(self, scene, frame, action, terminal, command_context=None):
            ts = scene.system.GetChTime()
            if self._start_time is None:
                self._start_time = ts
            row = self._power(scene)
            row.update(self._fast_risk(scene))
            for key, value in {"frame": frame, "is_terminal": terminal, "chrono_time_s": ts, "time_s": ts-self._start_time,
                               **{f"{name}_{suffix}": 0. for name in ("tire_fl", "tire_fr", "tire_rl", "tire_rr")
                                  for suffix in ("longitudinal_slip", "force_world_vertical_n")}}.items():
                self._put(row, key, value, "test", "deterministic fake fixture")
            return row
    clock = SimpleNamespace(t=7.)
    engine = SimpleNamespace(GetMotorSpeed=lambda: 120., GetOutputMotorshaftTorque=lambda: 1000. if clock.t < 7.02 else -2000.)
    # Missing output driveshaft torque deliberately tests NaN propagation.
    transmission = SimpleNamespace(GetOutputMotorshaftSpeed=lambda: 100.)
    vehicle = SimpleNamespace(GetEngine=lambda: engine, GetTransmission=lambda: transmission,
                              GetDriveline=lambda: SimpleNamespace(GetOutputDriveshaftSpeed=lambda: 30.),
                              GetRoll=lambda: .1 if clock.t < 7.02 else .3, GetPitch=lambda: -.2)
    body = SimpleNamespace(GetContactForce=lambda: SimpleNamespace(x=5. if clock.t >= 7.02 else 0., y=0., z=0.))
    scene = SimpleNamespace(system=SimpleNamespace(GetChTime=lambda: clock.t),
                            hmmwv=SimpleNamespace(GetVehicle=lambda: vehicle, GetChassis=lambda: SimpleNamespace(GetBody=lambda: body)),
                            asset_bodies=[], config={"fixture": True})
    with tempfile.TemporaryDirectory(prefix="fdm_rich_telemetry_check_") as tmp:
        observer = FakeTelemetry(tmp)
        observer.on_frame(scene, -1, None, None, (0., 1., 0.))
        observer.on_frame(scene, 0, None, None, (0., 1., 0.))
        for sub, dt in enumerate((.02, .03)):
            observer.on_substep(scene, 0, sub, dt)
            clock.t += dt
            observer.on_post_substep(scene, 0, sub)
        observer.finish(scene, 1, None, None, (0., 1., 0.))
        result = verify(tmp, require_solver_steps=True)
        with np.load(Path(tmp)/"rich_intervals.npz") as intervals:
            np.testing.assert_allclose(intervals["engine_interface_positive_work_kj"], [2.])
            np.testing.assert_allclose(intervals["engine_interface_negative_magnitude_work_kj"], [6.])
            np.testing.assert_allclose(intervals["max_abs_roll_rad"], [.3])
            np.testing.assert_allclose(intervals["max_chassis_contact_resultant_n"], [5.])
            assert np.isnan(intervals["driveshaft_positive_work_kj"]).all()
        # Missing on_substep must be disclosed, never silently claim 500 Hz.
        clock.t = 7.
        coarse = FakeTelemetry(Path(tmp)/"coarse")
        coarse.on_frame(scene, 0, None, None, (0., 1., 0.))
        clock.t += .05
        coarse.finish(scene, 1, None, None, (0., 1., 0.))
        assert verify(Path(tmp)/"coarse")["solver_step_intervals"] == 0
        try:
            verify(Path(tmp)/"coarse", require_solver_steps=True)
        except AssertionError:
            pass
        else:
            raise AssertionError("Coarse quadrature was falsely accepted as solver-step integration")
        # A missing physics substep must fail coverage even if frame counts match.
        clock.t = 7.
        broken = FakeTelemetry(Path(tmp)/"broken")
        broken.on_frame(scene, 0, None, None, (0., 1., 0.))
        broken.on_substep(scene, 0, 0, .02)
        clock.t += .05
        try:
            broken.finish(scene, 1, None, None, (0., 1., 0.))
        except ValueError as exc:
            assert "does not cover" in str(exc)
        else:
            raise AssertionError("Incomplete physics-step coverage was accepted")
    result["checks"] = ["variable physics dt", "signed/positive/negative energy", "missing API stays NaN", "post-step extrema", "settling excluded", "coarse work labelled", "incomplete coverage rejected"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?")
    parser.add_argument("--trajectory")
    parser.add_argument("--require-solver-steps", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    elif args.directory:
        result = verify(args.directory, trajectory=args.trajectory, require_solver_steps=args.require_solver_steps)
    else:
        parser.error("Provide directory or --self-test")
    output = json.dumps(result, indent=2)+"\n"
    if args.out:
        Path(args.out).write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
