"""Passive dynamics telemetry for FDM collection; no rendering or physics writes.

``on_frame`` is compatible with the existing 20 Hz observer. New collectors can
also call ``on_substep`` after Synchronize/before Advance and
``on_post_substep`` after Advance. Only the latter pair supports solver-step
work integration and post-step contact/attitude extrema. Missing measurements
stay NaN with explicit capability information, never fabricated zero targets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


POWER_CHANNELS = ("engine_interface", "engine_rotor", "driveshaft")
WHEEL_NAMES = ("tire_fl", "tire_fr", "tire_rl", "tire_rr")


def _xyz(value):
    return np.array([value.x, value.y, value.z], dtype=np.float64)


def _finite_extreme(values, *, maximum=True):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float((np.max if maximum else np.min)(values)) if len(values) else np.nan


class RichTelemetry:
    """Save N+1 measured states and N aligned interval targets.

    ``scene`` must expose hmmwv, system, terrain, and optionally config and
    asset_bodies. Settling frames (<0) are ignored. Input arrays state17/pose3
    are accepted for observer compatibility; physical getters supply the rich
    values. The simulation timestamp, not frame*nominal_dt, defines integration.
    """

    def __init__(self, out_dir, case_path=None, *, record_dt_s=.05):
        self.out_dir = Path(out_dir)
        self.case_path = Path(case_path).resolve() if case_path else None
        self.record_dt_s = float(record_dt_s)
        self.rows: list[dict[str, float]] = []
        self.fields: dict[str, dict[str, str]] = {}
        self.capabilities: dict[str, dict[str, Any]] = {}
        self._unsupported_getters: set[str] = set()
        self.intervals: dict[int, dict[str, Any]] = {}
        self._wheel_specs = None
        self._suspensions = {}
        self._start_time = None
        self._finished = False

    def _read(self, key, getter, fallback=np.nan):
        info = self.capabilities.setdefault(key, {"attempts": 0, "successes": 0, "failures": 0})
        info["attempts"] += 1
        if key in self._unsupported_getters:
            info["failures"] += 1
            info["skipped_after_unsupported"] = info.get("skipped_after_unsupported", 0)+1
            return fallback
        info["getter_invocations"] = info.get("getter_invocations", 0)+1
        try:
            value = getter()
            info["successes"] += 1
            return value
        except (AttributeError, TypeError, RuntimeError, ValueError, IndexError) as exc:
            info["failures"] += 1
            info.setdefault("first_error", f"{type(exc).__name__}: {exc}")
            if isinstance(exc, (AttributeError, TypeError)):
                self._unsupported_getters.add(key)
            return fallback

    def _put(self, row, key, value, unit, source):
        row[key] = float(value)
        self.fields.setdefault(key, {"unit": unit, "source": source})

    def _scalar(self, row, key, getter, unit, source):
        self._put(row, key, self._read(source, lambda: float(getter())), unit, source)

    def _vector(self, row, prefix, getter, unit, source, suffix):
        vector = self._read(source, lambda: _xyz(getter()), np.full(3, np.nan))
        for axis, value in zip("xyz", vector):
            self._put(row, f"{prefix}_{axis}_{suffix}", value, unit, source)
        return vector

    def _power(self, scene):
        row = {}
        vehicle = scene.hmmwv.GetVehicle()
        getters = (
            ("engine_motor_speed_radps", lambda: vehicle.GetEngine().GetMotorSpeed(), "rad/s", "engine.GetMotorSpeed"),
            ("engine_motorshaft_torque_nm", lambda: vehicle.GetEngine().GetOutputMotorshaftTorque(), "N m", "engine.GetOutputMotorshaftTorque"),
            ("transmission_motorshaft_speed_radps", lambda: vehicle.GetTransmission().GetOutputMotorshaftSpeed(), "rad/s", "transmission.GetOutputMotorshaftSpeed"),
            ("transmission_driveshaft_torque_nm", lambda: vehicle.GetTransmission().GetOutputDriveshaftTorque(), "N m", "transmission.GetOutputDriveshaftTorque"),
            ("driveline_driveshaft_speed_radps", lambda: vehicle.GetDriveline().GetOutputDriveshaftSpeed(), "rad/s", "driveline.GetOutputDriveshaftSpeed"),
        )
        for key, getter, unit, source in getters:
            self._scalar(row, key, getter, unit, source)
        products = {
            "engine_interface": ("engine_motorshaft_torque_nm", "transmission_motorshaft_speed_radps"),
            "engine_rotor": ("engine_motorshaft_torque_nm", "engine_motor_speed_radps"),
            "driveshaft": ("transmission_driveshaft_torque_nm", "driveline_driveshaft_speed_radps"),
        }
        for channel, (torque, speed) in products.items():
            self._put(row, f"{channel}_power_kw", row[torque]*row[speed]/1000., "kW", f"{torque} * {speed} / 1000")
        return row

    def _fast_risk(self, scene):
        vehicle = scene.hmmwv.GetVehicle()
        body = scene.hmmwv.GetChassis().GetBody()
        row = {}
        for axis in ("roll", "pitch"):
            self._scalar(row, f"{axis}_rad", lambda axis=axis: getattr(vehicle, f"Get{axis.title()}")(), "rad", f"vehicle.Get{axis.title()}")
        contact = self._read("chassis.GetContactForce", lambda: _xyz(body.GetContactForce()), np.full(3, np.nan))
        self._put(row, "chassis_contact_resultant_n", np.linalg.norm(contact), "N", "norm(chassis.GetContactForce world vector)")
        # Existing assets are fixed bodies. A zero force is meaningful only if
        # every declared body's getter succeeded. Empty explicit list means no assets.
        assets = getattr(scene, "asset_bodies", None)
        asset_values = []
        if assets is not None:
            for index, asset in enumerate(assets):
                asset_body = asset[1] if isinstance(asset, (tuple, list)) else asset
                asset_values.append(self._read(f"asset[{index}].GetContactForce", lambda b=asset_body: np.linalg.norm(_xyz(b.GetContactForce()))))
        asset_max = max(asset_values, default=0.) if assets is not None and np.isfinite(asset_values).all() else np.nan
        self._put(row, "asset_contact_max_resultant_n", asset_max, "N", "max norm(GetContactForce) over declared asset bodies; no contact-pair identity")
        return row

    def _snapshot(self, scene, frame, action, terminal, command_context=None):
        vehicle, body = scene.hmmwv.GetVehicle(), scene.hmmwv.GetChassis().GetBody()
        ref = body.GetFrameRefToAbs()
        row = self._power(scene)
        row.update(self._fast_risk(scene))
        ts = float(scene.system.GetChTime())
        if self._start_time is None:
            self._start_time = ts
        for key, value, unit, source in (
            ("frame", frame, "index", "integer telemetry anchor"),
            ("is_terminal", terminal, "bool", "actual endpoint after last interval"),
            ("chrono_time_s", ts, "s", "system.GetChTime"),
            ("time_s", ts-self._start_time, "s", "chrono_time_s minus first recorded chrono_time_s"),
        ):
            self._put(row, key, value, unit, source)
        for name, value in zip(("steering", "throttle", "braking"), action):
            self._put(row, f"driver_{name}", value, "1", "applied driver input at pre-interval sample; terminal repeats last applied input")
        for key, value in (command_context or {}).items():
            if not isinstance(value, (bool, int, float, np.number)):
                raise TypeError(f"command_context.{key} must be a numeric scalar")
            self._put(row, f"command_{key}", value, "m/s" if key.endswith("mps") else "1", "collector supplied command context, not a measured response")
        self._vector(row, "pos_world", ref.GetPos, "m", "chassis REF position", "m")
        self._vector(row, "vel_world", ref.GetPosDt, "m/s", "chassis REF.GetPosDt", "mps")
        self._vector(row, "vel_body", lambda: ref.TransformDirectionParentToLocal(ref.GetPosDt()), "m/s", "chassis REF velocity transformed to REF frame", "mps")
        self._vector(row, "acc_com_world", lambda: body.GetPosDt2(), "m/s^2", "chassis body COM.GetPosDt2", "mps2")
        self._vector(row, "ang_vel_world", lambda: ref.GetAngVelParent(), "rad/s", "chassis REF.GetAngVelParent", "radps")
        self._vector(row, "ang_vel_body", lambda: ref.GetAngVelLocal(), "rad/s", "chassis REF.GetAngVelLocal", "radps")
        self._vector(row, "ang_acc_world", lambda: body.GetAngAccParent(), "rad/s^2", "chassis body.GetAngAccParent", "radps2")
        for i in range(4):
            self._scalar(row, f"quat_e{i}", lambda i=i: getattr(ref.GetRot(), f"e{i}"), "1", f"chassis REF rotation e{i}")
        for axis in ("roll", "pitch", "yaw"):
            self._scalar(row, f"{axis}_rate_radps", lambda axis=axis: getattr(vehicle, f"Get{axis.title()}Rate")(), "rad/s", f"vehicle.Get{axis.title()}Rate")
        self._scalar(row, "yaw_rad", lambda: ref.GetRot().GetCardanAnglesZYX().z, "rad", "chassis REF.GetRot.GetCardanAnglesZYX.z")
        self._scalar(row, "body_slip_rad", lambda: vehicle.GetSlipAngle(), "rad", "vehicle.GetSlipAngle")
        self._scalar(row, "transmission_current_gear", lambda: vehicle.GetTransmission().GetCurrentGear(), "index", "transmission.GetCurrentGear (-1 reverse, 0 neutral)")
        self._vector(row, "chassis_contact_force_world", lambda: body.GetContactForce(), "N", "chassis.GetContactForce", "n")
        self._vector(row, "chassis_contact_torque_world", lambda: body.GetContactTorque(), "N m", "chassis.GetContactTorque", "nm")
        if self._wheel_specs is None:
            import pychrono.vehicle as veh
            self._wheel_specs = tuple(zip(WHEEL_NAMES, (0, 0, 1, 1), (veh.LEFT, veh.RIGHT, veh.LEFT, veh.RIGHT)))
        for name, axle, side in self._wheel_specs:
            tire = self._read(f"{name}.GetTire", lambda: vehicle.GetTire(axle, side), None)
            suspension = self._read(f"{name}.GetSuspension", lambda: vehicle.GetSuspension(axle), None)
            for key, method, unit in (("longitudinal_slip", "GetLongitudinalSlip", "1"), ("slip_angle_rad", "GetSlipAngle", "rad"),
                                     ("camber_angle_rad", "GetCamberAngle", "rad"), ("deflection_m", "GetDeflection", "m"), ("radius_m", "GetRadius", "m")):
                self._scalar(row, f"{name}_{key}", lambda method=method: getattr(tire, method)(), unit, f"{name}.{method} native tire-model convention")
            tire_force = self._read(f"{name}.ReportTireForce", lambda: tire.ReportTireForce(scene.terrain), None)
            force = self._vector(row, f"{name}_force_world", lambda: tire_force.force, "N", f"{name}.ReportTireForce.force", "n")
            self._vector(row, f"{name}_moment_world", lambda: tire_force.moment, "N m", f"{name}.ReportTireForce.moment", "nm")
            self._vector(row, f"{name}_hub_world", lambda: vehicle.GetSpindlePos(axle, side), "m", f"{name}.GetSpindlePos", "m")
            velocity = self._vector(row, f"{name}_hub_velocity_world", lambda: vehicle.GetSpindleLinVel(axle, side), "m/s", f"{name}.GetSpindleLinVel", "mps")
            angular = self._vector(row, f"{name}_angular_velocity_world", lambda: vehicle.GetSpindleAngVel(axle, side), "rad/s", f"{name}.GetSpindleAngVel", "radps")
            spin = self._read(f"{name}.GetSpindleRot.GetAxisY", lambda: _xyz(vehicle.GetSpindleRot(axle, side).GetAxisY()), np.full(3, np.nan))
            heading = np.cross(spin, (0., 0., 1.))
            norm = np.linalg.norm(heading)
            heading = heading/norm if norm > 1e-12 else np.full(3, np.nan)
            omega, vx = np.dot(angular, spin), np.dot(velocity, heading)
            circum = omega*row[f"{name}_radius_m"]
            derived = (
                ("spindle_omega_radps", omega, "rad/s", "world spindle angular velocity dot spin axis; positive forward rolling"),
                ("wheel_vx_horizontal_mps", vx, "m/s", "hub world velocity dot horizontal wheel heading"),
                ("circumferential_speed_mps", circum, "m/s", "spin rate * undeformed radius"),
                ("horizontal_slip_speed_mps", circum-vx, "m/s", "circumferential speed minus horizontal hub speed; approximate, not native slip"),
                ("horizontal_slip_ratio", (circum-vx)/max(abs(vx), .1), "1", "horizontal_slip_speed / max(abs(horizontal hub speed),0.1 m/s)"),
                ("force_horizontal_heading_n", np.dot(force, heading), "N", "world tire force dot horizontal wheel heading"),
                ("force_world_vertical_n", force[2], "N", "world tire force Z component, NOT terrain-normal load"),
            )
            for key, value, unit, source in derived:
                self._put(row, f"{name}_{key}", value, unit, f"{name}: {source}")
            self._scalar(row, f"{name}_axle_speed_radps", lambda: suspension.GetAxleSpeed(side), "rad/s", f"{name}.suspension.GetAxleSpeed native sign")
            self._scalar(row, f"{name}_driveline_spindle_torque_nm", lambda: vehicle.GetDriveline().GetSpindleTorque(axle, side), "N m", f"{name}.driveline.GetSpindleTorque native sign; save factors without assuming wheel power sign")
            # The AMD SWIG build does not wrap vector<ForceTSDA>; calling
            # ReportSuspensionForce allocates an unowned opaque object on every
            # frame. The typed HMMWV double-wishbone getters expose the same
            # spring/shock quantities as scalars, without that binding leak.
            if axle not in self._suspensions:
                import pychrono.vehicle as veh
                self._suspensions[axle] = self._read(f"suspension[{axle}].CastToChDoubleWishbone",
                    lambda: veh.CastToChDoubleWishbone(suspension), None)
            typed = self._suspensions[axle]
            for key, method, unit in (
                ("spring_force_n", "GetSpringForce", "N"), ("spring_length_m", "GetSpringLength", "m"),
                ("spring_deformation_m", "GetSpringDeformation", "m"), ("shock_force_n", "GetShockForce", "N"),
                ("shock_length_m", "GetShockLength", "m"), ("shock_velocity_mps", "GetShockVelocity", "m/s"),
            ):
                self._scalar(row, f"{name}_suspension_{key}", lambda method=method: getattr(typed, method)(side),
                    unit, f"{name}.CastToChDoubleWishbone.{method} native convention")
        return row

    def on_frame(self, scene, frame, state17, pose3, action3, *, command_context=None):
        if frame < 0:
            return
        if self._finished or frame != len(self.rows):
            raise ValueError("Telemetry anchors must be contiguous, start at frame zero, and precede finish")
        self.rows.append(self._snapshot(scene, frame, action3, False, command_context))

    def on_substep(self, scene, frame, substep, dt_s, action3=None):
        """Call after Synchronize and before Advance; integrate that step's power."""
        if frame < 0:
            return
        if self._finished or frame != len(self.rows)-1 or dt_s <= 0:
            raise ValueError("on_substep needs the current pre-interval on_frame and a positive step")
        interval = self.intervals.setdefault(frame, {"pre": [], "post": []})
        if substep != len(interval["pre"]):
            raise ValueError("Substeps must start at zero and be contiguous within each interval")
        ts = float(scene.system.GetChTime())
        powers = self._power(scene)
        interval["pre"].append({"time_s": ts, "dt_s": float(dt_s), **{f"{channel}_power_kw": powers[f"{channel}_power_kw"] for channel in POWER_CHANNELS}, **self._fast_risk(scene)})

    def on_post_substep(self, scene, frame, substep):
        """Call immediately after Advance for post-solve attitude/contact extrema."""
        if frame < 0:
            return
        interval = self.intervals.get(frame)
        if interval is None or substep != len(interval["post"]) or substep >= len(interval["pre"]):
            raise ValueError("on_post_substep must follow its corresponding on_substep")
        interval["post"].append({"time_s": float(scene.system.GetChTime()), **self._fast_risk(scene)})

    def _interval_arrays(self):
        rows = []
        for index, (start, end) in enumerate(zip(self.rows, self.rows[1:])):
            duration = end["chrono_time_s"]-start["chrono_time_s"]
            if duration <= 0:
                raise ValueError("Recorded Chrono timestamps must be strictly increasing")
            samples = self.intervals.get(index, {"pre": [], "post": []})
            pre, post = samples["pre"], samples["post"]
            solver = bool(pre)
            if solver:
                expected = start["chrono_time_s"]
                for sample in pre:
                    if not np.isclose(sample["time_s"], expected, rtol=0., atol=1e-7):
                        raise ValueError(f"Noncontiguous physics timestamps in interval {index}")
                    expected += sample["dt_s"]
                if not np.isclose(expected, end["chrono_time_s"], rtol=0., atol=1e-7):
                    raise ValueError(f"Physics integration does not cover interval {index}")
                if post and len(post) != len(pre):
                    raise ValueError(f"Partial post-step risk coverage in interval {index}")
                for before, after in zip(pre, post):
                    if not np.isclose(before["time_s"]+before["dt_s"], after["time_s"], rtol=0., atol=1e-7):
                        raise ValueError(f"Post-step timestamp mismatch in interval {index}")
            row = {"frame": index, "start_time_s": start["time_s"], "end_time_s": end["time_s"], "duration_s": duration,
                   "physics_power_sample_count": len(pre), "post_step_risk_sample_count": len(post), "solver_step_work": int(solver)}
            for channel in POWER_CHANNELS:
                # Frame-only fallback is explicitly marked as a coarse left
                # endpoint estimate. It is never advertised as solver-rate work.
                powers = np.asarray([sample[f"{channel}_power_kw"] for sample in pre] if solver else [start[f"{channel}_power_kw"]])
                widths = np.asarray([sample["dt_s"] for sample in pre] if solver else [duration])
                valid = np.isfinite(powers).all()
                for kind, values in (("signed", powers), ("positive", np.maximum(powers, 0.)), ("negative_magnitude", np.maximum(-powers, 0.))):
                    row[f"{channel}_{kind}_work_kj"] = float(np.dot(values, widths)) if valid else np.nan
                row[f"{channel}_mean_power_kw"] = row[f"{channel}_signed_work_kj"]/duration
            risk = [start, *post, end] if post else [start, *pre, end]
            for axis in ("roll", "pitch"):
                values = [abs(sample[f"{axis}_rad"]) for sample in risk]
                row[f"max_abs_{axis}_rad"] = _finite_extreme(values) if np.isfinite(values).all() else np.nan
            for field in ("chassis_contact_resultant_n", "asset_contact_max_resultant_n"):
                values = [sample[field] for sample in risk]
                row[f"max_{field}"] = _finite_extreme(values) if np.isfinite(values).all() else np.nan
            slips = [abs(sample[f"{name}_longitudinal_slip"]) for sample in (start, end) for name in WHEEL_NAMES]
            loads = [sample[f"{name}_force_world_vertical_n"] for sample in (start, end) for name in WHEEL_NAMES]
            row["endpoint_max_abs_native_longitudinal_slip"] = _finite_extreme(slips) if np.isfinite(slips).all() else np.nan
            row["endpoint_min_world_vertical_tire_force_n"] = _finite_extreme(loads, maximum=False) if np.isfinite(loads).all() else np.nan
            rows.append(row)
        return {key: np.asarray([row[key] for row in rows], dtype=np.float64) for key in rows[0]} if rows else {}

    def finish(self, scene, N, terminal_state17, terminal_pose3, last_action3):
        if self._finished or N != len(self.rows) or N < 1:
            raise ValueError("finish requires exactly N collected nonterminal samples")
        self.rows.append(self._snapshot(scene, N, last_action3, True))
        arrays = {key: np.asarray([row.get(key, np.nan) for row in self.rows], dtype=np.float64) for key in sorted(self.fields)}
        intervals = self._interval_arrays()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.out_dir / "rich_telemetry.npz", **arrays)
        np.savez_compressed(self.out_dir / "rich_intervals.npz", **intervals)
        for key, spec in self.fields.items():
            spec["finite_count"] = int(np.isfinite(arrays[key]).sum())
            spec["missing_count"] = int((~np.isfinite(arrays[key])).sum())
        source = Path(__file__).resolve()
        metadata = {
            "schema": "fdm_rich_telemetry_v2", "sample_rows_including_terminal": len(self.rows), "interval_rows": N,
            "record_dt_nominal_s": self.record_dt_s, "start_chrono_time_s": self._start_time,
            "observer_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "case_sha256": hashlib.sha256(self.case_path.read_bytes()).hexdigest() if self.case_path else None,
            "scene_configuration_labels_only": getattr(scene, "config", None),
            "passive_getters_only": True, "fields": self.fields, "getter_capabilities": self.capabilities,
            "solver_step_work_interval_count": int(intervals["solver_step_work"].sum()),
            "post_step_risk_interval_count": int((intervals["post_step_risk_sample_count"] > 0).sum()),
            "alignment": "Sample i is pre-interval state and applied command. Interval i covers sample i to i+1. N is an actual terminal state, never a duplicated training row. Settling is excluded.",
            "quadrature": "sum(power at synchronized pre-step state * actual physics dt), signed and positive/negative split. Without on_substep: explicit pre-frame left-endpoint estimate. No claim of continuous-time exactness.",
            "power_semantics": {
                "engine_interface": "Engine output motorshaft torque times transmission-reported motorshaft speed (feedback to engine). Matches existing runner power definition. Mechanical interface power, not fuel consumption.",
                "engine_rotor": "Engine output motorshaft torque times engine.GetMotorSpeed. Dynamic engine rotor speed can differ from interface feedback speed; raw factors retained.",
                "driveshaft": "Transmission output driveshaft torque times driveline output driveshaft speed. Matched transmission/driveline interface; gearbox-side mechanical power.",
            },
            "unavailable_by_design": {"fuel_mass_flow_kgps": "No fuel-flow model/getter established; do not infer from mechanical work", "contact_pair_identity": "Body force resultants only", "soil_sinkage_m": "Rigid heightfield does not model soil deformation"},
            "limitations": [
                "Native tire slip convention depends on tire model. Derived horizontal slip speed/ratio is an approximation on slopes and is separately named.",
                "World-Z tire force is not terrain-normal load. No unvalidated terrain-normal query is used for supervision.",
                "Suspension scalars use typed double-wishbone getters. Other suspension types remain NaN with capability metadata; opaque SWIG vector getters are never called.",
                "Contact resultant can miss cancelling or between-sample impulses. Full post-step contact/attitude coverage requires on_post_substep.",
                "Slip/load interval summaries use measured endpoints at the recording rate, not physics-rate maxima.",
                "Rigid terrain plus TMeasy provides tire/vehicle dynamics, not deformable-soil terramechanics. Terrain and friction configs are labels/provenance, never automatically NN inputs.",
            ],
        }
        (self.out_dir / "rich_telemetry.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
        self._finished = True
        return metadata
