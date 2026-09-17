"""Passive 20 Hz smooth-hill diagnostics for the existing frame-observer hook.

This observer only reads state/forces and terrain queries. It never advances,
synchronizes, renders, adds bodies, or changes the driver/solver. A chassis
contact resultant is not a contact-pair census; terrain-relative reference and
hub heights are explicitly not underbody clearances.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _xyz(value):
    return np.asarray([value.x, value.y, value.z], dtype=np.float64)


def _unit(value):
    value = np.asarray(value, dtype=np.float64)
    length = np.linalg.norm(value)
    return value / length if length > 1e-12 else np.full(3, np.nan)


def surface_projections(spin_axis, velocity, force, normal, omega, radius):
    """Measured force/speed projected onto the queried terrain tangent plane."""
    normal = _unit(normal)
    heading = _unit(np.cross(spin_axis, [0., 0., 1.]))
    tangent = _unit(heading - np.dot(heading, normal) * normal)
    along = float(np.dot(velocity, tangent))
    return {
        "surface_tangent_speed_mps": along,
        "surface_tangent_force_n": float(np.dot(force, tangent)),
        "surface_normal_force_n": float(np.dot(force, normal)),
        "circumferential_speed_mps": float(omega * radius),
        "surface_slip_speed_mps": float(omega * radius - along),
        "surface_grade_rad": float(np.arctan2(tangent[2], np.linalg.norm(tangent[:2]))),
    }


class SmoothHillDiagnostics:
    """Use ``args.frame_observer = SmoothHillDiagnostics(out_dir, case_path)``.

    Saves ``slope_diagnostics.npz`` (one column per numeric measurement, including
    the measured terminal row) and a units/availability/provenance JSON. Optional
    unsupported getters produce NaN plus an explicit availability error.
    """

    def __init__(self, out_dir, case_path):
        self.out_dir = Path(out_dir)
        self.case_path = Path(case_path).resolve()
        self.case = json.loads(self.case_path.read_text())
        self.rows: list[dict[str, float]] = []
        self.errors: dict[str, str] = {}
        self._chrono = self._capture_row = self._wheel_specs = None
        self.dt = .05
        self.profile_offsets = np.arange(-3., 3.01, .5)

    def _optional(self, name, getter, default=np.nan):
        try:
            return getter()
        except (AttributeError, TypeError, RuntimeError, ValueError) as exc:
            self.errors.setdefault(name, f"{type(exc).__name__}: {exc}")
            return default

    def _ground(self, scene, x, y):
        # RigidTerrain casts down FROM the query Z. Querying at zero inside a
        # positive hill silently returns the fallback flat height/normal.
        top = float(scene.config["terrain"]["height_max_m"]) + 5.
        query = self._chrono.ChVector3d(float(x), float(y), top)
        return float(scene.terrain.GetHeight(query)), _unit(_xyz(scene.terrain.GetNormal(query)))

    def _record(self, scene, frame, state17, pose3, action3, terminal):
        if self._chrono is None:
            import pychrono as chrono
            from nedm.hmmwv_data import WHEEL_SPECS, capture_row
            self._chrono, self._capture_row, self._wheel_specs = chrono, capture_row, WHEEL_SPECS
        vehicle = scene.hmmwv.GetVehicle()
        body = scene.hmmwv.GetChassis().GetBody()
        action = np.asarray(action3)
        inputs = SimpleNamespace(m_steering=float(action[0]), m_throttle=float(action[1]), m_braking=float(action[2]))
        captured = self._capture_row(scene.hmmwv, scene.terrain, "smooth_hill_probe", "diagnostic",
            self.case["id"], "geometry_probe", int(frame), frame*self.dt, inputs, include_tires=True)
        row = {key: float(value) for key, value in captured.items()
               if isinstance(value, (float, int, np.number))}
        row.update(frame=float(frame), is_terminal=float(terminal), time_s=float(frame*self.dt),
                   chrono_time_s=float(scene.system.GetChTime()))
        contact = self._optional("chassis.GetContactForce", lambda: _xyz(body.GetContactForce()), np.full(3, np.nan))
        torque = self._optional("chassis.GetContactTorque", lambda: _xyz(body.GetContactTorque()), np.full(3, np.nan))
        for axis, force, moment in zip("xyz", contact, torque):
            row[f"chassis_contact_force_world_{axis}_n"] = float(force)
            row[f"chassis_contact_torque_world_{axis}_nm"] = float(moment)
        row["chassis_contact_resultant_n"] = float(np.linalg.norm(contact))
        engine, transmission = vehicle.GetEngine(), vehicle.GetTransmission()
        row["engine_motor_speed_radps"] = float(engine.GetMotorSpeed())
        row["engine_motorshaft_torque_nm"] = float(engine.GetOutputMotorshaftTorque())
        row["transmission_output_speed_radps"] = float(transmission.GetOutputMotorshaftSpeed())
        row["runner_shaft_power_kw"] = row["engine_motorshaft_torque_nm"]*row["transmission_output_speed_radps"]/1000.
        row["transmission_current_gear"] = float(self._optional("transmission.GetCurrentGear", transmission.GetCurrentGear)) if hasattr(transmission, "GetCurrentGear") else np.nan
        if not hasattr(transmission, "GetCurrentGear"):
            self.errors.setdefault("transmission.GetCurrentGear", "Getter not exposed by this binding")
        x, y, yaw = map(float, pose3)
        ground_z, normal = self._ground(scene, x, y)
        row["terrain_height_at_chassis_ref_m"] = ground_z
        row["chassis_ref_height_above_terrain_m"] = row["pos_z_m"] - ground_z
        for axis, value in zip("xyz", normal):
            row[f"terrain_normal_at_chassis_ref_{axis}"] = float(value)
        heights = np.asarray([self._ground(scene, x+s*np.cos(yaw), y+s*np.sin(yaw))[0]
                              for s in self.profile_offsets])
        grade = np.arctan(np.diff(heights)/.5)
        row["longitudinal_profile_max_abs_grade_rad"] = float(np.max(np.abs(grade)))
        row["longitudinal_profile_max_grade_change_rad_per_m"] = float(np.max(np.abs(np.diff(grade)))/.5)
        for index, height in enumerate(heights):
            row[f"longitudinal_terrain_height_{index:02d}_m"] = float(height)
        for name, axle, side in self._wheel_specs:
            tire = vehicle.GetTire(axle, side)
            position = self._optional(f"{name}.GetSpindlePos", lambda: _xyz(vehicle.GetSpindlePos(axle, side)), np.full(3, np.nan))
            for axis, value in zip("xyz", position):
                row[f"{name}_hub_world_{axis}_m"] = float(value)
            radius = float(tire.GetRadius())
            row[f"{name}_radius_m"] = radius
            if np.isfinite(position).all():
                height, normal = self._ground(scene, *position[:2])
            else:
                height, normal = np.nan, np.full(3, np.nan)
            row[f"{name}_terrain_height_at_hub_xy_m"] = height
            row[f"{name}_hub_vertical_gap_minus_radius_m"] = float(position[2]-height-radius)
            spin = _xyz(vehicle.GetSpindleRot(axle, side).GetAxisY())
            velocity = _xyz(vehicle.GetSpindleLinVel(axle, side))
            force = np.asarray([row[f"{name}_force_world_{axis}_n"] for axis in "xyz"])
            for axis, value in zip("xyz", normal):
                row[f"{name}_queried_terrain_normal_{axis}"] = float(value)
            projected = surface_projections(spin, velocity, force, normal,
                row[f"{name}_spindle_omega_radps"], radius)
            row.update({f"{name}_{key}": value for key, value in projected.items()})
        self.rows.append(row)

    def on_frame(self, scene, frame, state17, pose3, action3):
        self._record(scene, frame, state17, pose3, action3, False)

    def finish(self, scene, N, terminal_state17, terminal_pose3, last_action3):
        self._record(scene, N, terminal_state17, terminal_pose3, last_action3, True)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for row in self.rows for key in row})
        arrays = {key: np.asarray([row.get(key, np.nan) for row in self.rows], np.float64) for key in keys}
        arrays["longitudinal_profile_offsets_m"] = self.profile_offsets
        np.savez_compressed(self.out_dir / "slope_diagnostics.npz", **arrays)
        contact = arrays["chassis_contact_resultant_n"]
        hits = np.flatnonzero(np.isfinite(contact) & (contact > 1.))
        finite_contact = contact[np.isfinite(contact)]
        source = Path(__file__).resolve()
        metadata: dict[str, Any] = {
            "schema": 1, "case_id": self.case["id"], "rows_including_terminal": len(self.rows),
            "dt_s": self.dt, "sampling": "Pre-interval 20 Hz plus actual terminal state; contacts can occur between samples",
            "passive_getters_only": True, "case_sha256": hashlib.sha256(self.case_path.read_bytes()).hexdigest(),
            "observer_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "optional_getter_errors": self.errors,
            "contact_pair_identity_available": False, "exact_underbody_clearance_available": False,
            "maximum_sampled_chassis_contact_resultant_n": float(finite_contact.max()) if len(finite_contact) else None,
            "first_sampled_chassis_contact_over_1n_s": float(arrays["time_s"][hits[0]]) if len(hits) else None,
            "chassis_reference_height_range_m": [float(arrays["chassis_ref_height_above_terrain_m"].min()), float(arrays["chassis_ref_height_above_terrain_m"].max())],
            "profile_offsets_m": self.profile_offsets.tolist(),
            "terrain_query_origin_z_m": float(scene.config["terrain"]["height_max_m"])+5.,
            "terrain_configuration": scene.config["terrain"],
            "smooth_hill_design": self.case.get("smooth_hill"),
            "field_semantics": {
                "pos_z_m": "Chassis REF world Z, not center of mass or underbody bottom",
                "longitudinal_slip": "Direct TMeasy GetLongitudinalSlip getter; retain its native model convention",
                "force_wheel_fz_n": "Existing capture_row convention: WORLD vertical force, not terrain normal",
                "force_wheel_fx_n": "Existing capture_row convention: horizontal wheel heading projection",
                "surface_normal_force_n": "Tire world force dotted with unit terrain normal queried beneath wheel hub",
                "surface_tangent_force_n": "Tire world force along horizontal wheel heading projected into queried terrain tangent plane",
                "surface_slip_speed_mps": "omega*undeformed_radius minus wheel-hub terrain-tangent speed; diagnostic approximation, not direct tire-model slip",
                "hub_vertical_gap_minus_radius_m": "Hub Z - ground beneath hub - undeformed radius; NOT clearance/contact penetration on a slope",
                "chassis_ref_height_above_terrain_m": "REF Z - ground beneath REF; NOT underbody clearance",
                "runner_shaft_power_kw": "Matches runner: engine output motorshaft torque times transmission output motorshaft speed /1000; retain both raw factors",
                "longitudinal_profile_max_grade_change_rad_per_m": "Maximum adjacent 0.5 m secant grade change over sampled +/-3 m chassis-heading profile",
            },
            "limitations": [
                "Chassis contact force is a resultant with no opponent identity. A nonzero force flags chassis collision involvement; zero sampled resultant cannot exclude brief or mutually cancelling contacts.",
                "Asset contact logs alone do not exclude chassis-terrain interaction. No claim of exact underbody clearance is made.",
                "Terrain height/normal beneath hub approximates tire support geometry on curved or steep terrain; direct tire slip, deflection and forces are also saved.",
                "Twenty Hz contact sampling is not the runner's 500 Hz asset-contact census. Persistent absence supports but does not prove absence of every chassis collision.",
                "A smooth analytic profile can still cause geometric breakover blocking. Interpret the actual quantized profile, chassis contact and wheel effort together.",
            ],
        }
        (self.out_dir / "slope_diagnostics.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
