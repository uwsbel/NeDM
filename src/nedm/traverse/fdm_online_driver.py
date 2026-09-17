"""Mutable native Chrono PID reference, preserving the speed-controller state.

Initial construction matches the diverse collector exactly. A changed path
mutates the existing ChBezierCurve and resets only its steering tracker. An
unchanged path or a speed-profile-only update never resets either controller.
Simulator terrain heights are used only for the native driver's 3-D reference;
this adapter is not a learned-model input or a terrain-aware route planner.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


DRIVER_CONTRACT = {
    "name": "ChPathFollowerDriver",
    "steering_gains": [.8, 0., 0.],
    "speed_gains": [.6, .05, 0.],
    "lookahead_m": 5.,
    "path_min_spacing_m": 2.,
    "path_z": "privileged terrain height + 0.5 m",
    "route_update": "Mutate shared ChBezierCurve.setPoints; reset steering only for changed 3-D knots",
    "speed_controller_reset_on_replan": False,
}


def _route_arrays(route: dict[str, Any]):
    xy = np.asarray(route["waypoints"], dtype=np.float64)
    station = np.asarray(route["stations"], dtype=np.float64)
    speed = np.asarray(route["speeds"], dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2:
        raise ValueError("An open reference requires at least two XY points")
    if station.shape != (len(xy),) or speed.shape != (len(xy),):
        raise ValueError("One station and desired speed are required per waypoint")
    if not all(np.isfinite(a).all() for a in (xy, station, speed)):
        raise ValueError("Reference coordinates, stations and speeds must be finite")
    spacing = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    if (spacing <= 1e-8).any() or not np.allclose(station-station[0], np.r_[0., spacing.cumsum()], atol=1e-4, rtol=1e-5):
        raise ValueError("Reference station encoding disagrees with its geometry")
    if (speed < 0.).any() or (speed > 10.).any():
        raise ValueError("Reference is outside the declared forward speed range")
    return tuple(a.copy() for a in (xy, station, speed))


def _fingerprint(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def reference_knots(route, terrain_height):
    """The collector's exact station filter and height query order, no resampling."""
    xy, station, _ = _route_arrays(route)
    points = []
    last_station = -10.
    for (x, y), position in zip(xy, station):
        if position-last_station < 2. and position != station[-1]:
            continue
        last_station = position
        points.append([float(x), float(y), float(terrain_height.height(x, y))+.5])
    result = np.asarray(points, dtype=np.float64)
    if len(result) < 2 or not np.isfinite(result).all():
        raise ValueError("Native driver needs at least two finite 3-D spline knots")
    return result


def _vector(chrono, points):
    vector = chrono.vector_ChVector3d()
    for x, y, z in points:
        vector.append(chrono.ChVector3d(float(x), float(y), float(z)))
    return vector


def _xyz(vector):
    return np.array([vector.x, vector.y, vector.z], dtype=np.float64)


def native_spline_control_points(chrono, knots):
    """Recover a native interpolating spline's control vertices, in memory.

    This build exposes setPoints(nodes,inCV,outCV), without a nodes-only
    setter or direct control-vertex getters. At each interval endpoint the
    cubic derivative is 3*(outCV-node) or 3*(node-inCV). A temporary native
    interpolant therefore provides the exact mathematical control polygon
    without duplicating Chrono's spline solver or lossy text serialization.
    Each update verifies reconstructed geometry against the temporary native
    spline; unchanged references never take this reconstruction path.
    """
    native_points = _vector(chrono, knots)
    temporary = chrono.ChBezierCurve(native_points)
    incoming, outgoing = knots.copy(), knots.copy()
    for index in range(len(knots)-1):
        outgoing[index] = knots[index]+_xyz(temporary.EvalDer(index, 0.))/3.
        incoming[index+1] = knots[index+1]-_xyz(temporary.EvalDer(index, 1.))/3.
    return native_points, _vector(chrono, incoming), _vector(chrono, outgoing), temporary


class OnlinePathFollower:
    """A persistent native PID driver with measured-pose path replacement.

    Instantiate once, before the same 0.8 s settlement used by collection.
    The native driver is initialized here, exactly as in the collector.
    The caller continues scheduling desired speed and the existing control
    rate/steering slew limit. update_route does not advance time, change the
    vehicle, call the speed controller's Reset, or choose a new target speed.
    """

    def __init__(self, chrono, veh, vehicle, route, terrain_height):
        self.chrono, self.veh = chrono, veh
        self.vehicle, self.terrain_height = vehicle, terrain_height
        self._arrays = _route_arrays(route)
        self.knots_xyz = reference_knots(route, terrain_height)
        self.curve = chrono.ChBezierCurve(_vector(chrono, self.knots_xyz))
        self.native_driver = veh.ChPathFollowerDriver(vehicle, self.curve, "route", float(self._arrays[2][0]))
        self.native_steering_controller = self.native_driver.GetSteeringController()
        self.native_speed_controller = self.native_driver.GetSpeedController()
        self.native_steering_controller.SetLookAheadDistance(5.)
        self.native_steering_controller.SetGains(.8, 0., 0.)
        self.native_speed_controller.SetGains(.6, .05, 0.)
        self.native_driver.Initialize()
        if not callable(getattr(self.curve, "setPoints", None)) or not callable(getattr(self.native_steering_controller, "Reset", None)):
            raise RuntimeError("This Chrono build lacks mutable spline / steering-only reset bindings")
        self.route_fingerprint = _fingerprint(self._arrays)
        self.updates = []
        self.update_calls = 0
        self.curve_mutations = 0
        self.steering_resets = 0
        self.speed_controller_resets = 0

    def SetDesiredSpeed(self, speed):
        return self.native_driver.SetDesiredSpeed(float(speed))

    def Synchronize(self, *args, **kwargs):
        return self.native_driver.Synchronize(*args, **kwargs)

    def GetInputs(self):
        return self.native_driver.GetInputs()

    def Advance(self, step):
        return self.native_driver.Advance(float(step))

    def GetSteeringController(self):
        return self.native_steering_controller

    def GetSpeedController(self):
        return self.native_speed_controller

    def update_route(self, route, current_pose):
        pose = np.asarray(current_pose, dtype=np.float64)
        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("current_pose must be measured [world_x, world_y, yaw]")
        new_arrays = _route_arrays(route)
        fingerprint = _fingerprint(new_arrays)
        old_fingerprint = self.route_fingerprint
        identical = all(np.array_equal(a, b) for a, b in zip(self._arrays, new_arrays))
        speed_changed = not np.array_equal(self._arrays[2], new_arrays[2])
        geometry_inputs_unchanged = all(np.array_equal(a, b) for a, b in zip(self._arrays[:2], new_arrays[:2]))
        # Static terrain and unchanged path imply unchanged 3-D knots. Avoid
        # even querying terrain for ordinary repeated or speed-only updates.
        new_knots = self.knots_xyz if geometry_inputs_unchanged else reference_knots(route, self.terrain_height)
        geometry_changed = not np.array_equal(self.knots_xyz, new_knots)
        native_curve_error = None
        if geometry_changed:
            points, incoming, outgoing, temporary = native_spline_control_points(self.chrono, new_knots)
            self.curve.setPoints(points, incoming, outgoing)
            # Reset the tracker to the current measured vehicle frame; the
            # same native speed-controller object and its integral survive.
            self.native_steering_controller.Reset(self.vehicle.GetRefFrame())
            self.curve_mutations += 1
            self.steering_resets += 1
            probes = ((index, fraction) for index in range(len(new_knots)-1) for fraction in (.25, .5, .75))
            native_curve_error = max(float(np.linalg.norm(_xyz(self.curve.Eval(index, fraction))-_xyz(temporary.Eval(index, fraction))))
                                     for index, fraction in probes)
            if native_curve_error > 1e-9:
                raise RuntimeError(f"Mutable native curve disagrees with fresh native interpolant: {native_curve_error} m")
            self.knots_xyz = new_knots.copy()
        self._arrays = new_arrays
        self.route_fingerprint = fingerprint
        self.update_calls += 1
        record = {
            "update_index": self.update_calls,
            "chrono_time_s": float(self.vehicle.GetChTime()),
            "current_pose": pose.tolist(),
            "previous_route_fingerprint": old_fingerprint,
            "route_fingerprint": fingerprint,
            "identical_reference": identical,
            "geometry_changed": geometry_changed,
            "speed_profile_changed": speed_changed,
            "curve_mutated": geometry_changed,
            "steering_reset": geometry_changed,
            "speed_controller_reset": False,
            "whole_driver_reset": False,
            "native_curve_identity_preserved": True,
            "knot_count": len(self.knots_xyz),
            "native_interpolant_max_error_m": native_curve_error,
            "current_desired_speed_changed_by_update": False,
        }
        self.updates.append(record)
        return record


__all__ = ["OnlinePathFollower", "reference_knots", "native_spline_control_points", "DRIVER_CONTRACT"]
