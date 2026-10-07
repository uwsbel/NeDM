#!/usr/bin/env python
"""Measure, in Chrono, the HMMWV geometry that its reduced state does not carry.

The numbers at the top of ``hmmwv_nrd_rollout.py`` come from this script: the chassis height
at rest, where each wheel sits in the chassis frame, the tire radius, how far the front wheels
turn at full steering, and the sign of the wheel speed when driving forward. The vehicle is
built with the settings NeDM collects data with (``configs/hmmwv``): HMMWV_Full, SMC contact,
TMEASY tires, pitman-arm steering, on flat rigid ground.

    python scripts/render/hmmwv_measure_geometry.py          # needs pychrono with the vehicle module
"""

from __future__ import annotations

import json
import math
import os

import pychrono as chrono
import pychrono.vehicle as veh

STEP_S = 0.002
WHEELS = (("fl", 0, veh.LEFT), ("fr", 0, veh.RIGHT), ("rl", 1, veh.LEFT), ("rr", 1, veh.RIGHT))


def main() -> None:
    # Chrono 10 renamed vehicle.SetDataPath to vehicle.SetVehicleDataPath.
    (getattr(veh, "SetVehicleDataPath", None) or veh.SetDataPath)(chrono.GetChronoDataPath() + "vehicle/")
    hmmwv = veh.HMMWV_Full()
    hmmwv.SetContactMethod(chrono.ChContactMethod_SMC)
    hmmwv.SetChassisFixed(False)
    hmmwv.SetInitPosition(chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 1.6), chrono.QuatFromAngleZ(0)))
    hmmwv.SetEngineType(veh.EngineModelType_SHAFTS)
    hmmwv.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SHAFTS)
    hmmwv.SetDriveType(veh.DrivelineTypeWV_AWD)
    hmmwv.SetSteeringType(veh.SteeringTypeWV_PITMAN_ARM)
    hmmwv.SetTireType(veh.TireModelType_TMEASY)
    hmmwv.SetTireStepSize(0.001)
    hmmwv.Initialize()
    system = hmmwv.GetSystem()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    terrain = veh.RigidTerrain(system)
    material = chrono.ChContactMaterialSMC()
    material.SetFriction(0.9)
    terrain.AddPatch(material, chrono.CSYSNORM, 400.0, 400.0)
    terrain.Initialize()
    vehicle = hmmwv.GetVehicle()

    def run(steering: float, throttle: float, braking: float, seconds: float) -> None:
        for _ in range(int(seconds / STEP_S)):
            t = system.GetChTime()
            inputs = veh.DriverInputs()
            inputs.m_steering, inputs.m_throttle, inputs.m_braking = steering, throttle, braking
            terrain.Synchronize(t)
            hmmwv.Synchronize(t, inputs, terrain)
            terrain.Advance(STEP_S)
            hmmwv.Advance(STEP_S)

    def wheels() -> dict:
        ref = hmmwv.GetChassis().GetBody().GetFrameRefToAbs()
        out = {}
        for name, axle, side in WHEELS:
            position = ref.TransformPointParentToLocal(vehicle.GetSpindlePos(axle, side))
            rotation = vehicle.GetSpindleRot(axle, side)
            axis = ref.TransformDirectionParentToLocal(rotation.GetAxisY())
            out[name] = {
                "position_in_chassis_m": [round(position.x, 4), round(position.y, 4), round(position.z, 4)],
                "turn_deg": round(math.degrees(math.atan2(-axis.x, axis.y)), 3),
                "speed_about_axle_radps": round(vehicle.GetSpindleAngVel(axle, side).Dot(rotation.GetAxisY()), 3),
            }
        return out

    run(0.0, 0.0, 1.0, 3.0)                     # settle on the brake
    result = {
        "chassis_height_at_rest_m": round(hmmwv.GetChassis().GetBody().GetFrameRefToAbs().GetPos().z, 4),
        "wheels_at_rest": wheels(),
        "tire_radius_m": round(vehicle.GetTire(0, veh.LEFT).GetRadius(), 4),
        "tire_width_m": round(vehicle.GetTire(0, veh.LEFT).GetWidth(), 4),
    }
    for steering in (1.0, -1.0, 0.5):
        run(steering, 0.0, 1.0, 1.5)
        result[f"turn_deg_at_steering_{steering:+.1f}"] = {name: w["turn_deg"] for name, w in wheels().items()}
    run(0.0, 0.0, 1.0, 1.0)
    run(0.0, 0.5, 0.0, 2.0)
    result["driving_forward"] = {
        "speed_mps": round(hmmwv.GetChassis().GetBody().GetPosDt().x, 3),
        "wheel_speed_radps": {name: w["speed_about_axle_radps"] for name, w in wheels().items()},
    }
    folder = chrono.GetChronoDataPath() + "vehicle/hmmwv/"
    result["mesh_folder"] = folder
    result["meshes"] = sorted(f for f in os.listdir(folder) if f.endswith(".obj"))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
