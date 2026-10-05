"""Chrono pool table: two balls on cloth inside four cushions (SMC contact).

Ball A starts at a fixed spot with launch velocity (vx, vy) and no spin
(centre-ball strike); ball B starts at rest at a fixed spot. Cloth, ball-ball
and ball-cushion contacts get their own friction and restitution in an
add-contact callback, because Chrono's default composition takes the minimum
of the two bodies' values.

Recorded every record step: both balls' full 3D state (position, velocity,
angular velocity in the world frame) and, per impulsive contact pair (order
PAIRS), the maximum overlap over the preceding interval (> 0 means contact).
"""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

BALL_FIELDS = ["x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz"]
# Model state: A then B, each [x, y, vx, vy, wx, wy, wz].
STATE_COLUMNS = [0, 1, 3, 4, 6, 7, 8]
STATE_FIELDS = [f"{ball}_{BALL_FIELDS[c]}" for ball in "AB" for c in STATE_COLUMNS]
CUSHIONS = ["xp", "xm", "yp", "ym"]
PAIRS = ["AB", *[f"A_{c}" for c in CUSHIONS], *[f"B_{c}" for c in CUSHIONS]]


def load_config(path: Path | str) -> dict[str, Any]:
    config = json.loads(Path(path).read_text())
    sim, scene = config["simulation"], config["scene"]
    ratio = sim["record_step_s"] / sim["step_s"]
    if ratio < 1 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("record_step_s must be an integer multiple of step_s")
    for key in ("radius_m", "mass_kg", "half_length_m", "half_width_m"):
        if not scene[key] > 0:
            raise ValueError(f"{key} must be positive")
    for ball in ("ball_a_xy_m", "ball_b_xy_m"):
        x, y = scene[ball]
        if abs(x) > scene["half_length_m"] - scene["radius_m"] or abs(y) > scene["half_width_m"] - scene["radius_m"]:
            raise ValueError(f"{ball} is outside the cushions")
    return config


def separation(config: dict) -> tuple[np.ndarray, float]:
    a, b = (np.asarray(config["scene"][k], dtype=float) for k in ("ball_a_xy_m", "ball_b_xy_m"))
    d = b - a
    return d / np.linalg.norm(d), float(np.linalg.norm(d))


def aim_limit_rad(config: dict, cut_deg: float | None = None) -> float:
    """Aim offset from the A->B line that gives the requested cut angle."""
    _, distance = separation(config)
    cut = math.radians(config["launch"]["max_cut_angle_deg"] if cut_deg is None else cut_deg)
    return math.asin(2 * config["scene"]["radius_m"] * math.sin(cut) / distance)


def launch_velocity(config: dict, speed: float, aim_rad: float) -> tuple[float, float]:
    """(vx, vy) for a launch `aim_rad` to the left (+) of the A->B line."""
    u, _ = separation(config)
    c, s = math.cos(aim_rad), math.sin(aim_rad)
    return speed * (c * u[0] - s * u[1]), speed * (s * u[0] + c * u[1])


def polar_action(config: dict, vx, vy):
    """Inverse of launch_velocity, vectorised."""
    u, _ = separation(config)
    vx, vy = np.asarray(vx, float), np.asarray(vy, float)
    speed = np.hypot(vx, vy)
    aim = np.arctan2(u[0] * vy - u[1] * vx, u[0] * vx + u[1] * vy)
    return speed, aim


def _classify(info, chrono) -> str:
    if abs(info.vN.z) > 0.5:
        return "cloth"
    sphere = getattr(chrono.ChCollisionShape, "Type_SPHERE", None)
    if sphere is None:
        sphere = chrono.ChCollisionShape.SPHERE
    if info.shapeA.GetType() == sphere and info.shapeB.GetType() == sphere:
        return "ball_ball"
    return "ball_cushion"


def build_scene(config: dict, vx: float, vy: float, *, ball_b: bool = True, visualize: bool = False):
    """SMC (penalty) contact. NSC's convex friction relaxation lifts a sliding
    ball off the cloth at about mu*|slip| (measured 0.43 m/s at 2.5 m/s), so it
    cannot model a ball sliding on cloth; SMC friction has no such dilation."""
    import pychrono as chrono

    sim, scene, contact = config["simulation"], config["scene"], config["contact"]
    R, m = scene["radius_m"], scene["mass_kg"]
    system = chrono.ChSystemSMC()
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -sim["gravity_mps2"]))
    system.SetContactForceModel(chrono.ChSystemSMC.Hertz)
    # "None" keeps only the viscous tangential term, min(gt*|v_t|, mu*F_n).
    # Chrono's OneStep/MultiStep add kt*|v_t|*dt, which makes cushion friction
    # (and B's path after a cushion) depend on the physics step.
    name = contact["tangential_model"]
    system.SetTangentialDisplacementModel(getattr(chrono.ChSystemSMC, "_None" if name == "None" else name))
    system.SetAdhesionForceModel(chrono.ChSystemSMC.Constant)
    system.UseMaterialProperties(True)
    system.SetSlipVelocityThreshold(contact["slip_velocity_threshold_mps"])
    materials = {key: contact[key] for key in ("cloth", "ball_ball", "ball_cushion")}

    class PairMaterial(chrono.AddContactCallback):
        def OnAddContact(self, info, material):
            values = materials[_classify(info, chrono)]
            composite = chrono.CastToChContactMaterialCompositeSMC(material)
            composite.mu_eff = values["friction"]
            composite.cr_eff = values["restitution"]
            composite.adhesion_eff = 0.0

    # Keep the Python director alive as long as the system uses it.
    system._pair_material = PairMaterial()
    system.GetContactContainer().RegisterAddContactCallback(system._pair_material)

    base = chrono.ChContactMaterialSMC()
    base.SetYoungModulus(contact["young_modulus_pa"])
    base.SetPoissonRatio(contact["poisson_ratio"])
    base.SetFriction(0.2)
    base.SetRestitution(0.5)
    hx, hy = scene["half_length_m"], scene["half_width_m"]
    ft, ct, ch = scene["floor_thickness_m"], scene["cushion_thickness_m"], scene["cushion_height_m"]
    floor = chrono.ChBodyEasyBox(2 * hx + 2 * ct + 0.2, 2 * hy + 2 * ct + 0.2, ft, 1000, visualize, True, base)
    floor.SetPos(chrono.ChVector3d(0, 0, -ft / 2))
    floor.SetFixed(True)
    floor.SetName("floor")
    bodies = {"floor": floor}
    # Cushion nose (box bottom edge) at nose_height above the cloth, as on a
    # real table (1.27 R): the rebound force then presses the ball into the
    # cloth instead of letting cushion friction lift it.
    nose = scene["cushion_nose_height_m"]
    cz, chh = nose + (ch - nose) / 2, ch - nose
    for name, size, position in [
        ("xp", (ct, 2 * hy + 2 * ct, chh), (hx + ct / 2, 0, cz)),
        ("xm", (ct, 2 * hy + 2 * ct, chh), (-hx - ct / 2, 0, cz)),
        ("yp", (2 * hx, ct, chh), (0, hy + ct / 2, cz)),
        ("ym", (2 * hx, ct, chh), (0, -hy - ct / 2, cz)),
    ]:
        cushion = chrono.ChBodyEasyBox(*size, 1000, visualize, True, base)
        cushion.SetPos(chrono.ChVector3d(*position))
        cushion.SetFixed(True)
        cushion.SetName(f"cushion_{name}")
        bodies[f"cushion_{name}"] = cushion
    density = m / (4 / 3 * math.pi * R**3)
    z0 = R - contact["initial_sink_m"]
    for label, xy, velocity in [("A", scene["ball_a_xy_m"], (vx, vy)), ("B", scene["ball_b_xy_m"], (0.0, 0.0))]:
        if label == "B" and not ball_b:
            continue
        ball = chrono.ChBodyEasySphere(R, density, visualize, True, base)
        ball.SetName(f"ball_{label}")
        ball.SetPos(chrono.ChVector3d(xy[0], xy[1], z0))
        ball.SetPosDt(chrono.ChVector3d(velocity[0], velocity[1], 0))
        ball.SetAngVelParent(chrono.ChVector3d(0, 0, 0))
        bodies[f"ball_{label}"] = ball
    for body in bodies.values():
        body.GetCollisionModel().SetEnvelope(contact["collision_envelope_m"])
        body.GetCollisionModel().SetSafeMargin(contact["collision_safe_margin_m"])
        system.Add(body)
    return system, bodies


def simulate_episode(config: dict, vx: float, vy: float, *, ball_b: bool = True,
                     duration_s: float | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Run one shot. Contact is labelled from geometry: with penalty contact a
    pair carries force exactly when the shapes overlap. Cloth rolling
    resistance and spinning friction (not applied by Chrono's SMC contact) are
    added as torques rho*m*g opposing the horizontal / vertical spin."""
    import pychrono as chrono

    sim, scene, contact, acceptance = config["simulation"], config["scene"], config["contact"], config["acceptance"]
    dt = sim["step_s"]
    stride = round(sim["record_step_s"] / dt)
    duration = sim["duration_s"] if duration_s is None else duration_s
    steps = round(duration / dt)
    if abs(steps * dt - duration) > 1e-9 or steps % stride:
        raise ValueError("duration must be a whole number of record steps")
    system, bodies = build_scene(config, vx, vy, ball_b=ball_b)
    balls = [bodies[k] for k in ("ball_A", "ball_B") if k in bodies]
    accumulators = [ball.AddAccumulator() for ball in balls]
    R, m, g = scene["radius_m"], scene["mass_kg"], sim["gravity_mps2"]
    hx, hy, nose = scene["half_length_m"], scene["half_width_m"], scene["cushion_nose_height_m"]
    inertia = 0.4 * m * R * R
    cloth = contact["cloth"]
    roll_torque, spin_torque = cloth["rolling_resistance_m"] * m * g, cloth["spinning_resistance_m"] * m * g
    eps_w = contact["resistance_smoothing_radps"]
    on_cloth = contact["on_cloth_gap_m"]

    def read():
        out = np.empty((len(balls), 9))
        for i, ball in enumerate(balls):
            p, v, w = ball.GetPos(), ball.GetPosDt(), ball.GetAngVelParent()
            out[i] = (p.x, p.y, p.z, v.x, v.y, v.z, w.x, w.y, w.z)
        return out

    def apply_resistance(state):
        for i, ball in enumerate(balls):
            ball.EmptyAccumulator(accumulators[i])
            if state[i, 2] - R > on_cloth:
                continue
            wx, wy, wz = state[i, 6:9]
            h = math.sqrt(wx * wx + wy * wy + eps_w * eps_w)
            s = math.sqrt(wz * wz + eps_w * eps_w)
            ball.AccumulateTorque(accumulators[i], chrono.ChVector3d(-roll_torque * wx / h, -roll_torque * wy / h,
                                                                    -spin_torque * wz / s), False)

    def overlaps(state):
        """Penetration per pair (positive = in contact), order PAIRS[:9]."""
        out = np.zeros(9)
        if len(balls) == 2:
            out[0] = 2 * R - math.hypot(state[0, 0] - state[1, 0], state[0, 1] - state[1, 1])
        for i in range(len(balls)):
            x, y, dz = state[i, 0], state[i, 1], nose - state[i, 2]
            # Distance from the ball centre to each cushion's nose edge line.
            for k, gap in enumerate((hx - x, hx + x, hy - y, hy + y)):
                out[1 + 4 * i + k] = R - math.hypot(max(gap, 0.0), dz) if gap > -R else R
        return out

    def energy(state):
        return float(sum(0.5 * m * (s[3:6] @ s[3:6]) + 0.5 * inertia * (s[6:9] @ s[6:9]) + m * g * s[2] for s in state))

    n_records = steps // stride + 1
    states = np.empty((n_records, len(balls), 9))
    contact_depth = np.zeros((n_records - 1, 9))
    energies = np.empty(n_records)
    states[0] = read()
    energies[0] = energy(states[0])
    events, active = [], np.zeros(9, dtype=bool)
    worst = dict(penetration_m=0.0, height_deviation_m=0.0, vertical_speed_mps=0.0, energy_increase_j=0.0)
    previous_energy, previous_state = energies[0], states[0]
    reason, start = "complete", time.perf_counter()
    rest_z = R - contact["initial_sink_m"]
    state = states[0]
    for step in range(1, steps + 1):
        apply_resistance(state)
        system.DoStepDynamics(dt)
        state = read()
        if not np.isfinite(state).all():
            reason = "nonfinite_state"
            break
        depth = overlaps(state)
        now = depth > 0
        for index in np.flatnonzero(now != active):
            events.append({"pair": PAIRS[index], "kind": "start" if now[index] else "end", "time_s": step * dt,
                           "before": previous_state.tolist(), "after": state.tolist()})
        active = now
        record = (step - 1) // stride
        contact_depth[record] = np.maximum(contact_depth[record], depth)
        e = energy(state)
        # Penalty contact stores elastic energy while shapes overlap, so the
        # check compares energy only between contact-free steps.
        if not now.any():
            worst["energy_increase_j"] = max(worst["energy_increase_j"], e - previous_energy)
            previous_energy = e
        previous_state = state
        worst["height_deviation_m"] = max(worst["height_deviation_m"], float(np.abs(state[:, 2] - rest_z).max()))
        worst["vertical_speed_mps"] = max(worst["vertical_speed_mps"], float(np.abs(state[:, 5]).max()))
        worst["penetration_m"] = max(worst["penetration_m"], float(depth.max()))
        if worst["penetration_m"] > 10 * acceptance["max_penetration_m"]:
            reason = "excess_penetration"
            break
        if step % stride == 0:
            states[step // stride] = state
            energies[step // stride] = e
    else:
        step = steps
    recorded = step // stride + 1 if reason == "complete" else step // stride
    starts = [e for e in events if e["kind"] == "start"]
    hits = [e["pair"] for e in starts]
    first_ab = next((e["time_s"] for e in starts if e["pair"] == "AB"), None)
    if reason == "complete" and len(balls) == 2:
        before = states[: int(first_ab / sim["record_step_s"]), 1, 3:5] if first_ab else states[:, 1, 3:5]
        if first_ab is None:
            reason = "missed_ball_b"
        elif any(p.startswith("A_") for p in hits[: hits.index("AB")]):
            reason = "cushion_before_ab"
        elif np.abs(before).max() > acceptance["max_b_speed_before_ab_mps"]:
            reason = "b_moved_before_ab"
    for key, limit in [("penetration_m", "max_penetration_m"), ("height_deviation_m", "max_height_deviation_m"),
                       ("vertical_speed_mps", "max_vertical_speed_mps"), ("energy_increase_j", "max_energy_increase_j")]:
        if reason == "complete" and worst[key] > acceptance[limit]:
            reason = f"excess_{key}"
    metadata = {
        "vx_mps": vx, "vy_mps": vy, "accepted": reason == "complete", "termination": reason,
        "duration_s": (recorded - 1) * sim["record_step_s"], "physics_steps": step,
        "wall_time_s": time.perf_counter() - start, "worst": worst,
        "contact_sequence": hits, "ab_contacts": hits.count("AB"),
        "a_cushion_hits": sum(p.startswith("A_") for p in hits), "b_cushion_hits": sum(p.startswith("B_") for p in hits),
        "first_ab_time_s": first_ab,
        "events": [{k: e[k] for k in ("pair", "kind", "time_s")} for e in events],
        "impacts": [e for e in events if e["kind"] == "start"],
        "events_full": events,
    }
    arrays = {"states": states[:recorded], "contact_depth": contact_depth[: recorded - 1], "energy": energies[:recorded]}
    return arrays, metadata


def with_overrides(config: dict, overrides: dict) -> dict:
    """Copy of config with dotted-path overrides, e.g. {'contact.solver': 'APGD'}."""
    out = copy.deepcopy(config)
    for dotted, value in overrides.items():
        node = out
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return out
