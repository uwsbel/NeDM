"""Soil-surface sinkage measured from SPH particle z, independent of the renderer.

This answers "is there a depression under the foot" without depending on whether
the sprite path can draw one. Lifted verbatim out of the sim loop, where it was
inline.
"""

from __future__ import annotations

import math

import numpy as np

from .constants import EJECTA_BAND, FOOT_BODIES, SOIL_CTRL_XY, SOIL_PROBE_R


def bind_probe(terrain):
    """The terrain's SPH system if it can report particle positions, else None."""
    if terrain is None:
        return None
    getter = getattr(terrain, "GetFluidSystemSPH", None)
    if getter is None:
        return None
    try:
        cand = getter()
    except Exception:  # noqa: BLE001
        return None
    return cand if hasattr(cand, "GetParticlePositionsNumpy") else None


def sample(sph_probe, robot):
    """(per-foot surface z95, control-patch z95). NaN where unavailable.

    95th percentile of particles within SOIL_PROBE_R of each foot's XY, against
    the same statistic for a fixed undisturbed patch. The difference is the local
    surface displacement. A STATIC proxy foot gave only 2 mm; a walking foot
    lands with well above its static share of body weight, so the two are not
    interchangeable and the walking case had to be measured separately.
    """
    soil_z = [float("nan")] * len(FOOT_BODIES)
    soil_ctrl = float("nan")
    if sph_probe is None:
        return soil_z, soil_ctrl
    try:
        P = np.asarray(sph_probe.GetParticlePositionsNumpy())
        # Control patch first: it is off the robot's path, so it has no ejecta
        # and its z95 is a clean surface estimate. Measured sd 0.00000 over a
        # run, which is what makes the filtered foot values trustworthy rather
        # than merely adjusted until they looked reasonable.
        cx, cy = SOIL_CTRL_XY
        d = np.hypot(P[:, 0] - cx, P[:, 1] - cy)
        sel = P[d < SOIL_PROBE_R]
        if len(sel):
            soil_ctrl = float(np.percentile(sel[:, 2], 95))
        # Foot patches: a bare z95 is NOT a surface estimate here, because a
        # footfall throws particles into the air and the top percentile then
        # tracks ejecta rather than the bed. A first pass reported 0.48 m under a
        # foot over a bed whose top is 0.20 m -- 28 cm of "surface". It announced
        # itself only because 28 cm is absurd; a 3 mm contamination would have
        # been reported as the finding. An estimator validated in one regime is
        # not validated in another.
        if not math.isnan(soil_ctrl):
            lid = soil_ctrl + EJECTA_BAND
            for k, n in enumerate(FOOT_BODIES):
                b = robot.body(n)
                if b is None:
                    continue
                fp = b.GetPos()
                d = np.hypot(P[:, 0] - fp.x, P[:, 1] - fp.y)
                sel = P[(d < SOIL_PROBE_R) & (P[:, 2] < lid)]
                if len(sel):
                    soil_z[k] = float(np.percentile(sel[:, 2], 95))
    except Exception:  # noqa: BLE001 - a diagnostic must not break the run
        pass
    return soil_z, soil_ctrl
