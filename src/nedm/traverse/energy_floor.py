"""Route-geometry energy floor (WP5): a lower bound on imagined energy against the optimiser's curse.

Fitted by ``scripts/traverse_wp5_energy_floor.py`` on every route the tracker has driven in Chrono
(energy ~ length, length-weighted v^2, positive climb, peak speed, re-acceleration). Explains only
~half the Chrono variance, so it is NOT an estimator; ``floor = fit - k * sigma`` is the energy below
which no route of that geometry has been observed, and a sampled route whose imagined energy falls
under it is the dynamics model being exploited. Used as one more term in the pessimistic maximum.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nedm.traverse.terrain import TerrainMap


def route_features(waypoints, speeds, stations, tmap: TerrainMap) -> np.ndarray:
    w, v, s = np.asarray(waypoints, float), np.asarray(speeds, float), np.asarray(stations, float)
    L = s[-1]
    ds = np.diff(s)
    v_mid = 0.5 * (v[1:] + v[:-1])
    kin = float((v_mid ** 2 * ds).sum() / max(L, 1e-6))
    h = np.array([tmap.height(x, y) for x, y in w])
    climb = float(np.maximum(np.diff(h), 0).sum())
    accel = float(np.maximum(np.diff(v), 0).sum())
    return np.array([L, kin * L / 100.0, climb, float(v.max()) ** 2 / 10.0, accel ** 2 / 10.0, 1.0])


class EnergyFloor:
    def __init__(self, w, sigma_kj: float, k_sigma: float = 1.5):
        self.w, self.sigma, self.k = np.asarray(w, float), float(sigma_kj), float(k_sigma)

    @classmethod
    def load(cls, path: Path, k_sigma: float = 1.5) -> "EnergyFloor":
        d = json.loads(Path(path).read_text())
        return cls(d["w"], d["sigma_kj"], k_sigma)

    def fit_kj(self, plan, tmap: TerrainMap) -> float:
        return float(route_features(plan.waypoints, plan.speeds, plan.stations, tmap) @ self.w)

    def floor_kj(self, plan, tmap: TerrainMap) -> float:
        return max(0.0, self.fit_kj(plan, tmap) - self.k * self.sigma)
