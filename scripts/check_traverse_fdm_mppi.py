#!/usr/bin/env python3
"""CPU-only checks for constrained reference MPPI; no model training."""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nedm.traverse.fdm_mppi import MPPIConfig, ReferenceMPPI, deform_reference, validate_reference


def main():
    x = np.linspace(-12., 18., 61)
    route = {"waypoints": np.c_[x, np.zeros_like(x)], "speeds": np.full_like(x, 4.)}
    anchor = np.array([-12., 0., 0.])
    cfg = MPPIConfig(samples=64, iterations=3, lateral_sigma_m=0.0, max_lateral_m=0.0)
    candidate = deform_reference(route, np.zeros(6), anchor, cfg)
    assert np.array_equal(candidate["waypoints"], route["waypoints"])
    assert np.array_equal(candidate["speeds"], route["speeds"])
    assert validate_reference(candidate, [], cfg)["valid"]
    assert "obstacle" in validate_reference(candidate, [(0., 0., 0.5)], cfg)["reasons"]
    assert "arena" in validate_reference(candidate, [], MPPIConfig(arena_half_extent_m=10.))["reasons"]

    def score(routes):
        # Synthetic preference for a modest faster command; this is a numerical
        # optimizer test, not a dynamics model or navigation result.
        return np.array([float(((r["speeds"] - 4.4) ** 2).mean()) for r in routes])

    result = ReferenceMPPI(cfg, seed=11).optimize(route, anchor, [], score)
    assert not result["abstained"]
    assert result["cost"] <= score([candidate])[0] + 1e-12
    assert validate_reference(result["route"], [], cfg)["valid"]
    second = ReferenceMPPI(cfg, seed=11).optimize(route, anchor, [], score)
    assert result["parameters"] == second["parameters"]
    # Entire route family blocked: explicitly abstain without evaluating an
    # unchecked path. An optimizer mean must never bypass geometry validation.
    blocked = ReferenceMPPI(cfg, seed=11).optimize(route, anchor, [(0., 0., 5.)], score)
    assert blocked["abstained"] and blocked["model_evaluations"] == 0
    rejected = ReferenceMPPI(cfg).optimize(route, anchor, [], lambda rs: np.full(len(rs), np.inf))
    assert rejected["abstained"]
    try:
        ReferenceMPPI(cfg).optimize(route, anchor, [], lambda rs: np.full(len(rs), np.nan))
    except ValueError:
        pass
    else:
        raise AssertionError("nonfinite scorer output was accepted")
    # Offset endpoints preserve reference start/end and requested zero speeds.
    parked = dict(route, speeds=np.r_[np.full(len(x)-1, 4.), 0.])
    modified = deform_reference(parked, np.ones(6) * .2, anchor, cfg)
    assert np.allclose(modified["waypoints"][[0, -1]], route["waypoints"][[0, -1]])
    assert modified["speeds"][-1] == 0.
    print(json.dumps({"passed": True, "checks": ["geometry", "optimizer_improvement", "determinism",
          "blocked_abstention", "scorer_abstention", "nan_rejection", "endpoint_parking"],
          "synthetic_best_cost": result["cost"], "model_evaluations": result["model_evaluations"]}, indent=2))


if __name__ == "__main__":
    main()
