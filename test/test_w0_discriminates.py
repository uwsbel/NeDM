"""W0 must be able to return the null. This test proves it can.

The repo's own rule: before trusting a check, ask whether the data could have produced
the opposite answer (docs/state/lessons/experiment-design.md). W0 decides whether an
event-indexed map is viable on soil, so a W0 that says "predictable" on everything would
license W4 on no evidence.

Two synthetic worlds, identical except for one thing:

  predictable    the section-to-section map is a function of the section state
  hidden field   a SPATIALLY VARYING soil value perturbs each footfall, and the robot
                 walks into new ground each step, so x_k cannot carry it

A uniform per-episode soil property is deliberately NOT the negative case: the state
accumulates it and it stays predictable. That was measured while building this test, and
it is the reason the threat model is spatial variation rather than terrain softness.

The test also pins the reason the decision metric is the INCREMENT. Level-R^2 scores ~0.9
in BOTH worlds because x_k is most of x_{k+1}; it cannot separate them and must never be
read as the verdict.
"""
from __future__ import annotations
import csv, os, subprocess, sys, json, tempfile
import numpy as np
import pytest

sys.path.insert(0, "src")
from nedm.quadruped.dataset import csv_field_names, LEG_ORDER

SCRIPT = "scripts/evaluation/w0_poincare_decidability.py"


def _write_world(root: str, field_sd: float, seed: int, n_ep: int = 80, T: int = 640, dt: float = 0.02):
    cols = csv_field_names()
    jc = sorted(c for c in cols
                if c.startswith("joint_") and (c.endswith("_pos_rad") or c.endswith("_vel_radps")))
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(seed)
    for e in range(n_ep):
        field = rng.normal(0, field_sd, 64)
        zk, vzk = 0.32 + rng.normal(0, 0.04), rng.normal(0, 0.15)
        jbase = rng.normal(0, 0.25, len(jc))
        rows = []
        for i in range(T):
            ph, x = i / 40.0, 0.5 * i * dt
            if i % 40 == 0 and i > 0:
                soil = field[int(x / 0.4) % 64]
                zk, vzk = (0.90 * zk + 0.10 * 0.32 + 0.05 * vzk + rng.normal(0, 0.002) + soil * 0.03,
                           0.85 * vzk - 0.5 * (zk - 0.32) + rng.normal(0, 0.01) + soil * 0.10)
                jbase = 0.92 * jbase + rng.normal(0, 0.02, len(jc)) + soil * 0.05
            r = {c: 0.0 for c in cols}
            r["episode_id"], r["time_s"] = e, i * dt
            r["scenario_name"] = r["scenario_family"] = "synth"
            r["split"] = "train"
            for k, leg in enumerate(LEG_ORDER):
                off = 0.0 if leg in ("fl", "rr") else 0.5
                r[f"foot_{leg}_force_fz_n"] = 100.0 * (np.sin(2 * np.pi * (ph - off)) > 0)
                for ax, val in zip("xyz", (0.1 * k, 0.05 * k, 0.0)):
                    r[f"foot_{leg}_pos_{ax}_m"] = val
            r["pos_z_m"] = zk + 0.005 * np.sin(2 * np.pi * ph)
            r["vel_world_z_mps"], r["pos_x_m"] = vzk, x
            r["vel_world_x_mps"] = 0.5 + 0.2 * vzk
            r["roll_rad"], r["pitch_rad"] = 0.05 * vzk, 0.03 * zk
            for n, c in enumerate(jc):
                r[c] = jbase[n] + 0.3 * np.sin(2 * np.pi * ph + n)
            rows.append(r)
        with open(f"{root}/ep{e:03d}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    return jc


def _run(glob_pat: str, label: str, out_json: str) -> dict:
    subprocess.run([sys.executable, SCRIPT, "--glob", glob_pat, "--label", label,
                    "--summary-json", out_json], check=True, capture_output=True)
    return json.load(open(out_json))


@pytest.mark.slow
def test_increment_r2_separates_predictable_from_hidden_field():
    with tempfile.TemporaryDirectory() as td:
        _write_world(f"{td}/pred", field_sd=0.0, seed=1)
        _write_world(f"{td}/hidden", field_sd=1.0, seed=2)
        pred = _run(f"{td}/pred/*.csv", "pred", f"{td}/p.json")
        hid = _run(f"{td}/hidden/*.csv", "hidden", f"{td}/h.json")

        pz = pred["components"].index("pz_rel")
        r_pred = pred["r2_increment"]["ridge"][pz]
        r_hid = hid["r2_increment"]["ridge"][pz]

        # THE POINT: the increment metric must separate the two worlds.
        assert r_pred > 0.4, f"predictable world should be learnable, got {r_pred}"
        assert r_hid < 0.3, f"hidden-field world should not be, got {r_hid}"
        assert r_pred > 2 * r_hid, f"separation too weak: {r_pred} vs {r_hid}"

        # AND: the level metric must NOT separate them, which is why it is not the verdict.
        lp = pred["r2_level_median"]["ridge(level)"]
        lh = hid["r2_level_median"]["ridge(level)"]
        assert lp > 0.8 and lh > 0.8, (lp, lh)
        assert abs(lp - lh) < 0.2, ("level-R^2 accidentally separates; the persistence "
                                    "argument in the docstring needs revisiting", lp, lh)
