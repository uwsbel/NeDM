"""Command schedules: what the robot is asked to do during an episode.

WHY FAMILIES AND NOT JUST RANDOM DIRECTIONS.

The command is not a model input. The NN-ROM maps (state, action) -> next state and never
sees the command; the command only decides WHICH (state, action) pairs get visited. So
this is a state-space excitation mechanism, and the question is what covers that space.

Families buy two things a random constant command cannot:

  TEMPORAL STRUCTURE. vel_step, stop_and_go and weave produce transients -- accelerations,
  gait transitions, starts and stops. A constant command, however randomly drawn, reaches
  steady state and stays there. Those transients are the dynamically richest rows in the
  corpus and a constant-command corpus contains none of them.

  AXIS-ISOLATED REGIMES. Pure pivot and pure lateral are behaviours a uniform draw over
  (vx, vy, wz) would essentially never produce, because a random triple is almost surely
  mixed.

But the structured families are mostly axis-aligned or simple pairs, so the INTERIOR of
the command space is barely covered: only `arc` mixes axes, and nothing drives all three
at once. That is the real gap, and it is why `random` and `random_walk` are here as
additional families rather than as a replacement.

RANGES ARE MEASURED FOR THIS POLICY. The previous tree's wide ranges were measured on a
different checkpoint, and its own notes record that the narrow ranges before them sampled
only the band where that policy worked worst -- 0.3 m/s achieving 0.01 of command -- which
made the plant look like it tracked at 63%. Inheriting a range is the same mistake as
inheriting a sign.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

FAMILY_PARAMS = {
    "constant":    ["vx"],
    "lateral":     ["vy"],
    "pivot":       ["wz"],
    "arc":         ["vx", "wz"],
    "vel_step":    ["vx0", "vx1", "t_switch"],
    "yaw_step":    ["vx", "wz1", "t_switch"],
    "weave":       ["vx", "wz_amp", "freq"],
    "stop_and_go": ["vx", "period"],
    # Interior coverage: all three axes at once, which no structured family provides.
    "random":      ["vx", "vy", "wz"],
    # Continuous direction variation, and transients everywhere rather than at scheduled
    # times. tau is the correlation time of the command itself, not of the action noise.
    "random_walk": ["vx", "vy", "wz", "tau"],
}

FAMILIES = tuple(FAMILY_PARAMS)


def family_seed(corpus: str, family: str, index: int) -> int:
    """crc32, NOT hash(). PYTHONHASHSEED salting once made a whole collection
    non-reproducible, because hash() of a str differs between interpreter runs."""
    return zlib.crc32(f"{corpus}|{family}|{index}".encode()) & 0x7FFFFFFF


def draw_params(family: str, rng: np.random.Generator, ranges: dict) -> dict:
    """One parameter draw. Draws in the order FAMILY_PARAMS lists, always the same count
    for a given family, so the stream position does not depend on the values drawn."""
    out = {}
    for name in FAMILY_PARAMS[family]:
        lo, hi = ranges[name]
        out[name] = float(rng.uniform(lo, hi))
    return out


def schedule(family: str, p: dict, duration_s: float, rng: np.random.Generator | None = None):
    """Return f(t) -> (vx, vy, wz)."""
    if family == "constant":
        return lambda t: (p["vx"], 0.0, 0.0)
    if family == "lateral":
        return lambda t: (0.0, p["vy"], 0.0)
    if family == "pivot":
        return lambda t: (0.0, 0.0, p["wz"])
    if family == "arc":
        return lambda t: (p["vx"], 0.0, p["wz"])
    if family == "vel_step":
        return lambda t: (p["vx0"] if t < p["t_switch"] else p["vx1"], 0.0, 0.0)
    if family == "yaw_step":
        return lambda t: (p["vx"], 0.0, 0.0 if t < p["t_switch"] else p["wz1"])
    if family == "weave":
        return lambda t: (p["vx"], 0.0,
                          p["wz_amp"] * math.sin(2 * math.pi * p["freq"] * t))
    if family == "stop_and_go":
        return lambda t: ((p["vx"] if (t % (2 * p["period"])) < p["period"] else 0.0),
                          0.0, 0.0)
    if family == "random":
        return lambda t: (p["vx"], p["vy"], p["wz"])
    if family == "random_walk":
        # Precomputed on a fixed grid so the schedule is a pure function of t and consumes
        # no randomness when evaluated -- the collector may call it any number of times.
        if rng is None:
            raise ValueError("random_walk needs an rng")
        dt = 0.05
        n = int(duration_s / dt) + 2
        tau = max(p["tau"], 2 * dt)
        alpha = math.exp(-dt / tau)
        amp = np.array([p["vx"], p["vy"], p["wz"]], dtype=float)
        x = np.zeros((n, 3))
        x[0] = rng.normal(size=3)
        for i in range(1, n):
            x[i] = alpha * x[i - 1] + math.sqrt(1 - alpha * alpha) * rng.normal(size=3)
        traj = np.clip(x, -2.0, 2.0) / 2.0 * amp

        def f(t):
            i = min(int(t / dt), n - 1)
            return tuple(traj[i])
        return f
    raise KeyError(family)


def stratified_families(n_episodes: int, rng: np.random.Generator,
                        families=FAMILIES) -> list[str]:
    """Balanced assignment, then shuffled. Drawing the family independently per episode
    leaves the balance to chance, and a corpus short one family is a corpus with a
    behaviour missing rather than one that is merely uneven."""
    base = list(families) * (n_episodes // len(families))
    base += list(rng.permutation(list(families))[: n_episodes % len(families)])
    return list(rng.permutation(base))
