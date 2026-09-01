"""Per-episode randomized asset layout for the traversal arena (plan §3.2).

The arena heightmap is FIXED; each episode samples rocks, trees, the goal house,
and the vehicle start pose, subject to placement constraints, then proves the
episode feasible with the privileged oracle (``plan_to_ring``). The sampled
layout is logged as a JSON manifest — the same manifest later drives costmap
label rasterization, so it must stay the single source of truth for obstacle
geometry.

Collision semantics: rocks, tree TRUNKS, and the house are collidable; tree
canopies are visual-only (a canopy overhangs well above the roofline, and the
planner should not treat it as ground footprint).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nedm.traverse.oracle import Obstacle, PlanCandidate, PlannerParams, plan_to_ring
from nedm.traverse.terrain import TerrainMap

FLAT_PLACEMENT_SLOPE = math.tan(math.radians(8.0))


@dataclass
class LayoutParams:
    n_rocks_range: tuple[int, int] = (6, 10)
    rock_size_range_m: tuple[float, float] = (0.8, 2.0)  # box edge length
    n_trees_range: tuple[int, int] = (8, 14)
    trunk_radius_range_m: tuple[float, float] = (0.22, 0.40)
    trunk_height_range_m: tuple[float, float] = (2.8, 4.5)
    canopy_radius_range_m: tuple[float, float] = (1.3, 2.2)
    house_size_m: tuple[float, float, float] = (5.0, 4.0, 3.0)  # L x W x wall height
    keep_within_m: float = 36.0
    house_keep_within_m: float = 29.0  # most of the 7 m approach ring stays in bounds
    start_house_dist_range_m: tuple[float, float] = (42.0, 68.0)
    asset_min_separation_m: float = 3.5
    start_clear_radius_m: float = 6.0
    house_clear_radius_m: float = 3.0  # beyond footprint
    placement_slope_max: float = FLAT_PLACEMENT_SLOPE
    max_episode_tries: int = 25
    max_point_tries: int = 400


@dataclass
class Asset:
    kind: str  # "rock" | "tree" | "house"
    x_m: float
    y_m: float
    yaw_rad: float
    footprint_radius_m: float  # collidable ground footprint for the planner
    dims: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "yaw_rad": self.yaw_rad,
            "footprint_radius_m": self.footprint_radius_m,
            "dims": self.dims,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "Asset":
        return cls(
            kind=payload["kind"],
            x_m=float(payload["x_m"]),
            y_m=float(payload["y_m"]),
            yaw_rad=float(payload["yaw_rad"]),
            footprint_radius_m=float(payload["footprint_radius_m"]),
            dims=dict(payload.get("dims", {})),
        )


@dataclass
class EpisodeLayout:
    episode_id: str
    seed: int
    assets: list[Asset]
    house_xy: tuple[float, float]
    house_yaw: float
    start_xy: tuple[float, float]
    start_yaw: float

    def obstacles(self) -> list[Obstacle]:
        """(x, y, footprint radius) for every collidable asset, house included."""
        return [(a.x_m, a.y_m, a.footprint_radius_m) for a in self.assets]

    def to_json(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "assets": [a.to_json() for a in self.assets],
            "house_xy": list(self.house_xy),
            "house_yaw": self.house_yaw,
            "start_xy": list(self.start_xy),
            "start_yaw": self.start_yaw,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "EpisodeLayout":
        return cls(
            episode_id=payload["episode_id"],
            seed=int(payload["seed"]),
            assets=[Asset.from_json(a) for a in payload["assets"]],
            house_xy=tuple(payload["house_xy"]),
            house_yaw=float(payload["house_yaw"]),
            start_xy=tuple(payload["start_xy"]),
            start_yaw=float(payload["start_yaw"]),
        )

    def save(self, path: Path) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_json(), handle, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EpisodeLayout":
        with path.open("r", encoding="utf-8") as handle:
            return cls.from_json(json.load(handle))


def _flat_point(
    tmap: TerrainMap,
    rng: np.random.Generator,
    bound: float,
    slope_max: float,
    tries: int,
    accept: Any = None,
) -> tuple[float, float] | None:
    """A point with low local slope (center + 4 offsets), optionally filtered."""
    for _ in range(tries):
        x = float(rng.uniform(-bound, bound))
        y = float(rng.uniform(-bound, bound))
        xs = np.array([x, x + 1.5, x - 1.5, x, x])
        ys = np.array([y, y, y, y + 1.5, y - 1.5])
        if float(tmap.slope(xs, ys).max()) > slope_max:
            continue
        if accept is not None and not accept(x, y):
            continue
        return x, y
    return None


def _clear_of(points: list[tuple[float, float, float]], x: float, y: float, margin: float) -> bool:
    return all(math.hypot(x - px, y - py) >= pr + margin for px, py, pr in points)


def sample_layout(tmap: TerrainMap, episode_id: str, seed: int, params: LayoutParams) -> EpisodeLayout | None:
    """One layout draw honoring placement constraints; no feasibility proof yet."""
    rng = np.random.default_rng(seed)
    hl, hw, _ = params.house_size_m
    house_radius = 0.5 * math.hypot(hl, hw) + 0.3

    house = _flat_point(
        tmap, rng, params.house_keep_within_m, params.placement_slope_max, params.max_point_tries
    )
    if house is None:
        return None

    def start_ok(x: float, y: float) -> bool:
        d = math.hypot(x - house[0], y - house[1])
        return params.start_house_dist_range_m[0] <= d <= params.start_house_dist_range_m[1]

    start = _flat_point(
        tmap, rng, params.keep_within_m - 2.0, params.placement_slope_max,
        params.max_point_tries, accept=start_ok,
    )
    if start is None:
        return None

    house_yaw = float(rng.choice([0.0, math.pi / 2.0]) + rng.uniform(-0.3, 0.3))
    assets: list[Asset] = [
        Asset(
            kind="house",
            x_m=house[0],
            y_m=house[1],
            yaw_rad=house_yaw,
            footprint_radius_m=house_radius,
            dims={"length_m": hl, "width_m": hw, "wall_height_m": params.house_size_m[2]},
        )
    ]
    # keep-out list as (x, y, radius): existing assets + start + house apron.
    placed: list[tuple[float, float, float]] = [
        (house[0], house[1], house_radius + params.house_clear_radius_m),
        (start[0], start[1], params.start_clear_radius_m),
    ]

    n_rocks = int(rng.integers(*params.n_rocks_range, endpoint=True))
    n_trees = int(rng.integers(*params.n_trees_range, endpoint=True))
    for kind, count in (("rock", n_rocks), ("tree", n_trees)):
        for _ in range(count):
            for _ in range(params.max_point_tries):
                x = float(rng.uniform(-params.keep_within_m, params.keep_within_m))
                y = float(rng.uniform(-params.keep_within_m, params.keep_within_m))
                if kind == "rock":
                    edge = float(rng.uniform(*params.rock_size_range_m))
                    radius = 0.5 * math.sqrt(2.0) * edge
                    dims = {"edge_m": edge, "height_m": float(rng.uniform(0.6, 1.4) * edge)}
                else:
                    trunk_r = float(rng.uniform(*params.trunk_radius_range_m))
                    radius = trunk_r + 0.4
                    dims = {
                        "trunk_radius_m": trunk_r,
                        "trunk_height_m": float(rng.uniform(*params.trunk_height_range_m)),
                        "canopy_radius_m": float(rng.uniform(*params.canopy_radius_range_m)),
                    }
                if not _clear_of(placed, x, y, radius + params.asset_min_separation_m):
                    continue
                assets.append(
                    Asset(kind=kind, x_m=x, y_m=y, yaw_rad=float(rng.uniform(0, 2 * math.pi)),
                          footprint_radius_m=radius, dims=dims)
                )
                placed.append((x, y, radius))
                break

    start_yaw = math.atan2(house[1] - start[1], house[0] - start[0])
    return EpisodeLayout(
        episode_id=episode_id, seed=seed, assets=assets,
        house_xy=house, house_yaw=house_yaw, start_xy=start, start_yaw=start_yaw,
    )


def sample_episode(
    tmap: TerrainMap,
    episode_id: str,
    seed: int,
    params: LayoutParams | None = None,
    planner_params: PlannerParams | None = None,
) -> tuple[EpisodeLayout, PlanCandidate]:
    """Layout draw + oracle feasibility proof; resamples until a plan exists."""
    params = params or LayoutParams()
    planner_params = planner_params or PlannerParams()
    for attempt in range(params.max_episode_tries):
        layout = sample_layout(tmap, episode_id, seed + 1000 * attempt, params)
        if layout is None:
            continue
        plan = plan_to_ring(tmap, layout.obstacles(), layout.start_xy, layout.house_xy, planner_params)
        if plan is None:
            continue
        layout.start_yaw = float(plan.headings[0])
        plan.meta["layout_seed"] = layout.seed
        plan.meta["layout_attempt"] = attempt
        return layout, plan
    raise RuntimeError(f"no feasible layout after {params.max_episode_tries} tries (seed {seed})")
