"""Missions for continuous waypoint navigation: 5-8 waypoints, 150-250 m of travel, in an 80 m arena.

Same gates as the five-goal missions (scripts/gen_missions.py): every point on slope < 7 deg and inside the arena,
legs long enough to be worth planning, turn limited at each waypoint, at least half the legs crossing a hill or a
crater, the start passing the collector's launch-footprint gate, and every leg's straight route validating from a
pose whose heading is the previous leg's direction. Different: 5-8 waypoints instead of exactly 5, a total travel
window, and a slightly wider turn limit (a chain that long inside 80 m cannot stay under 110 deg).
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src')); sys.path.insert(0, str(ROOT / 'scripts'))
from nedm.traverse.terrain import TerrainMap
from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference
from generate_traverse_f104_collection import footprint, start_ok
import gen_planner as P

CFG = MPPIConfig(max_speed_mps=6., min_speed_mps=0.0, max_curvature_inv_m=.125, arena_half_extent_m=40.)


def hazard_flags(tmap, a, B, ground):
    t = np.linspace(0, 1, 60)[None, :, None]
    pts = a[None, None, :] * (1 - t) + B[:, None, :] * t
    h = tmap.height(pts[..., 0].ravel(), pts[..., 1].ravel()).reshape(len(B), 60)
    return (h > ground + 2.0).any(1), (h < ground - 0.7).any(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arena', type=Path, required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--n', type=int, required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--goals-min', type=int, default=5); ap.add_argument('--goals-max', type=int, default=8)
    ap.add_argument('--leg-min', type=float, default=22.); ap.add_argument('--leg-max', type=float, default=35.)
    ap.add_argument('--turn-max-deg', type=float, default=120.)
    ap.add_argument('--total-min', type=float, default=150.); ap.add_argument('--total-max', type=float, default=250.)
    a = ap.parse_args()
    arena = a.arena.resolve(); tmap = TerrainMap.from_dir(arena)
    grid = np.arange(-38., 38.01, .5); gx, gy = np.meshgrid(grid, grid)
    ground = float(np.median(tmap.height(gx.ravel(), gy.ravel())))
    rng = np.random.default_rng(a.seed)
    coords = np.arange(-33., 33.01, 1.); cx, cy = np.meshgrid(coords, coords)
    flat = np.c_[cx.ravel(), cy.ravel()]
    flat = flat[tmap.slope(flat[:, 0], flat[:, 1]) < math.tan(math.radians(7))]
    missions, attempts = [], 0
    a.out.mkdir(parents=True, exist_ok=True)
    while len(missions) < a.n and attempts < 200000:
        attempts += 1
        n_goals = int(rng.integers(a.goals_min, a.goals_max + 1))
        p0 = flat[int(rng.integers(len(flat)))] + rng.uniform(-.35, .35, 2)
        pts, legs, heading, ok = [p0], [], None, True
        for _ in range(n_goals):
            prev = pts[-1]
            d = np.linalg.norm(flat - prev[None], axis=1)
            cand = flat[(d >= a.leg_min) & (d <= a.leg_max)]
            if heading is not None and len(cand):
                bearing = np.arctan2(cand[:, 1] - prev[1], cand[:, 0] - prev[0])
                turn = np.abs((bearing - heading + np.pi) % (2 * np.pi) - np.pi)
                cand = cand[turn <= math.radians(a.turn_max_deg)]
            if len(cand) == 0:
                ok = False; break
            # keep away from points already visited, so the chain does not fold back on itself
            if len(pts) > 1:
                prior = np.asarray(pts[:-1])
                keep = np.linalg.norm(cand[:, None, :] - prior[None], axis=-1).min(1) > 12.
                if keep.any():
                    cand = cand[keep]
            hills, craters = hazard_flags(tmap, prev, cand, ground)
            haz = hills | craters
            pref = np.where(haz)[0] if (haz.any() and rng.random() < .7) else np.arange(len(cand))
            order = rng.permutation(pref)
            placed = False
            for j in order[:25]:
                q = cand[j] + rng.uniform(-.35, .35, 2)
                th = math.atan2(q[1] - prev[1], q[0] - prev[0])
                pose = [prev[0], prev[1], th if heading is None else heading]
                if not validate_reference(P.base_route(pose, q), [], CFG, np.asarray(pose))['valid']:
                    continue
                turn = 0.0 if heading is None else math.degrees((th - heading + math.pi) % (2 * math.pi) - math.pi)
                legs.append(dict(length_m=float(np.linalg.norm(q - prev)), turn_deg=turn,
                                 hill=bool(hills[j]), crater=bool(craters[j])))
                pts.append(q); heading = th; placed = True
                break
            if not placed:
                ok = False; break
        if not ok:
            continue
        total = sum(L['length_m'] for L in legs)
        if not (a.total_min <= total <= a.total_max):
            continue
        if sum(L['hill'] or L['crater'] for L in legs) < max(3, n_goals // 2):
            continue
        yaw0 = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
        if not start_ok(footprint(tmap, pts[0], yaw0)):
            continue
        mid = f'{a.tag}_nav_{len(missions):03d}'
        m = dict(id=mid, arena=str(arena.relative_to(ROOT)), goal_radius_m=2.5,
                 goals=[[float(x), float(y)] for x, y in pts[1:]],
                 layout=dict(episode_id=mid, seed=int(a.seed + len(missions)), assets=[],
                             house_xy=[float(pts[1][0]), float(pts[1][1])], house_yaw=yaw0,
                             start_xy=[float(pts[0][0]), float(pts[0][1])], start_yaw=yaw0),
                 legs=legs, ground_m=ground, n_goals=n_goals, total_length_m=float(total),
                 hazard_legs=int(sum(L['hill'] or L['crater'] for L in legs)),
                 max_turn_deg=float(max(abs(L['turn_deg']) for L in legs)))
        json.dump(m, open(a.out / f'{mid}.json', 'w'), indent=1)
        missions.append(m)
    if missions:
        print(f'{a.tag}: {len(missions)}/{a.n} after {attempts} attempts; goals '
              f'{np.mean([m["n_goals"] for m in missions]):.1f}; total '
              f'{np.mean([m["total_length_m"] for m in missions]):.0f} m '
              f'[{min(m["total_length_m"] for m in missions):.0f}, {max(m["total_length_m"] for m in missions):.0f}]; '
              f'hazard legs {np.mean([m["hazard_legs"] for m in missions]):.1f}')
    else:
        print(f'{a.tag}: 0/{a.n} after {attempts} attempts')


if __name__ == '__main__':
    main()
