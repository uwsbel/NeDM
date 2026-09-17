"""Five-goal missions for the continuous multi-goal test (gen_v1 PLAN.md section B).

Geometry (PLAN.md amendment 1): legs 20-35 m; heading change at each goal <= 110 deg; all six points on slope < 7 deg
and within +-32 m;
at least 3 of 5 leg chords cross a hill (> median ground + 2 m) or crater (< median ground - 0.7 m) zone; the start
passes the frozen generator's launch footprint gate; every leg's straight route validates from a pose whose heading
is the previous leg's direction.
"""
import argparse, hashlib, json, math, sys
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
    """For chords from point a to each row of B: (crosses hill zone, crosses crater zone)."""
    t = np.linspace(0, 1, 60)[None, :, None]
    pts = a[None, None, :] * (1 - t) + B[:, None, :] * t
    h = tmap.height(pts[..., 0].ravel(), pts[..., 1].ravel()).reshape(len(B), 60)
    return (h > ground + 2.0).any(1), (h < ground - 0.7).any(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arena', type=Path, required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--n', type=int, required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    arena = a.arena.resolve(); tmap = TerrainMap.from_dir(arena)
    grid = np.arange(-38., 38.01, .5); gx, gy = np.meshgrid(grid, grid)
    ground = float(np.median(tmap.height(gx.ravel(), gy.ravel())))
    rng = np.random.default_rng(a.seed)
    coords = np.arange(-32., 32.01, 1.); cx, cy = np.meshgrid(coords, coords)
    flat = np.c_[cx.ravel(), cy.ravel()]; flat = flat[tmap.slope(flat[:, 0], flat[:, 1]) < math.tan(math.radians(7))]
    missions, attempts = [], 0
    a.out.mkdir(parents=True, exist_ok=True)
    while len(missions) < a.n and attempts < 60000:
        attempts += 1
        p0 = flat[int(rng.integers(len(flat)))] + rng.uniform(-.35, .35, 2)
        pts, legs, heading, ok = [p0], [], None, True
        for k in range(5):
            prev = pts[-1]
            d = np.linalg.norm(flat - prev[None], axis=1)
            cand = flat[(d >= 20) & (d <= 35)]
            if heading is not None and len(cand):
                bearing = np.arctan2(cand[:, 1] - prev[1], cand[:, 0] - prev[0])
                turn = np.abs((bearing - heading + np.pi) % (2 * np.pi) - np.pi)
                cand = cand[turn <= math.radians(110)]
            if not len(cand):
                ok = False; break
            hills, craters = hazard_flags(tmap, prev, cand, ground)
            haz = hills | craters
            # prefer hazard-crossing legs 70% of the time, so >= 3 of 5 is reachable without rejection-sampling
            pref = np.where(haz)[0] if (haz.any() and rng.random() < .7) else np.arange(len(cand))
            order = rng.permutation(pref)
            placed = False
            for j in order[:25]:
                q = cand[j] + rng.uniform(-.35, .35, 2)
                th = math.atan2(q[1] - prev[1], q[0] - prev[0])
                pose = [prev[0], prev[1], th if heading is None else heading]
                if not validate_reference(P.base_route(pose, q), [], CFG, np.asarray(pose))['valid']:
                    continue
                hill, crater = bool(hills[j]), bool(craters[j])
                turn = 0.0 if heading is None else math.degrees((th - heading + math.pi) % (2 * math.pi) - math.pi)
                legs.append(dict(length_m=float(np.linalg.norm(q - prev)), turn_deg=turn, hill=hill, crater=crater))
                pts.append(q); heading = th; placed = True
                break
            if not placed:
                ok = False; break
        if not ok or sum(L['hill'] or L['crater'] for L in legs) < 3:
            continue
        yaw0 = math.atan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
        if not start_ok(footprint(tmap, pts[0], yaw0)):
            continue
        mid = f'{a.tag}_mission_{len(missions):03d}'
        m = dict(id=mid, arena=str(arena.relative_to(ROOT)), goal_radius_m=2.5,
                 goals=[[float(x), float(y)] for x, y in pts[1:]],
                 layout=dict(episode_id=mid, seed=int(a.seed + len(missions)), assets=[], house_xy=[float(pts[1][0]), float(pts[1][1])],
                             house_yaw=yaw0, start_xy=[float(pts[0][0]), float(pts[0][1])], start_yaw=yaw0),
                 legs=legs, ground_m=ground, hazard_legs=int(sum(L['hill'] or L['crater'] for L in legs)))
        json.dump(m, open(a.out / f'{mid}.json', 'w'), indent=1)
        missions.append(m)
    print(f'{a.tag}: {len(missions)} missions after {attempts} attempts; mean hazard legs '
          f'{np.mean([m["hazard_legs"] for m in missions]):.2f}; mean length {np.mean([sum(L["length_m"] for L in m["legs"]) for m in missions]):.0f} m')


if __name__ == '__main__':
    main()
