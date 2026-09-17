"""Training tensors for the routes collected on the new arenas (same keys as night2_v1/station_ds_all.npz, plus arena).

Uses f104_n2_dataset.one() unchanged (labels with the corrected throttle column, event station for the survival
loss, 5-channel corridor), with the corridor sampler pointed at each arena's heightmap (gen_planner.set_map).
  python scripts/gen_build_dataset.py --runs gen_v1/data/runs [gen_v1/test/runs ...] --out gen_v1/station_ds_gen_v1.npz
"""
import argparse, glob, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _init(arena):
    import gen_planner as P
    P.set_map(arena)


def _one(item):
    import f104_n2_dataset as DS
    d, source = item
    r = DS.one((d, None, source))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=14)
    a = ap.parse_args()
    by_arena = {}
    for root in a.runs:
        source = 'designed' if '/data/' in root else 'gen_test_arm'
        for d in sorted(glob.glob(root + '/*')):
            name = os.path.basename(d); arena = name.split('_')[0]
            if not os.path.exists(d + '/outcome.json'):
                continue
            by_arena.setdefault(arena, []).append((d, source))
    rows = []
    for arena, items in sorted(by_arena.items()):
        adir = 'assets/traverse/' + ('arena_f104_50h_v1' if arena == 'f104' else f'arena_{arena}')
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(adir,)) as ex:
            got = [r for r in ex.map(_one, items, chunksize=16) if r is not None]
        for r in got:
            r['arena'] = arena
        rows += got
        print(arena, len(got), 'of', len(items), 'unsafe %.3f' % np.mean([r['unsafe'] for r in got]), flush=True)
    keys = ['X', 'ctx', 'id', 'group', 'split', 'source', 'profile', 'fail', 'unsafe', 'status', 'event_idx', 'route_len', 'min_vx', 'back_s', 'arena']
    out = {k: np.stack([r[k] for r in rows]) if k in ('X', 'ctx') else np.array([r[k] for r in rows]) for k in keys}
    out['X'] = out['X'].astype(np.float16)
    np.savez(a.out, **out)
    print('wrote', a.out, len(rows), 'routes')


if __name__ == '__main__':
    main()
