#!/usr/bin/env python3
"""Map / arena consistency check (REVIEW_R1 amendment 7a, REVIEW_R2 amendment 6) for arena_gator_20260925.

A planner or dataset-builder call gets ONE map root and a set of cases. The check passes only if every case names the
same arena (case["arena"], relative to --source-root) and that arena's BMP sha256 equals the map root's
static_map_v1/observation.json "arena_bmp_sha256". Nothing in the older planners/builders compares these.

  python scripts/ag_map_check.py --map-root <root> --cases <case dir | case json> [...] [--source-root .]
  (exit 0 = consistent; prints the arena and hashes; exit 1 with the first mismatches otherwise)
"""
import argparse, hashlib, json, sys
from pathlib import Path


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def case_files(paths):
    out = []
    for p in map(Path, paths):
        out += sorted(q for q in p.glob('*.json') if q.name != 'cases.json') if p.is_dir() else [p]
    return out


def check(map_root, cases, source_root='.'):
    obs = json.load(open(Path(map_root) / 'static_map_v1' / 'observation.json'))
    arenas = {}
    for c in case_files(cases):
        arenas.setdefault(json.load(open(c))['arena'], []).append(c.name)
    problems = []
    if len(arenas) != 1:
        problems.append(f'{len(arenas)} arenas in one call: {sorted(arenas)}')
    res = dict(map_root=str(map_root), map_arena=obs.get('arena'), map_arena_bmp_sha256=obs.get('arena_bmp_sha256'),
               map_observation_sha256=obs.get('observation_sha256'), n_cases=sum(map(len, arenas.values())), arenas={})
    for arena, names in arenas.items():
        d = Path(source_root) / arena
        meta = json.load(open(d / 'arena_meta.json'))
        h = sha256(d / meta['bmp'])
        res['arenas'][arena] = dict(n=len(names), bmp_sha256=h, match=h == obs.get('arena_bmp_sha256'))
        if h != obs.get('arena_bmp_sha256'):
            problems.append(f'{arena} BMP {h[:16]} != map {str(obs.get("arena_bmp_sha256"))[:16]} ({len(names)} cases, e.g. {names[0]})')
    res['ok'] = not problems
    res['problems'] = problems
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--map-root', required=True)
    ap.add_argument('--cases', nargs='+', required=True)
    ap.add_argument('--source-root', default='.')
    a = ap.parse_args(argv)
    r = check(a.map_root, a.cases, a.source_root)
    print(json.dumps(r, indent=1))
    sys.exit(0 if r['ok'] else 1)


if __name__ == '__main__':
    main()
