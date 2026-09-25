#!/usr/bin/env python3
"""Lock the arena-study case sets (arena_gator E1): one sha256 over every case and route file of a set.

Same scheme as ga_suite.write_suite_lock / PICKS_LOCKED: line 1 = sha256 over (relative path + sha256 digest of the
content) of every file, sorted by path; then one `<sha256>  <path>` line per file, paths relative to the repo root, so
from the repo root `tail -n +2 <lock> | sha256sum -c --quiet` re-checks every file. A small manifest json per set
(groups, arena, strata, splits, map, seed, hash) is written next to the lock and hashed into ALL_LOCKED.sha256.

  --set NAME=CASES_DIR[:ONPOLICY_DIR]   (repeatable) CASES_DIR holds cases.json, <group>.json and routes/<group>/*.json;
                                        ONPOLICY_DIR (training pools) holds routes.json and routes/<group>/op_*.json
  --kind suite|train                    suite -> <out>/<NAME>.SUITE_LOCKED.sha256, train -> <out>/<NAME>.CASES_LOCKED.sha256
  --check                               only re-verify the existing locks in --out (nothing is written)
"""
from __future__ import annotations

import argparse, fnmatch, hashlib, json, subprocess, sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT))


def lock_files(files):
    files = sorted(files, key=rel)
    h = hashlib.sha256(); lines = []
    for f in files:
        d = sha(f); h.update(rel(f).encode()); h.update(bytes.fromhex(d)); lines.append(f'{d}  {rel(f)}')
    return h.hexdigest(), lines


def blacklist_hits(groups):
    import ci_train, ga_build_mixed, ci_a5data
    out = {}
    for name, pats in (('ci_train.SUITE_GROUPS', ci_train.SUITE_GROUPS), ('ga_build_mixed.BLACKLIST', ga_build_mixed.BLACKLIST),
                       ('ci_a5data.SUITE_PATTERNS', ci_a5data.SUITE_PATTERNS)):
        out[name] = sum(any(fnmatch.fnmatch(g, p) for p in pats) for g in groups)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--kind', choices=['suite', 'train'], default='suite')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--map-roots', type=Path, default=None, help='<dir>/<arena short name>/static_map_v1/observation.json')
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.check:
        bad = 0
        for lk in sorted(a.out.glob('*LOCKED.sha256')):
            if lk.name == 'ALL_LOCKED.sha256':
                continue
            lines = lk.read_text().splitlines()
            r = subprocess.run(['sha256sum', '-c', '--quiet'], input='\n'.join(lines[1:]) + '\n', text=True, cwd=ROOT,
                               capture_output=True)
            files = [ROOT / ln.split('  ', 1)[1] for ln in lines[1:]]
            comb, _ = lock_files(files)
            ok = r.returncode == 0 and comb == lines[0].split()[0]
            bad += not ok
            print(f"{lk.name}: {len(files)} files, {'OK' if ok else 'MISMATCH'} {r.stdout.strip()[:300]}")
        raise SystemExit(bad)
    summary = {}
    for spec in a.set:
        name, paths = spec.split('=', 1)
        cdir, _, odir = paths.partition(':')
        cdir = Path(cdir); odir = Path(odir) if odir else None
        man = json.loads((cdir / 'cases.json').read_text())
        recs = man['records']; groups = [r['scene_id'] for r in recs]
        files = [cdir / 'cases.json']
        for r in recs:
            files.append(cdir / r['case'])
            files += [cdir / p for p in r['routes']]
            assert sha(cdir / r['case']) == r['case_sha256'], r['scene_id']
            assert [sha(cdir / p) for p in r['routes']] == r['route_sha256'], r['scene_id']
        # nothing in the folder that the manifest does not list
        on_disk = {p.resolve() for p in cdir.rglob('*.json')}
        assert on_disk == {f.resolve() for f in files}, f'{name}: files on disk differ from cases.json'
        n_op = 0
        if odir:
            rows = json.loads((odir / 'routes.json').read_text())
            op_files = sorted({ROOT / r['route'] if not Path(r['route']).is_absolute() else Path(r['route']) for r in rows})
            assert {p.resolve() for p in odir.rglob('op_*.json')} == {p.resolve() for p in op_files}
            files += [odir / 'routes.json'] + op_files; n_op = len(op_files)
            per_group = Counter(r['group'] for r in rows)
        arena = recs[0]['arena']; assert all(r['arena'] == arena for r in recs)
        short = Path(arena).name.replace('arena_', '')
        info = {'set': name, 'kind': a.kind, 'cases_dir': rel(cdir), 'groups': len(recs), 'arena': arena,
                'arena_bmp_sha256': sha(ROOT / arena / 'arena_000.bmp'), 'seed': man['seed'], 'prefix': man['prefix'],
                'wave': man['wave'], 'strata_mode': man['strata'],
                'strata': dict(Counter(r['evaluation_stratum'] for r in recs)),
                'splits_written_in_cases': dict(Counter(r['split'] for r in recs)),
                'designed_routes': sum(len(r['routes']) for r in recs), 'files': len(files)}
        if odir:
            info.update(onpolicy_dir=rel(odir), onpolicy_routes=n_op,
                        onpolicy_per_group=dict(Counter(per_group.values())),
                        groups_without_onpolicy=len(set(groups) - set(per_group)))
        if a.map_roots:
            obs = json.loads((a.map_roots / short / 'static_map_v1/observation.json').read_text())
            info['map_root'] = rel(a.map_roots / short)
            info['map_observation_sha256'] = obs['observation_sha256']
            info['map_bmp_matches_arena'] = obs['arena_bmp_sha256'] == info['arena_bmp_sha256']
            assert info['map_bmp_matches_arena'], name
        info['blacklist_hits'] = blacklist_hits(groups)
        tag = 'SUITE_LOCKED' if a.kind == 'suite' else 'CASES_LOCKED'
        comb, lines = lock_files(files)
        info['lock_sha256'] = comb
        (a.out / f'{name}.manifest.json').write_text(json.dumps(info, indent=1) + '\n')
        (a.out / f'{name}.{tag}.sha256').write_text(
            f'{comb}  {name}: {len(files)} files (cases.json, case files, designed routes'
            + (', on-policy routes.json + op_*.json' if odir else '') + '; relative path + content, sorted)\n'
            + '\n'.join(lines) + '\n')
        summary[name] = info
        print(f"{name}: {len(recs)} groups, {len(files)} files, lock {comb[:16]}..., blacklist hits {info['blacklist_hits']}", flush=True)
    # one combined record over every lock + manifest in the folder
    allf = sorted(p for p in a.out.glob('*') if p.name != 'ALL_LOCKED.sha256' and (p.name.endswith('LOCKED.sha256') or p.name.endswith('.manifest.json')))
    comb, lines = lock_files(allf)
    (a.out / 'ALL_LOCKED.sha256').write_text(f'{comb}  every lock and manifest in {rel(a.out)} (relative path + content, sorted)\n'
                                             + '\n'.join(lines) + '\n')
    print('ALL_LOCKED', comb)


if __name__ == '__main__':
    main()
