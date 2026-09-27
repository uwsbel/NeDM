#!/usr/bin/env python3
"""Lock every E6b pick directory of a suite set before any drive (arena_gator_20260925, module E6b).

For each ag_picks.py output directory under the given roots: re-verify PICKS_LOCKED.sha256 against the route files
(name + content, ga_planner's scheme, as ag_picks.lock_routes), check that every pick record's route file hash equals
the manifest, and record the manifest's sha256, the model checkpoints (sha256), the groups hash and the pick count.
Writes <out>.json (per directory) and <out>.sha256 (one line per directory: lock  manifest-sha256  dir; last line the
sha256 of all lines). Refuses to overwrite an existing lock with different content.

  PYTHONPATH=src:scripts python scripts/ag_e6b_lock.py --dirs "$K3/e6/picks/rigid/g2[0-9][0-9]/*" --out $K3/e6/picks/LOCK_unseen
"""
import argparse, glob, hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ag_picks  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True, help='globs of pick directories')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    dirs = sorted({d for g in a.dirs for d in glob.glob(g) if (Path(d) / 'ag_picks.json').exists()})
    assert dirs, 'no pick directory'
    rec, lines = [], []
    for d in dirs:
        d = Path(d)
        m = json.load(open(d / 'ag_picks.json'))
        lk = (d / 'PICKS_LOCKED.sha256').read_text().split()[0]
        assert lk == m['picks_locked_sha256'] == ag_picks.lock_routes(d), f'{d}: lock mismatch'
        n = 0
        for g, e in m['picks'].items():
            if e is None:
                continue
            assert sha(d / e['route_file']) == e['file_sha256'], f'{d}/{e["route_file"]}: changed'
            n += 1
        ms = sha(d / 'ag_picks.json')
        rec.append(dict(dir=str(d), arena=m['arena'], world=m['world'], mode=m['mode'], model_tag=m.get('model_tag'), n_groups=m['n_groups'],
                        n_picks=n, groups_sha256=m['groups_sha256'], picks_locked_sha256=lk, manifest_sha256=ms,
                        models=[dict(path=x['path'], sha256=x['sha256']) for x in (m.get('models') or [])],
                        numerics=m.get('numerics'), created=m.get('created')))
        lines.append(f'{lk}  {ms}  {d}')
    allh = hashlib.sha256(('\n'.join(lines) + '\n').encode()).hexdigest()
    out = Path(a.out)
    txt = '\n'.join(lines) + f'\n{allh}  ALL ({len(lines)} pick directories)\n'
    p = Path(str(out) + '.sha256')
    if p.exists() and p.read_text() != txt:
        raise SystemExit(f'{p} exists with different content (pick directories changed?)')
    p.write_text(txt)
    json.dump(dict(schema='ag_e6b_lock_v1', all_sha256=allh, n_dirs=len(rec), dirs=rec), open(str(out) + '.json', 'w'), indent=1)
    print(json.dumps(dict(dirs=len(rec), picks=sum(r['n_picks'] for r in rec), all_sha256=allh, out=str(p))))


if __name__ == '__main__':
    main()
