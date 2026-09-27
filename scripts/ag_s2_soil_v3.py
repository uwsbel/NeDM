#!/usr/bin/env python3
"""Soil task file v3 for arena_gator_20260925 (soil track step 2; PLAN 7.10, e3/README.md section 0 relaunch recipe).

v3 = every soil_v2 row unchanged and in its order, followed by the soil evaluation rows of the given
scripts/ag_eval_tasks.py builds. crm_worker.py sorts each worker's shuffled rows by tier only, so the declared evaluation
order is carried by the tiers of the builds (each build was made with its own --tier and with every earlier build in
--existing, so a route shared by two priority groups is one drive, in the earlier group):
    tier -9  A primary (M1a, M1b, M3a, A3, straight 6 on the 8 unseen arenas) + B primary (G and H on the Gator, f104 800)
    tier -8  M3b, M2 (unseen)
    tier -7  anchors: straight 6 on the Gator, H and straight 6 on the HMMWV (f104 800)
    tier -6  in distribution: M3a, A3 on the 200 f104 groups (M1a there = the H-on-HMMWV anchor drive); M1a, M3a, A3 on
             the g203 / g228 held-out groups
    tier -5  extra, not in the declared order: straight 6 on the g203 / g228 held-out groups
All of them sort ahead of the unfinished training tiers 7-12 (tiers 0-6 are complete); tiers -2/-1 of soil_v2 are done.
Checks: soil_v2 is a prefix of v3 (every row byte-equal as json), unique ids, unique episode seeds among the evaluation
rows and against soil_v2, every evaluation row kind 'eval' with an explicit ['--vehicle', hmmwv|gator] (never a
runtime fingerprint: the frozen soil dispatcher refuses it), tier equal to its build's --tier and < 0, every G3-relative
case/route path present locally (the staging lists put them on G3), and no evaluation id shaped like a training id.
  PYTHONPATH=src:scripts python scripts/ag_s2_soil_v3.py --v2 $K3/e3/tasks/soil_v2.json \
      --builds $K3/e6/tasks/soil_eval_p1.json ... --out $K3/e3/tasks/soil_v3.json [--check-out ... --check-n 6]
"""
import argparse, hashlib, json, re, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
G3 = '/work1/dannegrut/harry/experiments/arena_gator_20260925'
SOIL_V2_SHA = 'acf1e98c3c995c10'


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def local_of(p):
    """G3-relative (or absolute G3) path -> the local copy (K3 for K3 files; G3/ext/<repo path> for repo files)."""
    p = str(p)
    if p.startswith(G3 + '/'):
        p = p[len(G3) + 1:]
    if p.startswith('/'):
        return None                      # absolute path outside G3 (e.g. crm_f104_20260916): not staged by us
    if p.startswith('ext/'):
        return ROOT / p[4:]
    return K3 / p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v2', required=True)
    ap.add_argument('--builds', nargs='+', required=True, help='ag_eval_tasks.py outputs, in the declared order')
    ap.add_argument('--out', required=True)
    ap.add_argument('--check-out', help='also write a small check file (the first --check-n rows of each vehicle of tier -9)')
    ap.add_argument('--check-n', type=int, default=3)
    a = ap.parse_args(argv)
    assert sha256_file(a.v2).startswith(SOIL_V2_SHA), 'not the soil_v2 task file'
    v2 = json.load(open(a.v2))
    ids = {r['id'] for r in v2}
    seeds_v2 = {r.get('episode_seed') for r in v2 if r.get('episode_seed') is not None}
    eval_rows, per_build, last_tier = [], [], None
    for b in a.builds:
        meta = json.load(open(b + '.meta.json'))
        assert meta['file_sha256'] == sha256_file(b), f'{b}: changed since ag_eval_tasks.py wrote it'
        argv_b = meta['argv']
        tier = int(argv_b[argv_b.index('--tier') + 1])
        assert tier < 0, (b, tier)
        assert last_tier is None or tier > last_tier, 'builds must be given in the declared order (rising tier)'
        last_tier = tier
        ex = []
        if '--existing' in argv_b:
            for x in argv_b[argv_b.index('--existing') + 1:]:
                if x.startswith('--'):
                    break
                ex.append(x)
        rows = json.load(open(b))
        for r in rows:
            assert r['tier'] == tier and r['kind'] == 'eval' and r['world'] == 'crm', r['id']
            assert r['extra'] == ['--vehicle', r['vehicle']] and r['vehicle'] in ('hmmwv', 'gator'), r['id']
            assert not re.search(r'_(route|op)_\d{2}$', r['id']), r['id']
            assert r['id'] not in ids, f'duplicate id {r["id"]}'
            ids.add(r['id'])
            for k in ('case', 'route'):
                lp = local_of(r[k])
                if lp is not None:
                    assert lp.exists(), (r['id'], k, r[k])
        per_build.append(dict(file=b, sha256=sha256_file(b), tier=tier, rows=len(rows), existing=ex,
                              by_vehicle=dict(Counter(r['vehicle'] for r in rows)), by_arm_first=dict(Counter(r['arms'][0] for r in rows)),
                              by_arena=dict(Counter(r['arena'] for r in rows))))
        eval_rows += rows
    seeds = [r['episode_seed'] for r in eval_rows]
    assert len(set(seeds)) == len(seeds), 'duplicate episode seeds among evaluation rows'
    assert not set(seeds) & seeds_v2, 'evaluation seed already used in soil_v2'
    first_training_tier = min(r['tier'] for r in v2 if r['tier'] >= 0)          # 0: every training tier sorts after the evaluation rows
    assert all(r['tier'] < first_training_tier for r in eval_rows)
    v3 = v2 + eval_rows
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(v3, open(out, 'w'), indent=None)
    # superset check on the written file
    back = json.load(open(out))
    assert len(back) == len(v2) + len(eval_rows)
    assert all(json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(back[:len(v2)], v2)), 'soil_v2 not a prefix'
    meta = dict(tool='scripts/ag_s2_soil_v3.py', tool_sha256=sha256_file(__file__), created=time.strftime('%Y-%m-%d %H:%M:%S'),
                v2=dict(file=a.v2, sha256=sha256_file(a.v2), rows=len(v2)), builds=per_build, rows=len(v3), eval_rows=len(eval_rows),
                eval_by_tier=dict(sorted(Counter(r['tier'] for r in eval_rows).items())),
                eval_by_vehicle=dict(Counter(r['vehicle'] for r in eval_rows)), file_sha256=sha256_file(out),
                order_note='tiers -9 .. -5 = the declared evaluation order; soil_v2 rows unchanged (prefix)')
    json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    print(json.dumps({k: meta[k] for k in ('rows', 'eval_rows', 'eval_by_tier', 'eval_by_vehicle', 'file_sha256')}, indent=1))
    if a.check_out:
        first = [r for r in eval_rows if r['tier'] == per_build[0]['tier']]
        pick = []
        for veh in ('hmmwv', 'gator'):
            rs = sorted((r for r in first if r['vehicle'] == veh), key=lambda r: hashlib.md5(r['id'].encode()).hexdigest())
            pick += rs[:a.check_n]
        json.dump(pick, open(a.check_out, 'w'), indent=None)
        print(f'check file {a.check_out}: {len(pick)} rows (identical to their v3 rows) sha256 {sha256_file(a.check_out)[:16]}')


if __name__ == '__main__':
    main()
