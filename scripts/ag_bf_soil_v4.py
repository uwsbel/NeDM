#!/usr/bin/env python3
"""Soil task file v4 for arena_gator_20260925 (task B stage 2, "Bfull"; PLAN 7.10, e3/README.md section 0 recipe).

v4 = every soil_v3 row unchanged and in its order (soil_v3 = soil_v2 + the stage-1 evaluation rows, every one of them
driven by 23:33 on 09-25), followed by the rows of the given scripts/ag_eval_tasks.py builds (the Bfull closed-loop
drives: the all-tier Gator-trained planner G_full and the all-tier HMMWV-trained planner H_full on the Gator, then
H_full on the HMMWV). Each build was made with its own --tier and with soil_v3 (and every earlier build) in
--existing, so routes already driven by stage 1 (straight 6 m/s, the stage-1 G / H picks, and any new pick that equals
one of them) are reused by content and are not rows here.
Checks (as scripts/ag_s2_soil_v3.py, which is tied to the soil_v2 hash and is not edited): soil_v3 is a prefix of v4
(every row equal as json), unique ids, unique episode seeds among the new rows and against soil_v3, every new row kind
'eval' with an explicit ['--vehicle', hmmwv|gator] (never a runtime fingerprint: the frozen soil dispatcher refuses
it), tier equal to its build's --tier and < 0, every G3-relative case/route path present locally (the staging lists put
them on G3), and no new id shaped like a training id.
  PYTHONPATH=src:scripts python scripts/ag_bf_soil_v4.py --v3 $K3/e3/tasks/soil_v3.json \
      --builds $K3/e6/tasks/soil_eval_bf1.json $K3/e6/tasks/soil_eval_bf2.json --out $K3/e3/tasks/soil_v4.json [--check-out ... --check-n 2]
"""
import argparse, hashlib, json, re, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
G3 = '/work1/dannegrut/harry/experiments/arena_gator_20260925'
SOIL_V3_SHA = '654ca61b'


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def local_of(p):
    """G3-relative (or absolute G3) path -> the local copy (K3 for K3 files; G3/ext/<repo path> for repo files)."""
    p = str(p)
    if p.startswith(G3 + '/'):
        p = p[len(G3) + 1:]
    if p.startswith('/'):
        return None
    if p.startswith('ext/'):
        return ROOT / p[4:]
    return K3 / p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v3', required=True)
    ap.add_argument('--builds', nargs='+', required=True, help='ag_eval_tasks.py outputs, in the declared order')
    ap.add_argument('--out', required=True)
    ap.add_argument('--check-out', help='also write a small check file (the first --check-n rows of each vehicle of the first build)')
    ap.add_argument('--check-n', type=int, default=2)
    a = ap.parse_args(argv)
    assert sha256_file(a.v3).startswith(SOIL_V3_SHA), 'not the soil_v3 task file'
    v3 = json.load(open(a.v3))
    ids = {r['id'] for r in v3}
    seeds_v3 = {r.get('episode_seed') for r in v3 if r.get('episode_seed') is not None}
    new_rows, per_build, last_tier = [], [], None
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
        assert any(sha256_file(x).startswith(SOIL_V3_SHA) for x in ex), f'{b}: soil_v3 not in --existing'
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
        new_rows += rows
    seeds = [r['episode_seed'] for r in new_rows]
    assert len(set(seeds)) == len(seeds), 'duplicate episode seeds among the new rows'
    assert not set(seeds) & seeds_v3, 'new seed already used in soil_v3'
    v4 = v3 + new_rows
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(v4, open(out, 'w'), indent=None)
    back = json.load(open(out))
    assert len(back) == len(v3) + len(new_rows)
    assert all(json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True) for x, y in zip(back[:len(v3)], v3)), 'soil_v3 not a prefix'
    meta = dict(tool='scripts/ag_bf_soil_v4.py', tool_sha256=sha256_file(__file__), created=time.strftime('%Y-%m-%d %H:%M:%S'),
                v3=dict(file=a.v3, sha256=sha256_file(a.v3), rows=len(v3)), builds=per_build, rows=len(v4), new_rows=len(new_rows),
                new_by_tier=dict(sorted(Counter(r['tier'] for r in new_rows).items())),
                new_by_vehicle=dict(Counter(r['vehicle'] for r in new_rows)), file_sha256=sha256_file(out),
                order_note='soil_v3 rows unchanged (prefix, all driven); new rows: tier -9 Bfull primary (Gator), tier -8 secondary (H_full on the HMMWV)')
    json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    print(json.dumps({k: meta[k] for k in ('rows', 'new_rows', 'new_by_tier', 'new_by_vehicle', 'file_sha256')}, indent=1))
    if a.check_out:
        first = [r for r in new_rows if r['tier'] == per_build[0]['tier']]
        pick = []
        for veh in ('hmmwv', 'gator'):
            rs = sorted((r for r in first if r['vehicle'] == veh), key=lambda r: hashlib.md5(r['id'].encode()).hexdigest())
            pick += rs[:a.check_n]
        json.dump(pick, open(a.check_out, 'w'), indent=None)
        print(f'check file {a.check_out}: {len(pick)} rows (identical to their v4 rows) sha256 {sha256_file(a.check_out)[:16]}')


if __name__ == '__main__':
    main()
