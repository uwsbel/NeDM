#!/usr/bin/env python3
"""Soil collection task file of offroad_vehicles_20260927 for one vehicle (module M3; PLAN section 3; S3 3.1, 3.5).

Rows = the Gator's 15,235 soil rows gator__<id> of arena_gator_20260925/e3/tasks/soil_v2.json (every f104 collect_v1
id, tiers 0-12: tier 0 1,199, tiers 1-11 1,200, tier 12 836) with only the prefix and the vehicle flag changed:
  id <v>__<collect_v1 id>, extra ['--vehicle', <v>], vehicle / arm <v>, the Gator row's case, route (absolute
  crm_f104_20260916 paths), episode seed, tier, split, kind, group, stratum, ref_runs (the HMMWV twins); 'wheel' (a
  Gator field) dropped; M113 rows (m113, m113_g4) also carry "config": "configs/crm_m113.json" and "timeout_s".
Versions (every version is a new file, never an edit; K3/e3/README.md recipe):
  --previous F   the new file is a checked superset of F (the smoke file or an earlier collection version): every row of
                 F is kept with every field unchanged, except
                   * rows whose id is one of the new collection rows (the smoke's sample-A rows of this vehicle, e.g.
                     polaris__<id> at smoke tier -3/-2): they take their collect_v1 tier (field 'retiered_from' keeps the
                     old one) so the dataset builder sees tiers 0-12; the run folder is shared (same id, same output
                     folder), so these drives are not repeated;
                   * the 'run' flag, only where asked (--disable-prefix / --enable-prefix / --run-tiers for this vehicle).
  --run-tiers A-B | all | none   collection rows outside the range get run: false (e.g. 0-6 for stage 1; none to
                 queue a second vehicle behind the first, S3 3.5); rows already in --previous keep their run flag
                 unless --rerun-tiers is given.
  --disable-prefix P ...   set run: false on every row whose id starts with P__ (e.g. sensitivity arms of the smoke)
  --enable-prefix P ...    set run: true on those rows
  --append FILE [--append-tier T]   insert evaluation rows (e.g. module M4's ov_eval_tasks output) at a negative tier
                 (their own tier, or T); they must name their vehicle in extra, have unique ids and tier < 0.
Checks (the build stops on failure): unique ids; 15,235 collection rows for the vehicle with the tier counts above;
seeds, cases, routes, splits equal the Gator rows; extra exactly ['--vehicle', v]; M113 rows carry the 0.5 ms config and
a timeout >= 3,600 s, no other row a config; superset of --previous (only the declared changes); appended rows valid.
--check-cluster: every case / route path exists on the cluster (read-only ssh).

  PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
  $PY scripts/ov_soil_tasks.py --vehicle polaris --previous $K4/tasks/smoke_v1.json --run-tiers all \
      --disable-prefix polaris_pc polaris_4wd polaris_w08 --out $K4/tasks/soil_v1.json
  $PY scripts/ov_soil_tasks.py --vehicle polaris --previous $K4/tasks/soil_v1.json --append <eval rows> --append-tier -5 \
      --out $K4/tasks/soil_v2.json
"""
import argparse, json, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ov_smoke_tasks as S  # noqa: E402  (paths, vehicle names, cluster path check)

TIER_COUNTS = {0: 1199, **{t: 1200 for t in range(1, 12)}, 12: 836}
M113_ARMS = ('m113', 'm113_g4')
SAME_FIELDS = ('pair_id', 'group', 'stratum', 'arena', 'case', 'route', 'episode_seed', 'kind', 'split', 'vehicle', 'extra',
               'config', 'timeout_s')


def parse_tiers(spec):
    if spec == 'all':
        return set(range(13))
    if spec == 'none':
        return set()
    lo, _, hi = spec.partition('-')
    return set(range(int(lo), int(hi or lo) + 1))


def collection_rows(v, run_tiers, timeout_s):
    assert v in S.VEHICLES and v not in ('hmmwv',) and '__' not in v, v
    out = []
    for g in sorted(S.load_gator_rows().values(), key=lambda g: (g['tier'], g['group'], g['pair_id'])):
        r = dict(id=f'{v}__{g["pair_id"]}', pair_id=g['pair_id'], group=g['group'], stratum=g['stratum'], arena=g['arena'],
                 case=g['case'], route=g['route'], tier=g['tier'], collect_tier=g['tier'], episode_seed=g['episode_seed'],
                 run=g['tier'] in run_tiers, kind=g['kind'], split=g['split'], vehicle=v, arm=v, sample='collection',
                 extra=['--vehicle', v], ref_runs=g['ref_runs'], stored_gator_run=f'{S.GATOR_RUNS}/{g["id"]}')
        if v in M113_ARMS:
            r['config'] = S.M113_CONFIG
            r['timeout_s'] = timeout_s
        out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vehicle', required=True, choices=[v for v in S.VEHICLES if v != 'hmmwv'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--previous', default=None)
    ap.add_argument('--run-tiers', default='all')
    ap.add_argument('--rerun-tiers', action='store_true', help='also apply --run-tiers to collection rows already in --previous')
    ap.add_argument('--disable-prefix', nargs='*', default=[])
    ap.add_argument('--enable-prefix', nargs='*', default=[])
    ap.add_argument('--append', nargs='*', default=[])
    ap.add_argument('--append-tier', type=int, default=None)
    ap.add_argument('--m113-timeout-s', type=int, default=4000)
    ap.add_argument('--check-cluster', action='store_true')
    a = ap.parse_args(argv)
    v = a.vehicle
    assert a.m113_timeout_s >= 3600
    run_tiers = parse_tiers(a.run_tiers)
    coll = collection_rows(v, run_tiers, a.m113_timeout_s)
    cmap = {r['id']: r for r in coll}
    prev = json.load(open(a.previous)) if a.previous else []
    prev_ids = {r['id'] for r in prev}
    rows, retiered, flips = [], Counter(), Counter()
    for r in prev:
        n = dict(r)
        if r['id'] in cmap:
            c = cmap[r['id']]
            diff = [k for k in SAME_FIELDS if r.get(k) != c.get(k)]
            assert not diff, f"{r['id']}: previous row differs from the collection row in {diff}"
            if r.get('tier') != c['tier']:
                n['retiered_from'] = r['tier']; n['tier'] = c['tier']; retiered[str(r['tier'])] += 1
            if a.rerun_tiers:
                n['run'] = c['run']
        pre = r['id'].split('__', 1)[0] if '__' in r['id'] else ''
        if pre in a.disable_prefix:
            n['run'] = False
        if pre in a.enable_prefix:
            n['run'] = True
        if n.get('run', True) != r.get('run', True):
            flips[f"{pre}:{r.get('run', True)}->{n.get('run', True)}"] += 1
        rows.append(n)
    new_coll = [r for r in coll if r['id'] not in prev_ids]
    for r in new_coll:
        pre = r['id'].split('__', 1)[0]
        if pre in a.disable_prefix:
            r['run'] = False
        if pre in a.enable_prefix:
            r['run'] = True
    rows += new_coll
    appended = []
    for f in a.append:
        for r in json.load(open(f)):
            r = dict(r)
            if a.append_tier is not None:
                r['tier_set_by'] = 'ov_soil_tasks --append-tier'; r['tier'] = a.append_tier
            assert int(r['tier']) < 0, f"appended row {r['id']} must sit at a negative tier"
            assert r.get('vehicle') in S.VEHICLES and r.get('extra', [])[:2] == ['--vehicle', r['vehicle']], r['id']
            assert r['id'] not in prev_ids and r['id'] not in cmap and r['id'] not in {x['id'] for x in appended}, f"appended id {r['id']} already present"
            appended.append(r)
    rows += appended
    rows.sort(key=lambda r: int(r.get('tier', 0)))       # stable

    # ---------------------------------------------------------------- checks
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)), 'duplicate ids'
    mine = [r for r in rows if r['id'] in cmap]
    assert len(mine) == 15235 and Counter(r['tier'] for r in mine) == Counter(TIER_COUNTS), Counter(r['tier'] for r in mine)
    gator = S.load_gator_rows()
    for r in mine:
        g = gator[r['pair_id']]
        assert all(r[k] == g[k] for k in ('case', 'route', 'episode_seed', 'split', 'kind', 'group', 'tier')), r['id']
        assert r['extra'] == ['--vehicle', v]
    assert len({r['episode_seed'] for r in mine}) == 15235
    for r in rows:
        if r.get('config'):
            assert r.get('vehicle') in M113_ARMS or r.get('arm') == 'gatorh', r['id']
            assert r['config'] == S.M113_CONFIG and int(r.get('timeout_s', 0)) >= 3600, r['id']
        elif r.get('vehicle') in M113_ARMS:
            raise AssertionError(f"{r['id']}: M113 row without its 0.5 ms config")
        assert r.get('extra', [])[:2] == ['--vehicle', r.get('vehicle')], r['id']
    if a.previous:
        new = {r['id']: r for r in rows}
        for r in prev:
            n = new[r['id']]
            ch = {k for k in set(r) | set(n) if r.get(k) != n.get(k)} - {'run', 'tier', 'retiered_from'}
            assert not ch, f"{r['id']}: changed fields {ch}"
            if n.get('tier') != r.get('tier'):
                assert r['id'] in cmap and n['retiered_from'] == r['tier']
    cl = None
    if a.check_cluster:
        cl = S.check_cluster(sorted({p for r in rows if r.get('run', True) for p in (r['case'], r['route'])}))
        assert cl['n_missing'] == 0, cl
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f'{out} exists: task files are never overwritten (write a new version)')
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, 'w'))
    run_rows = [r for r in rows if r.get('run', True)]
    meta = dict(tool='scripts/ov_soil_tasks.py', tool_sha256=S.sha256_file(__file__), created=time.strftime('%F %T'),
                argv=sys.argv[1:] if argv is None else argv, vehicle=v, file_sha256=S.sha256_file(out), rows=len(rows),
                run_rows=len(run_rows), previous=dict(file=a.previous, sha256=S.sha256_file(a.previous), rows=len(prev)) if a.previous else None,
                retiered=dict(retiered), run_flips=dict(flips), new_collection_rows=len(new_coll), appended_rows=len(appended),
                collection_run_tiers=sorted(run_tiers), max_timeout_s=max([int(r.get('timeout_s', 2400)) for r in run_rows] + [2400]),
                by_tier_run={str(t): n for t, n in sorted(Counter(int(r['tier']) for r in run_rows).items())},
                by_prefix_run=dict(Counter(r['id'].split('__', 1)[0] if '__' in r['id'] else 'hmmwv' for r in run_rows)),
                inputs={'soil_v2.json': S.sha256_file(S.SOIL_V2)}, cluster_paths=cl)
    json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
