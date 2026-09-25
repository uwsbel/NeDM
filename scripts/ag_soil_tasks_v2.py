#!/usr/bin/env python3
"""Soil task file v2 for arena_gator_20260925 (module E3b2; PLAN 7.8, e3/README.md section 6, REVIEW_R2 amendment 8).

Two stages, so the HMMWV part is built by the E3a tool itself:
  1. scripts/ag_soil_tasks.py (unchanged, soil_v1 defaults) with --head <g217 dev rows> <spread headroom rows>
     --check-superset soil_v1.json -> the HMMWV rows (every soil_v1 row unchanged + the 597 spread headroom rows,
     tier -1). The Gator rows cannot go through its --append: it asserts unique episode seeds over all rows, and a Gator
     row must carry the SAME seed as its HMMWV twin (10 of them are also the soil_v1 drift rows).
  2. this script adds
     - tier -2: 3 'bitid__<id>' HMMWV rows = already completed soil_v1 training rows (run by crm_collect.py directly in
       job 436080/436092) re-driven through the Gator dispatcher ag_crm_collect.py with no vehicle flag; compared array
       by array with the originals (the cluster check that HMMWV physics is the same through the new collector);
     - tiers 0-12: the Gator rows = the 15,235 HMMWV collect_v1 ids (every f104 run with an outcome, none QA-flagged;
       tier 0 has 1,199, tier 12 836), id 'gator__<id>', the collect_v1 case / route (absolute crm_f104_20260916 paths)
       and episode_seed, extra ['--vehicle', 'gator'], built exactly as scripts/ag_pilot_tasks.py builds its
       calibrated rows (the 144 pilot rows are asserted equal to their production rows),
     and writes the rows in the order: tier -2, tier -1 (dev, drift, spread headroom), then per tier k = 0..12 the Gator
     rows, the g203 rows, the g228 rows. (crm_worker.py sorts every worker's shuffled rows by tier, so only the tiers
     decide what runs first; the order inside a tier is random per worker.) Nothing above tier 12.
Checks: every soil_v1 row and every stage-1 row present unchanged, unique ids, unique seeds per vehicle among the
training rows, Gator seeds = their twins' collect_v1 seeds, Gator extra exactly ['--vehicle', 'gator'], no HMMWV row
names a vehicle, every G3-relative path exists in K3.
  PYTHONPATH=src:scripts python scripts/ag_soil_tasks_v2.py --out artifacts/traverse/arena_gator_20260925/e3/tasks/soil_v2.json
"""
import argparse, hashlib, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_tasklib as L
import ag_soil_tasks

CRM_F104 = '/work1/dannegrut/harry/experiments/crm_f104_20260916'
COLLECT_V1_LOCAL = L.ROOT / 'artifacts/traverse/crm_f104_v1/collect_v1'
F104_CASES = L.ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/cases_night2/cases/cases.json'
F104_TASKS = L.K3 / 'e3/tasks/ref/crm_f104_tasks_train.json'
F104_TASKS_SHA = 'b4e2fdbaeb455e1d400131a49ee3e3d7da29f0254810d1b6557e753a6467d68e'   # collect_v1_inputs.sha256
SOIL_V1 = L.K3 / 'e3/tasks/soil_v1.json'
SOIL_V1_SHA = 'db0ba97f4b763ae6'
DEV = L.K3 / 'e3/dev_headroom/tasks_dev.json'
SPREAD = L.K3 / 'e3/tasks/spread_headroom_rows.json'
PILOT = L.K3 / 'e3/tasks/pilot_gator_soil.json'
ARENA_ORDER = {'f104': 0, 'g203': 1, 'g228': 2}


def md5hex(s):
    return hashlib.md5(s.encode()).hexdigest()


def gator_rows():
    assert L.sha256_file(F104_TASKS) == F104_TASKS_SHA, 'not the collect_v1 task file'
    ref = json.load(open(F104_TASKS))
    qa = json.load(open(COLLECT_V1_LOCAL / 'qa.json'))
    flagged = {f['id'] if isinstance(f, dict) else f for f in qa['flagged_ids']}
    stratum = {r['scene_id']: r['evaluation_stratum'] for r in json.load(open(F104_CASES))['records']}
    rows = []
    for r in sorted(ref, key=lambda r: (r['tier'], r['group'], r['id'])):
        run = COLLECT_V1_LOCAL / 'runs' / r['id']
        if r['id'] in flagged or not (run / 'outcome.json').exists():
            continue
        assert r['tier'] <= 12, r
        case = json.load(open(run / 'case.json'))
        rows.append(dict(id='gator__' + r['id'], pair_id=r['id'], group=r['group'], stratum=stratum[r['group']],
                         arena='f104', case=f"{CRM_F104}/{r['case']}", route=f"{CRM_F104}/{r['route']}",
                         tier=r['tier'], episode_seed=r['episode_seed'], run=True,
                         kind='on_policy' if '_op_' in r['id'] else 'designed', split=case['split'],
                         vehicle='gator', wheel='calibrated', extra=['--vehicle', 'gator'],
                         ref_runs=f'{CRM_F104}/collect_v1/runs'))
    return rows


def bitid_rows(v1, n_each=1):
    """one completed tier-0 soil_v1 row per (arena, kind) class g203/designed, g228/designed, g228/on_policy
    (lowest md5 of the id), re-driven as bitid__<id> at tier -2 with no extra."""
    out = []
    for arena, kind in (('g203', 'designed'), ('g228', 'designed'), ('g228', 'on_policy')):
        cand = sorted((r for r in v1 if r.get('arena') == arena and r.get('kind') == kind and r['tier'] == 0),
                      key=lambda r: md5hex(r['id']))[:n_each]
        for r in cand:
            b = dict(r)
            b.update(id='bitid__' + r['id'], tier=-2, kind='bitid', ref_id=r['id'], ref_runs='soil_v1/runs', vehicle='hmmwv')
            assert 'extra' not in b
            out.append(b)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--stage1', default='/tmp/ag_e3b2/soil_v2_stage1.json')
    ap.add_argument('--check-subset', help='also write this many-row check file: the bitid rows + the N Gator tier-0 rows '
                    'with the lowest md5 of the id (rows identical to soil_v2 rows)')
    ap.add_argument('--check-n', type=int, default=12)
    a = ap.parse_args(argv)
    assert L.sha256_file(SOIL_V1).startswith(SOIL_V1_SHA)
    # ---- stage 1: the unchanged E3a builder, soil_v1 defaults + the spread headroom head rows
    s1_argv = ['--head', str(DEV), str(SPREAD), '--f104-tasks', str(F104_TASKS), '--check-superset', str(SOIL_V1),
               '--out', a.stage1]
    ag_soil_tasks.main(s1_argv)
    stage1 = json.load(open(a.stage1))
    v1 = json.load(open(SOIL_V1))
    # ---- stage 2
    gator = gator_rows()
    bitid = bitid_rows(v1)
    pilot = {r['id']: r for r in json.load(open(PILOT)) if r['id'].startswith('gator__')}
    gmap = {r['id']: r for r in gator}
    assert len(pilot) == 144 and all(gmap[i] == r for i, r in pilot.items()), 'pilot rows differ from production rows'
    head2 = sorted(bitid, key=lambda r: r['id'])
    tier_m1 = [r for r in stage1 if r['tier'] < 0]
    train = [r for r in stage1 if r['tier'] >= 0] + gator
    order = {id(r): i for i, r in enumerate(train)}
    train.sort(key=lambda r: (r['tier'], ARENA_ORDER[r['arena']], order[id(r)]))
    rows = head2 + tier_m1 + train

    # ---- checks
    ids = [r['id'] for r in rows]
    assert len(set(ids)) == len(ids), 'duplicate ids'
    new = {r['id']: r for r in rows}
    for name, old in (('soil_v1', v1), ('stage1', stage1)):
        changed = [r['id'] for r in old if new.get(r['id']) != r]
        assert not changed, f'{len(changed)} rows of {name} missing or changed, e.g. {changed[:3]}'
    assert all(r['tier'] <= 12 for r in rows)
    hm_train = [r for r in rows if r['tier'] >= 0 and r['arena'] in ('g203', 'g228')]
    assert len({r['episode_seed'] for r in hm_train}) == len(hm_train)
    assert len({r['episode_seed'] for r in gator}) == len(gator)
    ref = {r['id']: r for r in json.load(open(F104_TASKS))}
    assert all(r['episode_seed'] == ref[r['pair_id']]['episode_seed'] and r['tier'] == ref[r['pair_id']]['tier'] for r in gator)
    for r in rows:
        ex = r.get('extra', [])
        if r['id'].startswith('gator__'):
            assert ex == ['--vehicle', 'gator'] and r['vehicle'] == 'gator', r['id']
        else:
            assert not any('vehicle' in str(x) or 'gator' in str(x) for x in ex), r['id']
            assert r.get('vehicle', 'hmmwv') == 'hmmwv', r['id']
    assert len(gator) == 15235 and len({r['pair_id'] for r in gator}) == 15235
    missing = [(r['id'], p) for r in rows for p in (r['case'], r['route']) if L.local_path(p) is not None and not L.local_path(p).exists()]
    assert not missing, f'{len(missing)} referenced files missing locally, e.g. {missing[:2]}'
    # absolute crm_f104_20260916 paths (Gator rows) are checked on the cluster login node before the launch

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, 'w'))
    by_tier = Counter((r['arena'], r['tier']) for r in rows)
    meta = dict(tool='scripts/ag_soil_tasks_v2.py', tool_sha256=L.sha256_file(__file__),
                stage1_tool='scripts/ag_soil_tasks.py', stage1_tool_sha256=L.sha256_file(ag_soil_tasks.__file__),
                stage1_argv=s1_argv, stage1_sha256=L.sha256_file(a.stage1), stage1_rows=len(stage1),
                superset_of=dict(soil_v1=L.sha256_file(SOIL_V1), stage1=L.sha256_file(a.stage1)),
                n_rows=len(rows), file_sha256=L.sha256_file(out),
                by_kind=dict(Counter(r.get('kind') for r in rows)), by_arena=dict(Counter(r['arena'] for r in rows)),
                by_vehicle=dict(Counter('gator' if r['id'].startswith('gator__') else 'hmmwv' for r in rows)),
                gator_rows=len(gator), bitid_rows=[r['id'] for r in bitid],
                pilot_rows_equal_production=len(pilot),
                by_arena_tier={f'{k[0]}:{k[1]}': n for k, n in sorted(by_tier.items(), key=lambda kv: (kv[0][1], ARENA_ORDER.get(kv[0][0], -1), kv[0][0]))})
    json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    if a.check_subset:
        pick = sorted((r for r in gator if r['tier'] == 0), key=lambda r: md5hex(r['id']))[:a.check_n]
        sub = head2 + pick
        assert all(new[r['id']] == r for r in sub)
        json.dump(sub, open(a.check_subset, 'w'))
        meta['check_subset'] = dict(path=a.check_subset, rows=len(sub), sha256=L.sha256_file(a.check_subset))
        json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k != 'by_arena_tier'}, indent=1))


if __name__ == '__main__':
    main()
