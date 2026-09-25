#!/usr/bin/env python3
"""Gator cluster pilot task files (arena_gator_20260925, module E3b1; PLAN sections 3 and 7.7).

Group rule (declared before any Gator pilot row ran): the 24 f104 night-2 groups = the 4 training-split groups with the
lowest md5(group id) hex digest in each of the 6 collection strata (evaluation_stratum of cases_night2/cases.json).

Soil rows (crm_worker.py, CRM_COLLECTOR = G3/source/scripts/ag_crm_collect.py):
  for each group the HMMWV collect_v1 rows of tiers 0-5 (crm_f104_20260916/tasks_train.json, the crm_tasks.py order),
  unchanged case / route (absolute crm_f104_20260916 paths, as the collect_v1 row) and episode_seed;
  id  gator__<collect_v1 id>    extra ["--vehicle", "gator"]                     tier = collect_v1 tier (0-5)
  id  gatorR8__<collect_v1 id>  extra + calibrated radius + 0.08 m on both axles tier = 10 + collect_v1 tier
  (the worker runs lower tiers first: the calibrated rows before the sensitivity rows).
Rigid rows (gen_runner_g-style runner, GEN_COLLECTOR = G3/source/scripts/ag_gen_collect_ext.py):
  the 12 designed routes of each group, case / route = the absolute fdm_f104_50h_20260909/cases_night2_v1 paths behind
  production_v3 (the HMMWV reference runs), id gator__<group>_route_NN, extra ["--vehicle", "gator",
  "--runtime-fingerprint", G3/runtime/gator_runtime_fingerprint.json], shard = md5(group) % --rigid-shards (every
  route of a group on one node).
Every row carries pair_id (the HMMWV id) and ref_runs (where the HMMWV run lives).

  PYTHONPATH=src:scripts python scripts/ag_pilot_tasks.py --out-dir artifacts/traverse/arena_gator_20260925/e3/tasks
"""
import argparse, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ag_tasklib as L
import ag_vehicle as agv

CRM_F104 = '/work1/dannegrut/harry/experiments/crm_f104_20260916'
F104_CAMPAIGN = '/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909'
F104_LOCAL = L.ROOT / 'artifacts/traverse/fdm_f104_50h_20260909'
COLLECT_V1_LOCAL = L.ROOT / 'artifacts/traverse/crm_f104_v1/collect_v1'
F104_TASKS_SHA = 'b4e2fdbaeb455e1d400131a49ee3e3d7da29f0254810d1b6557e753a6467d68e'   # collect_v1_inputs.sha256
FINGERPRINT = f'{L.G3}/runtime/gator_runtime_fingerprint.json'
R_OFFSET_M = 0.08


def md5hex(s):
    return hashlib.md5(s.encode()).hexdigest()


def pick_groups(per_stratum):
    recs = json.load(open(F104_LOCAL / 'cases_night2/cases/cases.json'))['records']
    by = defaultdict(list)
    for r in recs:
        if r['split'] == 'train':
            by[r['evaluation_stratum']].append(r['scene_id'])
    out = []
    for s in sorted(by):
        out += [(s, g) for g in sorted(by[s], key=md5hex)[:per_stratum]]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--f104-tasks', default=str(L.K3 / 'e3/tasks/ref/crm_f104_tasks_train.json'))
    ap.add_argument('--per-stratum', type=int, default=4)
    ap.add_argument('--soil-tiers', type=int, default=6)
    ap.add_argument('--rigid-shards', type=int, default=2)
    ap.add_argument('--out-dir', required=True)
    a = ap.parse_args(argv)
    assert L.sha256_file(a.f104_tasks) == F104_TASKS_SHA, 'not the collect_v1 task file'
    groups = pick_groups(a.per_stratum)
    gset = {g for _, g in groups}
    stratum = {g: s for s, g in groups}
    ref = [r for r in json.load(open(a.f104_tasks)) if r['group'] in gset and r['tier'] < a.soil_tiers]
    assert len(ref) == len(gset) * a.soil_tiers, len(ref)
    qa = json.load(open(COLLECT_V1_LOCAL / 'qa.json'))
    flagged = {f['id'] for f in qa['flagged_ids']}
    r8 = {k: round(v['radius_m'] + R_OFFSET_M, 5) for k, v in agv.GATOR_SOIL_WHEEL.items()}

    soil = []
    for variant, prefix, dtier, extra in (
            ('calibrated', 'gator__', 0, ['--vehicle', 'gator']),
            ('radius_plus_0.08', 'gatorR8__', 10, ['--vehicle', 'gator', '--gator-soil-radius-front', str(r8['front']),
                                                   '--gator-soil-radius-rear', str(r8['rear'])])):
        for r in sorted(ref, key=lambda r: (r['tier'], r['group'])):
            assert r['id'] not in flagged and (COLLECT_V1_LOCAL / 'runs' / r['id'] / 'outcome.json').exists(), r['id']
            case = json.load(open(COLLECT_V1_LOCAL / 'runs' / r['id'] / 'case.json'))
            soil.append(dict(id=prefix + r['id'], pair_id=r['id'], group=r['group'], stratum=stratum[r['group']],
                             arena='f104', case=f"{CRM_F104}/{r['case']}", route=f"{CRM_F104}/{r['route']}",
                             tier=dtier + r['tier'], episode_seed=r['episode_seed'], run=True,
                             kind='on_policy' if '_op_' in r['id'] else 'designed', split=case['split'],
                             vehicle='gator', wheel=variant, extra=extra, ref_runs=f'{CRM_F104}/collect_v1/runs'))
    rigid = []
    for s, g in groups:
        for k in range(12):
            rid = f'{g}_route_{k:02d}'
            assert (F104_LOCAL / 'production_v3/runs' / rid / 'outcome.json').exists(), rid
            assert (F104_LOCAL / 'cases_night2/cases/routes' / g / f'route_{k:02d}.json').exists()
            rigid.append(dict(id='gator__' + rid, pair_id=rid, group=g, stratum=s, arena='f104',
                              case=f'{F104_CAMPAIGN}/cases_night2_v1/{g}.json',
                              route=f'{F104_CAMPAIGN}/cases_night2_v1/routes/{g}/route_{k:02d}.json',
                              shard=int(md5hex(g), 16) % a.rigid_shards, run=True, tier=k, episode_seed=L.md5_int(rid),
                              kind='designed', profile=k % 4, split='train', vehicle='gator',
                              extra=['--vehicle', 'gator', '--runtime-fingerprint', FINGERPRINT],
                              ref_runs=f'{F104_CAMPAIGN}/production_v3/runs'))
    for rows in (soil, rigid):
        ids = [r['id'] for r in rows]
        assert len(set(ids)) == len(ids)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in (('pilot_gator_soil.json', soil), ('pilot_gator_rigid.json', rigid)):
        json.dump(rows, open(out / name, 'w'))
        files[name] = dict(sha256=L.sha256_file(out / name), rows=len(rows),
                           by_tier=dict(sorted(Counter(r['tier'] for r in rows).items())),
                           by_kind=dict(Counter(r['kind'] for r in rows)),
                           by_shard=dict(sorted(Counter(r.get('shard') for r in rows).items())) if name.endswith('rigid.json') else None)
    meta = dict(tool='scripts/ag_pilot_tasks.py', tool_sha256=L.sha256_file(__file__),
                group_rule='4 training-split groups with the lowest md5(group id) hex per evaluation_stratum (6 strata)',
                groups=[dict(stratum=s, group=g) for s, g in groups], f104_tasks=a.f104_tasks, f104_tasks_sha256=F104_TASKS_SHA,
                soil_tiers=list(range(a.soil_tiers)), r8_radius_m=r8, calibrated_radius_m={k: v['radius_m'] for k, v in agv.GATOR_SOIL_WHEEL.items()},
                files=files)
    json.dump(meta, open(out / 'pilot_gator.meta.json', 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    main()
