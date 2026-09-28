#!/usr/bin/env python3
"""Outcome index of soil evaluation drives, for ANY vehicle (offroad_vehicles_20260927, module M4; PLAN 4.2-4.4).
Vehicle-generalised copy of scripts/ag_eval_index.py (frozen; its labels, case / cluster information and run lookup are
imported). One row per (world, vehicle, group, arm), with the drive checks of scripts/ag_s2_extras.py folded in.

Inputs: mapping files of scripts/ov_eval_tasks.py build (schema ov_eval_mapping_v1: every arm entry names its study,
'g4' = this study's drives, 'g3' = a reused arena_gator drive) or of scripts/ag_eval_tasks.py (ag_eval_mapping_v1,
all 'g3'); run folders: --runs-g4 (local sync of G4's soil runs), --runs-g3 (local sync of G3's, e.g.
arena_gator_20260925/e6/runs_soil), --runs (searched for every study, after the study's own folders).

Per drive (ag_eval_index.outcome = ga_analyze.safe_labels after the 1 s settle): fail (goal not reached), unsafe
(not (goal and < 0.05 s rolling back under throttle and min forward speed > -0.30 m/s)), unsafe_noback, backward_only,
tilt30, elapsed, status, positive work; plus
  vehicle_block  the outcome.json vehicle block name; ASSERTED to name the row's vehicle: (block or 'hmmwv') == vehicle,
                 and for this study's drives (g4, the new dispatcher writes a block for every vehicle) block present
  qa_ok, qa_flag crm_qa.check (unreadable / shape / nonfinite / launch / explosion / unstalled_break); 'not_synced' when
                 the run folder lacks the files it reads (initial_state_validation.json, collection_request.json,
                 crm_extra.npz)
  launch_ok      initial_state_validation.json passed (None when not synced)
  belly_*        where vehicle_extra.npz has belly_clearance_min_m (Gator, Polaris; the ag_s1_gator_soil_qa rule):
                 belly_flag = lowest hull point > 0.05 m under the undisturbed surface for > 1 s in a row;
                 fail_belly / unsafe_belly = fail / unsafe OR belly_flag (the PLAN 4.3(c) sensitivity; = fail / unsafe
                 when the vehicle has no belly record)
Per group: stratum, nearest terrain feature -> cluster (ag_eval_index.case_info), set, suite_part ('fresh600' =
f104_pair_group_*, 'tuned200' = f104_crm_eval_group_*: CEM 4 x 64 was tuned on the 200). Missing drives: missing=true.

  PYTHONPATH=src:scripts python scripts/ov_eval_index.py --mapping <tasks>.mapping.json --runs-g4 <sync> \
      --runs-g3 artifacts/traverse/arena_gator_20260925/e6/runs_soil --out <index.json>
  python scripts/ov_eval_index.py --mapping ... --list-files <out.txt>     # run-relative files to sync, per study
"""
import argparse, hashlib, json, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / 'src'))
import ag_eval_index as AEI              # noqa: E402  (frozen: case_info, outcome, find_run, arena_role, group_set)
import ov_eval_tasks as OET              # noqa: E402  (G4 / G3 path mapping)

DT = 0.05
SYNC_FILES = ('outcome.json', 'trajectory.npz', 'episode_complete.json', 'case.json', 'vehicle_extra.npz', 'crm_extra.npz',
              'initial_state_validation.json', 'collection_request.json')


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def longest_run(m):
    best = cur = 0
    for v in m:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def case_local(case, schema):
    """ov mappings write every case G4-relative (ext/... for repository files); ag mappings G3-relative."""
    if schema == 'ag_eval_mapping_v1':
        rel = case[len(OET.G3) + 1:] if case.startswith(OET.G3 + '/') else case
        return AEI.local_of(rel) if not rel.startswith('/') else Path(rel)
    return OET.local_of(case)


def extras(d):
    """crm_qa flags, launch check, belly flag of one run folder."""
    import crm_qa
    x = {}
    need = ('initial_state_validation.json', 'collection_request.json', 'crm_extra.npz')
    if all((d / f).exists() for f in need):
        q = crm_qa.check(str(d))
        x.update(qa_ok=bool(q['ok']), qa_flag=q.get('flag'))
        x['launch_ok'] = bool(json.load(open(d / 'initial_state_validation.json')).get('passed', False))
    else:
        x.update(qa_ok=None, qa_flag='not_synced', launch_ok=None)
    x['belly_available'] = False
    ve = d / 'vehicle_extra.npz'
    if ve.exists():
        try:
            z = np.load(ve)
            if 'belly_clearance_min_m' in z.files:
                b = z['belly_clearance_min_m'].astype(float)
                deep = b < -0.05
                x.update(belly_available=True, belly_min_m=float(np.nanmin(b)) if len(b) else None, belly_run_s=longest_run(deep) * DT,
                         belly_total_s=float(deep.sum() * DT))
                x['belly_flag'] = int(x['belly_run_s'] > 1.0)
        except Exception as e:  # noqa: BLE001
            x['belly_error'] = type(e).__name__
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mapping', nargs='+', required=True)
    ap.add_argument('--runs-g4', nargs='*', default=[]); ap.add_argument('--runs-g3', nargs='*', default=[])
    ap.add_argument('--runs', nargs='*', default=[], help='searched for every study after its own folders')
    ap.add_argument('--out'); ap.add_argument('--list-files', help='write <study>\\t<run id>/<file> lines for the sync and exit')
    ap.add_argument('--no-extras', action='store_true')
    a = ap.parse_args(argv)
    maps = []
    for mp in a.mapping:
        m = json.load(open(mp))
        assert m.get('schema') in ('ov_eval_mapping_v1', 'ag_eval_mapping_v1'), f'{mp}: schema {m.get("schema")}'
        assert m['world'] == 'crm', f'{mp}: soil mappings only'
        maps.append((mp, m))
    if a.list_files:
        lines = sorted({f"{e.get('study', 'g3')}\t{e['run_id']}/{f}" for _, m in maps for g in m['groups'].values() for e in g.values() for f in SYNC_FILES})
        Path(a.list_files).write_text('\n'.join(lines) + '\n')
        print(f'{len(lines)} files of {len({l.split(chr(9))[1].split("/")[0] for l in lines})} runs -> {a.list_files}')
        return
    assert a.out, '--out required'
    indist = set(json.load(open(AEI.INDIST))['groups'])
    me = json.load(open(AEI.MAPERR))['arenas']
    rows, meta_in, t0, cinfo, xcache = [], [], time.time(), {}, {}
    roots = dict(g4=list(a.runs_g4) + list(a.runs), g3=list(a.runs_g3) + list(a.runs))
    for mp, m in maps:
        default_study = 'g4' if m['schema'] == 'ov_eval_mapping_v1' else 'g3'
        meta_in.append(dict(mapping=str(mp), sha256=sha256_file(mp), schema=m['schema'], tasks=m.get('tasks'), tasks_sha256=m.get('tasks_sha256'),
                            arms=[dict(name=x['name'], vehicle=x['vehicle'], modes=x['modes'], model_tags=x['model_tags'], arenas=x['arenas']) for x in m['arms']]))
        arm_meta = {x['name']: x for x in m['arms']}
        for g, arms in sorted(m['groups'].items()):
            for name, e in arms.items():
                study = e.get('study', default_study)
                key = (m['schema'], e['case'])
                if key not in cinfo:
                    cinfo[key] = AEI.case_info(case_local(e['case'], m['schema']))
                ci = cinfo[key]
                arena = e['arena']; mv = me.get(arena, {})
                row = dict(world='crm', vehicle=e['vehicle'], arena=arena, role=AEI.arena_role(arena), set=AEI.group_set(g, arena, indist), group=g,
                           suite_part='fresh600' if g.startswith('f104_pair_group_') else 'tuned200' if g.startswith('f104_crm_eval_group_') else None,
                           arm=name, mode=e.get('mode') or arm_meta[name]['modes'][0], model_tag=(arm_meta[name]['model_tags'] or [None])[0],
                           run_id=e['run_id'], study=study, reused_from=e.get('reused_from'), route_sha256=e['route_sha256'], P=e.get('P'), z=e.get('z'),
                           stratum=ci['stratum'], feature=ci['feature'], feature_kind=ci['feature_kind'], cluster=f"{arena}:{ci['feature']}",
                           design_feature=ci['design_feature'],
                           cluster_design=f"{arena}:{ci['design_feature'] if ci['design_feature'] is not None else 'none'}",
                           map_err_rmse_m=(mv.get('all') or {}).get('rmse'), dist_nearest_training=mv.get('distance_nearest_training'),
                           nearest_training_arena=mv.get('nearest_training_arena'), dist_f104=mv.get('distance_f104'))
                d = AEI.find_run(e['run_id'], roots[study])
                if d is None:
                    row['missing'] = True
                else:
                    row.update(AEI.outcome(d), missing=False, run_dir=str(d))
                    vb = row['vehicle_block']
                    assert (vb or 'hmmwv') == e['vehicle'], f"{e['run_id']}: vehicle block {vb!r} but the row asked for {e['vehicle']}"
                    if study == 'g4':
                        assert vb is not None, f"{e['run_id']}: a drive of the new dispatcher without a vehicle block"
                    if not a.no_extras:
                        if (study, e['run_id']) not in xcache:
                            xcache[(study, e['run_id'])] = extras(Path(d))
                        row.update(xcache[(study, e['run_id'])])
                    row['fail_belly'] = int(row['fail'] or row.get('belly_flag', 0))
                    row['unsafe_belly'] = int(row['unsafe'] or row.get('belly_flag', 0))
                rows.append(row)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    have = [r for r in rows if not r['missing']]
    arm_key = lambda r: f"{r['world']}|{r['vehicle']}|{r['arm']}"
    summary = dict(rows=len(rows), driven=len(have), missing=len(rows) - len(have),
                   missing_by_arm=dict(Counter(arm_key(r) for r in rows if r['missing'])), driven_by_arm=dict(Counter(arm_key(r) for r in have)),
                   studies=dict(Counter(r['study'] for r in rows)), sets=dict(Counter(r['set'] for r in rows)),
                   suite_parts=dict(Counter(str(r['suite_part']) for r in rows)), arenas=dict(Counter(r['arena'] for r in rows)),
                   qa_flags=dict(Counter(f"{arm_key(r)}|{r.get('qa_flag')}" for r in have if r.get('qa_flag'))),
                   belly_flagged=dict(Counter(arm_key(r) for r in have if r.get('belly_flag'))))
    json.dump(dict(schema='ov_eval_index_v1', tool='scripts/ov_eval_index.py', tool_sha256=sha256_file(__file__),
                   created=time.strftime('%Y-%m-%d %H:%M:%S'), inputs=meta_in, runs=dict(g4=a.runs_g4, g3=a.runs_g3, any=a.runs),
                   labels=dict(fail='goal not reached', unsafe='not (goal and back_s < 0.05 and min_vx > -0.30), after the 1 s settle',
                               unsafe_noback='fail', backward_only='unsafe and not fail', tilt30='max |roll|,|pitch| after the settle > 30 deg',
                               fail_belly='fail or belly flag (lowest hull point > 0.05 m under the undisturbed surface for > 1 s)',
                               unsafe_belly='unsafe or belly flag', cluster='arena : nearest TerrainMap feature to the start-goal midpoint'),
                   summary=summary, rows=rows, wall_s=round(time.time() - t0, 1)), open(out, 'w'), indent=None, default=float)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
