#!/usr/bin/env python3
"""Outcome index of evaluation drives (arena_gator_20260925, module E6a): one row per (world, vehicle, group, arm).

Inputs: the mapping files written by scripts/ag_eval_tasks.py build (<tasks>.mapping.json: group -> arm -> run id, incl.
rows reused from another task file) and one or more run folders (--runs; the first folder holding <run id> with
episode_complete.json, or outcome.json + trajectory.npz, wins; e.g. local syncs of G3/rigid_eval/runs and
G3/soil_v1/runs, or ag_eval_tasks.py smoke outputs).

Per drive (labels of scripts/ga_analyze.py / scripts/f104_n2_analyze.labels, after the 1 s settle):
  fail           goal not reached
  unsafe         not (goal and < 0.05 s rolling backwards under throttle and min forward speed > -0.30 m/s)
  unsafe_noback  unsafe without the backward-motion clauses = fail (the NOTES_E3b1 definition); kept as its own column
  backward_only  unsafe and not fail (the part of unsafe that only the backward clauses add)
  tilt30         max |roll| or |pitch| after the settle > 30 deg
  elapsed_s, status, positive work, the pick's predicted P and route logit (from the mapping)
Per group: evaluation stratum; start/goal midpoint -> nearest terrain feature of its arena (TerrainMap.features, the
Chrono frame, as scripts/n2_cluster_ci.py) -> cluster '<arena>:<feature index>' and the feature kind; the case's design
feature (routes/<g>/route_00.json meta feature_index, same index order, the rule of suites/*groups_per_feature) ->
cluster_design (a robustness clustering; the two agree on 88 % of the unseen soil subset groups); arena role
(near / spread test, training arena, dev), set (unseen / indist_f104 / heldout / dev / f104_suite), map-lookup error
and distance to the nearest training arena (arenas/map_lookup_error.json).
Missing drives are rows with missing=true (no outcome fields).

  PYTHONPATH=src:scripts python scripts/ag_eval_index.py --mapping $K3/e6/tasks/rigid_eval_v1.json.mapping.json \
      --runs /tmp/ag_eval/rigid_runs --out $K3/e6/index/rigid_eval_v1.json
"""
import argparse, hashlib, json, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / 'src'))
import ag_tasklib as L                    # noqa: E402
from ga_analyze import safe_labels        # noqa: E402

NEAR = ('g260', 'g271', 'g251', 'g247')
SPREAD = ('g258', 'g268', 'g263', 'g241')
TRAIN = ('f104', 'g203', 'g228')
DEV = ('g217',)
INDIST = L.K3 / 'suites/f104_indist_200.json'
MAPERR = L.K3 / 'arenas/map_lookup_error.json'
G3 = L.G3


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def local_of(p):
    rel = p[len(G3) + 1:] if p.startswith(G3 + '/') else p
    if rel.startswith('/'):
        return Path(rel)
    if rel.startswith('ext/'):
        return ROOT / rel[4:]
    return L.K3 / rel


def arena_role(a):
    return 'near' if a in NEAR else 'spread' if a in SPREAD else 'training' if a in TRAIN else 'dev' if a in DEV else 'other'


def group_set(g, arena, indist):
    if '_test_group_' in g:
        return 'unseen'
    if g in indist:
        return 'indist_f104'
    if g.startswith('f104_pair_group_') or g.startswith('f104_crm_eval_group_'):
        return 'f104_suite'
    if '_heldout_group_' in g:
        return 'heldout'
    if '_dev_group_' in g:
        return 'dev'
    return 'other'


_FEAT = {}


def features(arena_dir):
    if arena_dir not in _FEAT:
        from nedm.traverse.terrain import TerrainMap
        fts = TerrainMap.from_dir(ROOT / arena_dir).features
        _FEAT[arena_dir] = (np.array([[f['x_m'], f['y_m']] for f in fts], float), [f.get('kind', f.get('type', '?')) for f in fts])
    return _FEAT[arena_dir]


def case_info(case_path):
    c = json.load(open(case_path))
    s = np.asarray(c['layout']['start_xy'], float); e = np.asarray(c['goal_xy'], float); mid = (s + e) / 2
    xy, kinds = features(c['arena'])
    d = np.linalg.norm(xy - mid[None], axis=1); i = int(np.argmin(d))
    r0 = Path(case_path).parent / 'routes' / c['id'] / 'route_00.json'
    fi = json.load(open(r0))['meta'].get('feature_index') if r0.exists() else None
    return dict(stratum=c.get('evaluation_stratum'), arena_dir=c['arena'], feature=i, feature_kind=kinds[i], feature_dist_m=float(d[i]),
                design_feature=None if fi is None else int(fi), start_xy=s.tolist(), goal_xy=e.tolist())


def find_run(run_id, roots):
    for r in roots:
        d = Path(r) / run_id
        if (d / 'episode_complete.json').exists() or ((d / 'outcome.json').exists() and (d / 'trajectory.npz').exists()):
            return d
    return None


def outcome(d):
    o = json.load(open(d / 'outcome.json'))
    lab = safe_labels(str(d))
    unsafe, fail = int(lab['unsafe']), int(lab['fail'])
    return dict(status=lab['status'], fail=fail, unsafe=unsafe, unsafe_noback=fail, backward_only=int(unsafe and not fail),
                tilt30=int(lab['max_tilt'] > 30.0), max_tilt_deg=float(lab['max_tilt']), back_s=float(lab['back_s']), min_vx=float(lab['min_vx']),
                elapsed=float(lab['elapsed']), work_kj=o.get('positive_work_kj'), vehicle_block=(o.get('vehicle') or {}).get('name') if isinstance(o.get('vehicle'), dict) else o.get('vehicle'),
                complete_marker=(d / 'episode_complete.json').exists())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mapping', nargs='+', required=True)
    ap.add_argument('--runs', nargs='+', required=True, help='run folders (runs/<id>), searched in order')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    indist = set(json.load(open(INDIST))['groups'])
    me = json.load(open(MAPERR))['arenas']
    rows, meta_in, t0 = [], [], time.time()
    cinfo = {}
    for mp in a.mapping:
        m = json.load(open(mp))
        world = m['world']
        meta_in.append(dict(mapping=str(mp), sha256=sha256_file(mp), world=world, tasks=m.get('tasks'), tasks_sha256=m.get('tasks_sha256'),
                            arms=[dict(name=x['name'], vehicle=x['vehicle'], modes=x['modes'], model_tags=x['model_tags'], arenas=x['arenas']) for x in m['arms']]))
        arm_meta = {x['name']: x for x in m['arms']}
        for g, arms in sorted(m['groups'].items()):
            for name, e in arms.items():
                key = e['case']
                if key not in cinfo:
                    cinfo[key] = case_info(local_of(key))
                ci = cinfo[key]
                arena = e['arena']
                mv = me.get(arena, {})
                row = dict(world=world, vehicle=e['vehicle'], arena=arena, role=arena_role(arena), set=group_set(g, arena, indist), group=g,
                           arm=name, mode=e.get('mode') or arm_meta[name]['modes'][0], model_tag=(arm_meta[name]['model_tags'] or [None])[0],
                           run_id=e['run_id'], reused_from=e.get('reused_from'), route_sha256=e['route_sha256'], P=e.get('P'), z=e.get('z'),
                           stratum=ci['stratum'], feature=ci['feature'], feature_kind=ci['feature_kind'], cluster=f"{arena}:{ci['feature']}",
                           design_feature=ci['design_feature'],
                           cluster_design=f"{arena}:{ci['design_feature'] if ci['design_feature'] is not None else 'none'}",
                           map_err_rmse_m=(mv.get('all') or {}).get('rmse'), dist_nearest_training=mv.get('distance_nearest_training'),
                           nearest_training_arena=mv.get('nearest_training_arena'), dist_f104=mv.get('distance_f104'))
                d = find_run(e['run_id'], a.runs)
                if d is None:
                    row['missing'] = True
                else:
                    row.update(outcome(d), missing=False, run_dir=str(d))
                    vb = row['vehicle_block']
                    assert (vb == 'gator') == (e['vehicle'] == 'gator'), f"{e['run_id']}: vehicle block {vb!r} but the row asked for {e['vehicle']}"
                rows.append(row)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    have = [r for r in rows if not r['missing']]
    summary = dict(rows=len(rows), driven=len(have), missing=len(rows) - len(have),
                   by_world_arm={f"{r['world']}|{r['vehicle']}|{r['arm']}": v for r, v in []},
                   missing_by_arm=dict(Counter(f"{r['world']}|{r['vehicle']}|{r['arm']}" for r in rows if r['missing'])),
                   driven_by_arm=dict(Counter(f"{r['world']}|{r['vehicle']}|{r['arm']}" for r in have)),
                   sets=dict(Counter(r['set'] for r in rows)), arenas=dict(Counter(r['arena'] for r in rows)))
    summary.pop('by_world_arm')
    json.dump(dict(schema='ag_eval_index_v1', tool='scripts/ag_eval_index.py', tool_sha256=sha256_file(__file__),
                   created=time.strftime('%Y-%m-%d %H:%M:%S'), inputs=meta_in, runs=[str(r) for r in a.runs],
                   labels=dict(fail='goal not reached', unsafe='not (goal and back_s < 0.05 and min_vx > -0.30), after the 1 s settle',
                               unsafe_noback='fail (unsafe without the backward-motion clauses)', backward_only='unsafe and not fail',
                               tilt30='max |roll|,|pitch| after the settle > 30 deg', cluster='arena : nearest TerrainMap feature to the start-goal midpoint', cluster_design='arena : the case design feature (route_00 meta feature_index)'),
                   summary=summary, rows=rows, wall_s=round(time.time() - t0, 1)), open(out, 'w'), indent=None, default=float)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
