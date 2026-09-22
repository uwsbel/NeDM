#!/usr/bin/env python3
"""Cluster-side check of the rigid A4 / B3 and CRM B3 task files (login node, read-only, python3 stdlib).
For every row: case/route (and the recorded run dir for A4) exist; their sha256 equal the row's recorded hashes
(A4: also equal the recording's outcome.json hashes, frames > F); the recordings' frozen-source hashes are one set."""
import hashlib, json, sys
from collections import Counter
from pathlib import Path
G = Path('/work1/dannegrut/harry/experiments/generalist_20260921')
R = Path('/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909')
C = Path('/work1/dannegrut/harry/experiments/crm_f104_20260916')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
gman = json.load(open(G / 'source/source_manifest.json'))['files']
v1man = json.load(open(R / 'source_v1/source_manifest.json'))['files']
report, misses = {}, []

rows = json.load(open(G / 'tasks/tasks_a4_rigid.json'))
src_sets, statuses = Counter(), Counter()
for t in rows:
    ex = dict(zip(t['extra'][0::2], t['extra'][1::2]))
    rec = Path(ex['--recorded'])
    for p in (t['case'], t['route'], rec / 'trajectory.npz', rec / 'outcome.json', rec / 'episode_complete.json', rec / 'collection_request.json'):
        if not Path(p).exists():
            misses.append((t['id'], 'missing', str(p)))
    if misses and misses[-1][0] == t['id']:
        continue
    o = json.load(open(rec / 'outcome.json'))
    cq = json.load(open(rec / 'collection_request.json'))
    if not (sha(t['case']) == t['case_sha256'] == o['case_sha256']):
        misses.append((t['id'], 'case sha', t['case']))
    if not (sha(t['route']) == t['route_sha256'] == o['route_sha256']):
        misses.append((t['id'], 'route sha', t['route']))
    if not (int(o['frames']) == t['recorded_frames'] > int(ex['--branch-frame'])):
        misses.append((t['id'], 'frames', f"{o['frames']} vs {t['recorded_frames']} F={ex['--branch-frame']}"))
    if o['status'] != t['recorded_status']:
        misses.append((t['id'], 'status', o['status']))
    src_sets[json.dumps(cq['source_sha256'], sort_keys=True)] += 1
    statuses[o['status']] += 1
assert len(src_sets) == 1, f'{len(src_sets)} distinct frozen-source sets among the recordings'
rec_src = json.loads(next(iter(src_sets)))
report['a4'] = {'rows': len(rows), 'recorded_status': dict(statuses), 'distinct_recorded_source_sets': len(src_sets),
                'recorded_source_equals_source_v1_manifest': all(v1man[k] == v for k, v in rec_src.items()),
                'G_source_differs_from_recorded_source': sorted(k for k, v in rec_src.items() if gman[k] != v)}

for name, root in (('tasks_b3_rigid.json', R), ('tasks_b3_crm.json', C)):
    rows = json.load(open(G / 'tasks' / name)); n_ok = 0
    for t in rows:
        case, route = (Path(t['case']) if t['case'].startswith('/') else root / t['case']), (Path(t['route']) if t['route'].startswith('/') else root / t['route'])
        if not case.exists() or not route.exists():
            misses.append((t['id'], 'missing', f'{case} | {route}')); continue
        if sha(case) != t['case_sha256'] or sha(route) != t['route_sha256']:
            misses.append((t['id'], 'sha', f'{case} | {route}')); continue
        if json.load(open(case))['split'] != 'train':
            misses.append((t['id'], 'split', str(case))); continue
        n_ok += 1
    report[name] = {'rows': len(rows), 'ok': n_ok}
report['config'] = {'C/configs/crm_main.json': (C / 'configs/crm_main.json').exists(), 'fingerprint': (R / 'pilot_runtime_412394.json').exists(),
                    'chrono_data_hmmwv': Path('/work1/dannegrut/harry/nrd/chrono-build/data/vehicle/hmmwv').is_dir(),
                    'crm_collect_ext': (G / 'source/scripts/crm_collect_ext.py').exists(), 'crm_sbatch_C': (C / 'source/scripts/crm_collect.sbatch').exists()}
report['misses'] = misses[:20]; report['n_misses'] = len(misses)
print(json.dumps(report, indent=1))
sys.exit(1 if misses else 0)
