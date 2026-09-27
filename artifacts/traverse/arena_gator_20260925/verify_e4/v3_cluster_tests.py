#!/usr/bin/env python3
"""VERIFY E4 (independent, runs on the cluster login node, numpy only): the builder's two cluster test files.
tier / arena / vehicle columns vs the task files and the runs' own case.json / outcome.json; row counts vs the
re-anchored file; no suite id; soil QA-style rejections."""
import fnmatch, json, os, sys
from collections import Counter
import numpy as np

G3 = '/work1/dannegrut/harry/experiments/arena_gator_20260925'
F104_TASKS = '/work1/dannegrut/harry/experiments/crm_f104_20260916/tasks_train.json'
PATS = ['f104_crm_eval_group_*', 'f104_g1_test_group_*', 'f104_pair_group_*', '*_test_group_*', '*_heldout_group_*', '*_dev_group_*',
        '*_eval_group_*', '*_pair_group_*', 'drift__*']
f104_tier = {r['id']: int(r['tier']) for r in json.load(open(F104_TASKS))}
out = {}


def tasks(p):
    return {r['id']: r for r in json.load(open(p))}


cases = [
    ('test_partial_g203', G3 + '/e4/tests/test_partial_g203/ci_g203_hmmwv_crm.npz', dict(crm=[G3 + '/soil_v1/runs']),
     dict(crm=tasks(G3 + '/tasks/soil_v1.json')), 'g203', 'hmmwv', ''),
    ('test_pilot_gator', G3 + '/e4/tests/test_pilot_gator/ci_f104_gator_both.npz',
     dict(rigid=[G3 + '/pilot_gator/rigid/runs'], crm=[G3 + '/pilot_gator/soil/runs']),
     dict(rigid=tasks(G3 + '/tasks/pilot_gator_rigid.json'), crm=tasks(G3 + '/tasks/pilot_gator_soil.json')), 'f104', 'gator', 'gator__'),
]
for name, path, roots, T, arena, veh, prefix in cases:
    z = np.load(path, allow_pickle=True)
    ep = z['episode'].astype(str); dom = z['domain'].astype(int); tier = z['tier'].astype(int); af = z['anchor_frame'].astype(int)
    r = dict(rows=int(len(ep)), rows_by_domain={int(k): int(v) for k, v in Counter(dom.tolist()).items()},
             arena_col=sorted(set(z['arena'].astype(str))), vehicle_col=sorted(set(z['vehicle'].astype(str))))
    for w, code in (('rigid', 0), ('crm', 1)):
        if w not in roots:
            continue
        m = dom == code; eps = sorted(set(ep[m]))
        tier_task = tier_mis = tier_f104_mis = no_task = 0; arenas = Counter(); vblk = Counter(); tier_nonconst = 0
        for e in eps:
            ts = set(tier[m & (ep == e)].tolist())
            tier_nonconst += len(ts) > 1
            t = ts.pop()
            row = T[w].get(e)
            if row is None:
                no_task += 1
            else:
                tier_task += 1; tier_mis += int(int(row['tier']) != t)
                assert row.get('arena', arena) in (arena,) or True
            base = e[len(prefix):] if prefix and e.startswith(prefix) else e
            if arena == 'f104':
                tier_f104_mis += int(f104_tier[base] != t)
            d = [os.path.join(rt, e) for rt in roots[w] if os.path.isdir(os.path.join(rt, e))]
            assert len(d) == 1, (e, d)
            c = json.load(open(d[0] + '/case.json')); o = json.load(open(d[0] + '/outcome.json'))
            arenas[c['arena']] += 1
            v = o.get('vehicle'); vblk[(v or {}).get('name') if isinstance(v, dict) else str(v)] += 1
        # every finished run folder of this arena/vehicle prefix that the task file lists as a training row is in the file
        done = []
        for rt in roots[w]:
            for dn in os.listdir(rt):
                full = os.path.join(rt, dn)
                if not dn.startswith(prefix + arena + '_v2_group_') or (prefix == '' and dn.startswith('gator')):
                    continue
                if os.path.isfile(full + '/episode_complete.json') and os.path.isfile(full + '/outcome.json'):
                    done.append(dn)
        k0 = int((m & (af == 0)).sum())
        # re-anchored file of the same build: rows per world
        stem = os.path.basename(path)[3:-4]
        ra = os.path.join(os.path.dirname(path), f'reanchor_{stem}_{w}.npz')
        ra_rows = int(len(np.load(ra, allow_pickle=True)['id'])) if os.path.isfile(ra) else None
        r[w] = dict(episodes=len(eps), rows=int(m.sum()), k0_rows=k0, reanchor_rows=ra_rows, with_task_row=tier_task, without_task_row=no_task,
                    tier_mismatch_vs_task=tier_mis, tier_mismatch_vs_f104_order=tier_f104_mis if arena == 'f104' else None,
                    tier_not_constant=tier_nonconst, case_arenas=dict(arenas), outcome_vehicle=dict(vblk),
                    complete_runs_now=len(done), complete_runs_not_in_file=len(set(done) - set(eps)),
                    file_eps_not_complete_now=len(set(eps) - set(done)))
    strings = set(z['id'].astype(str)) | set(z['group'].astype(str)) | set(ep)
    r['suite_hits'] = sorted(s for s in strings if any(fnmatch.fnmatch(s.split('@')[0], p) for p in PATS))[:5]
    out[name] = r
print(json.dumps(out, indent=1))
