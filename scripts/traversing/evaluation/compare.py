"""Compare one arm of a run folder with the released record; only DONE runs count.

    PYTHONPATH=src python scripts/traversing/evaluation/compare.py --out OUT --arm NAME [--task ID ...] \\
        [--released 'data:artifacts/traverse/crm_improve_20260922/s4/runs/{task}__L0p5_HnG_B'] \\
        [--csv traversing/results/m2_shared_risk_soil.csv [--column NAME] [--candidate DIR]] \\
        [--picks data:.../picks_crm_HnG] [--decisions data:.../poses_crm_all.json] [--json FILE]
--released (data: or absolute template): every npz of the run against the reference's, key by key (dtype and values;
state, action, pose required), desired speeds vs command_reference.npz, branch pose, status, frames, work; files the
reference lacks are listed unchecked; a reference without outcome.json (a drive the study shared with another arm) is
counted as missing: compare it with --task and that arm's folder. --csv: codes vs the table (unlabelled cells:
labels.ON_MISSING): a wide table's column --column (default: the arm name), or a long table's (arm column; m1, m3)
outcome_code of the rows of arm --column and of this ground; --candidate (wide tables): traversing/results with the
column replaced, for recount_milestones.py --results. --picks: pick.json (and a gradient pick's B) vs
picks/<task>.json. --decisions: the run's own pass-1 decision state vs the released one.
"""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

from nedm.traversing.evaluation.config import REPO_ROOT, Env, ReleaseError, write_atomic
from nedm.traversing.evaluation.labels import ON_MISSING
from nedm.traversing.evaluation.runner import Record, cells, load_run

CHECKS = ('status_equal', 'frames_equal', 'work_equal', 'code_equal', 'pick_equal', 'B_equal', 'decision_equal')


def table_codes(path, column, ground):
    """{task: code} of one arm in a results CSV: a wide table keyed by its first column, or a long table (an arm
    column) keyed by its *_id column, rows of arm `column` (and of this ground where the table has a world column)."""
    rows = list(csv.DictReader(open(path)))
    head = list(rows[0])
    if 'arm' not in head:
        if column not in head:
            raise SystemExit(f'{path}: no column {column!r}; give --column (one of {head[1:]})')
        return {r[head[0]]: r[column] for r in rows}
    world, key = {'soil': 'crm'}.get(ground, ground), next(h for h in head if h.endswith('_id'))
    out = {r[key]: r['outcome_code'] for r in rows if r['arm'] == column and r.get('world', world) == world}
    if not out:
        raise SystemExit(f'{path}: no rows of arm {column!r} on {world}; give --column (arms: '
                         f'{sorted({r["arm"] for r in rows})})')
    return out


def npz(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def released(env, ref, d, rec):
    def file(name):
        try:
            return env.file(f'{ref}/{name}')
        except (ReleaseError, OSError):
            return None
    if (o := file('outcome.json')) is None:                 # the study shared some drives between arms
        return dict(reference='missing')
    o, same, unchecked = json.loads(o.read_text()), {}, []
    for p in sorted(d.glob('*.npz')):
        if (q := file(p.name)) is None:
            unchecked.append(p.name)
            continue
        x, y = npz(p), npz(q)
        same |= {f'{p.stem}.{k}': x[k].dtype == y[k].dtype and np.array_equal(x[k], y[k]) for k in x if k in y}
        if p.stem == 'trajectory' and rec['branch']:
            same['branch_pose'] = np.array_equal(y['branch_pose'], rec['branch']['pose'])
        if p.stem == 'trajectory' and (c := file('command_reference.npz')) is not None:
            same['desired_speed_mps'] = np.array_equal(npz(c)['desired_speed_mps'], x['desired_speed_mps'])
    core = {'trajectory.state', 'trajectory.action', 'trajectory.pose'}
    return dict(bitwise=all(same.values()) and core <= set(same), differ=[k for k, v in same.items() if not v],
                unchecked=unchecked + ([] if 'desired_speed_mps' in same else ['command_reference.npz']),
                status_equal=o['status'] == rec['status'], frames_equal=o['frames'] == rec['frames'],
                work_equal=o['positive_work_kj'] == rec['positive_work_kj'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ('--out', '--arm', '--released', '--csv', '--column', '--candidate', '--picks', '--decisions', '--json'):
        p.add_argument(k, required=k in ('--out', '--arm'))
    p.add_argument('--task', action='append', help='compare these tasks only (repeatable)')
    a = p.parse_args()
    env, out = Env.from_environ(), Path(a.out)
    arms, tasks, _ = load_run(out, env)
    if (arm := {x.cfg.name: x for x in arms}.get(a.arm)) is None:
        p.error(f'--arm {a.arm}: the arms of {out} are {[x.cfg.name for x in arms]}')
    col = a.column or a.arm
    ref = table_codes(a.csv, col, arm.cfg.ground) if a.csv else {}
    gap = ON_MISSING.get(Path(a.csv).stem, '-').replace('drop_pair', '-') if a.csv else None     # unlabelled cells
    code = {t: c if c is None else c['code'] if isinstance(c, dict) else gap or c
            for t, c in ((t, x.get(a.arm)) for t, x in cells(out, env).items())}
    rows = []
    for t, task in ((t, x) for t, x in tasks.items() if not a.task or t in a.task):
        d, row = out / 'runs' / t / a.arm, dict(task=t, code=code[t])
        rows.append(row)
        if code[t] is None:
            continue
        rec = json.loads((d / 'record.json').read_text())
        row.update(status=rec['status'], frames=rec['frames'])
        if a.released:                                   # a run without trajectory (crash, ...) is not bitwise
            row.update(released(env, env.expand(a.released, task), d, rec))
        if a.csv:
            row.update(csv=ref.get(t), code_equal=ref.get(t) == code[t])        # a task the table lacks: unequal
        if a.picks and (d / 'pick.json').exists():
            pk = json.loads((d / 'pick.json').read_text())
            rel = json.loads(env.file(f'{a.picks}/picks/{t}.json', task).read_text())['arms']
            sel, B = rel[{'straight': 'S'}.get(pk['arm'], pk['arm'])], pk['record'].get('B')
            row.update(pick_equal=sel['route_sha256'] == pk['route_sha256'], z_mean=[pk['z_mean'], sel.get('z_mean')],
                       B_equal=B and rel['B']['route_sha256'] == B['route_sha256'],
                       abstained=pk['record'].get('grad', {}).get('abstained'))
        if a.decisions and (d / 'pass1/record.json').exists():
            from nedm.traversing.evaluation.planner import Decision
            mine, rel = Decision.after_approach(task, Record.load(d / 'pass1'), arm.cfg.F), Decision.from_release(
                task, a.decisions, env)
            row['decision_equal'] = all(np.array_equal(getattr(mine, k), getattr(rel, k)) for k in ('pose', 'hist',
                                                                                                     'hmask'))
    done = [r for r in rows if r['code'] is not None]
    summary = dict(arm=a.arm, tasks=len(rows), done=len(done), statuses=dict(Counter(r['status'] for r in done)))
    if a.released:
        n = [r for r in done if 'reference' not in r]
        summary.update(bitwise=f'{sum(r["bitwise"] for r in n)}/{len(n)}', reference_missing=len(done) - len(n),
                       unchecked=dict(Counter(f for r in n for f in r['unchecked'])))
    for k in CHECKS:
        if rs := [r[k] for r in rows if r.get(k) is not None]:
            summary[k] = f'{sum(rs)}/{len(rs)}'
    write_atomic(Path(a.json or out / f'compare_{a.arm}.json'), json.dumps(dict(summary=summary, rows=rows), indent=1))
    if a.candidate:
        table = list(csv.DictReader(open(a.csv)))
        if 'arm' in table[0]:
            raise SystemExit('--candidate replaces a wide table\'s column; a long table (m1, m3) also holds metrics')
        shutil.copytree(REPO_ROOT / 'traversing/results', a.candidate)
        with open(Path(a.candidate) / Path(a.csv).name, 'w', newline='') as f:
            w = csv.DictWriter(f, list(table[0]), lineterminator='\n')
            w.writeheader()
            w.writerows({**r, col: code.get(r[next(iter(r))]) or r[col]} for r in table)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
