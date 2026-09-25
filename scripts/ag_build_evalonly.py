#!/usr/bin/env python3
"""Evaluation-only row files from suite drives (arena_gator_20260925, module E6a; REVIEW_R1 5c; NOTES_E4 problem 3).

The designed-route drives on the unseen test groups (rigid: rigid_hmmwv_v1 near rows, rigid_v2 shards 2000-2124
spread rows) and, when they exist, drives on held-out / dev groups carry suite ids, so every training builder refuses
them. This tool builds the same trainer-format file for them, for OFFLINE SCORING ONLY:
  - it runs the unchanged per-episode tools of scripts/ag_build_ds.py on exactly the selected episodes (its inspect()
    checks, the map/arena check, f104_n2_dataset.py, n2_reanchor_dataset.py, then ag_build_ds.ci_one_world);
  - suite ids are accepted ONLY with --allow-suite-ids and ONLY when --out lies in the evaluation-only folder
    (K3/e4/evalonly/ locally, G3/e4/evalonly/ on the cluster); ga_build_mixed.blacklisted is disabled for this process
    only; nothing else is changed;
  - every row's split is set to 'evalonly' (the case split is kept in 'case_split'), so ci_train.py cannot fit a
    single row of these files (it fits split == 'train' only); every group must be a suite group (the opposite of the
    training builders' check), and a marker file EVAL_ONLY_NOT_FOR_TRAINING.txt is written next to the file;
  - selection: run folders (--runs), ids matching --id-regex (default <arena>_(test|heldout|dev)_group_NNNN_route_NN,
    i.e. designed routes), optionally only rows of --tasks with kind in --kinds (default test_designed) and this arena;
    incomplete runs are left out and counted (a partial collection gives a partial file; rebuild when complete).
Output <out>/ci_<stem>.npz (+ station / re-anchor intermediates, <stem>_record.json, logs/). numpy only: it can also run
on the cluster login node from a copy in G3/tools with PYTHONPATH=$G3/tools:$G3/source/scripts:$G3/source/src (the
dataset tools and the map check then come from G3/source; --out under G3/e4/evalonly).

  PYTHONPATH=src:scripts python scripts/ag_build_evalonly.py --arena g268 --world rigid --runs <G3 or local>/rigid_v1/runs \
      --tasks $K3/e3/tasks/rigid_v2.json --map-root $K3/map_roots/g268 --out $K3/e4/evalonly/g268_rigid_test --allow-suite-ids
"""
import argparse, json, os, re, shutil, sys, time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ag_build_ds as B                   # noqa: E402  (unchanged; its functions are reused)
import ag_blacklist                       # noqa: E402
import ag_map_check                       # noqa: E402
import ag_tasklib as L                    # noqa: E402

EVAL_DIRS = [str((L.K3 / 'e4' / 'evalonly').resolve()), L.G3 + '/e4/evalonly']
TOOLS = Path(B.__file__).resolve().parent     # the unchanged dataset tools next to ag_build_ds.py (scripts/, or G3/source/scripts)


def in_eval_dir(out):
    o = str(Path(out).resolve())
    return any(o == d or o.startswith(d + '/') for d in EVAL_DIRS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arena', required=True)
    ap.add_argument('--world', choices=['rigid', 'crm'], required=True)
    ap.add_argument('--vehicle', choices=['hmmwv', 'gator'], default='hmmwv')
    ap.add_argument('--runs', nargs='+', required=True)
    ap.add_argument('--map-root', required=True)
    ap.add_argument('--source-root', default=str(TOOLS.parent), help='root holding assets/traverse/<arena> (map check)')
    ap.add_argument('--tasks', nargs='*', default=[])
    ap.add_argument('--kinds', default='test_designed')
    ap.add_argument('--id-regex', default=None)
    ap.add_argument('--out', required=True); ap.add_argument('--stem', default=None)
    ap.add_argument('--allow-suite-ids', action='store_true')
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--keep-work', action='store_true')
    a = ap.parse_args(argv)
    t0 = time.time()
    out = Path(a.out)
    if not in_eval_dir(out):
        raise SystemExit(f'{out} is not inside an evaluation-only folder {EVAL_DIRS}')
    if not a.allow_suite_ids:
        raise SystemExit('suite ids are only accepted with --allow-suite-ids (evaluation-only output)')
    out.mkdir(parents=True, exist_ok=True); (out / 'logs').mkdir(exist_ok=True)
    stem = a.stem or f'{a.arena}_{a.vehicle}_{a.world}_evalonly'
    prefix = B.VEHICLE_PREFIX[a.vehicle]
    pat = re.compile(a.id_regex or rf'^{re.escape(prefix)}({re.escape(a.arena)}_(test|heldout|dev)_group_\d{{4}}_route_\d{{2}})$')
    kinds = {k for k in a.kinds.split(',') if k}
    trows = B.load_task_rows(a.tasks) if a.tasks else None
    cnt, seen = Counter(), {}
    for r in a.runs:
        for e in os.scandir(r):
            if not e.is_dir():
                continue
            cnt['dirs'] += 1
            if not pat.match(e.name):
                cnt['other_ids'] += 1; continue
            if trows is not None:
                t = trows.get(e.name)
                if t is None or t.get('kind') not in kinds or t.get('arena') != a.arena or t.get('vehicle', 'hmmwv') != a.vehicle:
                    cnt['not_a_selected_task_row'] += 1; continue
            assert e.name not in seen, f'{e.name} in two run folders'
            seen[e.name] = os.path.join(r, e.name)
    with ProcessPoolExecutor(a.workers) as ex:
        info = list(ex.map(B.inspect, [(d, a.world, True) for d in seen.values()], chunksize=64))
    rejected = Counter(i['reason'] for i in info if not i['ok'])
    ok = [i for i in info if i['ok']]
    assert ok, f'nothing selected ({dict(cnt)}, rejected {dict(rejected)})'
    vbad = [i['id'] for i in ok if (i['vehicle'] != 'gator' if a.vehicle == 'gator' else i['vehicle'] is not None)]
    assert not vbad, f'vehicle block mismatch: {vbad[:5]}'
    arenas = Counter(B.arena_short(i['arena']) for i in ok)
    assert set(arenas) == {a.arena}, f'runs name other arenas: {dict(arenas)}'
    notsuite = [i['id'] for i in ok if not ag_blacklist.is_suite(i['group'] or '')]
    assert not notsuite, f'evaluation-only files take suite groups only; not suite groups: {notsuite[:5]}'
    mc = ag_map_check.check(a.map_root, [i['case'] for i in ok], a.source_root)
    if not mc['ok']:
        raise SystemExit(f'map/arena check failed: {mc["problems"]}')
    Lk = out / 'work' / f'links_{stem}'
    if Lk.exists():
        shutil.rmtree(Lk)
    Lk.mkdir(parents=True)
    for i in ok:
        os.symlink(os.path.abspath(seen[i['id']]), Lk / i['id'])
    env = dict(os.environ); env['PYTHONPATH'] = os.pathsep.join([str(TOOLS.parent / 'src'), str(TOOLS)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    env.setdefault('OMP_NUM_THREADS', '1')
    st = out / f'station_{stem}.npz'
    rc, secs = B.run_logged([sys.executable, '-u', TOOLS / 'f104_n2_dataset.py', '--root', a.map_root, '--runs', f'{Lk}/*_route_*:designed',
                             '--out', st, '--workers', a.workers], out / 'logs' / f'station_{stem}.log', env)
    assert rc == 0, 'f104_n2_dataset.py failed, see logs'
    sid = np.load(st, allow_pickle=True)['id'].astype(str)
    assert set(sid) == {i['id'] for i in ok}, f'station rows {len(sid)} != selected {len(ok)}'
    ra = out / f'reanchor_{stem}.npz'
    rc2, secs2 = B.run_logged([sys.executable, '-u', TOOLS / 'n2_reanchor_dataset.py', '--root', a.map_root, '--ids', st, '--runs', Lk, '--out', ra,
                               '--workers', a.workers], out / 'logs' / f'reanchor_{stem}.log', env)
    assert rc2 == 0, 'n2_reanchor_dataset.py failed, see logs'
    ci = out / f'ci_{stem}.npz'

    def extra_cols(d):
        n = len(d['id'])
        return dict(arena=np.array([a.arena] * n, object), vehicle=np.array([a.vehicle] * n, object), tier=np.full(n, -1, np.int16),
                    case_split=np.asarray(d['split'], object).copy(), split=np.array(['evalonly'] * n, object))

    orig = B.GBM.blacklisted
    B.GBM.blacklisted = lambda s: False          # this process only: the rows ARE suite rows, by construction
    try:
        r = B.ci_one_world(ra, a.world, [Lk], ci, extra_cols, a.workers)
    finally:
        B.GBM.blacklisted = orig
    assert not r['missing_episodes'] and r['raw_found_fraction'] == 1.0, r
    z = np.load(ci, allow_pickle=True)
    ids = z['id'].astype(str); grp = z['group'].astype(str); sp = z['split'].astype(str)
    assert set(sp) == {'evalonly'} and len(set(ids)) == len(ids)
    assert all(ag_blacklist.is_suite(g) for g in set(grp)), 'a non-suite group in an evaluation-only file'
    rec = dict(tool='scripts/ag_build_evalonly.py', tool_sha256=B.sha256_file(__file__), argv=sys.argv[1:] if argv is None else argv,
               started=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t0)), host=os.uname().nodename, arena=a.arena, world=a.world, vehicle=a.vehicle,
               map_root=os.path.abspath(a.map_root), map_check={k: v for k, v in mc.items()}, runs=[os.path.abspath(r) for r in a.runs],
               tasks={p: B.sha256_file(p) for p in a.tasks}, kinds=sorted(kinds), selection=dict(counts=dict(cnt), candidates=len(seen), rejected=dict(rejected),
               rejected_ids=[(i['id'], i['reason']) for i in info if not i['ok']][:200], selected=len(ok), groups=len({i['group'] for i in ok}),
               status=dict(Counter(i['status'] for i in ok))),
               stages=dict(station=dict(rows=int(len(sid)), secs=secs), reanchor=dict(secs=secs2), ci=r),
               output=dict(path=str(ci), sha256=B.sha256_file(ci), rows=int(len(ids)), startup_rows=int((z['anchor_frame'].astype(int) == 0).sum()),
                           groups=int(len(set(grp))), case_split=dict(Counter(z['case_split'].astype(str)))),
               tool_sha256_used={t: B.sha256_file(TOOLS / t) for t in ('ag_build_ds.py', 'f104_n2_dataset.py', 'n2_reanchor_dataset.py', 'ga_build_mixed.py')},
               elapsed_s=round(time.time() - t0, 1))
    json.dump(rec, open(out / f'{stem}_record.json', 'w'), indent=1, default=str)
    (out / 'EVAL_ONLY_NOT_FOR_TRAINING.txt').write_text('Evaluation-only rows (suite groups, split = evalonly). Never pass these files to a trainer.\n')
    if not a.keep_work:
        shutil.rmtree(Lk, ignore_errors=True)
    print(json.dumps(dict(out=str(ci), rows=rec['output']['rows'], startup_rows=rec['output']['startup_rows'], groups=rec['output']['groups'],
                          selected=len(ok), rejected=dict(rejected), counts=dict(cnt), secs=rec['elapsed_s'])), flush=True)


if __name__ == '__main__':
    main()
