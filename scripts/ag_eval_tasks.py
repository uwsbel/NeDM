#!/usr/bin/env python3
"""Evaluation drive rows from pick directories (arena_gator_20260925, module E6a; PLAN 2.3, 3, 7.1-7.8; REVIEW_R2 11).

  build   one world per call. Arms are --arm NAME=GLOB[@vehicle]: GLOB matches one or more ag_picks.py output
          directories (one per arena) of that world; vehicle hmmwv (default) or gator. For every (vehicle, group) the
          picks of all arms are collected; arms whose routes have the same content sha256 are driven ONCE (one row
          whose 'arms' lists them all; the mapping file resolves every arm to its row).
          Run ids: <v><group>__<first arm of the row> (v = '' for the HMMWV, 'gator__' for the Gator); episode_seed =
          md5(id)[:8] (salted as md5(id#k) if it collides with an --existing or earlier row; field episode_seed_salt);
          tier --tier (default -1). The vehicle is always explicit in 'extra':
            HMMWV  ['--vehicle', 'hmmwv']
            Gator  rigid ['--vehicle', 'gator', '--runtime-fingerprint', G3/runtime/gator_runtime_fingerprint.json]
                   soil  ['--vehicle', 'gator']  (the soil_v2 Gator row form: the frozen soil dispatcher ag_crm_collect.py
                         has no fingerprint gate and stops with ValueError on --runtime-fingerprint; VERIFY_E6a fix)
          so the collector must be the dispatching wrapper (rigid: ag_gen_collect_ext.py; soil: ag_crm_collect.py, whose
          name contains crm_collect so crm_worker forwards --crm-config). NEDM_VEHICLE must stay unset.
          rigid: every row of a group (every arm and both vehicles) in ONE shard, so they share a node (Chrono rigid is
                 deterministic per node only); groups sorted by md5(group id), consecutive blocks of --groups-per-shard,
                 shard = --shard-base + block. Paths are absolute G3 paths (so the rows run from G3/source or from the
                 second tree G3/r2/source, whose allowlist lists the spread arenas). Existing rigid runs are never
                 reused (a reference drive on another node is not a paired drive).
          soil:  the soil_v2.json row format (paths relative to G3, kind 'eval', vehicle, arms, sha256). With
                 --existing <task files> a route that is already a row there (same group, vehicle, case and route
                 content sha256, e.g. the spread headroom straight 6 m/s drives <g>__straight6) is NOT re-driven: the
                 mapping points to that id (soil drives are identical across GPU types, NOTES_E3b2 section 2); an id
                 clash with different content stops the build. --no-reuse re-drives everything under new ids.
          Writes <out> (rows), <out>.mapping.json ({group: {arm: {run_id, vehicle, route_sha256, reused_from}}} + arm
          specs), <out>.staging.tsv (local file -> G3 path, sha256: every route and case file the rows name; files
          outside K3 go to G3/ext/<repo-relative path>), <out>.meta.json (counts, shards, collector contract).
  smoke   drive a few rigid rows locally (--local gates, local Chrono build, single thread each), with exactly the
          collector arguments the cluster runner gen_runner_g.py uses (+ --local; a Gator row's --runtime-fingerprint
          is dropped because the local gate is bypassed). Output <out>/runs/<id>, <out>/logs/<id>.log.
  stage   copy the staging list to the cluster (rsync --ignore-existing, new paths only) and verify every sha256 there;
          without --execute only a dry run and the remote hash check of what already exists.

  PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
  $PY scripts/ag_eval_tasks.py build --world rigid --arm M1a_free="$K3/e6/picks/rigid/*/M1a_free" \
      --arm M1a_fx2="$K3/e6/picks/rigid/*/M1a_fixed2" --arm straight2="$K3/e6/picks/rigid/*/straight2" --out $K3/e6/tasks/rigid_eval_v1.json
  $PY scripts/ag_eval_tasks.py smoke --tasks $K3/e6/tasks/rigid_eval_v1.json --ids <id,id,...> --out /tmp/ag_smoke --workers 6
  $PY scripts/ag_eval_tasks.py stage --tasks $K3/e6/tasks/rigid_eval_v1.json [--execute]
"""
import argparse, glob, hashlib, json, os, re, subprocess, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ag_tasklib as L                    # noqa: E402

G3 = L.G3
GATOR_FP = f'{G3}/runtime/gator_runtime_fingerprint.json'
VPREFIX = dict(hmmwv='', gator='gator__')
ARM_RE = re.compile(r'^[A-Za-z0-9](?:[A-Za-z0-9]|_(?!_))*$')        # no '__' inside an arm name
LOCAL_PY = '/usr/bin/python3.12'
LOCAL_CHRONO_PY = '/home/harry/chrono/build/bin'
LOCAL_CHRONO_DATA = '/home/harry/chrono/data'


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def md5hex(s):
    return hashlib.md5(s.encode()).hexdigest()


def route_content_sha(path):
    import numpy as np
    r = json.load(open(path))
    s = json.dumps({k: np.asarray(r[k], float).tolist() for k in ('waypoints', 'speeds', 'stations', 'headings')})
    return hashlib.sha256(s.encode()).hexdigest()


def g3_rel(local):
    """Local file -> path relative to G3: under K3 the same relative path, elsewhere ext/<repo-relative path>."""
    p = Path(local).resolve()
    try:
        return str(p.relative_to(L.K3.resolve()))
    except ValueError:
        return 'ext/' + str(p.relative_to(ROOT.resolve()))


def local_of(g3path):
    """G3 path (relative or absolute) -> local file."""
    rel = g3path[len(G3) + 1:] if g3path.startswith(G3 + '/') else g3path
    if rel.startswith('/'):
        return None
    if rel.startswith('ext/'):
        return ROOT / rel[4:]
    return L.K3 / rel


def parse_arm(spec):
    name, rest = spec.split('=', 1)
    vehicle = 'hmmwv'
    if '@' in rest:
        rest, vehicle = rest.rsplit('@', 1)
    assert ARM_RE.match(name), f'arm name {name!r}: letters, digits and single underscores only'
    assert vehicle in VPREFIX, vehicle
    dirs = sorted(d for d in glob.glob(rest) if (Path(d) / 'ag_picks.json').exists())
    assert dirs, f'arm {name}: no ag_picks.py directory matches {rest}'
    return name, vehicle, [Path(d) for d in dirs]


# ------------------------------------------------------------------------------------------------------------------
def build(a):
    arms, picks, man_by = [], {}, {}
    for spec in a.arm:
        name, vehicle, dirs = parse_arm(spec)
        assert name not in {x['name'] for x in arms}, f'arm {name} given twice'
        seen_arena = {}
        for d in dirs:
            m = json.load(open(d / 'ag_picks.json'))
            assert m['world'] == a.world, f'{d}: world {m["world"]} != --world {a.world}'
            assert m['arena'] not in seen_arena, f'arm {name}: two pick dirs for arena {m["arena"]}: {seen_arena[m["arena"]]} and {d}'
            lk = (d / 'PICKS_LOCKED.sha256').read_text().split()[0]
            assert lk == m['picks_locked_sha256'], f'{d}: PICKS_LOCKED changed since ag_picks wrote it'
            seen_arena[m['arena']] = str(d)
            man_by[str(d)] = m
            for g, e in m['picks'].items():
                if e is None:
                    continue
                pf = d / e['route_file']
                assert sha256_file(pf) == e['file_sha256'], f'{pf}: route file changed since the lock'
                case = ROOT / m['cases_dir'] / f'{g}.json'
                picks.setdefault((vehicle, g), {})[name] = dict(route=pf, case=case, sha=e['route_sha256'], arena=m['arena'],
                                                                 mode=m['mode'], model_tag=m.get('model_tag'), picks_dir=str(d),
                                                                 P=e.get('P'), z=e.get('z_mean'))
        arms.append(dict(name=name, vehicle=vehicle, dirs=[str(d) for d in dirs], arenas=sorted(seen_arena),
                         modes=sorted({man_by[str(d)]['mode'] for d in dirs}), model_tags=sorted({str(man_by[str(d)].get('model_tag')) for d in dirs}),
                         models=sorted({mm['sha256'] for d in dirs for mm in (man_by[str(d)].get('models') or [])})))
        assert len(arms[-1]['modes']) == 1, f'arm {name} mixes modes {arms[-1]["modes"]}'
    order = [x['name'] for x in arms]
    existing = {}
    for f in a.existing or []:
        for r in json.load(open(f)):
            existing[r['id']] = dict(r, _file=str(f))
    by_content = defaultdict(list)
    for r in existing.values():
        if r.get('sha256'):
            by_content[(r['group'], r.get('vehicle', 'hmmwv'), r['sha256'])].append(r)
    rows, mapping, reused, stage = [], defaultdict(dict), Counter(), {}
    used_seeds = {r['episode_seed'] for r in existing.values() if r.get('episode_seed') is not None}

    def new_seed(rid):
        # md5(id)[:8] (the task-file convention); 32-bit seeds of ~40k rows collide by chance, so a clash with an existing
        # row or an earlier new row is salted (md5(id + '#k')), recorded in the row; seeds are provenance only
        k, seed = 0, L.md5_int(rid)
        while seed in used_seeds:
            k += 1; seed = L.md5_int(f'{rid}#{k}')
        used_seeds.add(seed)
        return seed, k
    for (vehicle, g), pa in sorted(picks.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        cases = {str(v['case']) for v in pa.values()}
        assert len(cases) == 1, f'{g}: arms name different case files {cases}'
        arenas = {v['arena'] for v in pa.values()}
        assert len(arenas) == 1, (g, arenas)
        by_sha = {}
        for name in order:
            if name in pa:
                by_sha.setdefault(pa[name]['sha'], []).append(name)
        for sha, names in by_sha.items():
            v = pa[names[0]]
            rid = f'{VPREFIX[vehicle]}{g}__{names[0]}'
            case_rel = g3_rel(v['case']); route_rel = g3_rel(v['route'])
            # the fingerprint argument exists only in the rigid dispatcher (ag_gen_collect_ext.py); the soil dispatcher
            # (ag_crm_collect.py, frozen) raises ValueError on it, so soil Gator rows carry the soil_v2 form ['--vehicle', 'gator']
            extra = ['--vehicle', vehicle] + (['--runtime-fingerprint', GATOR_FP] if vehicle == 'gator' and a.world == 'rigid' else [])
            reuse = None
            def same_drive(er):
                lc, lr = local_of(er['case']), local_of(er['route'])
                return (er['group'] == g and er.get('vehicle', 'hmmwv') == vehicle and lc is not None and lr is not None and lr.exists()
                        and str(lc.resolve()) == str(Path(v['case']).resolve()) and route_content_sha(lr) == sha)
            if a.world == 'crm' and not a.no_reuse:
                cands = ([existing[rid]] if rid in existing else []) + by_content.get((g, vehicle, sha), [])
                reuse = next((er for er in cands if same_drive(er)), None)
                if reuse is None and rid in existing:
                    raise SystemExit(f'{rid}: id already in {existing[rid]["_file"]} with a different drive (rename the arm)')
            elif rid in existing:
                raise SystemExit(f'{rid}: id already in {existing[rid]["_file"]}')
            if reuse is not None:
                for n in names:
                    mapping[g][n] = dict(run_id=reuse['id'], vehicle=vehicle, route_sha256=sha, reused_from=reuse['_file'], arena=v['arena'],
                                         case=case_rel, mode=pa[n]['mode'], P=pa[n]['P'], z=pa[n]['z'])
                reused[reuse.get('kind', '?')] += 1
                continue
            seed, salt = new_seed(rid)
            row = dict(id=rid, group=g, arena=v['arena'], case=case_rel, route=route_rel, tier=a.tier, episode_seed=seed,
                       run=True, kind='eval', arms=list(names), sha256=sha, vehicle=vehicle, extra=extra, split='eval', world=a.world)
            if salt:
                row['episode_seed_salt'] = salt
            if a.world == 'rigid':
                row['case'] = f'{G3}/{case_rel}'; row['route'] = f'{G3}/{route_rel}'
            rows.append(row)
            for n in names:
                mapping[g][n] = dict(run_id=rid, vehicle=vehicle, route_sha256=sha, reused_from=None, arena=v['arena'],
                                     case=case_rel, mode=pa[n]['mode'], P=pa[n]['P'], z=pa[n]['z'])
            for loc, rel in ((v['case'], case_rel), (v['route'], route_rel)):
                stage[rel] = str(Path(loc).resolve())
    # rigid shards: every row of a group in one shard
    shards = {}
    if a.world == 'rigid':
        groups = sorted({r['group'] for r in rows}, key=md5hex)
        shards = {g: a.shard_base + i // a.groups_per_shard for i, g in enumerate(groups)}
        for r in rows:
            r['shard'] = shards[r['group']]
        gs = defaultdict(set)
        for r in rows:
            gs[r['group']].add(r['shard'])
        assert all(len(v) == 1 for v in gs.values())
    ids = [r['id'] for r in rows]
    assert len(set(ids)) == len(ids), 'duplicate run ids'
    seeds = [r['episode_seed'] for r in rows]
    assert len(set(seeds)) == len(seeds), 'duplicate episode seeds'
    assert not any(re.search(r'_(route|op)_\d{2}$', i) for i in ids), 'an eval id looks like a training id'
    clash = {r['episode_seed'] for r in rows} & {r['episode_seed'] for r in existing.values()}
    assert not clash, f'{len(clash)} episode seeds of new rows already used in --existing files'
    assert not {r['id'] for r in rows} & set(existing), 'new row id already in an --existing file'
    missing = [p for p in stage.values() if not Path(p).exists()]
    assert not missing, missing[:3]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, 'w'), indent=None)
    cover = {x['name']: sum(1 for g in mapping if x['name'] in mapping[g]) for x in arms}
    groups_by_arena = defaultdict(set)
    for (vehicle, g), pa in picks.items():
        groups_by_arena[next(iter(pa.values()))['arena']].add(g)
    incomplete = {}
    for g, mg in mapping.items():
        arena = next(iter(mg.values()))['arena']
        want = {x['name'] for x in arms if arena in x['arenas']}
        if want - set(mg):
            incomplete[g] = sorted(want - set(mg))
    json.dump(dict(schema='ag_eval_mapping_v1', world=a.world, tasks=str(out), tasks_sha256=sha256_file(out), arms=arms, order=order,
                   groups={g: mapping[g] for g in sorted(mapping)}), open(str(out) + '.mapping.json', 'w'), indent=1, default=str)
    with open(str(out) + '.staging.tsv', 'w') as f:
        f.write('local\tg3_path\tsha256\n')
        for rel, loc in sorted(stage.items()):
            f.write(f'{os.path.relpath(loc, ROOT)}\t{rel}\t{sha256_file(loc)}\n')
    n_arm_drives = sum(len(r['arms']) for r in rows)
    meta = dict(tool='scripts/ag_eval_tasks.py', tool_sha256=sha256_file(__file__), created=time.strftime('%Y-%m-%d %H:%M:%S'),
                argv=sys.argv[1:], world=a.world, n_rows=len(rows), file_sha256=sha256_file(out),
                arm_group_pairs=int(sum(len(v) for v in mapping.values())), drives_saved_by_identical_routes=n_arm_drives - len(rows),
                reused_existing_rows=dict(reused), existing_files={f: sha256_file(f) for f in a.existing or []},
                by_arena=dict(Counter(r['arena'] for r in rows)), by_vehicle=dict(Counter(r['vehicle'] for r in rows)),
                by_arm_first=dict(Counter(r['arms'][0] for r in rows)), arm_coverage_groups=cover,
                groups_missing_some_arm=len(incomplete), groups_missing_some_arm_examples=dict(list(incomplete.items())[:10]),
                groups_by_arena={k: len(v) for k, v in sorted(groups_by_arena.items())},
                shards=dict(base=a.shard_base, groups_per_shard=a.groups_per_shard, n=len(set(shards.values())),
                            range=[min(shards.values()), max(shards.values())] if shards else None,
                            rows_per_shard=dict(Counter(Counter(r['shard'] for r in rows).values()))) if a.world == 'rigid' else None,
                staging=dict(files=len(stage), list=str(out) + '.staging.tsv'),
                collector_contract=(
                    'rigid: gen_runner_g.py-style runner (every row of a shard in one array task / pool step), GEN_COLLECTOR = '
                    f'{G3}/r2/source/scripts/ag_gen_collect_ext.py (the r2 tree lists all 14 arenas; G3/source lists only 10), '
                    'FDM_RUNTIME_FINGERPRINT = the f104 HMMWV fingerprint in the environment (Gator rows override it with their '
                    '--runtime-fingerprint), NEDM_VEHICLE unset, paths absolute' if a.world == 'rigid' else
                    f'soil: crm_worker.py with CRM_COLLECTOR = {G3}/source/scripts/ag_crm_collect.py (frozen dispatcher), CRM_CONFIG = '
                    'configs/crm_main.json, CRM_OUT = the soil output folder of the superset task file, NEDM_VEHICLE unset'))
    json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ('groups_missing_some_arm_examples',)}, indent=1))
    return meta


# ------------------------------------------------------------------------------------------------------------------
def smoke(a):
    rows = {r['id']: r for r in json.load(open(a.tasks))}
    ids = [s for s in a.ids.split(',') if s] if a.ids else []
    assert ids, '--ids required'
    out = Path(a.out); (out / 'runs').mkdir(parents=True, exist_ok=True); (out / 'logs').mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=f'{LOCAL_CHRONO_PY}:{ROOT / "src"}:{HERE}', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    env.pop('NEDM_VEHICLE', None); env.pop('FDM_RUNTIME_FINGERPRINT', None)

    def one(rid):
        t = rows[rid]
        assert t.get('world', 'rigid') == 'rigid' and 'shard' in t, f'{rid}: smoke drives rigid rows only'
        d = out / 'runs' / rid
        if (d / 'episode_complete.json').exists():
            return rid, 'cached', 0.0
        extra = list(t.get('extra', []))
        if '--runtime-fingerprint' in extra:
            i = extra.index('--runtime-fingerprint'); del extra[i:i + 2]
        cmd = [LOCAL_PY, '-P', '-u', str(HERE / 'ag_gen_collect_ext.py'), '--source-root', str(ROOT), '--case', str(local_of(t['case'])),
               '--route', str(local_of(t['route'])), '--out', str(d), '--chrono-data', LOCAL_CHRONO_DATA, '--horizon-s', '120'] + extra + ['--local']
        t0 = time.time()
        with open(out / 'logs' / f'{rid}.log', 'w') as lg:
            lg.write(' '.join(cmd) + '\n'); lg.flush()
            rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=lg, stderr=subprocess.STDOUT, timeout=3600).returncode
        return rid, ('ok' if rc == 0 else f'rc={rc}'), time.time() - t0

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, ids))
    rec = []
    for rid, st, wall in res:
        d = out / 'runs' / rid
        o = json.load(open(d / 'outcome.json')) if (d / 'outcome.json').exists() else {}
        rec.append(dict(id=rid, result=st, wall_s=round(wall, 1), status=o.get('status'), elapsed_s=o.get('elapsed_s'),
                        complete=(d / 'episode_complete.json').exists(), vehicle_block=o.get('vehicle') is not None))
    json.dump(dict(tasks=str(a.tasks), tasks_sha256=sha256_file(a.tasks), runs=rec), open(out / 'smoke_record.json', 'w'), indent=1)
    for r in rec:
        print(r, flush=True)
    return rec


# ------------------------------------------------------------------------------------------------------------------
def stage(a):
    lines = [l.rstrip('\n').split('\t') for l in open(str(a.tasks) + '.staging.tsv')][1:]
    k3_rel = [(loc, rel, h) for loc, rel, h in lines if not rel.startswith('ext/')]
    ext = [(loc, rel, h) for loc, rel, h in lines if rel.startswith('ext/')]
    res = {}
    for name, items, src, dst in (('k3', k3_rel, str(L.K3) + '/', f'amd:{G3}/'), ('ext', ext, str(ROOT) + '/', f'amd:{G3}/ext/')):
        if not items:
            continue
        lst = Path(f'/tmp/ag_stage_{os.getpid()}_{name}.txt')
        lst.write_text('\n'.join(rel[4:] if name == 'ext' else rel for _, rel, _ in items) + '\n')
        cmd = ['rsync', '-a', '--ignore-existing', '--itemize-changes', f'--files-from={lst}', src, dst] + ([] if a.execute else ['-n'])
        p = subprocess.run(cmd, capture_output=True, text=True)
        res[name] = dict(files=len(items), rc=p.returncode, changes=len([l for l in p.stdout.splitlines() if l.startswith('<f') or l.startswith('cd')]),
                         dry_run=not a.execute, stderr=p.stderr[-500:])
    # remote hash check of every listed file that exists there
    script = 'cd ' + G3 + ' && while IFS= read -r f; do [ -f "$f" ] && sha256sum "$f" || echo "MISSING  $f"; done'
    p = subprocess.run(['ssh', 'amd', script], input='\n'.join(rel for _, rel, _ in lines) + '\n', capture_output=True, text=True)
    remote = {}
    for l in p.stdout.splitlines():
        h, f = l.split(None, 1)
        remote[f.strip()] = h
    want = {rel: h for _, rel, h in lines}
    bad = [f for f, h in want.items() if remote.get(f) not in (h, 'MISSING')]
    miss = [f for f in want if remote.get(f) == 'MISSING']
    res['remote'] = dict(listed=len(want), equal=sum(1 for f, h in want.items() if remote.get(f) == h), missing=len(miss), different=bad[:20])
    print(json.dumps(res, indent=1))
    if bad:
        raise SystemExit(f'{len(bad)} cluster files differ from the local ones')
    if a.execute and miss:
        raise SystemExit(f'{len(miss)} files still missing on the cluster after the copy')
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--world', choices=['rigid', 'crm'], required=True)
    b.add_argument('--arm', action='append', required=True, help='NAME=<glob of ag_picks dirs>[@hmmwv|@gator]')
    b.add_argument('--out', required=True)
    b.add_argument('--tier', type=int, default=-1)
    b.add_argument('--existing', nargs='*', default=[], help='task files whose rows may be reused (soil) / must not clash (rigid)')
    b.add_argument('--no-reuse', action='store_true')
    b.add_argument('--shard-base', type=int, default=3000)
    b.add_argument('--groups-per-shard', type=int, default=8)
    s = sub.add_parser('smoke')
    s.add_argument('--tasks', required=True); s.add_argument('--ids', required=True); s.add_argument('--out', required=True)
    s.add_argument('--workers', type=int, default=6)
    st = sub.add_parser('stage')
    st.add_argument('--tasks', required=True); st.add_argument('--execute', action='store_true')
    a = ap.parse_args(argv)
    return dict(build=build, smoke=smoke, stage=stage)[a.cmd](a)


if __name__ == '__main__':
    main()
