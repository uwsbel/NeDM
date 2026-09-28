#!/usr/bin/env python3
"""Soil evaluation drive rows from pick directories, for ANY vehicle (offroad_vehicles_20260927, module M4; PLAN 4.2).
Vehicle-generalised, G4-rooted copy of the soil half of scripts/ag_eval_tasks.py (frozen; its route-content hash and
arm-name rule are imported). One call = one tier of rows.

  build   --arm NAME=GLOB@VEHICLE (repeatable; GLOB matches ag_picks.py / ov_grad_picks.py output folders of the soil
          world, one per arena; VEHICLE one of VEHICLES, required - no default vehicle). For every (vehicle, group) the
          picks of all arms are collected; arms of the same vehicle whose routes have the same content sha256 are driven
          ONCE (one row whose 'arms' lists them all).
          Run ids: <prefix><group>__<first arm of the row>, prefix '' for the HMMWV and '<vehicle>__' otherwise;
          episode_seed = md5(id)[:8] (salted md5(id#k) on a clash with an --existing / --avoid or earlier row; seeds
          are provenance only). The vehicle is always explicit: extra = ['--vehicle', VEHICLE]; the collector is the
          frozen dispatcher G4/source/scripts/ov_crm_collect.py (crm_worker forwards --crm-config because the name
          contains crm_collect). M113 arms (PLAN 1.3) carry the per-row soil config configs/crm_m113.json (0.5 ms) and
          timeout_s (>= 3,600 s), as ov_smoke_tasks.py writes them. NEDM_VEHICLE must stay unset.
          Paths in the rows are relative to G4 (crm_worker resolves them against CRM_ROOT = G4): files under K4 keep
          their K4-relative path, every other repository file goes to ext/<repo-relative path>.
          --existing FILE...: task files (G4-relative rows, e.g. the smoke's sample-B rows, or G3 rows with absolute
          G3 paths) whose drives may be reused: a new route that is already a row there (same group, vehicle, case file
          and route content) is NOT re-driven; the mapping points to that row's id and names the file (so the index
          looks the run up in that study's run folder). --no-reuse re-drives everything.
          --avoid FILE...: task files whose ids / seeds the new rows must not reuse (never reused as drives).
          Writes <out> (rows), <out>.mapping.json (schema ov_eval_mapping_v1: {group: {arm: {run_id, vehicle,
          route_sha256, reused_from, study, ...}}} + arm specs), <out>.staging.tsv (local file -> G4 path, sha256),
          <out>.meta.json (counts, reuse, collector contract).
  stage   copy the staging list to G4 (rsync --ignore-existing) and verify every sha256 there; without --execute only a
          dry run and the remote hash check (read-only).

  PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
  K3=artifacts/traverse/arena_gator_20260925; K4=artifacts/traverse/offroad_vehicles_20260927
  $PY scripts/ov_eval_tasks.py build --tier -9 --arm Ggrad_gator="$K4/e6/picks/f104/G_full_grad@gator" \
      --arm Hgrad_hmmwv="$K4/e6/picks/f104/H_full_grad@hmmwv" --out $K4/e6/tasks/soil_eval_ref.json
"""
import argparse, glob, hashlib, json, os, re, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import ag_eval_tasks as AET               # noqa: E402  (frozen: route_content_sha, ARM_RE)
import ag_tasklib as L                    # noqa: E402

K3, G3 = L.K3, L.G3
K4 = ROOT / 'artifacts/traverse/offroad_vehicles_20260927'
G4 = '/work1/dannegrut/harry/experiments/offroad_vehicles_20260927'
VEHICLES = ('hmmwv', 'gator', 'polaris', 'polaris_pc', 'polaris_4wd', 'polaris_w08', 'm113', 'm113_g4')
HALF_STEP = ('m113', 'm113_g4')                       # PLAN 1.3: 0.5 ms soil step, per-row config
M113_CONFIG = 'configs/crm_m113.json'
COLLECTOR = f'{G4}/source/scripts/ov_crm_collect.py'


def vprefix(vehicle):
    assert vehicle in VEHICLES, f'unknown vehicle {vehicle!r} (known: {VEHICLES})'
    return '' if vehicle == 'hmmwv' else f'{vehicle}__'


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def g4_rel(local):
    """Local file -> path relative to G4: under K4 the K4-relative path, else ext/<repo-relative path>."""
    p = Path(local).resolve()
    try:
        return str(p.relative_to(K4.resolve()))
    except ValueError:
        return 'ext/' + str(p.relative_to(ROOT.resolve()))


def local_of(path):
    """A row path (G4-relative, absolute G4, absolute G3, or G3-relative when the row is a G3 row) -> local file."""
    for root, loc in ((G4, K4), (G3, K3)):
        if path.startswith(root + '/'):
            rel = path[len(root) + 1:]
            return ROOT / rel[4:] if rel.startswith('ext/') else loc / rel
    if path.startswith('/'):
        return None
    return ROOT / path[4:] if path.startswith('ext/') else K4 / path


def local_of_row(path, study):
    """G3 task files write G3-relative paths; G4 files G4-relative ones."""
    if not path.startswith('/') and study == 'g3':
        return ROOT / path[4:] if path.startswith('ext/') else K3 / path
    return local_of(path)


def study_of(task_file):
    p = str(Path(task_file).resolve())
    return 'g3' if p.startswith(str(K3.resolve())) else 'g4'


def parse_arm(spec):
    name, rest = spec.split('=', 1)
    assert '@' in rest, f'arm {name}: the vehicle is required (NAME=GLOB@VEHICLE)'
    rest, vehicle = rest.rsplit('@', 1)
    assert AET.ARM_RE.match(name), f'arm name {name!r}: letters, digits and single underscores only'
    vprefix(vehicle)
    dirs = sorted(d for d in glob.glob(rest) if (Path(d) / 'ag_picks.json').exists())
    assert dirs, f'arm {name}: no pick directory with ag_picks.json matches {rest}'
    return name, vehicle, [Path(d) for d in dirs]


def build(a):
    arms, picks, man_by = [], {}, {}
    for spec in a.arm:
        name, vehicle, dirs = parse_arm(spec)
        assert name not in {x['name'] for x in arms}, f'arm {name} given twice'
        seen_arena = {}
        for d in dirs:
            m = json.load(open(d / 'ag_picks.json'))
            assert m['world'] == 'crm', f'{d}: world {m["world"]} (soil rows only)'
            assert m['arena'] not in seen_arena, f'arm {name}: two pick dirs for arena {m["arena"]}'
            lk = (d / 'PICKS_LOCKED.sha256').read_text().split()[0]
            assert lk == m['picks_locked_sha256'], f'{d}: PICKS_LOCKED changed since the manifest was written'
            seen_arena[m['arena']] = str(d)
            man_by[str(d)] = m
            for g, e in m['picks'].items():
                if e is None:
                    continue
                pf = d / e['route_file']
                assert sha256_file(pf) == e['file_sha256'], f'{pf}: route file changed since the lock'
                assert AET.route_content_sha(pf) == e['route_sha256'], f'{pf}: route content != pick record'
                case = ROOT / m['cases_dir'] / f'{g}.json'
                picks.setdefault((vehicle, g), {})[name] = dict(route=pf, case=case, sha=e['route_sha256'], arena=m['arena'],
                                                                 mode=m['mode'], model_tag=m.get('model_tag'), picks_dir=str(d),
                                                                 P=e.get('P'), z=e.get('z_mean'))
        arms.append(dict(name=name, vehicle=vehicle, dirs=[str(d) for d in dirs], arenas=sorted(seen_arena),
                         modes=sorted({man_by[str(d)]['mode'] for d in dirs}), model_tags=sorted({str(man_by[str(d)].get('model_tag')) for d in dirs}),
                         models=sorted({mm['sha256'] for d in dirs for mm in (man_by[str(d)].get('models') or [])}),
                         locks=sorted({man_by[str(d)]['picks_locked_sha256'] for d in dirs})))
        assert len(arms[-1]['modes']) == 1, f'arm {name} mixes modes {arms[-1]["modes"]}'
    order = [x['name'] for x in arms]
    existing, avoid_ids, used_seeds = {}, set(), set()
    for f in a.existing or []:
        st = study_of(f)
        for r in json.load(open(f)):
            existing[r['id']] = dict(r, _file=str(f), _study=st)
            if r.get('episode_seed') is not None:
                used_seeds.add(r['episode_seed'])
    for f in a.avoid or []:
        for r in json.load(open(f)):
            avoid_ids.add(r['id'])
            if r.get('episode_seed') is not None:
                used_seeds.add(r['episode_seed'])
    by_content = defaultdict(list)
    for r in existing.values():
        if r.get('sha256'):
            by_content[(r['group'], r.get('vehicle', 'hmmwv'), r['sha256'])].append(r)

    def new_seed(rid):
        k, seed = 0, L.md5_int(rid)
        while seed in used_seeds:
            k += 1; seed = L.md5_int(f'{rid}#{k}')
        used_seeds.add(seed)
        return seed, k

    def same_drive(er, g, vehicle, case, sha):
        lc, lr = local_of_row(er['case'], er['_study']), local_of_row(er['route'], er['_study'])
        return (er['group'] == g and er.get('vehicle', 'hmmwv') == vehicle and er.get('run', True) is not False and lc is not None
                and lr is not None and lr.exists() and str(lc.resolve()) == str(Path(case).resolve()) and AET.route_content_sha(lr) == sha)

    rows, mapping, reused, stage = [], defaultdict(dict), Counter(), {}
    for (vehicle, g), pa in sorted(picks.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        cases = {str(v['case']) for v in pa.values()}
        assert len(cases) == 1, f'{g}: arms name different case files {cases}'
        by_sha = {}
        for name in order:
            if name in pa:
                by_sha.setdefault(pa[name]['sha'], []).append(name)
        for sha, names in by_sha.items():
            v = pa[names[0]]
            rid = f'{vprefix(vehicle)}{g}__{names[0]}'
            assert rid not in avoid_ids, f'{rid}: id already used in an --avoid file (rename the arm)'
            reuse = None
            if not a.no_reuse:
                cands = ([existing[rid]] if rid in existing else []) + by_content.get((g, vehicle, sha), [])
                reuse = next((er for er in cands if same_drive(er, g, vehicle, v['case'], sha)), None)
                if reuse is None and rid in existing:
                    raise SystemExit(f'{rid}: id already in {existing[rid]["_file"]} with a different drive (rename the arm)')
            elif rid in existing:
                raise SystemExit(f'{rid}: id already in {existing[rid]["_file"]}')
            if reuse is not None:
                for n in names:
                    mapping[g][n] = dict(run_id=reuse['id'], vehicle=vehicle, route_sha256=sha, reused_from=reuse['_file'], study=reuse['_study'],
                                         arena=v['arena'], case=g4_rel(v['case']), mode=pa[n]['mode'], P=pa[n]['P'], z=pa[n]['z'])
                reused[f"{reuse['_study']}:{reuse.get('kind', '?')}"] += 1
                continue
            seed, salt = new_seed(rid)
            case_rel, route_rel = g4_rel(v['case']), g4_rel(v['route'])
            row = dict(id=rid, group=g, arena=v['arena'], case=case_rel, route=route_rel, tier=a.tier, episode_seed=seed,
                       run=True, kind='eval', arms=list(names), sha256=sha, vehicle=vehicle, extra=['--vehicle', vehicle],
                       split='eval', world='crm')
            if vehicle in HALF_STEP:
                row['config'] = M113_CONFIG; row['timeout_s'] = a.m113_timeout_s
            if salt:
                row['episode_seed_salt'] = salt
            rows.append(row)
            for n in names:
                mapping[g][n] = dict(run_id=rid, vehicle=vehicle, route_sha256=sha, reused_from=None, study='g4', arena=v['arena'],
                                     case=case_rel, mode=pa[n]['mode'], P=pa[n]['P'], z=pa[n]['z'])
            for loc, rel in ((v['case'], case_rel), (v['route'], route_rel)):
                stage[rel] = str(Path(loc).resolve())
    ids = [r['id'] for r in rows]
    assert len(set(ids)) == len(ids), 'duplicate run ids'
    seeds = [r['episode_seed'] for r in rows]
    assert len(set(seeds)) == len(seeds), 'duplicate episode seeds'
    assert not any(re.search(r'_(route|op)_\d{2}$', i) for i in ids), 'an eval id looks like a training id'
    assert not set(ids) & set(existing), 'new row id already in an --existing file'
    assert all(r['extra'] == ['--vehicle', r['vehicle']] for r in rows)
    missing = [p for p in stage.values() if not Path(p).exists()]
    assert not missing, missing[:3]
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists() or a.force, f'{out} exists (--force to replace)'
    json.dump(rows, open(out, 'w'), indent=None)
    cover = {x['name']: sum(1 for g in mapping if x['name'] in mapping[g]) for x in arms}
    json.dump(dict(schema='ov_eval_mapping_v1', world='crm', tasks=str(out), tasks_sha256=sha256_file(out), arms=arms, order=order,
                   existing={f: dict(sha256=sha256_file(f), study=study_of(f)) for f in a.existing or []},
                   groups={g: mapping[g] for g in sorted(mapping)}), open(str(out) + '.mapping.json', 'w'), indent=1, default=str)
    with open(str(out) + '.staging.tsv', 'w') as f:
        f.write('local\tg4_path\tsha256\n')
        for rel, loc in sorted(stage.items()):
            f.write(f'{os.path.relpath(loc, ROOT)}\t{rel}\t{sha256_file(loc)}\n')
    n_arm_drives = sum(len(r['arms']) for r in rows)
    meta = dict(tool='scripts/ov_eval_tasks.py', tool_sha256=sha256_file(__file__), created=time.strftime('%Y-%m-%d %H:%M:%S'),
                argv=sys.argv[1:], world='crm', tier=a.tier, n_rows=len(rows), file_sha256=sha256_file(out),
                arm_group_pairs=int(sum(len(v) for v in mapping.values())), drives_saved_by_identical_routes=n_arm_drives - len(rows),
                reused_existing_rows=dict(reused), existing_files={f: sha256_file(f) for f in a.existing or []},
                avoid_files={f: sha256_file(f) for f in a.avoid or []},
                by_vehicle=dict(Counter(r['vehicle'] for r in rows)), by_arm_first=dict(Counter(r['arms'][0] for r in rows)),
                arm_coverage_groups=cover, half_step_rows=sum(1 for r in rows if r.get('config')),
                staging=dict(files=len(stage), list=str(out) + '.staging.tsv'),
                collector_contract=(f'soil: crm_worker.py with CRM_ROOT = {G4}, CRM_COLLECTOR = {COLLECTOR} (frozen dispatcher), CRM_CONFIG = '
                                    'configs/crm_main.json (M113 rows: per-row config configs/crm_m113.json and timeout_s), NEDM_VEHICLE unset '
                                    '(submit with env -u NEDM_VEHICLE); append the rows to a superset task file at their tier'))
    json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    print(json.dumps(meta, indent=1))
    return meta


def stage(a):
    lines = [l.rstrip('\n').split('\t') for l in open(str(a.tasks) + '.staging.tsv')][1:]
    k4 = [(loc, rel, h) for loc, rel, h in lines if not rel.startswith('ext/')]
    ext = [(loc, rel, h) for loc, rel, h in lines if rel.startswith('ext/')]
    res = {}
    for name, items, src, dst in (('k4', k4, str(K4) + '/', f'amd:{G4}/'), ('ext', ext, str(ROOT) + '/', f'amd:{G4}/ext/')):
        if not items:
            continue
        lst = Path(f'/tmp/ov_stage_{os.getpid()}_{name}.txt')
        lst.write_text('\n'.join(rel[4:] if name == 'ext' else rel for _, rel, _ in items) + '\n')
        cmd = ['rsync', '-a', '--ignore-existing', '--itemize-changes', f'--files-from={lst}', src, dst] + ([] if a.execute else ['-n'])
        p = subprocess.run(cmd, capture_output=True, text=True)
        res[name] = dict(files=len(items), rc=p.returncode, changes=len([l for l in p.stdout.splitlines() if l.startswith('<f') or l.startswith('cd')]),
                         dry_run=not a.execute, stderr=p.stderr[-500:])
    script = 'cd ' + G4 + ' 2>/dev/null || exit 0; while IFS= read -r f; do [ -f "$f" ] && sha256sum "$f" || echo "MISSING  $f"; done'
    p = subprocess.run(['ssh', 'amd', script], input='\n'.join(rel for _, rel, _ in lines) + '\n', capture_output=True, text=True)
    remote = {}
    for l in p.stdout.splitlines():
        h, f = l.split(None, 1)
        remote[f.strip()] = h
    want = {rel: h for _, rel, h in lines}
    bad = [f for f, h in want.items() if remote.get(f) not in (h, 'MISSING', None)]
    miss = [f for f in want if remote.get(f) in ('MISSING', None)]
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
    b.add_argument('--arm', action='append', required=True, help='NAME=<glob of pick dirs>@<vehicle>')
    b.add_argument('--out', required=True); b.add_argument('--tier', type=int, required=True)
    b.add_argument('--existing', nargs='*', default=[]); b.add_argument('--avoid', nargs='*', default=[])
    b.add_argument('--no-reuse', action='store_true'); b.add_argument('--force', action='store_true')
    b.add_argument('--m113-timeout-s', type=int, default=4000)
    st = sub.add_parser('stage')
    st.add_argument('--tasks', required=True); st.add_argument('--execute', action='store_true')
    a = ap.parse_args(argv)
    if a.cmd == 'build':
        assert a.m113_timeout_s >= 3600, 'PLAN 1.3: M113 episode timeout >= 3,600 s'
    return dict(build=build, stage=stage)[a.cmd](a)


if __name__ == '__main__':
    main()
