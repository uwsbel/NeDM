#!/usr/bin/env python3
"""Smoke task file of offroad_vehicles_20260927 (module M3; PLAN sections 1.5, 2.2, 2.3).

One soil task file for crm_worker.py (unchanged) with the dispatcher G4/source/scripts/ov_crm_collect.py (module M1,
frozen before the first job). Tiers, in the order the workers drive them (lowest tier first, random inside a tier):

  tier -4  bit-identity rows (already-driven collect_v1 ids, not in sample A): 3 x bitid_hmmwv__<id> (--vehicle hmmwv,
           reference = the collect_v1 run) and the same 3 ids as bitid_gator__<id> (--vehicle gator, reference = the
           arena_gator soil_v1 Gator run). Judged at the outcome level; arrays reported.
  tier -3  quick look: sample A tiers 0-1 (48 routes) for polaris and gatorctl (the Gator re-driven through the new
           dispatcher); with --m113 quick|all also m113 and m113_g4 (per-row soil config configs/crm_m113.json, 0.5 ms,
           and a per-row timeout_s that the launcher enforces job-wide).
  tier -2  rest of sample A (tiers 2-5, 96 routes) for polaris and gatorctl; with --m113 all also the M113 arms.
  tier -1  sample A (144) for polaris_pc, polaris_4wd, polaris_w08; sample B (96 suite pairs x {straight 6 m/s, the
           HMMWV-trained H_full pick}) for polaris (--b-vehicles).
  optional --gator-half-step: gatorh__<id> rows (the Gator at the M113's 0.5 ms soil step, config crm_m113.json) at the
           same tiers as the M113 rows (the CRITIC's control for the step difference; not in PLAN 2.2, off by default).

Sample-A rows = the Gator rows gator__<id> of arena_gator_20260925/e3/tasks/soil_v2.json (same case, route, episode seed,
split, kind; the collect_v1 tier is kept as collect_tier), id <arm>__<collect_v1 id>, extra ['--vehicle', <vehicle>]
(gatorctl -> gator). Sample-B rows = the stored Gator evaluation rows of soil_v4.json (resolved through
e6/index/soil_eval_bfull.json), case / route as absolute arena_gator_20260925 paths (read only), same route content
sha256, the Gator row's episode seed (provenance only), id <v>__<group>__<arm>_<v>, arms straight6_<v> / Hfull_free_<v>
(one row when both arms name the same route content, as ag_eval_tasks.py does).
NEDM_VEHICLE is never used: every row names its vehicle.

Checks (the build stops on any failure): row counts per tier and arm, unique ids, no '__' inside arm names, sample A =
the 144 declared ids, every A row equal to its Gator row in case / route / seed / split / kind / group, seeds unique per
arm among the A rows, extra exactly ['--vehicle', v], per-row config only on 0.5 ms rows, B route content sha256 equal
to the stored Gator and HMMWV rows, superset of --check-superset (every old row unchanged except 'run' flips, which are
listed). --check-cluster: every case / route / stored reference path exists on the cluster (read-only ssh).

  PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
  $PY scripts/ov_smoke_tasks.py --out artifacts/traverse/offroad_vehicles_20260927/tasks/smoke_v1.json --check-cluster
  $PY scripts/ov_smoke_tasks.py --m113 quick --check-superset <smoke_v1.json> --out <smoke_v2.json>
"""
import argparse, hashlib, json, shlex, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
K4 = ROOT / 'artifacts/traverse/offroad_vehicles_20260927'
G3 = '/work1/dannegrut/harry/experiments/arena_gator_20260925'
G4 = '/work1/dannegrut/harry/experiments/offroad_vehicles_20260927'
CRM_F104 = '/work1/dannegrut/harry/experiments/crm_f104_20260916'
COLLECT_V1_RUNS = f'{CRM_F104}/collect_v1/runs'
GATOR_RUNS = f'{G3}/soil_v1/runs'           # stored Gator soil runs (the pilot's 144 copied in unchanged)
SOIL_V2 = K3 / 'e3/tasks/soil_v2.json'
SOIL_V4 = K3 / 'e3/tasks/soil_v4.json'
SOIL_V4_SHA = 'd73fe77ba2b2375aa4e299488eef67073392b097d2a9dcffd07152c4b3511aca'
BFULL_INDEX = K3 / 'e6/index/soil_eval_bfull.json'
SAMPLE_A = K4 / 'scratch/S3/sample_A.json'
SAMPLE_B = K4 / 'scratch/S3/sample_B.json'

VEHICLES = ('hmmwv', 'gator', 'polaris', 'polaris_pc', 'polaris_4wd', 'polaris_w08', 'm113', 'm113_g4')
ARM_VEHICLE = {'gatorctl': 'gator', 'gatorh': 'gator', 'polaris': 'polaris', 'polaris_pc': 'polaris_pc',
               'polaris_4wd': 'polaris_4wd', 'polaris_w08': 'polaris_w08', 'm113': 'm113', 'm113_g4': 'm113_g4'}
HALF_STEP_ARMS = ('m113', 'm113_g4', 'gatorh')          # rows that run at 0.5 ms (per-row soil config)
M113_CONFIG = 'configs/crm_m113.json'                    # G4-relative (crm_worker: ROOT / config)
MAIN_CONFIG_SHA = '90cd049ee6b3479a74237e7cdf2ff3567c26d580c117e2bab5ab8eff2218b248'
DEFAULT_TIMEOUT_S = 2400                                 # crm_worker.py EP_TIMEOUT default
B_ARMS = (('straight6', 'straight6_gator', 'straight6'), ('Hfull_free', 'Hfull_free_gator', 'Hfull_free'))


def md5hex(s):
    return hashlib.md5(s.encode()).hexdigest()


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def g3_abs(p):
    return p if p.startswith('/') else f'{G3}/{p}'


def load_gator_rows():
    rows = [r for r in json.load(open(SOIL_V2)) if r['id'].startswith('gator__')]
    by = {r['pair_id']: r for r in rows}
    assert len(by) == len(rows) == 15235, (len(by), len(rows))
    return by


def a_row(arm, g, tier, a):
    """sample-A row of one arm from the stored Gator row g (soil_v2)."""
    v = ARM_VEHICLE[arm]
    r = dict(id=f'{arm}__{g["pair_id"]}', pair_id=g['pair_id'], group=g['group'], stratum=g['stratum'], arena=g['arena'],
             case=g['case'], route=g['route'], tier=tier, collect_tier=g['tier'], episode_seed=g['episode_seed'], run=True,
             kind=g['kind'], split=g['split'], vehicle=v, arm=arm, sample='A', extra=['--vehicle', v],
             ref_runs=g['ref_runs'], stored_gator_run=f'{GATOR_RUNS}/{g["id"]}')
    if arm in HALF_STEP_ARMS:
        r['config'] = M113_CONFIG
        r['timeout_s'] = a.m113_timeout_s
    return r


def bitid_rows(gator, sample_groups):
    """3 collect_v1 ids outside the sample-A groups, tier 0, lowest md5(id): 2 designed + 1 on-policy."""
    cand = [g for g in gator.values() if g['tier'] == 0 and g['group'] not in sample_groups and g['split'] == 'train']
    des = sorted((g for g in cand if g['kind'] == 'designed'), key=lambda g: md5hex(g['pair_id']))[:2]
    onp = sorted((g for g in cand if g['kind'] == 'on_policy'), key=lambda g: md5hex(g['pair_id']))[:1]
    out = []
    for g in des + onp:
        for v, ref in (('hmmwv', f'{COLLECT_V1_RUNS}/{g["pair_id"]}'), ('gator', f'{GATOR_RUNS}/{g["id"]}')):
            out.append(dict(id=f'bitid_{v}__{g["pair_id"]}', pair_id=g['pair_id'], group=g['group'], stratum=g['stratum'],
                            arena=g['arena'], case=g['case'], route=g['route'], tier=-4, collect_tier=g['tier'],
                            episode_seed=g['episode_seed'], run=True, kind='bitid', route_kind=g['kind'], split=g['split'],
                            vehicle=v, arm=f'bitid_{v}', sample='bitid', extra=['--vehicle', v], ref_run=ref))
    return out


def b_rows(vehicle, tier):
    B = json.load(open(SAMPLE_B))
    idx = json.load(open(BFULL_INDEX))['rows']
    v4 = {r['id']: r for r in json.load(open(SOIL_V4))}
    ix = {(r['arm'], r['group']): r for r in idx}
    out, notes = [], Counter()
    for gname in B['groups']:
        per = []     # (arm name, gator row, hmmwv run id)
        for short, garm, harm in B_ARMS:
            gi, hi = ix[(garm, gname)], ix[(harm, gname)]
            gr = v4[gi['run_id']]
            assert gr['vehicle'] == 'gator' and gr['group'] == gname and gr['sha256'] == gi['route_sha256'], (gname, garm)
            assert gi['route_sha256'] == hi['route_sha256'], f'{gname} {short}: Gator and HMMWV routes differ'
            per.append((short, gr, gi, hi))
        by_sha = defaultdict(list)
        for p in per:
            by_sha[p[1]['sha256']].append(p)
        for sha, ps in by_sha.items():
            short, gr, gi, hi = ps[0]
            arms = [f'{p[0]}_{vehicle}' for p in ps]
            rid = f'{vehicle}__{gname}__{arms[0]}'
            out.append(dict(id=rid, group=gname, stratum=gi.get('stratum'), arena='f104', case=g3_abs(gr['case']),
                            route=g3_abs(gr['route']), tier=tier, episode_seed=gr['episode_seed'], run=True, kind='eval',
                            arms=arms, sha256=sha, vehicle=vehicle, arm=vehicle, sample='B', extra=['--vehicle', vehicle],
                            split='eval', world='crm',
                            stored_gator={p[0]: f'{GATOR_RUNS}/{p[2]["run_id"]}' for p in ps},
                            stored_hmmwv={p[0]: f'{GATOR_RUNS}/{p[3]["run_id"]}' for p in ps},
                            source_row=gr['id']))
            if len(ps) > 1:
                notes['shared_route_rows'] += 1
    return out, dict(notes)


def build(a):
    gator = load_gator_rows()
    SA = json.load(open(SAMPLE_A))
    pids = [r['pair_id'] for r in SA['rows']]
    assert len(pids) == len(set(pids)) == 144
    A = [gator[p] for p in pids]
    groups = sorted({g['group'] for g in A})
    assert len(groups) == 24 and Counter(g['tier'] for g in A) == Counter({t: 24 for t in range(6)})
    quick = [g for g in A if g['tier'] <= 1]
    rest = [g for g in A if g['tier'] >= 2]
    m113_arms = ['m113', 'm113_g4'] if a.m113 in ('quick', 'all') else []
    half = (['gatorh'] if a.gator_half_step and m113_arms else [])
    rows = bitid_rows(gator, set(groups))
    for arm in ['polaris', 'gatorctl'] + m113_arms + half:
        rows += [a_row(arm, g, -3, a) for g in quick]
    for arm in ['polaris', 'gatorctl'] + (m113_arms + half if a.m113 == 'all' else []):
        rows += [a_row(arm, g, -2, a) for g in rest]
    for arm in ('polaris_pc', 'polaris_4wd', 'polaris_w08'):
        rows += [a_row(arm, g, -1, a) for g in A]
    bnotes = {}
    for v in a.b_vehicles:
        br, bnotes[v] = b_rows(v, a.b_tier)
        rows += br
    rows.sort(key=lambda r: r['tier'])      # stable: file order = tier order, arms in the order above
    return rows, dict(sample_A=pids, groups=groups, b_notes=bnotes, gator=gator)


def check(rows, ctx, a):
    ids = [r['id'] for r in rows]
    assert len(ids) == len(set(ids)), 'duplicate ids'
    gator = ctx['gator']
    for r in rows:
        v = r['vehicle']
        assert v in VEHICLES, r['id']
        assert r['extra'] == ['--vehicle', v], r['id']
        pre = r['id'].split('__', 1)[0]
        assert '__' not in pre and pre and r['id'].count('__') >= 1, r['id']
        assert all('NEDM' not in str(x) for x in r['extra'])
        if r.get('config'):
            assert r['arm'] in HALF_STEP_ARMS and r['config'] == M113_CONFIG and r.get('timeout_s', 0) >= 3600, r['id']
        else:
            assert r['arm'] not in HALF_STEP_ARMS, r['id']
        if r['sample'] in ('A', 'bitid'):
            g = gator[r['pair_id']]
            for k in ('case', 'route', 'episode_seed', 'split', 'group', 'stratum', 'arena'):
                assert r[k] == g[k], (r['id'], k)
            assert r['collect_tier'] == g['tier']
            if r['sample'] == 'A':
                assert r['kind'] == g['kind'] and r['id'] == f"{r['arm']}__{r['pair_id']}"
                assert r['tier'] == (-3 if g['tier'] <= 1 else -2) or (r['tier'] == -1 and r['arm'].startswith('polaris_')), r['id']
    by_arm = defaultdict(list)
    for r in rows:
        if r['sample'] == 'A':
            by_arm[r['arm']].append(r)
    for arm, rs in by_arm.items():
        assert len({r['episode_seed'] for r in rs}) == len(rs), f'{arm}: seeds not unique'
        assert len({r['pair_id'] for r in rs}) == len(rs)
    counts = Counter((r['tier'], r['arm'], r['sample']) for r in rows)
    exp = {(-4, 'bitid_hmmwv', 'bitid'): 3, (-4, 'bitid_gator', 'bitid'): 3,
           (-3, 'polaris', 'A'): 48, (-3, 'gatorctl', 'A'): 48, (-2, 'polaris', 'A'): 96, (-2, 'gatorctl', 'A'): 96,
           (-1, 'polaris_pc', 'A'): 144, (-1, 'polaris_4wd', 'A'): 144, (-1, 'polaris_w08', 'A'): 144}
    if a.m113 in ('quick', 'all'):
        for arm in ['m113', 'm113_g4'] + (['gatorh'] if a.gator_half_step else []):
            exp[(-3, arm, 'A')] = 48
            if a.m113 == 'all':
                exp[(-2, arm, 'A')] = 96
    for k, n in exp.items():
        assert counts.get(k) == n, (k, counts.get(k), n)
    other = {k: n for k, n in counts.items() if k not in exp}
    for (t, arm, s), n in other.items():
        assert s == 'B' and t == a.b_tier and arm in a.b_vehicles, (t, arm, s, n)
    for v in a.b_vehicles:
        br = [r for r in rows if r['sample'] == 'B' and r['vehicle'] == v]
        arms = Counter(x for r in br for x in r['arms'])
        assert arms == Counter({f'straight6_{v}': 96, f'Hfull_free_{v}': 96}), arms
        assert len({r['group'] for r in br}) == 96
    # the A arms must cover exactly the declared 144 ids
    for arm, rs in by_arm.items():
        want = set(ctx['sample_A']) if arm not in ('m113', 'm113_g4', 'gatorh') or a.m113 == 'all' else \
            {p for p in ctx['sample_A'] if gator[p]['tier'] <= 1}
        assert {r['pair_id'] for r in rs} == want, arm
    return dict(counts={f'{t}:{arm}:{s}': n for (t, arm, s), n in sorted(counts.items())})


def check_superset(rows, old_path):
    old = json.load(open(old_path))
    new = {r['id']: r for r in rows}
    missing = [r['id'] for r in old if r['id'] not in new]
    changed, flips = [], Counter()
    for r in old:
        n = new.get(r['id'])
        if n is None:
            continue
        a_ = {k: v for k, v in r.items() if k != 'run'}
        b_ = {k: v for k, v in n.items() if k != 'run'}
        if a_ != b_:
            changed.append(r['id'])
        elif r.get('run', True) != n.get('run', True):
            flips[f"{r.get('run', True)}->{n.get('run', True)}"] += 1
    assert not missing and not changed, f'not a superset of {old_path}: {len(missing)} missing, {len(changed)} changed, e.g. {(missing + changed)[:3]}'
    return dict(old=str(old_path), old_sha256=sha256_file(old_path), old_rows=len(old), run_flips=dict(flips),
                new_rows=len(rows) - len(old))


def cluster_paths(rows):
    paths = set()
    for r in rows:
        paths.add(r['case']); paths.add(r['route'])
        if r.get('stored_gator_run'):
            paths.add(r['stored_gator_run'] + '/episode_complete.json')
        if r.get('ref_run'):
            paths.add(r['ref_run'] + '/episode_complete.json')
        if r.get('ref_runs') and r.get('pair_id'):
            paths.add(f"{r['ref_runs']}/{r['pair_id']}/outcome.json")
        for d in (r.get('stored_gator') or {}).values():
            paths.add(d + '/episode_complete.json')
        for d in (r.get('stored_hmmwv') or {}).values():
            paths.add(d + '/episode_complete.json')
    return sorted(paths)


def check_cluster(paths):
    code = ('import os,sys,json\nps=[l.strip() for l in sys.stdin if l.strip()]\n'
            'miss=[p for p in ps if not os.path.exists(p)]\nprint(json.dumps(dict(n=len(ps),missing=miss[:20],n_missing=len(miss))))')
    out = subprocess.run(['ssh', 'amd', 'python3 -c ' + shlex.quote(code)], input='\n'.join(paths), capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f'cluster check failed: {out.stderr[-500:]}')
    return json.loads(out.stdout.strip().splitlines()[-1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--m113', choices=('none', 'quick', 'all'), default='none',
                    help='M113 arms: none (default; PLAN 1.3 bounded path), quick = tier -3 only, all = tiers -3 and -2')
    ap.add_argument('--gator-half-step', action='store_true', help='add gatorh__ rows (Gator at 0.5 ms) next to the M113 rows')
    ap.add_argument('--m113-timeout-s', type=int, default=4000, help='per-row timeout_s of 0.5 ms rows (enforced job-wide by ov_soil_launch.sh)')
    ap.add_argument('--b-vehicles', nargs='*', default=['polaris'])
    ap.add_argument('--b-tier', type=int, default=-1, help='tier of the sample-B rows (PLAN amendment 9.1 item S8: -2)')
    ap.add_argument('--check-superset', default=None)
    ap.add_argument('--check-cluster', action='store_true')
    a = ap.parse_args(argv)
    assert a.m113_timeout_s >= 3600, 'PLAN 1.3: M113 episode timeout >= 3,600 s'
    assert sha256_file(SOIL_V4) == SOIL_V4_SHA, 'soil_v4.json changed'
    rows, ctx = build(a)
    chk = check(rows, ctx, a)
    sup = check_superset(rows, a.check_superset) if a.check_superset else None
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    cl = None
    if a.check_cluster:
        cl = check_cluster(cluster_paths(rows))
        assert cl['n_missing'] == 0, f'paths missing on the cluster: {cl}'
    if out.exists():
        raise SystemExit(f'{out} exists: task files are never overwritten (write a new version)')
    json.dump(rows, open(out, 'w'))
    if (ROOT / M113_CONFIG).exists() and any(r.get('config') for r in rows):
        m113_cfg_sha = sha256_file(ROOT / M113_CONFIG)
    else:
        m113_cfg_sha = None
    meta = dict(tool='scripts/ov_smoke_tasks.py', tool_sha256=sha256_file(__file__), created=time.strftime('%F %T'),
                argv=sys.argv[1:] if argv is None else argv, file_sha256=sha256_file(out), rows=len(rows),
                inputs={str(p.relative_to(ROOT)): sha256_file(p) for p in (SOIL_V2, SOIL_V4, BFULL_INDEX, SAMPLE_A, SAMPLE_B)},
                collector_contract='CRM_COLLECTOR = G4/source/scripts/ov_crm_collect.py (frozen); CRM_CONFIG = configs/crm_main.json '
                                   f'(sha256 {MAIN_CONFIG_SHA}); 0.5 ms rows carry config {M113_CONFIG}',
                max_timeout_s=max([r.get('timeout_s', DEFAULT_TIMEOUT_S) for r in rows if r.get('run', True)]),
                m113_config_sha256_local=m113_cfg_sha, by_tier=dict(Counter(str(r['tier']) for r in rows)),
                by_arm=dict(Counter(r['arm'] for r in rows)), counts=chk['counts'], sample_A_groups=ctx['groups'],
                bitid_ids=[r['id'] for r in rows if r['sample'] == 'bitid'], b_notes=ctx['b_notes'],
                superset=sup, cluster_paths=cl)
    json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in meta.items() if k not in ('sample_A_groups',)}, indent=1))


if __name__ == '__main__':
    main()
