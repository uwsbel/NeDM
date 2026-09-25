#!/usr/bin/env python3
"""arena_gator_20260925 module E4: training files for one (arena, vehicle, world) from collected episodes.

Pipeline (each stage runs the unchanged earlier tool as a subprocess, on exactly the selected episodes):
  0. select   run folders of the requested worlds; keep the episodes whose id is <prefix><arena>_v2_group_NNNN_route_NN
              or _op_NN (prefix '' = HMMWV, 'gator__' = Gator), that are training rows of the task files (--tasks, tier
              >= 0, same arena and vehicle; --tasks for both worlds, --tasks-rigid / --tasks-crm per world; without task
              files every matching id), inside --tiers, not in --exclude-ids-<world>,
              with every file the builders read, the completion marker (when --require-marker, default: on when any
              selected folder has markers), a passed launch check and crm_qa.py's validity check (soil) / finite
              arrays (rigid) where the collector wrote those records.
              Checks: one arena, and its BMP sha256 equals the map root's (scripts/ag_map_check.py); the run's
              vehicle block matches --vehicle (Gator runs carry vehicle.name == 'gator', HMMWV runs carry none);
              no suite or planner id; ids unique across the run folders.
  1. station  scripts/f104_n2_dataset.py --root <map root> on a folder of links to the selected episodes
              -> station_<world>.npz (one row per episode); every selected episode must be labelled.
  2. anchors  scripts/n2_reanchor_dataset.py (default flags, = the night-2 f104 files) -> reanchor_<world>.npz;
              every episode must give its k = 0 row.
  3. ci file  both worlds: scripts/ga_build_mixed.py (exit status 0, no missing episode required), then the columns
              arena, vehicle, tier are appended to its npz; one world: the same arrays for that world alone (thin
              variant below, using ga_build_mixed's own functions and checks) -> ci_<arena>_<vehicle>_<world>.npz.
              Tier = the task-file tier; for ids without a task row (the f104 pools) the crm_tasks.py per-group order
              (ag_tasklib.group_routes), which is also asserted equal to every task-file tier.
  4. checks   no id / group / episode matches a suite pattern (ga_build_mixed.BLACKLIST + generic suite patterns);
              every group is a training-pool group of the arena; ids unique; per-world x split x start counts.
Writes <out>/<stem>_record.json (selection counts, map check, commands, tool hashes, row counts, output sha256).
--compare-to REF.npz: restrict the new ci file to the episodes present in REF (a ga_build_mixed file) and compare row
counts and arrays (the twin-subset check of the f104 rebuild).

  PYTHONPATH=src:scripts python scripts/ag_build_ds.py --arena f104 --vehicle hmmwv --world both \
    --rigid-runs .../production_v3/runs .../production_v4/runs --crm-runs .../collect_v1/runs \
    --map-root artifacts/traverse/crm_f104_v1/map_root --out <dir>
numpy only (runs on the cluster login node or a compute node too).
"""
import argparse, fnmatch, functools, hashlib, json, os, re, shutil, subprocess, sys, time, zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ag_map_check                      # noqa: E402
import ag_tasklib                        # noqa: E402
import ga_build_mixed as GBM             # noqa: E402

WORLDS = ('rigid', 'crm')
NEED = dict(rigid=['trajectory.npz', 'outcome.json', 'case.json', 'command_reference.npz', 'anchor_state.npz'],
            crm=['trajectory.npz', 'outcome.json', 'case.json', 'command_reference.npz', 'anchor_state.npz', 'crm_extra.npz'])
SUITE_GENERIC = ['*_test_group_*', '*_heldout_group_*', '*_dev_group_*', '*_eval_group_*', '*_pair_group_*', 'drift__*']
VEHICLE_PREFIX = dict(hmmwv='', gator='gator__')
TOOLS = ['f104_n2_dataset.py', 'n2_reanchor_dataset.py', 'ga_build_mixed.py', 'ag_build_ds.py', 'ag_map_check.py', 'ag_tasklib.py', 'crm_qa.py']


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def arena_short(arena_path):
    """'assets/traverse/arena_f104_50h_v1' -> 'f104'; 'assets/traverse/arena_g203' -> 'g203'."""
    b = os.path.basename(str(arena_path).rstrip('/'))
    b = b[len('arena_'):] if b.startswith('arena_') else b
    return b.split('_')[0]


def suite_hit(s):
    return GBM.blacklisted(s) or any(fnmatch.fnmatch(s, p) for p in SUITE_GENERIC)


def parse_tiers(s):
    if not s:
        return None
    lo, _, hi = s.partition('-')
    return int(lo), int(hi or lo)


@functools.lru_cache(maxsize=None)
def group_order(g):
    return {rid: t for t, rid, _, _ in ag_tasklib.group_routes(g, 'd', 'o')}


def f104_order_tier(base_id):
    """Tier of <group>_route_NN / <group>_op_NN in the crm_tasks.py per-group shuffle (ag_tasklib.group_routes)."""
    return group_order(re.sub(r'_(route|op)_\d{2}$', '', base_id))[base_id]


def load_task_rows(paths):
    rows = {}
    for p in paths or []:
        for r in json.load(open(p)):
            if r['id'] in rows and rows[r['id']] != r:
                raise SystemExit(f'task id {r["id"]} differs between task files')
            rows[r['id']] = r
    return rows


def load_exclude(paths):
    out = set()
    for p in paths or []:
        if p.endswith('.json'):
            j = json.load(open(p))
            if isinstance(j, dict) and 'flagged_ids' in j:     # crm_qa.py qa.json
                out |= {r['id'] for r in j['flagged_ids']}
            elif isinstance(j, list):
                out |= {r if isinstance(r, str) else r['id'] for r in j}
            else:
                raise SystemExit(f'{p}: unknown exclude format')
        else:
            out |= {l.strip() for l in open(p) if l.strip()}
    return out


def inspect(item):
    """One run dir -> dict(ok, reason, ...). Reads only small records (+ the arrays for the finite check)."""
    d, world, require_marker = item
    rid = os.path.basename(d)
    miss = [f for f in NEED[world] if not os.path.isfile(os.path.join(d, f))]
    if miss:
        return dict(id=rid, ok=False, reason='missing_files:' + ','.join(miss))
    has_marker = os.path.isfile(os.path.join(d, 'episode_complete.json'))
    if require_marker and not has_marker:
        return dict(id=rid, ok=False, reason='no_completion_marker')
    try:
        c = json.load(open(os.path.join(d, 'case.json'))); o = json.load(open(os.path.join(d, 'outcome.json')))
    except Exception as e:
        return dict(id=rid, ok=False, reason=f'unreadable:{type(e).__name__}')
    veh = o.get('vehicle')
    row = dict(id=rid, ok=True, reason=None, case=os.path.join(d, 'case.json'), arena=c.get('arena'), group=c.get('id'),
               split=c.get('split'), status=o.get('status'), vehicle=(veh or {}).get('name') if isinstance(veh, dict) else veh,
               marker=has_marker)
    launch = os.path.join(d, 'initial_state_validation.json')
    if world == 'crm' and os.path.isfile(launch) and os.path.isfile(os.path.join(d, 'collection_request.json')):
        import crm_qa
        q = crm_qa.check(d)
        if not q['ok']:
            return dict(row, ok=False, reason=f'crm_qa:{q["flag"]}')
    else:
        if os.path.isfile(launch):
            try:
                if not json.load(open(launch)).get('passed', False):
                    return dict(row, ok=False, reason='launch_check_failed')
            except Exception:
                return dict(row, ok=False, reason='launch_check_unreadable')
        try:
            z = np.load(os.path.join(d, 'trajectory.npz'))
            st, ac, po = z['state'], z['action'], z['pose']
        except Exception as e:
            return dict(row, ok=False, reason=f'trajectory_unreadable:{type(e).__name__}')
        if not (len(st) > 0 and st.shape[1] == 17 and ac.shape == (len(st), 3) and po.shape[0] == len(st)):
            return dict(row, ok=False, reason='shape')
        if not (np.isfinite(st).all() and np.isfinite(ac).all() and np.isfinite(po).all()):
            return dict(row, ok=False, reason='nonfinite')
    return row


def run_logged(cmd, log, env):
    t0 = time.time()
    with open(log, 'w') as f:
        f.write('$ ' + ' '.join(map(str, cmd)) + '\n'); f.flush()
        p = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, env=env)
    return p.returncode, round(time.time() - t0, 1)


def append_columns(npz_path, cols):
    """Add per-row arrays to an existing npz (zip) without rewriting it; np.load sees them as ordinary keys."""
    with zipfile.ZipFile(npz_path, 'a', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        have = set(zf.namelist())
        for k, v in cols.items():
            assert f'{k}.npy' not in have, f'{npz_path} already has {k}'
            with zf.open(f'{k}.npy', 'w', force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(v), allow_pickle=True)


def ci_one_world(ra_path, world, run_roots, out_path, extra_cols, workers):
    """ga_build_mixed.py's output for one world alone: same keys, id suffix, domain code, history window, privileged
    context and invariants (its functions are reused, nothing re-implemented but the two-file bookkeeping)."""
    t0 = time.time()
    z = np.load(ra_path, allow_pickle=True)
    out = {k: z[k] for k in z.files if k != 'X'}
    n = len(out['id']); dm = GBM.DOMAIN_CODE[world]
    domain = np.full(n, dm, np.int8); out['domain'] = domain
    out['id'] = np.array([f'{i}@{world}' for i in out['id'].astype(str)], object)
    episode = out['episode'].astype(str); group = out['group'].astype(str); af = out['anchor_frame'].astype(int)
    assert len(set(zip(episode, af.tolist()))) == n, '(episode, anchor_frame) not unique'
    assert len(set(out['id'].astype(str))) == n, 'ids not unique'
    bad = [s for s in set(out['id'].astype(str)) | set(group) | set(episode) if GBM.blacklisted(s)]
    assert not bad, f'blacklisted ids/groups present: {bad[:10]}'
    T = GBM.T
    hist = np.zeros((n, T, 15), np.float16); hmask = np.zeros((n, T), bool); priv = np.zeros((n, 8), np.float32)
    by_ep = defaultdict(list)
    for i in range(n):
        by_ep[episode[i]].append(i)
    tasks = [(ep, [af[i] for i in idx], [str(r) for r in run_roots], world == 'crm', idx) for ep, idx in sorted(by_ep.items())]
    found = np.zeros(n, bool); misses = []
    with ProcessPoolExecutor(max(1, min(workers, 8))) as ex:
        for (ep, ks, roots, is_crm, idx), (ep2, res) in zip(tasks, ex.map(GBM.cut_episode, [t[:4] for t in tasks], chunksize=32)):
            assert ep == ep2
            if res is None:
                misses.append(ep); continue
            h, m, p = res; idx = np.asarray(idx)
            hist[idx] = h; hmask[idx] = m; priv[idx] = p; found[idx] = True
    k0 = af == 0; est = ~k0
    assert not hmask[k0].any(), 'k = 0 rows must be all-masked'
    assert np.all(hmask[found & est, :].sum(1) == np.minimum(af[found & est], T)), 'hmask count != min(k, 40)'
    ref = out['ctx'][found & est][:, GBM.HIST_STATE_COLS].astype(np.float32); got = hist[found & est, T - 1, :12].astype(np.float32)
    assert np.all(np.abs(ref - got) <= 2e-3 * np.maximum(np.abs(ref), 1.0)), 'hist last step != ctx state at anchor'
    if world == 'crm':
        assert np.all(priv[found, 7] == 1)
    else:
        assert np.all(priv[:, 5:8] == 0)
    out['hist'] = hist; out['hmask'] = hmask; out['privileged'] = priv
    out['hist_cols'] = np.asarray(GBM.HIST_STATE_COLS, np.int16); out['priv_names'] = np.asarray(GBM.PRIV_NAMES, object)
    X = z['X']; assert X.dtype == np.float16 and X.shape == (n, 5, 96, 32), (X.dtype, X.shape)
    out['X'] = X
    out.update(extra_cols(out))
    np.savez_compressed(out_path, **out)
    return dict(rows=n, missing_episodes=misses, raw_found_fraction=float(found.mean()), elapsed_s=round(time.time() - t0, 1))


def row_counts(domain, split, af):
    t = {}
    for w, code in GBM.DOMAIN_CODE.items():
        for sp in ('train', 'val', 'test'):
            for kind, m in (('startup', af == 0), ('established', af != 0)):
                v = int(((domain == code) & (split == sp) & m).sum())
                if v:
                    t[f'{w}|{sp}|{kind}'] = v
    return t


def compare(new_path, ref_path):
    """Rows of the new ci file whose (domain, episode) is in REF: counts per domain and exact array equality by id."""
    A = np.load(new_path, allow_pickle=True); B = np.load(ref_path, allow_pickle=True)
    ida, idb = A['id'].astype(str), B['id'].astype(str)
    epb = set(zip(B['domain'].astype(int).tolist(), B['episode'].astype(str)))
    sel = np.array([(int(dm), e) in epb for dm, e in zip(A['domain'], A['episode'].astype(str))])
    pos = {s: i for i, s in enumerate(ida)}
    res = dict(ref=os.path.abspath(ref_path), ref_rows=int(len(idb)), new_rows=int(len(ida)), new_rows_on_ref_episodes=int(sel.sum()),
               by_domain={})
    for w, code in GBM.DOMAIN_CODE.items():
        res['by_domain'][w] = dict(ref=int((B['domain'] == code).sum()), new_on_ref_episodes=int((sel & (A['domain'] == code)).sum()),
                                   ref_episodes=int(len({e for dm, e in epb if dm == code})))
    common = [i for i in idb if i in pos]
    res['ids_in_both'] = len(common); res['ref_ids_missing_in_new'] = int(len(idb) - len(common))
    res['new_ids_on_ref_episodes_missing_in_ref'] = int(sel.sum() - len(common))
    ia = np.array([pos[i] for i in common]); ib = np.array([i for i, s in enumerate(idb) if s in pos])
    eq = {}
    for k in sorted(set(A.files) & set(B.files) - {'hist_cols', 'priv_names'}):
        a, b = A[k], B[k]
        if a.ndim == 0 or a.shape[0] != len(ida):
            eq[k] = bool(np.array_equal(a, b)); continue
        a, b = a[ia], b[ib]
        if a.dtype == object or b.dtype == object:
            eq[k] = bool(np.array_equal(a.astype(str), b.astype(str)))
        else:
            eq[k] = bool(np.array_equal(a, b, equal_nan=np.issubdtype(a.dtype, np.floating)))
            if not eq[k] and np.issubdtype(a.dtype, np.floating):
                eq[k + '_max_abs_diff'] = float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    res['arrays_equal'] = eq
    res['identical'] = (res['ref_ids_missing_in_new'] == 0 and res['new_ids_on_ref_episodes_missing_in_ref'] == 0
                        and all(v for k, v in eq.items() if not k.endswith('_max_abs_diff')))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arena', required=True, help='short arena name as in the ids: f104, g203, g228, ...')
    ap.add_argument('--vehicle', choices=sorted(VEHICLE_PREFIX), default='hmmwv')
    ap.add_argument('--world', choices=['rigid', 'crm', 'both'], required=True)
    ap.add_argument('--rigid-runs', nargs='*', default=[]); ap.add_argument('--crm-runs', nargs='*', default=[])
    ap.add_argument('--map-root', required=True)
    ap.add_argument('--source-root', default=str(HERE.parent), help='root that holds assets/traverse/<arena> (map check)')
    ap.add_argument('--tasks', nargs='*', default=[], help='task files for every world; only their training rows (tier >= 0) of this arena/vehicle are used')
    ap.add_argument('--tasks-rigid', nargs='*', default=[], help='task files for the rigid world only (e.g. tasks/rigid_hmmwv_v1.json)')
    ap.add_argument('--tasks-crm', nargs='*', default=[], help='task files for the soil world only (e.g. tasks/soil_v1.json)')
    ap.add_argument('--tiers', default=None, help='keep tiers A-B (inclusive)')
    ap.add_argument('--exclude-ids-rigid', nargs='*', default=[], help='rigid ids to drop: json list or text file')
    ap.add_argument('--exclude-ids-crm', nargs='*', default=[], help='soil ids to drop: crm_qa.py qa.json (flagged_ids), json list or text file')
    ap.add_argument('--require-marker', choices=['auto', 'yes', 'no'], default='auto')
    ap.add_argument('--id-prefix', default=None, help="run id prefix (default by vehicle: '' HMMWV, 'gator__' Gator)")
    ap.add_argument('--out', required=True); ap.add_argument('--stem', default=None)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--compare-to', default=None)
    ap.add_argument('--keep-work', action='store_true', help='keep the link folders')
    a = ap.parse_args(argv)
    t0 = time.time()
    worlds = list(WORLDS) if a.world == 'both' else [a.world]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / 'logs').mkdir(exist_ok=True)
    stem = a.stem or f'{a.arena}_{a.vehicle}_{a.world}'
    prefix = VEHICLE_PREFIX[a.vehicle] if a.id_prefix is None else a.id_prefix
    pat = re.compile(rf'^{re.escape(prefix)}({re.escape(a.arena)}_v2_group_\d{{4}}_(route|op)_\d{{2}})$')
    tiers = parse_tiers(a.tiers)
    task_rows_by = dict(rigid=load_task_rows(a.tasks + a.tasks_rigid), crm=load_task_rows(a.tasks + a.tasks_crm))
    exclude_by = dict(rigid=load_exclude(a.exclude_ids_rigid), crm=load_exclude(a.exclude_ids_crm))
    obs = json.load(open(Path(a.map_root) / 'static_map_v1' / 'observation.json'))
    rec = dict(tool='scripts/ag_build_ds.py', argv=sys.argv[1:] if argv is None else argv, started=time.strftime('%Y-%m-%d %H:%M:%S'),
               host=os.uname().nodename, python=sys.executable, arena=a.arena, vehicle=a.vehicle, worlds=worlds, map_root=os.path.abspath(a.map_root),
               map_observation_sha256=obs.get('observation_sha256'), map_arena_bmp_sha256=obs.get('arena_bmp_sha256'),
               tasks={w: {p: sha256_file(p) for p in a.tasks + getattr(a, f'tasks_{w}')} for w in WORLDS}, exclude_ids={w: sorted(v)[:50] for w, v in exclude_by.items()},
               n_exclude_ids={w: len(v) for w, v in exclude_by.items()}, tiers=a.tiers,
               tool_sha256={t: sha256_file(HERE / t) for t in TOOLS}, selection={}, stages={})
    env = dict(os.environ); env['PYTHONPATH'] = os.pathsep.join([str(HERE.parent / 'src'), str(HERE)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    env.setdefault('OMP_NUM_THREADS', '1')
    tier_of, links = {}, {}
    for w in worlds:
        roots = a.rigid_runs if w == 'rigid' else a.crm_runs
        assert roots, f'--{w}-runs needed for world {w}'
        seen, cnt = {}, Counter()
        for r in roots:
            for e in os.scandir(r):
                if not e.is_dir():
                    continue
                cnt['dirs'] += 1
                m = pat.match(e.name)
                if not m:
                    cnt['other_ids'] += 1; continue
                assert e.name not in seen, f'id {e.name} in two run folders: {seen[e.name]} and {r}'
                seen[e.name] = os.path.join(r, e.name)
        cand = {}
        for rid, d in sorted(seen.items()):
            base = pat.match(rid).group(1)
            t_order = f104_order_tier(base)
            task_rows = task_rows_by[w]
            if task_rows:
                tr = task_rows.get(rid)
                if tr is None or tr.get('tier', -1) < 0 or tr.get('arena', a.arena) != a.arena or tr.get('vehicle', 'hmmwv') != a.vehicle:
                    cnt['not_a_training_task_row'] += 1; continue
                assert int(tr['tier']) == t_order, f'{rid}: task tier {tr["tier"]} != per-group order tier {t_order}'
            t = t_order
            if tiers and not (tiers[0] <= t <= tiers[1]):
                cnt['outside_tiers'] += 1; continue
            if rid in exclude_by[w]:
                cnt['excluded_by_list'] += 1; continue
            cand[rid] = (d, t)
        require = a.require_marker == 'yes' or (a.require_marker == 'auto' and any(
            os.path.isfile(os.path.join(d, 'episode_complete.json')) for d, _ in list(cand.values())[:200]))
        with ProcessPoolExecutor(a.workers) as ex:
            info = list(ex.map(inspect, [(d, w, require) for d, _ in cand.values()], chunksize=64))
        rejected = Counter(i['reason'] for i in info if not i['ok'])
        ok = [i for i in info if i['ok']]
        # vehicle provenance: Gator runs carry vehicle.name == 'gator'; HMMWV runs carry no vehicle block
        vbad = [i['id'] for i in ok if (i['vehicle'] != 'gator' if a.vehicle == 'gator' else i['vehicle'] is not None)]
        assert not vbad, f'{len(vbad)} runs whose vehicle block does not match --vehicle {a.vehicle}: {vbad[:5]}'
        arenas = Counter(arena_short(i['arena']) for i in ok)
        assert set(arenas) == {a.arena}, f'runs name other arenas: {dict(arenas)}'
        bad_ids = [i['id'] for i in ok if suite_hit(i['id']) or suite_hit(i['group'] or '')]
        assert not bad_ids, f'suite ids among the selected runs: {bad_ids[:5]}'
        gpat = re.compile(rf'^{re.escape(a.arena)}_v2_group_\d{{4}}$')
        assert all(gpat.match(i['group'] or '') for i in ok), 'a run whose case is not a training-pool group of the arena'
        assert all(pat.match(i['id']).group(1).startswith(i['group'] + '_') for i in ok), 'run id and case group disagree'
        mc = ag_map_check.check(a.map_root, [i['case'] for i in ok], a.source_root)
        mc_short = {k: v for k, v in mc.items() if k != 'arenas'}; mc_short['arenas'] = mc['arenas']
        if not mc['ok']:
            raise SystemExit(f'map/arena check failed for {w}: {mc["problems"]}')
        L = out / 'work' / f'links_{stem}_{w}'
        if L.exists():
            shutil.rmtree(L)
        L.mkdir(parents=True)
        for i in ok:
            os.symlink(os.path.abspath(cand[i['id']][0]), L / i['id'])
            tier_of[(w, i['id'])] = cand[i['id']][1]
        links[w] = L
        rec['selection'][w] = dict(run_folders=[os.path.abspath(r) for r in roots], counts=dict(cnt), candidates=len(cand), require_marker=require,
                                   rejected=dict(rejected), rejected_ids=[(i['id'], i['reason']) for i in info if not i['ok']][:200],
                                   selected=len(ok), groups=len({i['group'] for i in ok}),
                                   split_episodes=dict(Counter(i['split'] for i in ok)), split_groups={s: len({i['group'] for i in ok if i['split'] == s}) for s in ('train', 'val', 'test')},
                                   tiers=dict(sorted(Counter(cand[i['id']][1] for i in ok).items())), status=dict(Counter(i['status'] for i in ok)),
                                   kinds=dict(Counter('on_policy' if '_op_' in i['id'] else 'designed' for i in ok)),
                                   vehicle_blocks=dict(Counter(str(i['vehicle']) for i in ok)), map_check=mc_short)
        print(f'[{w}] {len(ok)} episodes selected ({dict(cnt)}; rejected {dict(rejected)}); map check ok ({mc["map_arena_bmp_sha256"][:16]})', flush=True)
        # 1. station rows
        st = out / f'station_{stem}_{w}.npz'
        rc, secs = run_logged([sys.executable, '-u', HERE / 'f104_n2_dataset.py', '--root', a.map_root, '--runs', f'{L}/*_route_*:designed', f'{L}/*_op_*:on_policy',
                               '--out', st, '--workers', a.workers], out / 'logs' / f'station_{stem}_{w}.log', env)
        assert rc == 0, f'f104_n2_dataset.py failed ({rc}), see logs'
        sid = np.load(st, allow_pickle=True)['id'].astype(str)
        assert len(sid) == len(set(sid)) and set(sid) == {i['id'] for i in ok}, f'station rows {len(sid)} != selected {len(ok)} (unlabelled episodes)'
        # 2. re-anchored rows
        ra = out / f'reanchor_{stem}_{w}.npz'
        rc2, secs2 = run_logged([sys.executable, '-u', HERE / 'n2_reanchor_dataset.py', '--root', a.map_root, '--ids', st, '--runs', L, '--out', ra,
                                 '--workers', a.workers], out / 'logs' / f'reanchor_{stem}_{w}.log', env)
        assert rc2 == 0, f'n2_reanchor_dataset.py failed ({rc2}), see logs'
        z = np.load(ra, allow_pickle=True); rep = z['episode'].astype(str); raf = z['anchor_frame'].astype(int)
        assert set(rep[raf == 0]) == set(sid) and (raf == 0).sum() == len(sid), 'an episode without its k = 0 row'
        rec['stages'][w] = dict(station=dict(path=str(st), rows=int(len(sid)), secs=secs, sha256=sha256_file(st)),
                                reanchor=dict(path=str(ra), rows=int(len(rep)), episodes=int(len(set(rep))), secs=secs2, sha256=sha256_file(ra),
                                              rows_per_episode=round(len(rep) / max(len(sid), 1), 3)))
        print(f'[{w}] station {len(sid)} rows ({secs} s), re-anchored {len(rep)} rows ({secs2} s)', flush=True)
        del z

    # 3. ci_train-ready file
    ci = out / f'ci_{stem}.npz'

    def extra_cols(d):
        dom = d['domain'].astype(int); ep = d['episode'].astype(str)
        wname = {0: 'rigid', 1: 'crm'}
        tier = np.array([tier_of[(wname[x], e)] for x, e in zip(dom, ep)], np.int16)
        return dict(arena=np.array([a.arena] * len(ep), object), vehicle=np.array([a.vehicle] * len(ep), object), tier=tier)

    if len(worlds) == 2:
        rc3, secs3 = run_logged([sys.executable, '-u', HERE / 'ga_build_mixed.py', '--rigid', out / f'reanchor_{stem}_rigid.npz', '--crm', out / f'reanchor_{stem}_crm.npz',
                                 '--rigid-runs', links['rigid'], '--crm-runs', links['crm'], '--out', ci, '--workers', min(a.workers, 8)],
                                out / 'logs' / f'mixed_{stem}.log', env)
        bj = json.load(open(out / f'ci_{stem}_build.json'))
        rec['stages']['mixed'] = dict(tool='ga_build_mixed.py', exit=rc3, secs=secs3, missing_episodes=len(bj['missing_episodes']),
                                      raw_found_fraction=bj['raw_found_fraction'], build_json=str(out / f'ci_{stem}_build.json'))
        assert rc3 == 0 and not bj['missing_episodes'] and bj['raw_found_fraction'] == 1.0, \
            f'ga_build_mixed exit {rc3}, missing {len(bj["missing_episodes"])} episodes'
        z = np.load(ci, allow_pickle=True)
        append_columns(ci, extra_cols({k: z[k] for k in ('domain', 'episode')}))
        del z
    else:
        r = ci_one_world(out / f'reanchor_{stem}_{worlds[0]}.npz', worlds[0], [links[worlds[0]]], ci, extra_cols, a.workers)
        rec['stages']['mixed'] = dict(tool='ag_build_ds.ci_one_world (ga_build_mixed format, one world)', **{k: (len(v) if k == 'missing_episodes' else v) for k, v in r.items()})
        assert not r['missing_episodes'] and r['raw_found_fraction'] == 1.0, f'{len(r["missing_episodes"])} episodes without a raw run'

    # 4. checks on the final file
    z = np.load(ci, allow_pickle=True)
    ids = z['id'].astype(str); grp = z['group'].astype(str); ep = z['episode'].astype(str); dom = z['domain'].astype(int)
    sp = z['split'].astype(str); af = z['anchor_frame'].astype(int); tier = z['tier'].astype(int)
    assert len(set(ids)) == len(ids), 'duplicate ids'
    bad = sorted({s for s in set(ids) | set(grp) | set(ep) if suite_hit(s.split('@')[0])})
    assert not bad, f'suite / blacklisted ids in the training file: {bad[:5]}'
    assert set(z['arena'].astype(str)) == {a.arena} and set(z['vehicle'].astype(str)) == {a.vehicle}
    assert set(dom.tolist()) == {GBM.DOMAIN_CODE[w] for w in worlds}
    for k in ('X', 'hist', 'hmask', 'privileged', 'ctx'):
        assert z[k].shape[0] == len(ids), k
    rec['output'] = dict(path=str(ci), size_gb=round(os.path.getsize(ci) / 1e9, 3), sha256=sha256_file(ci), rows=int(len(ids)),
                         rows_by_world={w: int((dom == GBM.DOMAIN_CODE[w]).sum()) for w in worlds},
                         episodes_by_world={w: int(len(set(ep[dom == GBM.DOMAIN_CODE[w]]))) for w in worlds},
                         groups_by_split={s: int(len(set(grp[sp == s]))) for s in ('train', 'val', 'test')},
                         counts=row_counts(dom, sp, af), tiers={w: dict(sorted(Counter(tier[(dom == GBM.DOMAIN_CODE[w]) & (af == 0)].tolist()).items())) for w in worlds},
                         keys=sorted(z.files))
    del z
    if a.compare_to:
        rec['compare'] = compare(ci, a.compare_to)
        print('compare:', json.dumps({k: v for k, v in rec['compare'].items() if k != 'arrays_equal'}), flush=True)
        print('arrays equal:', rec['compare']['arrays_equal'], flush=True)
    if not a.keep_work:
        for L in links.values():
            shutil.rmtree(L, ignore_errors=True)
        rec['links_removed'] = True
    rec['elapsed_s'] = round(time.time() - t0, 1); rec['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
    json.dump(rec, open(out / f'{stem}_record.json', 'w'), indent=1, default=str)
    print(f'wrote {ci} ({rec["output"]["rows"]} rows: {rec["output"]["rows_by_world"]}; {rec["output"]["size_gb"]} GB) in {rec["elapsed_s"]} s', flush=True)
    print(json.dumps(rec['output']['counts']), flush=True)


if __name__ == '__main__':
    main()
