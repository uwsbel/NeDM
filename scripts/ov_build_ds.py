#!/usr/bin/env python3
"""Training file for one (arena, vehicle, world) from collected episodes, for ANY vehicle (offroad_vehicles_20260927,
module M4; PLAN 4.1). Vehicle-generalised copy of scripts/ag_build_ds.py (frozen; its functions are imported, not
copied: inspect, the thin one-world cutter ci_one_world, append_columns, compare, the suite patterns and the f104 tier
order). Differences from ag_build_ds.py, and only these:
  - --vehicle takes any name of VEHICLE_PREFIX (hmmwv, gator, gatorctl-style re-drives via --id-prefix, polaris and its
    sensitivity arms, m113, m113_g4); the run-id prefix defaults to '<vehicle>__' ('' for the HMMWV);
  - the vehicle-block check is generic: every selected run's outcome.json vehicle block must name --vehicle; HMMWV runs
    may carry no block (the collect_v1 / arena_gator runs) or a block named 'hmmwv' (the new dispatcher
    scripts/ov_crm_collect.py writes one for every vehicle); --require-vehicle-block yes also refuses block-less HMMWV
    runs. ag_build_ds.py's rule (Gator runs carry 'gator', HMMWV runs none) is the special case;
  - --compare-subset REF.npz: when the new file is built from a SUBSET of REF's episodes (e.g. a local copy of part of
    the runs), REF's rows are restricted to the new file's episodes and every array is compared by row id.
Everything else is ag_build_ds.py's pipeline and checks, unchanged: selection by id pattern + training task rows
(tier >= 0, same arena and vehicle, task tier = the crm_tasks per-group order), completion marker, launch check +
crm_qa.check (soil), map / arena check (BMP sha256), refusal of any suite / test-pool / eval / dev / held-out id or
group (ga_build_mixed.BLACKLIST + generic suite patterns; every group must be a training-pool group
<arena>_v2_group_NNNN of the arena), unique ids, then f104_n2_dataset.py -> n2_reanchor_dataset.py -> one-world ci file
(+ arena, vehicle, tier columns). Writes <out>/<stem>_record.json.

  PYTHONPATH=src:scripts python scripts/ov_build_ds.py --world crm --vehicle polaris --arena f104 \
     --crm-runs $G4/soil/runs --tasks-crm $G4/tasks/soil_polaris.json --tiers 0-6 \
     --map-root $G3/e4/map_roots/f104 --source-root $G4/source --out $G4/e4/soil_s1/f104_polaris --workers 16
numpy only (runs on the cluster login node or a compute node too).
"""
import argparse, json, os, re, shutil, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ag_build_ds as B                  # noqa: E402  (frozen; reused functions)
import ag_map_check                      # noqa: E402

GBM = B.GBM
WORLDS = B.WORLDS
VEHICLES = ('hmmwv', 'gator', 'polaris', 'polaris_pc', 'polaris_4wd', 'polaris_w08', 'm113', 'm113_g4')
VEHICLE_PREFIX = {v: ('' if v == 'hmmwv' else f'{v}__') for v in VEHICLES}
TOOLS = ['f104_n2_dataset.py', 'n2_reanchor_dataset.py', 'ga_build_mixed.py', 'ag_build_ds.py', 'ov_build_ds.py', 'ag_map_check.py',
         'ag_tasklib.py', 'crm_qa.py']


def vehicle_ok(block_name, vehicle, require_block):
    """The run's vehicle block names the requested vehicle; HMMWV runs may carry none unless a block is required."""
    if block_name is None:
        return vehicle == 'hmmwv' and not require_block
    return block_name == vehicle


def compare_subset(new_path, ref_path):
    """REF restricted to the new file's (domain, episode) pairs; row ids and every per-row array compared by id."""
    A = np.load(new_path, allow_pickle=True); Bz = np.load(ref_path, allow_pickle=True)
    ida, idb = A['id'].astype(str), Bz['id'].astype(str)
    epa = set(zip(A['domain'].astype(int).tolist(), A['episode'].astype(str)))
    selb = np.array([(int(d), e) in epa for d, e in zip(Bz['domain'], Bz['episode'].astype(str))])
    posb = {s: i for i, s in enumerate(idb)}
    res = dict(ref=os.path.abspath(ref_path), ref_rows=int(len(idb)), new_rows=int(len(ida)), new_episodes=len(epa),
               ref_rows_on_new_episodes=int(selb.sum()), ref_episodes_found=int(len({e for d, e in zip(Bz['domain'][selb], Bz['episode'][selb].astype(str))})))
    set_a, set_b = set(ida), set(idb[selb])
    res['new_ids_not_in_ref'] = sorted(set_a - set(idb))[:10]; res['n_new_ids_not_in_ref'] = len(set_a - set(idb))
    res['ref_ids_on_new_episodes_missing_in_new'] = sorted(set_b - set_a)[:10]; res['n_ref_ids_missing_in_new'] = len(set_b - set_a)
    common = [i for i in ida if i in posb]
    ia = np.array([i for i, s in enumerate(ida) if s in posb]); ib = np.array([posb[s] for s in common])
    eq = {}
    for k in sorted(set(A.files) & set(Bz.files)):
        a, b = A[k], Bz[k]
        if a.ndim == 0 or a.shape[0] != len(ida) or b.shape[0] != len(idb):
            eq[k] = bool(np.array_equal(a, b)) if a.shape == b.shape else False; continue
        a, b = a[ia], b[ib]
        if a.dtype == object or b.dtype == object:
            eq[k] = bool(np.array_equal(a.astype(str), b.astype(str)))
        else:
            eq[k] = bool(np.array_equal(a, b, equal_nan=np.issubdtype(a.dtype, np.floating)))
            if not eq[k] and np.issubdtype(a.dtype, np.floating):
                eq[k + '_max_abs_diff'] = float(np.nanmax(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    res['keys_only_new'] = sorted(set(A.files) - set(Bz.files)); res['keys_only_ref'] = sorted(set(Bz.files) - set(A.files))
    res['ids_compared'] = len(common); res['arrays_equal'] = eq
    res['identical'] = bool(res['n_new_ids_not_in_ref'] == 0 and res['n_ref_ids_missing_in_new'] == 0 and not res['keys_only_new'] and not res['keys_only_ref']
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
    ap.add_argument('--tasks', nargs='*', default=[]); ap.add_argument('--tasks-rigid', nargs='*', default=[]); ap.add_argument('--tasks-crm', nargs='*', default=[])
    ap.add_argument('--tiers', default=None, help='keep tiers A-B (inclusive)')
    ap.add_argument('--exclude-ids-rigid', nargs='*', default=[]); ap.add_argument('--exclude-ids-crm', nargs='*', default=[])
    ap.add_argument('--require-marker', choices=['auto', 'yes', 'no'], default='auto')
    ap.add_argument('--require-vehicle-block', choices=['auto', 'yes', 'no'], default='auto',
                    help="auto: yes for every vehicle but the HMMWV (whose older runs carry no block)")
    ap.add_argument('--id-prefix', default=None, help="run id prefix (default '<vehicle>__', '' for the HMMWV; e.g. gatorctl__ with --vehicle gator)")
    ap.add_argument('--out', required=True); ap.add_argument('--stem', default=None)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--compare-to', default=None, help='REF built from a superset or the same runs: ag_build_ds.compare (new restricted to REF)')
    ap.add_argument('--compare-subset', default=None, help='REF built from a superset of these runs: REF restricted to the new episodes')
    ap.add_argument('--keep-work', action='store_true')
    a = ap.parse_args(argv)
    t0 = time.time()
    worlds = list(WORLDS) if a.world == 'both' else [a.world]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); (out / 'logs').mkdir(exist_ok=True)
    stem = a.stem or f'{a.arena}_{a.vehicle}_{a.world}'
    prefix = VEHICLE_PREFIX[a.vehicle] if a.id_prefix is None else a.id_prefix
    require_block = a.require_vehicle_block == 'yes' or (a.require_vehicle_block == 'auto' and a.vehicle != 'hmmwv')
    pat = re.compile(rf'^{re.escape(prefix)}({re.escape(a.arena)}_v2_group_\d{{4}}_(route|op)_\d{{2}})$')
    tiers = B.parse_tiers(a.tiers)
    task_rows_by = dict(rigid=B.load_task_rows(a.tasks + a.tasks_rigid), crm=B.load_task_rows(a.tasks + a.tasks_crm))
    exclude_by = dict(rigid=B.load_exclude(a.exclude_ids_rigid), crm=B.load_exclude(a.exclude_ids_crm))
    obs = json.load(open(Path(a.map_root) / 'static_map_v1' / 'observation.json'))
    rec = dict(tool='scripts/ov_build_ds.py', argv=sys.argv[1:] if argv is None else argv, started=time.strftime('%Y-%m-%d %H:%M:%S'),
               host=os.uname().nodename, python=sys.executable, arena=a.arena, vehicle=a.vehicle, id_prefix=prefix, require_vehicle_block=require_block,
               worlds=worlds, map_root=os.path.abspath(a.map_root),
               map_observation_sha256=obs.get('observation_sha256'), map_arena_bmp_sha256=obs.get('arena_bmp_sha256'),
               tasks={w: {p: B.sha256_file(p) for p in a.tasks + getattr(a, f'tasks_{w}')} for w in WORLDS}, exclude_ids={w: sorted(v)[:50] for w, v in exclude_by.items()},
               n_exclude_ids={w: len(v) for w, v in exclude_by.items()}, tiers=a.tiers,
               tool_sha256={t: B.sha256_file(HERE / t) for t in TOOLS}, selection={}, stages={})
    env = dict(os.environ); env['PYTHONPATH'] = os.pathsep.join([str(HERE.parent / 'src'), str(HERE)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    env.setdefault('OMP_NUM_THREADS', '1')
    env.pop('NEDM_VEHICLE', None)
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
                if not pat.match(e.name):
                    cnt['other_ids'] += 1; continue
                assert e.name not in seen, f'id {e.name} in two run folders: {seen[e.name]} and {r}'
                seen[e.name] = os.path.join(r, e.name)
        cand = {}
        for rid, d in sorted(seen.items()):
            base = pat.match(rid).group(1)
            t_order = B.f104_order_tier(base)
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
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            info = list(ex.map(B.inspect, [(d, w, require) for d, _ in cand.values()], chunksize=64))
        rejected = Counter(i['reason'] for i in info if not i['ok'])
        ok = [i for i in info if i['ok']]
        assert ok, f'[{w}] no episode selected (prefix {prefix!r}, counts {dict(cnt)}, rejected {dict(rejected)})'
        # vehicle provenance (generic): the block names --vehicle; HMMWV runs may carry none unless a block is required
        vbad = [(i['id'], i['vehicle']) for i in ok if not vehicle_ok(i['vehicle'], a.vehicle, require_block)]
        assert not vbad, f'{len(vbad)} runs whose vehicle block does not match --vehicle {a.vehicle}: {vbad[:5]}'
        arenas = Counter(B.arena_short(i['arena']) for i in ok)
        assert set(arenas) == {a.arena}, f'runs name other arenas: {dict(arenas)}'
        bad_ids = [i['id'] for i in ok if B.suite_hit(i['id']) or B.suite_hit(i['group'] or '')]
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
        print(f'[{w}] {len(ok)} episodes selected ({dict(cnt)}; rejected {dict(rejected)}); vehicle blocks {dict(Counter(str(i["vehicle"]) for i in ok))}; '
              f'map check ok ({mc["map_arena_bmp_sha256"][:16]})', flush=True)
        st = out / f'station_{stem}_{w}.npz'
        rc, secs = B.run_logged([sys.executable, '-u', HERE / 'f104_n2_dataset.py', '--root', a.map_root, '--runs', f'{L}/*_route_*:designed', f'{L}/*_op_*:on_policy',
                                 '--out', st, '--workers', a.workers], out / 'logs' / f'station_{stem}_{w}.log', env)
        assert rc == 0, f'f104_n2_dataset.py failed ({rc}), see logs'
        sid = np.load(st, allow_pickle=True)['id'].astype(str)
        assert len(sid) == len(set(sid)) and set(sid) == {i['id'] for i in ok}, f'station rows {len(sid)} != selected {len(ok)} (unlabelled episodes)'
        ra = out / f'reanchor_{stem}_{w}.npz'
        rc2, secs2 = B.run_logged([sys.executable, '-u', HERE / 'n2_reanchor_dataset.py', '--root', a.map_root, '--ids', st, '--runs', L, '--out', ra,
                                   '--workers', a.workers], out / 'logs' / f'reanchor_{stem}_{w}.log', env)
        assert rc2 == 0, f'n2_reanchor_dataset.py failed ({rc2}), see logs'
        z = np.load(ra, allow_pickle=True); rep = z['episode'].astype(str); raf = z['anchor_frame'].astype(int)
        assert set(rep[raf == 0]) == set(sid) and (raf == 0).sum() == len(sid), 'an episode without its k = 0 row'
        rec['stages'][w] = dict(station=dict(path=str(st), rows=int(len(sid)), secs=secs, sha256=B.sha256_file(st)),
                                reanchor=dict(path=str(ra), rows=int(len(rep)), episodes=int(len(set(rep))), secs=secs2, sha256=B.sha256_file(ra),
                                              rows_per_episode=round(len(rep) / max(len(sid), 1), 3)))
        print(f'[{w}] station {len(sid)} rows ({secs} s), re-anchored {len(rep)} rows ({secs2} s)', flush=True)
        del z

    ci = out / f'ci_{stem}.npz'

    def extra_cols(d):
        dom = d['domain'].astype(int); ep = d['episode'].astype(str)
        wname = {0: 'rigid', 1: 'crm'}
        tier = np.array([tier_of[(wname[x], e)] for x, e in zip(dom, ep)], np.int16)
        return dict(arena=np.array([a.arena] * len(ep), object), vehicle=np.array([a.vehicle] * len(ep), object), tier=tier)

    if len(worlds) == 2:
        rc3, secs3 = B.run_logged([sys.executable, '-u', HERE / 'ga_build_mixed.py', '--rigid', out / f'reanchor_{stem}_rigid.npz', '--crm', out / f'reanchor_{stem}_crm.npz',
                                   '--rigid-runs', links['rigid'], '--crm-runs', links['crm'], '--out', ci, '--workers', min(a.workers, 8)],
                                  out / 'logs' / f'mixed_{stem}.log', env)
        bj = json.load(open(out / f'ci_{stem}_build.json'))
        rec['stages']['mixed'] = dict(tool='ga_build_mixed.py', exit=rc3, secs=secs3, missing_episodes=len(bj['missing_episodes']),
                                      raw_found_fraction=bj['raw_found_fraction'], build_json=str(out / f'ci_{stem}_build.json'))
        assert rc3 == 0 and not bj['missing_episodes'] and bj['raw_found_fraction'] == 1.0, f'ga_build_mixed exit {rc3}, missing {len(bj["missing_episodes"])} episodes'
        z = np.load(ci, allow_pickle=True)
        B.append_columns(ci, extra_cols({k: z[k] for k in ('domain', 'episode')}))
        del z
    else:
        r = B.ci_one_world(out / f'reanchor_{stem}_{worlds[0]}.npz', worlds[0], [links[worlds[0]]], ci, extra_cols, a.workers)
        rec['stages']['mixed'] = dict(tool='ag_build_ds.ci_one_world (ga_build_mixed format, one world)', **{k: (len(v) if k == 'missing_episodes' else v) for k, v in r.items()})
        assert not r['missing_episodes'] and r['raw_found_fraction'] == 1.0, f'{len(r["missing_episodes"])} episodes without a raw run'

    z = np.load(ci, allow_pickle=True)
    ids = z['id'].astype(str); grp = z['group'].astype(str); ep = z['episode'].astype(str); dom = z['domain'].astype(int)
    sp = z['split'].astype(str); af = z['anchor_frame'].astype(int); tier = z['tier'].astype(int)
    assert len(set(ids)) == len(ids), 'duplicate ids'
    bad = sorted({s for s in set(ids) | set(grp) | set(ep) if B.suite_hit(s.split('@')[0])})
    assert not bad, f'suite / blacklisted ids in the training file: {bad[:5]}'
    assert set(z['arena'].astype(str)) == {a.arena} and set(z['vehicle'].astype(str)) == {a.vehicle}
    assert set(dom.tolist()) == {GBM.DOMAIN_CODE[w] for w in worlds}
    for k in ('X', 'hist', 'hmask', 'privileged', 'ctx'):
        assert z[k].shape[0] == len(ids), k
    rec['output'] = dict(path=str(ci), size_gb=round(os.path.getsize(ci) / 1e9, 3), sha256=B.sha256_file(ci), rows=int(len(ids)),
                         rows_by_world={w: int((dom == GBM.DOMAIN_CODE[w]).sum()) for w in worlds},
                         episodes_by_world={w: int(len(set(ep[dom == GBM.DOMAIN_CODE[w]]))) for w in worlds},
                         groups_by_split={s: int(len(set(grp[sp == s]))) for s in ('train', 'val', 'test')},
                         counts=B.row_counts(dom, sp, af), tiers={w: dict(sorted(Counter(tier[(dom == GBM.DOMAIN_CODE[w]) & (af == 0)].tolist()).items())) for w in worlds},
                         keys=sorted(z.files))
    del z
    if a.compare_to:
        rec['compare'] = B.compare(ci, a.compare_to)
        print('compare:', json.dumps({k: v for k, v in rec['compare'].items() if k != 'arrays_equal'}), flush=True)
        print('arrays equal:', rec['compare']['arrays_equal'], flush=True)
    if a.compare_subset:
        rec['compare_subset'] = compare_subset(ci, a.compare_subset)
        print('compare_subset:', json.dumps({k: v for k, v in rec['compare_subset'].items() if k != 'arrays_equal'}), flush=True)
        print('arrays equal:', rec['compare_subset']['arrays_equal'], flush=True)
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
