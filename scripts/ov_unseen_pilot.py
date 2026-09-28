#!/usr/bin/env python3
"""16-row pilot of the Polaris unseen-arena evaluation (offroad_vehicles_20260927, PLAN amendment 9.2).

  build    per unseen test arena, the lowest-md5 group of the declared soil subset (K3 suites/soil_unseen_subset.json,
           the declared order of every subset of the Gator study) gets two rows:
             - a stored-reference row: the Gator study's HMMWV M1a_free drive of that group (same case, same route file,
               vehicle hmmwv) re-driven through this study's frozen dispatcher: id <group>__u1ref_M1a_free, kind
               'refcheck', ref_run = the stored G3 run (its outcome must come back: end state = outcome status);
             - the Polaris polaris_u_grad row of that group, copied unchanged from the full evaluation rows file
               (ov_eval_tasks.py output), so the full launch later finds it complete.
           Writes <out> (16 rows), <out>.ref_rows.json (the 8 reference rows alone, for appending to the superset),
           <out>.staging.tsv (ov_eval_tasks.py stage format), <out>.meta.json. Paths G4-relative (ext/ for files
           outside K4), as ov_eval_tasks.py writes them.
  compare  after the pilot: per reference row the re-driven end state against the stored run's (outcome.json status),
           plus exact array equality (trajectory / crm_extra / anchor_state / command_reference npz, reported), and the
           Polaris rows' outcomes (status, goal, unsafe, belly flag). Rule (task U1): the reference holds if >= 7 of 8
           end states match.

  PY=/home/harry/miniconda3/envs/nedm/bin/python; export PYTHONPATH=src:scripts
  $PY scripts/ov_unseen_pilot.py build --eval-rows $K4/e6/tasks/soil_eval_polaris_unseen_v1.json \
      --existing $K4/tasks/soil_v6_polaris.json --out $K4/tasks/pilot_unseen_v1.json
  $PY scripts/ov_unseen_pilot.py compare --pilot $K4/tasks/pilot_unseen_v1.json --runs <sync of G4 soil_v1/runs> \
      --ref-runs <sync of the stored G3 runs> --out $K4/e6/unseen/pilot_compare.json
"""
import argparse, hashlib, json, os, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / 'src'))
import ag_eval_tasks as AET                # noqa: E402  (frozen: route_content_sha)
import ag_tasklib as L                     # noqa: E402  (md5_int)
import ov_eval_tasks as OET                # noqa: E402  (g4_rel, G3, K3, K4)

ARENAS = ('g260', 'g271', 'g251', 'g247', 'g258', 'g268', 'g263', 'g241')
SUBSET = OET.K3 / 'suites/soil_unseen_subset.json'
K3_MAPS = [OET.K3 / f'e6/tasks/soil_eval_p{i}.json.mapping.json' for i in (1, 2)]
REF_ARM = 'M1a_free'
REF_NAME = 'u1ref_M1a_free'
G3_RUNS = f'{OET.G3}/soil_v1/runs'


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def md5hex(s):
    return hashlib.md5(s.encode()).hexdigest()


def pilot_groups():
    sub = json.load(open(SUBSET))
    return {a: sorted(sub['arenas'][a]['groups'], key=md5hex)[0] for a in ARENAS}


def build(a):
    groups = pilot_groups()
    ev = json.load(open(a.eval_rows))
    used_seeds = {r['episode_seed'] for r in ev if r.get('episode_seed') is not None}
    used_ids = {r['id'] for r in ev}
    for f in a.existing or []:
        for r in json.load(open(f)):
            used_ids.add(r['id'])
            if r.get('episode_seed') is not None:
                used_seeds.add(r['episode_seed'])
    k3map = {}
    for mp in K3_MAPS:
        m = json.load(open(mp))
        for g, arms in m['groups'].items():
            if REF_ARM in arms and g not in k3map:
                k3map[g] = dict(arms[REF_ARM], mapping=str(mp.relative_to(ROOT)))
    ref_rows, pol_rows, stage = [], [], {}
    for ar, g in groups.items():
        # the stored HMMWV M1a_free drive of this group and its route file
        e = k3map[g]
        man = json.load(open(OET.K3 / f'e6/picks/crm/{ar}/{REF_ARM}/ag_picks.json'))
        pk = man['picks'][g]
        pdir = OET.K3 / f'e6/picks/crm/{ar}/{REF_ARM}'
        route = pdir / pk['route_file']
        assert sha256_file(route) == pk['file_sha256'], f'{route}: changed since the lock'
        sha = AET.route_content_sha(route)
        assert sha == pk['route_sha256'] == e['route_sha256'], f'{g}: route content differs from the stored drive'
        case = ROOT / man['cases_dir'] / f'{g}.json'
        rid = f'{g}__{REF_NAME}'
        assert rid not in used_ids, f'{rid} already used'
        k, seed = 0, L.md5_int(rid)
        while seed in used_seeds:
            k += 1; seed = L.md5_int(f'{rid}#{k}')
        used_seeds.add(seed)
        row = dict(id=rid, group=g, arena=ar, case=OET.g4_rel(case), route=OET.g4_rel(route), tier=a.tier, episode_seed=seed, run=True,
                   kind='refcheck', arms=[REF_NAME], sha256=sha, vehicle='hmmwv', extra=['--vehicle', 'hmmwv'], split='eval', world='crm',
                   ref_run=f"{G3_RUNS}/{e['run_id']}", ref_run_id=e['run_id'], ref_reused_from=e.get('reused_from'), ref_mapping=e['mapping'])
        if k:
            row['episode_seed_salt'] = k
        ref_rows.append(row)
        for loc, rel in ((case, row['case']), (route, row['route'])):
            stage[rel] = str(Path(loc).resolve())
        # the Polaris gradient-arm row of the same group, unchanged
        pr = [r for r in ev if r['group'] == g and r['vehicle'] == 'polaris' and 'polaris_u_grad' in r['arms']]
        assert len(pr) == 1, f'{g}: {len(pr)} polaris_u_grad rows in {a.eval_rows}'
        pol_rows.append(dict(pr[0]))
        for rel in (pr[0]['case'], pr[0]['route']):
            loc = OET.local_of(rel)
            assert loc is not None and loc.exists(), rel
            stage[rel] = str(loc.resolve())
    rows = ref_rows + pol_rows
    for r in rows:
        r['tier'] = a.tier
    out = Path(a.out)
    for p in (out, Path(str(out) + '.ref_rows.json')):
        assert not p.exists(), f'{p} exists: task files are never overwritten'
    json.dump(rows, open(out, 'w'))
    json.dump(ref_rows, open(str(out) + '.ref_rows.json', 'w'))
    with open(str(out) + '.staging.tsv', 'w') as f:
        f.write('local\tg4_path\tsha256\n')
        for rel, loc in sorted(stage.items()):
            f.write(f'{os.path.relpath(loc, ROOT)}\t{rel}\t{sha256_file(loc)}\n')
    meta = dict(tool='scripts/ov_unseen_pilot.py', tool_sha256=sha256_file(__file__), created=time.strftime('%F %T'), argv=sys.argv[1:],
                file_sha256=sha256_file(out), ref_rows_sha256=sha256_file(str(out) + '.ref_rows.json'), rows=len(rows),
                groups=groups, rule='per arena the lowest-md5 group of the declared subset (declared order)',
                eval_rows=dict(file=str(a.eval_rows), sha256=sha256_file(a.eval_rows)),
                existing={f: sha256_file(f) for f in a.existing or []}, by_vehicle=dict(Counter(r['vehicle'] for r in rows)),
                staging=dict(files=len(stage), list=str(out) + '.staging.tsv'),
                collector_contract=f'crm_worker.py with CRM_ROOT = {OET.G4}, CRM_COLLECTOR = {OET.COLLECTOR}, CRM_CONFIG = configs/crm_main.json, NEDM_VEHICLE unset')
    json.dump(meta, open(str(out) + '.meta.json', 'w'), indent=1)
    print(json.dumps(meta, indent=1))


def compare(a):
    import numpy as np
    import ov_smoke_analyze as SA           # npz_equal
    import ov_eval_index as OEI             # extras (QA, launch, belly)
    import ag_eval_index as AEI             # outcome labels
    rows = json.load(open(a.pilot))
    res = dict(tool='scripts/ov_unseen_pilot.py compare', tool_sha256=sha256_file(__file__), created=time.strftime('%F %T'),
               pilot=str(a.pilot), pilot_sha256=sha256_file(a.pilot), ref=[], polaris=[])
    for r in rows:
        d = Path(a.runs) / r['id']
        done = (d / 'episode_complete.json').exists()
        x = dict(id=r['id'], arena=r['arena'], group=r['group'], vehicle=r['vehicle'], complete=done)
        if done:
            o = AEI.outcome(d)
            x.update(status=o['status'], fail=o['fail'], unsafe=o['unsafe'], elapsed=o['elapsed'], vehicle_block=o['vehicle_block'])
            x.update({k: v for k, v in OEI.extras(d).items() if k in ('qa_ok', 'qa_flag', 'launch_ok', 'belly_flag', 'belly_min_m', 'belly_run_s')})
            try:
                ec = json.load(open(d / 'episode_complete.json')); x['sim_s'] = ec.get('actual_elapsed_s')
            except Exception:  # noqa: BLE001
                pass
        if r.get('kind') == 'refcheck':
            sd = Path(a.ref_runs) / r['ref_run_id']
            sdone = (sd / 'outcome.json').exists()
            x.update(ref_run=r['ref_run'], ref_complete=sdone)
            if done and sdone:
                so = AEI.outcome(sd)
                x.update(ref_status=so['status'], ref_fail=so['fail'], ref_elapsed=so['elapsed'], same_end_state=so['status'] == x['status'])
                eq, why = SA.npz_equal(str(d), str(sd))
                x.update(identical_arrays=eq, first_difference=why, arrays_compared=[n for n in ('trajectory.npz', 'crm_extra.npz', 'anchor_state.npz', 'command_reference.npz')
                                                                                      if (d / n).exists() and (sd / n).exists()])
            res['ref'].append(x)
        else:
            x['unsafe_belly'] = None if not done else int(x['unsafe'] or (x.get('belly_flag') or 0))
            res['polaris'].append(x)
    same = sum(1 for x in res['ref'] if x.get('same_end_state'))
    res['ref_summary'] = dict(n=len(res['ref']), compared=sum(1 for x in res['ref'] if 'same_end_state' in x), same_end_state=same,
                              identical_arrays=sum(1 for x in res['ref'] if x.get('identical_arrays')), holds=bool(same >= 7))
    pol = [x for x in res['polaris'] if x['complete']]
    res['polaris_summary'] = dict(n=len(res['polaris']), complete=len(pol), goal=sum(1 for x in pol if not x['fail']),
                                  reached_safely=sum(1 for x in pol if not x['unsafe_belly']), statuses=dict(Counter(x['status'] for x in pol)),
                                  belly_flags=sum(1 for x in pol if x.get('belly_flag')), launch_fail=sum(1 for x in pol if x.get('launch_ok') is False))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=str)
    L_ = [f"reference rows: same end state {same}/{res['ref_summary']['compared']} (rule >= 7 of 8: {'holds' if res['ref_summary']['holds'] else 'FAILS'}), "
          f"identical arrays {res['ref_summary']['identical_arrays']}"]
    for x in res['ref']:
        L_.append(f"  {x['id']}: {x.get('status')} vs stored {x.get('ref_status')} same {x.get('same_end_state')} arrays {x.get('identical_arrays')} {x.get('first_difference') or ''}")
    L_.append(f"Polaris rows: {res['polaris_summary']}")
    for x in res['polaris']:
        L_.append(f"  {x['id']}: {x.get('status')} goal {None if not x['complete'] else not x['fail']} safe {None if not x['complete'] else not x['unsafe_belly']} "
                  f"belly {x.get('belly_flag')} launch {x.get('launch_ok')} qa {x.get('qa_flag')} {x.get('elapsed')}")
    txt = '\n'.join(L_)
    Path(str(a.out).rsplit('.', 1)[0] + '.txt').write_text(txt + '\n')
    print(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    b = sub.add_parser('build')
    b.add_argument('--eval-rows', required=True); b.add_argument('--existing', nargs='*', default=[])
    b.add_argument('--out', required=True); b.add_argument('--tier', type=int, default=-10)
    c = sub.add_parser('compare')
    c.add_argument('--pilot', required=True); c.add_argument('--runs', required=True); c.add_argument('--ref-runs', required=True)
    c.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    assert not os.environ.get('NEDM_VEHICLE'), 'NEDM_VEHICLE is set'
    dict(build=build, compare=compare)[a.cmd](a)


if __name__ == '__main__':
    main()
