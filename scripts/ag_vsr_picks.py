#!/usr/bin/env python3
"""arena_gator_20260925, skeptical check of the soil results: pick locks, route identity and the content-based reuse.

Own code (hashlib + json only). Checks:
  1. LOCK_crm.sha256: every line re-derived from the pick directory now (routes lock = sha256 over sorted route file
     names + their sha256 digests, ga_planner's scheme; manifest sha256) and the ALL line.
  2. Every soil mapping entry (group, arm): its route hash equals the pick manifest's route hash, the pick's route file
     is unchanged (bytes), and the drive's task row names a route file whose drive-relevant content (waypoints, speeds,
     stations, headings: the only fields crm_collect / the frozen read_route and make_driver read) equals the pick's.
  3. Reused soil_v2 drives: same group, vehicle (HMMWV, no Gator switch), case file, and drive-relevant route content;
     how many arm results, distinct drives, and of which headroom kind.
  4. soil_v3.json = soil_v2 prefix + the 12,000 evaluation rows, and every planned new drive is a soil_v3 row.

  python scripts/ag_vsr_picks.py --out artifacts/traverse/arena_gator_20260925/verify_soil_results
"""
import argparse, hashlib, json, os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
FIELDS = ('waypoints', 'speeds', 'stations', 'headings')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def drive_content(p):
    v = json.loads(Path(p).read_text())
    if 'route' in v:
        v = v['route']
    return json.dumps({k: v[k] for k in FIELDS}, sort_keys=True)


def loc(p):
    return ROOT / p[4:] if p.startswith('ext/') else K3 / p


def lk(d):
    h = hashlib.sha256()
    for p in sorted((d / 'routes').glob('*.json')):
        h.update(p.name.encode()); h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    R = {}
    # 1. lock file
    lines = (K3 / 'e6/picks/LOCK_crm.sha256').read_text().splitlines()
    body, last = lines[:-1], lines[-1]
    bad = []
    for ln in body:
        l1, ms, d = ln.split('  ')
        dd = ROOT / d
        if lk(dd) != l1 or sha(dd / 'ag_picks.json') != ms or (dd / 'PICKS_LOCKED.sha256').read_text().split()[0] != l1:
            bad.append(d)
    allh = hashlib.sha256(('\n'.join(body) + '\n').encode()).hexdigest()
    st = os.stat(K3 / 'e6/picks/LOCK_crm.sha256')
    created = sorted(json.load(open(ROOT / ln.split('  ')[2] / 'ag_picks.json'))['created'] for ln in body)
    R['lock'] = dict(dirs=len(body), dirs_not_matching_now=bad, all_line=last.split()[0], all_recomputed=allh, all_equal=last.split()[0] == allh,
                     lock_file_mtime=st.st_mtime, manifests_created_first_last=[created[0], created[-1]])
    # 2. mapping entries vs picks and task rows
    rows = {}
    for f in ['e3/tasks/soil_v2.json'] + [f'e6/tasks/soil_eval_p{i}.json' for i in range(1, 6)]:
        for r in json.load(open(K3 / f)):
            assert r['id'] not in rows or f == 'e3/tasks/soil_v2.json', r['id']
            rows.setdefault(r['id'], dict(r, _file=f))
    man_cache = {}
    n = 0; e_route = []; e_file = []; e_content = []; e_case = []; e_group = []; reuse = []
    for i in range(1, 6):
        m = json.load(open(K3 / f'e6/tasks/soil_eval_p{i}.json.mapping.json'))
        dir_of = {}
        for arm in m['arms']:
            for d, ar in zip(arm['dirs'], arm['arenas']):
                dir_of[(arm['name'], ar)] = ROOT / d
        for g, arms in m['groups'].items():
            for name, e in arms.items():
                n += 1
                d = dir_of[(name, e['arena'])]
                if d not in man_cache:
                    man_cache[d] = json.load(open(d / 'ag_picks.json'))
                pk = man_cache[d]['picks'][g]
                if pk['route_sha256'] != e['route_sha256']:
                    e_route.append((g, name))
                pf = d / pk['route_file']
                if sha(pf) != pk['file_sha256']:
                    e_file.append((g, name))
                row = rows[e['run_id']]
                rp = loc(row['route'])
                if drive_content(rp) != drive_content(pf):
                    e_content.append((g, name, e['run_id']))
                if row['group'] != g:
                    e_group.append((g, name, e['run_id']))
                if loc(row['case']).resolve() != loc(e['case']).resolve() and sha(loc(row['case'])) != sha(loc(e['case'])):
                    e_case.append((g, name, e['run_id']))
                if e.get('reused_from') and 'soil_v2' in e['reused_from']:
                    reuse.append(dict(group=g, arm=name, run_id=e['run_id'], kind=row.get('kind'), veh_row=row.get('vehicle', 'hmmwv'),
                                      extra=row.get('extra'), mapping_vehicle=e['vehicle'], row_arms=row.get('arms'),
                                      file_bytes_equal=sha(rp) == sha(pf)))
    R['entries'] = n
    R['mapping_vs_pick_route_hash_mismatch'] = e_route
    R['pick_route_file_changed'] = e_file
    R['task_row_route_content_differs_from_pick'] = e_content
    R['task_row_group_mismatch'] = e_group
    R['task_row_case_mismatch'] = e_case
    R['reuse_soil_v2'] = dict(arm_results=len(reuse), distinct_drives=len({x['run_id'] for x in reuse}),
                              by_arm=dict(Counter(x['arm'] for x in reuse)), by_drive_kind=dict(Counter(x['run_id'].split('__')[1] for x in reuse)),
                              by_drive_kind_distinct=dict(Counter(r.split('__')[1] for r in {x['run_id'] for x in reuse})),
                              row_kinds=dict(Counter(x['kind'] for x in reuse)), gator_rows=sum(1 for x in reuse if x['veh_row'] != 'hmmwv' or x['mapping_vehicle'] != 'hmmwv'),
                              rows_with_extra=sum(1 for x in reuse if x['extra']), file_bytes_equal=sum(x['file_bytes_equal'] for x in reuse),
                              model_arms_on_straight6_drive=sum(1 for x in reuse if x['arm'] != 'straight6' and x['run_id'].endswith('__straight6')),
                              model_arms_on_frozen_model_drive=sum(1 for x in reuse if x['run_id'].endswith('__Scrm_B')),
                              arenas=dict(Counter(x['group'].split('_')[0] for x in reuse)))
    # 4. soil_v3 composition
    v2 = json.load(open(K3 / 'e3/tasks/soil_v2.json')); v3 = json.load(open(K3 / 'e3/tasks/soil_v3.json'))
    ev = [r for f in range(1, 6) for r in json.load(open(K3 / f'e6/tasks/soil_eval_p{f}.json'))]
    R['soil_v3'] = dict(rows=len(v3), sha256=sha(K3 / 'e3/tasks/soil_v3.json'), v2_prefix_equal=v3[:len(v2)] == v2,
                        tail_equals_eval_rows=v3[len(v2):] == ev, eval_rows=len(ev), eval_tiers=dict(Counter(r['tier'] for r in ev)),
                        eval_vehicles=dict(Counter(r['vehicle'] for r in ev)))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    json.dump(R, open(out / 'picks_reuse.json', 'w'), indent=1, default=str)
    print(json.dumps({k: (v if not isinstance(v, list) else (len(v), v[:3])) for k, v in R.items()}, indent=1, default=str))


if __name__ == '__main__':
    main()
