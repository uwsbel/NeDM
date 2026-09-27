#!/usr/bin/env python3
"""arena_gator_20260925, skeptical check of the soil results: descriptive read-outs + the login-node results.

Own code. Local part (e6/runs_soil): unsafe vs fail per arm, tilt > 30 deg, median paired time ratio on joint
successes, mean predicted risk of the picks, simulated hours per vehicle, drive route bytes = the planned route file
(episode_complete's reference.json hash). Cluster part (verify_soil_results/cluster/{eval,gator,pilot}.json from
scripts/ag_vsr_cluster.py): QA of the evaluation drives, first drive time vs the pick lock, collector records, belly
flags, the Gator collection read-out (validated, belly, failure by speed profile / tier / split vs the HMMWV twins,
simulated hours) and the pilot wheel-radius sensitivity.

  PYTHONPATH=src:scripts python scripts/ag_vsr_readouts.py --dir artifacts/traverse/arena_gator_20260925/verify_soil_results
"""
import argparse, hashlib, json, os, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
K3 = ROOT / 'artifacts/traverse/arena_gator_20260925'
RUNS = K3 / 'e6/runs_soil'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def loc(p):
    return ROOT / p[4:] if p.startswith('ext/') else K3 / p


def labels(d):
    o = json.load(open(d / 'outcome.json')); z = np.load(d / 'trajectory.npz')
    st, ac = z['state'], z['action']
    goal = o['status'] == 'goal_reached'
    if len(st) < 22:
        return dict(fail=int(not goal), unsafe=1, tilt30=0, elapsed=float(o['elapsed_s']))
    vx = st[20:, 0]; thr = ac[20:, 1]
    back = float(((vx < -0.10) & (thr > 0.3)).sum()) * 0.05
    tilt = float(np.degrees(np.abs(st[20:, 2:4])).max())
    return dict(fail=int(not goal), unsafe=int(not (goal and back < 0.05 and vx.min() > -0.30)), tilt30=int(tilt > 30.0), elapsed=float(o['elapsed_s']))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--dir', required=True); a = ap.parse_args()
    D = Path(a.dir); R = {}
    plan = {}
    rows = {}
    for f in ['e3/tasks/soil_v2.json'] + [f'e6/tasks/soil_eval_p{i}.json' for i in range(1, 6)]:
        for r in json.load(open(K3 / f)):
            rows.setdefault(r['id'], r)
    for i in range(1, 6):
        m = json.load(open(K3 / f'e6/tasks/soil_eval_p{i}.json.mapping.json'))
        for g, arms in m['groups'].items():
            for arm, e in arms.items():
                plan[(e['vehicle'], g, arm)] = e
    lab = {rid: labels(RUNS / rid) for rid in {e['run_id'] for e in plan.values()}}
    # drive used the planned route bytes: episode_complete lists reference.json (the collector's copy of --route)
    bad_ref = []
    for rid in lab:
        ec = json.load(open(RUNS / rid / 'episode_complete.json'))['artifacts_sha256']
        if ec.get('reference.json') != sha(loc(rows[rid]['route'])):
            bad_ref.append(rid)
    R['reference_json_equals_task_route_file'] = dict(checked=len(lab), mismatch=len(bad_ref), examples=bad_ref[:5])
    sub = json.load(open(K3 / 'suites/soil_unseen_subset.json'))['groups']
    indist = json.load(open(K3 / 'suites/f104_indist_200.json'))['groups']
    s800 = sorted({g for (v, g, arm) in plan if v == 'gator'})
    sets = dict(unseen=sub, indist=indist, f104_800=s800, heldout=sorted({g for (v, g, arm) in plan if '_heldout_' in g}))
    # unsafe - fail, tilt per arm and set
    tab = {}
    for (v, g, arm), e in plan.items():
        for sn, G in sets.items():
            pass
    per = defaultdict(list)
    for (v, g, arm), e in plan.items():
        s = 'unseen' if '_test_group_' in g else 'heldout' if '_heldout_' in g else 'f104_800'
        per[(v, arm, s)].append(lab[e['run_id']] | dict(P=e.get('P')))
        if g in set(indist):
            per[(v, arm, 'indist')].append(lab[e['run_id']] | dict(P=e.get('P')))
    for k, xs in sorted(per.items()):
        f = 100 * np.mean([x['fail'] for x in xs]); u = 100 * np.mean([x['unsafe'] for x in xs]); t = 100 * np.mean([x['tilt30'] for x in xs])
        Ps = [x['P'] for x in xs if x['P'] is not None]
        tab['|'.join(k)] = dict(n=len(xs), fail=round(f, 2), unsafe=round(u, 2), unsafe_minus_fail=round(u - f, 2), tilt30=round(t, 2),
                                mean_P_pct=round(100 * float(np.mean(Ps)), 2) if Ps else None)
    R['per_arm'] = tab
    R['max_unsafe_minus_fail'] = max(v['unsafe_minus_fail'] for v in tab.values())

    def F(v, g, arm, key='fail'):
        if arm in ('M1', 'M3'):
            a_, b_ = ('M1a_free', 'M1b_free') if arm == 'M1' else ('M3a_free', 'M3b_free')
            return (lab[plan[(v, g, a_)]['run_id']][key] + lab[plan[(v, g, b_)]['run_id']][key]) / 2
        return lab[plan[(v, g, arm)]['run_id']][key]
    R['unsafe_P1_P2'] = dict(P1=100 * (np.mean([F('hmmwv', g, 'M3', 'unsafe') for g in sub]) - np.mean([F('hmmwv', g, 'M1', 'unsafe') for g in sub])),
                             P2=100 * (np.mean([F('hmmwv', g, 'A3_free', 'unsafe') for g in sub]) - np.mean([F('hmmwv', g, 'M1', 'unsafe') for g in sub])))

    def tr(v, t, r, G):
        js = [g for g in G if not F(v, g, t) and not F(v, g, r)]
        ratio = [lab[plan[(v, g, t)]['run_id']]['elapsed'] / lab[plan[(v, g, r)]['run_id']]['elapsed'] for g in js]
        return dict(n=len(js), median_ratio=round(float(np.median(ratio)), 3),
                    median_t=float(np.median([lab[plan[(v, g, t)]['run_id']]['elapsed'] for g in js])),
                    median_r=float(np.median([lab[plan[(v, g, r)]['run_id']]['elapsed'] for g in js])))
    R['time_ratio'] = dict(M3a_vs_M1a=tr('hmmwv', 'M3a_free', 'M1a_free', sub), A3_vs_M1a=tr('hmmwv', 'A3_free', 'M1a_free', sub),
                           M1a_vs_straight6=tr('hmmwv', 'M1a_free', 'straight6', sub), G_vs_H_gator=tr('gator', 'G_free_gator', 'H_free_gator', s800))
    # simulated hours of the distinct evaluation drives
    vh = Counter()
    for rid, x in lab.items():
        vh['gator' if rid.startswith('gator__') else 'hmmwv'] += x['elapsed']
    R['eval_sim_hours'] = {k: round(v / 3600, 2) for k, v in vh.items()}
    R['eval_sim_hours_new_drives_only'] = {k: round(sum(x['elapsed'] for rid, x in lab.items() if (rid.startswith('gator__')) == (k == 'gator') and
                                                        rows[rid].get('kind') == 'eval') / 3600, 2) for k in ('gator', 'hmmwv')}
    # ---------------- cluster outputs
    C = D / 'cluster'
    if (C / 'eval.json').exists():
        ev = {r['id']: r for r in json.load(open(C / 'eval.json'))['rows']}
        new = [r for rid, r in ev.items() if rows[rid].get('kind') == 'eval']
        lock_t = os.stat(K3 / 'e6/picks/LOCK_crm.sha256').st_mtime
        spec_t = os.stat(K3 / 'e6/analysis/spec_soil_v1.json').st_mtime
        first = min(r['req_mtime'] for r in new)
        loc_ok = 0; loc_bad = []
        for rid, r in ev.items():
            ec = json.load(open(RUNS / rid / 'episode_complete.json'))
            if r['ec_sha'] == sha(RUNS / rid / 'episode_complete.json') and r['outcome_sha'] == sha(RUNS / rid / 'outcome.json'):
                loc_ok += 1
            else:
                loc_bad.append(rid)
        route_bad = [rid for rid, r in ev.items() if not (r['route_sha256'] == r['reference_sha'] == r['route_file_now_sha'] == sha(loc(rows[rid]['route'])))]
        case_bad = [rid for rid, r in ev.items() if r['case_sha256'] != sha(loc(rows[rid]['case']))]
        gat = [r for r in ev.values() if r['id'].startswith('gator__')]
        bel = [r for r in gat if r.get('belly')]
        R['eval_cluster'] = dict(
            runs=len(ev), new_eval_drives=len(new), qa_flags=dict(Counter(r['flag'] for r in ev.values() if not r['ok'])),
            qa_flag_ids=[(r['id'], r['flag']) for r in ev.values() if not r['ok']],
            configs=dict(Counter(f"{r['config']}|{r['step']}|{r['spacing']}" for r in ev.values())),
            local_copy_equal_cluster=loc_ok, local_copy_differs=loc_bad[:5],
            route_bytes_mismatch=len(route_bad), route_bad=route_bad[:5], case_bytes_mismatch=len(case_bad),
            gator_runs=len(gat), gator_wheels=dict(Counter(str(r.get('wheel')) for r in gat)), gator_spawn=dict(Counter(str(r.get('spawn_dz')) for r in gat)),
            gator_wrapper_switch=dict(Counter(f"{str(r.get('wrapper_sha'))[:8]}|{str(r.get('switch_sha'))[:8]}" for r in gat)),
            hmmwv_with_vehicle_block=sum(1 for r in ev.values() if not r['id'].startswith('gator__') and r.get('vehicle')),
            lock_mtime=time.strftime('%F %T', time.localtime(lock_t)), spec_mtime=time.strftime('%F %T', time.localtime(spec_t)),
            first_new_drive_request=time.strftime('%F %T', time.localtime(first)),
            last_new_drive_outcome=time.strftime('%F %T', time.localtime(max(r['out_mtime'] for r in new))),
            reused_drive_requests=[time.strftime('%F %T', time.localtime(min(r['req_mtime'] for rid, r in ev.items() if rows[rid].get('kind') != 'eval'))),
                                   time.strftime('%F %T', time.localtime(max(r['req_mtime'] for rid, r in ev.items() if rows[rid].get('kind') != 'eval')))],
            lock_before_first_drive=lock_t < first, spec_before_first_drive=spec_t < first)
        # belly flags of the Gator evaluation drives, per arm
        arm_of = defaultdict(list)
        for (v, g, arm), e in plan.items():
            if v == 'gator':
                arm_of[arm].append(e['run_id'])
        R['eval_belly'] = dict(all=round(100 * np.mean([r['belly']['flag'] for r in bel]), 2), n=len(bel),
                               by_arm={arm: round(100 * np.mean([ev[x]['belly']['flag'] for x in ids]), 2) for arm, ids in arm_of.items()},
                               flagged_but_goal=sum(1 for r in bel if r['belly']['flag'] and r['status'] == 'goal_reached'),
                               noncontiguous=sum(1 for r in bel if not r['belly']['frames_contiguous']))
    if (C / 'gator.json').exists():
        gr = json.load(open(C / 'gator.json'))['rows']
        comp = [x for x in gr if x['complete']]; val = [x for x in comp if x.get('ok')]
        both = [x for x in val if x.get('h_status') is not None]
        fl = lambda s: s != 'goal_reached'
        pc = lambda xs, f: round(100 * float(np.mean([f(x) for x in xs])), 2) if xs else None
        des = [x for x in gr if x['kind'] == 'designed']
        R['gator_collection'] = dict(
            rows=len(gr), complete=len(comp), validated=len(val), qa_flags=dict(Counter(x.get('flag') for x in comp if not x.get('ok'))),
            vehicle=dict(Counter(str(x.get('vehicle')) for x in comp)), rear_radius=dict(Counter(str(x.get('rear_radius')) for x in comp)),
            wheel_field=dict(Counter(str(x.get('wheel')) for x in gr)),
            belly_flag=pc(val, lambda x: x['belly']['flag']), belly_total_gt1s=pc(val, lambda x: x['belly']['total_s'] > 1.0),
            belly_noncontiguous=sum(1 for x in val if not x['belly']['frames_contiguous']),
            fail_gator=pc(both, lambda x: fl(x['status'])), fail_hmmwv=pc(both, lambda x: fl(x['h_status'])), n_pairs=len(both),
            gator_only=sum(1 for x in both if fl(x['status']) and not fl(x['h_status'])), hmmwv_only=sum(1 for x in both if fl(x['h_status']) and not fl(x['status'])),
            status_gator=dict(Counter(x['status'] for x in val)), status_hmmwv=dict(Counter(x['h_status'] for x in both)),
            sim_h_gator=round(sum(x['sim_s'] for x in both) / 3600, 2), sim_h_hmmwv=round(sum(x['h_sim_s'] for x in both) / 3600, 2),
            profile_vs_index_mismatch=sum(1 for x in des if x['profile'] != ['constant_2', 'constant_4', 'constant_6', 'smooth_2_6_2'][int(x['id'].split('_route_')[1]) % 4]),
            by_profile={p: dict(n=len(xs), fail_gator=pc(xs, lambda x: fl(x['status'])), fail_hmmwv=pc(xs, lambda x: fl(x['h_status'])), belly=pc(xs, lambda x: x['belly']['flag']))
                        for p, xs in sorted(defaultdict(list, {p: [x for x in both if x['profile'] == p] for p in {x['profile'] for x in both}}).items())},
            by_tier={t: dict(n=len(xs), fail_gator=pc(xs, lambda x: fl(x['status']))) for t, xs in sorted({t: [x for x in both if x['tier'] == t] for t in {x['tier'] for x in both}}.items())},
            by_split={s: dict(n=len(xs), fail_gator=pc(xs, lambda x: fl(x['status'])), fail_hmmwv=pc(xs, lambda x: fl(x['h_status']))) for s, xs in sorted({s: [x for x in both if x['split'] == s] for s in {x['split'] for x in both}}.items())},
            tiers0_6=dict(n=len([x for x in both if x['tier'] <= 6]), fail_gator=pc([x for x in both if x['tier'] <= 6], lambda x: fl(x['status'])),
                          fail_hmmwv=pc([x for x in both if x['tier'] <= 6], lambda x: fl(x['h_status'])), belly=pc([x for x in val if x['tier'] <= 6], lambda x: x['belly']['flag']),
                          sim_h_gator=round(sum(x['sim_s'] for x in both if x['tier'] <= 6) / 3600, 2), sim_h_hmmwv=round(sum(x['h_sim_s'] for x in both if x['tier'] <= 6) / 3600, 2)))
        # ids = the HMMWV collect_v1 ids (the E4 f104 soil file's 15,235 drives)
        R['gator_collection']['distinct_ids'] = len({x['id'] for x in gr})
    if (C / 'pilot.json').exists():
        pr = json.load(open(C / 'pilot.json'))['rows']
        by = defaultdict(dict)
        for x in pr:
            if x.get('complete'):
                by[x['pid']][x['prefix']] = x
        pairs = [v for v in by.values() if 'gator' in v and 'gatorR8' in v]
        fl = lambda s: s != 'goal_reached'
        R['pilot_wheel'] = dict(routes=len(pairs), fail_calibrated=round(100 * np.mean([fl(v['gator']['status']) for v in pairs]), 2),
                                fail_r8=round(100 * np.mean([fl(v['gatorR8']['status']) for v in pairs]), 2),
                                fail_hmmwv=round(100 * np.mean([fl(v['gator']['h_status']) for v in pairs]), 2),
                                flips_to_goal=sum(1 for v in pairs if fl(v['gator']['status']) and not fl(v['gatorR8']['status'])),
                                flips_to_fail=sum(1 for v in pairs if not fl(v['gator']['status']) and fl(v['gatorR8']['status'])),
                                radii=dict(Counter(f"{v['gator']['rear_radius']}|{v['gatorR8']['rear_radius']}" for v in pairs)),
                                qa_ok=sum(1 for v in pairs if v['gator']['ok'] and v['gatorR8']['ok']))
    json.dump(R, open(D / 'readouts.json', 'w'), indent=1, default=float)
    print(json.dumps(R, indent=1, default=float)[:20000])


if __name__ == '__main__':
    main()
