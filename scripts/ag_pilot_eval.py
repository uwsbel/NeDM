#!/usr/bin/env python3
"""Evaluate the Gator cluster pilot (arena_gator_20260925, module E3b1) against the PLAN 7.7 / REVIEW_R1 3 criteria.

Inputs (local copies, numpy only):
  --pilot    local copy of G3/pilot_gator (soil/{runs,failed,workers,logs}, rigid/{runs,logs}, logs/)
  --tasks    the two pilot task files (e3/tasks/pilot_gator_{soil,rigid}.json)
  HMMWV references on the identical routes: crm_f104_v1/collect_v1/runs (soil) and fdm_f104_50h_20260909/production_v3/runs
  (rigid), both read-only local copies of the earlier roots.
Definitions:
  soil validated   = crm_qa.check (completion marker, launch check passed, consistent finite arrays, no explosion, no
                     breakthrough without a preceding stall)
  rigid validated  = completion marker, launch check and native-height check passed, finite state / terminal state
  fail             = status != goal_reached
  belly-in-soil    = lowest chassis-hull point more than 0.05 m under the undisturbed surface for more than 1 s in a row
                     (vehicle_extra.npz, 0.05 s frames); the cumulative version (> 1 s in total) is reported as well
  rigid unsafe     = f104_n2_dataset.py's label: not (goal and backward-motion time after the 1 s settle < 0.05 s and
                     min vx after the settle > -0.30 m/s); "without the backward clause" = fail only
  profiles         = designed route index % 4 (constant 2 / 4 / 6 m/s, smooth 2-6-2); on-policy = planner proposals
Writes a JSON summary (--out) and prints a text report.
"""
import argparse, glob, json, math, os, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import crm_qa  # noqa: E402

SOIL_REF = ROOT / 'artifacts/traverse/crm_f104_v1/collect_v1/runs'
RIGID_REF = ROOT / 'artifacts/traverse/fdm_f104_50h_20260909/production_v3/runs'
PROFILES = {0: 'constant_2', 1: 'constant_4', 2: 'constant_6', 3: 'smooth_2_6_2', -1: 'planner_proposal'}
DT, SETTLE_FRAMES = 0.05, 20


def profile(pair_id):
    return int(pair_id.split('_route_')[1]) % 4 if '_route_' in pair_id else -1


def mcnemar(b, c):
    """exact two-sided McNemar on the discordant counts b (Gator-only fail) and c (HMMWV-only fail)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def longest_run(mask):
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def rate(xs):
    xs = list(xs)
    return (float(np.mean(xs)) if xs else None, len(xs))


def pct(x):
    return 'n/a' if x is None else f'{100 * x:.1f} %'


# ---------------------------------------------------------------------------------------------------------- soil
def soil_rows(pilot, tasks, host_part):
    rows = []
    for t in tasks:
        d = pilot / 'soil/runs' / t['id']
        r = dict(id=t['id'], pair_id=t['pair_id'], wheel=t['wheel'], group=t['group'], stratum=t['stratum'], kind=t['kind'],
                 profile=PROFILES[profile(t['pair_id'])])
        fail_rec = pilot / 'soil/failed' / f"{t['id']}.json"
        r['worker_failures'] = json.load(open(fail_rec))['attempts'] if fail_rec.exists() else 0
        r['collection_failure'] = (d / 'collection_failure.json').exists()
        if not (d / 'episode_complete.json').exists():
            r['complete'] = False
            rows.append(r)
            continue
        r['complete'] = True
        q = crm_qa.check(str(d))
        o = json.load(open(d / 'outcome.json'))
        launch = json.load(open(d / 'initial_state_validation.json'))
        req = json.load(open(d / 'collection_request.json'))
        vx = np.load(d / 'vehicle_extra.npz')
        belly = vx['belly_clearance_min_m'].astype(float)
        deep = belly < -0.05
        z = np.load(d / 'trajectory.npz')
        r.update(validated=bool(q['ok']), qa_flag=q.get('flag'), launch_passed=bool(launch.get('passed')), status=o['status'],
                 fail=o['status'] != 'goal_reached', elapsed_s=float(o['elapsed_s']), rtf=float(o['crm']['rtf_sim_over_wall']),
                 loop_wall_s=float(o['wall_s']), host=req.get('host'), gpu_type=host_part.get(req.get('host'), '?'),
                 finite=bool(np.isfinite(z['state']).all() and np.isfinite(z['action']).all() and np.isfinite(z['pose']).all()),
                 belly_min_m=float(belly.min()), belly_deep_run_s=longest_run(deep) * DT, belly_deep_total_s=float(deep.sum() * DT),
                 belly_below0_frac=float((belly < 0).mean()), anchor_belly_m=float(belly[0]),
                 max_sinkage_m=float(o['crm'].get('max_wheel_sinkage_below_bmp_m', np.nan)),
                 slip_p95=float(o['crm'].get('max_abs_slip_ratio_p95', np.nan)),
                 soil_wheel=o['vehicle']['soil_wheel_geometry']['front']['radius_m'], vehicle=o['vehicle']['name'],
                 ag_vehicle_sha=o['vehicle']['ag_vehicle_sha256'][:12], step_s=float(o['crm']['physics_dt_s']),
                 goal_progress_m=float(o.get('goal_progress_m', np.nan)),
                 end_mtime=os.path.getmtime(d / 'episode_complete.json'))
        r['belly_flag'] = r['belly_deep_run_s'] > 1.0
        r['belly_flag_total'] = r['belly_deep_total_s'] > 1.0
        h = SOIL_REF / t['pair_id']
        ho = json.load(open(h / 'outcome.json'))
        r.update(h_status=ho['status'], h_fail=ho['status'] != 'goal_reached', h_elapsed_s=float(ho['elapsed_s']),
                 h_max_sinkage_m=float(ho['crm'].get('max_wheel_sinkage_below_bmp_m', np.nan)),
                 h_slip_p95=float(ho['crm'].get('max_abs_slip_ratio_p95', np.nan)), h_rtf=float(ho['crm']['rtf_sim_over_wall']))
        rows.append(r)
    return rows


def paired(rows, a_key='fail', b_key='h_fail'):
    both = sum(r[a_key] and r[b_key] for r in rows); a_only = sum(r[a_key] and not r[b_key] for r in rows)
    b_only = sum(r[b_key] and not r[a_key] for r in rows); none = sum(not r[a_key] and not r[b_key] for r in rows)
    return dict(n=len(rows), both_fail=both, gator_only_fail=a_only, hmmwv_only_fail=b_only, both_goal=none,
                mcnemar_p=mcnemar(a_only, b_only))


def soil_summary(rows, variant):
    rs = [r for r in rows if r['wheel'] == variant]
    done = [r for r in rs if r['complete']]
    val = [r for r in done if r['validated']]
    out = dict(rows=len(rs), complete=len(done), validated=len(val), validated_share=len(val) / len(rs) if rs else None,
               worker_failed_ids=sum(r['worker_failures'] > 0 for r in rs), collection_failure_files=sum(r['collection_failure'] for r in rs),
               qa_flags=dict(Counter(r['qa_flag'] for r in done if r['qa_flag'])),
               nonfinite=sum(not r['finite'] for r in done), launch_failures=sum(not r['launch_passed'] for r in done),
               launch_failure_share=(sum(not r['launch_passed'] for r in done) / len(done)) if done else None,
               crash_or_nan_share=((sum(r['worker_failures'] > 0 and not r['complete'] for r in rs) + sum(not r['finite'] for r in done)
                                    + sum(r['qa_flag'] in ('explosion', 'nonfinite', 'shape', 'unreadable') for r in done)) / len(rs)) if rs else None,
               status=dict(Counter(r['status'] for r in val)), h_status=dict(Counter(r['h_status'] for r in val)),
               fail=rate(r['fail'] for r in val), h_fail=rate(r['h_fail'] for r in val), paired=paired(val),
               by_profile={p: dict(gator=rate(r['fail'] for r in val if r['profile'] == p), hmmwv=rate(r['h_fail'] for r in val if r['profile'] == p),
                                   paired=paired([r for r in val if r['profile'] == p]))
                           for p in PROFILES.values()},
               by_stratum={s: dict(gator=rate(r['fail'] for r in val if r['stratum'] == s), hmmwv=rate(r['h_fail'] for r in val if r['stratum'] == s))
                           for s in sorted({r['stratum'] for r in val})},
               belly_flag=rate(r['belly_flag'] for r in val), belly_flag_total=rate(r['belly_flag_total'] for r in val),
               belly_flag_among_fail=rate(r['belly_flag'] for r in val if r['fail']),
               belly_flag_among_goal=rate(r['belly_flag'] for r in val if not r['fail']),
               fail_with_flag_as_fail=rate(r['fail'] or r['belly_flag'] for r in val),
               belly_min_m_median=float(np.median([r['belly_min_m'] for r in val])) if val else None,
               anchor_belly_m_median=float(np.median([r['anchor_belly_m'] for r in val])) if val else None,
               sim_hours=sum(r['elapsed_s'] for r in val) / 3600, h_sim_hours=sum(r['h_elapsed_s'] for r in val) / 3600,
               mean_elapsed_s=float(np.mean([r['elapsed_s'] for r in val])) if val else None,
               h_mean_elapsed_s=float(np.mean([r['h_elapsed_s'] for r in val])) if val else None,
               max_sinkage_median=(float(np.nanmedian([r['max_sinkage_m'] for r in val])), float(np.nanmedian([r['h_max_sinkage_m'] for r in val]))) if val else None,
               slip_p95_median=(float(np.nanmedian([r['slip_p95'] for r in val])), float(np.nanmedian([r['h_slip_p95'] for r in val]))) if val else None,
               wheel_radius_front=sorted({r['soil_wheel'] for r in done}), vehicles=sorted({r['vehicle'] for r in done}),
               ag_vehicle_sha=sorted({r['ag_vehicle_sha'] for r in done}), step_s=sorted({r['step_s'] for r in done}),
               by_gpu={g: dict(episodes=sum(r['gpu_type'] == g for r in done),
                               loop_wall_per_sim_s=float(np.median([1 / r['rtf'] for r in done if r['gpu_type'] == g])))
                       for g in sorted({r['gpu_type'] for r in done})})
    groups = defaultdict(list)
    for r in val:
        if r['kind'] == 'designed':
            groups[r['group']].append(r)
    out['groups_any_designed_goal'] = dict(gator=rate(any(not r['fail'] for r in v) for v in groups.values()),
                                           hmmwv=rate(any(not r['h_fail'] for r in v) for v in groups.values()))
    des = [p for p in ('constant_2', 'constant_4', 'constant_6', 'smooth_2_6_2')]
    in_band = [p for p in des if out['by_profile'][p]['gator'][0] is not None and 0.10 <= out['by_profile'][p]['gator'][0] <= 0.90]
    out['informative'] = dict(overall_in_10_90=out['fail'][0] is not None and 0.10 <= out['fail'][0] <= 0.90,
                              designed_profiles_in_band=in_band)
    return out


def sensitivity(rows):
    cal = {r['pair_id']: r for r in rows if r['wheel'] == 'calibrated' and r.get('validated')}
    r8 = {r['pair_id']: r for r in rows if r['wheel'] != 'calibrated' and r.get('validated')}
    both = sorted(set(cal) & set(r8))
    if not both:
        return dict(n=0)
    fc = np.mean([cal[k]['fail'] for k in both]); f8 = np.mean([r8[k]['fail'] for k in both])
    same = np.mean([cal[k]['status'] == r8[k]['status'] for k in both])
    rs = [dict(fail=r8[k]['fail'], h_fail=cal[k]['fail']) for k in both]
    pr = paired(rs)
    return dict(n=len(both), fail_calibrated=float(fc), fail_r8=float(f8), change_points=float(100 * (f8 - fc)),
                same_status=float(same), r8_only_fail=pr['gator_only_fail'], calibrated_only_fail=pr['hmmwv_only_fail'],
                mcnemar_p=pr['mcnemar_p'],
                belly_flag_r8=float(np.mean([r8[k]['belly_flag'] for k in both])),
                max_sinkage_median=(float(np.nanmedian([cal[k]['max_sinkage_m'] for k in both])), float(np.nanmedian([r8[k]['max_sinkage_m'] for k in both]))),
                anchor_belly_median=(float(np.median([cal[k]['anchor_belly_m'] for k in both])), float(np.median([r8[k]['anchor_belly_m'] for k in both]))),
                by_profile={p: (float(np.mean([cal[k]['fail'] for k in both if cal[k]['profile'] == p] or [np.nan])),
                                float(np.mean([r8[k]['fail'] for k in both if cal[k]['profile'] == p] or [np.nan])),
                                sum(cal[k]['profile'] == p for k in both)) for p in PROFILES.values()},
                decision='depends on the wheel model' if abs(100 * (f8 - fc)) > 15 else 'within 15 points')


# ---------------------------------------------------------------------------------------------------------- rigid
def back_stats(d):
    z = np.load(d / 'trajectory.npz')
    vx = z['state'][:, 0].astype(float); thr = z['action'][:, 1].astype(float)
    back = (vx < -0.10) & (thr > 0.3)
    back_s = float(back[SETTLE_FRAMES:].sum() * DT)
    min_vx = float(vx[SETTLE_FRAMES:].min()) if len(vx) > SETTLE_FRAMES else float(vx.min())
    st = z['state']
    fin = bool(np.isfinite(st).all() and ('terminal_state' not in z.files or np.isfinite(z['terminal_state']).all()))
    return back_s, min_vx, fin, float(vx[0]), float(np.degrees(np.abs(st[:, 2]).max())), float(np.degrees(np.abs(st[:, 3]).max()))


def rigid_rows(pilot, tasks):
    rows = []
    fails = set()
    for f in glob.glob(str(pilot / 'rigid/logs/rigid_*.out')) + glob.glob(str(pilot / 'logs/*.out')):
        for line in open(f):
            m = re.match(r'\s+FAIL (\S+) (\S+)', line)
            if m:
                fails.add(m.group(1))
    for t in tasks:
        d = pilot / 'rigid/runs' / t['id']
        r = dict(id=t['id'], pair_id=t['pair_id'], group=t['group'], stratum=t['stratum'], profile=PROFILES[t['profile']],
                 runner_fail=t['id'] in fails)
        if not (d / 'episode_complete.json').exists():
            r['complete'] = False
            rows.append(r)
            continue
        o = json.load(open(d / 'outcome.json'))
        launch = json.load(open(d / 'initial_state_validation.json'))
        nh = json.load(open(d / 'native_height_check.json'))
        back_s, min_vx, fin, vx0, roll, pitch = back_stats(d)
        chassis = None
        if (d / 'rich_intervals.npz').exists():
            ri = np.load(d / 'rich_intervals.npz')
            ks = [k for k in ri.files if 'chassis_contact' in k and 'max' in k]
            chassis = float(ri[ks[0]].max()) if ks else None
        b = o['vehicle']['belly']
        fail = o['status'] != 'goal_reached'
        r.update(complete=True, launch_passed=bool(launch['passed']), native_height_passed=bool(nh['passed']), finite=fin,
                 status=o['status'], fail=fail, unsafe=bool(fail or back_s >= 0.05 or min_vx <= -0.30), unsafe_noback=fail,
                 back_s=back_s, min_vx=min_vx, anchor_vx=vx0, max_roll_deg=roll, max_pitch_deg=pitch, chassis_contact_n=chassis,
                 belly_min_m=float(b['min_clearance_m']), belly_frames_below=int(b['frames_below_surface']),
                 elapsed_s=float(o['elapsed_s']), wall_s=float(o['wall_s']), vehicle=o['vehicle']['name'],
                 fingerprint_sha=(o['vehicle'].get('runtime_fingerprint') or {}).get('sha256', '')[:12])
        r['validated'] = r['launch_passed'] and r['native_height_passed'] and fin
        h = RIGID_REF / t['pair_id']
        ho = json.load(open(h / 'outcome.json'))
        hb, hmin, hfin, hvx0, hroll, hpitch = back_stats(h)
        hfail = ho['status'] != 'goal_reached'
        r.update(h_status=ho['status'], h_fail=hfail, h_unsafe=bool(hfail or hb >= 0.05 or hmin <= -0.30), h_unsafe_noback=hfail,
                 h_back_s=hb, h_anchor_vx=hvx0, h_max_roll_deg=hroll, h_max_pitch_deg=hpitch, h_elapsed_s=float(ho['elapsed_s']),
                 h_wall_s=float(ho['wall_s']))
        # same-node HMMWV re-drive (pilot_gator/rigid_hmmwv, same shard = same node as the Gator row)
        n = pilot / 'rigid_hmmwv/runs' / ('hmmwv__' + t['pair_id'])
        if (n / 'episode_complete.json').exists():
            no = json.load(open(n / 'outcome.json'))
            nb, nmin, nfin, nvx0, nroll, npitch = back_stats(n)
            nfail = no['status'] != 'goal_reached'
            nch = None
            if (n / 'rich_intervals.npz').exists():
                ri = np.load(n / 'rich_intervals.npz')
                ks = [k for k in ri.files if 'chassis_contact' in k and 'max' in k]
                nch = float(ri[ks[0]].max()) if ks else None
            nl = json.load(open(n / 'initial_state_validation.json'))['passed'] and json.load(open(n / 'native_height_check.json'))['passed']
            r.update(n_status=no['status'], n_fail=nfail, n_unsafe=bool(nfail or nb >= 0.05 or nmin <= -0.30), n_unsafe_noback=nfail,
                     n_chassis_contact_n=nch, n_valid=bool(nl and nfin), n_elapsed_s=float(no['elapsed_s']), n_wall_s=float(no['wall_s']),
                     n_vehicle_block='vehicle' in no, n_same_as_production=no['status'] == ho['status'],
                     n_traj_identical=bool(np.array_equal(np.load(n / 'trajectory.npz')['state'], np.load(h / 'trajectory.npz')['state'])))
        rows.append(r)
    return rows


def rigid_same_node(rows):
    val = [r for r in rows if r.get('validated') and r.get('n_valid')]
    if not val:
        return dict(n=0)
    return dict(n=len(val), gator_fail=rate(r['fail'] for r in val), hmmwv_fail=rate(r['n_fail'] for r in val),
                paired_fail=paired(val, 'fail', 'n_fail'), gator_unsafe=rate(r['unsafe'] for r in val),
                hmmwv_unsafe=rate(r['n_unsafe'] for r in val), paired_unsafe=paired(val, 'unsafe', 'n_unsafe'),
                hmmwv_status=dict(Counter(r['n_status'] for r in val)),
                by_profile={p: dict(gator_fail=rate(r['fail'] for r in val if r['profile'] == p), hmmwv_fail=rate(r['n_fail'] for r in val if r['profile'] == p),
                                    gator_unsafe=rate(r['unsafe'] for r in val if r['profile'] == p), hmmwv_unsafe=rate(r['n_unsafe'] for r in val if r['profile'] == p))
                            for p in ('constant_2', 'constant_4', 'constant_6', 'smooth_2_6_2')},
                hmmwv_chassis_contact=dict(runs_with_contact=sum((r['n_chassis_contact_n'] or 0) > 0 for r in val),
                                           max_n=max([r['n_chassis_contact_n'] or 0 for r in val] or [0]),
                                           on_gator_contact_routes=sum((r['n_chassis_contact_n'] or 0) > 0 for r in val if (r['chassis_contact_n'] or 0) > 0),
                                           gator_contact_routes=sum((r['chassis_contact_n'] or 0) > 0 for r in val)),
                hmmwv_vs_production_v3=dict(same_status=sum(r['n_same_as_production'] for r in val), identical_state=sum(r['n_traj_identical'] for r in val),
                                            production_fail=rate(r['h_fail'] for r in val), rerun_fail=rate(r['n_fail'] for r in val)),
                hmmwv_rerun_has_vehicle_block=sum(r['n_vehicle_block'] for r in val),
                wall_per_sim_s_median=dict(gator=float(np.median([r['wall_s'] / r['elapsed_s'] for r in val])),
                                           hmmwv_same_node=float(np.median([r['n_wall_s'] / r['n_elapsed_s'] for r in val]))),
                wall_fit_s=dict(gator=np.linalg.lstsq(np.c_[np.ones(len(val)), [r['elapsed_s'] for r in val]], np.array([r['wall_s'] for r in val]), rcond=None)[0].tolist(),
                                hmmwv_same_node=np.linalg.lstsq(np.c_[np.ones(len(val)), [r['n_elapsed_s'] for r in val]], np.array([r['n_wall_s'] for r in val]), rcond=None)[0].tolist()))


def rigid_summary(rows):
    done = [r for r in rows if r['complete']]
    val = [r for r in done if r['validated']]
    fit = None
    if len(val) > 5:
        A = np.c_[np.ones(len(val)), [r['elapsed_s'] for r in val]]
        fit = np.linalg.lstsq(A, np.array([r['wall_s'] for r in val]), rcond=None)[0].tolist()
        hA = np.c_[np.ones(len(val)), [r['h_elapsed_s'] for r in val]]
        hfit = np.linalg.lstsq(hA, np.array([r['h_wall_s'] for r in val]), rcond=None)[0].tolist()
    out = dict(rows=len(rows), complete=len(done), validated=len(val), validated_share=len(val) / len(rows) if rows else None,
               runner_failures=sum(r['runner_fail'] for r in rows), nonfinite=sum(not r['finite'] for r in done),
               launch_failures=sum(not r['launch_passed'] for r in done), native_height_failures=sum(not r['native_height_passed'] for r in done),
               status=dict(Counter(r['status'] for r in val)), h_status=dict(Counter(r['h_status'] for r in val)),
               fail=rate(r['fail'] for r in val), h_fail=rate(r['h_fail'] for r in val), paired_fail=paired(val),
               unsafe=rate(r['unsafe'] for r in val), h_unsafe=rate(r['h_unsafe'] for r in val),
               paired_unsafe=paired(val, 'unsafe', 'h_unsafe'),
               unsafe_noback=rate(r['unsafe_noback'] for r in val), h_unsafe_noback=rate(r['h_unsafe_noback'] for r in val),
               backward_only_unsafe=dict(gator=sum(r['unsafe'] and not r['fail'] for r in val), hmmwv=sum(r['h_unsafe'] and not r['h_fail'] for r in val)),
               by_profile={p: dict(gator_fail=rate(r['fail'] for r in val if r['profile'] == p), hmmwv_fail=rate(r['h_fail'] for r in val if r['profile'] == p),
                                   gator_unsafe=rate(r['unsafe'] for r in val if r['profile'] == p), hmmwv_unsafe=rate(r['h_unsafe'] for r in val if r['profile'] == p))
                           for p in ('constant_2', 'constant_4', 'constant_6', 'smooth_2_6_2')},
               chassis_contact=dict(runs_with_contact=sum((r['chassis_contact_n'] or 0) > 0 for r in val),
                                    runs_with_record=sum(r['chassis_contact_n'] is not None for r in val),
                                    max_n=max([r['chassis_contact_n'] or 0 for r in val] or [0])),
               belly=dict(min_m=min([r['belly_min_m'] for r in val] or [np.nan]), runs_below_surface=sum(r['belly_frames_below'] > 0 for r in val),
                          median_min_m=float(np.median([r['belly_min_m'] for r in val])) if val else None),
               anchor_vx=dict(gator_median=float(np.median([r['anchor_vx'] for r in val])) if val else None,
                              gator_min=float(min([r['anchor_vx'] for r in val] or [np.nan])),
                              hmmwv_median=float(np.median([r['h_anchor_vx'] for r in val])) if val else None,
                              hmmwv_min=float(min([r['h_anchor_vx'] for r in val] or [np.nan]))),
               tilt=dict(gator_max_roll=float(max([r['max_roll_deg'] for r in val] or [np.nan])), hmmwv_max_roll=float(max([r['h_max_roll_deg'] for r in val] or [np.nan])),
                         gator_max_pitch=float(max([r['max_pitch_deg'] for r in val] or [np.nan])), hmmwv_max_pitch=float(max([r['h_max_pitch_deg'] for r in val] or [np.nan]))),
               sim_hours=sum(r['elapsed_s'] for r in val) / 3600, h_sim_hours=sum(r['h_elapsed_s'] for r in val) / 3600,
               wall_fit_s=dict(gator_mi3501x_cpu=fit, hmmwv_production_v3=hfit) if fit else None,
               wall_per_sim_s_median=dict(gator=float(np.median([r['wall_s'] / r['elapsed_s'] for r in val])) if val else None,
                                          hmmwv=float(np.median([r['h_wall_s'] / r['h_elapsed_s'] for r in val])) if val else None),
               vehicles=sorted({r['vehicle'] for r in done}), fingerprints=sorted({r['fingerprint_sha'] for r in done}))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pilot', required=True)
    ap.add_argument('--tasks-dir', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pilot = Path(a.pilot)
    host_part = {}
    for f in glob.glob(str(pilot / 'logs/*.out')) + glob.glob(str(pilot / 'soil/logs/*.out')):
        for line in open(f):
            m = re.match(r'host=(\S+) job=\S+ array=\S+ part=(\S+)', line)
            if m:
                host_part[m.group(1)] = {'mi3501x': 'MI350X', 'devel': 'MI210', 'mi2101x': 'MI210', 'mi2104x': 'MI210'}.get(m.group(2), m.group(2))
    soil_t = json.load(open(Path(a.tasks_dir) / 'pilot_gator_soil.json'))
    rigid_t = json.load(open(Path(a.tasks_dir) / 'pilot_gator_rigid.json'))
    srows = soil_rows(pilot, soil_t, host_part)
    rrows = rigid_rows(pilot, rigid_t)
    workers = [json.load(open(f)) for f in glob.glob(str(pilot / 'soil/workers/*.json'))]
    for w in workers:
        w['gpu_type'] = host_part.get(w['host'], '?')
        w['wall_per_sim_s'] = w['wall_s'] / w['sim_s'] if w['sim_s'] else None
    # CPU load windows on the pilot nodes (rigid shard k ran on the node of array task k): does rigid work slow soil?
    shard_host = {}
    for f in glob.glob(str(pilot / 'rigid/logs/rigid_*.out')) + glob.glob(str(pilot / 'rigid_hmmwv/logs/*.out')):
        txt = open(f).read()
        m = re.search(r'host=(\S+).*?shard[= ](\d+)', txt, re.S)
        if m:
            shard_host.setdefault(int(m.group(2)), m.group(1))
    windows = defaultdict(list)
    for sub, tl in (('rigid', rigid_t), ('rigid_hmmwv', [dict(t, id='hmmwv__' + t['pair_id']) for t in rigid_t])):
        by = defaultdict(list)
        for t in tl:
            d = pilot / sub / 'runs' / t['id']
            if (d / 'episode_complete.json').exists():
                by[t['shard']].append((os.path.getmtime(d / 'collection_request.json'), os.path.getmtime(d / 'episode_complete.json')))
        for k, v in by.items():
            if k in shard_host:
                windows[shard_host[k]].append((min(a for a, _ in v) - 60, max(b for _, b in v)))
    for r in srows:
        if r.get('complete') and r.get('host') in windows:
            s0, s1 = r['end_mtime'] - r['loop_wall_s'], r['end_mtime']
            r['cpu_busy'] = any(s0 < b and s1 > a for a, b in windows[r['host']])
    load = {}
    for h in windows:
        busy = [1 / r['rtf'] for r in srows if r.get('host') == h and r.get('cpu_busy') is True]
        idle = [1 / r['rtf'] for r in srows if r.get('host') == h and r.get('cpu_busy') is False]
        load[h] = dict(busy_n=len(busy), busy_loop_wall_per_sim_s=float(np.median(busy)) if busy else None,
                       idle_n=len(idle), idle_loop_wall_per_sim_s=float(np.median(idle)) if idle else None,
                       windows=[(time.strftime('%H:%M:%S', time.localtime(a)), time.strftime('%H:%M:%S', time.localtime(b))) for a, b in windows[h]])
    res = dict(soil_calibrated=soil_summary(srows, 'calibrated'), soil_r8=soil_summary(srows, 'radius_plus_0.08'),
               sensitivity=sensitivity(srows), rigid=rigid_summary(rrows), rigid_same_node=rigid_same_node(rrows),
               soil_under_rigid_load=load, soil_workers=workers, host_gpu=host_part)
    json.dump(dict(summary=res, soil_rows=srows, rigid_rows=rrows), open(a.out, 'w'), indent=1, default=float)
    report(res)


def fr(x):
    return 'n/a' if x is None or x[0] is None else f'{100 * x[0]:.1f} % (n {x[1]})'


def report(res):
    for key in ('soil_calibrated', 'soil_r8'):
        s = res[key]
        print(f'== {key}: rows {s["rows"]} complete {s["complete"]} validated {s["validated"]} ({pct(s["validated_share"])}); '
              f'worker-failed ids {s["worker_failed_ids"]}, collection_failure {s["collection_failure_files"]}, qa flags {s["qa_flags"]}, '
              f'nonfinite {s["nonfinite"]}, launch failures {s["launch_failures"]} ({pct(s["launch_failure_share"])}), crash/NaN {pct(s["crash_or_nan_share"])}')
        print(f'   fail Gator {fr(s["fail"])} vs HMMWV {fr(s["h_fail"])}; paired {s["paired"]}')
        print(f'   status Gator {s["status"]}  HMMWV {s["h_status"]}')
        for p, v in s['by_profile'].items():
            print(f'   {p:17s} Gator {fr(v["gator"]):18s} HMMWV {fr(v["hmmwv"]):18s} disc G-only {v["paired"]["gator_only_fail"]} H-only {v["paired"]["hmmwv_only_fail"]} p {v["paired"]["mcnemar_p"]:.3g}')
        for st, v in s['by_stratum'].items():
            print(f'   {st:24s} Gator {fr(v["gator"]):18s} HMMWV {fr(v["hmmwv"])}')
        print(f'   belly flag (run > 1 s) {fr(s["belly_flag"])}, cumulative {fr(s["belly_flag_total"])}; among fails {fr(s["belly_flag_among_fail"])}, '
              f'among goals {fr(s["belly_flag_among_goal"])}; fail counting flagged as fail {fr(s["fail_with_flag_as_fail"])}; '
              f'median belly min {s["belly_min_m_median"]}, anchor {s["anchor_belly_m_median"]}')
        print(f'   sim h Gator {s["sim_hours"]:.2f} vs HMMWV {s["h_sim_hours"]:.2f}; mean ep {s["mean_elapsed_s"]} vs {s["h_mean_elapsed_s"]}; '
              f'max sinkage median (G, H) {s["max_sinkage_median"]}; slip p95 median {s["slip_p95_median"]}')
        print(f'   any designed goal per group {s["groups_any_designed_goal"]}; informative {s["informative"]}; by GPU {s["by_gpu"]}; '
              f'wheel {s["wheel_radius_front"]} step {s["step_s"]} ag_vehicle {s["ag_vehicle_sha"]}')
    print('== sensitivity', json.dumps(res['sensitivity'], default=float))
    r = res['rigid']
    print(f'== rigid: rows {r["rows"]} complete {r["complete"]} validated {r["validated"]} ({pct(r["validated_share"])}); runner failures {r["runner_failures"]}, '
          f'nonfinite {r["nonfinite"]}, launch failures {r["launch_failures"]}, native-height failures {r["native_height_failures"]}')
    print(f'   fail Gator {fr(r["fail"])} vs HMMWV production_v3 {fr(r["h_fail"])}; paired {r["paired_fail"]}')
    print(f'   unsafe Gator {fr(r["unsafe"])} vs HMMWV {fr(r["h_unsafe"])} (without backward clause {fr(r["unsafe_noback"])} vs {fr(r["h_unsafe_noback"])}); backward-only {r["backward_only_unsafe"]}')
    for p, v in r['by_profile'].items():
        print(f'   {p:14s} fail G {fr(v["gator_fail"]):18s} H {fr(v["hmmwv_fail"]):18s} unsafe G {fr(v["gator_unsafe"]):18s} H {fr(v["hmmwv_unsafe"])}')
    print(f'   status G {r["status"]} H {r["h_status"]}')
    print(f'   chassis contact {r["chassis_contact"]}; belly {r["belly"]}; anchor vx {r["anchor_vx"]}; tilt {r["tilt"]}')
    print(f'   sim h G {r["sim_hours"]:.2f} H {r["h_sim_hours"]:.2f}; wall fit [a, b] {r["wall_fit_s"]}; wall/sim median {r["wall_per_sim_s_median"]}')
    print('== rigid same node', json.dumps(res['rigid_same_node'], default=float))
    print('== soil under rigid CPU load', json.dumps(res['soil_under_rigid_load'], default=float))
    for w in res['soil_workers']:
        print(f'   worker {w["job"]} {w["host"]} {w["gpu_type"]}: {w["episodes"]} ep, {w["sim_s"]:.0f} sim s in {w["wall_s"]:.0f} wall s -> {w["wall_per_sim_s"]}')


if __name__ == '__main__':
    main()
