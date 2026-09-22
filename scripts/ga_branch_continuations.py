#!/usr/bin/env python3
"""A4 stage 'continuations' (PLAN v2 A4, CRM two-pass): from the pass-1 prefix replays, check every anchor's replayed
state against the recording (class and pose), sample 3 continuations from the REPLAYED pose with the patched sampler
(speed floor from the replayed vx, 15-degree start-heading acceptance), and write the pass-2 task rows for
crm_collect_ext.py --mode branch.

  python scripts/ga_branch_continuations.py --anchors K/A_adapt/a4/anchors/anchors_crm.json --pass1 K/A_adapt/a4/pass1_out_crm/runs \
      --tasks-pass1 K/A_adapt/a4/tasks_pass1_crm.json --out K/A_adapt/a4/cont_crm --cluster-route-prefix generalist/a4/cont_crm/routes
Rows: {id: <episode>__c<j>, group, case, route (the recorded route, relative to CRM_ROOT), run, tier, episode_seed,
       extra: ['--mode','branch','--branch-frame',F,'--branch-route',<CRM_ROOT-relative continuation>,'--horizon-s','120']}.
"""
import argparse, hashlib, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gc_control as GC


def stall_onset(vx, thr, start=20, run=20):
    m = (np.abs(vx) < 0.3) & (thr > 0.3); c = 0
    for k in range(start, len(m)):
        c = c + 1 if m[k] else 0
        if c >= run:
            return k - run + 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--anchors', required=True); ap.add_argument('--pass1', required=True, help='dir with <episode>__p1/trajectory.npz')
    ap.add_argument('--tasks-pass1', required=True, help='pass-1 task rows (case/route relative to CRM_ROOT)')
    ap.add_argument('--out', required=True); ap.add_argument('--cluster-route-prefix', required=True)
    ap.add_argument('--n-cont', type=int, default=3); ap.add_argument('--pose-tol-m', type=float, default=0.5)
    ap.add_argument('--horizon-s', default='120'); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    anchors = json.load(open(a.anchors)); p1 = {r['id']: r for r in json.load(open(a.tasks_pass1))}
    rdir = os.path.join(a.out, 'routes'); os.makedirs(rdir, exist_ok=True)
    rows, report = [], dict(n_anchors=len(anchors), missing_pass1=0, pose_mismatch=0, class_mismatch=0, sampler_failed=0, accepted=0, per_class={})
    for i, an in enumerate(anchors[:a.limit] if a.limit else anchors):
        ep, F = an['episode'], int(an['F']); pid = ep + '__p1'
        tp = os.path.join(a.pass1, pid, 'trajectory.npz')
        if not os.path.exists(tp) or pid not in p1:
            report['missing_pass1'] += 1; continue
        z = np.load(tp); pose, st, act = z['pose'], z['state'], z['action']
        if len(pose) <= F:
            report['missing_pass1'] += 1; continue
        dpos = float(np.linalg.norm(pose[F, :2] - np.asarray(an['pose_F'][:2])))
        onset = stall_onset(st[:, 0], act[:, 1])
        if an['cls'] == 'low_progress':
            rec = int(an['onset_frame']); ok_cls = onset is not None and (rec - 20) <= onset <= (rec + 30)
        else:
            ok_cls = (onset is None or onset > F + 30) and float(st[F, 0]) > 1.0
        entry = dict(anchor_id=an['anchor_id'], episode=ep, cls=an['cls'], F=F, pose_dist_m=round(dpos, 3), replay_onset=onset, recorded_onset=an['onset_frame'],
                     vx_F_replay=float(st[F, 0]), vx_F_recorded=an['vx_F'])
        if dpos > a.pose_tol_m:
            report['pose_mismatch'] += 1; entry['drop'] = 'pose'; report.setdefault('dropped', []).append(entry); continue
        if not ok_cls:
            report['class_mismatch'] += 1; entry['drop'] = 'class'; report.setdefault('dropped', []).append(entry); continue
        seed = int(hashlib.md5(an['anchor_id'].encode()).hexdigest()[:8], 16)
        try:
            conts = GC.sample_continuations(pose[F].tolist(), an['goal_xy'], a.n_cont, seed, v0=float(st[F, 0]))
        except RuntimeError as e:
            report['sampler_failed'] += 1; entry['drop'] = f'sampler: {e}'[:120]; report.setdefault('dropped', []).append(entry); continue
        for j, r in enumerate(conts):
            rid = f'{ep}__c{j}'
            json.dump({k: (np.asarray(v).tolist() if k in ('waypoints', 'speeds', 'stations', 'headings') else v) for k, v in r.items()}, open(os.path.join(rdir, rid + '.json'), 'w'))
            rows.append(dict(id=rid, group=an['group'], split=an['split'], cls=an['cls'], anchor_id=an['anchor_id'], F=F, case=p1[pid]['case'], route=p1[pid]['route'], run=True,
                             tier=i, episode_seed=int(hashlib.md5(rid.encode()).hexdigest()[:8], 16),
                             extra=['--mode', 'branch', '--branch-frame', str(F), '--branch-route', f'{a.cluster_route_prefix}/{rid}.json', '--horizon-s', a.horizon_s]))
        report['accepted'] += 1; report['per_class'][an['cls']] = report['per_class'].get(an['cls'], 0) + 1
    json.dump(rows, open(os.path.join(a.out, 'tasks_pass2_crm.json'), 'w'), indent=1)
    json.dump(report, open(os.path.join(a.out, 'report.json'), 'w'), indent=1)
    print({k: v for k, v in report.items() if k != 'dropped'}, '->', len(rows), 'pass-2 rows')


if __name__ == '__main__':
    main()
