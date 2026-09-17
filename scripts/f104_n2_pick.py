"""Night-2 closed loop, step 2: pick one route per arm per fresh test group, and build the cluster task list.

Arms (fixed in PLAN.md before any result):
  1 control  : night-1 deployed model (H1 ensemble, 22-d context with the settled state imputed to the training
               median -- verified to leave 56/60 night-1 picks unchanged) on night-1 candidates
  2 model    : night-2 model on night-1 candidates          -> isolates the model
  3 sampler  : night-2 model on night-2 candidates          -> isolates the proposal
  4 abstain  : arm 3, but if the best OR second-best candidate scores >= 5%, drive the anchor the model rates
               safest instead of the argmin
  5 anchor6  : always the 6 m/s straight line (fixed policy reference)
  7/8 fixed2 : 2 m/s fixed, geometry-only candidates, night-1 vs night-2 model (the constrained regime)
  6 pessimist: night-2 model on night-2 candidates, but ranked by the ensemble's MOST PESSIMISTIC member
               (max over seeds) instead of its mean -- aimed at last night's failure mode, where the argmin
               landed on a single seed's over-optimistic score (declared before any night-2 result was seen)
All arms of a group share one array task so Chrono is node-deterministic; identical picks are driven once.
"""
import argparse, glob, hashlib, json, os, sys
import numpy as np, torch
sys.path.insert(0, 'scripts')
from f104_night_train import HazardNet, route_logit, DEV
from f104_n2_train import Net

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
CAND = ROOT + '/night2_v1/testcand'
OUT = ROOT + '/night2_v1/closed'
ABSTAIN = 0.05


def load_night1():
    out = []
    for p in sorted(glob.glob(ROOT + '/night_v1/final/H1_full_s*.pt')):
        ck = torch.load(p, map_location=DEV, weights_only=False)
        m = HazardNet(6, len(ck['ctx_mu'])).to(DEV); m.load_state_dict(ck['state']); m.eval(); out.append((m, ck))
    return out


def load_night2(pattern):
    out = []
    for p in sorted(glob.glob(pattern)):
        ck = torch.load(p, map_location=DEV, weights_only=False)
        m = Net(ck['cin'], ck['nctx'], arch=ck['arch'], layers=ck.get('layers', 2)).to(DEV)
        m.load_state_dict(ck['state']); m.eval(); out.append((m, ck))
    return out


def risk(models, z, ctx_full, kind, agg='mean'):
    zs = []
    for m, ck in models:
        nm = ck['norm']; X = z['X'].astype(np.float32).copy()
        X[:, :4] = (X[:, :4] - nm['mu'][None, :, None, None]) / nm['sd'][None, :, None, None]
        X = np.concatenate([X, np.ones((len(X), 1, 96, 32), np.float32)], 1)
        cols = ck['ctx_cols'] if kind == 'n2' else list(range(22))
        C = (ctx_full[:, cols] - ck['ctx_mu']) / ck['ctx_sd']
        with torch.no_grad():
            zs.append(route_logit(m(torch.tensor(X, device=DEV), torch.tensor(C, device=DEV))).cpu().numpy())
    z = np.max(zs, 0) if agg == 'max' else np.mean(zs, 0)
    return 1 - np.exp(-np.exp(z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n2-models', default=ROOT + '/night2_v1/final/N2_s*.pt')
    ap.add_argument('--shards', type=int, default=40)
    a = ap.parse_args()
    os.makedirs(OUT + '/routes', exist_ok=True)
    m1, m2 = load_night1(), load_night2(a.n2_models)
    print(f'night-1 ensemble {len(m1)}, night-2 ensemble {len(m2)}')
    med = np.median(np.load(ROOT + '/night2_v1/station_ds_fix.npz', allow_pickle=True)['ctx'][:, :17], 0)
    groups = [g['group'] for g in json.load(open(ROOT + '/night2_v1/fresh_test_groups.json'))['groups']]
    rows, tasks = [], []
    for g in groups:
        z1 = np.load(f'{CAND}/{g}__night1.npz'); z2 = np.load(f'{CAND}/{g}__night2.npz')
        z3 = np.load(f'{CAND}/{g}__fixed2.npz')
        full = lambda z: np.concatenate([np.repeat(med[None], len(z['geom_ctx']), 0), z['geom_ctx']], 1).astype(np.float32)
        p_c1 = risk(m1, z1, full(z1), 'n1')
        p_n1 = risk(m2, z1, full(z1), 'n2')
        p_n2 = risk(m2, z2, full(z2), 'n2')
        p_n2_max = risk(m2, z2, full(z2), 'n2', agg='max')
        anch = z2['anchor'].astype(bool)
        srt = np.sort(p_n2)
        pick = {'control': ('night1', int(np.argmin(p_c1)), float(p_c1.min())),
                'model': ('night1', int(np.argmin(p_n1)), float(p_n1.min())),
                'sampler': ('night2', int(np.argmin(p_n2)), float(p_n2.min()))}
        if srt[0] >= ABSTAIN or srt[1] >= ABSTAIN:
            ai = int(np.where(anch)[0][np.argmin(p_n2[anch])])
            pick['abstain'] = ('night2', ai, float(p_n2[ai]))
        else:
            pick['abstain'] = pick['sampler']
        pick['pessimist'] = ('night2', int(np.argmin(p_n2_max)), float(p_n2_max.min()))
        p_f2_old = risk(m1, z3, full(z3), 'n1'); p_f2_new = risk(m2, z3, full(z3), 'n2')
        pick['fixed2_control'] = ('fixed2', int(np.argmin(p_f2_old)), float(p_f2_old.min()))
        pick['fixed2_new'] = ('fixed2', int(np.argmin(p_f2_new)), float(p_f2_new.min()))
        six = np.where(anch & (z2['cruise'] == 6.0) & (np.abs(z2['offset']) < 1e-6))[0]
        pick['anchor6'] = ('night2', int(six[0]), float(p_n2[six[0]])) if len(six) else pick['sampler']
        shard = int(hashlib.md5(g.encode()).hexdigest(), 16) % a.shards
        seen = {}
        for arm, (cs, idx, p) in pick.items():
            key = (cs, idx)
            first = key not in seen
            seen.setdefault(key, arm)
            rid = f'{g}__{seen[key]}'
            if first:
                z = {'night1': z1, 'night2': z2, 'fixed2': z3}[cs]
                json.dump({'waypoints': z['wp'][idx].tolist(), 'speeds': z['sp'][idx].tolist(),
                           'stations': z['st'][idx].tolist(), 'headings': z['hd'][idx].tolist(),
                           'meta': {'candidate': f'n2_{arm}', 'scene_id': g, 'cand_set': cs, 'cand_index': idx}},
                          open(f'{OUT}/routes/{rid}.json', 'w'))
            tasks.append(dict(id=rid, group_id=g, arm=arm, cand_set=cs, cand_index=idx, risk=p,
                              shard=shard, run=first))
        rows.append(dict(group=g, abstained=bool(pick['abstain'] != pick['sampler']),
                         p_control=float(p_c1.min()), p_model=float(p_n1.min()), p_sampler=float(p_n2.min()),
                         p_second_sampler=float(srt[1]),
                         lat_control=float(z1['max_lateral'][pick['control'][1]]) if not np.isnan(z1['max_lateral'][pick['control'][1]]) else None,
                         speed_sampler=float(z2['mean_speed'][pick['sampler'][1]]),
                         speed_control=float(z1['mean_speed'][pick['control'][1]]) if 'mean_speed' in z1 else None))
    json.dump(rows, open(OUT + '/picks.json', 'w'), indent=1)
    json.dump(tasks, open(OUT + '/tasks_cluster.json', 'w'), indent=1)
    n_run = sum(t['run'] for t in tasks)
    print(f'{len(groups)} groups, {len(tasks)} arm-picks, {n_run} episodes to drive, {sum(r["abstained"] for r in rows)} abstentions')
    for arm in ('control', 'model', 'sampler', 'abstain', 'anchor6', 'pessimist', 'fixed2_control', 'fixed2_new'):
        ps = [t['risk'] for t in tasks if t['arm'] == arm]
        print(f'  {arm:8s} model risk of its pick: median {np.median(ps):.4f}  p90 {np.percentile(ps,90):.3f}')


if __name__ == '__main__':
    main()
