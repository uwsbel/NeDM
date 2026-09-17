"""Paired closed-loop test: OLD (production multihead_seed0) vs NEW (hazard ensemble) choose from the
SAME 256 fixed-speed candidate routes per held-out group; both picks later run on the same node.

Test groups reuse the exact candidate generator + seeds of the earlier fixed-speed smooth arm, so the
OLD pick must reproduce that run's smooth_best route (checked). Val groups add power.
"""
import json, glob, os, sys, hashlib
import numpy as np, torch
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from nedm.traverse.fdm_mppi import MPPIConfig, validate_reference
from f104_fixed_speed_geometry_test import deform_smooth
from score_f104_prospective import feats
from train_f104_multihead import MultiHead
from f104_night_dataset import station_tensor
from f104_night_train import HazardNet, route_logit, DEV

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
RUNS = ROOT + '/production_v2/runs'
OUT = ROOT + '/night_v1/paired'
N = 256


def bases(split):
    """offset-0 constant_2 base route per group, from locally available designed routes."""
    info = {}
    pros = {t['id']: t for t in json.load(open(ROOT + '/prospective/f104_undriven_test.json'))}
    for d in sorted(glob.glob(RUNS + '/*_route_00')):
        try:
            c = json.load(open(d + '/case.json'))
        except Exception:
            continue
        if c['split'] != split: continue
        cr = np.load(d + '/command_reference.npz')
        info[c['id']] = dict(case=c, dirname=d, wp=cr['reference_waypoints'], sp=cr['reference_speeds'],
                             st=cr['reference_stations'], hd=cr['reference_headings'])
    if split == 'test':
        for pid, t in pros.items():
            if not pid.endswith('_route_00'): continue
            g = t['group_id']
            if g in info: continue
            anyd = sorted(glob.glob(f'{RUNS}/{g}_route_*'))
            if not anyd: continue
            info[g] = dict(case=json.load(open(anyd[0] + '/case.json')), dirname=anyd[0],
                           wp=np.asarray(t['waypoints'], float), sp=np.asarray(t['speeds'], float),
                           st=np.asarray(t['stations'], float), hd=np.asarray(t['headings'], float))
    return info


def main(new_glob):
    os.makedirs(OUT + '/routes', exist_ok=True)
    old = torch.load(ROOT + '/multihead_v1/multihead_seed0.pt', map_location=DEV, weights_only=False)
    om = MultiHead(old['n_vec']).to(DEV); om.load_state_dict(old['state_dict']); om.eval()
    news = []
    for p in sorted(glob.glob(new_glob)):
        ck = torch.load(p, map_location=DEV, weights_only=False)
        m = HazardNet(6, len(ck['ctx_mu'])).to(DEV); m.load_state_dict(ck['state']); m.eval()
        news.append((m, ck))
    print(f'NEW ensemble: {len(news)} checkpoints from {new_glob}')
    prior = {x['group_id']: x for x in json.load(open(ROOT + '/fixed_speed_geom_v1/tasks.json'))
             if x.get('arm') == 'smooth' and x['pick'] == 'best'}
    tasks, rows, reproduced = [], [], []
    for split, seed0 in (('test', 1000), ('val', 5000)):
        B = bases(split)
        for gi, (g, b) in enumerate(sorted(B.items())):
            base = {'waypoints': b['wp'], 'speeds': b['sp'], 'stations': b['st'], 'headings': b['hd'], 'meta': {}}
            lay = b['case']['layout']
            anchor = np.array([lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']])
            state0 = np.asarray(np.load(b['dirname'] + '/anchor_state.npz', allow_pickle=True)['state'], np.float32)
            gxy = np.asarray(b['case']['goal_xy'], np.float32); sxy = np.asarray(lay['start_xy'], np.float32)
            rel = gxy - sxy
            extra = np.array([rel[0], rel[1], np.linalg.norm(rel), float(lay['start_yaw'])], np.float32)
            cfg = MPPIConfig(samples=N, knots=3, lateral_sigma_m=2.6, speed_sigma_mps=0.0, max_lateral_m=6.0,
                             max_speed_delta_mps=0.0, max_speed_mps=6.0, max_curvature_inv_m=0.125,
                             arena_half_extent_m=40.0)
            rng = np.random.default_rng(seed0 + gi); cand = []
            for _ in range(N * 4):
                if len(cand) >= N: break
                p = np.r_[np.clip(rng.normal(0, 2.6, 3), -6, 6), np.zeros(3)]
                r = deform_smooth(base, p, anchor, cfg)
                if validate_reference(r, [], cfg, anchor)['valid']:
                    cand.append(r)
            if len(cand) < 8: continue
            # OLD scores (exactly as the earlier fixed-speed arm)
            P, V, XS, LS = [], [], [], []
            for r in cand:
                patch, scal = feats(r['waypoints'], r['speeds'], r['stations'], r['headings'])
                P.append(patch); V.append(np.concatenate([scal, state0, extra]))
                X, L = station_tensor(np.asarray(r['waypoints']), np.asarray(r['speeds']), np.asarray(r['stations']))
                XS.append(X); LS.append(L)
            P = np.stack(P); V = np.stack(V); XS = np.stack(XS)
            with torch.no_grad():
                a, _, _ = om(torch.tensor((P - old['pmu']) / old['psd'], device=DEV),
                             torch.tensor((V - old['mu']) / old['sd'], device=DEV))
                p_old = torch.sigmoid(a).cpu().numpy()
                zs = []
                for m, ck in news:
                    Xn = XS.copy(); nm = ck['norm']
                    Xn[:, :4] = (XS[:, :4] - nm['mu'][None, :, None, None]) / nm['sd'][None, :, None, None]
                    Xn = np.concatenate([Xn, np.ones((len(Xn), 1, 96, 32), np.float32)], 1)
                    ctx = np.stack([np.concatenate([state0, [rel[0], rel[1], np.linalg.norm(rel),
                                                             float(lay['start_yaw']), L]]) for L in LS]).astype(np.float32)
                    ctx = (ctx - ck['ctx_mu']) / ck['ctx_sd']
                    zs.append(route_logit(m(torch.tensor(Xn, device=DEV), torch.tensor(ctx, device=DEV))).cpu().numpy())
            z_new = np.mean(zs, 0)
            p_new = 1 - np.exp(-np.exp(z_new))
            io, inew = int(np.argmin(p_old)), int(np.argmin(p_new))
            if split == 'test' and g in prior:
                reproduced.append(abs(prior[g]['predicted_fail'] - float(p_old[io])) < 1e-3)
            shard = int(hashlib.md5(g.encode()).hexdigest(), 16) % 20
            for arm, i in (('old', io), ('new', inew)):
                r = cand[i]; rid = f'{g}__night_{arm}'
                json.dump({'waypoints': np.asarray(r['waypoints']).tolist(), 'speeds': np.asarray(r['speeds']).tolist(),
                           'stations': np.asarray(r['stations']).tolist(), 'headings': np.asarray(r['headings']).tolist(),
                           'meta': {'candidate': f'night_{arm}', 'scene_id': g, 'cand_index': i}},
                          open(f'{OUT}/routes/{rid}.json', 'w'))
                tasks.append(dict(id=rid, group_id=g, split=split, arm=arm, cand_index=i, shard=shard,
                                  same_as_other=(io == inew), p_old=float(p_old[i]), p_new=float(p_new[i])))
            rows.append(dict(group=g, split=split, io=io, inew=inew, same=(io == inew),
                             p_old_of_old=float(p_old[io]), p_new_of_old=float(p_new[io]),
                             p_old_of_new=float(p_old[inew]), p_new_of_new=float(p_new[inew])))
    json.dump(tasks, open(OUT + '/tasks.json', 'w'), indent=1)
    json.dump(rows, open(OUT + '/picks.json', 'w'), indent=1)
    same = sum(r['same'] for r in rows)
    print(f'groups {len(rows)} (test {sum(r["split"]=="test" for r in rows)}, val {sum(r["split"]=="val" for r in rows)}); '
          f'identical picks {same}; OLD pick reproduces earlier run: {sum(reproduced)}/{len(reproduced)}')


if __name__ == '__main__':
    main(sys.argv[1])
