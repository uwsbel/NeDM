"""The user's test: fix the speed profile, search geometry only, ask whether MPPI +
the learned model produce sensible route geometry.

Two arms, identical in every respect except how offsets are interpolated between knots:
  tent   - stock fdm_mppi.deform_reference (piecewise-linear; caps lateral at ~1.2 m)
  smooth - C1 smoothstep interpolation (reaches the +-4 m offsets the designed routes use)

Speed is pinned to each group's constant_2 profile and never perturbed.
"""
import json, glob, os, sys
import numpy as np, torch

sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from nedm.traverse.fdm_mppi import MPPIConfig, deform_reference, validate_reference
from nedm.traverse.planner_s import _curvature_max
from score_f104_prospective import feats
from train_f104_multihead import MultiHead

ROOT = '/home/harry/NeDM-traverse_mppi/artifacts/traverse/fdm_f104_50h_20260909'
RUNS = ROOT + '/production_v2/runs'
OUT = ROOT + '/fixed_speed_geom_v1'
os.makedirs(OUT + '/routes', exist_ok=True)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMPLE = int(os.environ.get('N_SAMPLE', '256'))


def deform_smooth(base, pars, anchor, cfg):
    """deform_reference with smoothstep (C1) interpolation instead of a linear tent."""
    xy = np.asarray(base['waypoints'], float); station = np.asarray(base['stations'], float)
    speed = np.asarray(base['speeds'], float)
    k = cfg.knots
    span = max(station[-1] - station[0], 1e-6)
    f = np.clip((station - station[0]) / span, 0.0, 1.0)
    env = np.sin(np.pi * f) ** 2
    kx = np.linspace(0.0, 1.0, k + 2); kv = np.r_[0.0, pars[:k], 0.0]
    seg = np.clip(np.searchsorted(kx, f, side='right') - 1, 0, k)
    u = (f - kx[seg]) / (kx[seg + 1] - kx[seg])
    w = u * u * (3 - 2 * u)                      # smoothstep: C1 at every knot
    lateral = (kv[seg] + (kv[seg + 1] - kv[seg]) * w) * env
    t = np.gradient(xy, axis=0); t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    n = np.stack([-t[:, 1], t[:, 0]], 1)
    pts = xy + lateral[:, None] * n
    st = np.r_[0.0, np.linalg.norm(np.diff(pts, axis=0), axis=1).cumsum()]
    hd = np.arctan2(np.gradient(pts[:, 1]), np.gradient(pts[:, 0]))
    return {'waypoints': pts, 'speeds': speed.copy(), 'stations': st, 'headings': hd,
            'meta': {'candidate': 'smoothstep'}}


def load_bases():
    """Base geometry + constant_2 speed profile for every test group."""
    pros = {t['id']: t for t in json.load(open(ROOT + '/prospective/f104_undriven_test.json'))}
    info = {}
    for d in glob.glob(RUNS + '/*'):
        n = os.path.basename(d)
        if not os.path.exists(d + '/reference.json'):
            continue
        try:
            m = json.load(open(d + '/reference.json'))['meta']; c = json.load(open(d + '/case.json'))
        except Exception:
            continue
        if c['split'] != 'test' or m.get('cruise_speed_mps') != 2.0 or m.get('lateral_offset_m') != 0.0:
            continue
        cr = np.load(d + '/command_reference.npz')
        info[c['id']] = dict(case=c, dirname=d, nominal_id=n,
                             wp=cr['reference_waypoints'], sp=cr['reference_speeds'],
                             st=cr['reference_stations'], hd=cr['reference_headings'])
    for pid, t in pros.items():
        m = t.get('meta') or {}
        if m.get('cruise_speed_mps') != 2.0 or m.get('lateral_offset_m') != 0.0:
            continue
        g = t['group_id']
        if g in info:
            continue
        anyd = sorted(glob.glob(f'{RUNS}/{g}_route_*'))
        if not anyd:
            continue
        info[g] = dict(case=json.load(open(anyd[0] + '/case.json')), dirname=anyd[0], nominal_id=pid,
                       wp=np.asarray(t['waypoints'], float), sp=np.asarray(t['speeds'], float),
                       st=np.asarray(t['stations'], float), hd=np.asarray(t['headings'], float))
    return info


def main():
    ck = torch.load(f'{ROOT}/multihead_v1/multihead_seed0.pt', map_location=DEV, weights_only=False)
    net = MultiHead(ck['n_vec']).to(DEV); net.load_state_dict(ck['state_dict']); net.eval()
    bases = load_bases()
    print(f'test groups with a constant_2 offset-0 base: {len(bases)}')
    tasks, summary = [], []
    for gi, (g, b) in enumerate(sorted(bases.items())):
        base = {'waypoints': b['wp'], 'speeds': b['sp'], 'stations': b['st'],
                'headings': b['hd'], 'meta': {}}
        lay = b['case']['layout']
        anchor = np.array([lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']])
        an = np.load(b['dirname'] + '/anchor_state.npz', allow_pickle=True)
        state0 = np.asarray(an['state'], np.float32)
        gxy = np.asarray(b['case']['goal_xy'], np.float32)
        sxy = np.asarray(lay['start_xy'], np.float32); rel = gxy - sxy
        extra = np.array([rel[0], rel[1], np.linalg.norm(rel), float(lay['start_yaw'])], np.float32)
        rng = np.random.default_rng(1000 + gi)
        for arm, fn, sig in (('tent', deform_reference, 1.0), ('smooth', deform_smooth, 2.6)):
            cfg = MPPIConfig(samples=N_SAMPLE, knots=3, lateral_sigma_m=sig, speed_sigma_mps=0.0,
                             max_lateral_m=6.0, max_speed_delta_mps=0.0, max_speed_mps=6.0,
                             max_curvature_inv_m=0.125, arena_half_extent_m=40.0)
            cand = []
            for _ in range(N_SAMPLE * 4):
                if len(cand) >= N_SAMPLE:
                    break
                p = np.r_[np.clip(rng.normal(0, sig, 3), -6, 6), np.zeros(3)]
                r = fn(base, p, anchor, cfg)
                if validate_reference(r, [], cfg, anchor)['valid']:
                    cand.append(r)
            if len(cand) < 8:
                continue
            P, V = [], []
            for r in cand:
                patch, scal = feats(r['waypoints'], r['speeds'], r['stations'], r['headings'])
                P.append(patch); V.append(np.concatenate([scal, state0, extra]))
            P = np.stack(P); V = np.stack(V)
            with torch.no_grad():
                a, s, t = net(torch.tensor((P - ck['pmu']) / ck['psd'], device=DEV),
                              torch.tensor((V - ck['mu']) / ck['sd'], device=DEV))
                pf = torch.sigmoid(a).cpu().numpy()
            t0 = np.gradient(base['waypoints'], axis=0)
            t0 /= np.maximum(np.linalg.norm(t0, axis=1, keepdims=True), 1e-9)
            n0 = np.stack([-t0[:, 1], t0[:, 0]], 1)
            reach = np.array([np.abs(((c['waypoints'] - base['waypoints']) * n0).sum(-1)).max()
                              for c in cand])
            for tag, idx in (('best', int(np.argmin(pf))), ('worst', int(np.argmax(pf)))):
                r = cand[idx]
                rid = f'{g}__{arm}_{tag}'
                json.dump({'waypoints': r['waypoints'].tolist(), 'speeds': r['speeds'].tolist(),
                           'stations': r['stations'].tolist(), 'headings': r['headings'].tolist(),
                           'meta': {'candidate': f'fixed_speed_{arm}', 'scene_id': g, 'pick': tag}},
                          open(f'{OUT}/routes/{rid}.json', 'w'))
                tasks.append(dict(id=rid, group_id=g, arm=arm, pick=tag,
                                  predicted_fail=float(pf[idx]), reach_m=float(reach[idx]),
                                  nominal_id=b['nominal_id']))
            summary.append(dict(group=g, arm=arm, n=len(cand), reach_med=float(np.median(reach)),
                                reach_max=float(reach.max()), pf_min=float(pf.min()),
                                pf_max=float(pf.max())))
    json.dump(tasks, open(OUT + '/tasks.json', 'w'), indent=2)
    json.dump(summary, open(OUT + '/summary.json', 'w'), indent=2)
    for arm in ('tent', 'smooth'):
        s = [x for x in summary if x['arm'] == arm]
        print(f'  {arm:7s}: {len(s)} groups, median reach {np.median([x["reach_med"] for x in s]):.2f} m, '
              f'max reach {max(x["reach_max"] for x in s):.2f} m, '
              f'mean candidates {np.mean([x["n"] for x in s]):.0f}')
    print(f'wrote {len(tasks)} Chrono tasks -> {OUT}/tasks.json')


if __name__ == '__main__':
    main()
