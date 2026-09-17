"""How much does the frozen planner lose when it can only see a limited range? 1,200 already-labelled choices.

The continuous runner can crop the overhead frame to a radius around the vehicle (`--sense-radius-m`), which is
what an onboard sensor would give. That leaves the far half of a 25-35 m route unobserved, and the frozen
checkpoints never saw an unobserved sample in training. Thirty Chrono missions cannot resolve what that costs, so
this measures it where the labels already exist: the stored v2 corridors of the held-out arenas, masked in corridor
space exactly as the runner would mask them (the vehicle sits at the route start, so a corridor sample at station
`s` and lateral offset `l` is `hypot(s, l)` from the sensor), then scored with the frozen ensembles and passed
through the same route-choice metric.

Both masks are applied: the vehicle footprint exclusion at the route start and the range limit.
"""
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_riskmodel import Net, route_logit
from sensor_train_v2 import Data, VARIANTS, route_choice, CH
from vehicle_corridor import HALF_LENGTH_M, HALF_WIDTH_M_VEH, MIN_VALID_PER_STATION

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
HALF_W = 6.0


def remask(X12, route_len, margin, radius_m):
    """Vehicle exclusion at the route start plus, if radius_m is finite, a sensing-range limit."""
    X = X12.astype(np.float32).copy()
    n, _, S, L = X.shape
    lateral = np.linspace(-HALF_W, HALF_W, L)
    across_ok = np.abs(lateral) <= HALF_WIDTH_M_VEH + margin
    frac_valid = np.empty(n)
    for i in range(n):
        ds = route_len[i] / (S - 1)
        along = np.arange(S) * ds
        zone = (along <= HALF_LENGTH_M + margin)[:, None] & across_ok[None, :]
        valid = (X[i, 4] > 0.5) & ~zone
        if np.isfinite(radius_m):
            valid &= np.hypot(along[:, None], lateral[None, :]) <= radius_m
        z_rel = X[i, 0]
        ref = next((s for s in range(S) if valid[s].sum() >= MIN_VALID_PER_STATION), None)
        z0 = float(z_rel[ref][valid[ref]].mean()) if ref is not None else 0.0
        r0 = float(X[i, 5][ref][valid[ref]].mean()) if ref is not None else float(X[i, 5].mean())
        zf = np.where(valid, z_rel - z0, 0.0)
        X[i, 0] = zf
        X[i, 1] = np.clip(np.gradient(zf, max(ds, 1e-3), axis=0), -2, 2)
        X[i, 2] = np.clip(np.gradient(zf, 2 * HALF_W / (L - 1), axis=1), -2, 2)
        X[i, 4] = valid.astype(np.float32)
        X[i, 6] = np.where(valid, X[i, 5] - r0, 0.0)
        X[i, 5] = np.where(valid, X[i, 5], 110.0)
        X[i, 7] = np.where(valid, X[i, 7], 0.0)
        X[i, 11] = np.where(valid, X[i, 11], 0.0)
        frac_valid[i] = valid.mean()
    return X, frac_valid


def score(Xuse, ctx, ck_dir, tag, v):
    zs = []
    for p in sorted(glob.glob(f'{ck_dir}/{tag}_{v}_s*.pt')):
        ck = torch.load(p, map_location=DEV, weights_only=False)
        m = Net(ck['cin'], ck['nctx'], arch=ck['arch'], layers=ck.get('layers', 2)).to(DEV)
        m.load_state_dict(ck['state']); m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(Xuse), 512):
                out.append(route_logit(m(torch.tensor(Xuse[i:i + 512], device=DEV),
                                         torch.tensor(ctx[i:i + 512], device=DEV))).float().cpu().numpy())
        zs.append(np.concatenate(out))
    return np.mean(zs, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True); ap.add_argument('--ck-dir', required=True)
    ap.add_argument('--variants', default='Dabs,H'); ap.add_argument('--tag', default='matched')
    ap.add_argument('--margin', type=float, default=1.5); ap.add_argument('--out', required=True)
    ap.add_argument('--radii', nargs='+', type=float, default=[1e9, 30., 25., 20., 15.])
    ap.add_argument('--train-arenas', default='f104,g228,g203,g217'); ap.add_argument('--eval-arenas', default='g216,g231')
    a = ap.parse_args()
    res = {}
    raw = np.concatenate([np.load(f, allow_pickle=True)['X12'] for f in a.files])
    rl = []
    for f in a.files:
        z = np.load(f, allow_pickle=True)
        rl.append(z['route_len12'] if 'route_len12' in z else z['route_len'])
    rl = np.concatenate(rl).astype(float)
    for v in a.variants.split(','):
        D = Data(a.files, v, a.train_arenas.split(','), a.eval_arenas.split(','))
        sel = [CH.index(c) for c in VARIANTS[v]]
        idx = np.where(D.eval_designed)[0]
        sub = {k: D.d[k][idx] for k in ('group', 'profile', 'unsafe', 'fail')}
        full = np.ones(len(idx), bool)
        res[v] = {}
        for R in a.radii:
            Xm, fv = remask(raw[idx], rl[idx], a.margin, R)
            nm = D.norm; cont = nm['cont_index']
            Xm = Xm[:, sel]
            Xm[:, cont] = (Xm[:, cont] - nm['mu'][None, :, None, None]) / nm['sd'][None, :, None, None]
            Xm = np.concatenate([Xm, np.ones((len(Xm), 1, Xm.shape[2], Xm.shape[3]), np.float32)], 1)
            z = score(Xm, D.ctx[idx], a.ck_dir, a.tag, v)
            key = 'unlimited' if R > 1e6 else f'{R:.0f} m'
            res[v][key] = route_choice(z, sub, full) | {'mean_valid_fraction': float(fv.mean())}
            print(v, key, 'valid %.3f' % fv.mean(), json.dumps(res[v][key]), flush=True)
    json.dump(res, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
