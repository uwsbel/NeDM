"""Does the start-of-route exclusion zone cost planning quality? Frozen checkpoints, 1,200 already-labelled choices.

Applies the vehicle exclusion in corridor space to the stored v2 corridors of the held-out arenas (samples within
along <= HALF_LENGTH+margin and |lateral| <= HALF_WIDTH+margin of the route start are marked invalid, the height and
range references move to the first station with valid samples outside the zone, slopes are recomputed), then scores
with the frozen matched ensembles and recomputes route choice at matched speed.
"""
import argparse, glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_riskmodel import Net, route_logit
from sensor_train_v2 import Data, VARIANTS, route_choice, CH
from vehicle_corridor import HALF_LENGTH_M, HALF_WIDTH_M_VEH, MIN_VALID_PER_STATION

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
N_LAT, HALF_W = 32, 6.0


def mask_corridors(X12, route_len, margin):
    """X12: (N,12,96,32) raw (unnormalised) corridors. Returns a copy with the start exclusion applied."""
    X = X12.astype(np.float32).copy()
    n, _, S, L = X.shape
    lateral = np.linspace(-HALF_W, HALF_W, L)
    across_ok = np.abs(lateral) <= HALF_WIDTH_M_VEH + margin
    for i in range(n):
        ds = route_len[i] / (S - 1)
        along = np.arange(S) * ds
        zone = (along <= HALF_LENGTH_M + margin)[:, None] & across_ok[None, :]
        valid = (X[i, 4] > 0.5) & ~zone
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
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--files', nargs='+', required=True); ap.add_argument('--ck-dir', required=True)
    ap.add_argument('--variants', default='H,Dabs'); ap.add_argument('--tag', default='matched')
    ap.add_argument('--margin', type=float, default=1.5); ap.add_argument('--out', required=True)
    ap.add_argument('--train-arenas', default='f104,g228,g203,g217'); ap.add_argument('--eval-arenas', default='g216,g231')
    a = ap.parse_args()
    res = {}
    for v in a.variants.split(','):
        D = Data(a.files, v, a.train_arenas.split(','), a.eval_arenas.split(','))
        sel = [CH.index(c) for c in VARIANTS[v]]
        raw = np.concatenate([np.load(f, allow_pickle=True)['X12'] for f in a.files])   # unnormalised, for masking
        rl = np.concatenate([np.load(f, allow_pickle=True)[k] for f in a.files
                             for k in (['route_len12'] if 'route_len12' in np.load(f, allow_pickle=True) else ['route_len'])])
        idx = np.where(D.eval_designed)[0]
        masked = mask_corridors(raw[idx], rl[idx].astype(float), a.margin)
        nm = D.norm; cont = nm['cont_index']
        Xm = masked[:, sel]
        Xm[:, cont] = (Xm[:, cont] - nm['mu'][None, :, None, None]) / nm['sd'][None, :, None, None]
        Xm = np.concatenate([Xm, np.ones((len(Xm), 1, Xm.shape[2], Xm.shape[3]), np.float32)], 1)
        scores = {}
        for tag, Xuse in (('plain', D.X[idx]), ('masked', Xm)):
            zs = []
            for p in sorted(glob.glob(f'{a.ck_dir}/{a.tag}_{v}_s*.pt')):
                ck = torch.load(p, map_location=DEV, weights_only=False)
                m = Net(ck['cin'], ck['nctx'], arch=ck['arch'], layers=ck.get('layers', 2)).to(DEV)
                m.load_state_dict(ck['state']); m.eval(); out = []
                with torch.no_grad():
                    for i in range(0, len(Xuse), 512):
                        out.append(route_logit(m(torch.tensor(Xuse[i:i + 512], device=DEV),
                                                 torch.tensor(D.ctx[idx][i:i + 512], device=DEV))).float().cpu().numpy())
                zs.append(np.concatenate(out))
            scores[tag] = np.mean(zs, 0)
        sub = {k: D.d[k][idx] for k in ('group', 'profile', 'unsafe', 'fail')}
        full = np.ones(len(idx), bool)
        res[v] = {t: route_choice(scores[t], sub, full) for t in scores}
        res[v]['invalid_fraction_masked'] = float(1 - masked[:, 4].mean())
        res[v]['rank_shift_spearman'] = float(np.corrcoef(np.argsort(np.argsort(scores['plain'])),
                                                          np.argsort(np.argsort(scores['masked'])))[0, 1])
        p, mk = res[v]['plain'], res[v]['masked']
        print(f"{v}: picked-unsafe plain {100*p['picked_unsafe']:5.2f}% -> masked {100*mk['picked_unsafe']:5.2f}%  "
              f"(avoidable {100*p['avoidable_unsafe']:.2f} -> {100*mk['avoidable_unsafe']:.2f}; cells {p['cells']}; "
              f"invalid {100*res[v]['invalid_fraction_masked']:.1f}%)", flush=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
