"""Train the night-2 deployment ensemble on every training route (dev fold included; selection is already done)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, 'scripts')
from f104_n2_train import Data, train_one

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', default=ROOT + '/night2_v1/station_ds_all.npz')
    ap.add_argument('--ctx', default='none')
    ap.add_argument('--arch', default='gru')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--tag', default='N2')
    ap.add_argument('--sources', default='')      # comma list to restrict, e.g. 'designed'
    a = ap.parse_args()
    D = Data(a.ds, a.ctx)
    fit = D.fit | D.dev                                   # all training-split rows
    if a.sources:
        keep = set(a.sources.split(','))
        fit = fit & np.array([s in keep for s in D.d['source'].astype(str)])
    os.makedirs(ROOT + '/night2_v1/final', exist_ok=True)
    print(f'fit rows {int(fit.sum())} over {len(np.unique(D.d["group"][fit].astype(str)))} groups', flush=True)
    rows = []
    for s in range(a.seeds):
        p = f'{ROOT}/night2_v1/final/{a.tag}_s{s}.pt'
        _, m = train_one(D, arch=a.arch, seed=s, epochs=a.epochs, fit_mask=fit, save=p)
        rows.append(m); print(f'  seed {s}: G_unsafe {m["G_unsafe"]:.3f} (in-sample dev) -> {p}', flush=True)
    json.dump(rows, open(f'{ROOT}/night2_v1/final/{a.tag}_meta.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
