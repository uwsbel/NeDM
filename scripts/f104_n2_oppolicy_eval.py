"""Does training on the planner's own proposal distribution help ON that distribution?

The dev metric in scaling.json is computed on designed routes, which cannot see a distribution shift.
Here both models are fitted with the dev fold held out, and both are scored on the dev fold's ON-POLICY routes
(within-group ranking AUC and the unsafe rate of the model's own top pick among that group's on-policy routes).
"""
import json, sys
import numpy as np
sys.path.insert(0, 'scripts')
from f104_n2_train import Data, train_one, predict
from f104_night_train import cell_auc

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'


def main():
    D = Data(ROOT + '/night2_v1/station_ds_all.npz', 'none')
    src = D.d['source'].astype(str)
    dev_op = (D.d['split'].astype(str) == 'train') & np.array([bool(x) for x in (src == 'on_policy')]) & \
             np.array([__import__('hashlib').md5(g.encode()).hexdigest() and True for g in D.d['group'].astype(str)])
    from f104_night_train import dev_group
    isdev = np.array([dev_group(g) for g in D.d['group'].astype(str)])
    dev_op = (src == 'on_policy') & isdev
    fit_des = D.fit & (src == 'designed')
    fit_all = D.fit
    print(f'dev on-policy routes: {int(dev_op.sum())} over {len(np.unique(D.d["group"][dev_op].astype(str)))} groups')
    out = []
    for tag, m in (('designed only', fit_des), ('designed + on-policy', fit_all)):
        for seed in range(3):
            model, _ = train_one(D, arch='gru', seed=seed, epochs=30, fit_mask=m)
            s = predict(model, D, np.arange(D.n))
            grp = D.d['group'][dev_op].astype(str)
            a_un, n_un = cell_auc(D.d['unsafe'][dev_op].astype(float), s[dev_op], grp)
            a_fa, _ = cell_auc(D.d['fail'][dev_op].astype(float), s[dev_op], grp)
            pu = []
            for g in np.unique(grp):
                k = grp == g
                pu.append(D.d['unsafe'][dev_op][k][int(np.argmin(s[dev_op][k]))])
            out.append(dict(tag=tag, seed=seed, fit_rows=int(m.sum()), auc_unsafe=a_un, auc_fail=a_fa,
                            pick_unsafe=float(np.mean(pu)), pairs=n_un))
            print(f'  {tag:22s} s{seed}  within-group AUC on on-policy routes: unsafe {a_un:.3f} fail {a_fa:.3f}  '
                  f'top-pick unsafe {100*np.mean(pu):.1f}%', flush=True)
    json.dump(out, open(ROOT + '/night2_v1/onpolicy_eval.json', 'w'), indent=1)
    for tag in ('designed only', 'designed + on-policy'):
        r = [x for x in out if x['tag'] == tag]
        print(f'{tag:22s} mean AUC_unsafe {np.mean([x["auc_unsafe"] for x in r]):.3f}  '
              f'top-pick unsafe {100*np.mean([x["pick_unsafe"] for x in r]):.1f}%')


if __name__ == '__main__':
    main()
