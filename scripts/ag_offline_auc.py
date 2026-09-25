#!/usr/bin/env python3
"""arena_gator_20260925 module E5a: offline within-group AUC of ci_train ensembles on the held-out rows of each arena.

Every model is scored on the same rows of each evaluation file, so models trained on different arenas can be compared:
  val   rows of the val split (never fitted by any model of this study);
  dev   training-split rows of ga_train's dev fold (md5(group) % 5 == 0): not fitted by holdout-mode models, fitted by
        deploy-mode models of the same arena (excluded automatically, see the guard);
  test  rows are never read.
Evaluation files: scripts/ag_subset.py --groups <arena>=0 --keep-dev-fold (dev fold + val + test rows of one arena and
vehicle, no other training rows).

Guard: each model's fitted groups are read from its training run's <tag>_logits.npz (group[fit]); evaluation rows of
those groups are dropped and counted (group names carry no vehicle, so a model that fitted a route with one vehicle is
never scored on the same route driven by the other). A read-out with dropped rows is flagged 'partly_fitted'.

Scores: ci_train.load_ci_model + ci_train.score on the raw corridor and the raw geometry context (the checkpoint round
trip path, equal to the trainer's predict() to 1e-5); ensemble logit = mean of the member logits (as ci_train);
metrics = ci_train.metrics (ga_train.regime_metrics: W_* = within-group AUC, cells keyed group|domain; P_* pooled AUC;
pick_fail* = failure of the lowest-risk route per group) per domain x startup / established / all.

  PYTHONPATH=src:scripts python scripts/ag_offline_auc.py \
    --model M1a='e5/deploy/M1a/M1a_rigid_deploy_s*.pt' --model M3a='e5/deploy/M3a/*_s*.pt' \
    --eval f104=e5/eval/EV_f104_hmmwv_rigid.npz --eval g203=e5/eval/EV_g203_hmmwv_rigid.npz --out e5/offline/auc.json
--check-trainer: also compare the scorer's numbers with the trainer's own summary <tag>.json (ensemble.dev /
ensemble.heldout) where the evaluation rows are exactly the trainer's rows (reported, not asserted).
"""
import argparse, glob, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KEYS = ('W_unsafe', 'Wn_unsafe', 'W_fail', 'Wn_fail', 'P_unsafe', 'P_fail', 'pick_fail', 'pick_fail_unsafe', 'random_fail_unsafe',
        'oracle_fail_unsafe', 'rate_unsafe', 'rate_fail', 'brier_unsafe', 'n', 'n_groups')
LIGHT = ('id', 'group', 'split', 'domain', 'anchor_frame', 'unsafe', 'fail', 'episode', 'source', 'profile', 'arena', 'vehicle')


def md5_dev(g):
    return int(hashlib.md5(g.encode()).hexdigest(), 16) % 5 == 0


def sha256_file(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def slim(m):
    """ci_train.metrics block -> {domain: {regime: {KEYS}}}."""
    return {dk: {rg: {k: v for k, v in blk.items() if k in KEYS} for rg, blk in dv.items()} for dk, dv in m.items()}


def load_eval(path):
    z = np.load(path, allow_pickle=True)
    d = {k: z[k] for k in LIGHT if k in z.files}
    sp = d['split'].astype(str); grp = d['group'].astype(str)
    dev = (sp == 'train') & np.array([md5_dev(g) for g in grp]); val = sp == 'val'
    other_train = (sp == 'train') & ~dev
    assert not other_train.any(), f'{path}: {int(other_train.sum())} training rows outside the dev fold (not an evaluation file)'
    keep = np.where(dev | val)[0]
    d = {k: v[keep] for k, v in d.items()}
    X = z['X'][keep]; geom = None
    import ci_train as CT
    geom = z['ctx'][keep][:, CT.GEOM_COLS].astype(np.float32)
    return dict(path=os.path.abspath(path), sha256=sha256_file(path), d=d, X=X, geom=geom, dev=dev[keep], val=val[keep],
                vehicle=sorted(str(x) for x in set(d['vehicle'].astype(str))) if 'vehicle' in d else None,
                arena=sorted(str(x) for x in set(d['arena'].astype(str))) if 'arena' in d else None)


def fitted_groups(ck_paths, cks):
    """Groups fitted by the training run(s) behind the checkpoints: <out>/<tag>_logits.npz next to them."""
    out, files = set(), []
    for p, ck in zip(ck_paths, cks):
        lp = os.path.join(os.path.dirname(p), f"{ck['tag']}_logits.npz")
        if lp in files:
            continue
        assert os.path.isfile(lp), f'no training logits file {lp} (needed for the fitted-group guard)'
        z = np.load(lp, allow_pickle=True)
        out |= set(z['group'][z['fit'].astype(bool)].astype(str)); files.append(lp)
    return out, files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', action='append', required=True, help='NAME=checkpoint glob (one ensemble)')
    ap.add_argument('--eval', action='append', required=True, help='NAME=evaluation npz (ag_subset --groups <arena>=0 --keep-dev-fold)')
    ap.add_argument('--out', required=True); ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--check-trainer', action='store_true')
    a = ap.parse_args(argv)
    import torch
    import ci_train as CT
    t0 = time.time()
    evals = {}
    for e in a.eval:
        name, p = e.split('=', 1); evals[name] = load_eval(p)
        E = evals[name]
        print(f'eval {name}: {len(E["X"])} rows (dev {int(E["dev"].sum())}, val {int(E["val"].sum())}) arena {E["arena"]} vehicle {E["vehicle"]}', flush=True)
    res = dict(tool='scripts/ag_offline_auc.py', tool_sha256=sha256_file(__file__), ci_train_sha256=sha256_file(HERE / 'ci_train.py'),
               argv=sys.argv[1:] if argv is None else argv, started=time.strftime('%F %T'),
               evals={k: dict(path=v['path'], sha256=v['sha256'], rows_dev=int(v['dev'].sum()), rows_val=int(v['val'].sum()),
                              groups_dev=int(len(set(v['d']['group'][v['dev']].astype(str)))), groups_val=int(len(set(v['d']['group'][v['val']].astype(str)))),
                              arena=v['arena'], vehicle=v['vehicle']) for k, v in evals.items()},
               models={}, results={})
    for m in a.model:
        name, pat = m.split('=', 1)
        paths = sorted(glob.glob(pat)); assert paths, f'{name}: no checkpoint matches {pat}'
        members = [CT.load_ci_model(p) for p in paths]; cks = [ck for _, ck in members]
        fg, lfiles = fitted_groups(paths, cks)
        res['models'][name] = dict(checkpoints=[dict(path=os.path.abspath(p), sha256=sha256_file(p), seed=ck['seed'], tag=ck['tag'], mode=ck['mode'],
                                                     train_rows=ck['train_rows'], split_hash=ck['split_hash']) for p, ck in zip(paths, cks)],
                                   fit_logits=lfiles, fitted_groups=len(fg), ds=sorted({x for ck in cks for x in ck['ds']}),
                                   modes=sorted({ck['mode'] for ck in cks}))
        res['results'][name] = {}
        for en, E in evals.items():
            s_mem = []
            for model, ck in members:
                s_mem.append(CT.score(model, ck, E['X'], E['geom'], bs=a.bs))
            S = np.stack(s_mem); ens = S.mean(0)
            infit = np.array([g in fg for g in E['d']['group'].astype(str)])
            out = {}
            for sname, smask in (('val', E['val']), ('dev', E['dev']), ('dev+val', E['dev'] | E['val'])):
                mm = smask & ~infit
                blk = dict(rows=int(smask.sum()), rows_dropped_fitted=int((smask & infit).sum()), partly_fitted=bool((smask & infit).any()),
                           groups=int(len(set(E['d']['group'][mm].astype(str)))))
                if mm.any():
                    blk['ensemble'] = slim(CT.metrics(ens, E['d'], mm))
                    blk['members_W_unsafe'] = {rg: [slim(CT.metrics(S[i], E['d'], mm))[dk][rg]['W_unsafe'] for i in range(len(S))]
                                               for dk in blk['ensemble'] for rg in ('startup', 'established')}
                out[sname] = blk
            res['results'][name][en] = out
            v = out['val'].get('ensemble', {}); dv = out['dev+val'].get('ensemble', {})
            dk = next(iter(v), None)
            if dk:
                print(f"{name:22s} {en:12s} val W_unsafe start {v[dk]['startup']['W_unsafe']:.4f} moving {v[dk]['established']['W_unsafe']:.4f}"
                      + (f" | dev+val start {dv[dk]['startup']['W_unsafe']:.4f} moving {dv[dk]['established']['W_unsafe']:.4f}" if dv and not out['dev+val']['partly_fitted'] else ' | dev+val (fitted rows dropped)' if dv else ''), flush=True)
        if a.check_trainer:
            chk = {}
            for lp in lfiles:
                sj = lp[:-len('_logits.npz')] + '.json'
                if not os.path.isfile(sj):
                    continue
                J = json.load(open(sj))['ensemble']
                for en, E in evals.items():
                    for tset, sname in (('dev', 'dev'), ('heldout', 'val')):
                        tr = J[tset]['all']
                        blk = res['results'][name][en][sname]
                        if 'ensemble' not in blk or blk['partly_fitted']:
                            continue
                        for dk in blk['ensemble']:
                            if dk not in tr:
                                continue
                            for rg in ('startup', 'established'):
                                tn, sn = tr[dk][rg].get('n'), blk['ensemble'][dk][rg].get('n')
                                if tn == sn:
                                    chk[f'{os.path.basename(sj)}|{en}|{sname}|{dk}|{rg}'] = dict(
                                        n=tn, trainer_W_unsafe=tr[dk][rg]['W_unsafe'], scorer_W_unsafe=blk['ensemble'][dk][rg]['W_unsafe'],
                                        diff=abs(tr[dk][rg]['W_unsafe'] - blk['ensemble'][dk][rg]['W_unsafe']))
            res['models'][name]['trainer_check'] = chk
            if chk:
                print(f'  trainer check {name}: {len(chk)} matching row sets, max |diff W_unsafe| {max(c["diff"] for c in chk.values()):.2e}', flush=True)
        del members
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    res['elapsed_s'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    print(f'wrote {a.out} ({res["elapsed_s"]} s)', flush=True)


if __name__ == '__main__':
    main()
