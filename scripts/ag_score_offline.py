#!/usr/bin/env python3
"""Offline scores of ci_train ensembles on evaluation-only row files (arena_gator_20260925, module E6a; REVIEW_R1 5).

Loads every checkpoint of --models (ci_train.load_ci_model; model_kind 'ci_train') and scores the rows of each --ds
file with ci_train.score (the planner's own scoring path: raw float16 corridor -> float32, geometry context = ctx
columns 17-21, the per-row history window only for history models), ensemble logit = mean of the member route logits.
Rows: every row of the file, or --splits (e.g. val for a training file; evaluation-only files carry split
'evalonly'), optionally --startup-only.
Metrics per file and per arena, on startup rows (anchor frame 0: the decision the planner makes from a standing start),
established rows (moving re-anchored starts) and all rows, with ga_train.regime_metrics (the trainer's own numbers):
  W_unsafe / W_fail   within-group ranking AUC (cells group|world): does the ensemble rank the group's routes right?
  G_unsafe / G_fail   the same within (group, designed speed profile) cells
  P_*                 pooled AUC; pick_fail / random_fail / oracle_fail: failure of the lowest-risk route of each group
  brier / ece         calibration of P(unsafe) = 1 - exp(-exp(logit))
--check-logits <trainer _logits.npz>: rows present in both are compared with the trainer's stored ensemble logits
(max abs difference; the scorer must reproduce the trainer, e.g. on the val rows of the training file).
--save-logits writes <out>.logits.npz (ids, ensemble and member logits) for later per-row read-outs.
TF32 is off by default (exact float32 scoring; the trainer's stored logits are reproduced to < 1e-4).

  PYTHONPATH=src:scripts python scripts/ag_score_offline.py --models "$K3/e5/train/M1_rigid/M1a_rigid_deploy_s*.pt" --tag M1a \
      --ds $K3/e4/evalonly/g260_rigid_test/ci_g260_hmmwv_rigid_evalonly.npz --out $K3/e6/offline/M1a_g260.json
"""
import argparse, glob, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

KEYS = ('W_unsafe', 'Wn_unsafe', 'W_fail', 'Wn_fail', 'G_unsafe', 'G_fail', 'P_unsafe', 'P_fail', 'pick_fail', 'random_fail', 'oracle_fail',
        'pick_fail_unsafe', 'random_fail_unsafe', 'oracle_fail_unsafe', 'rate_unsafe', 'rate_fail', 'brier_unsafe', 'ece_unsafe', 'n', 'n_groups')


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def load_rows(path, splits, startup_only, world):
    z = np.load(path, allow_pickle=True)
    sp = z['split'].astype(str); af = z['anchor_frame'].astype(int); dom = z['domain'].astype(int)
    m = np.ones(len(sp), bool)
    if splits:
        m &= np.isin(sp, splits)
    if startup_only:
        m &= af == 0
    if world:
        m &= dom == {'rigid': 0, 'crm': 1}[world]
    idx = np.flatnonzero(m)
    d = {k: z[k][idx] for k in ('id', 'group', 'domain', 'anchor_frame', 'fail', 'unsafe', 'split') if k in z.files}
    for k in ('profile', 'arena', 'episode', 'source', 'case_split', 'vehicle'):
        if k in z.files:
            d[k] = z[k][idx]
    X = z['X']; ctx = z['ctx']
    hist = z['hist'] if 'hist' in z.files else None; hmask = z['hmask'] if 'hmask' in z.files else None
    return d, idx, X, ctx, hist, hmask


def score_file(members, CT, X, ctx, hist, hmask, idx, chunk=4096):
    geom_cols = list(members[0][1]['geom_cols'])
    out = np.zeros((len(members), len(idx)), np.float32)
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        x = np.asarray(X[ii], np.float32); g = np.asarray(ctx[ii][:, geom_cols], np.float32)
        for j, (m, ck) in enumerate(members):
            h = hm = None
            if m.use_hist:
                h, hm = np.asarray(hist[ii], np.float32), np.asarray(hmask[ii], bool)
            out[j, s:s + len(ii)] = CT.score(m, ck, x, g, hist=h, hmask=hm)
    return out


def metrics_by(d, s, GA):
    af = d['anchor_frame'].astype(int)
    res = {}
    arenas = sorted(set(d['arena'].astype(str))) if 'arena' in d else ['all']
    for ar in arenas + (['pooled'] if len(arenas) > 1 else []):
        am = np.ones(len(s), bool) if ar in ('all', 'pooled') else d['arena'].astype(str) == ar
        blk = {}
        for name, mm in (('startup', am & (af == 0)), ('established', am & (af != 0)), ('all', am)):
            if mm.sum() == 0:
                continue
            r = GA.regime_metrics(s[mm], d, mm)
            blk[name] = {k: r.get(k) for k in KEYS}
        res[ar] = blk
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', required=True); ap.add_argument('--tag', default=None)
    ap.add_argument('--ds', nargs='+', required=True)
    ap.add_argument('--splits', default=None, help='comma list of splits to score (default: every row)')
    ap.add_argument('--startup-only', action='store_true')
    ap.add_argument('--world', choices=['rigid', 'crm'], default=None)
    ap.add_argument('--check-logits', default=None)
    ap.add_argument('--save-logits', action='store_true')
    ap.add_argument('--device', default=None)
    ap.add_argument('--tf32', action='store_true', help='allow TF32 convolutions (torch default on this GPU; off here: with TF32 the '
                    'RTX 5090 differs from the trainer logits by up to 0.015, without it by < 1e-4)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    import torch
    import ci_train as CT
    import ga_train as GA
    torch.backends.cudnn.allow_tf32 = bool(a.tf32); torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
    dev = a.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    paths = sorted(glob.glob(a.models))
    assert paths, a.models
    members = [CT.load_ci_model(p, device=dev) for p in paths]
    world = a.world or {'rigid': 'rigid', 'crm': 'crm'}.get(members[0][1].get('domain_filter'))
    res = dict(tool='scripts/ag_score_offline.py', tool_sha256=sha256_file(__file__), created=time.strftime('%Y-%m-%d %H:%M:%S'), tag=a.tag,
               models=[dict(path=os.path.relpath(p, ROOT) if p.startswith(str(ROOT)) or not os.path.isabs(p) else p, sha256=sha256_file(p),
                            tag=ck.get('tag'), seed=ck.get('seed'), domain_filter=ck.get('domain_filter'), mode=ck.get('mode')) for p, (m, ck) in zip(paths, members)],
               world=world, splits=a.splits, startup_only=a.startup_only, tf32=bool(a.tf32), device=dev, files={})
    ref = np.load(a.check_logits, allow_pickle=True) if a.check_logits else None
    saved = {}
    for f in a.ds:
        d, idx, X, ctx, hist, hmask = load_rows(f, a.splits.split(',') if a.splits else None, a.startup_only, world)
        Z = score_file(members, CT, X, ctx, hist, hmask, idx)
        s = Z.mean(0)
        blk = dict(path=str(f), sha256=sha256_file(f), rows=int(len(idx)), groups=int(len(set(d['group'].astype(str)))),
                   splits=sorted(set(d['split'].astype(str))), metrics=metrics_by(d, s.astype(np.float64), GA),
                   members={Path(p).name: metrics_by(d, Z[j].astype(np.float64), GA) for j, p in enumerate(paths)} if len(paths) > 1 else None)
        if ref is not None:
            pos = {i: k for k, i in enumerate(ref['id'].astype(str))}
            common = [(k, pos[i]) for k, i in enumerate(d['id'].astype(str)) if i in pos]
            if common:
                a_, b_ = np.array([c[0] for c in common]), np.array([c[1] for c in common])
                blk['check_logits'] = dict(file=a.check_logits, common_rows=len(common), max_abs_diff_ensemble=float(np.abs(s[a_] - ref['ensemble_logit'][b_]).max()),
                                           max_abs_diff_members=float(np.abs(Z[:, a_] - ref['member_logits'][:, b_]).max()) if ref['member_logits'].shape[0] == len(paths) else None)
            else:
                blk['check_logits'] = dict(file=a.check_logits, common_rows=0)
        res['files'][str(f)] = blk
        if a.save_logits:
            saved[str(f)] = dict(id=d['id'], ensemble_logit=s, member_logits=Z)
        print(f'{f}: {len(idx)} rows, {blk["groups"]} groups', flush=True)
        for ar, m in blk['metrics'].items():
            for reg, v in m.items():
                print(f"  {ar:8s} {reg:11s} n {v['n']:6d} W_unsafe {v['W_unsafe']:.4f} W_fail {v['W_fail']:.4f} G_unsafe {v['G_unsafe']:.4f} "
                      f"pick_fail {v['pick_fail']:.3f} (random {v['random_fail']:.3f}, oracle {v['oracle_fail']:.3f}) rate_unsafe {v['rate_unsafe']:.3f}", flush=True)
        if 'check_logits' in blk:
            print('  check vs trainer logits:', blk['check_logits'], flush=True)
    res['wall_s'] = round(time.time() - t0, 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    if a.save_logits:
        np.savez_compressed(str(a.out).rsplit('.', 1)[0] + '.logits.npz',
                            **{f'{k}_{i}': v for i, (f, dd) in enumerate(saved.items()) for k, v in dd.items()}, files=np.array(list(saved), object))
    return res


if __name__ == '__main__':
    main()
