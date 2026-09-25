#!/usr/bin/env python3
"""arena_gator_20260925 module E4: training subsets of the per-arena ci files (scripts/ag_build_ds.py output).

Selection rule (declared, fixed, no seed): within each arena, the training-split groups that still have rows after the
world / tier / id filters are ranked by md5(group id) (hex digest, ascending) and the first N are kept. The ranking
depends only on the group name, so the sets are nested (the 272 f104 groups are inside the 363, inside the 545, ...)
and identical in both worlds whenever the same groups are available. Val and test groups are never subsampled: all
their rows of the chosen world are kept (--eval-rows intact, default) or only the rows passing the same tier / id
filters (--eval-rows filtered).

  --groups 'f104=363,g203=363,g228=363'   group count per arena ('all' = every available training group)
  --preset NAME                           named designs of PLAN 2.1 / 7.5 / 7.6 (table PRESETS below)
  --tiers A-B                             keep rows of tiers A..B only (the matched tier range of a design)
  --ids-file F [--ids-apply train|all]    keep only the episodes listed in F (text, one id per line, or a json list);
                                          a leading 'gator__' is stripped, so a list of validated Gator ids selects
                                          the HMMWV twins (task B's H set). Default: applied to every row.
  --keep-dev-fold                         also keep every training group of ga_train's dev fold (md5 % 5 == 0) of each
                                          arena: files for holdout-mode runs (learning curve), where ci_train leaves
                                          the dev fold out of the fit and scores it; such a file must NOT be used in
                                          deploy mode (the manifest says so).
The written npz has every per-row array of the inputs (rows in input order) and the metadata arrays; a manifest
<out>.manifest.json lists the selected groups, per-arena row counts, the expected fitted rows in deploy and holdout
mode, input and output sha256. Hard error if a suite / blacklisted id or group is present, if a requested count
exceeds the available groups (unless --allow-short) or if arenas / worlds are missing.

  PYTHONPATH=src:scripts python scripts/ag_subset.py --preset M1 --world rigid \
      --ds <e4>/f104_hmmwv/ci_f104_hmmwv_both.npz --out <e4>/subsets/M1_rigid.npz
numpy only.
"""
import argparse, hashlib, json, os, sys, time
from collections import Counter
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ag_build_ds as B                  # noqa: E402  (suite patterns, sha256)

PRESETS = {
    # task A (PLAN 2.1): matched designs and the additive one
    'M1': dict(groups={'f104': 1089}),
    'M2': dict(groups={'f104': 545, 'g203': 545}),
    'M3': dict(groups={'f104': 363, 'g203': 363, 'g228': 363}),
    'A3': dict(groups={'f104': 'all', 'g203': 'all', 'g228': 'all'}),
    # leave one training arena out at a matched 545 groups (PLAN 7.6, REVIEW_R1 5a); 2-arena sets: first arena 273
    'LOAO1_f104': dict(groups={'f104': 545}), 'LOAO1_g203': dict(groups={'g203': 545}), 'LOAO1_g228': dict(groups={'g228': 545}),
    'LOAO2_f104_g203': dict(groups={'f104': 273, 'g203': 272}), 'LOAO2_f104_g228': dict(groups={'f104': 273, 'g228': 272}),
    'LOAO2_g203_g228': dict(groups={'g203': 273, 'g228': 272}),
    # f104 learning curve (PLAN 7.6, REVIEW_R1 5b), holdout-mode files (dev fold kept for scoring)
    'LC272': dict(groups={'f104': 272}, keep_dev_fold=True), 'LC545': dict(groups={'f104': 545}, keep_dev_fold=True),
    'LC1089': dict(groups={'f104': 1089}, keep_dev_fold=True),
    # task B: the HMMWV planner H on exactly the ids the Gator validated (needs --ids-file)
    'H': dict(groups={'f104': 'all'}, needs_ids=True),
}
LIGHT = ('id', 'group', 'split', 'domain', 'arena', 'vehicle', 'tier', 'episode', 'anchor_frame')
META = ('hist_cols', 'priv_names')


def md5hex(s):
    return hashlib.md5(s.encode()).hexdigest()


def dev_fold(g):
    """ga_train.dev_group (imported there with torch; same formula)."""
    return int(md5hex(g), 16) % 5 == 0


def parse_groups(s):
    out = {}
    for kv in s.split(','):
        k, v = kv.split('=')
        out[k.strip()] = 'all' if v.strip() == 'all' else int(v)
    return out


def load_ids(p, strip='gator__'):
    txt = open(p).read()
    ids = json.loads(txt) if p.endswith('.json') else [l.strip() for l in txt.splitlines() if l.strip()]
    ids = [i if isinstance(i, str) else i['id'] for i in ids]
    return {i[len(strip):] if strip and i.startswith(strip) else i for i in ids}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ds', nargs='+', required=True, help='ci files (ag_build_ds.py); one or more per arena')
    ap.add_argument('--world', choices=['rigid', 'crm'], required=True)
    ap.add_argument('--preset', choices=sorted(PRESETS)); ap.add_argument('--groups', default=None)
    ap.add_argument('--tiers', default=None); ap.add_argument('--ids-file', default=None); ap.add_argument('--ids-apply', choices=['train', 'all'], default='all')
    ap.add_argument('--ids-strip-prefix', default='gator__')
    ap.add_argument('--vehicle', default=None, help='keep rows of this vehicle only (default: all in the inputs, must be one)')
    ap.add_argument('--eval-rows', choices=['intact', 'filtered'], default='intact')
    ap.add_argument('--keep-dev-fold', action='store_true'); ap.add_argument('--allow-short', action='store_true')
    ap.add_argument('--out', required=True); ap.add_argument('--no-compress', action='store_true'); ap.add_argument('--list-only', action='store_true')
    a = ap.parse_args(argv)
    t0 = time.time()
    P = PRESETS.get(a.preset, {})
    want = parse_groups(a.groups) if a.groups else dict(P.get('groups', {}))
    assert want, 'give --preset or --groups'
    keep_dev = a.keep_dev_fold or P.get('keep_dev_fold', False)
    if P.get('needs_ids'):
        assert a.ids_file, f'preset {a.preset} needs --ids-file'
    ids_keep = load_ids(a.ids_file, a.ids_strip_prefix) if a.ids_file else None
    tiers = B.parse_tiers(a.tiers); code = B.GBM.DOMAIN_CODE[a.world]

    # ---- light columns of every file
    files = []
    for p in a.ds:
        z = np.load(p, allow_pickle=True)
        miss = [k for k in LIGHT if k not in z.files]
        assert not miss, f'{p}: missing {miss} (not an ag_build_ds.py file?)'
        files.append(dict(path=p, z=z, **{k: z[k] for k in LIGHT}))
    veh = sorted({v for f in files for v in f['vehicle'][f['domain'].astype(int) == code].astype(str)})
    if a.vehicle:
        veh = [a.vehicle]
    assert len(veh) == 1, f'rows of several vehicles {veh}: pass --vehicle'
    # ---- candidate groups per arena
    cand, masks = {}, []
    for f in files:
        dom = f['domain'].astype(int); ar = f['arena'].astype(str); sp = f['split'].astype(str); grp = f['group'].astype(str)
        base = (dom == code) & (f['vehicle'].astype(str) == veh[0])
        tm = np.ones(len(dom), bool); idm = np.ones(len(dom), bool)
        if tiers:
            t = f['tier'].astype(int); tm = (t >= tiers[0]) & (t <= tiers[1])
        if ids_keep is not None:
            ep = f['episode'].astype(str)
            strip = [e[len(a.ids_strip_prefix):] if a.ids_strip_prefix and e.startswith(a.ids_strip_prefix) else e for e in ep]
            idm = np.array([e in ids_keep for e in strip])
        filt = base & tm & idm
        f.update(base=base, filt=filt, tm=tm, idm=idm, ar=ar, sp=sp, grp=grp)
        for arena in set(ar[filt & (sp == 'train')]):
            cand.setdefault(arena, set()).update(grp[filt & (sp == 'train') & (ar == arena)])
    missing = [k for k in want if k not in cand]
    assert not missing, f'no training rows for arenas {missing} in the inputs (have {sorted(cand)})'
    sel, per_arena = {}, {}
    for arena, n in want.items():
        ranked = sorted(cand[arena], key=md5hex)
        k = len(ranked) if n == 'all' else n
        if k > len(ranked):
            assert a.allow_short, f'{arena}: {n} groups requested, {len(ranked)} available (--allow-short to take all)'
            k = len(ranked)
        sel[arena] = set(ranked[:k])
        per_arena[arena] = dict(requested=n, available=len(ranked), selected=k, md5_last=md5hex(ranked[k - 1]) if k else None)
    dev_keep = {arena: {g for g in cand[arena] if dev_fold(g)} for arena in want} if keep_dev else {}
    # ---- rows
    for f in files:
        in_arena = np.isin(f['ar'], list(want))
        tr = f['filt'] & (f['sp'] == 'train') & in_arena
        g_ok = np.array([g in sel.get(ar, ()) or g in dev_keep.get(ar, ()) for g, ar in zip(f['grp'], f['ar'])])
        ev_src = f['base'].copy()                  # intact: every row of the world / vehicle
        if a.eval_rows == 'filtered':
            ev_src &= f['tm']
        if ids_keep is not None and a.ids_apply == 'all':
            ev_src &= f['idm']
        ev = ev_src & np.isin(f['sp'], ['val', 'test']) & in_arena
        f['keep'] = (tr & g_ok) | ev
        masks.append(f['keep'])
    n_out = int(sum(m.sum() for m in masks))
    assert n_out > 0
    # ---- checks
    allsel = lambda k: np.concatenate([f[k][f['keep']].astype(str) for f in files])
    ids, grp, ep, sp = allsel('id'), allsel('group'), allsel('episode'), allsel('split')
    ar, af = allsel('arena'), np.concatenate([f['anchor_frame'][f['keep']].astype(int) for f in files])
    assert len(set(ids)) == len(ids), 'duplicate ids across inputs'
    bad = sorted({s for s in set(ids) | set(grp) | set(ep) if B.suite_hit(s.split('@')[0])})
    assert not bad, f'suite / blacklisted ids: {bad[:5]}'
    trg = {arena: sorted(set(grp[(sp == 'train') & (ar == arena)])) for arena in want}
    for arena in want:
        assert set(trg[arena]) == sel[arena] | dev_keep.get(arena, set()), arena
    fit_holdout = np.array([(s == 'train') and not dev_fold(g) for s, g in zip(sp, grp)])
    man = dict(tool='scripts/ag_subset.py', tool_sha256=B.sha256_file(__file__), argv=sys.argv[1:] if argv is None else argv,
               preset=a.preset, world=a.world, vehicle=veh[0], rule='md5(group id) hex ascending within each arena, first N of the available training groups',
               tiers=a.tiers, ids_file=a.ids_file, ids_file_sha256=B.sha256_file(a.ids_file) if a.ids_file else None, n_ids=len(ids_keep) if ids_keep is not None else None,
               ids_apply=a.ids_apply if a.ids_file else None, eval_rows=a.eval_rows, keep_dev_fold=keep_dev,
               use='holdout mode only (the dev fold is kept for scoring; deploy mode would fit it)' if keep_dev else 'deploy or holdout',
               per_arena={arena: dict(per_arena[arena], dev_fold_groups_kept=len(dev_keep.get(arena, ())),
                                      train_groups_in_file=len(trg[arena]),
                                      train_rows=int(((sp == 'train') & (ar == arena)).sum()),
                                      train_rows_startup=int(((sp == 'train') & (ar == arena) & (af == 0)).sum()),
                                      train_episodes=int(len(set(ep[(sp == 'train') & (ar == arena)]))),
                                      val_groups=int(len(set(grp[(sp == 'val') & (ar == arena)]))), val_rows=int(((sp == 'val') & (ar == arena)).sum()),
                                      test_groups=int(len(set(grp[(sp == 'test') & (ar == arena)]))), test_rows=int(((sp == 'test') & (ar == arena)).sum()),
                                      fit_rows_holdout=int((fit_holdout & (ar == arena)).sum()),
                                      fit_groups_holdout=int(len(set(grp[fit_holdout & (ar == arena)]))))
                          for arena in want},
               rows=n_out, fit_rows_deploy=int((sp == 'train').sum()), fit_rows_holdout=int(fit_holdout.sum()),
               eval_rows_val=int((sp == 'val').sum()), inputs=[dict(path=os.path.abspath(f['path']), rows=int(len(f['id'])), rows_kept=int(f['keep'].sum())) for f in files],
               selected_groups={arena: sorted(sel[arena], key=md5hex) for arena in want})
    print(json.dumps({k: v for k, v in man.items() if k not in ('selected_groups', 'inputs')}, indent=1), flush=True)
    if a.list_only:
        return man
    # ---- write
    common = sorted(set.intersection(*[set(f['z'].files) for f in files]))
    dropped = sorted(set.union(*[set(f['z'].files) for f in files]) - set(common))
    out = {}
    for k in common:
        if k in META:
            vals = [np.asarray(f['z'][k]) for f in files]
            assert all(v.shape == vals[0].shape and (v == vals[0]).all() for v in vals[1:]), f'{k} differs between inputs'
            out[k] = vals[0]; continue
        parts = []
        for f in files:
            v = f['z'][k]
            assert v.shape[0] == len(f['id']), (f['path'], k, v.shape)
            parts.append(v[f['keep']]); del v
        out[k] = np.concatenate(parts) if len(parts) > 1 else parts[0]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    (np.savez if a.no_compress else np.savez_compressed)(a.out, **out)
    man.update(inputs=[dict(i, sha256=B.sha256_file(i['path'])) for i in man['inputs']], dropped_keys=dropped, keys=common,
               output=dict(path=os.path.abspath(a.out), sha256=B.sha256_file(a.out), size_gb=round(os.path.getsize(a.out) / 1e9, 3)),
               elapsed_s=round(time.time() - t0, 1))
    json.dump(man, open(os.path.splitext(a.out)[0] + '.manifest.json', 'w'), indent=1)
    print(f'wrote {a.out} ({n_out} rows, {man["output"]["size_gb"]} GB) in {man["elapsed_s"]} s', flush=True)
    return man


if __name__ == '__main__':
    main()
