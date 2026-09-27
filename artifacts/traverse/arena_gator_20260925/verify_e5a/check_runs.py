#!/usr/bin/env python3
"""VERIFY_E5a: every rigid training run's arguments (trainer summary json args + member records) against the declared
recipe and the job lists; fitted rows against my recomputed subsets; exit status from the run logs."""
import json, glob, os, sys, re
K3 = sys.argv[1]; RS = json.load(open(sys.argv[2]))['designs']; OUT = sys.argv[3]
G3 = '/work1/dannegrut/harry/experiments/arena_gator_20260925'
RECIPE = dict(arch='gru', cond='none', domain_filter='rigid', ctx='geom', split_eval='val', bs=256, epochs=30, seeds=5, lr=None, wd=None,
              data_frac_rigid=1.0, data_frac_crm=1.0, subsample=None, startup_only=False, row_weight='', keep_all_rows=False)
RUNS = {  # tag: (subset stem, mode, seed0, job)
    'M1a_rigid_deploy': ('M1_f104_hmmwv_rigid', 'deploy', 0, 436135), 'M1b_rigid_deploy': ('M1_f104_hmmwv_rigid', 'deploy', 5, 436135),
    'M1_rigid_holdout': ('M1_f104_hmmwv_rigid', 'holdout', 0, 436135), 'LC545_rigid_holdout': ('LC545_f104_hmmwv_rigid', 'holdout', 0, 436135),
    'LC272_rigid_holdout': ('LC272_f104_hmmwv_rigid', 'holdout', 0, 436135),
    'M2_rigid_deploy': ('M2_hmmwv_rigid', 'deploy', 0, 436234), 'M2_rigid_holdout': ('M2_hmmwv_rigid', 'holdout', 0, 436235),
    'LOAO1_g203_rigid_holdout': ('LOAO1_g203_hmmwv_rigid', 'holdout', 0, 436234), 'LOAO2_f104_g203_rigid_holdout': ('LOAO2_f104_g203_hmmwv_rigid', 'holdout', 0, 436234),
    'M3a_rigid_deploy': ('M3_hmmwv_rigid', 'deploy', 0, 436352), 'M3b_rigid_deploy': ('M3_hmmwv_rigid', 'deploy', 5, 436352),
    'M3_rigid_holdout': ('M3_hmmwv_rigid', 'holdout', 0, 436352), 'A3_rigid_deploy': ('A3_hmmwv_rigid', 'deploy', 0, 436352),
    'A3_rigid_holdout': ('A3_hmmwv_rigid', 'holdout', 0, 436352), 'G_rigid_deploy': ('G_f104_gator_rigid', 'deploy', 0, 436352),
    'G_rigid_holdout': ('G_f104_gator_rigid', 'holdout', 0, 436352), 'LOAO1_g228_rigid_holdout': ('LOAO1_g228_hmmwv_rigid', 'holdout', 0, 436352),
    'LOAO2_f104_g228_rigid_holdout': ('LOAO2_f104_g228_hmmwv_rigid', 'holdout', 0, 436352),
    'LOAO2_g203_g228_rigid_holdout': ('LOAO2_g203_g228_hmmwv_rigid', 'holdout', 0, 436352)}
jl = {}
for f in glob.glob(f'{K3}/e5/jobs/*.tsv'):
    for l in open(f):
        if l.startswith('#') or not l.strip():
            continue
        p = l.rstrip('\n').split('\t')
        tag, args = (p[1], p[3]) if p[0].isdigit() else (p[0], p[2])
        jl.setdefault(tag, []).append((os.path.basename(f), args))
res, bad = {}, []
for tag, (stem, mode, seed0, job) in RUNS.items():
    js = glob.glob(f'{K3}/e5/train/*/{tag}.json')
    assert len(js) == 1, (tag, js)
    j = json.load(open(js[0])); a = j['args']
    probs = []
    for k, v in RECIPE.items():
        if a.get(k) != v:
            probs.append(f'{k}={a.get(k)!r} (want {v!r})')
    if a['ds'] != [f'{G3}/e4/subsets/{stem}.npz']: probs.append(f'ds {a["ds"]}')
    if a['mode'] != mode: probs.append(f'mode {a["mode"]}')
    if a['seed0'] != seed0: probs.append(f'seed0 {a["seed0"]}')
    if a['tag'] != tag: probs.append('tag')
    if mode == 'deploy' and not a['roundtrip_check']: probs.append('no roundtrip check')
    seeds = [m['seed'] for m in j['members']]
    if seeds != list(range(seed0, seed0 + 5)): probs.append(f'member seeds {seeds}')
    for m in j['members']:
        for k in ('arch', 'cond', 'domain_filter', 'ctx', 'mode', 'epochs'):
            if m[k] != (a[k] if k != 'mode' else mode): probs.append(f'member {m["seed"]} {k}')
        if (m['lr'], m['wd']) != (0.002, 0.0001): probs.append(f'member lr/wd {m["lr"]}/{m["wd"]}')
        if mode == 'deploy' and (m.get('roundtrip') or {}).get('max_abs_diff') != 0.0: probs.append(f'member {m["seed"]} roundtrip {m.get("roundtrip")}')
    want_fit = RS[stem]['fit_deploy' if mode == 'deploy' else 'fit_holdout']
    if j['n_fit'] != want_fit: probs.append(f'n_fit {j["n_fit"]} != recomputed {want_fit}')
    if j['fit_by_domain'].get('crm', 0) != 0: probs.append('crm rows fitted')
    # the job list line
    lines = [x for x in jl.get(tag, []) if 'smoke' not in x[0]]
    want_args = f'--ds {G3}/e4/subsets/{stem}.npz --mode {mode} --arch gru --cond none --domain-filter rigid --ctx geom --split-eval val --bs 256 --epochs 30 --seeds 5 --seed0 {seed0}' + (' --roundtrip-check' if mode == 'deploy' else '')
    if not any(x[1] == want_args for x in lines): probs.append(f'job list line differs: {lines}')
    log = f'{K3}/e5/logs/{tag}_{job}.log'
    last = open(log).read().strip().splitlines()[-1] if os.path.exists(log) else 'NO LOG'
    if last != 'exit: 0': probs.append(f'log {log}: {last}')
    res[tag] = dict(json=os.path.relpath(js[0], K3), job=job, n_fit=j['n_fit'], recomputed_fit=want_fit, seeds=seeds,
                    ens=dict(startup=j['ensemble'].get('heldout', {}).get('rigid', {}).get('startup', {}).get('W_unsafe') if isinstance(j['ensemble'].get('heldout'), dict) else None),
                    job_lists=[x[0] for x in lines], log_last=last, problems=probs)
    if probs: bad.append(tag)
    print(f'{tag:32s} job {job} n_fit {j["n_fit"]:7d} (recomputed {want_fit:7d}) seeds {seeds} lists {[x[0] for x in lines]} {last} {"OK" if not probs else probs}')
json.dump(dict(runs=res, bad=bad), open(OUT, 'w'), indent=1)
print('runs with problems:', bad)
