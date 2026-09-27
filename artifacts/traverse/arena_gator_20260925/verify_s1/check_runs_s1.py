#!/usr/bin/env python3
"""VERIFY_S1: every soil stage-1 training run's arguments (trainer summary args + member records) against the declared
soil recipe and the job lists; fitted rows against my recomputed subsets (recompute_s1.json); exit status from the logs.
  python check_runs_s1.py <K3> <recompute_s1.json> <out json>"""
import json, glob, os, sys
K3 = sys.argv[1]; RS = json.load(open(sys.argv[2]))['designs']; OUT = sys.argv[3]
G3 = '/work1/dannegrut/harry/experiments/arena_gator_20260925'
RECIPE = dict(arch='gru', cond='none', domain_filter='crm', ctx='geom', split_eval='val', bs=256, epochs=30, seeds=5, lr=None, wd=None,
              data_frac_rigid=1.0, data_frac_crm=1.0, subsample=None, startup_only=False, row_weight='', keep_all_rows=False)
S = f'{G3}/e4/soil_s1/subsets'
RUNS = {  # tag: (subset stem, mode, seed0, job, summary json (local))
    'M1a_soil_deploy': ('M1_f104_hmmwv_soil', 'deploy', 0, 436354), 'M1b_soil_deploy': ('M1_f104_hmmwv_soil', 'deploy', 5, 436354),
    'M2_soil_deploy': ('M2_hmmwv_soil', 'deploy', 0, 436354), 'M3a_soil_deploy': ('M3_hmmwv_soil', 'deploy', 0, 436354),
    'M3b_soil_deploy': ('M3_hmmwv_soil', 'deploy', 5, 436354), 'A3_soil_deploy': ('A3_hmmwv_soil', 'deploy', 0, 436354),
    'M1_soil_holdout': ('M1_f104_hmmwv_soil', 'holdout', 0, 436354), 'M2_soil_holdout': ('M2_hmmwv_soil', 'holdout', 0, 436354),
    'M3_soil_holdout': ('M3_hmmwv_soil', 'holdout', 0, 436354), 'A3_soil_holdout': ('A3_hmmwv_soil', 'holdout', 0, 436354),
    'LC545_soil_holdout': ('LC545_f104_hmmwv_soil', 'holdout', 0, 436354), 'LC272_soil_holdout': ('LC272_f104_hmmwv_soil', 'holdout', 0, 436354),
    'LOAO1_g203_soil_holdout': ('LOAO1_g203_hmmwv_soil', 'holdout', 0, 436354),
    'LOAO2_f104_g203_soil_holdout': ('LOAO2_f104_g203_hmmwv_soil', 'holdout', 0, 436354),
    'LOAO2_f104_g228_soil_holdout': ('LOAO2_f104_g228_hmmwv_soil', 'holdout', 0, 436354),
    'LOAO2_g203_g228_soil_holdout': ('LOAO2_g203_g228_hmmwv_soil', 'holdout', 0, 436354),
    'LOAO1_g228_soil_holdout': ('LOAO1_g228_hmmwv_soil', 'holdout', 0, 436353),
    'G_soil_deploy': ('G_f104_gator_soil', 'deploy', 0, 436353), 'G_soil_holdout': ('G_f104_gator_soil', 'holdout', 0, 436353)}
jl = {}
for f in glob.glob(f'{K3}/e5/jobs/soil_s1_*.tsv'):
    for l in open(f):
        if l.startswith('#') or not l.strip():
            continue
        p = l.rstrip('\n').split('\t')
        jl.setdefault(p[1], []).append((os.path.basename(f), p[2], p[3]))
res, bad = {}, []
for tag, (stem, mode, seed0, job) in RUNS.items():
    js = sorted(set(glob.glob(f'{K3}/e5/train/soil_s1/*/{tag}.json') + glob.glob(f'{K3}/e5/deploy/*_soil/{tag}.json')))
    js = [p for p in js if '/deploy/' in p] or js   # G deploy lives only in the deploy copy
    j = json.load(open(js[0])); a = j['args']
    probs = []
    for k, v in RECIPE.items():
        if a.get(k) != v: probs.append(f'{k}={a.get(k)!r} (want {v!r})')
    if a['ds'] != [f'{S}/{stem}.npz']: probs.append(f'ds {a["ds"]}')
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
        if m['n_fit'] != j['n_fit']: probs.append(f'member {m["seed"]} n_fit {m["n_fit"]}')
        if mode == 'deploy' and (m.get('roundtrip') or {}).get('max_abs_diff') != 0.0: probs.append(f'member {m["seed"]} roundtrip {m.get("roundtrip")}')
    want_fit = RS[stem]['fit_deploy' if mode == 'deploy' else 'fit_holdout']
    if j['n_fit'] != want_fit: probs.append(f'n_fit {j["n_fit"]} != recomputed {want_fit}')
    if j['fit_by_domain'].get('rigid', 0) != 0: probs.append('rigid rows fitted')
    if j['files'][0]['rows'] != RS[stem]['rows']: probs.append(f'file rows {j["files"][0]["rows"]} != {RS[stem]["rows"]}')
    lines = jl.get(tag, [])
    want_args = f'--ds {S}/{stem}.npz --mode {mode} --arch gru --cond none --domain-filter crm --ctx geom --split-eval val --bs 256 --epochs 30 --seeds 5 --seed0 {seed0}' + (' --roundtrip-check' if mode == 'deploy' else '')
    if not any(x[2] == want_args for x in lines): probs.append(f'job list line differs: {lines}')
    log = f'{K3}/e5/logs/{tag}_{job}.log'
    txt = open(log).read() if os.path.exists(log) else ''
    last = txt.strip().splitlines()[-1] if txt else 'NO LOG'
    if last != 'exit: 0': probs.append(f'log {log}: {last}')
    res[tag] = dict(json=os.path.relpath(js[0], K3), job=job, n_fit=j['n_fit'], recomputed_fit=want_fit, file_rows=j['files'][0]['rows'], seeds=seeds,
                    roundtrip=[(m.get('roundtrip') or {}).get('max_abs_diff') for m in j['members']] if mode == 'deploy' else None,
                    sec_per_step=[m.get('sec_per_step') for m in j['members']],
                    job_lists=[x[0] for x in lines], log_last=last, problems=probs)
    if probs: bad.append(tag)
    print(f'{tag:30s} job {job} n_fit {j["n_fit"]:6d} (mine {want_fit:6d}) seeds {seeds} {[x[0] for x in lines]} {last} {"OK" if not probs else probs}')
json.dump(dict(runs=res, bad=bad), open(OUT, 'w'), indent=1)
print('runs with problems:', bad)
