"""Night-2 planner study, arms G/H: gradient refinement of the deployed 256-pool pick (offline picks, hashed before drives).

Per group: rebuild the deployed proposal pool exactly as crm_pools.build (same md5 tag; float16-rounded corridors so the
pool argmin equals eval_v1's), take the top-k by ensemble-mean logit plus the argmin's lateral mirror as starts, run the
batched Adam refinement of f104_n2_grad (arm G: objective 'logit'; arm H: expected cost C_fail*P + T + lambda_E*E with
the analytic energy refit on f104 CRM) together with the leave-one-member-out guard rows, re-shape the finals in float64,
validate, re-score them through the deployed numpy pipeline, and pick by the pessimistic objective with abstention.

  python scripts/planner_grad_arms.py --cases artifacts/traverse/crm_f104_v1/cases_eval/cases \\
      --map-root artifacts/traverse/crm_f104_v1/map_root --models 'artifacts/traverse/crm_f104_v1/train_v1/deploy/CRM_N2_s*.pt' \\
      --out artifacts/traverse/crm_night2_v1/planner/grad_crm --arms G,H

Outputs: <out>/picks/<group>.json, <out>/routes/<route_id>.json (collector format), <out>/tasks.json, <out>/audit.json,
<out>/energy_fit.json, <out>/config.json. Picks identical to the pool argmin reuse eval_v1's route id (<group>__crm) and
are listed with run=false (their drives exist in crm_f104_v1/eval_v1/runs).
"""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import f104_n2_grad as G                      # noqa: E402
import crm_pools as CP                        # noqa: E402
from f104_n2_grad import DS, P, ROOT          # noqa: E402

ARMS = {'G': dict(objective='logit', C_fail=0.0, lam_E=0.0, abstain=0.3, unit='logit'),
        'H': dict(objective='expected_cost_energy', C_fail=120.0, lam_E=1.0, abstain=1.0, unit='s'),
        'H0': dict(objective='expected_cost', C_fail=120.0, lam_E=0.0, abstain=1.0, unit='s')}
P_BINS = [(0, .01), (.01, .05), (.05, .2), (.2, 1.01)]


def p_of(z):
    return 1 - np.exp(-np.exp(np.asarray(z, float)))


def keep_objective(arm, z_pess, T, E):
    a = ARMS[arm]
    if a['objective'] == 'logit':
        return float(z_pess)
    return float(a['C_fail'] * p_of(z_pess) + T + a['lam_E'] * G.KJ_TO_S * E)


def np_time(route):
    v = np.asarray(route['speeds'], float); st = np.asarray(route['stations'], float)
    return float((np.diff(st) / np.maximum(0.5 * (v[1:] + v[:-1]), 0.25)).sum())


def np_energy(X, L, model, device):
    with torch.no_grad():
        return G.energy_kj(torch.tensor(np.asarray(X, np.float32), device=device), torch.tensor(np.asarray(L, np.float32), device=device), model).cpu().numpy()


def route_stats(route, X, L, Zc, Zr, model, device):
    """Deployed-pipeline scores and descriptors of one route (numpy corridor, float16-rounded, RiskModel.score)."""
    z = Zc.mean(0); zp = Zc.max(0)
    return dict(z_mean=float(z), z_pess=float(zp), P=float(p_of(z)), P_pess=float(p_of(zp)), z_members=[float(v) for v in Zc],
                z_rigid=float(Zr.mean(0)), T=np_time(route), E_analytic=float(np_energy(X, L, model, device)[0]),
                mean_speed=float(np.asarray(route['speeds'])[1:-1].mean()), length_m=float(np.asarray(route['stations'])[-1]),
                max_lateral_m=float(route.get('meta', {}).get('max_lateral_m', 0.0)))


def pair_sensitivity(X16, z, sd, k=32):
    """Baseline for the local-sensitivity guard: |dz| per unit normalised-corridor RMS between pool candidates (top-k pairs)."""
    o = np.argsort(z)[:k]; Xn = X16[o][:, :4] / sd[None, :, None, None]; zz = z[o]
    vals = []
    for i in range(len(o)):
        for j in range(i + 1, len(o)):
            rms = float(np.sqrt(((Xn[i] - Xn[j]) ** 2).mean()))
            if rms > 1e-6:
                vals.append(abs(zz[i] - zz[j]) / rms)
    return float(np.median(vals)) if vals else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', required=True); ap.add_argument('--map-root', required=True); ap.add_argument('--models', required=True)
    ap.add_argument('--out', required=True); ap.add_argument('--arms', default='G,H')
    ap.add_argument('--pool-tag', default='crm_proposal'); ap.add_argument('--topk', type=int, default=16)
    ap.add_argument('--rigid-models', default=G.RIGID_MODELS)
    ap.add_argument('--twin', default=str(ROOT / 'artifacts/traverse/crm_night2_v1/datasets/twin_crm.npz'), help='CRM twin dataset for the energy refit')
    ap.add_argument('--eval-picks', default=str(ROOT / 'artifacts/traverse/crm_f104_v1/eval_v1/picks'), help='locked picks: the pool argmin must match')
    ap.add_argument('--eval-results', default=str(ROOT / 'artifacts/traverse/crm_f104_v1/eval_v1/results.json'))
    ap.add_argument('--steps', type=int, default=100); ap.add_argument('--lr-a', type=float, default=0.02); ap.add_argument('--lr-dv', type=float, default=0.10)
    ap.add_argument('--patience', type=int, default=15); ap.add_argument('--trust', type=float, default=0.0)
    ap.add_argument('--lomo-starts', type=int, default=17, help='how many of the starts also run the 5 leave-one-member-out folds')
    ap.add_argument('--case-prefix', default='cases_eval/cases'); ap.add_argument('--route-prefix', default='night2_v1/planner/grad_crm/routes')
    ap.add_argument('--reuse-route-prefix', default='eval_v1/routes')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--device', default='cuda'); ap.add_argument('--verbose', action='store_true')
    a = ap.parse_args()
    arms = a.arms.split(',')
    out = Path(a.out); (out / 'picks').mkdir(parents=True, exist_ok=True); (out / 'routes').mkdir(exist_ok=True)
    dev = a.device
    DS.init_map(a.map_root); CP._S['P'] = P
    crm_rm = P.RiskModel(a.models, device=dev); rig_rm = P.RiskModel(a.rigid_models, device=dev)
    ens = G.Ensemble(a.models, dev, torch.float32, risk_model=crm_rm); rig = G.Ensemble(a.rigid_models, dev, torch.float32, risk_model=rig_rm)
    tmap = G.TMap(DS.G, dev, torch.float32)
    sd_norm = np.stack([m[2].cpu().numpy() for m in ens.members]).mean(0)
    t0 = time.time()
    energy = G.fit_energy_model(a.twin, dev) if a.twin else None
    if energy is not None:
        json.dump(energy, open(out / 'energy_fit.json', 'w'), indent=1)
        print(f'energy refit ({time.time() - t0:.0f}s): pool {energy["pool"]}, held-out MAE {energy["stats"]["heldout_mae_kj"]:.1f} kJ, '
              f'Spearman {energy["stats"]["heldout_spearman"]:.3f}; coefficients {energy["coefficients_nonzero"]}', flush=True)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except Exception:
        commit = None
    json.dump({**vars(a), 'arms_spec': {k: ARMS[k] for k in arms}, 'commit': commit, 'started': time.strftime('%Y-%m-%d %H:%M:%S'),
               'selection_rule': 'candidates = float64-validated finals of the full-ensemble rows; choose by the pessimistic objective '
                                 '(max member); abstain (keep the pool argmin) if the pessimistic gain over the pool argmin is below '
                                 'ARMS[arm].abstain (0.3 logit for G, 1.0 s of expected cost for H)'},
              open(out / 'config.json', 'w'), indent=1)
    try:
        results = json.load(open(a.eval_results))['per_group']
    except Exception:
        results = {}
    cases = sorted(str(p) for p in Path(a.cases).glob('*.json') if p.name != 'cases.json')
    if a.limit:
        cases = cases[:a.limit]
    tasks, audit_rows, route_ids = [], [], {}
    timing = []
    for gi, case_path in enumerate(cases):
        tc = time.time()
        case = json.load(open(case_path)); g = case['id']; lay = case['layout']
        pose = np.array([lay['start_xy'][0], lay['start_xy'][1], lay['start_yaw']], float); goal = np.asarray(case['goal_xy'], float)
        base = {k: np.asarray(v, float) for k, v in json.load(open(Path(case_path).parent / 'routes' / g / 'route_00.json')).items()
                if k in ('waypoints', 'speeds', 'stations', 'headings')}
        base['meta'] = {}
        # ---- the deployed pool, bit-for-bit (mirror sampler with parameters; cross-checked against gen_planner.proposal_pool)
        seed = CP._seed(g, a.pool_tag)
        cands, tries, params = G.propose_with_params(base, pose, np.random.default_rng(seed))
        ref, _ = P.proposal_pool(base, pose, np.random.default_rng(seed))
        mirror_err = max(max(np.abs(c['waypoints'] - r['waypoints']).max(), np.abs(c['speeds'] - r['speeds']).max()) for c, r in zip(cands, ref)) if len(ref) == len(cands) else float('inf')
        X, L = P.corridors(cands); X16 = X.astype(np.float16).astype(np.float32); ctx = P.geom_ctx(lay['start_xy'], case['goal_xy'], lay['start_yaw'], L)
        Zc = CP.member_logits(crm_rm, X16, ctx); Zr = CP.member_logits(rig_rm, X16, ctx)
        z_mean = Zc.mean(0); z_pess = Zc.max(0); i0 = int(np.argmin(z_mean)); order = np.argsort(z_mean)
        locked, locked_rid = None, f'{g}__crm'
        try:
            lk = json.load(open(Path(a.eval_picks) / f'{g}.json'))['arms']['crm']; locked, locked_rid = lk['index'], lk['route_id']
        except Exception:
            pass   # eval_v1 names a shared route file after the first arm that picked it (e.g. <g>__straight6), so reuse rows need its id
        pool_pick = route_stats(cands[i0], X16[i0:i0 + 1], L[i0:i0 + 1], Zc[:, i0], Zr[:, i0], energy, dev)
        pool_pick.update(index=i0, index_locked=locked, route_id_locked=locked_rid, pool_hash=G.route_hash(cands[i0]), z_pess_argmin_index=int(np.argmin(z_pess)),
                         driven=results.get(g, {}).get('crm'))
        base_sens = pair_sensitivity(X16, z_mean, sd_norm)
        t_pool = time.time() - tc
        # ---- starts: top-k + the argmin's lateral mirror
        starts = [int(i) for i in order[:a.topk]] + [i0]
        s_a, s_dv, s_l = [], [], []
        for si, i in enumerate(starts):
            aa, dd, ll = params[i]; aa = aa.copy(); ll = ll.copy()
            if si == len(starts) - 1:
                aa[0] = -aa[0]; ll = -ll
            s_a.append(aa); s_dv.append(dd); s_l.append(ll)
        rows = dict(a0=[], dv0=[], lat0=[], w=[], obj=[], C_fail=[], lam_E=[], arm=[], fold=[], start=[])
        for arm in arms:
            spec = ARMS[arm]
            for fold in [-1, 0, 1, 2, 3, 4]:
                n_s = len(starts) if fold < 0 else min(a.lomo_starts, len(starts))
                for si in range(n_s):
                    ww = np.ones(ens.M); ww[fold] = 0 if fold >= 0 else 1
                    rows['a0'].append(s_a[si]); rows['dv0'].append(s_dv[si]); rows['lat0'].append(s_l[si]); rows['w'].append(ww)
                    rows['obj'].append(G.OBJ_ID[spec['objective']]); rows['C_fail'].append(spec['C_fail']); rows['lam_E'].append(spec['lam_E'])
                    rows['arm'].append(arm); rows['fold'].append(fold); rows['start'].append(si)
        rows = {k: np.array(v) for k, v in rows.items()}
        prob = G.Problem(base, pose, goal, tmap, dev, torch.float32)
        res = G.refine(prob, ens, rows, steps=a.steps, lr_a=a.lr_a, lr_dv=a.lr_dv, patience=a.patience, trust=a.trust,
                       energy_model=energy, rigid=rig, verbose=a.verbose)
        t_ref = time.time() - tc - t_pool
        # ---- per arm: float64 re-shape, validate, deployed-pipeline re-score, pick, guards
        picks = {}; summ = dict(group=g, n_pool=len(cands), tries=tries, pool_mirror_max_abs_diff=float(mirror_err), pool_pick=pool_pick,
                                starts=[dict(pool_index=int(i), z_mean=float(z_mean[i]), z_pess=float(z_pess[i])) for i in starts[:-1]] + [dict(pool_index=i0, mirror=True)],
                                steps_run=int(res['steps_run']), zero_grad_frac=res['zero_grad_frac'], base_pair_sensitivity=base_sens, arms={})
        w_all = rows['w']; Z0, Z1 = res['Z0'], res['Z']
        for arm in arms:
            spec = ARMS[arm]; full = np.flatnonzero((rows['arm'] == arm) & (rows['fold'] < 0)); lomo = np.flatnonzero((rows['arm'] == arm) & (rows['fold'] >= 0))
            finals = []
            for r in full:
                route = G.np_route(base, res['a'][r], res['dv'][r], rows['lat0'][r])
                chk = P.safe_validate(route, [], P.CFG, pose)
                fin = dict(row=int(r), start=int(rows['start'][r]), valid=bool(chk['valid']), reasons=chk.get('reasons', []),
                           kappa_max=float(chk.get('max_curvature', res['kappa_max'][r])), a=[float(x) for x in res['a'][r]], dv=[float(x) for x in res['dv'][r]],
                           z_fit_torch=float(res['z_fit'][r]), z_pess_torch=float(res['z_pess'][r]), best_step=int(res['best_step'][r]),
                           mean_speed=float(res['mean_v'][r]), max_lateral_m=float(res['max_lat'][r]), max_abs_pt=float(res['max_abs_pt'][r]))
                if chk['valid']:
                    Xf, Lf = P.corridors([route]); Xf16 = Xf.astype(np.float16).astype(np.float32); cf = P.geom_ctx(lay['start_xy'], case['goal_xy'], lay['start_yaw'], Lf)
                    st = route_stats(route, Xf16, Lf, CP.member_logits(crm_rm, Xf16, cf)[:, 0], CP.member_logits(rig_rm, Xf16, cf)[:, 0], energy, dev)
                    st['valid_frac'] = float(Xf16[0, 4].mean()); fin.update(st); fin['J_keep'] = keep_objective(arm, st['z_pess'], st['T'], st['E_analytic'])
                    fin['route'] = route
                finals.append(fin)
            valid = [f for f in finals if f['valid']]
            J_pool = keep_objective(arm, pool_pick['z_pess'], pool_pick['T'], pool_pick['E_analytic'])
            best = min(valid, key=lambda f: f['J_keep']) if valid else None
            gain = (J_pool - best['J_keep']) if best else float('nan')
            abstain = (best is None) or not (gain >= spec['abstain'])
            chosen = best if not abstain else None
            # guards
            r_best = best['row'] if best else int(full[0]); s_best = int(rows['start'][r_best])
            lomo_rows = lomo[rows['start'][lomo] == s_best]
            fit_gain = []; ho_gain = []
            for r in lomo_rows:
                ww = w_all[r]; k = int(rows['fold'][r])
                fit_gain.append(float((Z1[:, r] * ww).sum() / ww.sum() - (Z0[:, r] * ww).sum() / ww.sum())); ho_gain.append(float(Z1[k, r] - Z0[k, r]))
            all_fit, all_ho = [], []
            for r in lomo:
                ww = w_all[r]; k = int(rows['fold'][r])
                all_fit.append(float((Z1[:, r] * ww).sum() / ww.sum() - (Z0[:, r] * ww).sum() / ww.sum())); all_ho.append(float(Z1[k, r] - Z0[k, r]))
            fg, hg = (np.mean(fit_gain) if fit_gain else np.nan), (np.mean(ho_gain) if ho_gain else np.nan)
            # gains are logit changes (negative = safer). Flag when the held-out member disagrees with the optimised members by more
            # than 0.5 logit in the unsafe direction, or when a fit improvement does not transfer to the held-out member.
            lomo_flag = bool(fit_gain) and ((hg - fg > 0.5) or (fg < 0 and hg > 0.5 * fg))
            path_ratio = float(res['path_dz'][r_best] / max(res['path_dX'][r_best], 1e-9))
            sens_ratio = path_ratio / base_sens if base_sens and np.isfinite(base_sens) else float('nan')
            pk = chosen or {}
            d_pess = (pk['z_pess'] - pool_pick['z_pess']) if chosen else 0.0
            d_rigid = (pk['z_rigid'] - pool_pick['z_rigid']) if chosen else 0.0
            d_mean = (pk['z_mean'] - pool_pick['z_mean']) if chosen else 0.0
            flags = dict(lomo=lomo_flag, pess_worse=bool(chosen and d_pess > 0), rigid_disagrees=bool(chosen and d_mean < 0 and d_rigid > 0.5),
                         sensitivity=bool(np.isfinite(sens_ratio) and sens_ratio > 10), speed=bool(chosen and pk['mean_speed'] > 5.0),
                         dv_clip=bool(chosen and int(np.sum(np.abs(best['dv']) >= G.DV_CLIP - 1e-6)) >= 3),
                         lateral_clip=bool(chosen and pk['max_lateral_m'] >= G.LAT_CLIP - 0.1),
                         edge=bool(chosen and (best['max_abs_pt'] > 34.0 or pk.get('valid_frac', 1.0) < 1.0)))
            guards = dict(lomo_fit_gain_at_pick=fg, lomo_heldout_gain_at_pick=hg, lomo_heldout_worst_at_pick=(max(ho_gain) if ho_gain else np.nan),
                          lomo_fit_gain_all=float(np.mean(all_fit)) if all_fit else np.nan, lomo_heldout_gain_all=float(np.mean(all_ho)) if all_ho else np.nan,
                          lomo_heldout_improved_frac=float(np.mean(np.array(all_ho) < 0)) if all_ho else np.nan,
                          d_z_pess=float(d_pess), d_z_mean=float(d_mean), d_z_rigid=float(d_rigid),
                          path_sensitivity=path_ratio, base_pair_sensitivity=base_sens, sensitivity_ratio=sens_ratio,
                          validator_pass_rate=float(np.mean([f['valid'] for f in finals])), n_valid=len(valid), n_finals=len(finals),
                          reasons=sorted({r for f in finals for r in f['reasons']}), flags=flags, n_flags=int(sum(flags.values())))
            # route id / file (dedup by content hash; pool-identical picks reuse eval_v1's id and drive)
            if chosen:
                h = G.route_hash(chosen['route'])
                if h == pool_pick['pool_hash']:
                    rid, reuse = locked_rid, True
                else:
                    rid, reuse = route_ids.get(h, f'{g}__{arm}'), False
                if h not in route_ids:
                    route_ids[h] = rid; rt = chosen['route']
                    json.dump({'waypoints': np.asarray(rt['waypoints']).tolist(), 'speeds': np.asarray(rt['speeds']).tolist(),
                               'stations': np.asarray(rt['stations']).tolist(), 'headings': np.asarray(rt['headings']).tolist(),
                               'meta': {**rt['meta'], 'candidate': f'n2_grad_{arm}', 'scene_id': g, 'pool': 'proposal', 'start_pool_index': starts[s_best],
                                        'objective': spec['objective'], 'C_fail': spec['C_fail'], 'lambda_E': spec['lam_E']}}, open(out / 'routes' / f'{rid}.json', 'w'))
                    tasks.append(dict(id=rid, group=g, case=f'{a.case_prefix}/{os.path.basename(case_path)}',
                                      route=f'{a.reuse_route_prefix}/{rid}.json' if reuse else f'{a.route_prefix}/{rid}.json',
                                      run=not reuse, tier=gi, episode_seed=int(hashlib.md5(rid.encode()).hexdigest()[:8], 16),
                                      **({'reuse': f'eval_v1/runs/{rid}'} if reuse else {})))
            else:
                rid, reuse = locked_rid, True
                if pool_pick['pool_hash'] not in route_ids:
                    route_ids[pool_pick['pool_hash']] = rid
                    tasks.append(dict(id=rid, group=g, case=f'{a.case_prefix}/{os.path.basename(case_path)}', route=f'{a.reuse_route_prefix}/{rid}.json',
                                      run=False, tier=gi, episode_seed=int(hashlib.md5(rid.encode()).hexdigest()[:8], 16), reuse=f'eval_v1/runs/{rid}'))
            picked = {k: v for k, v in (chosen or pool_pick).items() if k not in ('route', 'driven')}
            picks[arm] = dict(route_id=rid, abstained=bool(abstain), changed=bool(chosen and not reuse), reuse_eval_v1=bool(reuse),
                              objective=spec['objective'], C_fail=spec['C_fail'], lambda_E=spec['lam_E'],
                              gain=float(gain) if np.isfinite(gain) else None, gain_unit=spec['unit'], J_keep_pool=J_pool,
                              J_keep_best=(best['J_keep'] if best else None), theta=(dict(a=best['a'], dv=best['dv'], start=s_best, start_pool_index=starts[s_best]) if best else None),
                              pick=picked, best_final=({k: v for k, v in best.items() if k != 'route'} if best else None), guards=guards,
                              finals=[{k: v for k, v in f.items() if k != 'route'} for f in finals])
            audit_rows.append(dict(group=g, arm=arm, abstained=bool(abstain), changed=bool(chosen and not reuse), gain=float(gain) if np.isfinite(gain) else None,
                                   P_pool=pool_pick['P'], P_pess_pool=pool_pick['P_pess'], P_new=picked['P'], P_pess_new=picked['P_pess'],
                                   z_pool=pool_pick['z_mean'], z_new=picked['z_mean'], T_pool=pool_pick['T'], T_new=picked['T'], E_pool=pool_pick['E_analytic'], E_new=picked['E_analytic'],
                                   v_pool=pool_pick['mean_speed'], v_new=picked['mean_speed'], lat_pool=pool_pick['max_lateral_m'], lat_new=picked['max_lateral_m'],
                                   kappa_new=(best['kappa_max'] if best else None), pool_fail=(results.get(g, {}).get('crm') or {}).get('fail'),
                                   **{k: v for k, v in guards.items() if k not in ('flags', 'reasons')}, **{f'flag_{k}': v for k, v in flags.items()}))
        summ['arms'] = picks; summ['seconds'] = dict(pool=t_pool, refine=t_ref, total=time.time() - tc)
        timing.append(summ['seconds'])
        json.dump(summ, open(out / 'picks' / f'{g}.json', 'w'), indent=1, default=float)
        if locked is not None and locked != i0:
            print(f'WARNING {g}: pool argmin {i0} != locked eval_v1 pick {locked}', flush=True)
        msg = ' '.join(f"{arm}:{'abstain' if picks[arm]['abstained'] else ('P %.3f->%.3f' % (pool_pick['P'], picks[arm]['pick']['P']))}" for arm in arms)
        print(f'[{gi + 1}/{len(cases)}] {g} pool argmin {i0} (locked {locked}) P {pool_pick["P"]:.3f} v {pool_pick["mean_speed"]:.2f} | {msg} | '
              f'{summ["seconds"]["total"]:.1f}s (pool {t_pool:.1f}, refine {t_ref:.1f}, {res["steps_run"]} steps)', flush=True)
        json.dump(tasks, open(out / 'tasks.json', 'w'), indent=1)
        write_audit(out, audit_rows, arms, timing, energy)
    print(f'{len(cases)} groups, {len(tasks)} task rows ({sum(t["run"] for t in tasks)} new drives), {time.time() - t0:.0f}s')


def write_audit(out, rows, arms, timing, energy):
    def q(x):
        x = np.asarray([v for v in x if v is not None], float)
        return [float(v) for v in np.quantile(x, [0, .1, .25, .5, .75, .9, 1])] if len(x) else None
    summary = {'n_groups': len({r['group'] for r in rows}), 'seconds_per_case': {k: float(np.mean([t[k] for t in timing])) for k in ('pool', 'refine', 'total')} if timing else None,
               'energy_fit': ({k: energy[k] for k in ('pool', 'coefficients_nonzero', 'stats')} if energy else None), 'arms': {}}
    for arm in arms:
        R = [r for r in rows if r['arm'] == arm]
        if not R:
            continue
        risky = [r for r in R if r['P_pool'] > 0.01]; very = [r for r in R if r['P_pool'] > 0.20]
        binc = lambda key, rr: {f'{lo:.2f}-{hi:.2f}': int(sum(lo <= r[key] < hi for r in rr)) for lo, hi in P_BINS}
        flags = {k[5:]: int(sum(bool(r[k]) for r in R)) for k in R[0] if k.startswith('flag_')}
        chg = [r for r in R if r['changed']]
        summary['arms'][arm] = dict(
            n=len(R), changed=len(chg), abstained=int(sum(r['abstained'] for r in R)),
            gain_quantiles=q([r['gain'] for r in R]), gain_unit=ARMS[arm]['unit'],
            P_pool_bins=binc('P_pool', R), P_new_bins=binc('P_new', R), P_pool_quantiles=q([r['P_pool'] for r in R]), P_new_quantiles=q([r['P_new'] for r in R]),
            P_pess_pool_mean=float(np.mean([r['P_pess_pool'] for r in R])), P_pess_new_mean=float(np.mean([r['P_pess_new'] for r in R])),
            mean_P_pool=float(np.mean([r['P_pool'] for r in R])), mean_P_new=float(np.mean([r['P_new'] for r in R])),
            risky_pairs=dict(n=len(risky), changed=int(sum(r['changed'] for r in risky)), P_pool_mean=float(np.mean([r['P_pool'] for r in risky])) if risky else None,
                             P_new_mean=float(np.mean([r['P_new'] for r in risky])) if risky else None, pool_failures=int(sum(bool(r['pool_fail']) for r in risky)),
                             changed_among_pool_failures=int(sum(bool(r['pool_fail']) and r['changed'] for r in risky)),
                             groups=[dict(group=r['group'], P_pool=r['P_pool'], P_new=r['P_new'], changed=r['changed'], pool_fail=r['pool_fail'], v_pool=r['v_pool'], v_new=r['v_new']) for r in risky]),
            very_risky_pairs=dict(n=len(very), changed=int(sum(r['changed'] for r in very)), P_new_mean=float(np.mean([r['P_new'] for r in very])) if very else None),
            mean_speed=dict(pool=float(np.mean([r['v_pool'] for r in R])), new=float(np.mean([r['v_new'] for r in R])), changed_only_pool=float(np.mean([r['v_pool'] for r in chg])) if chg else None,
                            changed_only_new=float(np.mean([r['v_new'] for r in chg])) if chg else None, new_quantiles=q([r['v_new'] for r in R])),
            T=dict(pool=float(np.mean([r['T_pool'] for r in R])), new=float(np.mean([r['T_new'] for r in R]))),
            E_analytic=dict(pool=float(np.mean([r['E_pool'] for r in R])), new=float(np.mean([r['E_new'] for r in R]))),
            max_lateral=dict(pool=float(np.mean([r['lat_pool'] for r in R])), new=float(np.mean([r['lat_new'] for r in R]))),
            guards=dict(lomo_fit_gain_all_mean=float(np.nanmean([r['lomo_fit_gain_all'] for r in R])), lomo_heldout_gain_all_mean=float(np.nanmean([r['lomo_heldout_gain_all'] for r in R])),
                        lomo_heldout_improved_frac_mean=float(np.nanmean([r['lomo_heldout_improved_frac'] for r in R])),
                        lomo_fit_gain_at_pick_mean=float(np.nanmean([r['lomo_fit_gain_at_pick'] for r in R])), lomo_heldout_gain_at_pick_mean=float(np.nanmean([r['lomo_heldout_gain_at_pick'] for r in R])),
                        d_z_pess_changed=q([r['d_z_pess'] for r in chg]), d_z_mean_changed=q([r['d_z_mean'] for r in chg]), d_z_rigid_changed=q([r['d_z_rigid'] for r in chg]),
                        rigid_agrees_frac_changed=float(np.mean([r['d_z_rigid'] < 0 for r in chg])) if chg else None,
                        sensitivity_ratio_quantiles=q([r['sensitivity_ratio'] for r in R if r['sensitivity_ratio'] is not None and np.isfinite(r['sensitivity_ratio'])]),
                        validator_pass_rate_mean=float(np.mean([r['validator_pass_rate'] for r in R])), kappa_new_quantiles=q([r['kappa_new'] for r in R])),
            exploitation_flags=flags, picks_with_any_flag=int(sum(any(bool(r[k]) for k in r if k.startswith('flag_')) for r in chg)))
    json.dump({'summary': summary, 'rows': rows}, open(out / 'audit.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
