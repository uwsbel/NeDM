"""Night-2 report figures (static PNG, palette: blue #2a78d6 rigid, orange #eb6834 CRM, aqua #1baf7a, violet #4a3aa7).

  python scripts/n2_figures.py --root artifacts/traverse/crm_night2_v1 --out artifacts/traverse/crm_night2_v1/figures
fig_arch        stage A: within-group AUC per architecture (5 seeds, mean +- sd, ensemble marker), CRM and rigid at matched size
fig_energy      stage B: learned energy head vs the analytic baseline (log-RMSE and within-group Spearman)
fig_velocity    stage C: AUC on moving anchors by speed stratum for the velocity-input variants
fig_moving      moving-start ground truth: realised failure vs predicted risk by initial speed
fig_planner     planner arms: goal reached per arm (Wilson 95 %), CRM iterated/gradient arms + rigid sets
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7'
WC = {'crm': ORANGE, 'rigid': BLUE}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE})
ARCH_ORDER = ['mlp', 'gru', 'gru120', 'gru2', 'tx_conv', 'tx96_2', 'tx96_4', 'tx128_4', 'tx192_4', 'patch4_128_4', 'patch2_128_6']
ARCH_SHORT = {'gru': 'CNN-GRU', 'tx96_2': 'tx d96 L2', 'tx128_4': 'tx d128 L4'}
ARCH_LABEL = {'mlp': 'MLP', 'gru': 'CNN-GRU\n(current)', 'gru120': 'CNN-GRU\nwide', 'gru2': 'CNN-GRU\n2 layers', 'tx_conv': 'CNN tokens\n+ transformer',
              'tx96_2': 'transformer\nd96 L2', 'tx96_4': 'transformer\nd96 L4', 'tx128_4': 'transformer\nd128 L4', 'tx192_4': 'transformer\nd192 L4',
              'patch4_128_4': 'patch-4 tokens\nd128 L4', 'patch2_128_6': 'patch-2 tokens\nd128 L6'}


def style(ax):
    for s in ('top', 'right', 'left'):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)


def wilson(k, n, z=1.96):
    if n == 0: return 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def load_stage(d):
    rows = []
    for f in sorted(glob.glob(d + '/*.json')):
        if f.endswith('report_rows.json'): continue
        s = json.load(open(f))
        if s.get('skipped') or 'members' not in s: continue
        W = np.array([m['heldout']['W_fail'] for m in s['members']])
        r = dict(tag=s['tag'], world=s['args']['world'], arch=s['args']['arch'], ctx=s['args'].get('ctx', 'geom'), vplane=bool(s['args'].get('vplane')),
                 energy=float(s['args'].get('energy', 0)), lr=float(s['args']['lr']), n=len(W), W=W.mean(), Wsd=W.std(ddof=1) if len(W) > 1 else 0.0,
                 Wens=s['ensemble']['heldout']['W_fail'], params=int(s['members'][0]['params']))
        if 'energy_heldout' in s['members'][0]:
            for k in ('E_log_rmse', 'E_spearman_within', 'E_mape', 'T_log_rmse', 'T_spearman_within'):
                v = np.array([m['energy_heldout'][k] for m in s['members']]); r[k] = v.mean(); r[k + '_sd'] = v.std(ddof=1) if len(v) > 1 else 0.0
        rows.append(r)
    return rows


def fig_arch(root, out):
    rows = load_stage(root + '/stageA')
    if not rows: return
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for ax, w in zip(axes, ('crm', 'rigid')):
        R = [r for r in rows if r['world'] == w]
        best = {}
        for r in R:   # best learning rate per architecture by mean member AUC
            if r['arch'] not in best or r['W'] > best[r['arch']]['W']: best[r['arch']] = r
        archs = [a_ for a_ in ARCH_ORDER if a_ in best]; x = np.arange(len(archs))
        m = np.array([best[a_]['W'] for a_ in archs]); sd = np.array([best[a_]['Wsd'] for a_ in archs]); e = np.array([best[a_]['Wens'] for a_ in archs])
        ax.errorbar(x, m, yerr=sd, fmt='o', color=WC[w], ecolor=WC[w], capsize=3, ms=6, label='single network (mean +- sd over seeds)')
        ax.scatter(x, e, marker='_', s=260, color=INK, linewidths=2, label='5-member ensemble', zorder=3)
        ref = best.get('gru')
        if ref: ax.axhline(ref['Wens'], color=GRID, lw=1, ls='--')
        ax.set_xticks(x); ax.set_xticklabels([ARCH_LABEL.get(a_, a_) for a_ in archs], fontsize=6.8)
        ax.set_ylabel('within-group AUC (held-out groups)'); style(ax)
        ax.set_title(f"{'CRM soil' if w == 'crm' else 'rigid ground'}: {len(R)} arms, same 13.6 k training routes", loc='left', fontsize=10.5)
        lo, hi = (m - sd).min(), max(e.max(), (m + sd).max()); pad = 0.25 * (hi - lo + 1e-3); ax.set_ylim(lo - pad, hi + pad)
    axes[0].legend(frameon=False, fontsize=8.5, loc='lower left')
    fig.suptitle('Architecture at matched data: no arm beats the current CNN-GRU by more than seed noise', x=0.01, ha='left', fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f'{out}/fig_arch.png', dpi=160); plt.close(fig)


def fig_energy(root, out):
    rows = [r for r in load_stage(root + '/stageB') if 'E_log_rmse' in r]
    if not rows: return
    base = json.load(open(root + '/energy_baselines/summary.json'))
    ana = {}
    for w in ('crm', 'rigid'):   # clean-only fit scored on clean held-out arrivals, like-for-like with the trainer's energy read-out
        if w in base:
            h = base[w]['metrics'].get('heldout_clean', base[w]['metrics']['heldout']); ana[w] = h.get('analytic_clean', h['analytic'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, lab in zip(axes, ('E_log_rmse', 'E_spearman_within'), ('energy log-RMSE (lower is better)', 'within-group Spearman (higher is better)')):
        labels, vals, sds, cols = [], [], [], []
        for w in ('crm', 'rigid'):
            R = sorted([r for r in rows if r['world'] == w], key=lambda r: (r['arch'] != 'gru', r['arch'], r['energy']))
            for r in R:
                labels.append(f"{ARCH_SHORT.get(r['arch'], r['arch'])} {r['energy']:g}"); vals.append(r[key]); sds.append(r[key + '_sd']); cols.append(WC[w])
        x = np.arange(len(labels)); ax.bar(x, vals, 0.6, color=cols, yerr=sds, ecolor=INK2, capsize=2)
        ax.set_xlabel('architecture and energy-loss weight lambda', fontsize=8.5)
        for w, ls in (('crm', '-'), ('rigid', '--')):
            if w in ana:
                v = ana[w]['log_rmse'] if key == 'E_log_rmse' else ana[w]['spearman_within_group']
                ax.axhline(v, color=WC[w], lw=1.2, ls=ls, label=f"analytic baseline ({'CRM' if w == 'crm' else 'rigid'}) {v:.2f}")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7, rotation=60, ha='right'); ax.set_ylabel(lab); style(ax); ax.legend(frameon=False, fontsize=8)
    axes[0].set_title('Energy head (network) vs the analytic work model, held-out clean arrivals', loc='left', fontsize=10.5)
    fig.tight_layout(); fig.savefig(f'{out}/fig_energy.png', dpi=160); plt.close(fig)


def fig_velocity(root, out):
    p = root + '/stageC/report_rows.json'
    if not os.path.exists(p): return
    rows = [r for r in json.load(open(p)) if 'W_mov' in r]
    strata = [('W_start', 'standing\nstart'), ('W_mov', 'moving\n(all)'), ('W_v01', 'moving\n< 1 m/s'), ('W_v13', 'moving\n1-3 m/s'), ('W_v36', 'moving\n3-6 m/s')]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, w in zip(axes, ('crm', 'rigid')):
        R = sorted([r for r in rows if r['world'] == w], key=lambda r: (r['arch'] != 'gru', r['ctx'] != 'geom', r['vplane'], r['ctx']))
        if not R: continue
        nb = len(R); wbar = 0.8 / nb; x = np.arange(len(strata))
        shades = [GRID, WC[w], AQUA, VIOLET, INK2, '#b0aeaa'][:nb]
        for i, r in enumerate(R):
            name = f"{r['arch']} ctx={r['ctx']}" + (' + vx plane' if r['vplane'] else '')
            ax.bar(x + (i - (nb - 1) / 2) * wbar, [r[k] for k, _ in strata], wbar, color=shades[i], label=name)
        ax.set_xticks(x); ax.set_xticklabels([l for _, l in strata], fontsize=8.5); ax.set_ylabel('within-group AUC (re-anchored held-out rows)')
        vals = [r[k] for r in R for k, _ in strata]; ax.set_ylim(min(vals) - 0.02, min(1.0, max(vals) + 0.015)); style(ax)
        ax.legend(frameon=False, fontsize=7.5, loc='lower right'); ax.set_title('CRM soil' if w == 'crm' else 'rigid ground', loc='left', fontsize=10.5)
    fig.suptitle('Velocity as an input: AUC by anchor speed (geom = no vehicle state; vel = vx, vy, yaw rate; chassis = full state)', x=0.01, ha='left', fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(f'{out}/fig_velocity.png', dpi=160); plt.close(fig)


def fig_moving(root, out):
    p = root + '/moving_v1/moving_analysis.json'
    if not os.path.exists(p): return
    m = json.load(open(p)); v0 = ['0', '2', '4']; x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7.4, 4))
    ax.bar(x, [100 * m['fail_by_v0'][v] for v in v0], 0.5, color=GRID, label='realised failure (1,080 rigid drives)')
    cols = [BLUE, AQUA, VIOLET, ORANGE, INK2]
    for i, (tag, r) in enumerate(sorted(m['arms'].items())):
        name = tag.replace('rigid_', '').replace('_lr0.002_holdout', '').replace('_lr0.001_holdout', '')
        ax.plot(x, [100 * r['mean_P_by_v0'][v] for v in v0], '-o', color=cols[i % len(cols)], label=f"predicted P: {name} (AUC {r['auc_all']:.3f}, direction {r['direction_agree']}/{r['direction_total']})")
    ax.set_xticks(x); ax.set_xticklabels(['0 m/s', '2 m/s (1.5 measured)', '4 m/s (3.0 measured)']); ax.set_xlabel('initial speed at the mid-route anchor')
    ax.set_ylabel('%'); style(ax); ax.legend(frameon=False, fontsize=7.5)
    ax.set_title('Moving starts on rigid ground: realised outcome is flat in initial speed,\nvelocity-aware models predict a downward trend', loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(f'{out}/fig_moving.png', dpi=160); plt.close(fig)


ARM_LABEL = {'A': 'A one-shot\n256 (deployed)', 'B': 'B CEM\n4 x 64', 'C': 'C CEM\n8 x 64', 'D': 'D one-shot\n512', 'E': 'E CEM\npessimistic', 'F': 'F expected\ncost (60 s)',
             'G': 'G gradient\nrisk', 'H': 'H gradient\nrisk+energy'}


def fig_planner(root, out, sets):
    panels = []
    for name, path in sets:
        if os.path.exists(path):
            s = json.load(open(path))['summary']; panels.append((name, s))
    if not panels: return
    ncol = min(3, len(panels)); nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.6 * ncol, 4.3 * nrow), squeeze=False, sharey=False)
    for ax in axes.flat[len(panels):]: ax.axis('off')
    for ax, (name, s) in zip(axes.flat, panels):
        arms = s['arms']; n = s['PRIMARY_' + [k for k in s if k.startswith('PRIMARY_')][0][8:]]['n'] if any(k.startswith('PRIMARY_') for k in s) else s['groups']
        key = 'goal_reached' if s['label'] == 'fail' else 'safe'
        vals = [s['rates'][a_][key] if key in s['rates'][a_] else 100 - s['rates'][a_]['unsafe'] for a_ in arms]
        cis = [wilson(round(v * n / 100), n) for v in vals]; x = np.arange(len(arms))
        col = ORANGE if 'crm' in name.lower() else BLUE
        ax.bar(x, vals, 0.58, color=[col if a_ != 'A' else GRID for a_ in arms])
        ax.errorbar(x, vals, yerr=[[v - lo for v, (lo, hi) in zip(vals, cis)], [hi - v for v, (lo, hi) in zip(vals, cis)]], fmt='none', ecolor=INK2, elinewidth=1, capsize=3)
        for xi, v, (lo, hi) in zip(x, vals, cis): ax.text(xi, hi + 0.8, f'{v:.1f}', ha='center', va='bottom', fontsize=8.5)
        ax.set_xticks(x); ax.set_xticklabels([ARM_LABEL.get(a_, a_) for a_ in arms], fontsize=6.8)
        ax.set_ylabel('goal reached (%)' if s['label'] == 'fail' else 'goal reached, no unsafe event (%)', fontsize=9); style(ax)
        lo = min(c[0] for c in cis); ax.set_ylim(max(0, lo - 8), 104)
        ax.set_title(f'{name} ({n} start-goal pairs)', loc='left', fontsize=10)
    fig.suptitle('Planner sampling arms, closed loop in Chrono (95 % Wilson intervals; grey = deployed one-shot sampler)', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(f'{out}/fig_planner.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--root', default='artifacts/traverse/crm_night2_v1'); ap.add_argument('--out', default=None)
    a = ap.parse_args(); out = a.out or a.root + '/figures'; os.makedirs(out, exist_ok=True)
    fig_arch(a.root, out); fig_energy(a.root, out); fig_velocity(a.root, out); fig_moving(a.root, out)
    P = a.root + '/planner'
    fig_planner(a.root, out, [('CRM soil, f104', P + '/eval_iter_crm/results.json'), ('CRM soil, gradient arms', P + '/eval_grad_crm/results.json'),
                              ('rigid f104, fixed 2 m/s', P + '/eval_rigid_f104_fixed2/results.json'), ('rigid g216', P + '/eval_rigid_g216/results.json'),
                              ('rigid g231', P + '/eval_rigid_g231/results.json')])
    print('figures ->', out, sorted(os.listdir(out)))
