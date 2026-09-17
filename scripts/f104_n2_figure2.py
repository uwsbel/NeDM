"""Night-2 second figure: the three Chrono tests side by side, with the metric caveats made visible."""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N2 = 'artifacts/traverse/fdm_f104_50h_20260909/night2_v1'
SURF, INK, INK2 = '#fcfcfb', '#0b0b0b', '#52514e'
COL = {'control': '#b0692c', 'model': '#7b5cb8', 'sampler': '#2a78d6', 'sampler_noanchor': '#4da3e8',
       'pessimist': '#00897b', 'anchor6': '#52514e', 'fixed2_control': '#b8b5ae', 'fixed2_new': '#8a867c',
       'fixed2_old': '#b8b5ae', 'old': '#b0692c', 'new': '#2a78d6'}
NAME = {'control': "last night's\nplanner", 'model': 'new model,\nsame routes', 'sampler': 'new model +\nnew proposal',
        'sampler_noanchor': 'new proposal,\nno designed\nfallbacks', 'pessimist': 'pessimistic\nensemble',
        'anchor6': 'always 6 m/s\nstraight', 'fixed2_control': '2 m/s:\nold model', 'fixed2_new': '2 m/s:\nnew model',
        'fixed2_old': '2 m/s:\nold model', 'old': 'old model', 'new': 'new model'}


def bars(ax, res, arms, keys, title, sub):
    G = list(res)
    rng = np.random.default_rng(0)
    w = 0.8 / len(keys)
    hatch = {None: None}
    for ki, key in enumerate(keys):
        vals, lo, hi = [], [], []
        for a in arms:
            v = np.array([res[g][a][key] for g in G], float)
            b = [v[i].mean() for i in (rng.integers(0, len(G), len(G)) for _ in range(3000))]
            vals.append(100 * v.mean()); lo.append(100 * np.percentile(b, 2.5)); hi.append(100 * np.percentile(b, 97.5))
        x = np.arange(len(arms)) + (ki - (len(keys) - 1) / 2) * w
        ax.bar(x, vals, width=w * .92, color=[COL[a] for a in arms], alpha=1.0 if ki == 0 else .45,
               edgecolor=INK if ki else 'none', linewidth=.6 if ki else 0,
               label={'fail': 'did not reach goal', 'unsafe': 'failed or slid back',
                      'unsafe_or_tilt': 'that, or tilted past 35°'}[key])
        ax.errorbar(x, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt='none',
                    ecolor=INK, capsize=2.5, lw=1)
        for xi, v in zip(x, vals):
            ax.text(xi, v + .25, f'{v:.1f}', ha='center', fontsize=7.5, color=INK)
    ax.set_xticks(range(len(arms))); ax.set_xticklabels([NAME[a] for a in arms], fontsize=8, color=INK2)
    ax.set_ylabel('% of groups', fontsize=9, color=INK2)
    ax.set_title(title, fontsize=11.5, color=INK)
    ax.text(.5, -.30, sub, transform=ax.transAxes, ha='center', fontsize=8.5, color=INK2)
    ax.legend(fontsize=8, frameon=False, loc='upper right')


def main():
    main_res = json.load(open(N2 + '/closed/results.json'))
    ext_res = json.load(open(N2 + '/closed_ext/results.json'))
    haz = N2 + '/closed_haz/results.json'
    n = 3 if os.path.exists(haz) else 2
    fig = plt.figure(figsize=(7.2 * n, 6.4), facecolor=SURF)
    gs = fig.add_gridspec(1, n, wspace=.26, left=.05, right=.99, top=.80, bottom=.20)
    for r in main_res.values():
        for a in r: r[a]['unsafe_or_tilt'] = int(r[a]['unsafe'] or r[a]['max_tilt'] > 35)
    bars(fig.add_subplot(gs[0, 0]), main_res,
         ['control', 'model', 'sampler', 'pessimist', 'anchor6'], ['unsafe', 'unsafe_or_tilt'],
         'Test 1: 184 held-out start/goals, speed free',
         'mostly long flat traverses (15% target a hill or crater):\nevery arm is near the failure floor')
    for r in ext_res.values():
        for a in r: r[a]['unsafe_or_tilt'] = int(r[a]['unsafe'] or r[a]['max_tilt'] > 35)
    bars(fig.add_subplot(gs[0, 1]), ext_res, ['old', 'new'], ['unsafe', 'unsafe_or_tilt'],
         'Test 2: 523 held-out start/goals, speed fixed at 2 m/s',
         'identical candidate routes, only the model differs:\nslides 3.1% -> 1.3% (p = 0.022)')
    if n == 3:
        haz_res = json.load(open(haz))
        for r in haz_res.values():
            for a in r: r[a]['unsafe_or_tilt'] = int(r[a]['unsafe'] or r[a]['max_tilt'] > 35)
        bars(fig.add_subplot(gs[0, 2]), haz_res,
             ['control', 'sampler', 'sampler_noanchor', 'anchor6', 'fixed2_old', 'fixed2_new'],
             ['unsafe', 'unsafe_or_tilt'],
             'Test 3: 300 start/goals aimed at a hill or crater',
             'the terrain the first test filtered out')
    fig.text(.05, .935, 'Night 2: three Chrono tests, reported on both safety metrics', fontsize=16, color=INK, weight='bold')
    fig.text(.05, .885, 'Solid bar = failed or slid backwards. Outlined bar = the same, but also counting runs that leaned past 35 degrees, '
             'which is how the fast straight-line baseline buys its record. Bars are % of groups with 95% group-bootstrap intervals.',
             fontsize=9.5, color=INK2)
    for ax in fig.axes:
        ax.set_facecolor(SURF); ax.tick_params(colors=INK2, labelsize=8.5)
        for s in ('top', 'right'): ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'): ax.spines[s].set_color('#d9d8d4')
    fig.savefig(N2 + '/night2_tests.png', dpi=130, facecolor=SURF)
    print('wrote', N2 + '/night2_tests.png')


if __name__ == '__main__':
    main()
