#!/usr/bin/env python3
"""Verify and report the post-hoc parent/refinement physical interventions."""
from pathlib import Path
import argparse
import hashlib
import json
import importlib.util

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / 'refinement_summary'
    out.mkdir(exist_ok=True)
    helper = root / 'refinement_counterfactual_v2/diagnose_traverse_fdm_refinement_counterfactual.py'
    spec = importlib.util.spec_from_file_location('physical_parity_helper', helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows, sources, routes = [], {}, {}
    raw = root / 'decision_audit/inputs/raw'
    for directory in ('refinement_counterfactual_v2', 'refinement_best_base_v1'):
        base = root / directory
        protocol = json.loads((base / 'protocol.json').read_text())
        batch = json.loads((base / 'batch_result.json').read_text())
        assert batch['finished'] == batch['declared'] == len(protocol['tasks']) and not batch['failures']
        assert sha(base / 'diagnose_traverse_fdm_refinement_counterfactual.py') == protocol['wrapper_sha256']
        sources[str((base / 'protocol.json').relative_to(root))] = sha(base / 'protocol.json')
        for task in protocol['tasks']:
            run = base / 'runs' / task['id']
            audit = json.loads((run / 'counterfactual_audit.json').read_text())
            assert audit['protocol_sha256'] == sha(base / 'protocol.json')
            assert audit['runtime_exact'] and audit['online_protocol_exact'] and audit['requested_decision_exact']
            for name, expected in audit['artifact_sha256'].items():
                assert sha(run / name) == expected, (task['id'], name)
            original = raw / task['original_trial_id']
            old = json.loads((original / 'decisions/decision_00000.json').read_text())
            decision = json.loads((run / 'decisions/decision_00000.json').read_text())
            if task['mode'] == 'replay_selected':
                parity = module.archive_parity(run, original)
                assert parity['passed'] and parity['array_count'] == 241
                assert decision['decision'] == old['decision']
                reference_cost = old['decision']['cost']
            else:
                scores = old['decision']['family_scores']
                index = (min((s for s in scores if s['cost'] is not None), key=lambda s: s['cost'])['family_index']
                         if task['mode'] == 'best_unrefined' else old['decision']['selected_family_index'])
                assert decision['decision']['route'] == old['candidate_references'][index]
                reference_cost = next(s['cost'] for s in scores if s['family_index'] == index)
            assert decision['decision']['cost'] == reference_cost
            with np.load(run / 'rich_intervals.npz', allow_pickle=False) as z:
                contact = np.maximum(z['max_asset_contact_max_resultant_n'], z['max_chassis_contact_resultant_n']) > 1.
                # Native times include the 0.8 s settle; report elapsed traversal.
                first = float(z['start_time_s'][np.flatnonzero(contact)[0]] - z['start_time_s'][0]) if contact.any() else None
            row = {'trial_id': task['original_trial_id'], 'mode': task['mode'],
                   'model_cost': reference_cost, 'first_contact_interval_start_s': first,
                   **audit['outcome'], 'replay_arrays_exact': True if task['mode'] == 'replay_selected' else None,
                   'replay_parity_applicable': task['mode'] == 'replay_selected',
                   'run': str(run.relative_to(root))}
            rows.append(row)
            with np.load(run / 'trajectory.npz', allow_pickle=False) as z:
                routes[(task['original_trial_id'], task['mode'])] = np.vstack([z['pose'][:, :2], z['terminal_pose'][None, :2]])
    assert len(rows) == 9
    # Parent controls are identical across the two cost settings within a scene.
    for scene in ('rolling_hills', 'rough_mosaic'):
        left = next(r for r in rows if scene in r['trial_id'] and r['trial_id'].endswith('_time') and r['mode'] == 'parent_family')
        right = next(r for r in rows if scene in r['trial_id'] and r['trial_id'].endswith('_energy') and r['mode'] == 'parent_family')
        parent_parity = module.archive_parity(root / left['run'], root / right['run'])
        assert parent_parity['passed']
    result = {'scope': 'Post-hoc diagnostics; original protected results unchanged',
              'all_nine_trials_verified': True, 'four_selected_replays_exact_241_arrays_each': True,
              'paired_parent_arms_physically_identical': True, 'rows': rows, 'source_sha256': sources,
              'report_script_sha256': sha(__file__),
              'interpretation': 'Reference geometry AND speed profiles are intervened on. No new model inference or tuning. The best unrefined route is selected by original launch costs, not physical outcomes. Scenes were selected post-hoc.'}
    (out / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# Physical MPPI refinement counterfactuals', '',
        'All nine headless AMD trials pass source/runtime/input and artifact checks. All four selected-reference controls reproduce every one of 241 stored physical arrays exactly. Each parent reference also reproduces identically across the two cost settings. Original protected scores remain unchanged.', '',
        'Contact timestamps below are starts of the first positive 50 ms contact interval, not exact contact instants. For example, rough-time first contact lies in [11.70, 11.75] s.', '',
        '| Scene / cost | Executed reference | Frozen model cost | Safe goal | Duration | First contact interval start |',
        '|---|---|---:|---|---:|---:|']
    for r in rows:
        scene = 'Rolling hills' if 'rolling_hills' in r['trial_id'] else 'Rough mosaic'
        cost = 'time' if r['trial_id'].endswith('_time') else 'energy'
        first = 'none' if r['first_contact_interval_start_s'] is None else f"{r['first_contact_interval_start_s']:.2f} s"
        lines.append(f"| {scene} / {cost} | {r['mode']} | {r['model_cost']:.3f} | {r['schema_safe_goal_reached']} | {r['elapsed_s']:.2f} s | {first} |")
    lines += ['', 'The rough-mosaic time/risk comparison establishes a harmful ranking change in this case. The original best family (index 12, −44 m at 6 m/s) has model cost 47.608 and safely finishes in 41.30 s. Another parent (index 6, −22 m at 6 m/s) costs 48.721, but MPPI refinement lowers its predicted cost to 46.475 and makes it the winner. That executed route contacts within 12 s, blocks and times out at 180 s.', '',
        'All four selected parent references are already unsafe. The rough-mosaic parent eventually reaches the goal at 120.25 s after contact and blockage, while its refinements time out. Thus local reference changes worsen that outcome but do not create its initial unsafe classification. Rolling-hills parents and refinements both contact and block. Small MPPI changes are not necessary for every failure.', '',
        'This isolates the executed reference, including both geometry and speed. It does not show that removing MPPI improves a general success rate. The single additional best-base reference was chosen by the original model-cost argmin; it was not chosen using its subsequently measured success.', '',
        'The wrapper substitutes one archived launch decision and imports the immutable online_v9 physical loop and native PID adapter. It verifies the exact launch history, pose, map, goal and candidate-family arrays. Sidecar counterfactual_audit.json files identify the intervention; the inherited online_protocol.json describes the unchanged engine, not a fresh neural inference. Parent probabilities must be read from the matching original family_scores row.', '',
        'The earlier eight-job wrapper attempt 412221 stopped before traversal because its exact goal guard compared a float32 observation copy with the runner’s float64 case goal. Version 2 uses the same case goal as the frozen runner, retaining exact equality. That failed harness and all logs remain archived.', '',
        '[Cost and physical-route figure](ranking_counterfactual.png) · [Machine-readable report](report.json)', '']
    (out / 'report.md').write_text('\n'.join(lines))
    tid = 'diverse_v1_test_rough_mosaic_00_selected_rgbd_time'
    choices = [(mode, next(r for r in rows if r['trial_id'] == tid and r['mode'] == mode))
               for mode in ('best_unrefined', 'parent_family', 'replay_selected')]
    colors = ['#168663', '#ce8b26', '#c94454']
    labels = ['Best original candidate\nSafe goal: 41.30 s', 'Parent of MPPI winner\nUnsafe goal: 120.25 s', 'MPPI-refined winner\nContact/blockage: 180 s timeout']
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={'width_ratios':[1.05,1.2]})
    costs = [r['model_cost'] for _, r in choices]
    axes[0].barh(np.arange(3), costs, color=colors, height=.57)
    axes[0].set_yticks(np.arange(3), labels)
    axes[0].invert_yaxis(); axes[0].set_xlim(0, 55)
    for i, value in enumerate(costs):
        axes[0].text(value+.3, i, f'{value:.3f}', va='center', fontsize=10)
    axes[0].set_xlabel('Frozen FDM objective (lower is preferred; not measured travel time)')
    axes[0].set_title('MPPI prefers a physically failed route')
    axes[0].grid(axis='x', alpha=.18); axes[0].set_axisbelow(True)
    for i, (mode, r) in enumerate(choices):
        xy = routes[(tid, mode)]
        axes[1].plot(xy[:,0], xy[:,1], color=colors[i], linewidth=2.0,
                     label=labels[i].replace('\n', ': '))
        axes[1].scatter(*xy[-1], color=colors[i], s=26, zorder=5)
    old = json.loads((raw/tid/'decisions/decision_00000.json').read_text())
    start = np.asarray(old['anchor_pose'])[:2]
    goal = np.asarray(old['decision']['route']['waypoints'])[-1]
    axes[1].scatter(*start, marker='o', facecolor='white', edgecolor='black', s=65, zorder=6)
    axes[1].scatter(*goal, marker='*', color='black', s=115, zorder=6)
    axes[1].set_aspect('equal', adjustable='datalim'); axes[1].grid(alpha=.18)
    axes[1].set_xlabel('World X (m)'); axes[1].set_ylabel('World Y (m)')
    axes[1].set_title('Actual Chrono trajectories; same launch and PID')
    fig.suptitle('Rough mosaic, time/risk: exact frozen-reference counterfactual', fontsize=14)
    fig.text(.5, .015, 'Post-hoc selected failure. Four original-route controls reproduce all 241 physical arrays; no new model training or cost tuning.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    fig.savefig(out/'ranking_counterfactual.png', dpi=180)
    fig.savefig(out/'ranking_counterfactual.pdf')
    plt.close(fig)
    print(json.dumps({'report':str(out/'report.md'),'verified_trials':len(rows),'exact_controls':4}))


if __name__ == '__main__':
    main()
