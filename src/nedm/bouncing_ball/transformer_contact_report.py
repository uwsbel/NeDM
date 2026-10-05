"""Compact report and paired error distributions from completed AMD evidence."""
import argparse
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nedm.bouncing_ball.collection import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Plot on an AMD compute allocation')
    full=json.loads((args.root/'certification/paired_results.json').read_text())
    summaries=[]
    for name,record in full['models'].items():
        free=record['free_rollout'];teacher=record.get('teacher_forced',{})
        raw=full['raw_position_interpolation'][name]
        summaries.append(dict(name=name,p95_trajectory_rmse_mm=free['position_rmse_m']['p95']*1000,
            p95_endpoint_mm=free['endpoint_error_m']['p95']*1000,correct_order_fraction=free['contact_order_fraction'],
            finite_fraction=free['finite_fraction'],teacher_contact=teacher.get('contact_classification'),
            rollout_contact=free.get('contact_from_predicted_states'),
            raw_interpolated_p95_rmse_mm=raw['position_rmse_m']['p95']*1000,
            checkpoint_sha256=record['checkpoint_sha256']))
    report=dict(complete=True,fresh_test_episodes=full['fresh_test_episodes'],fresh_seed=full['fresh_seed'],
        selection_before_test=True,chosen=full['chosen_before_test'],dt_s=full['dt_s'],models=summaries,
        architecture='existing ContinuousTransformer core plus learned contact gate and bounce residual; no analytical flight/contact/timing in new inference',
        new_update='state + Transformer_delta5 + gate * bounce_delta5',
        primary_inputs=dict(transformer=5,gate=2,bounce=2),
        limitations=['One training seed per configuration','Original21k and refined54k budgets both preserved',
            'Historical V1/V2 have gravity and geometric priors; matched pure Transformer separately trained',
            'Raw position comparison uses linear native10ms interpolation, not an exact dense physical trajectory',
            'Hard gates are piecewise differentiable; finite gradients do not guarantee robust targeting'])
    output=args.root/'reports';output.mkdir(exist_ok=True)
    atomic_json(output/'summary.json',report)
    names=['historical_v1_refined','historical_v2_certified',full['chosen_before_test']['transformer_only'],
        full['chosen_before_test']['literal_primary'],full['chosen_before_test']['fullstate_gate']]
    labels=['Historical V1 refined','Accepted Transformer V2','Matched Transformer only','Exact x,z proposal','Full-state gate diagnostic']
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),constrained_layout=True)
    colors=['#a1a1a1','#236889','#b86d20','#aa3548','#40835a']
    for name,label,color in zip(names,labels,colors):
        values=full['models'][name]['free_rollout']['per_episode']
        for ax,key in zip(axes,['position_rmse_m','endpoint_error_m']):
            sorted_values=np.sort([r.get(key,1e6)*1000 for r in values])
            ax.plot(sorted_values,np.arange(1,len(values)+1)/len(values),label=label,color=color,lw=2)
    for ax,title in zip(axes,['Whole-trajectory position RMSE','End-of-horizon position error']):
        ax.set(xscale='log',xlabel='Error (mm)',ylabel='Fraction of 900 new episodes',title=title,ylim=(0,1.02));ax.grid(alpha=.25)
    axes[0].legend(fontsize=8,loc='lower right')
    fig.suptitle('Transformer core retained: learned contact residual experiment',fontsize=14)
    fig.savefig(output/'transformer_contact_comparison.png',dpi=150);plt.close(fig)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
