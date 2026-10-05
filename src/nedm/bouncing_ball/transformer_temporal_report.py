"""Compact measured report and Transformer-function diagnostics on AMD."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import numpy as np
import torch
from nedm.bouncing_ball.collection import atomic_json
from nedm.bouncing_ball.model import load_model


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(1)
    selection=json.loads((args.root/'selection.json').read_text())
    report=json.loads((args.root/'certification/paired_results.json').read_text())
    truth=np.load(args.root/'certification_data/model_data.npz')['states']
    primary_tail=sorted(report['models'][selection['chosen']]['free_rollout']['per_episode'],
        key=lambda r:r.get('endpoint_error_m',float('inf')),reverse=True)[:3]
    selected_candidates={c['name']:c for c in selection['candidates']}
    rows=[];diagnostics={}
    for name,result in report['models'].items():
        metrics=result['free_rollout'];mc=result['model_config']
        finite=[r for r in metrics['per_episode'] if r['finite']]
        trajectory=np.array([r['position_rmse_m'] for r in finite])
        rows.append(dict(name=name,trajectory_median_mm=1000*metrics['position_rmse_m']['median'],
            trajectory_p95_mm=1000*metrics['position_rmse_m']['p95'],endpoint_p95_mm=1000*metrics['endpoint_error_m']['p95'],
            maximum_error_p95_mm=1000*metrics['maximum_position_error_m']['p95'],finite=metrics['finite_fraction'],
            bounce_order=metrics['contact_order_fraction'],context=mc.get('context',1),embedding=mc.get('embedding'),
            layers=mc.get('layers'),bounce_inputs=mc.get('bounce_input_dim'),bounce_heads=mc.get('contact_modes'),
            gate_inputs=mc.get('gate_input_dim'),gate_outputs=mc.get('gate_modes',mc.get('contact_modes')),
            gate_pool=mc.get('gate_pool','logsumexp'),linear_gate_enabled=mc.get('linear_gate_enabled',True),
            functional_core=selected_candidates.get(name,{}).get('functional_core',result.get('note')),
            trajectory_p99_mm=float(1000*np.quantile(trajectory,.99)) if len(trajectory) else None,
            trajectory_worst_mm=1000*metrics['position_rmse_m']['max'],
            endpoint_worst_mm=1000*metrics['endpoint_error_m']['max'],
            trajectory_over_10mm=int((trajectory>.01).sum()),
            architecture=mc.get('architecture'),checkpoint_sha256=result['checkpoint_sha256']))
    for candidate in selection['candidates']:
        model,_=load_model(candidate['path'],'cpu')
        projections=[]
        for block in model.backbone.blocks:
            projections.append(dict(attention_projection_norm=float(block.attn.c_proj.weight.norm()),mlp_projection_norm=float(block.mlp.c_proj.weight.norm())))
        with torch.no_grad():
            state=model.state_mean.new_tensor([[0.,1.,5.5,-9.75,0.]])
            h=model.initial_history(state);changed=h.clone()
            if model.context>1:changed[:,:-1,0]+=.01
            shift=(model.core_delta(state,changed)-model.core_delta(state,h)).abs()
            same_current_history_effect=float(shift.max())
        all_zero=all(x['attention_projection_norm']==0 and x['mlp_projection_norm']==0 for x in projections)
        diagnostics[candidate['name']]=dict(projections=projections,all_residual_projections_zero=all_zero,
            functional_core='data-fitted current-state affine map' if all_zero and model.affine_preserving else 'trained nonlinear Transformer',
            older_history_changed_current_fixed_max_delta_change=same_current_history_effect,
            context=model.context,affine_preserving=model.affine_preserving,identity_initialization=model.config.get('identity_init',True))
        if candidate['name'] in (selection['chosen'],'direct_binary_mlp_shared5'):
            trace=np.load(args.root/'certification'/(candidate['name']+'_traces.npz'))
            trace_rows={int(idx):i for i,idx in enumerate(trace['indices'])}
            tails=[]
            for case in primary_tail:
                idx=case['index'];row=trace_rows[idx];n=int(trace['lengths'][row]);steps=[]
                predicted=torch.from_numpy(trace['predicted'][row,:n]).to(model.state_mean.dtype)
                for step in range(n-4,n-1):
                    h=predicted[torch.arange(step-model.context+1,step+1).clamp_min(0)].unsqueeze(0)
                    with torch.no_grad():_,info=model.details(predicted[step:step+1],h)
                    steps.append(dict(step=step,time_s=step*model.dt,
                        predicted_state=predicted[step].tolist(),true_state=truth[idx,step].tolist(),
                        logits=info['logits'][0].tolist(),gates=info['gates'][0].tolist(),
                        delta1=info['delta1'][0].tolist(),delta2_by_mode=info['delta2_by_mode'][0].tolist()))
                tails.append(dict(index=idx,terminal_time_s=(n-1)*model.dt,steps=steps))
            diagnostics[candidate['name']]['paired_primary_tail_diagnostics']=tails
    target_paths=list((args.root/'targeting').glob('*/chrono_verification.json')) if (args.root/'targeting').exists() else []
    targets={str(p.parent.name):json.loads(p.read_text()) for p in target_paths}
    summary=dict(fresh_test_episodes=report['episodes'],fresh_seed=report['fresh_seed'],chosen_before_test=selection['chosen'],
        metrics='Each episode: RMS Euclidean x,z error over native 10ms trajectory; table reports median/p95 across episodes. Endpoint is each valid trajectory terminal time, not1.7s.',
        rows=rows,transformer_diagnostics=diagnostics,targeting=targets,
        selected_raw_interpolation=report['selected_raw_interpolation'],selection_before_test=True)
    atomic_json(args.root/'summary.json',summary)
    with (args.root/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    chosen=selection['chosen']
    names=[chosen,'direct_binary_mlp_shared5','trained_shared5_w512',
        'standard_k1_w64_l2','standard_k16_w256_l6','ap_k1_b2','ap_k1_b3',
        'prior_fullstate_117mm','historical_transformer_v2','analytical_mlp_reference']
    labels={chosen:'Trained Transformer K8 + two bounce NNs',
        'direct_binary_mlp_shared5':'Scalar contact MLP + one bounce NN',
        'trained_shared5_w512':'Learned OR gate + one bounce NN',
        'standard_k1_w64_l2':'Standard Transformer K1 / L2 / width64',
        'standard_k16_w256_l6':'Standard Transformer K16 / L6 / width256',
        'ap_k1_b2':'Matched control: bounce input vx,vz',
        'ap_k1_b3':'Matched control: bounce input vx,vz,spin',
        'prior_fullstate_117mm':'Previous full-state gate diagnostic',
        'historical_transformer_v2':'Previous analytical-contact Transformer V2',
        'analytical_mlp_reference':'Analytical-contact MLP reference'}
    names=list(dict.fromkeys(n for n in names if n in report['models']))
    fig,axes=plt.subplots(1,2,figsize=(13,6.8))
    for name in names:
        entries=report['models'][name]['free_rollout']['per_episode']
        for ax,key in zip(axes,('position_rmse_m','endpoint_error_m')):
            x=np.sort([1000*r[key] for r in entries if r['finite']]);y=np.arange(1,len(x)+1)/len(entries)
            ax.plot(np.maximum(x,1e-6),y,label=labels.get(name,name),linewidth=2.5 if name==chosen else 1.3)
            ax.set_xscale('log');ax.grid(True,alpha=.25);ax.axhline(.95,color='gray',linestyle='--',linewidth=.8)
    axes[0].set_xlabel('Trajectory position RMSE (mm)');axes[1].set_xlabel('Endpoint error (mm)')
    for ax in axes:
        ax.set_ylabel('Fraction of fresh test episodes');ax.set_ylim(0,1.02)
        ax.set_xlim(.1,1000)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,fontsize=8,loc='lower center',ncol=2,bbox_to_anchor=(.5,.01))
    fig.subplots_adjust(bottom=.27,top=.90,wspace=.22)
    fig.suptitle('900 fresh Chrono launches: learned contact residuals and Transformer flight cores')
    fig.text(.5,.925,'Native 10ms samples. Axes show 0.1–1000mm; full out-of-range tails remain in comparison.csv.',ha='center',fontsize=9)
    fig.savefig(args.root/'comparison.png',dpi=180);plt.close(fig)
    print(json.dumps(dict(complete=True,selected=chosen,summary=str(args.root/'summary.json'))),flush=True)


if __name__=='__main__':main()
