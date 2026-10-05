"""Publish measured precision comparison figures from saved AMD evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from nedm.bouncing_ball.collection import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline-run',type=Path,required=True)
    args=p.parse_args()
    root=args.root
    report=json.loads((root/'runs/certified_v2/certification.json').read_text())
    choice=json.loads((root/'selection.json').read_text())
    run=root/'runs/launch_precision_v1'
    old=json.loads((args.baseline_run/'chrono_verification.json').read_text())
    new=json.loads((run/'chrono_verification.json').read_text())
    optimization=json.loads((run/'optimization.json').read_text())
    assert new['passed'] and optimization['nrd_passed']
    cases=[]
    for a,b,c in zip(old['cases'],new['cases'],optimization['cases'],strict=True):
        assert a['target_xz_m']==b['target_xz_m']==c['xz_m']
        cases.append(dict(name=c['name'],target_xz_m=c['xz_m'],velocity_mps=c['optimized_velocity_mps'],
                          original_target_miss_mm=a['optimized']['target_distance_m']*1000,
                          new_target_miss_mm=b['optimized']['target_distance_m']*1000,
                          target_miss_reduction_factor=a['optimized']['target_distance_m']/b['optimized']['target_distance_m'],
                          original_model_gap_mm=a['nrd_chrono_endpoint_gap_m']*1000,
                          new_model_gap_mm=b['nrd_chrono_endpoint_gap_m']*1000,
                          new_nrd_miss_mm=c['optimized_nrd_distance_m']*1000))
    fresh=report['comparisons']['fresh_test']
    probe=json.loads((root/'contact_normal_probe.json').read_text())
    normal={}
    for kind in ('ground','wall'):
        normals=[x['normal'] for e in probe['episodes'] for v in e['events'] for x in v['contacts'] if v['kind']==kind]
        axes=[0,1] if kind=='ground' else [1,2]
        normal[kind]={'max_off_axis_normal_magnitude':max(math.sqrt(sum(n[j]**2 for j in axes)) for n in normals),'contacts':len(normals)}
    result=dict(complete=True,checkpoint_sha256=report['checkpoint_sha256'],selected=choice['selected_name'],
                fresh_test_episodes=fresh['candidate']['episodes'],trajectory_p95_reduction_factor=fresh['trajectory_p95_reduction_factor'],
                endpoint_p95_reduction_factor=fresh['endpoint_p95_reduction_factor'],
                original_trajectory_p95_mm=fresh['baseline']['position_rmse_m']['p95']*1000,
                new_trajectory_p95_mm=fresh['candidate']['position_rmse_m']['p95']*1000,
                original_endpoint_p95_mm=fresh['baseline']['endpoint_error_m']['p95']*1000,
                new_endpoint_p95_mm=fresh['candidate']['endpoint_error_m']['p95']*1000,
                dense_maximum_position_error_mm=report['dense_fresh_test']['maximum_position_error_m']['max']*1000,
                dense_position_rmse_p95_mm=report['dense_fresh_test']['position_rmse_m']['p95']*1000,
                native_step_s=report['native_dt_s'],contact_physics_step_s=.000125,
                candidates_compared=len(choice['candidates']),cases=cases,contact_normal_diagnostic=normal,
                contact_normal_interpretation='Small observed normal perturbations could help explain remaining submillimeter-to-millimeter error; this is not a proven irreducible lower bound.',
                target_comparison_note='Same five targets; NRD optimization tolerance tightened from 5 mm to 0.05 mm. The trajectory comparison uses identical initial conditions and a matched 100 Hz grid.',
                research_sources=[{'title':'Learning Neural Event Functions for Ordinary Differential Equations','url':'https://arxiv.org/abs/2011.03902'},
                                  {'title':'Official differentiable bouncing-ball example','url':'https://github.com/rtqichen/torchdiffeq/blob/master/examples/bouncing_ball.py'},
                                  {'title':'Learning to Simulate Complex Physics with Graph Networks','url':'https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html'},
                                  {'title':'Chrono 10 NSC contact source','url':'https://github.com/projectchrono/chrono/blob/10.0.0/src/chrono/physics/ChContactNSC.cpp'}])
    out=root/'reports'
    out.mkdir(exist_ok=True)
    atomic_json(out/'precision_summary.json',result)
    plt.rcParams.update({'font.size':11})
    fig,axes=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for ax,key,title in zip(axes[:2],['position_rmse_m','endpoint_error_m'],['Whole-trajectory position RMSE','End-of-episode position error'],strict=True):
        for role,label,color in [('baseline','Original NRD','#bd4c25'),('candidate','Precision NRD','#176b91')]:
            values=np.sort([e[key]*1000 for e in fresh[role]['per_episode']])
            ax.plot(values,np.arange(1,len(values)+1)/len(values),label=label,color=color,lw=2)
        ax.set(xlabel='Error (mm)',ylabel='Fraction of 900 fresh episodes',title=title,xscale='log',ylim=(0,1.02))
        ax.grid(alpha=.2);ax.legend(loc='lower right')
    x=np.arange(5)
    axes[2].bar(x-.18,[c['original_target_miss_mm'] for c in cases],.36,label='Original targeting',color='#bd4c25')
    axes[2].bar(x+.18,[c['new_target_miss_mm'] for c in cases],.36,label='Precision targeting',color='#176b91')
    axes[2].set(xticks=x,xticklabels=[str(i+1) for i in x],xlabel='Target',ylabel='Physical endpoint miss (mm)',yscale='log',title='Chrono targeting at 1.7 s')
    axes[2].legend(loc='upper right');axes[2].grid(axis='y',alpha=.2)
    fig.suptitle(f"Chrono–NRD precision: {result['trajectory_p95_reduction_factor']:.1f}× lower trajectory p95 error",fontsize=18)
    fig.savefig(out/'precision_comparison.png',dpi=150)
    plt.close(fig)
    gradients=json.loads((run/'gradient_checks.json').read_text())
    atomic_json(out/'launch_gradient_checks.json',gradients)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
