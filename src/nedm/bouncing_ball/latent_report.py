"""Scientific comparison from frozen certification and physical replay files."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nedm.bouncing_ball.collection import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);args=p.parse_args()
    root=args.root
    data=json.loads((root/"certification/paired_results.json").read_text())
    selection=json.loads((root/"selection.json").read_text())
    rows=[]
    for name,result in data["models"].items():
        motion=result["free_rollout"];contact=result.get("teacher_forced",{}).get("contact")
        rows.append({"name":name,"p95_trajectory_rmse_mm":motion["position_rmse_m"]["p95"]*1000,
            "p95_endpoint_mm":motion["endpoint_error_m"]["p95"]*1000,"bounce_order_fraction":motion["contact_order_fraction"],
            "bounce_order_failures":motion["failure_count"],"rmse_over3mm_failures":motion["trajectory_accuracy_failure_count_3mm"],
            "teacher_ground_wall_f1":[c["f1"] for c in contact] if contact else None})
    targets={}
    for name in ("launch_learned_contact","launch_learned_contact_fixed_restarts"):
        folder=root/"runs"/name
        optimization=json.loads((folder/"optimization.json").read_text())
        physical=json.loads((folder/"chrono_verification.json").read_text())
        targets[name]={"all_passed":physical["passed"],"nrd_passed":optimization["nrd_passed"],"iterations":optimization["iterations"],
            "nrd_target_miss_mm":[c["optimized_nrd_distance_m"]*1000 for c in optimization["cases"]],
            "physical_target_miss_mm":[c["optimized"]["target_distance_m"]*1000 for c in physical["cases"]],
            "simulator_feedback_used":physical["simulator_feedback_used_by_optimizer"]}
    fig,axes=plt.subplots(1,3,figsize=(18,6.5),constrained_layout=True)
    main_name=selection["chosen"]["stage1"]["candidate"]
    for name,label,color in (("original_transformer","Original NRD","#6b6b6b"),
                             ("analytical_contact_precision","Analytical contact benchmark","#24788b"),
                             (main_name,"Learned contact and phase","#c96b16")):
        entries=data["models"][name]["free_rollout"]["per_episode"]
        for ax,key,title in ((axes[0],"position_rmse_m","Whole-trajectory position RMSE"),
                             (axes[1],"endpoint_error_m","Last common 50 ms endpoint")):
            values=np.sort([e[key]*1000 for e in entries if e["finite"]])
            ax.plot(values,np.arange(1,len(values)+1)/len(values),label=label,color=color,lw=2)
            ax.set_xscale("log");ax.set(title=title,xlabel="Position error (mm)",ylabel="Fraction of 900 fresh episodes",ylim=(0,1.01))
            ax.axhline(.95,color=".55",lw=.8,ls=":");ax.grid(alpha=.15,which="both")
    axes[0].legend(loc="lower right",fontsize=9)
    keys=["analytical_contact_precision",main_name,"original_transformer","state5_binary_cpu_native", "state5_phase_cpu_soft",
          "xz_phase_cpu_native","unified_gated_cpu_native","unified_aux_cpu_native","unified_noaux_cpu_native"]
    names=["Analytical contact benchmark","Learned contact + phase","Original NRD transformer","Binary label + midpoint",
           "Soft contact + learned phase","x,z-only latent (diverged)","One 5D decoder + learned gate","One 5D decoder + contact loss","One 5D decoder, no contact loss"]
    values=np.array([data["models"][k]["free_rollout"]["position_rmse_m"]["p95"]*1000 for k in keys])
    colors=["#24788b","#c96b16","#6b6b6b","#ad4141","#ad4141","#ad4141","#735997","#735997","#735997"]
    axes[2].barh(np.arange(len(keys)),np.minimum(values,2000),color=colors,alpha=.9)
    axes[2].set_yticks(np.arange(len(keys)),names,fontsize=9);axes[2].invert_yaxis();axes[2].set_xscale("log")
    axes[2].set(xlabel="p95 trajectory RMSE (mm)",title="Contact labels do not replace timing or dynamics",xlim=(.4,4000))
    for i,value in enumerate(values):
        text="diverged" if value>2000 else f"{value:.2f}" if value<10 else f"{value:.1f}"
        axes[2].text(min(value,2000)*1.1,i,text,va="center",fontsize=8)
    axes[2].grid(alpha=.15,axis="x",which="both")
    fig.suptitle("State-only contact latent study — fixed Chrono scene, frozen models, new seed 202610013",fontsize=15)
    out=root/"reports";out.mkdir(exist_ok=False)
    fig.savefig(out/"latent_contact_comparison.png",dpi=160,bbox_inches="tight");plt.close(fig)
    atomic_json(out/"summary.json",{"fresh_test_episodes":data["fresh_test_episodes"],"fresh_seed":data["fresh_seed"],
        "selection_before_test":True,"comparison_dt_s":.05,"full_raw_dense_dt_s":.0005,"models":rows,
        "raw_stage1":{k:v for k,v in data["raw_dense_stage1"].items() if k!="per_episode"},"targets":targets,
        "checkpoints":{k:{kk:vv for kk,vv in v.items() if kk!="validation"} for k,v in selection["chosen"].items()},
        "certification_sha256":hashlib.sha256((root/"certification/paired_results.json").read_bytes()).hexdigest(),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({"figure":str(out/"latent_contact_comparison.png"),"summary":str(out/"summary.json")}),flush=True)


if __name__=="__main__":main()
