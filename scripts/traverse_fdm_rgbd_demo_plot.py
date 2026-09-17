#!/usr/bin/env python3
"""Plot measured route-choice outcomes; this script never trains or selects routes."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def trajectory(path):
    with np.load(path) as data:
        poses = np.concatenate((data["pose"], data["terminal_pose"][None]), axis=0)
        dt = float(data["dt_s"])
    return poses, np.arange(len(poses))*dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.demo/"physical_comparison.json").read_text())
    fig, axes = plt.subplots(2,2,figsize=(12,8), constrained_layout=True)
    for col, (name, case_result) in enumerate(report["cases"].items()):
        case = json.loads((args.demo/"cases"/(name+".json")).read_text())
        goal = np.asarray(case["goal_xy"])
        selected, times = trajectory(args.demo/"physical_selected"/name/"trajectory.npz")
        outcome = json.loads((args.demo/"physical_selected"/name/"outcome.json").read_text())
        safe_families = [(k,v) for k,v in case_result["families"].items() if v["safe_goal_reached"]]
        best = min(safe_families,key=lambda kv:kv[1]["goal_time_s"])[0] if safe_families else None
        ax, progress_ax = axes[0,col],axes[1,col]
        for family, result in case_result["families"].items():
            poses, clock = trajectory(args.demo/"physical_families"/name/family/"trajectory.npz")
            color = "#98a2b3" if result["safe_goal_reached"] else "#e9a0a0"
            ax.plot(poses[:,0],poses[:,1],color=color,lw=1,alpha=.7)
            progress_ax.plot(clock,np.linalg.norm(poses[:,:2]-goal,axis=1),color=color,lw=1,alpha=.7)
        if best is not None:
            poses, clock = trajectory(args.demo/"physical_families"/name/best/"trajectory.npz")
            ax.plot(poses[:,0],poses[:,1],color="#169b62",lw=2,label="Fastest safe sampled route")
            progress_ax.plot(clock,np.linalg.norm(poses[:,:2]-goal,axis=1),color="#169b62",lw=2,label="Fastest safe sampled route")
        selected_color = "#2365b0" if outcome["safe_goal_reached"] else "#c0392b"
        ax.plot(selected[:,0],selected[:,1],color=selected_color,lw=2.5,label="RGB-D + MPPI selection")
        ax.scatter(*selected[0,:2],color="black",marker="o",s=28,zorder=5,label="Start")
        ax.scatter(*goal,color="#efab22",marker="*",s=160,zorder=5,label="Goal")
        for asset in case["layout"]["assets"]:
            if asset["kind"]=="rock":
                edge=asset["dims"]["edge_m"]
                ax.add_patch(Rectangle((asset["x_m"]-edge/2,asset["y_m"]-edge/2),edge,edge,color="#454545"))
        progress_ax.plot(times,np.linalg.norm(selected[:,:2]-goal,axis=1),color=selected_color,lw=2.5,label="RGB-D + MPPI selection")
        progress_ax.axhline(case["goal_radius_m"],color="#efab22",ls=":",lw=1.5,label="Goal tolerance")
        title = "Visible obstacle" if "rock" in name else "Hill crossing"
        result_text = (f"Reached safely in {outcome['goal_time_s']:.2f} s" if outcome["safe_goal_reached"]
                       else f"FAILED: contact and stopping; {outcome['goal_progress_m']:.2f} m progress")
        ax.set_title(title+"\n"+result_text,fontsize=12)
        ax.set_aspect("equal",adjustable="datalim");ax.set_xlabel("World x (m)");ax.set_ylabel("World y (m)")
        ax.grid(alpha=.2)
        progress_ax.set_xlabel("Actual execution time (s)");progress_ax.set_ylabel("Actual goal distance (m)")
        progress_ax.set_ylim(bottom=0);progress_ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc="best")
    fig.suptitle("First RGB-D pilot: measured Chrono outcomes\nFixed checkpoint and scoring settings; same starting state for every route",fontsize=14)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=170)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(args.out)


if __name__=="__main__":main()
