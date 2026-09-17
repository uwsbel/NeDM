"""Write the README that ships with the 15 demo videos."""
import json, os
import numpy as np

ROOT = 'artifacts/traverse/fdm_f104_50h_20260909'
OUT = ROOT + '/night2_v1/demo_v1'
VID = 'artifacts/f104_demo_v1'
ROLES = ('optimal', 'suboptimal', 'risky')
LAB = {'optimal': "planner's choice", 'suboptimal': 'middling alternative', 'risky': 'worst-ranked'}


def main():
    meta = {(m['scenario'], m['role']): m for m in json.load(open(OUT + '/video_meta.json'))}
    scen = sorted({k[0] for k in meta})
    L = ["# Five scenarios, three routes each (f104 arena, night-2 planner)", "",
         "Each scenario is one held-out start/goal pair on the hill-and-crater test set. For each, the planner",
         "generated 256 candidate routes, scored every one with the deployed 5-seed risk model, and three were",
         "then driven in Chrono: the model's best, a middling one, and the model's worst. Same model, same",
         "candidate pool, three ranks -- so the videos are a direct read on whether the predicted ordering",
         "matches the simulator.", "",
         "Every filmed run reproduced its screening run exactly (status and elapsed time to 0.05 s).", "",
         "| scenario | route | predicted risk | Chrono outcome | time | max tilt | backwards |",
         "|---|---|---|---|---|---|---|"]
    for s in scen:
        for r in ROLES:
            m = meta[(s, r)]
            out = m['status'].replace('_', ' ')
            L.append(f"| {s} | {LAB[r]} | {100*m['risk']:.2f}% | {out} | {m['elapsed']:.1f} s | "
                     f"{m['max_tilt']:.0f} deg | {m['back_s']:.1f} s |")
    L += ["", "## Files", "",
          "`scenarios.png`  the five scenarios on the terrain map: dashed = proposed route, solid = driven path.", ""]
    for s in scen:
        for i, r in enumerate(ROLES):
            L.append(f"`s{s}_{i+1}_{r}.mp4`")
    L += ["", "The green/amber/red ribbon on the ground in each video is the route the planner asked for; the",
          "yellow disc is the 2.5 m goal circle. Clips of stalled runs are cut at 35 s -- the vehicle was still",
          "stuck when the episode was terminated.", "",
          "## How these were produced", "",
          "1. `scripts/f104_demo_pick.py` -- score the cached 256-candidate pool of 28 hazard groups with",
          "   `night2_v1/final/N2_s*.pt` and stage best / middling / worst per group.",
          "2. Cluster screening (`demo_night2_v1`, 84 episodes): drive all three per group with the frozen",
          "   collector `source_v1/scripts/collect_traverse_f104.py`; all three arms of a group share one array",
          "   task because Chrono is only deterministic per node.",
          "3. `scripts/f104_demo_select.py` -- keep the five scenarios where the trio tells three different",
          "   stories, spread over the arena.",
          "4. Cluster re-run with Chrono's Blender exporter attached to the collector's own observer",
          "   (`demo_blend_v1`), which reproduces the recorded status and elapsed time exactly.",
          "5. `scripts/f104_demo_render_blender.py` + `f104_demo_videos.py` -- render locally with Blender",
          "   (EEVEE) and encode with ffmpeg captions.", "",
          "## Caveats", "",
          "- One arena. Every route in these videos was driven during training as part of some other episode;",
          "  'held out' here means the start/goal pair, not the terrain.",
          "- The model is a good ranker but badly calibrated: across the wider hazard test the median predicted",
          "  risk was 0.008% against a realised 0.33%. Read the numbers as an ordering, not a probability.",
          "- The routes are chosen once, before the run. There is no online replanning."]
    os.makedirs(VID, exist_ok=True)
    open(VID + '/README.md', 'w').write('\n'.join(L) + '\n')
    print('wrote', VID + '/README.md')


if __name__ == '__main__':
    main()
