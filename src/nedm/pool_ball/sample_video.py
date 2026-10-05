"""Top-down video of sample Chrono shots from a packed campaign (physics check).

Picks shots spread over speed and cut angle and animates both balls in real
time (25 fps), with their paths and the contact events listed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from nedm.pool_ball.evaluate import cut_angle_deg
from nedm.pool_ball.video import table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=6)
    args = parser.parse_args()
    index = json.loads((args.data / "campaign_index.json").read_text())
    config, scene = index["config"], index["config"]["scene"]
    R = scene["radius_m"]
    with np.load(args.data / "model_data.npz") as packet:
        states, polar = packet["states"], packet["polar"]
    cuts = cut_angle_deg(config, polar[:, 1])
    wanted = [(1.6, 0), (2.9, 0), (2.2, 30), (2.6, -45), (1.8, -60), (2.9, 60), (2.4, 15), (2.0, -20)][: args.shots]
    picks = [int(np.argmin((polar[:, 0] - u) ** 2 + ((cuts - c) / 30) ** 2)) for u, c in wanted]
    cols = 3
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.1 * rows + 0.5))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(picks):]:
        ax.axis("off")
    art = []
    for ax, k in zip(axes, picks):
        table(ax, scene)
        e = index["episodes"][k]
        events = ", ".join(f"{x['pair']}@{x['time_s']:.2f}s" for x in e["events"] if x["kind"] == "start")
        ax.set_title(f"v={polar[k, 0]:.2f} m/s, cut {cuts[k]:+.0f}°\n{events}", fontsize=7)
        a_line, = ax.plot([], [], color="white", lw=0.8, alpha=0.7)
        b_line, = ax.plot([], [], color="#ffcc00", lw=1.0)
        a_ball, b_ball = plt.Circle((0, 0), R, color="white", zorder=4), plt.Circle((0, 0), R, color="#d32f2f", zorder=4)
        ax.add_patch(a_ball)
        ax.add_patch(b_ball)
        art.append((k, a_line, b_line, a_ball, b_ball))
    title = fig.suptitle("")
    record = index["record_step_s"]
    stride = round(0.04 / record)
    writer = FFMpegWriter(fps=25, bitrate=2400)
    with writer.saving(fig, str(args.output), dpi=100):
        for m in range(0, states.shape[1], stride):
            for k, a_line, b_line, a_ball, b_ball in art:
                s = states[k, : m + 1]
                a_line.set_data(s[:, 0], s[:, 1])
                b_line.set_data(s[:, 7], s[:, 8])
                a_ball.center = (s[-1, 0], s[-1, 1])
                b_ball.center = (s[-1, 7], s[-1, 8])
            title.set_text(f"Chrono shots (A white, B red), t = {m * record:.2f} s, real time")
            writer.grab_frame()
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
