"""Iteration video and miss-vs-iteration figure for ball targeting (side view).

For the first targets, each saved optimiser iteration (every 5) was replayed
in Chrono by ball_verify; the video animates those replays one iteration after
another, with the target, Chrono's miss at t and the model's predicted
distance. All ball positions in the video are Chrono's, not the model's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
try:  # AMD compute nodes have no system ffmpeg; use the imageio-ffmpeg binary.
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except ImportError:
    pass
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--method", default="lm")
    parser.add_argument("--targets", type=int, default=10)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()
    report = json.loads((args.run / "optimization.json").read_text())
    traces = json.loads((args.run / "chrono_video_traces.json").read_text())
    scene = report["physics_config"]["scene"]
    R = scene["radius_m"]
    method = args.method
    method_name = {"lm": "Levenberg-Marquardt", "gd": "gradient descent"}.get(method, method)
    snaps = report["results"][method]["snapshots"]
    iterations = sorted({int(k) for target in traces[method].values() for k in target})
    n = min(args.targets, len(report["targets"]))
    t_end = report["target_time_s"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i in range(n):
        it = [s["iteration"] for s in snaps]
        nrd = [max(s["nrd_distance_m"][i] * 1000, 1e-4) for s in snaps]
        chrono = [traces[method][str(i)][f"{k:03d}"]["miss_m"] * 1000 for k in it if f"{k:03d}" in traces[method][str(i)]]
        axes[0].semilogy(it, nrd, color=f"C{i}", lw=1.2)
        axes[1].semilogy(it[: len(chrono)], chrono, color=f"C{i}", lw=1.2, marker="o", ms=3)
    for ax, title in zip(axes, ("model-predicted distance to target at t", "Chrono replay: ball's miss at t")):
        ax.axhline(3, color="k", ls="--", lw=0.8)
        ax.set(xlabel=f"{method_name} iteration", ylabel="mm", title=title)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{n} fresh targets, {method_name} on gradients of the unified NRD rollout; dashed line = 3 mm")
    fig.tight_layout()
    fig.savefig(args.run / f"iteration_vs_miss_{method}.png", dpi=130)
    plt.close(fig)

    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.9 * cols, 2.4 * rows + 0.5))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.9, bottom=0.01, wspace=0.03, hspace=0.06)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.axis("off")
    title = fig.suptitle("")
    wall = scene["wall_front_x_m"]
    artists = []
    for i in range(n):
        ax = axes[i]
        ax.axhline(scene["ground_z_m"], color="#795548", lw=3)
        ax.axvline(wall, color="#795548", lw=3)
        ax.set_xlim(-1.5, wall + 0.3)
        ax.set_ylim(-0.2, 6.0)
        ax.set_xticks([])
        ax.set_yticks([])
        tx, tz = report["targets"][i]["xy_m"]
        ax.add_patch(plt.Circle((tx, tz), 0.12, fill=False, ec="#e65100", lw=1.5, ls="--", zorder=3))
        path, = ax.plot([], [], color="#1565c0", lw=1.0)
        ball = plt.Circle((0, 0), R, color="#d32f2f", zorder=4)
        ax.add_patch(ball)
        label = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", fontsize=8,
                        bbox=dict(fc="white", alpha=0.7, lw=0))
        artists.append((path, ball, label))
    frames_per_iter = 50
    writer = FFMpegWriter(fps=args.fps, bitrate=2400)
    out = args.run / f"iterations_chrono_{method}.mp4"
    with writer.saving(fig, str(out), dpi=100):
        for k in iterations:
            for f in range(frames_per_iter + 15):
                frac = min(1.0, f / frames_per_iter)
                for i in range(n):
                    entry = traces[method][str(i)].get(f"{k:03d}")
                    if entry is None:
                        continue
                    trace = np.asarray(entry["trace_10ms"])  # [T, 2] = x, z
                    m = max(1, int(round(frac * (len(trace) - 1))) + 1)
                    path, ball, label = artists[i]
                    path.set_data(trace[:m, 0], trace[:m, 1])
                    ball.center = tuple(trace[m - 1])
                    snap = next(s for s in snaps if s["iteration"] == k)
                    label.set_text(f"target {i + 1}\nChrono miss {entry['miss_m'] * 1000:.1f} mm\n"
                                   f"model {snap['nrd_distance_m'][i] * 1000:.1f} mm")
                title.set_text(f"{method_name} iteration {k}: Chrono replay of the current launch, t = {frac * t_end:.2f} s of {t_end:.1f} s"
                               "   (target ring orange, 12 cm; ball drawn to scale)")
                writer.grab_frame()
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
