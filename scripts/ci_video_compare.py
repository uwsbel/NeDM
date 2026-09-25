#!/usr/bin/env python
"""Synchronised top-down comparison videos of recorded Chrono drives (no re-simulation).

For one case of the CRM-improvement study (a soil or a rigid start-goal pair) the six arms listed in
``artifacts/traverse/crm_improve_20260922/videos/manifest.json`` are drawn side by side in a 2 x 3 grid, all on the
same simulated clock from the start of the drive.  Every mark comes from the recorded files: ``trajectory.npz``
(pose x, y, yaw and the body-forward speed, one row per 50 ms control step, plus the terminal pose), ``outcome.json``
(status, elapsed time, branch block) and the branch route JSON the drive followed after the decision (checked by
content hash against the hash the collector wrote into the outcome).

The map view is rotated so the start sits on the left and the goal on the right (the straight start-goal line is
horizontal); the rotation is a rigid one, so distances and turns are preserved.  The terrain is the arena heightmap
(``nedm.traverse.terrain.TerrainMap``) resampled on the rotated grid: hillshade plus a tint over cells steeper than
15 degrees (slope from the same gradient the planner study uses).

Output (per case): ``compare_<case>.mp4`` (real time, 20 fps = one frame per recorded step), ``compare_<case>_2x.mp4``
(every other step at 20 fps), both holding the last frame for 2 s, ``compare_<case>_final.png`` (the last frame) and
``compare_<case>_checks.json`` (what was verified about the inputs).

Usage::

    PYTHONPATH=src:scripts python scripts/ci_video_compare.py --case soil
    PYTHONPATH=src:scripts python scripts/ci_video_compare.py --case rigid
    PYTHONPATH=src:scripts python scripts/ci_video_compare.py --case soil2     # any manifest key
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LightSource, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

import gc_control  # noqa: E402  (numpy-only; provides the collector's route content hash)
from nedm.traverse.terrain import TerrainMap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VID = ROOT / "artifacts/traverse/crm_improve_20260922/videos"
DT = 0.05
FPS = 20
HOLD_S = 2.0
W_PX, H_PX, DPI = 1600, 1000, 100

SURF = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
APPROACH = "#8d8b86"
STEEP = "#c9ad72"
FAIL = "#d03b3b"   # status: critical
GOOD = "#0ca30c"   # status: good
# Categorical slots 1-4 of the house palette plus two extra hues; the six-colour set was run through the dataviz
# validator with --pairs all (worst CVD dE 9.2, worst normal-vision dE 15.5; aqua and yellow need the visible labels
# the cell titles and the legend provide).
ARM_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e0a800", "#9e2a6b"]
# legend labels; a manifest arm's "short_label" overrides (the 3 s arm drives the previous study's CNN-GRU, so its
# manifest entry names that, and its "footnote" is drawn under the map key)
SHORT = {"H3s": "CNN-GRU (earlier), 3 s approach (old)", "L1_Hn": "CNN-GRU, 1 s", "L1_X": "Transformer, 1 s",
         "L0p5_Hn": "CNN-GRU, 0.5 s", "L0p5_X": "Transformer, 0.5 s", "HnG": "CNN-GRU + gradient, 0.5 s"}
WORLD = {"crm": "Soft soil (deformable ground)", "rigid": "Rigid ground"}
VEH_L, VEH_W = 4.7, 2.2
STEEP_DEG = 15.0
STEEP_A = 0.42
CONTOUR_M = 1.0
MARGIN_M = 4.5

plt.rcParams.update({"font.family": ["Lato", "DejaVu Sans"], "font.size": 11, "text.color": INK,
                     "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def fmt_t(t: float) -> str:
    """Times are multiples of the 50 ms step: print them exactly (two decimals, trailing zero dropped)."""
    s = f"{round(t / DT) * DT:.2f}"
    return s[:-1] if s.endswith("0") else s


def stuck_onset(vx: np.ndarray) -> float:
    """Start of the final stretch below 0.3 m/s (the speed the no-progress rules call stopped), in s."""
    i = len(vx)
    while i > 0 and abs(vx[i - 1]) < 0.3:
        i -= 1
    return i * DT


def outcome_text(status: str, t_end: float, stuck_from: float | None = None) -> tuple[bool, str]:
    if status == "goal_reached":
        return True, f"goal reached in {fmt_t(t_end)} s"
    if status == "soil_breakthrough_terminated":
        return False, f"bogged down at {fmt_t(t_end)} s: wheels dug through the soil"
    if status == "prolonged_blockage_terminated":
        since = f" from {fmt_t(stuck_from)} s" if stuck_from is not None else ""
        return False, f"stuck{since}; drive ended at {fmt_t(t_end)} s (no progress)"
    return False, f"did not reach the goal ({status.replace('_', ' ')}) at {fmt_t(t_end)} s"


def load_arm(a: dict, case: dict) -> dict:
    d = ROOT / a["run_dir"]
    z = np.load(d / "trajectory.npz")
    o = json.loads((d / "outcome.json").read_text())
    n = len(z["pose"])
    F = int(z["branch_frame"])
    assert str(z["state_fields"][0]) == "vel_body_x_mps", z["state_fields"][0]
    assert F == int(a["branch_frame"]) and abs(F * DT - a["approach_s"]) < 1e-9, (F, a)
    assert abs(n * DT - float(o["elapsed_s"])) < 1e-6 and o["status"] == a["status"], (n, o["elapsed_s"], o["status"])
    br = json.loads(Path(a["branch_route"]).read_text())
    blk = o.get("branch") or {}
    want = blk.get("branch_route_content_sha256") or o.get("branch_route_sha256")
    got = gc_control.route_sha256(br)
    assert got == want, f"{a['arm']}: branch route content hash {got} != recorded {want}"
    bpose = np.asarray(blk.get("branch_pose", o.get("branch_pose")), float)
    pose = np.vstack([z["pose"], np.asarray(z["terminal_pose"], float)[None]])
    vx = np.concatenate([z["state"][:, 0].astype(float), [float(z["terminal_state"][0])]])
    assert np.allclose(bpose[:2], pose[F, :2], atol=1e-9), "branch pose != recorded pose at the decision frame"
    route = np.asarray(br["waypoints"], float)
    ap = np.asarray(json.loads(Path(a["approach_route"]).read_text())["waypoints"], float)
    start = np.asarray(case["layout"]["start_xy"], float)
    e = (ap[-1] - ap[0]) / np.linalg.norm(ap[-1] - ap[0])
    rel = pose[:F + 1, :2] - ap[0]
    approach_dev = float(np.abs(rel[:, 0] * e[1] - rel[:, 1] * e[0]).max())
    goal = np.asarray(case["goal_xy"], float)
    stuck = stuck_onset(vx) if o["status"] == "prolonged_blockage_terminated" else None
    ok, txt = outcome_text(o["status"], n * DT, stuck)
    short = a.get("short_label", SHORT[a["arm"]])
    check = {"arm": a["arm"], "label": a["label"], "legend_label": short, "footnote": a.get("footnote"),
             "status": o["status"], "elapsed_s": float(o["elapsed_s"]),
             "frames": n, "decision_frame": F, "decision_time_s": F * DT,
             "branch_route_content_sha256_matches_outcome": True, "branch_pose_equals_recorded_pose": True,
             "route_start_to_decision_pose_m": float(np.linalg.norm(route[0] - pose[F, :2])),
             "approach_route_start_to_case_start_m": float(np.linalg.norm(ap[0] - start)),
             "approach_max_lateral_deviation_m": approach_dev,
             "final_goal_distance_m_recorded": float(o["final_goal_distance_m"]),
             "final_goal_distance_m_from_terminal_pose": float(np.linalg.norm(pose[-1, :2] - goal)),
             "vx_max_mps": float(vx.max()), "on_screen_outcome": txt}
    if stuck is not None:
        check["stuck_from_s"] = stuck
        check["movement_after_stuck_from_m"] = float(np.linalg.norm(pose[round(stuck / DT):, :2] - pose[round(stuck / DT), :2],
                                                                    axis=1).max())
    return dict(arm=a["arm"], label=a["label"], short=short, footnote=a.get("footnote"), status=o["status"], n=n, F=F,
                pose=pose, vx=vx, route=route,
                ok=ok, outcome=txt, check=check, stuck_from=stuck)


class Frame:
    """Rigid rotation of world xy into screen coordinates: x along start->goal, y to its left."""

    def __init__(self, start, goal):
        self.s = np.asarray(start, float)
        g = np.asarray(goal, float)
        self.dist = float(np.linalg.norm(g - self.s))
        self.u = (g - self.s) / self.dist
        self.v = np.array([-self.u[1], self.u[0]])
        self.psi = math.atan2(self.u[1], self.u[0])

    def fwd(self, xy):
        q = np.asarray(xy, float) - self.s
        return np.stack([q @ self.u, q @ self.v], -1)

    def inv(self, X, Y):
        return self.s[0] + X * self.u[0] + Y * self.v[0], self.s[1] + X * self.u[1] + Y * self.v[1]


def smooth(a: np.ndarray, sigma_px: float) -> np.ndarray:
    """Separable Gaussian blur with edge padding (display only: hides the 8-bit height quantisation ripples)."""
    r = int(math.ceil(3 * sigma_px))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_px) ** 2)
    k /= k.sum()
    b = np.pad(a, r, mode="edge")
    b = np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 0, b)
    return np.apply_along_axis(lambda x: np.convolve(x, k, mode="valid"), 1, b)


def terrain_rgb(tmap: TerrainMap, fr: Frame, ext, res=0.08):
    x0, x1, y0, y1 = ext
    nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    X, Y = np.meshgrid(x0 + (np.arange(nx) + 0.5) * (x1 - x0) / nx, y0 + (np.arange(ny) + 0.5) * (y1 - y0) / ny)
    wx, wy = fr.inv(X, Y)
    H = tmap.height(wx, wy)
    slope = np.degrees(np.arctan(tmap.slope(wx, wy)))
    ls = LightSource(azdeg=315, altdeg=42)
    Hs = smooth(H, 0.25 / res)
    shade = np.flipud(ls.hillshade(np.flipud(Hs), vert_exag=2.2, dx=res, dy=res))  # row 0 = bottom of the screen
    base = np.asarray(to_rgb(SURF))
    rgb = base[None, None] * (0.60 + 0.40 * shade[..., None])
    steep = slope > STEEP_DEG
    rgb[steep] = (1 - STEEP_A) * rgb[steep] + STEEP_A * np.asarray(to_rgb(STEEP))
    return np.clip(rgb, 0, 1), float(steep.mean()), (X, Y, Hs)


def vehicle_poly(fr: Frame, pose):
    """4.7 m x 2.2 m footprint centred on the chassis reference point, front corners chamfered so the heading reads."""
    X, Y = fr.fwd(pose[:2])
    h = pose[2] - fr.psi
    c, s = math.cos(h), math.sin(h)
    hl, hw, ch = VEH_L / 2, VEH_W / 2, 0.55
    loc = np.array([[hl, 0.0], [hl - ch, hw], [-hl, hw], [-hl, -hw], [hl - ch, -hw]])
    return np.stack([X + c * loc[:, 0] - s * loc[:, 1], Y + s * loc[:, 0] + c * loc[:, 1]], -1)


def climb_check(arms: list[dict], tmap: TerrainMap, start) -> dict:
    """Where the failed drives stopped: height above the start and ground slope under the terminal pose."""
    h0 = float(tmap.height(float(start[0]), float(start[1])))
    out = {}
    for a in arms:
        if a["ok"]:
            continue
        x, y = (float(v) for v in a["pose"][-1, :2])
        out[a["arm"]] = {"status": a["status"], "height_above_start_m": float(tmap.height(x, y)) - h0,
                         "slope_deg": float(np.degrees(np.arctan(tmap.slope(x, y))))}
    return out


def headline(world: str, arms: list[dict], climb: dict) -> str:
    by = {a["arm"]: a for a in arms}
    t = {a["arm"]: a["n"] * DT for a in arms}
    failed = {a["arm"] for a in arms if not a["ok"]}
    bogged = all(by[k]["status"] == "soil_breakthrough_terminated" for k in failed)
    # "on the climb" only when every failed drive stopped >= 1 m above the start on ground steeper than 15 degrees
    on_climb = bool(climb) and all(c["height_above_start_m"] >= 1.0 and c["slope_deg"] > STEEP_DEG for c in climb.values())
    where = " on the climb" if on_climb else ""
    if world == "crm" and failed == {"H3s", "L1_Hn"} and bogged:
        good = [t[k] for k in by if k not in failed]
        return (f"The old 3 s approach and the 1 s CNN-GRU bog down{where}; the 1 s transformer and every 0.5 s "
                f"planner get through ({fmt_t(min(good))}-{fmt_t(max(good))} s).")
    if world == "crm" and failed == {"H3s"}:
        good = [t[k] for k in by if k not in failed]
        h = by["H3s"]
        verb = {"soil_breakthrough_terminated": "bogs down",
                "prolonged_blockage_terminated": f"gets stuck {fmt_t((h['stuck_from'] or 0) - h['F'] * DT)} s after its "
                                                 "decision and stays stuck"}.get(h["status"], "fails")
        return (f"The old 3 s approach {verb}{where}; every planner that decides after 1 s or 0.5 s gets through "
                f"({fmt_t(min(good))}-{fmt_t(max(good))} s).")
    if world == "rigid" and not failed:
        tx = sorted([t["L1_X"], t["L0p5_X"]])
        th = sorted([t["L1_Hn"], t["L0p5_Hn"], t["HnG"]])
        return (f"Every planner arrives: the transformers in {fmt_t(tx[0])} and {fmt_t(tx[1])} s, the retrained CNN-GRU "
                f"planners in {fmt_t(th[0])}-{fmt_t(th[-1])} s, the old 3 s protocol in {fmt_t(t['H3s'])} s.")
    ok = [a for a in arms if a["ok"]]
    return f"{len(ok)} of {len(arms)} planners reach the goal from the same start."


def build(key: str, manifest: dict):
    C = manifest[key]
    case = json.loads(Path(C["case"]).read_text())
    arms = [load_arm(a, case) for a in C["arms"]]
    for a, col in zip(arms, ARM_COLOURS):
        a["col"] = col
    start = np.asarray(case["layout"]["start_xy"], float)
    goal = np.asarray(case["goal_xy"], float)
    R = float(case.get("goal_radius_m", 2.5))
    fr = Frame(start, goal)
    tmap = TerrainMap.from_dir(Path(C["arena"]))
    climb = climb_check(arms, tmap, start)

    fig = plt.figure(figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI)
    L, Rr, gapx = 0.028, 0.985, 0.018
    cw = (Rr - L - 2 * gapx) / 3
    map_h = 0.264
    map_bottoms = [0.573, 0.252]
    axes = []
    for i in range(6):
        r, c = divmod(i, 3)
        axes.append(fig.add_axes([L + c * (cw + gapx), map_bottoms[r], cw, map_h]))
    # common crop: start, goal circle, every driven track and planned route, plus a margin; then match the cell aspect
    pts = [fr.fwd(start)[None], fr.fwd(goal)[None] + np.array([[R, R], [-R, -R]])]
    for a in arms:
        pts += [fr.fwd(a["pose"][:, :2]), fr.fwd(a["route"])]
    P = np.vstack(pts)
    x0, y0 = P.min(0) - MARGIN_M
    x1, y1 = P.max(0) + MARGIN_M
    x0 -= 1.0  # room for the "start" label behind the vehicle
    bb = axes[0].get_position()
    aspect = (bb.width * W_PX) / (bb.height * H_PX)
    if (x1 - x0) / (y1 - y0) < aspect:
        cx, half = 0.5 * (x0 + x1), 0.5 * (y1 - y0) * aspect
        x0, x1 = cx - half, cx + half
    else:
        cy, half = 0.5 * (y0 + y1), 0.5 * (x1 - x0) / aspect
        y0, y1 = cy - half, cy + half
    ext = (x0, x1, y0, y1)
    rgb, steep_frac, (TX, TY, TH) = terrain_rgb(tmap, fr, ext)
    levels = np.arange(math.floor(TH.min() / CONTOUR_M) + 1, math.ceil(TH.max() / CONTOUR_M)) * CONTOUR_M

    halo = [pe.Stroke(linewidth=4.4, foreground=SURF, alpha=0.85), pe.Normal()]
    halo_thin = [pe.Stroke(linewidth=3.4, foreground=SURF, alpha=0.85), pe.Normal()]
    text_halo = [pe.withStroke(linewidth=2.6, foreground=SURF, alpha=0.9)]
    anim = []  # (artist) drawn every frame in this order
    cells = []
    goal_side = {}
    late = []   # artists drawn after every other animated artist (goal labels a stopped vehicle would cover)
    gx, gy = fr.fwd(goal)
    renderer = fig.canvas.get_renderer()
    title_fs = 11.5
    for a in arms:
        probe = fig.text(0, 0, a["label"], fontsize=title_fs, fontweight="semibold")
        wpx = probe.get_window_extent(renderer).width
        probe.remove()
        room = (cw - 0.013) * W_PX - 6
        if wpx > room:
            title_fs = math.floor(title_fs * room / wpx * 4) / 4
    for ax, a in zip(axes, arms):
        ax.imshow(rgb, origin="lower", extent=ext, interpolation="bilinear", zorder=0)
        if len(levels):
            ax.contour(TX, TY, TH, levels=levels, colors=INK2, linewidths=0.55, alpha=0.38, zorder=1)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.add_patch(Circle((gx, gy), R, fill=False, ec=INK, lw=1.3, zorder=3))
        # "goal" above the circle, unless the drive's last pose covers that spot: then below it; if both are covered,
        # the label is drawn on top of the vehicle
        ax.apply_aspect()
        vp = vehicle_poly(fr, a["pose"][-1])
        vlo, vhi = vp.min(0) - 0.3, vp.max(0) + 0.3
        glab = ax.text(gx, gy + R + 0.6, "goal", ha="center", va="bottom", fontsize=9, color=INK2, zorder=3,
                       path_effects=text_halo)
        side = None
        for name, yy, va in (("above", gy + R + 0.6, "bottom"), ("below", gy - R - 0.6, "top")):
            glab.set_y(yy)
            glab.set_va(va)
            bb = ax.transData.inverted().transform(glab.get_window_extent(renderer).get_points())
            if bb[1, 0] < vlo[0] or bb[0, 0] > vhi[0] or bb[1, 1] < vlo[1] or bb[0, 1] > vhi[1]:
                side = name
                break
        if side is None:
            glab.set_y(gy + R + 0.6)
            glab.set_va("bottom")
            glab.set_animated(True)
            late.append(glab)
            side = "above, drawn over the vehicle"
        goal_side[a["arm"]] = side
        ax.text(-VEH_L / 2 - 0.6, 0.0, "start", ha="right", va="center", fontsize=9, color=INK2, zorder=3,
                path_effects=text_halo)
        sb_x, sb_y = x0 + 1.2, y0 + 1.2
        ax.plot([sb_x, sb_x + 10], [sb_y, sb_y], color=INK2, lw=2.0, solid_capstyle="butt", zorder=3)
        ax.text(sb_x + 5, sb_y + 0.5, "10 m", ha="center", va="bottom", fontsize=8.5, color=INK2, zorder=3,
                path_effects=text_halo)
        bb = ax.get_position()
        fig.text(bb.x0, bb.y1 + 0.039, "■", color=a["col"], fontsize=13, va="center", ha="left")
        fig.text(bb.x0 + 0.013, bb.y1 + 0.039, a["label"], color=INK, fontsize=title_fs, va="center", ha="left",
                 fontweight="semibold")
        goal_fill = Circle((gx, gy), R, fc=a["col"], ec="none", alpha=0.22, zorder=2, visible=False, animated=True)
        ax.add_patch(goal_fill)
        ln_app, = ax.plot([], [], color=APPROACH, lw=2.4, solid_capstyle="round", zorder=4, animated=True,
                          path_effects=halo)
        ln_route, = ax.plot([], [], color=a["col"], lw=1.7, ls=(0, (4.0, 2.6)), zorder=5, animated=True,
                            path_effects=halo_thin)
        ln_trk, = ax.plot([], [], color=a["col"], lw=2.4, solid_capstyle="round", zorder=6, animated=True,
                          path_effects=halo)
        dia, = ax.plot([], [], "D", ms=7.5, mfc=a["col"], mec=INK, mew=1.1, zorder=8, animated=True)
        veh = Polygon(vehicle_poly(fr, a["pose"][0]), closed=True, fc=a["col"], ec=INK, lw=1.1, zorder=9,
                      animated=True, joinstyle="round")
        ax.add_patch(veh)
        sym = fig.text(bb.x0, bb.y1 + 0.016, "", fontsize=11, va="center", ha="left", animated=True)
        st = fig.text(bb.x0, bb.y1 + 0.016, "", fontsize=10.5, color=INK2, va="center", ha="left", animated=True)
        anim += [goal_fill, ln_app, ln_route, ln_trk, veh, dia, sym, st]  # diamond above the vehicle: it decides early
        XY = fr.fwd(a["pose"][:, :2])
        cells.append(dict(a=a, XY=XY, RT=fr.fwd(a["route"]), goal_fill=goal_fill, ln_app=ln_app, ln_route=ln_route,
                          ln_trk=ln_trk, dia=dia, veh=veh, sym=sym, st=st, bb=bb))

    # header and map key
    fig.text(L, 0.973, f"{WORLD[C['world']]}: one start and goal, six planners, the recorded drives side by side",
             fontsize=17, fontweight="bold", color=INK, va="center")
    fig.text(L, 0.944, headline(C["world"], arms, climb), fontsize=12.5, color=INK2, va="center")
    # one plain line per distinct note, between the map key and the cell titles (the layout has room for one)
    notes = list(dict.fromkeys(a["footnote"] for a in arms if a.get("footnote")))
    assert len(notes) <= 1, notes
    for j, t in enumerate(notes):
        fig.text(L, 0.897 - 0.019 * j, t, fontsize=10, color=INK2, va="center", ha="left")
    play = fig.text(Rr, 0.973, "", fontsize=10.5, color=INK2, ha="right", va="center", animated=True)
    anim.append(play)
    kax = fig.add_axes([L, 0.910, Rr - L, 0.018])
    kax.set_xlim(0, 100)
    kax.set_ylim(0, 1)
    kax.axis("off")
    kx = 0.0

    def key_item(draw, text, w):
        nonlocal kx
        draw(kx)
        kax.text(kx + 2.6, 0.5, text, fontsize=10, color=INK2, va="center")
        kx += w

    key_item(lambda x: kax.plot([x, x + 2.0], [0.5, 0.5], color=APPROACH, lw=2.4), "straight approach", 11.6)
    key_item(lambda x: kax.plot([x + 1.0], [0.5], "D", ms=7, mfc=INK2, mec=INK, mew=1.0), "decision point", 10.2)
    key_item(lambda x: kax.plot([x, x + 2.0], [0.5, 0.5], color=INK2, lw=1.7, ls=(0, (3.0, 2.0))),
             "route planned at the decision", 17.2)
    key_item(lambda x: kax.plot([x, x + 2.0], [0.5, 0.5], color=INK2, lw=2.4), "driven track", 9.4)
    key_item(lambda x: kax.add_patch(Rectangle((x + 0.2, 0.05), 1.6, 0.9, fc=STEEP, ec="none", alpha=0.8)),
             "ground steeper than 15°", 14.2)
    key_item(lambda x: kax.plot([x, x + 2.0], [0.5, 0.5], color=INK2, lw=0.7, alpha=0.6),
             f"{CONTOUR_M:g} m height contours", 13.4)
    any_fail = not all(a["ok"] for a in arms)
    if any_fail:
        key_item(lambda x: kax.add_patch(Rectangle((x + 0.1, 0.1), 1.8, 0.8, fc=SURF, ec=FAIL, lw=2.4)),
                 "red outline: drive ended short of the goal", 0.0)

    # speed strip, with a thin band above it holding one decision tick per planner (one row each, grid order)
    sax = fig.add_axes([0.058, 0.058, 0.715, 0.116])
    dax = fig.add_axes([0.058, 0.180, 0.715, 0.052], sharex=sax)
    T_end = max(a["n"] for a in arms) * DT
    vmax = max(float(a["vx"].max()) for a in arms)
    sax.set_xlim(0, T_end + 0.4)
    sax.set_ylim(-0.3, vmax * 1.12)
    sax.set_yticks(np.arange(0, math.floor(vmax) + 1, 2 if vmax > 5 else 1))
    sax.grid(axis="y", color=GRID, lw=0.8)
    sax.set_axisbelow(True)
    for sp in ("top", "right"):
        sax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        sax.spines[sp].set_color(INK2)
    sax.tick_params(labelsize=9.5, length=3)
    sax.set_xlabel("time since the start of the drive (s)", fontsize=10)
    sax.set_ylabel("forward speed (m/s)", fontsize=10)
    dax.set_ylim(0, len(arms))
    dax.set_yticks([])
    dax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for sp in ("top", "right", "left"):
        dax.spines[sp].set_visible(False)
    dax.spines["bottom"].set_color(GRID)
    fig.text(0.054, 0.206, "decides", fontsize=9.5, color=INK2, ha="right", va="center")
    strip = []
    for i, a in enumerate(arms):
        tt = np.arange(a["n"] + 1) * DT
        sax.plot(tt, a["vx"], color=a["col"], lw=1.1, alpha=0.22, zorder=2)
        row = len(arms) - 1 - i
        dax.plot([a["F"] * DT] * 2, [row + 0.04, row + 0.96], color=a["col"], lw=2.6, solid_capstyle="butt")
        ln, = sax.plot([], [], color=a["col"], lw=1.9, zorder=4, animated=True)
        endm, = sax.plot([], [], "o" if a["ok"] else "X", ms=7 if a["ok"] else 8.5,
                         mfc=a["col"] if a["ok"] else FAIL, mec=SURF, mew=1.2, zorder=6, animated=True)
        strip.append((a, tt, ln, endm))
        anim += [ln, endm]
    # the caption is drawn after the moving time cursor, on an opaque box, so the cursor stops at its edge instead of
    # running through the letters
    cap = dax.text(max(a["F"] for a in arms) * DT + 0.25, len(arms) / 2, "\u2190 the moment each planner picks its route "
                   "(0.5 s, 1 s or 3 s after the start)", fontsize=9.5, color=INK2, va="center", ha="left", zorder=9,
                   animated=True, bbox=dict(boxstyle="square,pad=0.25", fc=SURF, ec="none"))
    cur = sax.axvline(0, color=INK, lw=1.0, zorder=7, animated=True)
    cur_d = dax.axvline(0, color=INK, lw=1.0, zorder=7, animated=True)
    cur_t = sax.text(0, vmax * 1.12, "", fontsize=9.5, color=INK, ha="left", va="top", zorder=8, animated=True,
                     path_effects=text_halo)
    anim += [cur, cur_d, cap, cur_t] + late
    handles = [Line2D([], [], color=a["col"], lw=2.2) for a in arms]
    handles += [Line2D([], [], ls="none", marker="o", ms=7, mfc=INK2, mec=SURF)]
    labels = [a["short"] for a in arms] + ["goal reached"]
    if any_fail:
        handles += [Line2D([], [], ls="none", marker="X", ms=8, mfc=FAIL, mec=SURF)]
        labels += ["drive ended short of the goal"]
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.79, 0.045), frameon=False, fontsize=9.5,
               labelcolor=INK2, handlelength=1.8, ncol=1, borderaxespad=0.0, labelspacing=0.32)
    info = dict(fig=fig, anim=anim, cells=cells, strip=strip, cur=cur, cur_d=cur_d, cur_t=cur_t, play=play, fr=fr, arms=arms,
                ext=ext, steep_frac=steep_frac, K=max(a["n"] for a in arms), vmax=vmax, case=case, C=C, climb=climb,
                headline=headline(C["world"], arms, climb), goal_label_side=goal_side, title_fontsize=title_fs,
                footnotes=notes)
    return info


def update(info, k: int, play_text: str):
    fr = info["fr"]
    t = k * DT
    for c in info["cells"]:
        a = c["a"]
        kk = min(k, a["n"])
        F = a["F"]
        XY = c["XY"]
        ia = min(kk, F)
        c["ln_app"].set_data(XY[:ia + 1, 0], XY[:ia + 1, 1])
        if kk >= F:
            c["ln_trk"].set_data(XY[F:kk + 1, 0], XY[F:kk + 1, 1])
            c["ln_route"].set_data(c["RT"][:, 0], c["RT"][:, 1])
            c["dia"].set_data([XY[F, 0]], [XY[F, 1]])
        else:
            c["ln_trk"].set_data([], [])
            c["ln_route"].set_data([], [])
            c["dia"].set_data([], [])
        c["veh"].set_xy(vehicle_poly(fr, a["pose"][kk]))
        done = k >= a["n"]
        if done and not a["ok"]:
            c["veh"].set_edgecolor(FAIL)
            c["veh"].set_linewidth(3.2)
        else:
            c["veh"].set_edgecolor(INK)
            c["veh"].set_linewidth(1.1)
        c["goal_fill"].set_visible(done and a["ok"])
        if done:
            c["sym"].set_text("✓" if a["ok"] else "✕")
            c["sym"].set_color(GOOD if a["ok"] else FAIL)
            c["st"].set_text(a["outcome"])
            c["st"].set_color(INK)
            c["st"].set_x(c["bb"].x0 + 0.013)
        else:
            v = a["vx"][kk]
            v = 0.0 if abs(v) < 0.05 else v
            c["sym"].set_text("")
            c["st"].set_text(f"{fmt_t(t)} s  ·  speed {v:.1f} m/s")
            c["st"].set_color(INK2)
            c["st"].set_x(c["bb"].x0)
    for a, tt, ln, endm in info["strip"]:
        kk = min(k, a["n"])
        ln.set_data(tt[:kk + 1], a["vx"][:kk + 1])
        if k >= a["n"]:
            endm.set_data([tt[-1]], [a["vx"][-1]])
        else:
            endm.set_data([], [])
    info["cur"].set_xdata([t, t])
    info["cur_d"].set_xdata([t, t])
    right = t > 0.8 * info["K"] * DT
    info["cur_t"].set_position((t - 0.12 if right else t + 0.12, info["vmax"] * 1.12))
    info["cur_t"].set_horizontalalignment("right" if right else "left")
    info["cur_t"].set_text(f"{fmt_t(t)} s")
    info["play"].set_text(play_text)


def grab(info, bg, k, play_text):
    fig = info["fig"]
    update(info, k, play_text)
    fig.canvas.restore_region(bg)
    for art in info["anim"]:
        fig.draw_artist(art)
    return bytes(fig.canvas.buffer_rgba())


def ffmpeg(path: Path):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                             "-s", f"{W_PX}x{H_PX}", "-framerate", str(FPS), "-i", "-", "-c:v", "libx264",
                             "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                             str(path)], stdin=subprocess.PIPE)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--case", required=True, help="manifest key: soil, rigid, soil2, ...")
    ap.add_argument("--manifest", default=str(VID / "manifest.json"))
    ap.add_argument("--still-only", action="store_true", help="write only the last-frame PNG (layout check)")
    ap.add_argument("--still-frame", type=int, default=None, help="with --still-only: frame index to draw")
    args = ap.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text())
    info = build(args.case, manifest)
    fig = info["fig"]
    fig.canvas.draw()
    bg = fig.canvas.copy_from_bbox(fig.bbox)
    K = info["K"]
    stem = VID / f"compare_{args.case}"
    if args.still_only:
        k = K if args.still_frame is None else args.still_frame
        buf = grab(info, bg, k, "")
        Image.frombuffer("RGBA", (W_PX, H_PX), buf).convert("RGB").save(f"{stem}_check_{k}.png")
        print(f"{stem}_check_{k}.png")
        return 0
    hold = int(round(HOLD_S * FPS))
    ks2 = list(range(0, K + 1, 2))
    if ks2[-1] != K:
        ks2.append(K)
    p1, p2 = ffmpeg(Path(f"{stem}.mp4")), ffmpeg(Path(f"{stem}_2x.mp4"))
    n1 = n2 = 0
    last1 = last2 = None
    for k in range(K + 1):
        last1 = grab(info, bg, k, "playback: real time")
        p1.stdin.write(last1)
        n1 += 1
        if k in ks2:
            last2 = grab(info, bg, k, "playback: 2x real time")
            p2.stdin.write(last2)
            n2 += 1
    for _ in range(hold):
        p1.stdin.write(last1)
        p2.stdin.write(last2)
    n1 += hold
    n2 += hold
    for p in (p1, p2):
        p.stdin.close()
        assert p.wait() == 0
    buf = grab(info, bg, K, "")
    Image.frombuffer("RGBA", (W_PX, H_PX), buf).convert("RGB").save(f"{stem}_final.png")
    checks = {"case": args.case, "group": info["C"]["group"], "world": info["C"]["world"],
              "source": "recorded trajectory.npz / outcome.json / branch route JSON; no re-simulation",
              "view": "rigid rotation: screen x along start->goal, screen y to its left; same crop in every cell",
              "crop_screen_m": [round(float(v), 3) for v in info["ext"]],
              "steep_threshold_deg": STEEP_DEG, "steep_fraction_of_crop": info["steep_frac"],
              "headline": info["headline"], "failed_drive_end_terrain": info["climb"],
              "goal_label_side": info["goal_label_side"], "cell_title_fontsize": info["title_fontsize"],
              "footnotes": info["footnotes"], "frames_real_time": n1, "frames_2x": n2, "fps": FPS,
              "last_sim_frame": K, "hold_frames": hold, "arms": [a["check"] for a in info["arms"]]}
    Path(f"{stem}_checks.json").write_text(json.dumps(checks, indent=1))
    print(json.dumps({"real_time_frames": n1, "x2_frames": n2, "K": K}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
