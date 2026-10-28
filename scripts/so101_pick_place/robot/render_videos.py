"""Robot test videos from the logged student input images (read-only on robot_logs).

Source per episode: ep_NN/episode_live.npz qry_thumbs [n_query, 2 (external, wrist), 112, 112, 3] uint8 RGB and qry_t (s).
These are the student's 224 px input images (external: crop 52,103,288,288 of the 640x480 frame; wrist: the full frame
squashed to a square), logged at 112 px at every policy query (about 7.8 Hz). No full-resolution recording exists.

Outputs (H.264, CRF 23, yuv420p):
  videos/stress_tb10/<session>_<ep>.mp4   one per stress episode, real time, 25 fps (each image held until the next query)
  videos/tb10_stress_20261027_4x.mp4      all 50 stress episodes at 4x with a title card each + a final card
  videos/screening_20261027_4x.mp4        the 10-27 screening (8 sessions x 4, 7 students) at 4x

    python3 render_videos.py [all|per_episode|stress_comp|screen_comp] --logs <Hub robot/logs> \
        --results <results.json of build_results.py> --out DIR       (needs ffmpeg with libx264)

Ported from the study's ~/so101_r2s_claude/release/robot/scripts/render_videos.py: the folders are arguments.
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np

LOGS = OUT = RESULTS = None        # set from the arguments
FPS = 25
UP = 3
PANEL = 112 * UP                         # 336
W = 2 * PANEL                            # 672
HDR = 78
H = PANEL + HDR                          # 414
FONT = cv2.FONT_HERSHEY_SIMPLEX


def encoder(path):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
           "-pix_fmt", "yuv420p", "-threads", "4", "-movflags", "+faststart", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def text(img, s, y, scale=0.48, color=(255, 255, 255), x=6, th=1):
    cv2.putText(img, s, (x, y), FONT, scale, color, th, cv2.LINE_AA)


def panels(th):
    ext = cv2.resize(th[0], (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    wr = cv2.resize(th[1], (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
    im = np.concatenate([ext, wr], 1)
    text(im, "external", PANEL - 8, 0.45, (255, 255, 0))
    text(im, "wrist", PANEL - 8, 0.45, (255, 255, 0), x=PANEL + 6)
    return im


def frame(th, lines, colors=None):
    f = np.zeros((H, W, 3), np.uint8)
    f[:HDR] = 30
    for i, s in enumerate(lines):
        text(f, s, 20 + 24 * i, color=(colors or {}).get(i, (255, 255, 255)))
    f[HDR:] = panels(th)
    return f


def card(lines, big=None):
    f = np.zeros((H, W, 3), np.uint8)
    f[:] = 20
    y = 60
    if big:
        sz = cv2.getTextSize(big, FONT, 1.6, 3)[0]
        text(f, big, y + 30, 1.6, (255, 255, 255), x=(W - sz[0]) // 2, th=3)
        y += 90
    for s in lines:
        text(f, s, y, 0.55)
        y += 28
    return f


def load_ep(sess, ep):
    z = np.load(f"{LOGS}/{sess}/{ep}/episode_live.npz", allow_pickle=False)
    return z["qry_thumbs"], z["qry_t"], float(z["t"][-1])


def episode_frames(th, qt, t_end, speed, header, place):
    """Frames at FPS; output frame k shows real time k * speed / FPS (latest image logged at or before it)."""
    n = int(np.floor(t_end * FPS / speed)) + 1
    for k in range(n):
        tk = k * speed / FPS
        i = max(int(np.searchsorted(qt, tk, side="right")) - 1, 0)
        yield frame(th[i], header(tk), {1: outcome_color(place)})


def short(sess):
    return sess.split("_")[-1]


def outcome_color(place):
    return (120, 255, 120) if place else (255, 120, 120)


def main(which):
    R = json.load(open(RESULTS))
    eps = R["episodes"]
    stud = R["students"]
    sizes = {}
    stress = [r for r in eps if r["group"] == "stress_tb10"]
    tb = stud["tb10"]
    tb_ck = "v12_tb10_s1210 it_01000, sha256 " + tb["student_sha256"][:8]

    if which in ("all", "per_episode"):
        os.makedirs(f"{OUT}/videos/stress_tb10", exist_ok=True)
        for r in stress:
            th, qt, t_end = load_ep(r["session"], r["ep"])
            sid = f"{short(r['session'])}/{r['ep']}"
            geo = (f"{r['dir']}  cube r {r['cube_r_mm']:.0f} mm {r['cube_deg']:+.0f} deg, "
                   f"plate r {r['plate_r_mm']:.0f} mm {r['plate_deg']:+.0f} deg")
            res = "PLACED" if r["place"] else "NOT PLACED"

            def hdr(tk, r=r, sid=sid, geo=geo, res=res):
                return [f"tb10 stress {sid} | {r['typed']} | t {tk:5.1f} s",
                        f"{res} | {geo}",
                        f"real SO101 2026-10-27, student {tb_ck}, 1x"]
            p = f"{OUT}/videos/stress_tb10/{r['session']}_{r['ep']}.mp4"
            enc = encoder(p)
            for f in episode_frames(th, qt, t_end, 1.0, hdr, r["place"]):
                enc.stdin.write(f.tobytes())
            enc.stdin.close()
            assert enc.wait() == 0, p
            sizes[p] = os.path.getsize(p)
            print(p, sizes[p], flush=True)

    if which in ("all", "stress_comp"):
        p = f"{OUT}/videos/tb10_stress_20261027_4x.mp4"
        enc = encoder(p)
        rc = R["stress_recount"]
        intro = card(["Real SO101 pick-and-place stress test, 2026-10-27 14:55-15:48",
                      f"Student tb10: {tb_ck}",
                      "Trained in simulation; bundle built 09:32, before the first test",
                      "50 policy episodes, 6 sessions; 4x speed",
                      "Images: the student's own camera inputs (logged at 112 px)",
                      "left: external camera (crop), right: wrist camera",
                      "Outcome typed by the person at the robot"], big="tb10 stress test")
        for _ in range(FPS * 4):
            enc.stdin.write(intro.tobytes())
        for j, r in enumerate(stress, 1):
            th, qt, t_end = load_ep(r["session"], r["ep"])
            sid = f"{short(r['session'])}/{r['ep']}"
            res = "PLACED" if r["place"] else "NOT PLACED"
            c = card([f"{r['dir']}: cube r {r['cube_r_mm']:.0f} mm {r['cube_deg']:+.0f} deg, "
                      f"plate r {r['plate_r_mm']:.0f} mm {r['plate_deg']:+.0f} deg",
                      f"typed: {r['typed']}",
                      f"result: {res}"], big=f"{j}/50  {sid}")
            for _ in range(int(FPS * 1.2)):
                enc.stdin.write(c.tobytes())

            def hdr(tk, r=r, sid=sid, res=res, j=j):
                return [f"tb10 stress {sid} | {r['typed']} | t {tk:5.1f} s",
                        f"{res} | {r['dir']} | episode {j}/50",
                        "real SO101 2026-10-27, 4x speed"]
            for f in episode_frames(th, qt, t_end, 4.0, hdr, r["place"]):
                enc.stdin.write(f.tobytes())
        bd = rc["by_dir"]
        fin = card([f"R->L (plate left of the cube): {bd['R->L']['placed']}/{bd['R->L']['n']}",
                    f"L->R: {bd['L->R']['placed']}/{bd['L->R']['n']}",
                    "Session 150916 ep_03 was stopped at HOME before",
                    "the policy ran: not counted",
                    f"Student {tb_ck}"], big=f"{rc['placed']}/{rc['n_episodes']} placed")
        for _ in range(FPS * 5):
            enc.stdin.write(fin.tobytes())
        enc.stdin.close()
        assert enc.wait() == 0
        sizes[p] = os.path.getsize(p)
        print(p, sizes[p], flush=True)

    if which in ("all", "screen_comp"):
        p = f"{OUT}/videos/screening_20261027_4x.mp4"
        enc = encoder(p)
        scr = [r for r in eps if r["group"] == "screening_20261027"]
        intro = card(["Real SO101, 2026-10-27 13:30-14:04",
                      "8 sessions x 4 episodes, 7 students (D at the start and the end)",
                      "Every session: ep 1-2 cube right, ep 3-4 cube left. 4x speed",
                      "Images: the student's own camera inputs (112 px logs)",
                      "left: external camera (crop), right: wrist camera",
                      "Outcome typed by the person at the robot"], big="Screening 10-27")
        for _ in range(FPS * 4):
            enc.stdin.write(intro.tobytes())
        for j, r in enumerate(scr, 1):
            th, qt, t_end = load_ep(r["session"], r["ep"])
            sid = f"{short(r['session'])}/{r['ep']}"
            res = "PLACED" if r["place"] else "NOT PLACED"
            ck = stud[r["student"]]["student_ckpt_local"].split("/ckpt/")[-1]
            c = card([f"student {r['student']} ({ck}, sha256 {r['student_sha8']})",
                      f"typed: {r['typed']}", f"result: {res}"], big=f"{j}/32  {sid}")
            for _ in range(int(FPS * 1.2)):
                enc.stdin.write(c.tobytes())

            def hdr(tk, r=r, sid=sid, res=res, j=j):
                return [f"{r['student']} screening {sid} | {r['typed']} | t {tk:5.1f} s",
                        f"{res} | episode {j}/32",
                        "real SO101 2026-10-27, 4x speed"]
            for f in episode_frames(th, qt, t_end, 4.0, hdr, r["place"]):
                enc.stdin.write(f.tobytes())
        ps = R["per_student"]
        lines = [f"{lab}: {ps[lab]['screening_20261027']['placed']}/{ps[lab]['screening_20261027']['n']}"
                 for lab in ps]
        a = R["screening_all"]
        fin = card(lines + [f"all: {a['placed']}/{a['n']}"], big="Screening places")
        for _ in range(FPS * 5):
            enc.stdin.write(fin.tobytes())
        enc.stdin.close()
        assert enc.wait() == 0
        sizes[p] = os.path.getsize(p)
        print(p, sizes[p], flush=True)



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", nargs="?", default="all", choices=("all", "per_episode", "stress_comp", "screen_comp"))
    ap.add_argument("--logs", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True, help="the videos go to <out>/videos/")
    a = ap.parse_args()
    LOGS, RESULTS, OUT = a.logs, a.results, a.out
    main(a.which)
