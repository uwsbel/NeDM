"""Robot evidence tables for the SO101 pick-and-place release (read-only on robot_logs).

Reads the raw session logs (session.json + ep_*/episode_live.{json,npz}) and writes results.json / results.md.
Student identity = the bundle folder in ep_*/episode_live.json setup.kin_points (/home/sbel/<bundle>/assets/...);
the checkpoint and sha256 of each bundle come from the local copy of that bundle (MANIFEST.txt, assets/student.pt).
Place = session.json episodes[].place (the "p" letter typed by the person at the robot); re-checked against the typed
code and against ep_*/episode_live.json outcome.

    python3 build_results.py --logs <Hub robot/logs> --stress <Hub results/robot/stress_episodes.json> --out DIR \
        [--bundles DIR]

Ported from the study's ~/so101_r2s_claude/release/robot/scripts/build_results.py: the folders are arguments. Only
the tb10 bundle is released (bundle/so101_hw_v13_tb10/ on the Hub), so the bundle line, checkpoint and student.pt
sha256 of each robot bundle come from the record below (= the study's results.json "students"); with --bundles DIR
(a folder that holds <bundle>/MANIFEST.txt and <bundle>/assets/student.pt) a bundle found there is read and checked
against the record. The output equals the released results.json / results.md except the source paths it names.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

LOGS = OUT = STRESS_JSON = BUNDLE_DIR = None        # set by main() from the arguments

STRESS = ["eval_20261027_145520", "eval_20261027_150916", "eval_20261027_151136", "eval_20261027_151953",
          "eval_20261027_153045", "eval_20261027_153943"]
SCREEN = ["eval_20261027_133039", "eval_20261027_133521", "eval_20261027_133953", "eval_20261027_134411",
          "eval_20261027_134903", "eval_20261027_135259", "eval_20261027_135657", "eval_20261027_140036"]

# bundle folder on the robot computer -> (label, record of the study's local bundle copy: MANIFEST.txt line 1, line 2
# checkpoint, sha256 of assets/student.pt)
BUNDLES = {
    "so101_hw_v8_roll801": ("D", dict(
        built="built 2026-10-24T03:58:36-05:00 from /home/harry/.t3/worktrees/NeDM-push-ppo/r2s-v8 (git ddb08332d 0 changed files under the copied paths)",
        student_ckpt_local="/home/harry/so101_r2s_claude/ckpt/v8/v8_roll_s801/it_02000.pt",
        student_sha256="d9a35bfc23abde45472e05b1be3ed0de90108d17c3484e31511a669be5ab7a1b")),
    "so101_hw_v13_tb10": ("tb10", dict(
        built="built 2026-10-27T09:32:46-05:00 from /home/harry/.t3/worktrees/NeDM-push-ppo/r2s-v12-vis (git 2b6dc4d8d 0 changed files under the copied paths)",
        student_ckpt_local="/home/harry/so101_r2s_claude/ckpt/v12/students/v12_tb10_s1210/it_01000.pt",
        student_sha256="6458a4f791d1052148bb8f7621413259342a7018757668fde244cfc098f99f5c")),
    "so101_hw_v13_visR500": ("visR500", dict(
        built="built 2026-10-27T09:32:40-05:00 from /home/harry/.t3/worktrees/NeDM-push-ppo/r2s-v12-vis (git 2b6dc4d8d 0 changed files under the copied paths)",
        student_ckpt_local="/home/harry/so101_r2s_claude/ckpt/v12/students/v12_visR_s1231/it_00500.pt",
        student_sha256="516eb659f6720066212a2ed54ab8e6805928fdd66f8bf2634349f1dd0144280b")),
    "so101_hw_v13q_visR_s1304_it750": ("visR_s1304_it750", dict(
        built="built 2026-10-27T12:11:44-05:00 from /home/harry/.t3/worktrees/NeDM-push-ppo/r2s-v12-vis (git 2b6dc4d8d 0 changed files under the copied paths)",
        student_ckpt_local="/home/harry/so101_r2s_claude/ckpt/v13/quick/v13_visR_s1304/it_00750.pt",
        student_sha256="7485a7088b3be705399ea28c73f3d26163892ea50e9ebe477b7af2e2f4e45efa")),
    "so101_hw_v13q_visRtb05_s1301_it500": ("visRtb05_s1301_it500", dict(
        built="built 2026-10-27T11:56:00-05:00 from /home/harry/.t3/worktrees/NeDM-push-ppo/r2s-v12-vis (git 2b6dc4d8d 0 changed files under the copied paths)",
        student_ckpt_local="/home/harry/so101_r2s_claude/ckpt/v13/quick/v13_visRtb05_s1301/it_00500.pt",
        student_sha256="3886e4b90e18d24acefda6a3d3c1595263f7a1f54ca964cfa60a94b6ac152798")),
    "so101_hw_v13_tb05": ("tb05", dict(
        built="built 2026-10-27T09:32:44-05:00 from /home/harry/.t3/worktrees/NeDM-push-ppo/r2s-v12-vis (git 2b6dc4d8d 0 changed files under the copied paths)",
        student_ckpt_local="/home/harry/so101_r2s_claude/ckpt/v12/students/v12_tb05_s1205/it_01500.pt",
        student_sha256="2dea35124ca7a701c475fbfac68ab6f0d719804ff0ba1a836017f0c9cab4fd49")),
    "so101_hw_v13_visR0_250": ("visR0_250", dict(
        built="built 2026-10-27T09:32:50-05:00 from /home/harry/.t3/worktrees/NeDM-push-ppo/r2s-v12-vis (git 2b6dc4d8d 0 changed files under the copied paths)",
        student_ckpt_local="/home/harry/so101_r2s_claude/ckpt/v13vis/visR0_s1231/it_00250.pt",
        student_sha256="85a1983beb3840211d79e87c7dbd72cfd605714042507ec392e2d9f3d69f9e16")),
}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def bundle_info(name):
    label, rec = BUNDLES[name]
    path = f"{BUNDLE_DIR}/{name}" if BUNDLE_DIR else None
    if path is None or not os.path.exists(f"{path}/MANIFEST.txt"):
        return dict(label=label, bundle=name, local_bundle=None, **rec)
    man = open(f"{path}/MANIFEST.txt").read().splitlines()
    ck = man[1].split()[1] if man[1].startswith("student ") else None
    out = dict(label=label, bundle=name, local_bundle=path, built=man[0], student_ckpt_local=ck,
               student_sha256=sha256(f"{path}/assets/student.pt"))
    if out["student_sha256"] != rec["student_sha256"]:
        raise SystemExit(f"{path}: student.pt sha256 {out['student_sha256'][:16]} is not the recorded "
                         f"{rec['student_sha256'][:16]}")
    return out


def session_student(sess):
    names = set()
    for e in sorted(glob.glob(f"{LOGS}/{sess}/ep_*/episode_live.json")):
        kp = json.load(open(e))["setup"]["kin_points"]
        names.add(kp.split("/")[3])
    assert len(names) == 1, (sess, names)
    return names.pop()


def all_d_sessions():
    out = []
    for d in sorted(glob.glob(f"{LOGS}/eval_2026102[4-7]_*")):
        s = os.path.basename(d)
        if not os.path.exists(f"{d}/session.json"):
            continue
        if session_student(s) == "so101_hw_v8_roll801":
            out.append(s)
    return out


def parse_code(typed):
    code = (typed or "").strip().partition(" ")[0].lower()
    return dict(grasp="g" in code, lift="l" in code, place="p" in code)


def episodes(sess, group, binfo, stress_idx):
    S = json.load(open(f"{LOGS}/{sess}/session.json"))
    rows, excluded = [], []
    listed = {e["episode"] for e in S["episodes"]}
    for epdir in sorted(glob.glob(f"{LOGS}/{sess}/ep_*")):
        k = int(epdir.rsplit("_", 1)[1])
        J = json.load(open(f"{epdir}/episode_live.json"))
        if k not in listed:
            excluded.append(dict(session=sess, ep=f"ep_{k:02d}", end_reason=J.get("end_reason"),
                                 outcome=J.get("outcome"), episode_start=J.get("episode_start"),
                                 why="not in session.json episodes[]; no policy step was run"))
    for e in S["episodes"]:
        k = e["episode"]
        epdir = f"{LOGS}/{sess}/ep_{k:02d}"
        J = json.load(open(f"{epdir}/episode_live.json"))
        z = np.load(f"{epdir}/episode_live.npz", allow_pickle=False)
        t = z["t"]
        qt = z["qry_t"]
        pc = parse_code(e.get("typed"))
        o = J.get("outcome") or {}
        r = dict(session=sess, ep=f"ep_{k:02d}", group=group, student=binfo["label"], bundle=binfo["bundle"],
                 student_sha8=binfo["student_sha256"][:8], start=e.get("start"), typed=e.get("typed"),
                 grasp=e.get("grasp"), lift=e.get("lift"), place=int(bool(e.get("place"))),
                 end_reason=e.get("end_reason"), duration_s=round(float(t[-1]), 2), n_steps=int(len(t)),
                 n_queries=int(len(qt)),
                 check_typed_code=(pc["place"] == bool(e.get("place")) and pc["grasp"] == bool(e.get("grasp"))
                                   and pc["lift"] == bool(e.get("lift"))),
                 check_episode_json=(o.get("place") == e.get("place") and o.get("typed") == e.get("typed")))
        sid = f"{sess.split('_')[-1]}/ep_{k:02d}"
        if stress_idx is not None:
            s = stress_idx.get(sid)
            if s is None:
                r["stress_row"] = None
            else:
                r.update(dir=s["dir"], cube_side=s["side"], cube_r_mm=round(s["cube_r"], 1),
                         cube_deg=round(s["cube_deg"], 1), plate_r_mm=round(s["plate_r"], 1),
                         plate_deg=round(s["plate_deg"], 1),
                         end_to_plate_mm=(None if s.get("end_to_plate") is None else round(s["end_to_plate"], 1)),
                         fail_class=s.get("fail"),
                         plate_minus_cube_y_mm=round(s["plate_r"] * math.sin(math.radians(s["plate_deg"]))
                                                     - s["cube_r"] * math.sin(math.radians(s["cube_deg"])), 1),
                         check_stress_json=(bool(s["placed"]) == bool(e.get("place")) and s["typed"] == e.get("typed")))
        rows.append(r)
    return rows, excluded


def tally(rows):
    k, n = sum(r["place"] for r in rows), len(rows)
    lo, hi = wilson(k, n)
    return dict(placed=k, n=n, rate=round(k / n, 4) if n else None, wilson95=[round(lo, 4), round(hi, 4)])


def main():
    global LOGS, OUT, STRESS_JSON, BUNDLE_DIR
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--logs", required=True, help="robot/logs of the Hub folder (one folder per session)")
    p.add_argument("--stress", required=True, help="results/robot/stress_episodes.json of the Hub folder")
    p.add_argument("--out", required=True, help="folder for results.json and results.md")
    p.add_argument("--bundles", default="", help="optional folder with local bundle copies (checked against the record)")
    a = p.parse_args()
    LOGS, OUT, STRESS_JSON, BUNDLE_DIR = a.logs, a.out, a.stress, a.bundles or None
    os.makedirs(OUT, exist_ok=True)
    stress_list = json.load(open(STRESS_JSON))
    sidx = {s["id"]: s for s in stress_list}
    binfo = {}
    rows, excluded = [], []
    sessions = {}
    d_sessions = all_d_sessions()
    plan = [(s, "stress_tb10") for s in STRESS] + [(s, "screening_20261027") for s in SCREEN] + \
           [(s, "D_20261024_26") for s in d_sessions if s not in SCREEN]
    for sess, group in plan:
        bname = session_student(sess)
        if bname not in binfo:
            binfo[bname] = bundle_info(bname)
        rr, ex = episodes(sess, group, binfo[bname], sidx if group == "stress_tb10" else None)
        rows += rr
        excluded += ex
        sessions[sess] = dict(group=group, student=binfo[bname]["label"], bundle=bname, n_episodes=len(rr),
                              placed=sum(r["place"] for r in rr), start=json.load(open(f"{LOGS}/{sess}/session.json"))["start"])

    stress = [r for r in rows if r["group"] == "stress_tb10"]
    # independent recount (raw session.json only, no stress_episodes.json)
    raw_place = 0
    raw_n = 0
    for s in STRESS:
        S = json.load(open(f"{LOGS}/{s}/session.json"))
        raw_n += len(S["episodes"])
        raw_place += sum(1 for e in S["episodes"] if e.get("place") is True)
    by_dir = {}
    for d in ("R->L", "L->R"):
        by_dir[d] = tally([r for r in stress if r.get("dir") == d])
    stress_ids = {f"{r['session'].split('_')[-1]}/{r['ep']}" for r in stress}
    recount = dict(
        source="robot_logs/<session>/session.json episodes[].place for the 6 stress sessions",
        n_episodes=raw_n, placed=raw_place, report_claim="36/50", agrees=(raw_place == 36 and raw_n == 50),
        typed_code_agrees_all=all(r["check_typed_code"] for r in stress),
        episode_json_agrees_all=all(r["check_episode_json"] for r in stress),
        stress_json_agrees_all=all(r.get("check_stress_json") for r in stress),
        stress_json_ids_match=(stress_ids == set(sidx)),
        dir_matches_lateral_y=sum((r["plate_minus_cube_y_mm"] > 0) == (r["dir"] == "R->L") for r in stress),
        dir_near_ambiguous_lt15mm=[f"{r['session']}/{r['ep']}" for r in stress if abs(r["plate_minus_cube_y_mm"]) < 15],
        by_dir=by_dir,
        by_session={s: sessions[s]["placed"] for s in STRESS},
    )

    screen = [r for r in rows if r["group"] == "screening_20261027"]
    d_all = [r for r in rows if r["student"] == "D"]
    per_student = {}
    for lab in [v[0] for v in BUNDLES.values()]:
        per_student[lab] = dict(
            screening_20261027=tally([r for r in screen if r["student"] == lab]),
        )
    per_student["tb10"]["stress_20261027"] = tally(stress)
    per_student["tb10"]["stress_by_dir"] = by_dir
    per_student["tb10"]["all_20261027"] = tally([r for r in rows if r["student"] == "tb10"])
    per_student["D"]["all_20261024_27"] = tally(d_all)
    per_student["D"]["by_day"] = {day: tally([r for r in d_all if r["session"].startswith(f"eval_{day}")])
                                  for day in ("20261024", "20261025", "20261026", "20261027")}

    res = dict(
        built_by="scripts/so101_pick_place/robot/build_results.py", logs_root=LOGS, stress_meta=STRESS_JSON,
        students={b["label"]: {k: v for k, v in b.items() if k != "label"} for b in binfo.values()},
        sessions=sessions, excluded=excluded, stress_recount=recount, per_student=per_student,
        screening_all=tally(screen), episodes=rows)
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)

    # ---- markdown
    L = []
    L.append("# SO101 robot results (from the raw logs)\n")
    L.append(f"Source: `{LOGS}/<session>/session.json` (typed outcome) and "
             "`ep_NN/episode_live.{json,npz}`. Built by `scripts/build_results.py`. Place = the person at the robot "
             "typed `p` (cube on the plate).\n")
    L.append("## Students\n")
    L.append("| Label | Bundle on the robot computer | Checkpoint (local copy of the bundle) | student.pt sha256 (first 16) |")
    L.append("|---|---|---|---|")
    for b in binfo.values():
        L.append(f"| {b['label']} | {b['bundle']} | {b['student_ckpt_local'].replace('/home/harry', '~')} "
                 f"| {b['student_sha256'][:16]} |")
    L.append("")
    L.append("## Totals\n")
    L.append("| Student | Test | Places | Rate | 95 % (Wilson) |")
    L.append("|---|---|---|---|---|")

    def row(lab, test, t):
        L.append(f"| {lab} | {test} | {t['placed']}/{t['n']} | {100 * t['rate']:.0f} % | "
                 f"{100 * t['wilson95'][0]:.0f}-{100 * t['wilson95'][1]:.0f} % |")
    row("tb10", "stress 10-27 (6 sessions)", per_student["tb10"]["stress_20261027"])
    for d in ("R->L", "L->R"):
        row("tb10", f"stress, {d}", by_dir[d])
    row("tb10", "all 10-27 (screening + stress)", per_student["tb10"]["all_20261027"])
    row("D", "all 10-24..10-27 (7 sessions)", per_student["D"]["all_20261024_27"])
    for day, t in per_student["D"]["by_day"].items():
        row("D", f"day {day}", t)
    for lab in per_student:
        row(lab, "screening 10-27", per_student[lab]["screening_20261027"])
    row("all 7", "screening 10-27", tally(screen))
    L.append("")
    L.append("## Independent recount of the stress test\n")
    L.append(f"- session.json place flags, 6 sessions: **{raw_place}/{raw_n}** (report claim 36/50; agrees: "
             f"{recount['agrees']}).")
    L.append(f"- Per session: " + ", ".join(f"{s.split('_')[-1]} {sessions[s]['placed']}/{sessions[s]['n_episodes']}"
                                          for s in STRESS))
    L.append(f"- Typed code letters agree with the flags: {recount['typed_code_agrees_all']}; "
             f"ep_NN/episode_live.json outcome agrees: {recount['episode_json_agrees_all']}; "
             f"stress_episodes.json placed/typed agree: {recount['stress_json_agrees_all']}; "
             f"same 50 episode ids: {recount['stress_json_ids_match']}.")
    L.append(f"- Direction label (stress_episodes.json `dir`) = sign of the arm-frame lateral offset plate y - cube y "
             f"(y = r sin(deg)): {recount['dir_matches_lateral_y']}/50 agree; |offset| < 15 mm (near ambiguous): "
             f"{', '.join(recount['dir_near_ambiguous_lt15mm']) or 'none'}.")
    for x in excluded:
        L.append(f"- Excluded: {x['session']}/{x['ep']} ({x['end_reason']}; {x['why']}).")
    L.append("")
    L.append("## Episodes\n")
    L.append("| Session | Ep | Group | Student | Typed | Place | Dir | Cube r mm / deg | End reason | Duration s |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        geo = f"{r['cube_r_mm']:.0f} / {r['cube_deg']:+.0f}" if "cube_r_mm" in r else ""
        L.append(f"| {r['session']} | {r['ep']} | {r['group']} | {r['student']} | {r['typed']} | {r['place']} | "
                 f"{r.get('dir', '')} | {geo} | {r['end_reason']} | {r['duration_s']:.1f} |")
    open(f"{OUT}/results.md", "w").write("\n".join(L) + "\n")
    print(json.dumps(dict(recount=recount, per_student=per_student, excluded=excluded), indent=1))


if __name__ == "__main__":
    sys.exit(main())
