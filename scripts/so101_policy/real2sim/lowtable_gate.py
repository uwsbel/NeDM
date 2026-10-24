"""Lowered-table gate of camera students (real2sim v8 table randomisation, 2026-10-23/24; committed from the scratch
``~/so101_r2s_claude/v8/table/lowtable.py`` after review). Real robot: the real table is 12-17 mm lower than the sim
table where the cubes are (calib_v1 scene.table_plane), the real goal plate 3.9 mm thick (scenes 1 mm); the robot
students stop where the sim cube would touch and hover with the cube above the plate. This gate moves the PHYSICAL
table in CPU MuJoCo and checks that a student still grasps and releases.

Setup (G5 style, ``scripts/so101_policy/vision_viral.py eval`` of this repo): CPU MuJoCo (8 workers), GPU renders of
100 worlds (nominal visuals, the student's own scene), jt_h110, default hold cap 3.5 s, tasks dev100 + home45_dev100
(200 episodes). Lowered-table variants: the MuJoCo table on a plane (``--table-plane``: physics, task starts, task
rules), the renders on the same plane with the goal plate 3.9 mm thick (geometry override in the worker), the two drop
rules judged at the plate top (release_z + plate), the action-layer floor at arm-frame z = 0 (robot ``--table-z 0``)
unless ``floor`` = table (the floor follows the tilted table: robot ``--table-plane calib``). Optional robot servo
(``*_fitq``: ``<servo dir>/dyn_<set>.json`` via NEDM_SO101_DYN; 30 s episodes, hold cap 20 s, robot-like qdot).

PASS RULE (fixed 2026-10-24, before any table-trained student was evaluated; proposed by the builder after the
reference runs of C1 and s28 below, so not blind for those two):
- lenient >= 90 % on the 200 episodes at BOTH ``real_m5`` (the calibrated real plane 5 mm lower, ~19 mm below the sim
  table at x 0.25 m) and ``real_m5_floor`` (the same with the action-layer floor on the table);
- hold-cap hovers <= 10 of 200 in each of the two;
- ``nominal`` lenient not more than 3 points below the student's init (``REF_NOMINAL``: C1 92.0 %, s28 97.5 %).
Reported for every student, no rule: ``real`` (the calibrated plane = gate 8 / G5), ``real_m10`` (10 mm lower, ~24
mm: outside the 20 mm training range) and on request ``flat20`` / ``*_fitq``.
Reference runs (2026-10-23/24, 200 episodes, lenient % (hold-cap hovers); both FAIL the rule):
    | student | nominal | real | real_m5 | real_m5_floor | real_m10 |
    | C1      | 92.0 (7) | 85.0 (12) | 50.5 (73) | 30.5 (110) | 16.0 (108) |
    | s28     | 97.5 (4) | 96.0 (2)  | 79.0 (11) | 71.0 (19)  | 15.5 (34)  |
The sim reproduces C1's no-release hover but not its open-jaw grasp miss on the robot, and not s28's robot
no-release (s28 keeps 96 % on the calibrated plane): a pass is necessary, not sufficient.

    python scripts/so101_policy/real2sim/lowtable_gate.py run --ckpt X.pt --tag T [--variants ...] [--data D]
        [--out O] [--lock /home/harry/so101_r2s_claude/gpu.lock]
    python scripts/so101_policy/real2sim/lowtable_gate.py report --tags C1,s28 [--out O]
    python scripts/so101_policy/real2sim/lowtable_gate.py gate --tag T --ref C1 [--out O]   (pass rule; exit 1 = fail)

``--data`` (default $NEDM_R2S_GATE_DATA, else the studentlab stage ~/so101_studentlab_claude/trials/stage/data, read
only): ``jt_h110.pt``, ``calib/calib_v1.json``, ``tasks/dev100.npz``, ``tasks/home45_dev100.npz``, ``scene/*.json``
(the student's scene is found by its sha256 there or in ``--scene-dirs``). Per run (``<out>/runs/<tag>/<set>_<variant>
.{npz,log,json}``): lenient / strict, lifted (cube bottom >= 20 mm above the physics table), closed (jaw < 15 deg at
any step), hold-cap hovers (end = hold_cap: held > 3.5 s), lifted but never opened past 0.19 rad, open-jaw misses
(never lifted, jaw never below 15 deg), release height above the plate top (the step before the first open after
the lift), end causes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
DATA = Path(os.environ.get("NEDM_R2S_GATE_DATA", "~/so101_studentlab_claude/trials/stage/data")).expanduser()
OUT = Path(os.environ.get("NEDM_R2S_GATE_OUT", "~/so101_r2s_claude/v8/table_gate")).expanduser()
SCENE_DIRS = ("~/so101_r2s_claude/scene", "~/so101_r2s_claude/data/scene", "~/so101_studentlab_claude/scene")
SERVO = Path("~/so101_r2s_claude/sim_debug/servo").expanduser()
GATE_SETS = ("dev100", "home45_dev100")
PLANE = (-0.00468, -0.03641, 0.01836)   # calib_v1 scene.table_plane (gate 8 / table_test_final PLANE)
PLATE_REAL = 0.0039031332928005         # calib_v1 scene.plate_thickness
OPEN_JAW, LIFT_H, CUBE_HALF, JAW_CLOSED = 0.19, 0.02, 0.015, np.radians(15.0)
# the pass rule (module docstring)
PASS = dict(variants=("real_m5", "real_m5_floor"), min_lenient_pct=90.0, max_hovers=10, nominal_drop_pts=3.0)
REF_NOMINAL = {"C1": 92.0, "s28": 97.5}          # nominal lenient % of the inits (G5 reference, 200 episodes)
REPORT = ("nominal", "real", "real_m5", "real_m5_floor", "real_m10")


def shifted(dz):
    return (PLANE[0] + dz, PLANE[1], PLANE[2])


# name -> (plane or None, floor, dyn set or None)
VARIANTS = {
    "nominal": (None, "zero", None),
    "real": (PLANE, "zero", None),                       # = gate 8 realtable (G5)
    "real_m5": (shifted(-0.005), "zero", None),          # the real plane 5 mm lower (18.8 mm at x 0.25)
    "real_m10": (shifted(-0.010), "zero", None),         # 10 mm lower (23.8 mm: beyond the 20 mm training range)
    "flat20": ((-0.020, 0.0, 0.0), "zero", None),
    "real_floor": (PLANE, "table", None),                # robot --table-plane calib (tilted floor)
    "real_m5_floor": (shifted(-0.005), "table", None),
    "nominal_fitq": (None, "zero", "fit_d2q"),           # fitted real servo, queued 2-step delay
    "real_fitq": (PLANE, "zero", "fit_d2q"),
    "real_m5_fitq": (shifted(-0.005), "zero", "fit_d2q"),
}


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def student_scene(ckpt, data, scene_dirs):
    import torch
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    want = ck["cfg"].get("scene_sha256", "")
    cands = sorted((data / "scene").glob("*.json")) if (data / "scene").exists() else []
    for d in scene_dirs:
        d = Path(d).expanduser()
        if d.exists():
            cands += sorted(d.glob("*.json"))
    for p in cands:
        if not want or sha256(p) == want:
            return p, ck.get("it")
    raise SystemExit(f"no scene with the student's sha256 {want[:12]} in {data / 'scene'} or {list(scene_dirs)}")


# ---- worker: plate 3.9 mm in the render, drops judged at the plate top (= gate8_realtable.worker) ----------------
def worker(eval_args, plate, release_add):
    sys.path.insert(0, str(REPO / "src"))
    from nedm.so101_policy.vision import viral as VI
    orig_make_views = VI.make_views

    def make_views(cfg, num_worlds, **kw):
        geo = dict(kw.get("geometry_over") or {})
        if "table_plane" not in geo:
            raise SystemExit("worker: the plate patch is for a lowered-table run (--table-plane) only")
        geo["plate_thickness"] = float(plate)
        kw["geometry_over"] = geo
        views = orig_make_views(cfg, num_worlds, **kw)
        rs = views.real.real
        log(f"render: table plane {tuple(rs.table_plane)}, plate {rs.plate_thickness * 1e3:.2f} mm, mat "
            f"{rs.mat_thickness * 1e3:.2f} mm")
        if abs(rs.plate_thickness - plate) > 1e-12:
            raise SystemExit("worker: the plate patch did not reach the renderer")
        return views
    VI.make_views = make_views
    import importlib.util
    spec = importlib.util.spec_from_file_location("vision_viral", str(REPO / "scripts/so101_policy/vision_viral.py"))
    VV = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(VV)
    orig_get = VV.EnvCache.get

    def get(self, world, B):
        new = (world, B) not in self.c
        env = orig_get(self, world, B)
        if new and release_add:
            if env.task.cfg is not env.tcfg:
                raise SystemExit("worker: the task config is not the env's")
            z0 = float(env.tcfg.release_z)
            env.tcfg.release_z = z0 + float(release_add)
            log(f"rules: release_z {z0 * 1e3:.1f} -> {env.tcfg.release_z * 1e3:.1f} mm above the physics table")
        return env
    VV.EnvCache.get = get
    VV.main(["eval"] + list(eval_args))


# ---- analysis -------------------------------------------------------------------------------------------------------
def plane_h(plane, xy):
    if plane is None:
        return np.zeros(xy.shape[:-1])
    return plane[0] + plane[1] * xy[..., 0] + plane[2] * xy[..., 1]


def analyse(npz, plane, plate):
    sys.path.insert(0, str(REPO / "src"))
    from nedm.so101_policy import task as T
    z = np.load(npz, allow_pickle=True)
    q, cube, steps = z["state_q"], z["state_cube"], z["steps"].astype(int)
    n = len(steps)
    rel = np.full(n, np.nan)
    lifted = np.zeros(n, bool)
    closed = np.zeros(n, bool)
    for b in range(n):
        k = max(1, min(int(steps[b]), q.shape[1]))
        c = cube[b, :k, :3].astype(np.float64)
        h = c[:, 2] - plane_h(plane, c[:, :2]) - CUBE_HALF
        jaw = q[b, :k, 5]
        closed[b] = bool((jaw < JAW_CLOSED).any())
        up = np.flatnonzero(h >= LIFT_H)
        if len(up):
            lifted[b] = True
            op = np.flatnonzero((jaw > OPEN_JAW) & (np.arange(k) > up[0]))
            if len(op):
                rel[b] = (h[op[0] - 1] - plate) * 1e3
    le = np.asarray(z["lenient"]).astype(bool)
    end = np.asarray(z["end"]).astype(int)
    causes = {}
    for code in end[~le]:
        causes[T.ENDS[int(code)]] = causes.get(T.ENDS[int(code)], 0) + 1
    r = rel[np.isfinite(rel)]
    p = lambda v, k: None if not len(v) else round(float(np.percentile(v, k)), 2)  # noqa: E731
    return dict(n=n, lenient=int(le.sum()), strict=int(np.asarray(z["success"]).astype(bool).sum()),
                lifted=int(lifted.sum()), closed=int(closed.sum()), hover=int((end == T.EC["hold_cap"]).sum()),
                lifted_no_open=int((lifted & ~np.isfinite(rel)).sum()), miss_open=int((~lifted & ~closed).sum()),
                rel_plate_mm_p50=p(r, 50), rel_plate_mm_p90=p(r, 90), rel_gt5mm=int((r > 5.0).sum()),
                causes=causes)


def pooled(rows):
    keys = ("n", "lenient", "strict", "lifted", "closed", "hover", "lifted_no_open", "miss_open", "rel_gt5mm")
    out = {k: sum(r[k] for r in rows) for k in keys}
    c = {}
    for r in rows:
        for k, v in r["causes"].items():
            c[k] = c.get(k, 0) + v
    out["causes"] = c
    return out


def judge(rows, ref_nominal_pct):
    """The pass rule (module docstring) on the per-set result rows of one student -> dict(pass, checks)."""
    by = {}
    for r in rows:
        by.setdefault(r["variant"], {})[r["set"]] = r
    checks = []

    def pool(var):
        got = by.get(var, {})
        if set(GATE_SETS) - set(got):
            return None
        return pooled([got[s] for s in GATE_SETS])

    for var in PASS["variants"]:
        P = pool(var)
        if P is None:
            checks.append(dict(check=f"{var}: both sets run", ok=False))
            continue
        pct = 100.0 * P["lenient"] / P["n"]
        checks.append(dict(check=f"{var}: lenient >= {PASS['min_lenient_pct']} %", value=round(pct, 1),
                           ok=pct >= PASS["min_lenient_pct"]))
        checks.append(dict(check=f"{var}: hold-cap hovers <= {PASS['max_hovers']}", value=P["hover"],
                           ok=P["hover"] <= PASS["max_hovers"]))
    P = pool("nominal")
    if P is None:
        checks.append(dict(check="nominal: both sets run", ok=False))
    else:
        pct = 100.0 * P["lenient"] / P["n"]
        lim = float(ref_nominal_pct) - PASS["nominal_drop_pts"]
        checks.append(dict(check=f"nominal: lenient >= {lim:.1f} % (init {ref_nominal_pct} - "
                                 f"{PASS['nominal_drop_pts']})", value=round(pct, 1), ok=pct >= lim - 1e-9))
    info = {}
    for var in REPORT:
        P = pool(var)
        if P is not None:
            info[var] = dict(lenient_pct=round(100.0 * P["lenient"] / P["n"], 1), hovers=P["hover"], n=P["n"])
    return {"pass": all(c["ok"] for c in checks), "checks": checks, "report": info}


# ---- driver ---------------------------------------------------------------------------------------------------------
def run_one(a, ckpt, scene, tag, tset, var):
    plane, floor, dyn = VARIANTS[var]
    out = a.out / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    npz, lg = out / f"{tset}_{var}.npz", out / f"{tset}_{var}.log"
    args = ["--student", str(ckpt), "--scene-path", str(scene), "--calib-path", str(a.data / "calib/calib_v1.json"),
            "--render-worlds", "100", "--jt", str(a.data / "jt_h110.pt"), "--tasks", str(a.data / f"tasks/{tset}.npz"),
            "--world", "mujoco", "--workers", str(a.workers), "--record-states", "--out", str(npz)]
    if dyn:                 # the robot servo is slow: 30 s episodes, hold cap 20 s, robot-like qdot (= sim_eval.sh)
        args += ["--horizon", "1500", "--hold-cap-s", "20", "--qd-robot", "3"]
    if plane is not None:
        args += ["--table-plane=" + ",".join(f"{c:g}" for c in plane), "--floor", floor]
        cmd = [a.python, "-B", str(Path(__file__).resolve()), "worker", "--plate", repr(PLATE_REAL), "--release-add",
               repr(PLATE_REAL), "--"] + args
    else:
        cmd = [a.python, "-P", "-B", "scripts/so101_policy/vision_viral.py", "eval"] + args
    cache = a.out / "cache"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO / "src"), PYTHONUNBUFFERED="1",
               WARP_CACHE_PATH=str(cache / "warp"), NEDM_SO101_OVERLAP_CACHE=str(cache / "so101_overlap"))
    env["NEDM_SO101_DYN"] = (a.servo / f"dyn_{dyn}.json").read_text() if dyn else ""
    t0 = time.time()
    with open(lg, "w") as fo:
        fo.write(f"# {var}: plane {plane} floor {floor} dyn {dyn} | {' '.join(cmd)}\n")
        fo.flush()
        rc = subprocess.call((["flock", a.lock] if a.lock else []) + cmd, cwd=str(REPO), env=env, stdout=fo,
                             stderr=subprocess.STDOUT)
    wall = time.time() - t0
    if rc != 0 or not npz.exists():
        log(f"{tag} {tset} {var}: exit {rc} ({lg})")
        return None
    plate = PLATE_REAL if plane is not None else 0.0
    res = analyse(npz, plane, plate)
    m = re.findall(r"student mujoco: (\{.*\})\s*$", lg.read_text(), re.M)
    res.update(variant=var, set=tset, tag=tag, ckpt=str(ckpt), plane=plane, floor=floor, dyn=dyn,
               wall_s=round(wall, 1), summary=json.loads(m[-1]) if m else None)
    (out / f"{tset}_{var}.json").write_text(json.dumps(res, indent=1))
    log(f"{tag} {tset} {var}: lenient {res['lenient']}/{res['n']} hover {res['hover']} lifted-no-open "
        f"{res['lifted_no_open']} miss-open {res['miss_open']} rel p50 {res['rel_plate_mm_p50']} mm | "
        f"{res['causes']} | {wall:.0f} s")
    return res


def load_rows(out, tag):
    return [json.loads(f.read_text()) for f in sorted((out / "runs" / tag).glob("*.json"))]


def report(out, tags):
    lines = ["| student | variant | episodes | lenient % | strict % | lifted | closed | hold-cap hovers | "
             "lifted, never opened | open-jaw misses | release above plate p50 mm | causes |", "|" + "---|" * 12]
    for tag in tags:
        by = {}
        for a in load_rows(out, tag):
            by.setdefault(a["variant"], []).append(a)
        for var in VARIANTS:
            if var not in by:
                continue
            rows = by[var]
            P = pooled(rows)
            rel = [r["rel_plate_mm_p50"] for r in rows if r["rel_plate_mm_p50"] is not None]
            sets = "+".join(sorted(r["set"].replace("_dev100", "") for r in rows))
            lines.append(f"| {tag} | {var} ({sets}) | {P['n']} | {100 * P['lenient'] / P['n']:.1f} | "
                         f"{100 * P['strict'] / P['n']:.1f} | {P['lifted']} | {P['closed']} | {P['hover']} | "
                         f"{P['lifted_no_open']} | {P['miss_open']} | "
                         f"{'/'.join(f'{v:.1f}' for v in rel)} | {P['causes']} |")
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "worker":
        i = argv.index("--")
        ap = argparse.ArgumentParser()
        ap.add_argument("--plate", type=float, required=True)
        ap.add_argument("--release-add", type=float, default=0.0)
        w = ap.parse_args(argv[1:i])
        worker(argv[i + 1:], w.plate, w.release_add)
        return 0
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "report", "gate"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=OUT, help="results folder (runs/<tag>/...)")
        if name == "run":
            p.add_argument("--ckpt", required=True)
            p.add_argument("--tag", required=True)
            p.add_argument("--variants", default=",".join(REPORT))
            p.add_argument("--sets", default=",".join(GATE_SETS))
            p.add_argument("--skip-done", action="store_true")
            p.add_argument("--data", type=Path, default=DATA, help="jt_h110.pt, calib/, tasks/, scene/ (read only)")
            p.add_argument("--scene-dirs", default=",".join(SCENE_DIRS), help="more folders with scene json files")
            p.add_argument("--python", default=sys.executable)
            p.add_argument("--lock", default="", help="flock file for the GPU (e.g. ~/so101_r2s_claude/gpu.lock)")
            p.add_argument("--servo", type=Path, default=SERVO, help="dyn_<set>.json folder (*_fitq variants)")
            p.add_argument("--workers", type=int, default=8)
        elif name == "report":
            p.add_argument("--tags", required=True)
        else:
            p.add_argument("--tag", required=True)
            p.add_argument("--ref", required=True, help=f"the init's nominal reference: {sorted(REF_NOMINAL)} or a %")
    a = ap.parse_args(argv)
    a.out = Path(a.out).expanduser()
    if a.cmd == "report":
        txt = report(a.out, a.tags.split(","))
        print(txt)
        (a.out / "report_table.txt").write_text(txt + "\n")
        return 0
    if a.cmd == "gate":
        ref = REF_NOMINAL[a.ref] if a.ref in REF_NOMINAL else float(a.ref)
        res = judge(load_rows(a.out, a.tag), ref)
        print(json.dumps(res, indent=1))
        return 0 if res["pass"] else 1
    a.data = Path(a.data).expanduser()
    a.lock = os.path.expanduser(a.lock) if a.lock else ""
    ckpt = Path(a.ckpt).expanduser().resolve()
    scene, it = student_scene(ckpt, a.data, [s for s in a.scene_dirs.split(",") if s])
    log(f"{a.tag}: {ckpt} it {it} scene {scene}")
    for var in a.variants.split(","):
        if var not in VARIANTS:
            raise SystemExit(f"unknown variant {var}: {list(VARIANTS)}")
        for tset in a.sets.split(","):
            if a.skip_done and (a.out / "runs" / a.tag / f"{tset}_{var}.json").exists():
                continue
            run_one(a, ckpt, scene, a.tag, tset, var)
    return 0


if __name__ == "__main__":
    sys.exit(main())
