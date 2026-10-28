"""Release sim check of a camera student (v12 check, frozen runner check_eval.py sha256 f70011ab, ported): one
closed-loop CPU MuJoCo evaluation through scripts/so101_policy/vision_viral.py eval, with the recording of the planned
chunk per decision, the goal plate 3.9 mm thick in the renders and the drop rule at the plate top (plane runs), the
parts override of the arm (RL = configs/so101_pick_place/parts_override_v2.json, the real-look gripper parts; GR = the
parts of the scene real_scene_v2.json; own = the student's pinned scene), tasks re-seeded (--seed-base: seed = base +
row + N j for draw j), the measured table plane (or --plane), the flat robot floor and NEDM_SO101_DYN from the
environment (run_gates.sh sets the fitted servo model configs/so101_pick_place/dyn_fit_d2q.json).

    NEDM_SO101_DYN="$(cat configs/so101_pick_place/dyn_fit_d2q.json)" PYTHONPATH=src python -B \
        scripts/so101_pick_place/check/check_eval.py eval --student tb10 --arm own \
        --tasks artifacts/so101_pick_place/rl_inputs/check_tasks/jit224.npz --horizon 2000 --tag g2_a --out OUT.npz

Ported from the study's frozen runner (~/so101_r2s_claude/v12/check/lib/check_eval.py, sha256 f70011ab...): the
paths come from this repository (code, configs/so101_pick_place, artifacts/so101_pick_place) instead of the study's
folders; the "build68" command (it built rows 30-67 of h68.npz from robot logs with study-only scripts) is left out:
h68.npz is released (rl_inputs/check_tasks/). The eval arguments, the patches of vision_viral.py and the recorded
arrays are the frozen runner's (checked on one G2 smoke run: same arrays).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parents[3]                # the repository root
WT = Path(os.environ.get("CHECK_WT", str(CODE)))           # code under test: this checkout (scripts/, src/)
DATA = CODE / "artifacts" / "so101_pick_place"            # the Hub folder after reproduce.sh --steps download
CONF = CODE / "configs" / "so101_pick_place"
WB = Path(os.environ.get("CHECK_WB", str(CONF)))
CALIB = Path(os.environ.get("CHECK_CALIB", str(CONF / "calib_v1.json")))
JT = Path(os.environ.get("CHECK_JT", str(DATA / "teacher/t4_jt_h110/model_00110.pt")))   # loaded, not used for acting
PLANE = (-0.00468, -0.03641, 0.01836)
PLATE = 0.0039031332928005
STUDENTS = {"D": DATA / "student/s4_v8_roll_s801/it_02000.pt", "tb10": DATA / "student/s5_v12_tb10_s1210/it_01000.pt"}
SCENE_DIRS = [CONF, WB] + [Path(p) for p in os.environ.get("CHECK_SCENE_DIRS", "").split(":") if p]
BASE = {"v2": Path(os.environ.get("CHECK_BASE_V2", str(CONF / "real_scene_v2.json"))),
        "v4": Path(os.environ.get("CHECK_BASE_V4", str(CONF / "real_scene_v4.json")))}      # v4: not released
BASE_SHA = {"v2": "4bce0b8dd6f22b5499b556275bca0bf248d1e96b1cd3babf169b34a8cb539d82",
            "v4": "3e862c06395c6122587683dbbfc2d3d2bfb858e6c6ac47e51812634dd4f3fbef"}
RL_PARTS = {"v2": WB / "parts_override_v2.json", "v4": WB / "parts_override_v4.json"}           # v4: not released
RL_SHA = {"v2": "3840ca99745839ff693dca975f9853720176ab61b3b7d751a42995e0e5366217",
          "v4": "b1c96039a1614ccf38caf6aad84d8fd8dc1715d22ff8de21a047680e69bd2de3"}
REC = dict(on=False, delta=[], q=[], chunk=[], exec=[])


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] check_eval: {s}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ---- eval ----------------------------------------------------------------------------------------------------------
def find_scene(ck):
    import torch
    c = torch.load(ck, map_location="cpu", weights_only=False)["cfg"]
    get = (lambda k: c.get(k, "")) if isinstance(c, dict) else (lambda k: getattr(c, k, ""))
    want = get("scene_sha256")
    own = Path(str(get("scene") or "")).expanduser()
    if str(own) and own.is_file() and sha(own) == want:          # the pinned path itself (AMD runs)
        return own
    for dd in SCENE_DIRS:
        if not dd.is_dir():
            continue
        for p in sorted(dd.glob("*.json")):
            try:
                if sha(p) == want:
                    return p
            except OSError:
                continue
    raise SystemExit(f"no scene file with sha256 {want[:12]} for {ck} in {[str(d) for d in SCENE_DIRS]}")


def family(scene):
    names = [p.get("name", "") for p in json.loads(Path(scene).read_text())["geometry"].get("parts", [])]
    return "v4" if any(n.startswith("base_clamp") for n in names) else "v2"


def arm_parts(arm, fam):
    if arm == "RL":
        f = RL_PARTS[fam]
        if sha(f) != RL_SHA[fam]:
            raise SystemExit(f"{f}: sha256 {sha(f)[:12]} != the frozen {RL_SHA[fam][:12]}")
        return json.loads(f.read_text()), str(f)
    if arm == "GR":
        f = BASE[fam]
        if sha(f) != BASE_SHA[fam]:
            raise SystemExit(f"{f}: sha256 {sha(f)[:12]} != the frozen {BASE_SHA[fam][:12]}")
        return json.loads(f.read_text())["geometry"]["parts"], str(f)
    raise SystemExit(f"arm {arm}")


def look_of(scene, fam, arm):
    """The look a run renders (Addendum B2): 'grey' (the family's base scene parts), 'real' (the real-look parts of
    arm RL) or 'other'. Arm 'own' = the parts of the student's pinned scene file, compared with both lists."""
    if arm in ("RL", "GR"):
        return "real" if arm == "RL" else "grey"
    own = json.loads(Path(scene).read_text())["geometry"].get("parts", [])
    if own == arm_parts("GR", fam)[0]:
        return "grey"
    if own == arm_parts("RL", fam)[0]:
        return "real"
    return "other"


def tile(t, n, base):
    """The task set n times; seeds = base + row + N j (base 0: the file's own seeds, n must be 1)."""
    N = len(t["seeds"])
    if not base and n != 1:
        raise SystemExit("--draws > 1 needs --seed-base")
    out = {k: (np.concatenate([v] * n) if (np.ndim(v) and len(v) == N) else v) for k, v in t.items()}
    if base:
        out["seeds"] = np.concatenate([np.arange(N, dtype=np.int64) + base + j * N for j in range(n)])
    return out


def cmd_eval(a):
    import torch
    torch.set_num_threads(a.threads)
    from nedm.so101_policy.vision import viral as VI
    if not str(VI.__file__).startswith(str(WT)):
        raise SystemExit(f"nedm from {VI.__file__}, not {WT}")
    plane = None if a.plane == "sim" else (PLANE if a.plane == "measured" else tuple(float(v) for v in a.plane.split(",")))
    if a.plane_dz:
        if plane is None:
            raise SystemExit("--plane-dz needs a plane")
        plane = (plane[0] + a.plane_dz, plane[1], plane[2])
    orig_den = VI.denormalise

    def denormalise(st, p):
        out = orig_den(st, p)
        if REC["on"]:
            REC["delta"].append(out.detach().float())
        return out
    VI.denormalise = denormalise
    orig_call = VI.Actor.__call__

    def call(self, ws, goal_pose, k=None):
        REC["delta"], REC["on"] = [], True
        try:
            ex = orig_call(self, ws, goal_pose, k)
        finally:
            REC["on"] = False
        delta = torch.cat(REC["delta"])
        q = ws.q.to(delta.device, torch.float32).nan_to_num()
        if self.q_offset is not None:
            q = q + torch.as_tensor(self.q_offset, dtype=q.dtype, device=q.device)
        if delta.shape[0] != q.shape[0]:
            raise RuntimeError(f"chunk record: {delta.shape[0]} deltas for {q.shape[0]} worlds")
        REC["q"].append(q.cpu())
        REC["chunk"].append((q[:, None, :] + delta).cpu())
        REC["exec"].append(ex.float().cpu())
        return ex
    VI.Actor.__call__ = call
    ck = Path(STUDENTS.get(a.student, a.student)).expanduser().resolve()
    scene = find_scene(ck)
    fam = family(scene)
    parts, parts_src = arm_parts(a.arm, fam) if a.arm != "own" else (None, "own scene")
    look = look_of(scene, fam, a.arm)                      # Addendum B2: recorded in the run's "check" info
    info = {}
    orig_make_views = VI.make_views

    def make_views(cfg, num_worlds, **kw):
        geo = dict(kw.get("geometry_over") or {})
        if plane is not None:
            if "table_plane" not in geo:
                raise SystemExit("check_eval: the plate patch needs the table plane")
            geo["plate_thickness"] = PLATE
        if parts is not None:
            geo["parts"] = parts
        kw["geometry_over"] = geo or None
        views = orig_make_views(cfg, num_worlds, **kw)
        rs = views.real.real
        got = [p.get("name") if isinstance(p, dict) else getattr(p, "name", None) for p in rs.parts]
        if parts is not None and len(got) != len(parts):
            raise SystemExit(f"check_eval: the parts override did not reach the renderer ({len(got)} vs {len(parts)})")
        info.update(parts=got, table_plane=list(rs.table_plane) if rs.table_plane is not None else None,
                    plate_mm=float(rs.plate_thickness) * 1e3)
        log(f"render: arm {a.arm} ({fam}, {parts_src}): {len(got)} parts, table plane {info['table_plane']}, plate "
            f"{info['plate_mm']:.2f} mm")
        return views
    VI.make_views = make_views
    spec = importlib.util.spec_from_file_location("vision_viral", str(WT / "scripts/so101_policy/vision_viral.py"))
    VV = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(VV)
    if plane is not None:
        orig_get = VV.EnvCache.get

        def get(self, world, B):
            new = (world, B) not in self.c
            env = orig_get(self, world, B)
            if new:
                if env.task.cfg is not env.tcfg:
                    raise SystemExit("check_eval: the task config is not the env's")
                env.tcfg.release_z = float(env.tcfg.release_z) + PLATE
                log(f"rules: release_z -> {env.tcfg.release_z * 1e3:.1f} mm above the physics table (plate top)")
            return env
        VV.EnvCache.get = get
    orig_eval = VV.evaluate

    def evaluate(*p, **kw):
        r = orig_eval(*p, **kw)
        r["rec_q"] = torch.stack(REC["q"], 1).numpy().astype(np.float32)
        r["rec_chunk"] = torch.stack(REC["chunk"], 1).numpy().astype(np.float32)
        r["rec_exec"] = torch.stack(REC["exec"], 1).numpy().astype(np.float32)
        head = subprocess.run(["git", "-C", str(WT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        if not head and (WT / "SYNC_HEAD.txt").exists():
            head = (WT / "SYNC_HEAD.txt").read_text().strip()
        r["check"] = np.array(json.dumps(dict(
            tag=a.tag, student=a.student, ckpt=str(ck), ckpt_sha256=sha(ck), scene=str(scene), scene_sha256=sha(scene),
            family=fam, arm=a.arm, look=look, parts_src=parts_src, parts=info.get("parts"), tasks=str(a.tasks),
            tasks_sha256=sha(a.tasks), draws=a.draws, seed_base=a.seed_base, n_base=n_base, plane=plane,
            floor=a.floor, horizon=a.horizon, hold_cap_s=a.hold_cap_s, roll_hi_deg=a.roll_hi_deg,
            dyn=os.environ.get("NEDM_SO101_DYN", "")[:40] and json.loads(os.environ["NEDM_SO101_DYN"]).get("note"),
            wt=str(WT), wt_head=head, check_eval_sha256=sha(__file__))))
        return r
    VV.evaluate = evaluate
    t = dict(np.load(a.tasks, allow_pickle=True))
    n_base = len(t["seeds"])
    t = tile(t, a.draws, a.seed_base)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tf = out.parent / f"_tasks_{a.tag}.npz"
    np.savez(tf, **t)
    nw = len(t["seeds"])
    args = ["eval", "--student", str(ck), "--scene-path", str(scene), "--calib-path", str(CALIB),
            "--render-worlds", str(nw), "--jt", str(JT), "--tasks", str(tf), "--world", "mujoco",
            "--workers", str(a.workers), "--record-states", "--horizon", str(a.horizon), "--hold-cap-s", str(a.hold_cap_s),
            "--qd-robot", "3", "--floor", a.floor, "--seed", "0", "--out", str(out)]
    if plane is not None:
        args.insert(-2, f"--table-plane={plane[0]},{plane[1]},{plane[2]}")
    if a.roll_hi_deg is not None:
        args += ["--roll-hi-deg", str(a.roll_hi_deg)]
    args += [x for x in os.environ.get("CHECK_EVAL_EXTRA", "").split() if x]      # AMD: --nrd-ckpt, --mj-python
    log(f"eval {a.tag}: student {ck}, scene {scene} ({fam}), arm {a.arm} (look {look}), {nw} worlds ({a.draws} draws, seed base "
        f"{a.seed_base}), plane {plane}, dyn {'on' if os.environ.get('NEDM_SO101_DYN') else 'OFF'}")
    try:
        VV.main(args)
    finally:
        tf.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("eval")
    q.add_argument("--student", required=True, help="a STUDENTS name or a checkpoint path")
    q.add_argument("--tag", required=True)
    q.add_argument("--arm", choices=("RL", "GR", "own"), required=True)
    q.add_argument("--tasks", required=True)
    q.add_argument("--draws", type=int, default=1)
    q.add_argument("--seed-base", type=int, default=0, help="seed = base + row + N j (0: the file's seeds, 1 draw)")
    q.add_argument("--plane", default="measured", help="measured | sim | c0,c1,c2")
    q.add_argument("--plane-dz", type=float, default=0.0, help="m added to the plane's height term")
    q.add_argument("--floor", default="zero")
    q.add_argument("--horizon", type=int, default=400)
    q.add_argument("--hold-cap-s", type=float, default=40.0)
    q.add_argument("--roll-hi-deg", type=float, default=None)
    q.add_argument("--workers", type=int, default=6)
    q.add_argument("--threads", type=int, default=6)
    q.add_argument("--out", required=True, help="output .npz")
    q.set_defaults(fn=cmd_eval)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
