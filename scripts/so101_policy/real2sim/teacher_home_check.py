"""Real2sim study: can the build9 Markov joint teacher ``jt_h110`` start from ONE fixed pose (the real robot's HOME), and
which real cube / plate placements can it do? State only (no images), deterministic, 8 s, build9 rules (lenient and
strict success), CPU MuJoCo (the collector physics) and the NRD (c1 s62, where the student trains).

Task sets (all in the tray-task layout of ``tray.make_tasks``: start_* = MuJoCo warm-up input, state_* = the settled
state after the collector's 0.3 s warm-up in MuJoCo; seeds 594,500+ are not reserved and do not overlap build9's
590,000-592,999 or 640,000-656,383):

  dev100        build9's development tasks (~/so101_policy/runs/viral/tasks/dev100.npz), as they are (sanity)
  home100       arm at HOME (``hw_lib.HOME_SIM_DEG``), at rest; cube resting at a start-bank cube pose (xy, yaw of a
                random row of jt_h110's PPO bank, production_plus split 0); goal ``task.sample_goals`` (6-30 cm on the
                annulus, yaw +-30 deg) with the tray.make_tasks generator rule; seeds 594,500-594,599
  homenoise100  the same cubes and goals, HOME + U(-3, 3) deg on each of the 6 joints
  real15        the 15 real placements of calib_v1.json "episodes" from HOME: cube start xy + yaw (jaw yaw at the grasp),
                plate xy + yaw (mod 90), cube z at rest on the sim table; seeds 594,700-594,714
  real15sil     the same with the cube yaw from the silhouette fit (second estimate); seeds 594,720-594,734
  realjit300    20 jittered variants of each real placement (make_real_jitter); seeds 596,000-596,299
  outer100      cube and goal on r 0.27-0.34 m about the pan axis, angle +-75 deg, goal 6-30 cm from the cube, goal yaw
                = cube yaw + U(-30, 30) deg; from HOME; seeds 594,800-594,899
  p500_*        (``paired``) 500 more bank cubes + goals (seeds 595,000-595,499) from HOME (p500_home), from the
                training start sampler (p500_ctrl: Scene.sample_above within 40 / 60 deg of the cube direction), from the
                folded real rest pose (p500_rest) and from 5 fixed bank poses with |pan| <= 3 deg (p500_alt_*): paired

    export PYTHONPATH=src PY=~/miniconda3/envs/so101_vision/bin/python
    $PY -P scripts/so101_policy/real2sim/teacher_home_check.py tasks
    $PY -P scripts/so101_policy/real2sim/teacher_home_check.py paired --n 500
    $PY -P scripts/so101_policy/real2sim/teacher_home_check.py eval --set home100 --world mujoco
    $PY -P scripts/so101_policy/real2sim/teacher_home_check.py report

Outputs in ~/so101_r2s_claude/teacher_check/ (tasks/, evals/, REPORT.md is written by hand from ``report``).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

OUT = Path.home() / "so101_r2s_claude" / "teacher_check"
JT_PATH = Path.home() / "so101_policy" / "runs" / "viral" / "jt" / "jt_h110.pt"
NRD = Path.home() / "so101_policy" / "ckpt" / "nrd_c1_s62.pt"
DEV100 = Path.home() / "so101_policy" / "runs" / "viral" / "tasks" / "dev100.npz"
BANK = Path.home() / "so101_policy" / "runs" / "viral" / "data" / "bank_pp_split0.npz"
CALIB = Path.home() / "so101_r2s_claude" / "calib" / "calib_v1.json"
MJ_PY = Path.home() / "miniconda3" / "envs" / "nrdsim" / "bin" / "python"   # vision_viral_teacher.py default
SEED0 = dict(home=594_500, real=594_700, realsil=594_720, outer=594_800)
OUTER_R = (0.27, 0.34)


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def home_rad():
    sys.path.insert(0, str(Path(__file__).resolve().parent / "hw"))
    import hw_lib as H
    return np.radians(np.asarray(H.HOME_SIM_DEG, np.float64)), tuple(H.HOME_SIM_DEG)


# ---- task construction -----------------------------------------------------------------------------------------
class Warm:
    """The collector's 0.3 s warm-up in CPU MuJoCo (``tray.make_tasks``'s block and acceptance rule)."""

    def __init__(self):
        from nedm.contact_nrd.collect import so101_grasp as g
        from nedm.so101_policy.mujoco_env import _Block
        self.g = g
        self.blk = _Block(1, overlap=True)
        self.half = self.blk.s.half

    def __call__(self, q0, xy, cyaw):
        c = self.g.resting_cube(np.array([xy[0], xy[1], self.half]), cyaw)
        row = np.concatenate((q0, np.zeros(6), c["pos"], c["quat"], np.zeros(6)))[None]
        res = self.blk.reset([0], row, warmup=True)
        ok = res["code"][0] == 0 and not res["arm_table"][0] and not res["contact"][0, :2].any()
        why = dict(code=int(res["code"][0]), arm_table=bool(res["arm_table"][0]),
                   contact=res["contact"][0].tolist())
        return bool(ok), c, res["raw"][0], why


def goals_like_make_tasks(seed, raw):
    """tray.make_tasks's goal rule (generator seed * 7 + 3, cube xy / yaw of the settled state)."""
    from nedm.so101_policy import task as T
    gen = torch.Generator().manual_seed(int(seed) * 7 + 3)
    r = raw.astype(np.float32).astype(np.float64)
    cy, _ = T.cube_yaw_tilt(torch.as_tensor(r[None, 15:19]).double())
    gg, gy = T.sample_goals(torch.as_tensor(r[None, 12:14]).double(), cy, gen)
    return gg[0].numpy(), float(gy[0])


def pack(seeds, q, pos, quat, raw, gxy, gyaw, **extra):
    r = np.asarray(raw).astype(np.float32).astype(np.float64)
    out = dict(seeds=np.asarray(seeds, np.int64), start_q=np.asarray(q), start_cube_pos=np.asarray(pos),
               start_cube_quat=np.asarray(quat), goal_xy=np.asarray(gxy, np.float64),
               goal_yaw=np.asarray(gyaw, np.float64), stage=np.array("H"), dev=np.bool_(True))
    for name, sl in (("q", slice(0, 6)), ("qd", slice(6, 12)), ("cube_pos", slice(12, 15)), ("cube_quat", slice(15, 19)),
                     ("cube_v", slice(19, 22)), ("cube_w", slice(22, 25))):
        out[f"state_{name}"] = r[:, sl]
    out.update({k: np.asarray(v) for k, v in extra.items()})
    return out


def check_seeds(seeds):
    from nedm.so101_policy.mujoco_env import reserved_seeds
    from nedm.so101_policy.vec_env import reserved_seed_hits
    if reserved_seed_hits(seeds) or set(int(s) for s in seeds) & reserved_seeds():
        raise ValueError("reserved seeds")
    if ((np.asarray(seeds) >= 590_000) & (np.asarray(seeds) <= 592_999)).any():
        raise ValueError("build9 dev / fresh / video seeds")


def polar(xy):
    from nedm.so101_policy.kin import PAN_AXIS_XY
    xy = np.asarray(xy, np.float64)
    d = xy - np.asarray(PAN_AXIS_XY)
    return np.hypot(d[..., 0], d[..., 1]), np.degrees(np.arctan2(d[..., 1], d[..., 0]))


def wrap90_deg(a):
    return (np.asarray(a) + 45.0) % 90.0 - 45.0


def geometry(cube_xy, cube_yaw, goal_xy, goal_yaw):
    cr, ca = polar(cube_xy)
    gr, ga = polar(goal_xy)
    d = np.linalg.norm(np.asarray(goal_xy) - np.asarray(cube_xy), axis=-1)
    turn = wrap90_deg(np.degrees(np.asarray(goal_yaw) - np.asarray(cube_yaw)))
    cin = (cr >= 0.15) & (cr <= 0.27) & (np.abs(ca) <= 75)
    gin = (gr >= 0.15) & (gr <= 0.27) & (np.abs(ga) <= 75)
    din = (d >= 0.06) & (d <= 0.30)
    return dict(cube_r=cr, cube_ang=ca, goal_r=gr, goal_ang=ga, dist=d, turn_deg=turn, cube_in=cin, goal_in=gin,
                dist_in=din, in_dist=cin & gin & din & (np.abs(turn) <= 30.0 + 1e-6))


def make_home(warm, q_home, n=100, noise_deg=0.0, cubes=None, seed0=SEED0["home"]):
    """n tasks from HOME (+ U(-noise, noise) deg per joint). cubes: (xy, yaw, goal_xy, goal_yaw) of an earlier set to
    reuse, else bank cubes + goals by the make_tasks rule."""
    from nedm.so101_policy import task as T
    bank = np.load(BANK)
    N = bank["q"].shape[0]
    byaw, _ = T.cube_yaw_tilt(torch.as_tensor(bank["cube_quat"]).double())
    byaw = byaw.numpy()
    seeds = np.arange(n, dtype=np.int64) + seed0
    check_seeds(seeds)
    Q, P, Qt, R, GX, GY, BI, TR = [], [], [], [], [], [], [], []
    for k, s in enumerate(seeds):
        rng = np.random.default_rng([int(s), 11])
        nrng = np.random.default_rng([int(s), 13, int(round(noise_deg * 100))])
        tries = 0
        while True:
            tries += 1
            if tries > 200:
                raise RuntimeError(f"seed {s}: no start")
            if cubes is None:
                i = int(rng.integers(N))
                xy, cyaw = bank["cube_pos"][i, :2].astype(np.float64), float(byaw[i])
            else:
                i, xy, cyaw = int(cubes["bank_index"][k]), cubes["xy"][k], float(cubes["yaw"][k])
            q0 = q_home + (np.radians(nrng.uniform(-noise_deg, noise_deg, 6)) if noise_deg > 0 else 0.0)
            ok, c, raw, why = warm(q0, xy, cyaw)
            if ok:
                break
            if cubes is not None and noise_deg == 0:
                raise RuntimeError(f"seed {s}: fixed cube warm-up failed {why}")
        if cubes is None:
            gxy, gyaw = goals_like_make_tasks(s, raw)
        else:
            gxy, gyaw = cubes["goal_xy"][k], float(cubes["goal_yaw"][k])
        Q.append(q0), P.append(c["pos"]), Qt.append(c["quat"]), R.append(raw), GX.append(gxy), GY.append(gyaw)
        BI.append(i), TR.append(tries)
    t = pack(seeds, Q, P, Qt, R, GX, GY, bank_index=np.asarray(BI), tries=np.asarray(TR),
             noise_deg=np.float64(noise_deg))
    return t


def make_real(warm, q_home, yaw_key="cube_start_yaw_deg", seed0=SEED0["real"]):
    cal = json.loads(CALIB.read_text())
    eps = cal["episodes"]
    seeds = np.arange(len(eps), dtype=np.int64) + seed0
    check_seeds(seeds)
    Q, P, Qt, R, GX, GY, names, simchk = [], [], [], [], [], [], [], []
    for k, e in enumerate(eps):
        xy = np.asarray(e["cube_start_xyz"][:2], np.float64)
        cyaw = math.radians(float(e[yaw_key])) % (math.pi / 2)
        ok, c, raw, why = warm(q_home, xy, cyaw)
        if not ok:
            raise RuntimeError(f"real episode {k}: warm-up failed {why}")
        Q.append(q_home), P.append(c["pos"]), Qt.append(c["quat"]), R.append(raw)
        GX.append(np.asarray(e["plate_xy"], np.float64)), GY.append(math.radians(float(e["plate_yaw_deg"])) % (math.pi / 2))
        names.append(f"{e['dataset']}/{e['episode']}")
        simchk.append(bool(e["sim_check"]["task_in_sim_distribution"]))
    return pack(seeds, Q, P, Qt, R, GX, GY, episode=np.asarray(names), calib_in_dist=np.asarray(simchk),
                yaw_key=np.array(yaw_key))


def make_real_jitter(warm, q_home, K=20, seed0=596_000, dq_deg=3.0, dxy=0.005, dyaw_deg=10.0):
    """K variants of each real placement: HOME + U(-3, 3) deg per joint, cube and plate xy + U(-5, 5) mm, cube yaw (jaw
    yaw estimate) + U(-10, 10) deg (the calibration's two yaw estimates disagree; real2sim REVIEW), plate yaw as
    measured. Seeds seed0 + K * episode + j."""
    cal = json.loads(CALIB.read_text())
    eps = cal["episodes"]
    seeds = np.arange(len(eps) * K, dtype=np.int64) + seed0
    check_seeds(seeds)
    Q, P, Qt, R, GX, GY, names, simchk = [], [], [], [], [], [], [], []
    for k, e in enumerate(eps):
        for j in range(K):
            rng = np.random.default_rng([int(seeds[k * K + j]), 23])
            q0 = q_home + np.radians(rng.uniform(-dq_deg, dq_deg, 6))
            xy = np.asarray(e["cube_start_xyz"][:2], np.float64) + rng.uniform(-dxy, dxy, 2)
            cyaw = (math.radians(float(e["cube_start_yaw_deg"]) + rng.uniform(-dyaw_deg, dyaw_deg))) % (math.pi / 2)
            ok, c, raw, why = warm(q0, xy, cyaw)
            if not ok:
                raise RuntimeError(f"real episode {k} variant {j}: warm-up failed {why}")
            Q.append(q0), P.append(c["pos"]), Qt.append(c["quat"]), R.append(raw)
            GX.append(np.asarray(e["plate_xy"], np.float64) + rng.uniform(-dxy, dxy, 2))
            GY.append(math.radians(float(e["plate_yaw_deg"])) % (math.pi / 2))
            names.append(f"{e['dataset']}/{e['episode']}")
            simchk.append(bool(e["sim_check"]["task_in_sim_distribution"]))
    return pack(seeds, Q, P, Qt, R, GX, GY, episode=np.asarray(names), calib_in_dist=np.asarray(simchk))


def make_outer(warm, q_home, n=100, seed0=SEED0["outer"]):
    from nedm.so101_policy.kin import PAN_AXIS_XY
    px, py = PAN_AXIS_XY
    seeds = np.arange(n, dtype=np.int64) + seed0
    check_seeds(seeds)
    Q, P, Qt, R, GX, GY = [], [], [], [], [], []
    for s in seeds:
        rng = np.random.default_rng([int(s), 17])

        def pt():
            r, a = rng.uniform(*OUTER_R), math.radians(rng.uniform(-75.0, 75.0))
            return np.array([px + r * math.cos(a), py + r * math.sin(a)])

        while True:
            xy, cyaw = pt(), rng.uniform(0, math.pi / 2)
            ok, c, raw, why = warm(q_home, xy, cyaw)
            if ok:
                break
        for _ in range(10_000):
            g = pt()
            if 0.06 <= np.linalg.norm(g - xy) <= 0.30:
                break
        else:
            raise RuntimeError("no outer goal")
        gyaw = (cyaw + math.radians(rng.uniform(-30.0, 30.0))) % (math.pi / 2)
        Q.append(q_home), P.append(c["pos"]), Qt.append(c["quat"]), R.append(raw), GX.append(g), GY.append(gyaw)
    return pack(seeds, Q, P, Qt, R, GX, GY)


def alt_poses():
    """Candidate fixed start poses (sim rad, 6): bank arm poses with |pan| <= 3 deg, chosen at quantiles of the
    grasp-point height and reach (kinematics of the task env), plus HOME with the jaw at the bank median."""
    from nedm.so101_policy import vec_env as VE
    from nedm.so101_policy.joint_env import JointEnvConfig
    bank = np.load(BANK)
    q = bank["q"].astype(np.float64)
    kin = VE.kinematics(JointEnvConfig().kin_points, "cpu")
    m = np.abs(np.degrees(q[:, 0])) <= 3.0
    qq = q[m]
    gp = kin_grasp(kin, qq)
    rr = np.hypot(gp[:, 0] - 0.0388, gp[:, 1])
    z = gp[:, 2]
    out = {}
    for name, (rq, zq) in dict(alt_low_near=(0.25, 0.25), alt_mid=(0.5, 0.5), alt_high_far=(0.75, 0.75),
                               alt_high_near=(0.25, 0.75), alt_low_far=(0.75, 0.25)).items():
        rt, zt = np.quantile(rr, rq), np.quantile(z, zq)
        j = int(np.argmin(((rr - rt) / rr.std()) ** 2 + ((z - zt) / z.std()) ** 2))
        out[name] = dict(q=qq[j], grasp=gp[j].tolist())
    return out


def kin_grasp(kin, q):
    """Grasp point (world xyz) of poses q [N, 6] through the task env's kinematics."""
    pos, _ = kin.fk(torch.as_tensor(q, dtype=torch.float64))
    return pos.cpu().numpy()


REST_SIM_DEG = (8.3, -90.0, 72.6, 70.6, -71.9, 30.0)   # median mapped real first frame (calib_v1), jaw set to 30 deg


def cubes_of(t):
    return dict(bank_index=t["bank_index"], xy=t["start_cube_pos"][:, :2],
                yaw=np.asarray([2 * math.atan2(qq[3], qq[0]) for qq in t["start_cube_quat"]]),
                goal_xy=t["goal_xy"], goal_yaw=t["goal_yaw"])


def make_ctrl(warm, base):
    """The cubes / goals of ``base`` with the arm start of the training task sampler (``tray.make_tasks``:
    ``Scene.sample_above`` within 40 deg (70 %) or 60 deg (30 %) of the cube direction, warm-up acceptance)."""
    from nedm.contact_nrd.collect import so101_grasp as g
    from nedm.so101_policy.kin import PAN_AXIS_XY
    s = warm.blk.s
    cubes = cubes_of(base)
    seeds = np.asarray(base["seeds"])
    Q, P, Qt, R, TR = [], [], [], [], []
    for k, sd in enumerate(seeds):
        rng = np.random.default_rng([int(sd), 19])
        spread = 40.0 if rng.uniform() < 0.7 else 60.0
        xy, cyaw = cubes["xy"][k], float(cubes["yaw"][k])
        a = math.degrees(math.atan2(xy[1] - PAN_AXIS_XY[1], xy[0] - PAN_AXIS_XY[0]))
        tries = 0
        while True:
            tries += 1
            if tries > 500:
                raise RuntimeError(f"seed {sd}: no start")
            try:
                q0 = s.sample_above(rng, angle=(a - spread, a + spread))
            except g.PlanError:
                continue
            ok, c, raw, _ = warm(q0, xy, cyaw)
            if ok:
                break
        Q.append(q0), P.append(c["pos"]), Qt.append(c["quat"]), R.append(raw), TR.append(tries)
    return pack(seeds, Q, P, Qt, R, cubes["goal_xy"], cubes["goal_yaw"], bank_index=cubes["bank_index"],
                tries=np.asarray(TR))


def cmd_paired(a):
    """Paired comparison of fixed start poses on the SAME cubes / goals: ``<prefix>home`` (bank cubes + make_tasks
    goals from HOME, seeds ``seed0``+), ``<prefix>ctrl`` (training start sampler), ``<prefix>rest`` (the folded real
    rest pose), and the bank-derived alternatives ``alt_poses``."""
    q_home, _ = home_rad()
    d = OUT / "tasks"
    d.mkdir(parents=True, exist_ok=True)
    warm = Warm()
    pre = a.prefix
    if a.seed0 == SEED0["home"] and a.n <= 100:
        base = dict(np.load(d / "home100.npz"))
    else:
        base = make_home(warm, q_home, a.n, seed0=a.seed0)
    np.savez(d / f"{pre}home.npz", **base)
    np.savez(d / f"{pre}ctrl.npz", **make_ctrl(warm, base))
    cubes = cubes_of(base)
    poses = {k: dict(q=v["q"], grasp=v["grasp"]) for k, v in alt_poses().items()}
    poses["rest"] = dict(q=np.radians(np.asarray(REST_SIM_DEG)), grasp=None)
    meta = {}
    for name, p in poses.items():
        try:
            ta = make_home(warm, np.asarray(p["q"], np.float64), a.n, cubes=cubes, seed0=a.seed0)
        except RuntimeError as e:
            log(f"{name}: no tasks ({e})")
            meta[name] = dict(q_deg=np.round(np.degrees(p["q"]), 2).tolist(), error=str(e))
            continue
        np.savez(d / f"{pre}{name}.npz", **ta)
        meta[name] = dict(q_deg=np.round(np.degrees(p["q"]), 2).tolist(), grasp=p["grasp"])
        log(f"{pre}{name}: q deg {meta[name]['q_deg']} grasp {p['grasp']}")
    (d / f"{pre}poses.json").write_text(json.dumps(meta, indent=1))


def cmd_pose(a):
    """One more fixed pose (6 sim deg) on the cubes / goals of ``<prefix>home.npz`` -> ``<prefix><name>.npz``;
    ``--noise`` adds U(-noise, noise) deg per joint per task."""
    d = OUT / "tasks"
    base = dict(np.load(d / f"{a.prefix}home.npz"))
    q = np.radians(np.asarray([float(v) for v in a.deg.split(",")], np.float64))
    t = make_home(Warm(), q, len(base["seeds"]), noise_deg=a.noise, cubes=cubes_of(base), seed0=int(base["seeds"][0]))
    np.savez(d / f"{a.prefix}{a.name}.npz", **t)
    log(f"{a.prefix}{a.name}: q deg {np.round(np.degrees(q), 2).tolist()} noise {a.noise} deg, {len(t['seeds'])} tasks")


def cmd_realpose(a):
    """real15 / realjit300 from another fixed pose (6 sim deg) -> ``real15<suffix>.npz``, ``realjit300<suffix>.npz``
    (same seeds, so the same jitter draws as the HOME sets: paired)."""
    d = OUT / "tasks"
    q = np.radians(np.asarray([float(v) for v in a.deg.split(",")], np.float64))
    warm = Warm()
    np.savez(d / f"real15{a.suffix}.npz", **make_real(warm, q))
    np.savez(d / f"realjit300{a.suffix}.npz", **make_real_jitter(warm, q))
    log(f"real15{a.suffix}, realjit300{a.suffix}: q deg {np.round(np.degrees(q), 2).tolist()}")


def cmd_tasks(a):
    q_home, home_deg = home_rad()
    log(f"HOME sim deg {home_deg} -> rad {np.round(q_home, 4).tolist()}")
    d = OUT / "tasks"
    d.mkdir(parents=True, exist_ok=True)
    warm = Warm()
    t = make_home(warm, q_home, a.n)
    np.savez(d / "home100.npz", **t)
    log(f"home100: tries p50 {np.median(t['tries']):.0f} max {t['tries'].max()}")
    tn = make_home(warm, q_home, a.n, noise_deg=3.0, cubes=cubes_of(t))
    np.savez(d / "homenoise100.npz", **tn)
    log(f"homenoise100: tries p50 {np.median(tn['tries']):.0f} max {tn['tries'].max()}, |dq| max "
        f"{np.degrees(np.abs(tn['start_q'] - q_home).max()):.2f} deg")
    np.savez(d / "real15.npz", **make_real(warm, q_home))
    np.savez(d / "real15sil.npz", **make_real(warm, q_home, "cube_start_yaw_silhouette_deg", SEED0["realsil"]))
    np.savez(d / "outer100.npz", **make_outer(warm, q_home, a.n))
    log("real15, real15sil, outer100 written")
    np.savez(d / "realjit300.npz", **make_real_jitter(warm, q_home))
    log("realjit300 written")


# ---- evaluation --------------------------------------------------------------------------------------------------
def cmd_eval(a):
    from nedm.so101_policy import joint_env as JE
    from nedm.so101_policy import joint_teacher as JT
    from nedm.so101_policy.vision.demos import load_tasks
    path = DEV100 if a.set == "dev100" else OUT / "tasks" / f"{a.set}.npz"
    t = load_tasks(path, np.arange(a.n) if a.n else None)
    B = len(t["seeds"])
    jt = JE.JointTeacher(str(JT_PATH), a.device)
    t0 = time.time()
    env = JT.make_env(a.world, B, jt.codec, a.device, str(NRD), a.workers, str(MJ_PY), 400)
    r = JT.run_tasks(env, t, jt, log=None)
    env.close()
    s = JT.summarize(r)
    s["wall_s"] = time.time() - t0
    r["summary"] = np.array(json.dumps(s))
    r["jt"] = np.array(json.dumps(jt.info()))
    r["world"] = np.array(a.world)
    r["task_file"] = np.array(str(path))
    od = OUT / "evals"
    od.mkdir(parents=True, exist_ok=True)
    tag = f"{a.set}_{a.world}" + (f"_n{a.n}" if a.n else "")
    np.savez_compressed(od / f"{tag}.npz", **r)
    log(f"{tag}: {json.dumps(s)}")


# ---- report ------------------------------------------------------------------------------------------------------
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fmt(k, n):
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:.0f} % ({100 * lo:.0f}-{100 * hi:.0f})"


def cmd_report(a):
    from nedm.so101_policy import task as T
    od = OUT / "evals"
    rows = []
    for f in sorted(od.glob("*.npz")):
        r = np.load(f)
        s = json.loads(str(r["summary"]))
        tf = Path(str(r["task_file"]))
        t = dict(np.load(tf))
        le, st = np.asarray(r["lenient"]).astype(bool), np.asarray(r["success"]).astype(bool)
        n = len(le)
        yaw = np.asarray([2 * math.atan2(q[3], q[0]) for q in t["state_cube_quat"][:n]])
        geo = geometry(t["state_cube_pos"][:n, :2], yaw, t["goal_xy"][:n], t["goal_yaw"][:n])
        causes = {T.ENDS[int(k)]: int(v) for k, v in zip(*np.unique(np.asarray(r["end"])[~le], return_counts=True))}
        lt = np.asarray(r["lenient_t"])[le] * 0.02 if le.any() else np.zeros(0)
        rows.append(dict(file=f.name, n=n, lenient=fmt(int(le.sum()), n), strict=fmt(int(st.sum()), n),
                         causes=causes, lenient_t_p50=float(np.median(lt)) if len(lt) else None,
                         lifted=float(np.mean(r["lifted"]))))
        print(f"{f.name}: lenient {rows[-1]['lenient']}, strict {rows[-1]['strict']}, not lenient: {causes}, "
              f"lenient time p50 {rows[-1]['lenient_t_p50']}")
        if a.detail:
            ang = np.abs(geo["cube_ang"])
            for lo, hi in ((0, 30), (30, 50), (50, 60), (60, 75.1)):
                m = (ang >= lo) & (ang < hi)
                if m.any():
                    print(f"    |cube angle| {lo}-{hi}: lenient {fmt(int(le[m].sum()), int(m.sum()))}")
            for lo, hi in ((0.15, 0.20), (0.20, 0.24), (0.24, 0.27), (0.27, 0.30), (0.30, 0.345)):
                m = (geo["cube_r"] >= lo) & (geo["cube_r"] < hi)
                if m.any():
                    print(f"    cube r {lo}-{hi}: lenient {fmt(int(le[m].sum()), int(m.sum()))}")
            for lo, hi in ((0.15, 0.20), (0.20, 0.24), (0.24, 0.27), (0.27, 0.30), (0.30, 0.345)):
                m = (geo["goal_r"] >= lo) & (geo["goal_r"] < hi)
                if m.any():
                    print(f"    goal r {lo}-{hi}: lenient {fmt(int(le[m].sum()), int(m.sum()))}")
        if "episode" in t and n <= 20:
            for k in range(n):
                print(f"    {t['episode'][k]:<26} cube r {geo['cube_r'][k]:.3f} ang {geo['cube_ang'][k]:6.1f} | goal r "
                      f"{geo['goal_r'][k]:.3f} ang {geo['goal_ang'][k]:6.1f} | d {geo['dist'][k]:.3f} turn "
                      f"{geo['turn_deg'][k]:6.1f} | in-dist {bool(geo['in_dist'][k])} (calib "
                      f"{bool(t['calib_in_dist'][k])}) | lenient {int(le[k])} strict {int(st[k])} end "
                      f"{T.ENDS[int(r['end'][k])]} t {float(r['lenient_t'][k]) * 0.02 if le[k] else float('nan'):.2f}")
        if "episode" in t and n > 20:
            names = list(dict.fromkeys(t["episode"].tolist()))
            print("    per real placement (jittered variants), lenient / strict:")
            for nm in names:
                m = t["episode"] == nm
                ends = {T.ENDS[int(k)]: int(v) for k, v in zip(*np.unique(np.asarray(r["end"])[m & ~le],
                                                                          return_counts=True))}
                print(f"      {nm:<26} cube r {np.median(geo['cube_r'][m]):.3f} goal r {np.median(geo['goal_r'][m]):.3f} "
                      f"in-dist {bool(t['calib_in_dist'][m][0])}: lenient {int(le[m].sum())}/{int(m.sum())} strict "
                      f"{int(st[m].sum())}/{int(m.sum())} {ends}")
            for flag in (True, False):
                m = t["calib_in_dist"] == flag
                print(f"    in-dist {flag}: lenient {fmt(int(le[m].sum()), int(m.sum()))}, strict "
                      f"{fmt(int(st[m].sum()), int(m.sum()))}")
        if "outer" in f.name or "real" in f.name:
            rmax = np.maximum(geo["cube_r"], geo["goal_r"])
            for lo, hi in ((0.0, 0.27), (0.27, 0.29), (0.29, 0.31), (0.31, 0.345)):
                m = (rmax >= lo) & (rmax < hi)
                if m.any():
                    print(f"    max(cube r, goal r) {lo}-{hi}: lenient {fmt(int(le[m].sum()), int(m.sum()))}, strict "
                          f"{fmt(int(st[m].sum()), int(m.sum()))}")
            for lo, hi in ((0.27, 0.29), (0.29, 0.31), (0.31, 0.345)):
                m = (geo["cube_r"] >= lo) & (geo["cube_r"] < hi) & (geo["goal_r"] < 0.31)
                if m.any():
                    print(f"    cube r {lo}-{hi} with goal r < 0.31: lenient {fmt(int(le[m].sum()), int(m.sum()))}")
    (OUT / "summary.json").write_text(json.dumps(rows, indent=1))
    paired(od, T)


def mcnemar_exact(b, c):
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(od, T):
    """Paired lenient outcomes of every p500_* (f500_*) set against p500_home (f500_home): same cubes / goals."""
    for pre, world in [(p_, w_) for p_ in ("p500_", "f500_", "v500_") for w_ in ("mujoco", "nrd")]:
        ref = od / f"{pre}home_{world}.npz"
        if not ref.exists():
            continue
        h = np.asarray(np.load(ref)["lenient"]).astype(bool)
        hs = np.asarray(np.load(ref)["success"]).astype(bool)
        print(f"paired vs {pre}home ({world}, n {len(h)}; home lenient {fmt(int(h.sum()), len(h))}, strict "
              f"{fmt(int(hs.sum()), len(h))}):")
        for f in sorted(od.glob(f"{pre}*_{world}.npz")):
            if f == ref:
                continue
            r = np.load(f)
            x = np.asarray(r["lenient"]).astype(bool)
            xs = np.asarray(r["success"]).astype(bool)
            b, c = int((h & ~x).sum()), int((~h & x).sum())
            causes = {T.ENDS[int(k)]: int(v) for k, v in zip(*np.unique(np.asarray(r["end"])[~x], return_counts=True))}
            print(f"  {f.stem:<28} lenient {fmt(int(x.sum()), len(x))} strict {fmt(int(xs.sum()), len(x))} | home only "
                  f"{b}, this only {c}, exact McNemar p {mcnemar_exact(b, c):.3g} | not lenient {causes}")
    for base, other in (("real15", "real15_r45"), ("realjit300", "realjit300_r45")):
        for world in ("mujoco", "nrd"):
            fa, fb = od / f"{base}_{world}.npz", od / f"{other}_{world}.npz"
            if fa.exists() and fb.exists():
                xa = np.asarray(np.load(fa)["lenient"]).astype(bool)
                xb = np.asarray(np.load(fb)["lenient"]).astype(bool)
                b, c = int((xa & ~xb).sum()), int((~xa & xb).sum())
                print(f"  {other} vs {base} ({world}): {fmt(int(xb.sum()), len(xb))} vs {fmt(int(xa.sum()), len(xa))}, "
                      f"{base} only {b}, {other} only {c}, exact McNemar p {mcnemar_exact(b, c):.3g}")
    for name in ("dev100", "home100", "homenoise100", "real15", "real15sil", "realjit300", "outer100", "p500_home",
                 "p500_ctrl", "p500_rest", "v500_home", "v500_home_roll45", "real15_r45", "realjit300_r45"):
        fm, fn = od / f"{name}_mujoco.npz", od / f"{name}_nrd.npz"
        if fm.exists() and fn.exists():
            m = np.asarray(np.load(fm)["lenient"]).astype(bool)
            nn = np.asarray(np.load(fn)["lenient"]).astype(bool)
            print(f"  NRD vs MuJoCo agreement {name}: both {int((m & nn).sum())}, NRD only {int((nn & ~m).sum())}, "
                  f"MuJoCo only {int((m & ~nn).sum())}, neither {int((~m & ~nn).sum())}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("tasks")
    q.add_argument("--n", type=int, default=100)
    q.set_defaults(fn=cmd_tasks)
    q = sub.add_parser("paired")
    q.add_argument("--n", type=int, default=500)
    q.add_argument("--seed0", type=int, default=595_000)
    q.add_argument("--prefix", default="p500_")
    q.set_defaults(fn=cmd_paired)
    q = sub.add_parser("pose")
    q.add_argument("--prefix", default="p500_")
    q.add_argument("--name", required=True)
    q.add_argument("--deg", required=True, help="6 sim deg, comma-separated")
    q.add_argument("--noise", type=float, default=0.0)
    q.set_defaults(fn=cmd_pose)
    q = sub.add_parser("realpose")
    q.add_argument("--suffix", required=True)
    q.add_argument("--deg", required=True, help="6 sim deg, comma-separated")
    q.set_defaults(fn=cmd_realpose)
    q = sub.add_parser("eval")
    q.add_argument("--set", required=True)
    q.add_argument("--world", default="mujoco", choices=("mujoco", "nrd"))
    q.add_argument("--n", type=int, default=0)
    q.add_argument("--workers", type=int, default=8)
    q.add_argument("--device", default="cuda")
    q.set_defaults(fn=cmd_eval)
    q = sub.add_parser("report")
    q.add_argument("--detail", action="store_true")
    q.set_defaults(fn=cmd_report)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
