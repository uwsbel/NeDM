"""Bracket torque-perturbation magnitude AND verify it reaches the target modes.

Two questions in one run, because they share the episodes:
  (a) where does yield collapse -- torque has NO prior, unlike force which is
      measured flat to 120 N, so it must be bracketed from zero rather than
      extrapolated; the units and the failure mechanism both differ.
  (b) does it actually excite modes 3 (fl+fr, FRONT) and 12 (rl+rr, REAR), which a
      trot structurally cannot reach and which measure 0.91% and 0.06% of the
      existing 3.39M transitions.

The zero-torque level is the control. If target-mode occupancy does not rise above
it, the channel is not doing what it was added for, whatever the yield says.

Contact mode = fl*1 + fr*2 + rl*4 + rr*8, matching dorm-pc's numbering.
"""
import json, os, subprocess, sys, random
REPO = "/home/kyle/Documents/sbel/NeDM"
SEED_OFFSET = 5000000
ROOT = os.environ.get("NEDM_TQ_ROOT", "/home/kyle/sbel-artifacts/datasets/go2_torque_bracket_v2")
PY = "/home/kyle/miniconda3/envs/nedm-src/bin/python"
CKPT = "/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt"
LEVELS = [float(x) for x in os.environ.get("NEDM_TQ_LEVELS","0,5,10,20,40").split(",")]     # 80 N.m topples the robot at 5.6 s; see notes
N_PER = int(os.environ.get("NEDM_TQ_NPER", "8"))
CMD_SIGN = float(os.environ.get("NEDM_TQ_CMD_SIGN", "-1"))
# AXIS: "torque" sweeps the torque peak; "tilt" holds torque fixed and sweeps the
# ground pitch instead. REAR (front feet up) needs a SUSTAINED rearward weight shift,
# which a gravity tilt gives and a torque impulse does not -- the policy resists an
# impulse and recovers, but cannot recover from a slope.
AXIS = os.environ.get("NEDM_TQ_AXIS", "torque")
# The collection draws roll and pitch ~U(-3,3) deg per episode; the probe did not.
TILT_RAND = os.environ.get("NEDM_TQ_TILT_RAND") == "1"
FIXED_TQ = float(os.environ.get("NEDM_TQ_FIXED", "40"))
CONC = 8
os.makedirs(f"{ROOT}/logs", exist_ok=True)
env = dict(os.environ, NEDM_STAND_HIP=os.environ.get("NEDM_STAND_HIP","0"), PYTHONPATH="/home/kyle/Documents/sbel/chrono-build/bin",
           NEDM_SEED_OFFSET=str(SEED_OFFSET))

jobs = []
for li, tq in enumerate(LEVELS):
    for r in range(N_PER):
        rng = random.Random(SEED_OFFSET + 7919 * li + r)
        jobs.append(dict(tq=tq, li=li, r=r,
                         vx=round(rng.uniform(0.25, 0.75), 4),
                         prewalk=rng.uniform(0.0, 3.0),
                         roll=rng.uniform(-3.0, 3.0), pitch=rng.uniform(-3.0, 3.0),
                         x=3.0 + rng.uniform(-0.5, 0.5), y=rng.uniform(-1.0, 1.0),
                         h=rng.uniform(-15.0, 15.0), seed=2000 + 100 * li + r))
print(f"{len(jobs)} episodes: {len(LEVELS)} torque levels x {N_PER}", flush=True)

def cmd(j):
    tag = f"t{j['li']}_r{j['r']}"
    return [PY, "scripts/collection/collect_go2_smoke.py", "--terrain", "rigid",
            "--duration-s", "41.25", "--imported-ckpt", CKPT,
            "--command-family", "constant",
            "--command-params", json.dumps({"vx": CMD_SIGN * j["vx"]}),
            "--ground-size-m", "200",
            "--perturb-peak-n", "0.0",                 # ISOLATE torque: no linear push
            "--perturb-torque-peak-nm",
            f"{(FIXED_TQ if AXIS == 'tilt' else j['tq']):.1f}",
            *(["--ground-tilt-pitch-deg", f"{j['tq']:.1f}"] if AXIS == "tilt" else
              ["--ground-tilt-roll-deg", f"{j['roll']:.2f}",
               "--ground-tilt-pitch-deg", f"{j['pitch']:.2f}"] if TILT_RAND else []),
            "--prewalk-s", f"{j['prewalk']:.2f}",
            "--episode-index", str(j["li"] * 100 + j["r"]), "--seed", str(j["seed"]),
            "--spawn-x-m", f"{j['x']:.4f}", "--spawn-y-m", f"{j['y']:.4f}",
            "--heading-deg", f"{j['h']:.2f}", "--patch-y", "4.0",
            "--output-dir", f"{ROOT}/rigid_{tag}", "--overwrite",
            "--progress-interval-s", "99"]

running = []
for n, j in enumerate(jobs):
    log = open(f"{ROOT}/logs/{j['li']}_{j['r']}.log", "w")
    running.append(subprocess.Popen(cmd(j), stdout=log, stderr=log, env=env, cwd=REPO))
    if len(running) >= CONC:
        for p in running: p.wait()
        running = []
        print(f"  {n+1}/{len(jobs)}", flush=True)
for p in running: p.wait()
print("TORQUE_BRACKET_DONE", flush=True)
