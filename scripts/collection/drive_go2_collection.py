"""Stratified Go2 collection driver. Rigid in parallel, CRM sequential."""
import json, os, subprocess, sys, random
# Paths differ per box (dorm-pc has no "sbel/" segment, and its env is "nedm" not
# "nedm-src"), so every machine-specific path is an env var with this box's value as
# the fallback. Without this the driver silently runs the wrong interpreter.
REPO = os.environ.get("NEDM_REPO", "/home/kyle/Documents/sbel/NeDM")
sys.path.insert(0, f"{REPO}/src")
from nedm.quadruped.imported_policy import (COMMAND_FAMILIES, FAMILY_PARAMS,
                                            stratified_params, family_seed)

TERRAIN, N_PER_FAM, CONC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
# The comprehensive collection. Off by default so every earlier invocation
# reproduces; NEDM_WIDE=1 switches on the measured command envelope, 40 s
# episodes, the 200 m ground and the diversity mechanisms together, because they
# are one design and mixing halves of it would be neither dataset.
WIDE = os.environ.get("NEDM_WIDE", "0") == "1"
DURATION_S = float(os.environ.get("NEDM_DURATION_S", "41.25" if WIDE else "16"))
GROUND_M = float(os.environ.get("NEDM_GROUND_M", "200" if WIDE else "10"))
# Peak trunk impulse per episode, STRATIFIED ACROSS EPISODES AND INCLUDING ZERO,
# so the set spans clean-and-directly-comparable through heavily-disturbed rather
# than being uniformly noisy. 0 to 120 N against a ~158 N robot: the top of the
# range falls it, which is where fall coverage comes from at no extra mechanism.
PERTURB_MAX_N = float(os.environ.get("NEDM_PERTURB_MAX_N", "120"))
SEED_OFFSET = int(sys.argv[4]) if len(sys.argv) > 4 else 0
# The offset goes in the directory name as well as the metadata: metadata makes the
# origin recoverable, a path makes it obvious, and two boxes writing identically
# named directories is how a merge silently overwrites. Offset 0 is unsuffixed, so
# the baseline collection keeps its name and an unsuffixed directory MEANS offset 0.
_suffix = "" if SEED_OFFSET == 0 else f"_off{SEED_OFFSET}"
ROOT = os.environ.get("NEDM_DATASET_ROOT",
                      f"/home/kyle/sbel-artifacts/datasets/go2_stratified{_suffix}")
PY = os.environ.get("NEDM_PY", "/home/kyle/miniconda3/envs/nedm-src/bin/python")
CKPT = os.environ.get("NEDM_GO2_CKPT", "/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt")
os.makedirs(f"{ROOT}/logs", exist_ok=True)
env = dict(os.environ,
           PYTHONPATH=os.environ.get("NEDM_CHRONO_PYTHONPATH",
                                     "/home/kyle/Documents/sbel/chrono-build/bin"),
           NEDM_SEED_OFFSET=str(SEED_OFFSET))

def spawn_for(fam, p, terrain):
    """Spawn so the episode has room in the direction it will actually travel.

    vx and vy are now BOTH SIGNS, so a fixed spawn would give one sign the full
    bed and the other almost none -- which is how the old lateral family ended up
    100% truncated. Place the robot at the far end of its own travel direction.
    """
    rng = random.Random(hash((fam, json.dumps(p, sort_keys=True))) & 0xffffffff)
    vx = p.get("vx", p.get("vx0", 0.0)); vy = p.get("vy", 0.0)
    if terrain == "rigid":
        x = -3.0 if vx >= 0 else 3.0
        y = -3.0 if vy > 0 else (3.0 if vy < 0 else rng.uniform(-1, 1))
        return x + rng.uniform(-0.5, 0.5), y + rng.uniform(-0.3, 0.3), rng.uniform(-15, 15), 4.0
    # CRM: bed x [-0.6, 7.4] usable [0.2, 6.6]; lateral gets the wide bed
    x = 0.9 if vx >= 0 else 5.5
    if fam == "lateral":
        py = 8.0; y = -3.0 if vy > 0 else 3.0
    else:
        py = 4.0; y = rng.uniform(-0.2, 0.2)
    return x + rng.uniform(-0.3, 0.3), y, rng.uniform(-10, 10), py

jobs = []
for fam in COMMAND_FAMILIES:
    for i, p in enumerate(stratified_params(fam, N_PER_FAM, seed=family_seed(fam, SEED_OFFSET), wide=WIDE)):
        x, y, h, py = spawn_for(fam, p, TERRAIN)
        jobs.append((fam, i, p, x, y, h, py))
print(f"{len(jobs)} episodes: {len(COMMAND_FAMILIES)} families x {N_PER_FAM}")

def cmd(j):
    fam, i, p, x, y, h, py = j
    extra = []
    if WIDE:
        # One stratified bin per episode index, including exactly zero at i==0.
        peak = PERTURB_MAX_N * (i % 6) / 5.0
        tilt_rng = random.Random(family_seed(fam, SEED_OFFSET) + 977 * i)
        extra = ["--ground-size-m", f"{GROUND_M}",
                 "--perturb-peak-n", f"{peak:.1f}",
                 "--prewalk-s", f"{tilt_rng.uniform(0.0, 3.0):.2f}",
                 "--ground-tilt-roll-deg", f"{tilt_rng.uniform(-3.0, 3.0):.2f}",
                 "--ground-tilt-pitch-deg", f"{tilt_rng.uniform(-3.0, 3.0):.2f}"]
    return [PY, "scripts/collection/collect_go2_smoke.py", "--terrain", TERRAIN,
            "--duration-s", f"{DURATION_S}", "--imported-ckpt", CKPT, "--command-family", fam,
            *extra,
            "--command-params", json.dumps(p), "--episode-index", str(i),
            "--seed", str(1000 + i), "--spawn-x-m", f"{x:.4f}", "--spawn-y-m", f"{y:.4f}",
            "--heading-deg", f"{h:.2f}", "--patch-y", f"{py}",
            "--output-dir", f"{ROOT}/{TERRAIN}_{fam}_{i}", "--overwrite",
            "--progress-interval-s", "99"]

running = []
for n, j in enumerate(jobs):
    log = open(f"{ROOT}/logs/{TERRAIN}_{j[0]}_{j[1]}.log", "w")
    running.append(subprocess.Popen(cmd(j), stdout=log, stderr=log, env=env,
                                    cwd=REPO))
    if len(running) >= CONC:
        for r in running: r.wait()
        running = []
        print(f"  {n+1}/{len(jobs)}", flush=True)
for r in running: r.wait()
print("COLLECT_DONE")
