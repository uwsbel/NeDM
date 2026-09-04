"""Stratified Go2 collection driver. Rigid in parallel, CRM sequential."""
import json, os, subprocess, sys, random
sys.path.insert(0, "/home/kyle/Documents/sbel/NeDM/src")
from nedm.quadruped.imported_policy import (COMMAND_FAMILIES, FAMILY_PARAMS,
                                            stratified_params, family_seed)

TERRAIN, N_PER_FAM, CONC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
SEED_OFFSET = int(sys.argv[4]) if len(sys.argv) > 4 else 0
ROOT = "/home/kyle/sbel-artifacts/datasets/go2_stratified"; PY = "/home/kyle/miniconda3/envs/nedm-src/bin/python"
CKPT = "/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt"
os.makedirs(f"{ROOT}/logs", exist_ok=True)
env = dict(os.environ, PYTHONPATH="/home/kyle/Documents/sbel/chrono-build/bin")

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
    for i, p in enumerate(stratified_params(fam, N_PER_FAM, seed=family_seed(fam, SEED_OFFSET))):
        x, y, h, py = spawn_for(fam, p, TERRAIN)
        jobs.append((fam, i, p, x, y, h, py))
print(f"{len(jobs)} episodes: {len(COMMAND_FAMILIES)} families x {N_PER_FAM}")

def cmd(j):
    fam, i, p, x, y, h, py = j
    return [PY, "scripts/collection/collect_go2_smoke.py", "--terrain", TERRAIN,
            "--duration-s", "16", "--imported-ckpt", CKPT, "--command-family", fam,
            "--command-params", json.dumps(p), "--episode-index", str(i),
            "--seed", str(1000 + i), "--spawn-x-m", f"{x:.4f}", "--spawn-y-m", f"{y:.4f}",
            "--heading-deg", f"{h:.2f}", "--patch-y", f"{py}",
            "--output-dir", f"{ROOT}/{TERRAIN}_{fam}_{i}", "--overwrite",
            "--progress-interval-s", "99"]

running = []
for n, j in enumerate(jobs):
    log = open(f"{ROOT}/logs/{TERRAIN}_{j[0]}_{j[1]}.log", "w")
    running.append(subprocess.Popen(cmd(j), stdout=log, stderr=log, env=env,
                                    cwd="/home/kyle/Documents/sbel/NeDM"))
    if len(running) >= CONC:
        for r in running: r.wait()
        running = []
        print(f"  {n+1}/{len(jobs)}", flush=True)
for r in running: r.wait()
print("COLLECT_DONE")
