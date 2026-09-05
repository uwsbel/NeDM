"""Replicate a FIXED command across independent realisations.

WHY THIS IS NOT drive_go2_collection.py WITH A DIFFERENT SEED. In the driver the
spawn comes from hash((family, params)) and the tilt, prewalk and perturbation peak
come from the EPISODE INDEX -- none of them from --seed. Varying only --seed would
hold the whole realisation fixed and measure almost nothing, while looking like a
replicate set. So this varies the realisation the way the collection itself does,
at a command held constant.

WHAT IT MEASURES, precisely. Spread of the outcome at a fixed command across
realisations. That splits the baseline's residual (after family and command
magnitude are removed) into command-specific structure, which a paired design
cancels, and realisation noise, which it does not. It is an INDICATOR for the
treatment-by-realisation interaction, not a measurement of it: the interaction
needs both arms and the fine-tuned arm does not exist yet.
"""
import json, os, subprocess, sys, random
REPO = os.environ.get("NEDM_REPO", "/home/kyle/Documents/sbel/NeDM")
SEED_OFFSET = 4000000                      # unused; recorded in the storage doc
ROOT = os.environ.get("NEDM_PROBE_ROOT",
        f"/home/kyle/sbel-artifacts/datasets/go2_seedprobe_off{SEED_OFFSET}")
PY = "/home/kyle/miniconda3/envs/nedm-src/bin/python"
CKPT = os.environ.get("NEDM_GO2_CKPT",
        "/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt")
DURATION_S, GROUND_M, PERTURB_MAX_N, CONC = 41.25, 200.0, 120.0, 8
N_CMD, N_REP = 8, 8
os.makedirs(f"{ROOT}/logs", exist_ok=True)
env = dict(os.environ, PYTHONPATH="/home/kyle/Documents/sbel/chrono-build/bin",
           NEDM_SEED_OFFSET=str(SEED_OFFSET))

# Backward-low band, the cell the criterion has to live in.
CMDS = [round(-0.03 - i * (0.17 - 0.03) / (N_CMD - 1), 4) for i in range(N_CMD)]
jobs = []
for ci, vx in enumerate(CMDS):
    for r in range(N_REP):
        rng = random.Random(SEED_OFFSET + 7919 * ci + r)
        # Same distributions the driver uses, but drawn per REPLICATE so the
        # realisation actually varies at fixed command.
        peak = PERTURB_MAX_N * (r % 6) / 5.0
        jobs.append(dict(vx=vx, ci=ci, r=r, peak=peak,
                         prewalk=rng.uniform(0.0, 3.0),
                         roll=rng.uniform(-3.0, 3.0), pitch=rng.uniform(-3.0, 3.0),
                         x=3.0 + rng.uniform(-0.5, 0.5), y=rng.uniform(-1.0, 1.0),
                         h=rng.uniform(-15.0, 15.0), seed=1000 + 100 * ci + r))
print(f"{len(jobs)} episodes: {N_CMD} commands x {N_REP} realisations", flush=True)

def cmd(j):
    tag = f"c{j['ci']}_r{j['r']}"
    return [PY, "scripts/collection/collect_go2_smoke.py", "--terrain", "rigid",
            "--duration-s", f"{DURATION_S}", "--imported-ckpt", CKPT,
            "--command-family", "constant",
            "--command-params", json.dumps({"vx": j["vx"]}),
            "--ground-size-m", f"{GROUND_M}", "--perturb-peak-n", f"{j['peak']:.1f}",
            "--prewalk-s", f"{j['prewalk']:.2f}",
            "--ground-tilt-roll-deg", f"{j['roll']:.2f}",
            "--ground-tilt-pitch-deg", f"{j['pitch']:.2f}",
            "--episode-index", str(j["ci"] * 100 + j["r"]), "--seed", str(j["seed"]),
            "--spawn-x-m", f"{j['x']:.4f}", "--spawn-y-m", f"{j['y']:.4f}",
            "--heading-deg", f"{j['h']:.2f}", "--patch-y", "4.0",
            "--output-dir", f"{ROOT}/rigid_{tag}", "--overwrite",
            "--progress-interval-s", "99"]

running = []
for n, j in enumerate(jobs):
    log = open(f"{ROOT}/logs/rigid_c{j['ci']}_r{j['r']}.log", "w")
    running.append(subprocess.Popen(cmd(j), stdout=log, stderr=log, env=env, cwd=REPO))
    if len(running) >= CONC:
        for p in running: p.wait()
        running = []
        print(f"  {n+1}/{len(jobs)}", flush=True)
for p in running: p.wait()
print("PROBE_DONE", flush=True)
