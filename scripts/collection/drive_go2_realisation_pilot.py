#!/usr/bin/env python3
"""Command-realisation pilot: does the robot actually reach what it is commanded?

Two arms that differ ONLY in applied perturbation amplitude:

    treated   --perturb-peak-n 60 --perturb-scale 1.0
    control   --perturb-peak-n 60 --perturb-scale 0.0

Both take the same random draws at the same times (see --perturb-scale). Building
the control with --perturb-peak-n 0 would skip the impulse block's draws entirely
and desynchronise the two arms' random streams.

vel_body_x and yaw_rate are swept SEPARATELY -- `constant` moves only x, `pivot`
turns in place -- because a mixed command cannot attribute a shortfall to either.
Both channels are untried, so neither is assumed to behave like the other.

Levels span the policy's TRAINED range (vx +-0.5, wz +-1.0), not its deployment
clip. A ceiling found outside the trained range says nothing about the envelope
the dataset should cover.
"""
import argparse, hashlib, itertools, json, os, subprocess, sys, time

PY = "/home/kyle/miniconda3/envs/nedm-src/bin/python"
CK = "/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt"
VX = [round(-0.5 + 0.1 * i, 2) for i in range(11)]      # trained range, 11 levels
WZ = [round(-1.0 + 0.2 * i, 2) for i in range(11)]
ARMS = {"treated": "1.0", "control": "0.0"}


def cell_seed(chan, lv):
    """Stable per-cell seed offset. NOT hash() -- str.__hash__ is salted per process
    unless PYTHONHASHSEED is set, so a rerun would draw different perturbation times
    for the same cell and the two arms would not pair. This is the defect removed
    from drive_go2_collection.py in the sha256 seeding fix; it is easy to write again.
    """
    h = hashlib.sha256(f"{chan}:{lv:+.2f}".encode()).hexdigest()
    return int(h[:8], 16) % 100000


def jobs(reps):
    out = []
    for arm, scale in ARMS.items():
        for rep in range(reps):
            for lv in VX:
                out.append((arm, scale, "constant", {"vx": lv}, "vx", lv, rep))
            for lv in WZ:
                out.append((arm, scale, "pivot", {"wz": lv}, "wz", lv, rep))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--duration-s", type=float, default=25.0)
    ap.add_argument("--peak-n", type=float, default=60.0)
    ap.add_argument("--seed-base", type=int, default=7700000)
    a = ap.parse_args()

    J = jobs(a.reps)
    print(f"{len(J)} episodes: {len(ARMS)} arms x 2 channels x "
          f"{len(VX)} levels x {a.reps} reps")
    t0 = time.time()
    for i in range(0, len(J), a.concurrency):
        procs = []
        for k, (arm, scale, fam, params, chan, lv, rep) in enumerate(J[i:i + a.concurrency]):
            idx = i + k
            od = os.path.join(a.out_root, arm, f"{chan}_{lv:+.2f}_r{rep}")
            cmd = [PY, "scripts/collection/collect_go2_smoke.py", "--terrain", "rigid",
                   "--duration-s", str(a.duration_s), "--imported-ckpt", CK,
                   "--command-family", fam, "--command-params", json.dumps(params),
                   "--ground-size-m", "200.0",
                   "--perturb-peak-n", f"{a.peak_n:.1f}", "--perturb-scale", scale,
                   "--prewalk-s", "1.00",
                   "--ground-tilt-roll-deg", "0.00", "--ground-tilt-pitch-deg", "0.00",
                   "--episode-index", str(idx),
                   # SAME SEED FOR BOTH ARMS at the same (channel, level, rep), so the
                   # pair differs only in applied force. The seed keys on the cell, not
                   # on the running index, which differs between arms.
                   "--seed", str(a.seed_base + 977 * rep + cell_seed(chan, lv)),
                   "--spawn-x-m", "0.0", "--spawn-y-m", "0.0", "--heading-deg", "0.0",
                   "--output-dir", od, "--overwrite", "--progress-interval-s", "99"]
            procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL))
        for p in procs:
            p.wait()
        print(f"  {min(i + a.concurrency, len(J))}/{len(J)}  "
              f"({time.time() - t0:.0f}s)", flush=True)
    print("PILOT_DONE")


if __name__ == "__main__":
    main()
