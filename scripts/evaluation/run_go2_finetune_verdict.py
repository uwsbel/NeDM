"""Paired fine-tune verdict: both arms, identical episodes, one pass.

THE BASELINE ARM IS NOT RE-RUN. Existing collected episodes replay BIT-IDENTICALLY
from recorded metadata plus index-derived parameters, verified by sha256 before
anything else happens. So the collected episode IS the baseline arm, the treated arm
runs the same specs with only --imported-ckpt differing, and the pairing is exact
rather than nominal.

WHAT HAD TO BE RECONSTRUCTED, AND WHY IT IS SAFE. The collection records spawn,
heading, seed, duration and command, but NOT the three diversity mechanisms --
ground tilt, prewalk and perturbation peak. Those are derived deterministically in
the driver from (family, episode index, seed_offset) using integer seeds, so they
reconstruct exactly. Spawn could NOT have been reconstructed: it was drawn from
Python's per-process-randomised hash() until 2026-09-04, which is why it is taken
from metadata rather than recomputed.

THE REPLAY CHECK IS THE ABORT CONDITION. If a sampled baseline episode does not
reproduce bit-for-bit, the arms are not on identical episodes, the pairing is void,
and the criterion says to report "not measurable at this n" rather than a null. This
script refuses to produce a verdict in that case.
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os, random, subprocess, sys
from math import comb
import numpy as np

sys.path.insert(0, "src")
from nedm.quadruped.imported_policy import family_seed

BASE_CKPT = "/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt"
PY = "/home/kyle/miniconda3/envs/nedm-src/bin/python"
CHRONO = "/home/kyle/Documents/sbel/chrono-build/bin"
PERTURB_MAX_N, GROUND_M, SCORED_ROWS, LEAD_IN_S = 120.0, 200.0, 1000, 5.0


def episode_spec(json_path):
    """Full replayable spec, or None if the directory name is not parseable."""
    m = json.load(open(json_path))
    d = os.path.basename(os.path.dirname(os.path.dirname(json_path)))
    if not d.startswith("rigid_"):
        return None
    fam, idx = d[len("rigid_"):].rsplit("_", 1)
    idx = int(idx); off = int(m.get("seed_offset", 0))
    tr = random.Random(family_seed(fam, off) + 977 * idx)
    sx, sy = m["spawn_m"][0], m["spawn_m"][1]
    return dict(json=json_path, csv=json_path.replace(".json", ".csv"), fam=fam, idx=idx,
                params=m["command_params"], duration=m["duration_s"], seed=m["seed"],
                spawn_x=sx, spawn_y=sy, heading=m["heading_deg"],
                peak=PERTURB_MAX_N * (idx % 6) / 5.0, prewalk=tr.uniform(0.0, 3.0),
                roll=tr.uniform(-3.0, 3.0), pitch=tr.uniform(-3.0, 3.0), off=off)


def scored(csv_path):
    """(abs velocity error, ratio) over the scored window, or None if predicate fails."""
    try:
        rows = list(csv.DictReader(open(csv_path)))
    except Exception:
        return None
    if len(rows) < SCORED_ROWS + 500:
        return None
    c = np.array([float(r["cmd_vx_mps"]) for r in rows])
    v = np.array([float(r["vel_body_x_mps"]) for r in rows])
    if not (np.isfinite(c).all() and np.isfinite(v).all()):
        return None
    w = c[-SCORED_ROWS:]
    if w.std() > 1e-6:
        return None
    cmd = float(w[0])
    k = len(c) - SCORED_ROWS
    while k > 0 and abs(c[k - 1] - cmd) <= 1e-6:
        k -= 1
    if (len(c) - SCORED_ROWS - k) * 0.01 < LEAD_IN_S:      # clause 3b
        return None
    ach = float(v[-SCORED_ROWS:].mean())
    # |cmd| ~ 0 is STAND STILL, a different measurement scored on drift, and the
    # ratio has no denominator there. It is excluded by the cell filter below, but
    # the ratio must not be formed at all or the selection dies on the first
    # stationary episode -- which is what happened the first time this ran.
    ratio = ach / cmd if abs(cmd) > 1e-9 else float("nan")
    return ach - cmd, ratio, cmd


def arm_cmd(spec, ckpt, outdir):
    return [PY, "scripts/collection/collect_go2_smoke.py", "--terrain", "rigid",
           "--duration-s", str(spec["duration"]), "--imported-ckpt", ckpt,
           "--command-family", spec["fam"], "--command-params", json.dumps(spec["params"]),
           "--ground-size-m", str(GROUND_M), "--perturb-peak-n", f"{spec['peak']:.1f}",
           "--prewalk-s", f"{spec['prewalk']:.2f}",
           "--ground-tilt-roll-deg", f"{spec['roll']:.2f}",
           "--ground-tilt-pitch-deg", f"{spec['pitch']:.2f}",
           "--episode-index", str(spec["idx"]), "--seed", str(spec["seed"]),
           "--spawn-x-m", str(spec["spawn_x"]), "--spawn-y-m", str(spec["spawn_y"]),
           "--heading-deg", str(spec["heading"]), "--patch-y", "4.0",
            "--output-dir", outdir, "--overwrite", "--progress-interval-s", "99"]


def arm_env(spec):
    return dict(os.environ, PYTHONPATH=CHRONO, NEDM_SEED_OFFSET=str(spec["off"]))


def run_arm(spec, ckpt, outdir):
    """Blocking single-episode run, used by the replay check."""
    os.makedirs(outdir, exist_ok=True)
    p = subprocess.run(arm_cmd(spec, ckpt, outdir), env=arm_env(spec),
                       capture_output=True, text=True)
    got = glob.glob(f"{outdir}/episodes/*.csv")
    return (got[0] if got and p.returncode == 0 else None)


def popen_arm(spec, ckpt, outdir):
    """Same invocation as run_arm, launched without blocking so a batch runs in parallel."""
    os.makedirs(outdir, exist_ok=True)
    return subprocess.Popen(arm_cmd(spec, ckpt, outdir), env=arm_env(spec),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def exact_median_ci(x, alpha=0.05):
    x = np.sort(np.asarray(x)); n = len(x)
    ks = [i for i in range(1, n // 2 + 1)
          if 2 * sum(comb(n, j) for j in range(i)) / 2 ** n <= alpha]
    if not ks:
        return float("nan"), float("nan"), 0.0
    k = max(ks)
    cov = 1 - 2 * sum(comb(n, j) for j in range(k)) / 2 ** n
    return float(x[k - 1]), float(x[n - k]), cov


def mcnemar(b_bad, t_bad):
    n01 = int(np.sum(~b_bad & t_bad)); n10 = int(np.sum(b_bad & ~t_bad)); d = n01 + n10
    if d == 0:
        return n01, n10, 1.0, 1.0
    k = min(n01, n10)
    p = min(1.0, 2 * sum(comb(d, i) for i in range(k + 1)) / 2 ** d)
    return n01, n10, p, 2 / 2 ** d          # p, and the smallest attainable p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="fine-tuned checkpoint (treated arm)")
    ap.add_argument("--baseline-root", action="append", required=True)
    ap.add_argument("--out-root", default="/home/kyle/sbel-artifacts/datasets/go2_verdict")
    ap.add_argument("--cell-lo", type=float, default=0.02)
    ap.add_argument("--cell-hi", type=float, default=0.18)
    ap.add_argument("--replay-check", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # --- select the cell -----------------------------------------------------
    eligible = []
    for root in a.baseline_root:
        for j in sorted(glob.glob(f"{root}/rigid_*/episodes/*.json")):
            spec = episode_spec(j)
            if spec is None:
                continue
            s = scored(spec["csv"])
            if s is None:
                continue
            err, ratio, cmd = s
            if not (-a.cell_hi < cmd <= -a.cell_lo):     # backward, low
                continue
            spec.update(base_err=err, base_ratio=ratio, cmd=cmd)
            eligible.append(spec)
    from collections import Counter
    print(f"eligible baseline episodes in the cell: {len(eligible)}")
    for f, n in sorted(Counter(s['fam'] for s in eligible).items()):
        print(f"   {f:<14} n={n}")
    if a.dry_run:
        return 0
    if not eligible:
        print("VERDICT: INCOMPLETE -- no episodes satisfy the predicate")
        return 1

    # --- abort condition: episodes must replay bit-for-bit -------------------
    print(f"\nreplay check on {a.replay_check} baseline episodes (bit-identical required)")
    rng = random.Random(0)
    for spec in rng.sample(eligible, min(a.replay_check, len(eligible))):
        out = f"{a.out_root}/_replaycheck/{spec['fam']}_{spec['idx']}"
        got = run_arm(spec, BASE_CKPT, out)
        ok = got is not None and sha(got) == sha(spec["csv"])
        print(f"   {spec['fam']}_{spec['idx']:<5} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            print("\nVERDICT: NOT MEASURABLE -- the arms cannot be run on identical\n"
                  "episodes, so the pairing is void. This is not a null result.")
            return 2

    # --- treated arm ---------------------------------------------------------
    print(f"\nrunning treated arm on {len(eligible)} episodes, concurrency {a.concurrency}")
    for spec in eligible:
        spec["out"] = f"{a.out_root}/treated/{spec['fam']}_{spec['idx']}"
    done = 0
    for i in range(0, len(eligible), a.concurrency):
        batch = eligible[i:i + a.concurrency]
        procs = [(s, popen_arm(s, a.ckpt, s["out"])) for s in batch]
        for s, pr in procs:
            pr.wait()
            got = glob.glob(f"{s['out']}/episodes/*.csv")
            s["treated_csv"] = got[0] if (got and pr.returncode == 0) else None
        done += len(batch)
        print(f"   {done}/{len(eligible)}", flush=True)

    # --- pair and score ------------------------------------------------------
    pairs, dropped = [], 0
    for s in eligible:
        if not s.get("treated_csv"):
            dropped += 1; continue
        t = scored(s["treated_csv"])
        if t is None:
            dropped += 1; continue
        pairs.append((s, t[0], t[1]))
    n = len(pairs)
    print(f"\nsurviving pairs {n}/{len(eligible)}   dropped {dropped} "
          f"({100*dropped/max(len(eligible),1):.0f}%)")
    for f in sorted({s['fam'] for s, _, _ in pairs}):
        print(f"   {f:<14} n={sum(1 for s,_,_ in pairs if s['fam']==f)}")
    if n < 30:
        print("\nVERDICT: INCOMPLETE -- fewer than 30 surviving pairs")
        return 1

    D = np.array([te - s["base_err"] for s, te, _ in pairs])
    B = np.array([s["base_err"] for s, _, _ in pairs])
    T = np.array([te for _, te, _ in pairs])
    bw = np.array([s["base_ratio"] < 0 for s, _, _ in pairs])
    tw = np.array([tr < 0 for _, _, tr in pairs])
    lo, hi, cov = exact_median_ci(D)
    n01, n10, pmc, pmin = mcnemar(bw, tw)
    ratio_sd = T.std(ddof=1) / B.std(ddof=1)
    med = float(np.median(D))

    print(f"\n--- PRIMARY ---")
    print(f"  median paired difference {med:+.4f} m/s")
    print(f"  exact 95% CI [{lo:+.4f}, {hi:+.4f}] (coverage {cov:.3f}), half-width {(hi-lo)/2:.4f}")
    print(f"--- ANCHOR (wrong-way) ---")
    print(f"  baseline {bw.mean():.0%}  treated {tw.mean():.0%}   discordant {n01}/{n10}")
    print(f"  McNemar p {pmc:.3f}   smallest attainable p {pmin:.4f}"
          f"   {'(VACUOUS -- cannot reject)' if pmin > 0.05 else ''}")
    print(f"--- SPREAD GUARD ---")
    print(f"  treated sd / baseline sd {ratio_sd:.2f}   (limit 1.50)")

    fail, unevaluable = [], []
    if not (med <= -0.020 and hi < 0):
        fail.append("primary: median paired difference did not reach -0.020 m/s with a CI excluding 0")
    # A VACUOUS ANCHOR IS NOT A PASSING ANCHOR. With few discordant pairs the
    # smallest attainable McNemar p exceeds 0.05, so the test cannot reject no
    # matter what the data does -- silence then means "could not look", not "did
    # not happen". Reporting it as satisfied would be the same error as the n=5
    # sign test that started this whole line of work.
    if pmin > 0.05:
        unevaluable.append(
            f"anchor: only {n01+n10} discordant pairs, smallest attainable McNemar "
            f"p = {pmin:.4f} > 0.05, so the test CANNOT reject and a regression "
            f"cannot be excluded (needs >= 6 discordant pairs)")
    elif pmc < 0.05 and n01 > n10:
        fail.append("anchor: wrong-way fraction increased significantly")
    if ratio_sd > 1.5:
        fail.append("guard: treated spread exceeds 1.5x baseline")
    print()
    for u in unevaluable:
        print(f"  NOT EVALUABLE -- {u}")
    if unevaluable and not fail:
        print("\nVERDICT: INCOMPLETE -- the primary is satisfied but the anchor could\n"
              "not be evaluated, and an unevaluable rule is never a pass.")
        return 1
    if fail:
        for f in fail:
            print(f"  FAIL -- {f}")
        print("\nVERDICT: FAIL. This means no improvement of at least 0.020 m/s was\n"
              "DEMONSTRATED -- not that no improvement occurred. A true 0.015 m/s\n"
              "improvement fails this criterion about four times in five.")
    else:
        print("VERDICT: PASS (rigid terrain only; says nothing about soil)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
