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
import argparse, csv, glob, hashlib, json, os, random, re, socket, subprocess, sys
from math import comb
import numpy as np

sys.path.insert(0, "src")
from nedm.quadruped.imported_policy import family_seed

# FROM THE ENVIRONMENT, with this box's values as defaults, so neither machine has
# to edit the file. NEDM_GO2_ASSETS is not read here: the collector already takes it
# from the environment, and arm_env inherits os.environ, so it propagates untouched.
# Without it the collector dies at URDF load AFTER the process starts.
BASE_CKPT = os.environ.get("NEDM_GO2_CKPT",
                           "/home/kyle/sbel-artifacts/checkpoints/go2_cts_150k.pt")
PY = os.environ.get("NEDM_PY", "/home/kyle/miniconda3/envs/nedm-src/bin/python")
CHRONO = os.environ.get("NEDM_CHRONO_PYTHONPATH",
                        "/home/kyle/Documents/sbel/chrono-build/bin")
PERTURB_MAX_N, GROUND_M, SCORED_ROWS, LEAD_IN_S = 120.0, 200.0, 1000, 5.0
# PHYSICAL ADMISSIBILITY. Surviving collection is not the same as being physically
# real: an episode can blow up to absurd-but-FINITE values and pass every finiteness
# check. Measured over 1,481 episodes the population is cleanly bimodal -- p99 of
# max|joint angle| is 3.38 rad and p99 of max|joint target| is 3.64, then it jumps
# straight to 136 rad and 4.6e34. Every bound between 4 and 10 rad excludes exactly
# the same 10 episodes, so this threshold is not a judgement call. A Go2 joint moves
# within about +-3 rad.
JOINT_LIMIT_RAD = 5.0


def episode_spec(json_path):
    """Full replayable spec from the EPISODE SIDECAR, never from the directory path.

    An earlier version parsed (family, index) out of the directory name, which works
    only for the one-directory-per-episode layout this box collected. dorm-pc
    consolidates into a flat episodes/ tree, where that glob matches nothing and the
    dirname does not start with "rigid_" -- so all 3,003 of its episodes were skipped
    and the harness reported a SMALLER eligible count instead of an error. Running on
    the merged root would have returned this half's 37 and looked like the merged
    answer.

    command_family and the numeric suffix of episode_id recover the same pair and
    travel with the episode through consolidation. Verified against the directory
    name on all 1,762 episodes of this half: 1,762 agreements, 0 mismatches.
    """
    m = json.load(open(json_path))
    fam = m.get("command_family")
    hit = re.search(r"(\d+)$", str(m.get("episode_id", "")))
    if not fam or not hit:
        return None, "no command_family or episode_id suffix"
    idx = int(hit.group(1))
    # Cross-check against the directory when it is in the per-episode layout. Never
    # a fallback -- a disagreement means the reconstruction below would use the wrong
    # (family, index) and silently produce a mismatched realisation.
    d = os.path.basename(os.path.dirname(os.path.dirname(json_path)))
    if d.startswith("rigid_") and "_" in d[6:]:
        dfam, didx = d[6:].rsplit("_", 1)
        if didx.isdigit() and (dfam, int(didx)) != (fam, idx):
            return None, f"metadata ({fam},{idx}) disagrees with directory ({dfam},{didx})"
    off = int(m.get("seed_offset", 0))
    tr = random.Random(family_seed(fam, off) + 977 * idx)
    for key in ("spawn_m", "heading_deg", "duration_s", "seed", "command_params"):
        if key not in m:
            return None, f"missing {key}"
    # The sibling .csv is the per-episode layout's convention; a consolidated tree
    # may put it elsewhere, and the sidecar records where. Prefer what exists over
    # what is conventional -- a missing CSV would otherwise be indistinguishable
    # from an episode that failed the predicate.
    sib = json_path.replace(".json", ".csv")
    if os.path.exists(sib):
        csv_path = sib
    else:
        rel = m.get("csv_path")
        cand = [c for c in (rel, os.path.join(os.path.dirname(json_path),
                                              os.path.basename(str(rel)))) if rel]
        csv_path = next((c for c in cand if c and os.path.exists(c)), None)
        if csv_path is None:
            return None, "csv not found beside the sidecar or at csv_path"
    return dict(json=json_path, csv=csv_path, machine=m.get("machine"), fam=fam, idx=idx,
                params=m["command_params"], duration=m["duration_s"], seed=m["seed"],
                spawn_x=m["spawn_m"][0], spawn_y=m["spawn_m"][1], heading=m["heading_deg"],
                peak=PERTURB_MAX_N * (idx % 6) / 5.0, prewalk=tr.uniform(0.0, 3.0),
                roll=tr.uniform(-3.0, 3.0), pitch=tr.uniform(-3.0, 3.0), off=off), None


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
    # Admissibility, checked over the RETAINED episode rather than the scored window:
    # a run that went physically absurd earlier is not rehabilitated by ending calmly.
    for suffix, absurd in (("_target_rad", True), ("_rad", False)):
        cols = [c for c in rows[0]
                if c.startswith("joint_") and c.endswith(suffix)
                and (absurd or "target" not in c)]
        if not cols:
            continue
        M = np.abs(np.array([[float(r[c]) for c in cols] for r in rows]))
        M = M[np.isfinite(M)]
        if M.size and M.max() > JOINT_LIMIT_RAD:
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


# NOT PHYSICS. Four identity columns rewritten by consolidation, and three gravity
# channels derived post-hoc from the quaternion after collection. Excluded BY NAME.
NON_PHYSICS = ("episode_id", "scenario_name", "scenario_family", "split",
               "grav_body_x", "grav_body_y", "grav_body_z")
N_PHYSICS_COLS = 164
# SCHEMA ADDITION, 2026-09-05. The applied gravity vector is now logged, so episodes
# collected after that date carry 3 physics columns the earlier ones lack. The
# recorded VALUES of every shared column are unchanged -- logging gravity does not
# alter dynamics -- so an old baseline still replays correctly, but its column SET no
# longer matches.
#
# Handled by naming the difference rather than intersecting the sets. An intersection
# would silently absorb a genuinely missing physics column, which is the failure this
# file already carries a guard against; asserting that the set difference is EXACTLY
# this list keeps a real omission fatal.
SCHEMA_ADDITIONS = ("grav_world_x_mps2", "grav_world_y_mps2", "grav_world_z_mps2")


def physics_digest(path, extra_exclude=()):
    """(sorted physics column names, digest over just those columns).

    A raw sha256 of the file compares bytes, so an episode from a consolidated half
    -- three added gravity channels, three rewritten identity columns -- mismatches
    even when all 164 physics columns agree on every row. That would abort a
    PERFECTLY VALID pairing, the opposite of what the check exists for.

    DO NOT compute this over the intersection of the two files' columns. A genuinely
    missing physics column would drop out of the intersection and the digests would
    then agree -- a check whose success path no longer requires the thing it checks,
    which is the failure this file already carries three fixes for. The excluded
    names are listed above and the count is asserted by the caller.
    """
    with open(path, newline="") as h:
        r = csv.reader(h)
        head = next(r)
        drop = set(NON_PHYSICS) | set(extra_exclude)
        keep = [i for i, c in enumerate(head) if c not in drop]
        names = sorted(head[i] for i in keep)
        order = sorted(keep, key=lambda i: head[i])
        d = hashlib.sha256()
        for row in r:
            d.update(("\x1f".join(row[i] for i in order) + "\x1e").encode())
    return names, d.hexdigest()


def compare_replay(original, replay):
    """(ok, reason). Physics-only equality, with the column set checked first."""
    na, da = physics_digest(original)
    nb, db = physics_digest(replay)
    diff = set(na) ^ set(nb)
    if diff and diff <= set(SCHEMA_ADDITIONS):
        # One side predates the gravity columns. Drop them from BOTH and re-digest,
        # so the comparison is over the columns the two schemas share.
        na, da = physics_digest(original, extra_exclude=SCHEMA_ADDITIONS)
        nb, db = physics_digest(replay, extra_exclude=SCHEMA_ADDITIONS)
        expect = N_PHYSICS_COLS
    else:
        expect = None
    if expect is None:
        for n in (na, nb):
            if len(n) not in (N_PHYSICS_COLS, N_PHYSICS_COLS + len(SCHEMA_ADDITIONS)):
                return False, (f"physics column count {len(na)} vs {len(nb)}; expected "
                               f"{N_PHYSICS_COLS} or {N_PHYSICS_COLS + len(SCHEMA_ADDITIONS)}"
                               f" -- columns were lost, not renamed")
    elif len(na) != expect or len(nb) != expect:
        return False, (f"after dropping the schema additions, {len(na)} vs {len(nb)} "
                       f"physics columns, expected {expect}")
    if na != nb:
        miss = sorted(set(na) ^ set(nb))[:4]
        return False, f"physics columns differ: {miss}"
    note = "physics identical" if da == db else "physics digest differs"
    if diff:
        note += " (schema additions excluded from both)"
    return (da == db), note


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
    ap.add_argument("--allow-foreign", action="store_true",
                    help="proceed despite episodes from another machine; only valid if "
                         "cross-build replay has been verified")
    ap.add_argument("--summary-json", default=None,
                    help="write machine-tagged per-episode paired differences for "
                         "stratified combination across boxes")
    a = ap.parse_args()

    # --- select the cell -----------------------------------------------------
    eligible = []
    from collections import Counter
    rejected = Counter()
    for root in a.baseline_root:
        # RECURSIVE, so both layouts match: one-directory-per-episode and the
        # consolidated flat episodes/ tree.
        found = sorted(glob.glob(f"{root}/**/episodes/*.json", recursive=True))
        seen = 0
        for j in found:
            spec, why = episode_spec(j)
            if spec is None:
                rejected[why] += 1
                continue
            seen += 1
            sc = scored(spec["csv"])
            if sc is None:
                rejected["failed predicate or admissibility"] += 1
                continue
            err, ratio, cmd = sc
            if not (-a.cell_hi < cmd <= -a.cell_lo):
                rejected["outside the cell"] += 1
                continue
            spec.update(base_err=err, base_ratio=ratio, cmd=cmd)
            eligible.append(spec)
        # A ROOT THAT CONTRIBUTES NOTHING IS AN ERROR, NOT A SMALLER NUMBER. That is
        # exactly how the layout bug stayed invisible.
        print(f"  {root}: {len(found)} episode files, {seen} parsed")
        if not found or not seen:
            print(f"\nVERDICT: NOT MEASURABLE -- baseline root contributed no usable\n"
                  f"episodes ({len(found)} files found, {seen} parsed). Refusing to\n"
                  f"report a count that silently omits a whole half.")
            return 2
    print("  rejected: " + (", ".join(f"{v} {k}" for k, v in rejected.most_common()) or "none"))

    # EPISODES ARE ONLY BIT-REPRODUCIBLE ON THE MACHINE THAT PRODUCED THEM. Different
    # Chrono builds give different arithmetic in the last digits and a chaotic plant
    # amplifies it: a replay of an s3000000 episode on the other box differs in 147
    # columns from row 0, physics included. The baseline arm is unaffected -- it is
    # the recorded file, not a replay -- but the TREATED arm must run where its
    # baseline was collected, or the two arms differ by machine as well as by
    # checkpoint in a design whose whole claim is that only the checkpoint differs.
    #
    # So the verdict is a STRATIFIED paired design: each box scores its own episodes
    # and the paired differences are combined afterwards. Machine cancels within each
    # pair. This aborts rather than skipping, because silently dropping the foreign
    # half is how a partial answer would come back looking like the whole one.
    host = socket.gethostname()
    by_machine = Counter(sp.get("machine") for sp in eligible)
    print("  eligible by machine: " + ", ".join(f"{k}={v}" for k, v in by_machine.items()))
    foreign = {k: v for k, v in by_machine.items() if k and k != host}
    if foreign:
        print(f"\nVERDICT: NOT MEASURABLE -- {sum(foreign.values())} eligible episodes were\n"
              f"collected on {', '.join(foreign)} but this host is {host}. Episodes do not\n"
              f"replay bit-identically across machines, so the treated arm would differ\n"
              f"from its baseline by BUILD as well as by checkpoint. Run this harness on\n"
              f"each collecting machine over its own episodes and combine the paired\n"
              f"differences afterwards; --allow-foreign overrides only if you have\n"
              f"verified replay across the two builds.")
        if not a.allow_foreign:
            return 2
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
        if got is None:
            ok, why = False, "replay produced no CSV"
        else:
            ok, why = compare_replay(spec["csv"], got)
        print(f"   {spec['fam']}_{spec['idx']:<5} {'OK' if ok else 'MISMATCH'}  ({why})")
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

    if a.summary_json:
        # Machine-tagged, per-episode, so the strata can be combined later AND
        # reported separately. A machine-by-treatment interaction must stay visible;
        # pooling numbers that hide structure is the failure this study kept hitting.
        json.dump({"machine": host, "n": n, "cell": [a.cell_lo, a.cell_hi],
                   "median_paired_difference": med,
                   "exact_ci": [lo, hi], "coverage": cov,
                   "wrong_way_baseline": float(bw.mean()), "wrong_way_treated": float(tw.mean()),
                   "mcnemar_p": pmc, "mcnemar_min_p": pmin, "sd_ratio": ratio_sd,
                   "pairs": [{"family": sp["fam"], "idx": sp["idx"], "cmd": sp["cmd"],
                              "baseline_err": sp["base_err"], "treated_err": te,
                              "difference": te - sp["base_err"]}
                             for sp, te, _ in pairs]},
                  open(a.summary_json, "w"), indent=1)
        print(f"\nwrote {a.summary_json} ({n} pairs, machine {host})")
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
        # Computed from THIS run's half-width, not hardcoded. An earlier version
        # printed "about four times in five", which was the figure at n=33; the
        # same sentence at n=107 understates the criterion by an order of magnitude.
        # A number embedded in an explanation goes stale silently.
        from math import erf, sqrt as _sqrt
        se = max((hi - lo) / 2 / 1.96, 1e-12)
        p15 = 1 - 0.5 * (1 + erf(((0.015 - 0.020) / se) / _sqrt(2)))
        print("\nVERDICT: FAIL. This means no improvement of at least 0.020 m/s was\n"
              "DEMONSTRATED -- not that no improvement occurred. At this run's n=%d\n"
              "a true 0.015 m/s improvement fails this criterion about %.0f%% of the\n"
              "time, so a real but smaller effect is not excluded." % (n, 100 * p15))
    else:
        print("VERDICT: PASS (rigid terrain only; says nothing about soil)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
