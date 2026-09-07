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
def chrono_provenance():
    """The pychrono actually resolved, and its md5.

    This box has TWO pychrono installs -- a source build and one inside the conda
    env -- and which one wins is decided by PYTHONPATH. The default CHRONO above is
    the OTHER machine's layout, so an unset NEDM_CHRONO_PYTHONPATH silently selects
    the conda build, which breaks the bit-exact replay the paired design rests on.

    Recording the resolved path and md5 turns "which build produced these numbers?"
    from a question that cannot be answered from the artifacts into a grep. The
    replay check would catch a mismatch, but only after the episodes are collected.
    """
    import hashlib
    from pathlib import Path
    so = Path(CHRONO) / "pychrono" / "_core.so"
    if not so.exists():
        return f"pychrono NOT FOUND at {so} -- subprocesses will fall back to whatever is importable"
    h = hashlib.md5(so.read_bytes()).hexdigest()
    return f"pychrono {so}  md5 {h}"


PERTURB_MAX_N, GROUND_M, SCORED_ROWS, LEAD_IN_S = 120.0, 200.0, 1000, 5.0
LEGACY_DERIVATION = [False]   # set by --legacy-derivation; list so it is writable
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
    # READ the collection parameters, do not re-derive them. These were reconstructed
    # from an RNG duplicated in this file, which silently disagreed with the driver
    # after the ground-pitch cap (driver +-1.5, here +-3.0) and made every post-cap
    # corpus fail the replay check as if the simulator were non-deterministic.
    #
    # ABSENCE IS AMBIGUOUS AND THEREFORE FATAL BY DEFAULT. A sidecar without these
    # fields is either a pre-cap corpus, where the derivation is right, or a post-cap
    # one, where it is wrong -- and the two are indistinguishable here. Guessing is
    # what produced the original failure, so a caller who wants the legacy path has
    # to ask for it and thereby record the decision.
    have = all(k in m for k in ("prewalk_s", "ground_tilt_roll_deg",
                                "ground_tilt_pitch_deg", "perturb_peak_n"))
    if have:
        prewalk = float(m["prewalk_s"]); roll = float(m["ground_tilt_roll_deg"])
        pitch = float(m["ground_tilt_pitch_deg"]); peak = float(m["perturb_peak_n"])
    elif LEGACY_DERIVATION[0]:
        peak = PERTURB_MAX_N * (idx % 6) / 5.0
        prewalk = tr.uniform(0.0, 3.0)
        roll = tr.uniform(-3.0, 3.0); pitch = tr.uniform(-3.0, 3.0)
    else:
        return None, ("sidecar records no collection parameters, so they can only be "
                      "re-derived -- and the derivation is correct ONLY for corpora "
                      "collected before the ground-pitch cap (e09e45b). Pass "
                      "--legacy-derivation to use it anyway, which is right for "
                      "go2_joint_off3000000 and wrong for anything newer.")
    return dict(json=json_path, csv=csv_path, machine=m.get("machine"), fam=fam, idx=idx,
                params=m["command_params"], duration=m["duration_s"], seed=m["seed"],
                spawn_x=m["spawn_m"][0], spawn_y=m["spawn_m"][1], heading=m["heading_deg"],
                peak=peak, prewalk=prewalk, roll=roll, pitch=pitch, off=off), None


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


def arm_env(spec, action_mult=None):
    """Subprocess environment. action_mult is scoped to the TREATED arm only.

    NEDM_ACTION_MULT scales the policy's output, so inheriting it from the ambient
    environment silently applies it to the baseline replay check as well -- and the
    replay then does not reproduce bit-identically, aborting the run with NOT
    MEASURABLE. That is the guard working, but the contamination is the bug. The
    variable is therefore STRIPPED from the inherited environment and re-set only
    where it is meant to apply, so a k-scaled treated arm can be compared against an
    unscaled recorded baseline without disabling the replay check.
    """
    e = dict(os.environ, PYTHONPATH=CHRONO, NEDM_SEED_OFFSET=str(spec["off"]))
    e.pop("NEDM_ACTION_MULT", None)
    if action_mult is not None and action_mult != 1.0:
        e["NEDM_ACTION_MULT"] = repr(float(action_mult))
    return e


def run_arm(spec, ckpt, outdir, action_mult=None):
    """Blocking single-episode run, used by the replay check."""
    os.makedirs(outdir, exist_ok=True)
    p = subprocess.run(arm_cmd(spec, ckpt, outdir), env=arm_env(spec, action_mult),
                       capture_output=True, text=True)
    got = glob.glob(f"{outdir}/episodes/*.csv")
    return (got[0] if got and p.returncode == 0 else None)


def popen_arm(spec, ckpt, outdir, action_mult=None):
    """Same invocation as run_arm, launched without blocking so a batch runs in parallel."""
    os.makedirs(outdir, exist_ok=True)
    return subprocess.Popen(arm_cmd(spec, ckpt, outdir), env=arm_env(spec, action_mult),
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



def arms_record(a, base_ckpt):
    """What this run compared. Recorded on EVERY exit path, including the aborts.

    A summary written only on success means the runs most in need of provenance --
    the ones that stopped early and will be re-run under changed conditions -- leave
    no record of what they tried. Audited 2026-09-07: of eight summaries on this box,
    none named either arm, and two could not be classified at all.
    """
    return {"treated_ckpt": os.path.abspath(a.ckpt),
            "treated_action_mult": 1.0 if a.action_mult is None else float(a.action_mult),
            "baseline_ckpt": os.path.abspath(base_ckpt),
            "baseline_action_mult": 1.0,
            "matched_gain": a.action_mult is None or a.action_mult == 1.0}


def write_stub(a, base_ckpt, host, status, why):
    """Summary for a run that produced no verdict, so the attempt stays auditable."""
    if not a.summary_json:
        return
    import json as _j
    _j.dump({"machine": host, "n": 0, "cell": [a.cell_lo, a.cell_hi],
             "arms": arms_record(a, base_ckpt), "argv": sys.argv,
             "verdict": status, "why": why, "pairs": []},
            open(a.summary_json, "w"), indent=2)
    print(f"  wrote {a.summary_json} (no verdict; the attempt is recorded)")


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
    ap.add_argument("--legacy-derivation", action="store_true",
                    help="re-derive prewalk and ground tilt from a seeded RNG for "
                         "corpora that do not record them. Correct ONLY before the "
                         "ground-pitch cap (e09e45b); wrong and silent after it.")
    # DEFAULT None, NOT 1.0, so "not passed" is distinguishable from "passed as
    # nominal". arm_env already treats None as nominal. With default=1.0 the
    # ambient-variable guard below could never fire -- it tested `is None` on a
    # value that was never None, which is the second guard-that-cannot-fail of
    # the same day.
    ap.add_argument("--action-mult", type=float, default=None,
                    help="scale the TREATED policy's output by this factor. Applied only "
                         "to the treated arm; the baseline replay check runs unscaled, so "
                         "it stays a valid check. See imported_policy.NEDM_ACTION_MULT -- "
                         "a gain-margin diagnostic, not a fix.")
    ap.add_argument("--own-machine-only", action="store_true",
                    help="drop episodes collected on another machine and score only "
                         "this host's, which is the stratified design the abort message "
                         "recommends; prints exactly what was dropped. Ported from "
                         "dorm-pc's patch rather than adopting its file, which was "
                         "based on 04eac14 and predates the survivorship diagnostic and "
                         "the stratum_incomplete guard. Its schema exclusion exists "
                         "under a different name, NEW_PHYSICS -- absence of an "
                         "identifier is not absence of a capability, and I nearly "
                         "reported it missing from reading the source instead of "
                         "calling it.")
    ap.add_argument("--allow-foreign", action="store_true",
                    help="proceed despite episodes from another machine; only valid if "
                         "cross-build replay has been verified")
    ap.add_argument("--summary-json", default=None,
                    help="write machine-tagged per-episode paired differences for "
                         "stratified combination across boxes")
    a = ap.parse_args()
    host = socket.gethostname()

    # SETTING NEDM_ACTION_MULT IN THE ENVIRONMENT DOES NOTHING HERE, AND SAYING SO
    # IS THE POINT. arm_env pops it deliberately, so that a stray value in the
    # operator's shell cannot silently scale one arm. The cost is that an operator
    # who sets it the wrong way gets a clean run at nominal gain with no complaint:
    # a gain control launched that way returned exactly 0.000000 on 457 episodes,
    # which is what a comparison of a policy against itself returns.
    #
    # The harness cannot tell "the operator wants nominal" from "the operator asked
    # the wrong way", so it refuses instead of guessing.
    if "NEDM_ACTION_MULT" in os.environ and a.action_mult is None:
        raise SystemExit(
            "NEDM_ACTION_MULT is set in the environment but --action-mult was not "
            "passed. This harness strips the ambient variable by design, so the run "
            "would proceed at NOMINAL gain and silently ignore what you asked for. "
            "Pass --action-mult explicitly, or unset the variable to run at nominal.")
    LEGACY_DERIVATION[0] = a.legacy_derivation

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
            write_stub(a, BASE_CKPT, host, "NOT MEASURABLE", "baseline root contributed no usable episodes")
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
    # (hoisted above; kept for readability of the original flow)
    by_machine = Counter(sp.get("machine") for sp in eligible)
    print("  eligible by machine: " + ", ".join(f"{k}={v}" for k, v in by_machine.items()))
    foreign = {k: v for k, v in by_machine.items() if k and k != host}
    if foreign and a.own_machine_only:
        before = len(eligible)
        eligible = [sp for sp in eligible if sp.get("machine") == host]
        print(f"  --own-machine-only: dropped {before - len(eligible)} episode(s) from "
              f"{', '.join(foreign)}; scoring {len(eligible)} from {host}.")
        print("  THIS IS ONE STRATUM, NOT THE POOL. The dropped episodes are not missing\n"
              "  data; they are the other box's stratum and must be scored there and\n"
              "  combined as paired differences. Do not report this n as the comparison.")
        if not eligible:
            print("\nVERDICT: NOT MEASURABLE -- no episodes from this host.")
            write_stub(a, BASE_CKPT, host, "NOT MEASURABLE", "no eligible episodes from this host")
            return 2
        foreign = {}
    if foreign:
        print(f"\nVERDICT: NOT MEASURABLE -- {sum(foreign.values())} eligible episodes were\n"
              f"collected on {', '.join(foreign)} but this host is {host}. Episodes do not\n"
              f"replay bit-identically across machines, so the treated arm would differ\n"
              f"from its baseline by BUILD as well as by checkpoint. Run this harness on\n"
              f"each collecting machine over its own episodes and combine the paired\n"
              f"differences afterwards; --allow-foreign overrides only if you have\n"
              f"verified replay across the two builds.")
        if not a.allow_foreign:
            write_stub(a, BASE_CKPT, host, "NOT MEASURABLE", "eligible episodes come from a foreign machine and --allow-foreign was not given")
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
    print(f"  {chrono_provenance()}")
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
            write_stub(a, BASE_CKPT, host, "NOT MEASURABLE", "replay check mismatched; the arms cannot run on identical episodes")
            return 2

    # --- treated arm ---------------------------------------------------------
    print(f"\nrunning treated arm on {len(eligible)} episodes, concurrency {a.concurrency}")
    for spec in eligible:
        spec["out"] = f"{a.out_root}/treated/{spec['fam']}_{spec['idx']}"
    done = 0
    for i in range(0, len(eligible), a.concurrency):
        batch = eligible[i:i + a.concurrency]
        procs = [(s, popen_arm(s, a.ckpt, s["out"], a.action_mult)) for s in batch]
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
    # THE 30-PAIR MINIMUM IS A CONDITION ON THE POOLED COUNT, NOT ON A STRATUM.
    # This is a stratified paired design: each machine scores its own episodes and
    # the differences are combined afterwards, so a stratum below 30 is normal and
    # must still EMIT its per-episode differences or the pool cannot be formed.
    # Returning early here made a stratum that was individually short unpoolable,
    # which would have forced a re-run to recover data already computed.
    #
    # Separating the two concerns: the statistics and the summary are always
    # produced; only the VERDICT is withheld below the minimum.
    stratum_incomplete = n < 30

    D = np.array([te - s["base_err"] for s, te, _ in pairs])
    B = np.array([s["base_err"] for s, _, _ in pairs])
    T = np.array([te for _, te, _ in pairs])
    bw = np.array([s["base_ratio"] < 0 for s, _, _ in pairs])
    tw = np.array([tr < 0 for _, _, tr in pairs])
    lo, hi, cov = exact_median_ci(D)
    # AN ALL-DROPPED STRATUM IS A RESULT, NOT A CRASH. Arm A dropped 36 of 36 because
    # every treated episode diverged numerically, and mcnemar() then raised on empty
    # boolean arrays -- losing the diagnostic output after the run had already cost
    # the episodes. Report and stop cleanly instead.
    if len(bw) == 0:
        print("\nVERDICT: NOT MEASURABLE -- 0 surviving pairs. Every treated episode\n"
              "failed the scoring predicate. Check the treated episodes' joint-target\n"
              "magnitudes before concluding anything about control quality: unbounded\n"
              "output and falling are different failures and only one is about walking.")
        write_stub(a, BASE_CKPT, host, "NOT MEASURABLE", "0 surviving pairs; every treated episode failed the scoring predicate")
        return 2
    n01, n10, pmc, pmin = mcnemar(bw, tw)
    ratio_sd = T.std(ddof=1) / B.std(ddof=1)
    med = float(np.median(D))

    if a.summary_json:
        # Machine-tagged, per-episode, so the strata can be combined later AND
        # reported separately. A machine-by-treatment interaction must stay visible;
        # pooling numbers that hide structure is the failure this study kept hitting.
        # THE ARMS, RECORDED. Without these a summary cannot say what it compared:
        # the treated arm runs at action_mult and the baseline at nominal, so a file
        # holding only a median is a number whose comparison is unrecoverable. Audited
        # 2026-09-07: of eight summaries on this box, five encoded the gain in the
        # FILENAME by convention and two (anchor_sbel, v4_sbel) had neither the gain nor
        # an output directory, so which comparison produced them cannot be determined
        # from the artifacts at all.
        json.dump({"machine": host, "n": n, "cell": [a.cell_lo, a.cell_hi],
                   "arms": {
                       "treated_ckpt": os.path.abspath(a.ckpt),
                       "treated_action_mult": (1.0 if a.action_mult is None
                                               else float(a.action_mult)),
                       "baseline_ckpt": os.path.abspath(BASE_CKPT),
                       "baseline_action_mult": 1.0,
                       "matched_gain": (a.action_mult is None or a.action_mult == 1.0),
                   },
                   "argv": sys.argv,
                   "median_paired_difference": med,
                   "exact_ci": [lo, hi], "coverage": cov,
                   "wrong_way_baseline": float(bw.mean()), "wrong_way_treated": float(tw.mean()),
                   "mcnemar_p": pmc, "mcnemar_min_p": pmin, "sd_ratio": ratio_sd,
                   "survivorship": {
                       "n_survivors": len(pairs),
                       "n_dropped": len(eligible) - len(pairs),
                       "baseline_median_err_survivors": float(np.median(
                           [sp["base_err"] for sp in eligible
                            if (sp["fam"], sp["idx"]) in {(q["fam"], q["idx"])
                                                          for q, _, _ in pairs}])),
                       "baseline_median_err_dropped": (float(np.median(
                           [sp["base_err"] for sp in eligible
                            if (sp["fam"], sp["idx"]) not in {(q["fam"], q["idx"])
                                                              for q, _, _ in pairs}]))
                           if len(eligible) > len(pairs) else None)},
                   "pairs": [{"family": sp["fam"], "idx": sp["idx"], "cmd": sp["cmd"],
                              "baseline_err": sp["base_err"], "treated_err": te,
                              "difference": te - sp["base_err"]}
                             for sp, te, _ in pairs]},
                  open(a.summary_json, "w"), indent=1)
        print(f"\nwrote {a.summary_json} ({n} pairs, machine {host})")
    # SURVIVORSHIP DIAGNOSTIC. Surviving pairs are selected by the TREATMENT
    # completing, so the paired difference describes a subpopulation the treatment
    # defines. It is unbiased WITHIN that subpopulation and says nothing about
    # whether the subpopulation is representative. The failure it permits is
    # specific: a treatment that survives preferentially on episodes it happens to
    # track well shows a good per-survivor difference while the overall effect is
    # bad -- which is the DEFAULT expectation when a treatment removes its own
    # worst cases.
    #
    # The baseline ran every episode, so its error exists on both groups and the
    # comparison is free. Similar => selection is not on episode difficulty.
    # Lower on survivors => the treatment survives on easy episodes, and the gap
    # bounds how optimistic the paired figure is.
    surv_ids = {(sp["fam"], sp["idx"]) for sp, _, _ in pairs}
    b_surv = np.array([sp["base_err"] for sp in eligible
                       if (sp["fam"], sp["idx"]) in surv_ids])
    b_drop = np.array([sp["base_err"] for sp in eligible
                       if (sp["fam"], sp["idx"]) not in surv_ids])
    print("\n--- SURVIVORSHIP (baseline error, which exists for every episode) ---")
    print(f"  survivors     n={len(b_surv):<4} median baseline |err| {np.median(b_surv):+.4f} m/s")
    if len(b_drop):
        gap = float(np.median(b_drop) - np.median(b_surv))
        print(f"  dropped       n={len(b_drop):<4} median baseline |err| {np.median(b_drop):+.4f} m/s")
        print(f"  gap (dropped - survivors) {gap:+.4f} m/s")
        if len(b_drop) < 5:
            print("  NOT INTERPRETABLE: fewer than 5 dropped episodes, so this "
                  "comparison cannot distinguish selection from noise")
        elif gap > 0.005:
            print("  -> dropped episodes were HARDER for the baseline: the treatment "
                  "survived on easier ones, so the paired figure is optimistic by "
                  "roughly this gap")
        elif gap < -0.005:
            print("  -> dropped episodes were EASIER for the baseline: selection runs "
                  "against the treatment, so the paired figure is conservative")
        else:
            print("  -> no material difference: selection is not on episode difficulty "
                  "and the survivorship concern is weak")
    else:
        print("  dropped       n=0    no selection occurred")

    print(f"\n--- PRIMARY ---")
    print(f"  median paired difference {med:+.4f} m/s")
    print(f"  exact 95% CI [{lo:+.4f}, {hi:+.4f}] (coverage {cov:.3f}), half-width {(hi-lo)/2:.4f}")
    print(f"--- ANCHOR (wrong-way) ---")
    print(f"  baseline {bw.mean():.0%}  treated {tw.mean():.0%}   discordant {n01}/{n10}")
    print(f"  McNemar p {pmc:.3f}   smallest attainable p {pmin:.4f}"
          f"   {'(VACUOUS -- cannot reject)' if pmin > 0.05 else ''}")
    print(f"--- SPREAD GUARD ---")
    print(f"  treated sd / baseline sd {ratio_sd:.2f}   (limit 1.50)")

    if stratum_incomplete:
        print(f"\n  NOTE: {n} surviving pairs is below the 30-pair minimum. The figures\n"
              f"  above are reported so this stratum can be POOLED; no verdict is\n"
              f"  declared on it alone.")
        print("\nVERDICT: INCOMPLETE for this stratum -- pool with the other machine "
              "before scoring.")
        return 1

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
