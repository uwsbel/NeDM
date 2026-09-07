#!/usr/bin/env python3
"""Separate the action multiplier's effect from the fine-tune's.

The verdict harness runs the TREATED arm at --action-mult k and the BASELINE at
nominal gain (`arm_env` pops NEDM_ACTION_MULT for every other arm). Every paired
difference it reports is therefore fine-tune AND gain, not fine-tune alone.

Running the BASE checkpoint as the treated arm at the same k gives the third leg:

    as-reported   armA@k - base@1.0     what the harness printed
    multiplier    base@k - base@1.0     what the gain does by itself
    matched gain  armA@k - base@k       what the fine-tune does

Emits each leg as a summary JSON in the harness's own `pairs` format, so the
registered endpoints and the mechanism test can be recomputed on matched-gain
differences by the SAME scripts rather than a second implementation of them.
This matters for E3: it asks which VARIABLE explains an observed split, and it
would favour "family" whether the family structure comes from the multiplier or
from the fine-tune, so recomputing it on the matched-gain leg is the only way to
tell those apart.
"""
import argparse, csv, glob, json, os, re
import numpy as np
from math import comb

SCORED_ROWS = 1000


def scored_err(csv_path):
    try:
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
    except Exception:
        return None
    if len(rows) < SCORED_ROWS:
        return None
    w = rows[-SCORED_ROWS:]
    c = np.array([float(r["cmd_vx_mps"]) for r in w])
    if c.std() > 1e-6:
        return None
    v = np.array([float(r["vel_body_x_mps"]) for r in w])
    if not np.isfinite(v).all():
        return None
    return abs(float(v.mean()) - float(c[0])), float(c[0])


def from_arm(root):
    """Arm outputs are one directory per episode, named family_index."""
    out = {}
    for f in glob.glob(os.path.join(root, "**", "episodes", "*.csv"), recursive=True):
        key = os.path.basename(os.path.dirname(os.path.dirname(f)))
        e = scored_err(f)
        if e:
            out[key] = e
    return out


def from_corpus(root):
    """A collected corpus keyed the SAME WAY -- through the sidecar, because its
    directory names are collection-order (epNN) and will not intersect otherwise."""
    out = {}
    for jf in glob.glob(os.path.join(root, "**", "episodes", "*.json"), recursive=True):
        try:
            m = json.load(open(jf))
        except Exception:
            continue
        hit = re.search(r"(\d+)$", str(m.get("episode_id", "")))
        if not (m.get("command_family") and hit):
            continue
        e = scored_err(os.path.splitext(jf)[0] + ".csv")
        if e:
            out[f"{m['command_family']}_{int(hit.group(1))}"] = e
    return out


def exact_ci(d, alpha=0.05):
    x = np.sort(np.asarray(d, float)); n = len(x)
    ks = [i for i in range(1, n // 2 + 1)
          if 2 * sum(comb(n, j) for j in range(i)) / 2 ** n <= alpha]
    if not ks:
        return None
    k = max(ks)
    return float(x[k - 1]), float(x[n - k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="baseline corpus, run at nominal gain")
    ap.add_argument("--treated-root", required=True, help="arm outputs, fine-tune at k")
    ap.add_argument("--gaincontrol-root", required=True, help="arm outputs, BASE at k")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--straight", default="constant,vel_step")
    a = ap.parse_args()

    STRAIGHT = tuple(a.straight.split(","))
    b1, tk, bk = from_corpus(a.corpus), from_arm(a.treated_root), from_arm(a.gaincontrol_root)
    sh = sorted(set(b1) & set(tk) & set(bk))
    print(f"base@1.0 {len(b1)}, treated@k {len(tk)}, base@k {len(bk)}  ->  shared {len(sh)}")
    if len(sh) < 20:
        raise SystemExit("too few shared episodes to decompose")

    # HEADROOM. A null means little in a cell where the robot barely executes the
    # command: the baseline error is then mostly the unrealised command, which is
    # common to both arms and which no fine-tune acts on. Reporting the deadband
    # fraction turns "null in this cell" into a quantity a reader can weigh, and
    # makes a null across cells a claim about the range of headroom it held across.
    cmds = np.array([b1[k][1] for k in sh])
    errs = np.array([b1[k][0] for k in sh])
    real = np.array([abs(abs(b1[k][1]) - b1[k][0]) for k in sh])   # |realised| approx
    r = float(np.median(real) / np.median(np.abs(cmds)))
    deadband = (1.0 - r) * float(np.median(np.abs(cmds)))
    print(f"\n  HEADROOM: median |cmd| {np.median(np.abs(cmds)):.4f}, "
          f"realised ratio {r:.2f}")
    print(f"    baseline error {np.median(errs):.4f} m/s, of which deadband "
          f"~{deadband:.4f} ({100 * deadband / np.median(errs):.0f}%)")
    print(f"    headroom a tracking effect could act on: "
          f"~{np.median(errs) - deadband:+.4f} m/s")

    legs = (("as_reported", lambda k: tk[k][0] - b1[k][0]),
            ("multiplier",  lambda k: bk[k][0] - b1[k][0]),
            ("matched_gain", lambda k: tk[k][0] - bk[k][0]))
    for name, f in legs:
        print(f"\n--- {name} ---")
        pairs = []
        for k in sh:
            fam, idx = k.rsplit("_", 1)
            pairs.append(dict(family=fam, idx=int(idx), cmd=b1[k][1],
                              baseline_err=float("nan"), treated_err=float("nan"),
                              difference=float(f(k))))
        gs = np.array([p["difference"] for p in pairs if p["family"] in STRAIGHT])
        gt = np.array([p["difference"] for p in pairs if p["family"] not in STRAIGHT])
        al = np.array([p["difference"] for p in pairs])
        for nm, g in (("straight", gs), ("turning", gt), ("ALL", al)):
            c = exact_ci(g) if len(g) else None
            print(f"  {nm:<9} n={len(g):<4} median {np.median(g):+.5f}" +
                  (f"  95% CI [{c[0]:+.5f}, {c[1]:+.5f}]" if c else ""))
        print(f"  split (turning - straight) {np.median(gt) - np.median(gs):+.5f}")
        out = f"{a.out_prefix}_{name}.json"
        json.dump(dict(machine="kyle-sbel", n=len(pairs), cell=None,
                       median_paired_difference=float(np.median(al)),
                       exact_ci=list(exact_ci(al) or (None, None)),
                       leg=name, pairs=pairs), open(out, "w"), indent=2)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
