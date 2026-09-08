"""Truncate each episode at its first physically inadmissible frame.

The previous corpus dropped 500 whole episodes as inadmissible -- 499 of them
divergences. Measured afterwards: 156 of those 500 were admissible in SOME frames,
and their pre-divergence prefixes went out with the blown-up remainder. Those
prefixes are the approach to failure, which is the part a dynamics model most needs
and the part a whole-episode filter is guaranteed to remove.

This truncates instead. Rows up to the first violation are kept; the rest are
dropped. An episode left shorter than --min-rows is excluded, which is the same
decision as before but taken on the surviving prefix rather than on the episode.

BOUNDS. JOINT_LIMIT_RAD is imported from the verdict harness rather than restated,
so the corpus filter and the scoring predicate cannot drift apart. The other three
are the previous corpus's recorded bounds.

Writes a sibling <csv>.trunc.json per touched episode and never modifies a CSV, so
this is reversible and the original rows stay on disk.
"""
import argparse, csv, glob, json, math, os, sys

# Resolve the harness against the repo, not against __file__ -- this script gets
# copied to /tmp to run on other boxes, and a relative import then silently
# resolves to nothing.
import importlib.util
_CANDS = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "evaluation",
                       "run_go2_finetune_verdict.py"),
          os.path.join(os.environ.get("NEDM_REPO", ""), "scripts", "evaluation",
                       "run_go2_finetune_verdict.py"),
          "/home/kyle/Documents/sbel/NeDM/scripts/evaluation/run_go2_finetune_verdict.py",
          "/home/kyle/sbel/NeDM/scripts/evaluation/run_go2_finetune_verdict.py"]
_H = next((c for c in _CANDS if c and os.path.exists(c)), None)
if _H is None:
    raise SystemExit("FATAL: cannot locate run_go2_finetune_verdict.py to import "
                     "JOINT_LIMIT_RAD from; refusing to restate the bound")
_spec = importlib.util.spec_from_file_location("_v", _H)
_V = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_V)
JOINT_LIMIT_RAD = _V.JOINT_LIMIT_RAD          # 5.0, imported not restated

DQ_MAX, V_MAX, W_MAX = 60.2, 15.0, 50.0


def first_bad(rows, cols):
    jp, jv, vb, av = cols
    prev = None
    for i, r in enumerate(rows):
        # CONSECUTIVE-JUMP BOUND. The absolute-position and velocity-column checks
        # below both passed on episodes preprocess then rejected: a joint can sit
        # inside +-5 rad and still step 5.4 rad between two 10 ms rows, which is
        # 539 rad/s against a 60.2 bound, and the logged velocity column does not
        # reflect it. preprocess computes the difference itself and refuses.
        #
        # pi is preprocess's own threshold, used here so the corpus filter and the
        # consumer cannot disagree -- the same reason JOINT_LIMIT_RAD is imported
        # rather than restated.
        try:
            cur = [float(r[c]) for c in jp]
        except (ValueError, KeyError):
            return i, "unparseable"
        if prev is not None and any(abs(a - b) > math.pi for a, b in zip(cur, prev)):
            return i, "joint jump > pi"
        prev = cur
        try:
            if any(not math.isfinite(float(r[c])) for c in jp + jv + vb + av):
                return i, "non-finite"
            if any(abs(float(r[c])) > JOINT_LIMIT_RAD for c in jp):
                return i, "joint position"
            if any(abs(float(r[c])) > DQ_MAX for c in jv):
                return i, "joint velocity"
            if any(abs(float(r[c])) > V_MAX for c in vb):
                return i, "body velocity"
            if any(abs(float(r[c])) > W_MAX for c in av):
                return i, "angular velocity"
        except (ValueError, KeyError):
            return i, "unparseable"
    return None, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--min-rows", type=int, default=1500)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    paths = sorted(glob.glob(f"{a.root}/episodes/*.csv")) or \
            sorted(glob.glob(f"{a.root}/*/episodes/*.csv"))
    if not paths:
        print(f"FATAL: no episodes under {a.root}", file=sys.stderr); return 2

    clean = trunc_kept = trunc_dropped = 0
    reasons, cut_at = {}, []
    for p in paths:
        rows = list(csv.DictReader(open(p)))
        if not rows:
            trunc_dropped += 1; continue
        h = rows[0].keys()
        cols = ([c for c in h if c.startswith("joint_") and c.endswith("_pos_rad")],
                [c for c in h if c.startswith("joint_") and c.endswith("_vel_radps")],
                [c for c in h if c.startswith("vel_body_")],
                [c for c in h if c.startswith("ang_vel_body_")])
        i, why = first_bad(rows, cols)
        if i is None:
            clean += 1; continue
        reasons[why] = reasons.get(why, 0) + 1
        cut_at.append(i)
        if i >= a.min_rows:
            trunc_kept += 1
            if a.apply:
                json.dump({"original_rows": len(rows), "kept_rows": i, "reason": why},
                          open(p + ".trunc.json", "w"))
        else:
            trunc_dropped += 1
            if a.apply:
                json.dump({"original_rows": len(rows), "kept_rows": i, "reason": why,
                           "excluded": True}, open(p + ".trunc.json", "w"))

    n = len(paths)
    print(f"{a.root.split('/')[-1]}: {n} episodes")
    print(f"  clean throughout          {clean:>5}  ({100*clean/n:.1f}%)")
    print(f"  truncated, prefix kept    {trunc_kept:>5}  (>= {a.min_rows} good rows)")
    print(f"  excluded, prefix too short{trunc_dropped:>5}")
    print(f"  first-violation reasons: {reasons}")
    if cut_at:
        cut_at.sort()
        print(f"  cut row: min {cut_at[0]}  median {cut_at[len(cut_at)//2]}  max {cut_at[-1]}")
    print(f"  {'WROTE .trunc.json sidecars' if a.apply else 'dry run, pass --apply'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
