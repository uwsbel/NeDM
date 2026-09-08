"""Apply the action-sensitivity rule across SURROGATE SEEDS, not to one surrogate.

`gate_go2_action_sensitivity.py` measures one surrogate and puts a bootstrap CI
around it. That CI is over EPISODES, with the surrogate held fixed -- so it answers
"how well is this particular model's action response measured?" and not "what is
this arm's action response?". Those diverge badly here: on the gravity arms the
gate reversed between seed 1 and seed 2 (B_s1 looked good, B_s2 scored 4.374 --
as bad as baseline -- while C_s2, the permuted control carrying no tilt information
at all, scored 1.600), and the closed-loop Chrono evaluation reversed the same way.

So aggregate over seeds and let the between-seed spread be the uncertainty. Two
things are printed side by side on purpose:

  ci_halfwidth   the median within-run bootstrap CI half-width  (episodes vary)
  seed_sd        the standard deviation across surrogate seeds  (surrogates vary)

**When seed_sd exceeds ci_halfwidth, the single-surrogate CI is understating the
uncertainty, and any verdict read off one report is over-confident by that ratio.**
That ratio is the whole point of this tool.

The thresholds are imported from _gate_rule.py rather than restated, so the two
tools cannot drift apart.
"""
import argparse, importlib.util, json, math, os, statistics as st, sys

_rs = importlib.util.spec_from_file_location(
    "_gate_rule", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gate_rule.py"))
_R = importlib.util.module_from_spec(_rs); _rs.loader.exec_module(_R)

ap = argparse.ArgumentParser()
ap.add_argument("--arm", action="append", required=True,
                help="NAME=report1.json,report2.json,...  one gate report per surrogate seed")
ap.add_argument("--min-seeds", type=int, default=4,
                help="the seed-level permutation floor only reaches 0.05 at 4 per arm")
a = ap.parse_args()

print(f"rule (imported, not restated): family={_R.PRIMARY}  horizons={_R.VH}  "
      f"gain in [{_R.GLO},{_R.GHI}]  corr/cosine >= {_R.CMIN}  n >= {_R.NMIN}\n")

arms = {}
for spec in a.arm:
    name, files = spec.split("=", 1)
    reps = []
    for f in files.split(","):
        r = json.load(open(f))
        reps.append(r)
    # A gate report is only comparable to another if the perturbation matches.
    sig = {(r.get("rel_sigma"), tuple(r.get("horizons_s", []))) for r in reps}
    if len(sig) != 1:
        raise SystemExit(f"FATAL: arm {name} mixes perturbation settings {sig}; "
                         f"these reports do not measure the same thing")
    seeds = [r.get("perturb_seed") for r in reps]
    if len(set(seeds)) != len(seeds):
        print(f"  NOTE: arm {name} repeats a --perturb-seed {seeds}; those are not "
              f"independent draws of the perturbation", file=sys.stderr)
    arms[name] = reps

for name, reps in arms.items():
    print(f"=== arm {name}   {len(reps)} surrogate seeds ===")
    if len(reps) < a.min_seeds:
        print(f"  *** {len(reps)} seeds. The seed-level permutation test cannot reach "
              f"p<0.05 below {a.min_seeds}; read the spread, not a verdict. ***")
    for h in _R.VH:
        rows = []
        for r in reps:
            fam = r.get("families", {}).get(_R.PRIMARY, {}).get(h)
            if fam and fam.get("n", 0) >= _R.NMIN:
                rows.append(fam)
        if not rows:
            print(f"  {h}s: no seed has >= {_R.NMIN} usable pairs -- INCOMPLETE")
            continue
        print(f"  {h}s  ({len(rows)} of {len(reps)} seeds evaluable, "
              f"n per seed {[x['n'] for x in rows]})")
        for label, lo_ok, hi_ok in (("gain", _R.GLO, _R.GHI),
                                    ("corr", _R.CMIN, 1.0),
                                    ("cosine", _R.CMIN, 1.0)):
            vals = [x[label] for x in rows]
            his = [x[f"{label}_ci"][1] - x[f"{label}_ci"][0] for x in rows]
            m = sum(vals) / len(vals)
            sd = st.stdev(vals) if len(vals) > 1 else float("nan")
            ciw = st.median(his) / 2.0
            inside = all(lo_ok <= v <= hi_ok for v in vals)
            outside = all(v < lo_ok or v > hi_ok for v in vals)
            verdict = ("PASS every seed" if inside else
                       "FAIL every seed" if outside else
                       "SPLIT ACROSS SEEDS -- no arm-level verdict")
            ratio = (sd / ciw) if ciw and ciw == ciw and sd == sd and ciw > 0 else float("nan")
            flag = ""
            if ratio == ratio and ratio > 1.0:
                flag = f"   << seed_sd is {ratio:.1f}x the within-run CI half-width"
            print(f"    {label:7s} mean {m:7.3f}  seed_sd {sd:6.3f}  "
                  f"ci_halfwidth {ciw:6.3f}  [{min(vals):.3f}..{max(vals):.3f}]  "
                  f"{verdict}{flag}")
    print()

print("A verdict from a single gate report is a statement about one surrogate. Where")
print("the line above reads SPLIT ACROSS SEEDS, that arm has no gate verdict, and any")
print("previously reported one came from whichever seed happened to be run first.")
