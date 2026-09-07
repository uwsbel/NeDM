"""Split a verdict's paired differences by condition family, with episode bootstrap.

WHY. The pooled paired difference is now the LEAST informative summary available.
On the full 36-episode stratum it read +0.00006 while the two halves were
-0.0385 straight and +0.0395 turning -- two significant effects of opposite sign
cancelling to a flat line. A pre-registered, conservative, correctly-applied
single-endpoint criterion returned "no effect" on data containing two large ones.

Pre-registration was right; the single pooled endpoint was wrong. So report both,
and label which was registered.

Bootstrap resamples EPISODES, not pairs-as-independent-observations, because the
paired difference is one number per episode already -- but the interval must still
come from the episode count, which is what `n` is here.
"""
import argparse, json
import numpy as np

STRAIGHT = ("constant", "vel_step", "stop_and_go")


def bucket(fam):
    return "straight" if any(s in fam for s in STRAIGHT) else "turning"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_json")
    ap.add_argument("--boot", type=int, default=20000)
    a = ap.parse_args()
    d = json.loads(open(a.summary_json).read())
    pairs = d["pairs"]
    if not pairs:
        raise SystemExit(f"  no pairs in {a.summary_json} -- nothing survived to split")
    rng = np.random.default_rng(0)
    groups = {"ALL (registered endpoint)": pairs,
              "straight": [p for p in pairs if bucket(p["family"]) == "straight"],
              "turning": [p for p in pairs if bucket(p["family"]) == "turning"]}
    print(f"\n  {a.summary_json}   machine {d.get('machine')}   n {d.get('n')}")
    print(f"  {'group':28s} {'n':>4s} {'median diff':>12s} {'bootstrap 95% CI':>26s}")
    for g, ps in groups.items():
        if not ps:
            print(f"  {g:28s} {0:4d}   (no episodes in this stratum)")
            continue
        x = np.array([p["difference"] for p in ps])
        boot = np.array([np.median(rng.choice(x, size=len(x), replace=True))
                         for _ in range(a.boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        # (lo > 0) == (hi > 0) is TRUE for a degenerate interval at exactly zero --
        # two Falses compare equal -- so the base control's [0,0] reported
        # "excludes 0". Testing the sign explicitly instead. Same class as every
        # other failure that produced benign-looking output this week.
        excl = "excludes 0" if (lo > 0 or hi < 0) else "INCLUDES 0"
        print(f"  {g:28s} {len(x):4d} {np.median(x):+12.5f}   [{lo:+.5f},{hi:+.5f}]  {excl}")
    fams = {}
    for p in pairs:
        fams.setdefault(p["family"], []).append(p["difference"])
    print(f"\n  by family: " + "  ".join(f"{k}={len(v)}" for k, v in sorted(fams.items())))
    print("  ONLY the pooled row is the registered endpoint. The split is an")
    print("  unregistered observation, reported because a pooled null has twice")
    print("  concealed structure this project cared about.")
