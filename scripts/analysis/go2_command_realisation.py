#!/usr/bin/env python3
"""Realised-vs-commanded slope and ceiling, per channel, per arm.

SLOPE is fitted on the small-command half only. A single line through a
saturating response is neither the slope nor the ceiling -- it mixes them, and
its value depends on how far the sweep happened to extend.

CEILING is the median |realised| at the largest |command| level, which does not
assume the response ever flattens; if it has not saturated, the ceiling is
simply the largest realised value and the slope will say so.

Both are reported per arm (perturbation applied vs matched control). Untried
channels, so vel_body_x and yaw_rate are never pooled.
"""
import argparse, csv, glob, json, os
import numpy as np

SCORED_ROWS = 1000
CHAN_COL = {"vx": "vel_body_x_mps", "wz": "yaw_rate_radps"}


def realised(csv_path, col):
    try:
        rows = list(csv.DictReader(open(csv_path)))
    except Exception:
        return None
    if len(rows) < SCORED_ROWS:
        return None
    v = np.array([float(r[col]) for r in rows[-SCORED_ROWS:]])
    return float(v.mean()) if np.isfinite(v).all() else None


def boot(fn, data, n=4000, seed=0):
    r = np.random.default_rng(seed)
    k = len(data[0])
    # ONE index vector shared by every array. Drawing one per array resamples x and y
    # independently, which destroys the pairing and regresses shuffled x on shuffled
    # y -- the bootstrap then estimates the null slope regardless of the data. It is
    # visible only because the resulting interval did not contain its own point
    # estimate; a CI that excludes the statistic it is an interval for is impossible.
    vals = []
    for _ in range(n):
        i = r.integers(0, k, k)
        # A cluster resample can draw the same level k times. The design matrix is
        # then rank-deficient and lstsq raises rather than returning nan, so the
        # degenerate draw has to be discarded explicitly -- letting it through
        # aborts the whole analysis on a resampling artefact, not on the data.
        if len(np.unique(data[0][i])) < 2:
            continue
        try:
            vals.append(fn(*[d[i] for d in data]))
        except np.linalg.LinAlgError:
            continue
    vals = np.array([v for v in vals if np.isfinite(v)])
    if len(vals) < n // 10:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-json")
    a = ap.parse_args()

    obs = {}
    for arm in ("treated", "control"):
        for d in sorted(glob.glob(os.path.join(a.root, arm, "*"))):
            name = os.path.basename(d)
            chan = name.split("_")[0]
            if chan not in CHAN_COL:
                continue
            lv = float(name.split("_")[1])
            cs = glob.glob(os.path.join(d, "**", "episodes", "*.csv"), recursive=True)
            if not cs:
                continue
            v = realised(cs[0], CHAN_COL[chan])
            if v is not None:
                obs.setdefault((arm, chan), []).append((lv, v))

    out = {}
    for chan in ("vx", "wz"):
        unit = "m/s" if chan == "vx" else "rad/s"
        print(f"\n=== {CHAN_COL[chan]} ({unit}) ===")
        for arm in ("treated", "control"):
            rec = obs.get((arm, chan))
            if not rec:
                print(f"  {arm:<8} no episodes")
                continue
            c = np.array([r[0] for r in rec]); v = np.array([r[1] for r in rec])
            # REPLICATION CHECK. A control episode consumes no randomness once the
            # perturbation is scaled to zero -- spawn, heading, tilt and prewalk are
            # all fixed -- so its "reps" come out bit-identical and replicate nothing.
            # Treating them as independent inflates precision by sqrt(reps). Detect
            # it rather than assume reps are reps.
            dup = 0
            for lv in np.unique(c):
                g = v[c == lv]
                if len(g) > 1 and float(np.ptp(g)) == 0.0:
                    dup += 1
            if dup:
                print(f"  {arm:<8} NOTE: {dup} of {len(np.unique(c))} levels have "
                      "bit-identical reps -- they replicate nothing; "
                      "intervals below are clustered by level, not by episode")
            half = np.max(np.abs(c)) / 2.0
            m = np.abs(c) <= half
            n_ep, n_sl = len(c), int(m.sum())
            if int(m.sum()) < 3:
                print(f"  {arm:<8} n={n_ep} but only {int(m.sum())} in the linear "
                      "half -- slope NOT MEASURABLE")
                continue
            # CLUSTER BOOTSTRAP over distinct command levels. Resampling episodes
            # would count duplicated reps as independent evidence; the unit that was
            # actually varied is the level.
            lv_u = np.unique(c[m])
            lv_mean = np.array([v[m][c[m] == L].mean() for L in lv_u])
            slope = float(np.polyfit(lv_u, lv_mean, 1)[0])
            slo, shi = boot(lambda x, y: np.polyfit(x, y, 1)[0], (lv_u, lv_mean))
            n_sl = len(lv_u)
            if np.isfinite(slo) and not (slo <= slope <= shi):
                raise SystemExit(
                    f"bootstrap CI [{slo:.3f}, {shi:.3f}] excludes the point estimate "
                    f"{slope:.3f} for {chan}/{arm}. A percentile interval cannot omit "
                    "its own statistic; the resampling is wrong, not the data.")
            top = np.max(np.abs(c))
            tm = np.abs(np.abs(c) - top) < 1e-9
            ceil = float(np.median(np.abs(v[tm])))
            # Only +top and -top are distinct here, so this interval rests on 2
            # clusters and is reported for completeness rather than as inference.
            clo, chi = boot(lambda y: np.median(np.abs(y)), (v[tm],))
            frac = ceil / top if top > 1e-9 else float("nan")
            print(f"  {arm:<8} n={n_ep:<3} ({n_sl} levels) slope {slope:.3f} [{slo:.3f}, {shi:.3f}]"
                  f"   ceiling {ceil:.3f} [{clo:.3f}, {chi:.3f}] {unit}"
                  f"  at |cmd|={top:.2f}  ({frac:.0%} of commanded)")
            out[f"{chan}_{arm}"] = dict(n=n_ep, n_slope=n_sl, slope=slope,
                                        slope_ci=[slo, shi], ceiling=ceil,
                                        ceiling_ci=[clo, chi], top_cmd=float(top),
                                        realised_fraction=frac)
        t, c_ = out.get(f"{chan}_treated"), out.get(f"{chan}_control")
        if t and c_:
            ds, dc = t["slope"] - c_["slope"], t["ceiling"] - c_["ceiling"]
            print(f"  perturbation effect:  slope {ds:+.3f}   ceiling {dc:+.3f} {unit}")
            print("  (arms share seeds and draws; they differ only in applied force)")
            out[f"{chan}_delta"] = dict(slope=ds, ceiling=dc)

    if a.out_json:
        json.dump(out, open(a.out_json, "w"), indent=2)
        print(f"\nwrote {a.out_json}")


if __name__ == "__main__":
    main()
