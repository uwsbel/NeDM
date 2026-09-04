"""Trajectory of PPO tracking error over iterations, not just its final value.

WHY A TRAJECTORY. The Go2 policy reached 0.02 m position error by iteration 190
of 2000 -- inside the flat region of the reward, where exp(-(e/0.55)^2) has
almost no gradient. Little gradient means little pressure to HOLD position as
well as to improve it, so the failure to watch for is drift or entropy collapse
late in training, not a plateau. A final-value report cannot distinguish
"converged at 0.02" from "hit 0.015 at iteration 400 and drifted back".

Parses the rsl_rl console log rather than TensorBoard: the log is the artifact
that always exists, including for a run launched with --logger none.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

FIELDS = {
    "iteration": re.compile(r"Learning iteration (\d+)/"),
    "reward": re.compile(r"Mean total reward:\s+([-\d.]+)"),
    "ep_len": re.compile(r"Mean episode length:\s+([-\d.]+)"),
    "noise_std": re.compile(r"Mean action noise std:\s+([-\d.]+)"),
    "pos_err": re.compile(r"/tracking/position_error_m:\s+([-\d.]+)"),
    "yaw_err": re.compile(r"/tracking/yaw_error_abs_rad:\s+([-\d.]+)"),
    "track": re.compile(r"/tracking/track_reward:\s+([-\d.]+)"),
}


def parse(log: Path) -> list[dict[str, float]]:
    """One record per iteration block. A block is closed by the next iteration
    header, so a partially-written trailing block is dropped rather than
    reported with stale values from the block before it."""
    records: list[dict[str, float]] = []
    current: dict[str, float] = {}
    for line in log.read_text(errors="replace").splitlines():
        m = FIELDS["iteration"].search(line)
        if m:
            if current.get("iteration") is not None and len(current) > 1:
                records.append(current)
            current = {"iteration": float(m.group(1))}
            continue
        for name, pattern in FIELDS.items():
            if name == "iteration":
                continue
            m = pattern.search(line)
            if m and name not in current:
                current[name] = float(m.group(1))
    if current.get("iteration") is not None and len(current) > 1:
        records.append(current)
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--every", type=int, default=None,
                    help="Print every Nth record. Default picks ~25 rows.")
    a = ap.parse_args()

    records = [r for r in parse(a.log) if "pos_err" in r]
    if not records:
        raise SystemExit(f"no iteration records with tracking logs in {a.log}")
    every = a.every or max(1, len(records) // 25)

    print(f"{len(records)} iterations parsed from {a.log}")
    print(f"{'iter':>6} {'reward':>9} {'ep_len':>7} {'pos_err':>8} {'yaw_err':>8} "
          f"{'track':>7} {'noise':>7}")
    for r in records[::every] + ([records[-1]] if len(records) % every else []):
        print(f"{int(r['iteration']):>6} {r.get('reward', float('nan')):>9.2f} "
              f"{r.get('ep_len', float('nan')):>7.1f} {r['pos_err']:>8.4f} "
              f"{r.get('yaw_err', float('nan')):>8.4f} {r.get('track', float('nan')):>7.4f} "
              f"{r.get('noise_std', float('nan')):>7.3f}")

    best = min(records, key=lambda r: r["pos_err"])
    last = records[-1]
    print(f"\nbest pos_err {best['pos_err']:.4f} m at iteration {int(best['iteration'])}")
    print(f"last pos_err {last['pos_err']:.4f} m at iteration {int(last['iteration'])}")
    drift = last["pos_err"] - best["pos_err"]
    # Drift AFTER the best, which is the thing the flat reward region cannot
    # penalise. Reported as a fraction of the best because 0.005 m means one
    # thing at 0.02 and another at 0.20.
    print(f"drift since best: {drift:+.4f} m ({100 * drift / best['pos_err']:+.1f}%)"
          f"  over {int(last['iteration'] - best['iteration'])} iterations")
    if "noise_std" in records[0] and "noise_std" in last:
        print(f"action noise std: {records[0]['noise_std']:.3f} -> {last['noise_std']:.3f}"
              "   (collapse toward 0 is the other late-training failure mode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
