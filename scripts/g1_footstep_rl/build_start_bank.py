"""Build the G1 start bank: start states for policies in the NRD, from the TRAINING split of production v1.

Each start is a control step k of a recorded nominal or difficult training episode (split 0; validation, which also
selected the NRD checkpoint, and the sealed test are not used). Fields (nedm.g1_footstep_rl.bank.build_starts): the NRD
history (4 states, 20 ms step-average rates, ok mask), the HOMIE teacher memory before step k reconstructed from the
recorded states with averaged rates (6 frames + last clipped output), the recorded command, raw teacher output and
applied action of step k, and labels (family, segment kind, time in segment, commanded and measured speed, foot
contacts before / after, steps to the episode end, the difficult flags).

Sampling: n_nominal + n_difficult starts drawn uniformly without replacement over all valid (episode, k) of each
family: k >= 3 (a full 4-state history), at least one control step after the start, and for episodes that ended
early (self-contact etc.) not within the last 0.5 s before the end.

Checks written to the summary (every start):
- teacher reproduction: a HomieTeacher step from the start reproduces the recorded teacher targets of step k
  (default + 0.25 clip(policy_output[2k])); once with the bank memory (averaged rates, what the teacher sees in the
  NRD) and once with the same reconstruction from the stored instantaneous rates (what it saw in MuJoCo). The second
  differs only by the float32 rounding of the stored states, so the first minus the second is the effect of the
  averaged rates.
- data envelope of the training task data (nedm.g1_footstep_rl.sim.envelope_features): min / max over every valid record
  of the loaded episodes on the 20 ms grid, and quantiles of a subsample (every 5th grid state).

    PYTHONPATH=src python scripts/g1_footstep_rl/build_start_bank.py \
        [--data artifacts/contact_nrd/g1_locomotion/train] [--out artifacts/g1_footstep_rl/rebuilt/bank_v1.npz] \
        [--n-nominal 140000] [--n-difficult 60000] [--threads 8]

The released bank is g1_footstep_rl/rl_inputs/start_bank/ on the Hub; the default output keeps a rebuild apart from it.
The teacher comes from deploy.onnx of the vendor checkout (scripts/g1_footstep_rl/fetch_homie.py).
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch

from nedm.g1_footstep_rl import bank as B
from nedm.g1_footstep_rl.homie import HomieNet, HomieTeacher
from nedm.g1_footstep_rl.sim import ENVELOPE_FEATURES, envelope_features

MIN_K = 3                 # first control step with a full 4-state NRD history (t = 0.06 s)
EARLY_END_MARGIN = 25     # control steps (0.5 s) kept away from an early episode end


def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def pct(v, ps=(0, 1, 5, 50, 95, 99, 100)):
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return {f"p{p}": float(np.percentile(v, p)) for p in ps} if len(v) else None


def valid_k_range(lengths, terminal):
    """Per episode: (first, last) valid start control step (last < first = none)."""
    last = (lengths - 3) // B.STRIDE                       # record 2k + 2 must exist (one control step after)
    last = np.where(terminal != 0, last - EARLY_END_MARGIN, last)
    return np.full(len(lengths), MIN_K), last


def draw(rows, lengths, terminal, n, rng):
    """n starts (row, k) uniformly over all valid (episode, k) pairs of the given rows, without replacement."""
    first, last = valid_k_range(lengths, terminal)
    count = np.maximum(0, last - first + 1)
    cum = np.cumsum(count)
    if n > cum[-1]:
        raise ValueError(f"only {cum[-1]} valid starts, {n} requested")
    pick = np.sort(rng.choice(cum[-1], size=n, replace=False))
    ep = np.searchsorted(cum, pick, side="right")
    k = first[ep] + pick - (cum[ep] - count[ep])
    return rows[ep], k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=B.DATA)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[2] / "artifacts/g1_footstep_rl/rebuilt/bank_v1.npz")
    ap.add_argument("--n-nominal", type=int, default=140_000)
    ap.add_argument("--n-difficult", type=int, default=60_000)
    ap.add_argument("--chunk", type=int, default=256, help="episodes loaded at a time")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--limit-episodes", type=int, default=0, help="smoke test: use only this many episodes per family")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0, c0 = time.perf_counter(), cpu_s()
    rng = np.random.default_rng(args.seed)
    arr = B.open_arrays(args.data)
    splits, family = np.asarray(arr["splits"]), np.asarray(arr["family"])
    lengths, terminal = np.asarray(arr["lengths"]).astype(np.int64), np.asarray(arr["terminal_reason"])

    # ---- choose the starts --------------------------------------------------------------------------------------
    picks, mix = [], {}
    for name, n in (("nominal", args.n_nominal), ("difficult", args.n_difficult)):
        rows = np.flatnonzero((splits == 0) & (family == B.FAMILIES.index(name)))
        if args.limit_episodes:
            rows = rows[:args.limit_episodes]
        r, k = draw(rows, lengths[rows], terminal[rows], n, rng)
        picks.append((r, k))
        first, last = valid_k_range(lengths[rows], terminal[rows])
        mix[name] = dict(episodes_available=int(len(rows)), episodes_used=int(len(np.unique(r))), starts=int(n),
                         valid_starts_available=int(np.maximum(0, last - first + 1).sum()))
    rows_all = np.concatenate([p[0] for p in picks])
    k_all = np.concatenate([p[1] for p in picks])
    order = np.lexsort((k_all, rows_all))
    rows_all, k_all = rows_all[order], k_all[order]
    if (splits[rows_all] != 0).any():
        raise RuntimeError("a start outside the training split")
    episodes = np.unique(rows_all)
    print(f"{len(rows_all)} starts from {len(episodes)} training episodes", flush=True)

    # ---- load episode chunks, build the starts, run the checks ------------------------------------------------------
    net = HomieNet.load()
    parts, check = [], {"avg_err": [], "inst_err": [], "avg_minus_inst": [], "applied_err": [], "noise": []}
    feat_min = np.full(len(ENVELOPE_FEATURES), np.inf)
    feat_max = np.full(len(ENVELOPE_FEATURES), -np.inf)
    feat_sub = []
    env_rows = np.flatnonzero((splits == 0) & (family <= 1))     # envelope: every training nominal/difficult episode
    if args.limit_episodes:
        env_rows = np.intersect1d(env_rows, episodes)
    todo = np.union1d(episodes, env_rows)
    for a in range(0, len(todo), args.chunk):
        chunk = todo[a:a + args.chunk]
        ep = B.load_episodes(chunk, args.data, arr)
        # envelope over all valid grid states of these episodes
        grid = np.arange(0, ep["avg"].shape[1], B.STRIDE)
        valid = grid[None, :] <= (ep["lengths"][:, None] - 1)
        f = envelope_features(torch.as_tensor(ep["avg"][:, grid])).numpy()[valid]
        feat_min, feat_max = np.minimum(feat_min, f.min(0)), np.maximum(feat_max, f.max(0))
        feat_sub.append(f[::5].astype(np.float32))
        # starts in this chunk
        sel = np.isin(rows_all, chunk)
        if sel.any():
            e = np.searchsorted(chunk, rows_all[sel])
            k = k_all[sel]
            st = B.build_starts(ep, e, k)
            parts.append(st)
            inst_h, inst_l = B.teacher_memory(ep["raw"], ep["command"], ep["policy_output"], e, k)
            want = B.teacher_targets(st["policy_output"])
            cmd = torch.as_tensor(st["command"])
            with torch.no_grad():
                t_avg = HomieTeacher(len(e), net=net)
                t_avg.reset(None, torch.as_tensor(st["teacher_history"]), torch.as_tensor(st["teacher_last_action"]))
                _, legs_avg = t_avg.step(cmd, torch.as_tensor(st["hist"][:, -1]))
                t_inst = HomieTeacher(len(e), net=net)
                t_inst.reset(None, torch.as_tensor(inst_h), torch.as_tensor(inst_l))
                _, legs_inst = t_inst.step(cmd, torch.as_tensor(ep["raw"][e, k * B.STRIDE]))
            la, li = legs_avg.numpy(), legs_inst.numpy()
            check["avg_err"].append(np.abs(la - want))
            check["inst_err"].append(np.abs(li - want))
            check["avg_minus_inst"].append(np.abs(la - li))
            check["applied_err"].append(np.abs(li - st["action"][:, :12]))
            check["noise"].append(st["leg_noise"] > 0)
        print(f"episodes {a + len(chunk)}/{len(todo)}  starts {sum(len(p['k']) for p in parts)}  "
              f"{time.perf_counter() - t0:.0f} s", flush=True)

    bank = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    n = len(bank["k"])
    # ---- summary ----------------------------------------------------------------------------------------------------
    ce = {k: np.concatenate(v) for k, v in check.items()}
    noise = ce["noise"]

    def err_stats(x):
        per_start = x.max(1)
        return {"max_rad": float(x.max()), "p95_rad": float(np.percentile(x, 95)), "p50_rad": float(np.median(x)),
                "per_start_max_p95_rad": float(np.percentile(per_start, 95)),
                "per_joint_p95_rad": np.percentile(x, 95, axis=0).round(5).tolist()}

    teacher_check = {
        "what": "HomieTeacher step from each start vs the recorded teacher leg targets of step k "
                "(default + 0.25 clip(policy_output[2k])); errors over all 12 leg joints of all starts",
        "averaged_rates_bank_memory": err_stats(ce["avg_err"]),
        "instantaneous_rates_same_reconstruction": err_stats(ce["inst_err"]),
        "averaged_minus_instantaneous": err_stats(ce["avg_minus_inst"]),
        "instantaneous_vs_recorded_applied_action_no_leg_noise": err_stats(ce["applied_err"][~noise]),
        "starts": int(n), "starts_with_leg_noise": int(noise.sum()),
        "by_family": {name: {"averaged": err_stats(ce["avg_err"][bank["family"] == f]),
                             "instantaneous": err_stats(ce["inst_err"][bank["family"] == f])}
                      for f, name in enumerate(("nominal", "difficult"))},
        "by_kind_averaged_p95_rad": {kn: float(np.percentile(ce["avg_err"][bank["kind"] == i], 95))
                                     for i, kn in enumerate(B.KINDS) if (bank["kind"] == i).any()}}
    sub = np.concatenate(feat_sub)
    envelope = {"what": "nedm.g1_footstep_rl.sim.envelope_features of the averaged-rate states of all training nominal "
                        "and difficult episodes on the 20 ms grid (min/max: every valid grid state; quantiles: every "
                        "5th)", "features": list(ENVELOPE_FEATURES), "states": int(sum(len(x) for x in feat_sub) * 5),
                "min": feat_min.tolist(), "max": feat_max.tolist()}
    for p in (0.1, 0.5, 99.5, 99.9):
        envelope[f"p{p:g}"] = np.percentile(sub, p, axis=0).tolist()
    con = bank["contact_after"]
    kinds = {kn: float((bank["kind"] == i).mean()) for i, kn in enumerate(B.KINDS)}
    summary = {
        "data": str(args.data), "split": "training (0) only", "families": mix, "starts": int(n), "seed": args.seed,
        "sampling": f"uniform over valid (episode, control step k): k >= {MIN_K}, one control step after the start, "
                    f"early-ended episodes not within the last {EARLY_END_MARGIN} control steps",
        "share": {"nominal": float((bank["family"] == 0).mean()), "difficult": float((bank["family"] == 1).mean()),
                  "difficult_with_leg_noise": float(((bank["family"] == 1) & (bank["leg_noise"] > 0)).mean()),
                  "difficult_with_upper_motion": float(((bank["family"] == 1) & (bank["upper_motion"] > 0)).mean()),
                  "difficult_abrupt": float(((bank["family"] == 1) & (bank["abrupt_commands"] > 0)).mean()),
                  "difficult_perturbed_start": float(((bank["family"] == 1) & (bank["perturbed_start"] > 0)).mean()),
                  "kind": kinds,
                  "contact_after": {"both": float(con.all(1).mean()), "left_only": float((con[:, 0] & ~con[:, 1]).mean()),
                                    "right_only": float((con[:, 1] & ~con[:, 0]).mean()), "none": float((~con.any(1)).mean())}},
        "time_s": pct(bank["time_s"]), "t_seg_s": pct(bank["t_seg"]),
        "measured_avg_vel_fwd": pct(bank["vel_avg"][:, 0]), "measured_avg_vel_lat": pct(bank["vel_avg"][:, 1]),
        "measured_avg_wz": pct(bank["vel_avg"][:, 2]),
        "measured_0p5s_mean_fwd": pct(bank["vel_mean_0p5s"][:, 0]),
        "commanded_vx": pct(bank["command"][:, 0]), "commanded_vy": pct(bank["command"][:, 1]),
        "commanded_wz": pct(bank["command"][:, 2]), "commanded_height": pct(bank["command"][:, 3]),
        "pelvis_z": pct(bank["hist"][:, -1, 2]),
        "distance_from_origin_m": pct(np.hypot(bank["hist"][:, -1, 0], bank["hist"][:, -1, 1])),
        "teacher_check": teacher_check, "envelope": envelope,
        "fields": {k: [list(v.shape[1:]), str(v.dtype)] for k, v in bank.items()},
        "kinds": list(B.KINDS), "family_names": list(B.FAMILIES),
        "wall_s": time.perf_counter() - t0, "cpu_s": cpu_s() - c0, "threads": args.threads}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **bank)
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    (args.out.parent / "envelope_v1.json").write_text(json.dumps(envelope, indent=1) + "\n")
    print(json.dumps({k: teacher_check[k] for k in ("averaged_rates_bank_memory", "instantaneous_rates_same_reconstruction",
                                                    "averaged_minus_instantaneous")}, indent=1))
    print(json.dumps(summary["share"], indent=1))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.0f} MB), {summary['wall_s']:.0f} s wall, "
          f"{summary['cpu_s']:.0f} s CPU")


if __name__ == "__main__":
    main()
