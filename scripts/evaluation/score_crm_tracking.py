"""Score a policy on CRM episodes by VELOCITY TRACKING ERROR, not completion.

Why this exists rather than reusing the rigid scorer. On CRM the robot does not fall:
14 pilot episodes gave 0 falls, 0 divergences, 0 boundary truncations and full-length
rows, with the feet carrying body weight throughout. It survives and fails to move
instead, achieving 34-85% of commanded velocity. A completion-rate metric would score
every arm, the baseline AND a displacement-matched random control at ~100% -- a flat
table that reads as a null result and is actually a metric failure.

Why not `run_go2_finetune_verdict.scored()`. That predicate returns None unless the
episode has SCORED_ROWS + 500 rows AND the command is constant over the scored window.
CRM episodes are 12 s (bed-bounded, not budget-bounded) and the WIDE envelope is mostly
time-varying families (vel_step, weave, yaw_step, stop_and_go). It would reject nearly
the whole corpus, and the rejections would correlate with command family -- which is
exactly the selection-shaped bias that made the rigid base rates uninterpretable.

So: compare against the INSTANTANEOUS command, on whatever rows the episode has after
warm-up. Partial episodes stay valid, which matters because the wide envelope will push
fast commands into the bed boundary and discarding those would silently drop the fastest
commands -- the ones where the tracking deficit is largest.

Reports per episode so the seed-level evaluator can pair on episode identity, and emits
`completed` too, so the claim "completion does not discriminate on CRM" is measured
here rather than assumed from the pilot.
"""
import argparse, csv, glob, json, math, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor

AXES = (("cmd_vx_mps", "vel_body_x_mps"),
        ("cmd_vy_mps", "vel_body_y_mps"),
        ("cmd_wz_radps", "yaw_rate_radps"))

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--index", required=True, help="dataset_index.json of the CRM corpus")
ap.add_argument("--split", default="val")
ap.add_argument("--out", required=True)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--warmup-s", type=float, default=1.25,
                help="pose ramp + settle; recording already starts after it, this is a "
                     "second guard for episodes collected with --log-warmup")
ap.add_argument("--min-rows", type=int, default=200,
                help="~2 s at 93 rows/s. Below this the mean is noise, and an episode "
                     "that short means the run failed rather than tracked badly.")
a = ap.parse_args()

idx = json.load(open(a.index))["episodes"]
eps = [e for e in idx if e.get("split") == a.split] or idx
print(f"policy {a.policy}\nepisodes: {len(eps)}  (split {a.split!r})")

PY_ = sys.executable
CH = os.environ.get("PYTHONPATH", "").split(":")[0]
_SEEN = set()


def uid(cp):
    q = os.path.normpath(cp).split(os.sep)
    stem = os.path.splitext(q[-1])[0]
    return "_".join(q[-4:-2] + [stem]) if len(q) >= 4 else stem


def track_error(rows):
    """Mean |achieved - commanded| per axis over the post-warm-up rows."""
    keep = [r for r in rows if float(r["time_s"]) >= a.warmup_s]
    if len(keep) < a.min_rows:
        return None
    out = {}
    for cmd_c, act_c in AXES:
        if cmd_c not in keep[0] or act_c not in keep[0]:
            continue
        errs, cs, vs = [], [], []
        for r in keep:
            try:
                c, v = float(r[cmd_c]), float(r[act_c])
            except (ValueError, KeyError):
                return None
            if not (math.isfinite(c) and math.isfinite(v)):
                return None
            errs.append(abs(v - c)); cs.append(c); vs.append(v)
        n = len(errs)
        out[cmd_c.replace("cmd_", "").replace("_mps", "").replace("_radps", "")] = dict(
            mae=sum(errs) / n,
            mean_cmd=sum(cs) / n,
            mean_ach=sum(vs) / n)
    return (out, len(keep)) if out else None


def _patch(e, key, default):
    """Read a terrain patch dimension from the episode's own collector config."""
    cfg = e["csv_path"][:-4] + ".config.json"
    try:
        with open(cfg) as fh:
            return float(json.load(fh)["terrain"][key])
    except Exception:
        return float(default)


def run(e):
    m = json.load(open(e["csv_path"][:-4] + ".json"))
    out = tempfile.mkdtemp(prefix="crmtrk_")
    cmd = [PY_, "scripts/collection/collect_go2_smoke.py", "--terrain", "crm",
           "--duration-s", str(m["duration_s"]), "--imported-ckpt", a.policy,
           "--command-family", m["command_family"],
           "--command-params", json.dumps(m["command_params"]),
           "--spawn-x-m", f"{m['spawn_m'][0]:.4f}", "--spawn-y-m", f"{m['spawn_m'][1]:.4f}",
           "--heading-deg", f"{m.get('heading_deg', 0.0):.4f}",
           # patch_y_m lives ONLY in the per-episode collector config, never in the
           # episode sidecar -- consolidate_go2_dataset.py keeps that file for exactly
           # this reason. Reading it from the sidecar silently yields the 4.0 default,
           # which gives a bed of [-2, 2]; the lateral family is collected on an 8 m
           # bed and spawns at y = +/-3, so every lateral episode was refused by the
           # spawn guard. The default is the trap, not the guard.
           "--patch-x", f"{_patch(e, 'patch_x_m', 8.0):.2f}",
           "--patch-y", f"{_patch(e, 'patch_y_m', 4.0):.2f}",
           "--episode-index", "0", "--seed", str(m["seed"]),
           "--output-dir", out, "--overwrite", "--progress-interval-s", "99"]
    r = subprocess.run(cmd, env=dict(os.environ), capture_output=True, text=True)
    f = glob.glob(f"{out}/episodes/*.csv")
    rows = list(csv.DictReader(open(f[0]))) if f else []
    shutil.rmtree(out, ignore_errors=True)   # 16,352 leaked dirs / 67 GB once. Not again.
    if not rows:
        why = (r.stderr or r.stdout or "").strip().splitlines()
        msg = why[-1] if why else f"exit {r.returncode}"
        if msg not in _SEEN:
            _SEEN.add(msg); print(f"  [no episode] {msg}", file=sys.stderr, flush=True)
    t = track_error(rows) if rows else None
    rec = dict(episode_id=uid(e["csv_path"]), csv_path=e["csv_path"],
               family=m["command_family"], rows=len(rows),
               completed=int(len(rows) >= a.min_rows))
    if t:
        axes, n = t
        rec["scored_rows"] = n
        for k, v in axes.items():
            rec[f"mae_{k}"] = v["mae"]
            rec[f"cmd_{k}"] = v["mean_cmd"]
            rec[f"ach_{k}"] = v["mean_ach"]
    return rec


# PROGRESS. A CRM scoring pass is ~80 minutes and previously printed nothing until it
# finished, so a run that was merely slow looked identical to one that had hung -- and
# distinguishing those cost a diagnostic detour mid-round. Report every 10 episodes.
recs = []
with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
    for i, rec in enumerate(ex.map(run, eps), 1):
        recs.append(rec)
        if i % 10 == 0 or i == len(eps):
            done = [r for r in recs if "mae_vx" in r]
            avg = (sum(r["mae_vx"] for r in done) / len(done)) if done else float("nan")
            print(f"  [{i}/{len(eps)}] scored {len(done)}  running mean|err_vx| {avg:.4f} m/s",
                  flush=True)

keys = [r["episode_id"] for r in recs]
if len(set(keys)) != len(keys):
    raise SystemExit(f"FATAL: {len(recs)} records but {len(set(keys))} distinct keys -- "
                     f"pairing would be silently wrong")
ok = [r for r in recs if "mae_vx" in r]
# A RUN THAT SCORES NOTHING IS A BROKEN MACHINE, NOT A BAD POLICY -- FAIL, DO NOT WRITE.
#
# On 2026-09-11 an unattended driver upgrade left a3 and sliger with the NVIDIA kernel
# module at 580.173.02 while userspace had moved to 580.178.04. Every episode died in
# terrain.Initialize() with cudaErrorCompatNotSupportedOnDevice. The per-episode failure
# path is a `continue` -- correct, since one bad episode should not sink a run -- so the
# scorer sailed through all 80, printed "scored 0 of 80", exited 0, and WROTE A RESULT
# FILE containing zero scored episodes.
#
# That file is worse than no file. The dispatcher reads "a result exists here" and never
# retries the arm; the aggregator skips it as empty. Two boxes produced nothing for
# roughly an hour and reported success, and 13 such files had to be hunted down and
# deleted by hand before the arms could be rescored.
#
# The write is moved AFTER this check so the poisoned artefact is never created.
if not ok:
    raise SystemExit(
        f"FATAL: 0 of {len(recs)} episodes scored. Not a policy result -- this machine "
        "cannot run CRM at all. Check the GPU before anything else:\n"
        "  nvidia-smi                      (NVML version mismatch => driver upgraded "
        "under a running kernel module)\n"
        "  cat /proc/driver/nvidia/version (loaded module) vs the installed libnvidia-ml\n"
        "Fix without rebooting: stop the display manager (it pins nvidia_drm), rmmod "
        "nvidia_uvm nvidia_drm nvidia_modeset nvidia, modprobe nvidia nvidia_uvm, start "
        "it again. NO result file was written, so re-running will retry this arm.")

json.dump(recs, open(a.out, "w"), indent=1)
comp = sum(r["completed"] for r in recs)
print(f"  scored {len(ok)} of {len(recs)};  completed {comp} of {len(recs)} "
      f"({100*comp/max(len(recs),1):.1f}%)")
if ok:
    mvx = sum(r["mae_vx"] for r in ok) / len(ok)
    cvx = sum(r["cmd_vx"] for r in ok) / len(ok)
    avx = sum(r["ach_vx"] for r in ok) / len(ok)
    print(f"  vx: mean|err| {mvx:.4f} m/s   mean cmd {cvx:+.3f}   mean achieved {avx:+.3f}")
    print(f"  -> {a.out}")
print("  NOTE: completion is reported to DEMONSTRATE it does not discriminate on CRM,")
print("        not to be used as the metric. The metric is mae_vx.")
