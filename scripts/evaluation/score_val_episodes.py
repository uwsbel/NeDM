"""Run a policy in Chrono on the corpus's held-out episodes and record, per episode,
whether it completed and what ground pitch it was under.

Every arm gets the SAME episode list, taken from the merged index's val split, so
the comparison is paired per episode and the banded evaluator can use McNemar.

Completion is `rows >= SCORED_ROWS + 500`, imported from the verdict harness rather
than restated -- the same constant the corpus filter uses, so the three cannot drift.

Ground pitch is read from each episode's sidecar. It is NOT re-derived from a seeded
RNG: that reconstruction silently disagreed with the driver after the pitch cap and
produced half-scale values for hours tonight.
"""
import argparse, glob, json, math, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import importlib.util

_C = [os.path.join(os.environ.get("NEDM_REPO", ""), "scripts", "evaluation",
                   "run_go2_finetune_verdict.py"),
      "/home/kyle/Documents/sbel/NeDM/scripts/evaluation/run_go2_finetune_verdict.py",
      "/home/kyle/sbel/NeDM/scripts/evaluation/run_go2_finetune_verdict.py"]
_H = next((c for c in _C if c and os.path.exists(c)), None)
if _H is None:
    raise SystemExit("FATAL: cannot locate the verdict harness to import SCORED_ROWS")
_s = importlib.util.spec_from_file_location("_v", _H)
_V = importlib.util.module_from_spec(_s); _s.loader.exec_module(_V)
THRESH = _V.SCORED_ROWS + 500

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--split", default="val")
ap.add_argument("--out", required=True)
ap.add_argument("--concurrency", type=int, default=12)
a = ap.parse_args()

idx = json.load(open(f"{a.root}/dataset_index.json"))
eps = [e for e in idx["episodes"] if e.get("split") == a.split]
print(f"policy {a.policy}")
print(f"episodes in split '{a.split}': {len(eps)}   completion threshold {THRESH} rows")

PY_ = sys.executable
CH = os.environ.get("PYTHONPATH", "").split(":")[0]

_SEEN = set()   # distinct collector errors already reported


def run(e):
    j = e["csv_path"][:-4] + ".json"
    m = json.load(open(j))
    out = tempfile.mkdtemp(prefix="score_")
    cmd = [PY_, "scripts/collection/collect_go2_smoke.py", "--terrain", "rigid",
           "--duration-s", str(m["duration_s"]), "--imported-ckpt", a.policy,
           "--command-family", m["command_family"],
           "--command-params", json.dumps(m["command_params"]),
           "--ground-size-m", "200.0", "--perturb-peak-n", f"{m['perturb_peak_n']:.1f}",
           "--prewalk-s", f"{m['prewalk_s']:.2f}",
           "--ground-tilt-roll-deg", f"{m['ground_tilt_roll_deg']:.2f}",
           "--ground-tilt-pitch-deg", f"{m['ground_tilt_pitch_deg']:.2f}",
           "--episode-index", "0", "--seed", str(m["seed"]), "--spawn-x-m", "0.0",
           "--spawn-y-m", "0.0", "--heading-deg", "0.0", "--patch-y", "4.0",
           "--output-dir", out, "--overwrite", "--progress-interval-s", "99"]
    r = subprocess.run(cmd, env=dict(os.environ), capture_output=True, text=True)
    f = glob.glob(f"{out}/episodes/*.csv")
    n = (sum(1 for _ in open(f[0])) - 1) if f else 0
    # Remove the scratch episode. One pass writes 536 of these and only the row count
    # is ever read; leaving them behind put 67 GB across 16,352 /tmp/score_* directories
    # on sbel, which is disk the collector then competes for.
    shutil.rmtree(out, ignore_errors=True)
    # SURFACE THE COLLECTOR'S ERROR. Capturing stderr and discarding it is the exact
    # defect fixed in standing_screen.py earlier today, reintroduced here: all 536
    # episodes returned 0 rows and the only signal was "completed 0 of 536", which
    # reads as a policy that fails everything rather than a collector that never ran.
    if n == 0:
        why = (r.stderr or r.stdout or "").strip().splitlines()
        msg = why[-1] if why else f"exit {r.returncode}, no output"
        if msg not in _SEEN:
            _SEEN.add(msg)
            print(f"  [no episode] {msg}", file=sys.stderr, flush=True)
    # EPISODE_ID, NOT SEED. Seeds are assigned per family per index and collide
    # across families and shards: 536 scored episodes carried only 23 distinct
    # seeds, one repeated 29 times. Keying a comparison on seed silently collapsed
    # the paired set to 23 -- the same collision that dropped 679 of 789 episodes
    # in the merger, in a different file.
    return dict(episode_id=e["episode_id"], seed=m["seed"],
                pitch=float(m["ground_tilt_pitch_deg"]),
                roll=float(m["ground_tilt_roll_deg"]),
                family=m["command_family"], rows=n, completed=int(n >= THRESH))

with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
    recs = list(ex.map(run, eps))
json.dump(recs, open(a.out, "w"), indent=1)
k = sum(r["completed"] for r in recs)
print(f"completed {k} of {len(recs)}  ({100*k/max(len(recs),1):.1f}%)  -> {a.out}")
