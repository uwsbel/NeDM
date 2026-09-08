"""Score a policy on episodes the BASE controller FAILED.

The val split is selected for base succeeding, so it can only report degradation.
This is its complement: base completes 0 of these by construction, so it can only
report capability. Neither is interpretable alone; the pair is, and they are never
summed -- the same two-conditional-rates form settled on for v4's complement.

The set is disproportionately nose-down (443 of 522 in the two steepest bands
against 74 in the val split), because that is where base fails -- which makes it
the better test of tilt observability, not merely a complement to it.
"""
import argparse, glob, json, os, shutil, subprocess, sys, tempfile
from concurrent.futures import ThreadPoolExecutor
import importlib.util
_C = [os.path.join(os.environ.get("NEDM_REPO", ""), "scripts", "evaluation",
                   "run_go2_finetune_verdict.py"),
      "/home/kyle/Documents/sbel/NeDM/scripts/evaluation/run_go2_finetune_verdict.py"]
_H = next((c for c in _C if c and os.path.exists(c)), None)
if _H is None: raise SystemExit("FATAL: cannot locate the verdict harness")
_s = importlib.util.spec_from_file_location("_v", _H)
_V = importlib.util.module_from_spec(_s); _s.loader.exec_module(_V)
THRESH = _V.SCORED_ROWS + 500

ap = argparse.ArgumentParser()
ap.add_argument("--policy", required=True)
ap.add_argument("--index", required=True, help="basefail_index.json")
ap.add_argument("--out", required=True)
ap.add_argument("--concurrency", type=int, default=12)
a = ap.parse_args()
eps = json.load(open(a.index))

def _uid(e):
    """Unique key for an episode, derived from its path rather than trusted.

    The index's own `episode_id` namespaces on the scenario dir but NOT on the
    shard root, so the same scenario/episode in two shards collides: 522 episodes
    carried 416 distinct ids, 98 of them shared across off5000000/off7000000.
    Those are different episodes -- different RNG offsets -- and a downstream
    dict keyed on the id silently keeps one of each pair. Derive from csv_path,
    which was already distinct 522 of 522, and refuse to run if that is not
    unique either. Never trust an id you can recompute.
    """
    q = os.path.normpath(e["csv_path"]).split(os.sep)
    stem = os.path.splitext(q[-1])[0]
    return "_".join(q[-4:-2] + [stem]) if len(q) >= 4 else stem

_ids = [_uid(e) for e in eps]
if len(set(_ids)) != len(eps):
    raise SystemExit(f"FATAL: {len(eps)} episodes yield only {len(set(_ids))} distinct "
                     f"keys -- pairing would be silently wrong. Refusing to score.")
for e, u in zip(eps, _ids):
    e["_uid"] = u
print(f"policy {a.policy}\nbase-failed episodes: {len(eps)}   threshold {THRESH} rows")
_SEEN = set()

def run(e):
    m = json.load(open(e["csv_path"][:-4] + ".json"))
    out = tempfile.mkdtemp(prefix="bf_")
    cmd = [sys.executable, "scripts/collection/collect_go2_smoke.py", "--terrain", "rigid",
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
    if n == 0:
        why = (r.stderr or r.stdout or "").strip().splitlines()
        msg = why[-1] if why else f"exit {r.returncode}"
        if msg not in _SEEN:
            _SEEN.add(msg); print(f"  [no episode] {msg}", file=sys.stderr, flush=True)
    return dict(episode_id=e["_uid"], index_episode_id=e["episode_id"],
                csv_path=e["csv_path"], pitch=e["pitch"], rows=n,
                completed=int(n >= THRESH))

with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
    recs = list(ex.map(run, eps))
json.dump(recs, open(a.out, "w"), indent=1)
k = sum(r["completed"] for r in recs)
print(f"completed {k} of {len(recs)}  ({100*k/max(len(recs),1):.1f}%)  "
      f"-- base completes 0 of these by construction  -> {a.out}")
