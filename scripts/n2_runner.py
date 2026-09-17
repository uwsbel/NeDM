"""Night-2 closed loop: all arms of a group share one array task (node-deterministic). Identical picks run once."""
import json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
C = "/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909"
OUT = C + "/night2_closed_v1"
COLLECT = C + "/source_v1/scripts/collect_traverse_f104.py"
CASES = C + "/cases_test_final_v1"
PY = os.environ.get("NRD_PYTHON", "python3")
aid = int(os.environ["F104_ARRAY_ID"])
tasks = [t for t in json.load(open(OUT + "/tasks_cluster.json")) if t["shard"] == aid and t["run"]]
os.makedirs(OUT + "/runs", exist_ok=True); os.makedirs(OUT + "/logs", exist_ok=True)
print(f"shard {aid}: {len(tasks)} episodes", flush=True)
t0 = time.time()


def run(t):
    d = OUT + "/runs/" + t["id"]
    if os.path.exists(d + "/episode_complete.json"):
        return t["id"], "cached"
    cmd = [PY, "-P", "-u", COLLECT, "--source-root", C + "/source_v1",
           "--case", f"{CASES}/{t['group_id']}.json", "--route", f"{OUT}/routes/{t['id']}.json",
           "--out", d, "--chrono-data", "/work1/dannegrut/harry/nrd/chrono-build/data", "--horizon-s", "120"]
    with open(OUT + "/logs/" + t["id"] + ".log", "w") as lg:
        r = subprocess.run(cmd, stdout=lg, stderr=subprocess.STDOUT, timeout=3600)
    return t["id"], ("ok" if r.returncode == 0 else f"rc={r.returncode}")


with ThreadPoolExecutor(max_workers=int(os.environ.get("F104_WORKERS", "4"))) as ex:
    for tid, st in ex.map(run, tasks):
        if st not in ("ok", "cached"):
            print("  FAIL", tid, st, flush=True)
print(f"shard {aid} done in {time.time()-t0:.0f}s", flush=True)
