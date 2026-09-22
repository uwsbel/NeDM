#!/usr/bin/env python3
"""Build the rigid A4 (branch_auto) and B3 (pid_perturbed) task files and the CRM B3 twin (2026-09-21).

Runs locally from the repo root; writes only under C_collectors/launch/tasks/.  Cluster roots:
  R = /work1/dannegrut/harry/experiments/fdm_f104_50h_20260909   (rigid recordings, cases_night2_v1, cases_night2_onpolicy_v1)
  C = /work1/dannegrut/harry/experiments/crm_f104_20260916        (CRM_ROOT; cases/night2, cases/night2/routes)
Rigid rows (gen_runner_g.py): {id, group, case, route, run, tier, arena, shard, mode, extra, n_cont}; case/route absolute.
CRM rows (crm_worker.py):     {id, group, case, route, run, tier, episode_seed, extra}; case/route relative to CRM_ROOT.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent / "tasks"
R = "/work1/dannegrut/harry/experiments/fdm_f104_50h_20260909"
C = "/work1/dannegrut/harry/experiments/crm_f104_20260916"
LOCAL_R = "artifacts/traverse/fdm_f104_50h_20260909"
ANCHORS = REPO / "artifacts/traverse/generalist_20260921/A_adapt/a4/anchors/anchors_rigid.json"
TWIN = REPO / "artifacts/traverse/crm_night2_v1/datasets/twin_crm.npz"
SUITE_PREFIXES = ("f104_crm_eval_group_", "f104_g1_test_group_", "f104_pair_group_")
N_B3, N_ROUTES, N_SHARDS, SAMPLE_SEED = 1500, 12, 6, 20260921


def md5int(s):                     # 32-bit seed / shard hash from md5
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def shard_of(group):
    return int(hashlib.md5(group.encode()).hexdigest(), 16) % N_SHARDS


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def not_suite(name):
    assert not name.startswith(SUITE_PREFIXES), f"planner-suite id/group in a builder: {name}"


# ---------------------------------------------------------------- split (the twin group split is the only split)
z = np.load(TWIN, allow_pickle=True)
split_of = {}
for g, s in zip(z["group"].tolist(), z["split"].tolist()):
    assert split_of.setdefault(g, s) == s, g
train_groups = sorted(g for g, s in split_of.items() if s == "train")
held_out = {g for g, s in split_of.items() if s != "train"}
assert len(split_of) == 1200 and len(train_groups) == 1089 and len(held_out) == 111, (len(split_of), len(train_groups))

# ---------------------------------------------------------------- (a) A4 rigid: one branch_auto row per anchor
anchors = json.load(open(ANCHORS))
assert len(anchors) == 800 and len({a["anchor_id"] for a in anchors}) == 800
a4 = []
for i, a in enumerate(anchors):
    g, ep, F = a["group"], a["episode"], int(a["F"])
    not_suite(g); not_suite(ep)
    assert a["world"] == "rigid" and split_of[g] == a["split"], (a["anchor_id"], split_of[g], a["split"])
    assert a["route_file_matches_recording"] and a["route_sha256"] == a["recorded_route_sha256"]
    assert 1 <= F < a["n_frames"] and F * 0.05 < 120.0, (a["anchor_id"], F, a["n_frames"])
    rf = a["route_file"]
    if rf.startswith(f"{LOCAL_R}/cases_night2/cases/routes/"):
        route = R + "/cases_night2_v1/routes/" + rf[len(f"{LOCAL_R}/cases_night2/cases/routes/"):]
        assert a["route_kind"] == "designed"
    elif rf.startswith(f"{LOCAL_R}/cases_night2_onpolicy/routes/"):
        route = R + "/cases_night2_onpolicy_v1/routes/" + rf[len(f"{LOCAL_R}/cases_night2_onpolicy/routes/"):]
        assert a["route_kind"] == "on_policy"
    else:
        raise SystemExit(f"unknown route_file layout {rf}")
    rd = a["run_dir"]
    assert rd.startswith(f"{LOCAL_R}/production_v") and rd.endswith("/runs/" + ep), rd
    recorded = R + rd[len(LOCAL_R):]
    case = f"{R}/cases_night2_v1/{g}.json"
    rec_outcome = json.load(open(REPO / rd / "outcome.json"))
    assert rec_outcome["route_sha256"] == a["route_sha256"], a["anchor_id"]
    local_case = REPO / LOCAL_R / "cases_night2/cases" / f"{g}.json"
    assert sha256(local_case) == rec_outcome["case_sha256"], (a["anchor_id"], "local case != recorded case")
    assert sha256(REPO / rf) == a["route_sha256"]
    seed = md5int(a["anchor_id"])
    a4.append({"id": f"{ep}__a4", "group": g, "case": case, "route": route, "run": True, "tier": i, "arena": "f104",
               "shard": shard_of(g), "mode": "branch_auto", "n_cont": 3,
               "extra": ["--branch-frame", str(F), "--n-cont", "3", "--cont-seed", str(seed), "--recorded", recorded],
               "world": "rigid", "split": a["split"], "cls": a["cls"], "F": F, "anchor_id": a["anchor_id"],
               "episode": ep, "cont_seed": seed, "route_kind": a["route_kind"],
               "case_sha256": rec_outcome["case_sha256"], "route_sha256": a["route_sha256"],
               "recorded_frames": int(a["n_frames"]), "recorded_status": a["status"]})
assert len({r["id"] for r in a4}) == 800 and len({r["cont_seed"] for r in a4}) == 800

# ---------------------------------------------------------------- (b)/(c) B3: 1,500 (train group, designed route) pairs
rng = np.random.default_rng(SAMPLE_SEED)
pairs, seen = [], set()
while len(pairs) < N_B3:
    g = train_groups[int(rng.integers(len(train_groups)))]
    r = int(rng.integers(N_ROUTES))
    if (g, r) not in seen:
        seen.add((g, r)); pairs.append((g, r))
b3_rigid, b3_crm = [], []
for i, (g, r) in enumerate(pairs):
    not_suite(g)
    assert split_of[g] == "train" and g not in held_out
    rid = f"{g}_route_{r:02d}__b3"
    local_case = REPO / LOCAL_R / "cases_night2/cases" / f"{g}.json"
    local_route = REPO / LOCAL_R / "cases_night2/cases/routes" / g / f"route_{r:02d}.json"
    cj = json.load(open(local_case))
    assert cj["id"] == g and cj["split"] == "train" and cj["layout"]["assets"] == [] and cj["arena"] == "assets/traverse/arena_f104_50h_v1"
    seed = md5int(rid)
    csha, rsha = sha256(local_case), sha256(local_route)
    common = {"id": rid, "group": g, "run": True, "tier": i, "split": "train", "route_index": r,
              "case_sha256": csha, "route_sha256": rsha}
    b3_rigid.append({**common, "case": f"{R}/cases_night2_v1/{g}.json", "route": f"{R}/cases_night2_v1/routes/{g}/route_{r:02d}.json",
                     "arena": "f104", "shard": shard_of(g), "mode": "pid_perturbed",
                     "extra": ["--episode-seed", str(seed), "--near-stop-s", "40"], "world": "rigid"})
    b3_crm.append({**common, "case": f"cases/night2/{g}.json", "route": f"cases/night2/routes/{g}/route_{r:02d}.json",
                   "episode_seed": seed, "extra": ["--mode", "pid_perturbed", "--near-stop-s", "40"], "world": "crm"})
seeds = [int(r["extra"][1]) for r in b3_rigid]
assert seeds == [r["episode_seed"] for r in b3_crm] and len(set(seeds)) == N_B3, "seeds must be unique and identical across worlds"
assert len({r["id"] for r in b3_rigid}) == N_B3 and [r["id"] for r in b3_rigid] == [r["id"] for r in b3_crm]
assert not ({r["group"] for r in b3_rigid} & held_out)
assert len(set(seeds) & {r["cont_seed"] for r in a4}) == 0

OUT.mkdir(exist_ok=True)
for name, rows in (("tasks_a4_rigid.json", a4), ("tasks_b3_rigid.json", b3_rigid), ("tasks_b3_crm.json", b3_crm)):
    json.dump(rows, open(OUT / name, "w"), indent=0)
    print(name, len(rows), "sha256", sha256(OUT / name))
from collections import Counter
summary = {"a4": {"rows": len(a4), "split": dict(Counter(r["split"] for r in a4)), "cls": dict(Counter(r["cls"] for r in a4)),
                  "route_kind": dict(Counter(r["route_kind"] for r in a4)), "shard": dict(Counter(r["shard"] for r in a4)),
                  "recorded_root": dict(Counter(r["extra"][-1].split("/")[-3] for r in a4)),
                  "F_min_max": [min(r["F"] for r in a4), max(r["F"] for r in a4)]},
           "b3": {"rows": N_B3, "distinct_groups": len({g for g, _ in pairs}), "distinct_train_groups_available": len(train_groups),
                  "route_index": dict(Counter(r for _, r in pairs)), "shard": dict(Counter(r["shard"] for r in b3_rigid)),
                  "rows_per_group_max": max(Counter(g for g, _ in pairs).values()), "sample_seed": SAMPLE_SEED,
                  "seeds_unique": True, "held_out_groups_present": 0, "suite_ids_present": 0},
           "seed_rule": "int(md5(anchor_id or row id)[:8], 16); shard = int(md5(group), 16) % 6"}
json.dump(summary, open(OUT / "build_summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
