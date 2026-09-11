#!/usr/bin/env python3
"""Aggregate CRM tracking results, refusing to merge across eras.

Records written before the provenance commit carry no policy_sha256, so era is inferred
from file mtime: for each arm, only files within MERGE_WINDOW of that arm's NEWEST file
are merged. An arm whose older files are excluded is reported, so a stale era is visible
rather than silently averaged in.
"""
import json, glob, os, re, random, statistics as st, collections, sys, datetime as dt

MERGE_WINDOW = 36 * 3600
D = sys.argv[1] if len(sys.argv) > 1 else "."

def load(p):
    try: d = json.load(open(p))
    except Exception: return {}
    return {e["episode_id"]: e for e in d if isinstance(e, dict) and e.get("mae_vx") is not None}

files = collections.defaultdict(list)
for p in glob.glob(os.path.join(D, "crmtrack_*.json")):
    b = os.path.basename(p)
    m = re.match(r"crmtrack_(?:finetune_crm_)?(.+?)_(hpcfund|kyle-.+?)\.json$", b)
    if not m: continue
    arm = "BASE" if ("go2_cts_150k" in b or b.startswith("crmtrack_BASE")) else m.group(1)
    files[arm].append((os.path.getmtime(p), p))

merged, stale = {}, {}
for arm, fs in files.items():
    fs.sort(reverse=True)
    newest = fs[0][0]
    keep = [p for t, p in fs if newest - t <= MERGE_WINDOW]
    drop = [p for t, p in fs if newest - t > MERGE_WINDOW]
    d = {}
    for p in keep: d.update(load(p))
    merged[arm] = d
    if drop: stale[arm] = (len(keep), drop)

base = merged.pop("BASE", {})
print(f"base episodes: {len(base)}   arms: {len(merged)}   merge window: {MERGE_WINDOW//3600}h")
if stale:
    print("\nolder-era files EXCLUDED (arm name reused after retrain):")
    for arm, (nk, drop) in sorted(stale.items()):
        print(f"  {arm:22s} kept {nk}, dropped {len(drop)}: {', '.join(os.path.basename(x) for x in drop[:2])}")

rng = random.Random(0); B = 8000
rows = []
for n, d in merged.items():
    sh = sorted(set(d) & set(base))
    if len(sh) < 20: continue
    diff = [d[k]["mae_vx"] - base[k]["mae_vx"] for k in sh]
    bv = st.mean(base[k]["mae_vx"] for k in sh); m = st.mean(diff)
    w = sum(x < 0 for x in diff)
    hit = sum(1 for _ in range(B) if abs(sum(x if rng.getrandbits(1) else -x for x in diff)) >= abs(m * len(sh)))
    shas = {d[k].get("policy_sha256") for k in sh}
    tag = "" if len(shas) <= 1 else f"  MIXED-CKPT({len(shas)})"
    rows.append((100 * m / bv, n, len(sh), m, w, (hit + 1) / (B + 1), tag))
rows.sort()
print(f"\n{'arm':24s} {'n':>3} {'delta':>9} {'pct':>8} {'wins':>9} {'p':>7}")
for pct, n, ns, m, w, p, tag in rows:
    print(f"{n:24s} {ns:3d} {m:+9.4f} {pct:+7.1f}% {w:4d}/{ns:<4d}{p:7.4f}{tag}")
