#!/usr/bin/env python3
"""Merge sharded corpora into one, without losing what each shard recorded.

Collection is parallelised by running independent shards on separate nodes, each writing
its own corpus directory. Sharing one directory would race on `manifest.json`, and a
corrupted manifest costs more than this step saves.

WHAT MERGING MUST NOT DO

  Silently accept shards that are not comparable. Episodes collected against different
  Chrono builds, different soil, different excitation or different command ranges are not
  one corpus however similar the filenames look -- this project has already had a capacity
  arm read -9.5% on one build where another read -39.6% on the same policy. Every shard's
  provenance is compared and a mismatch is refused, not warned about.

  Renumber episodes in a way that collides. Each shard numbers its episodes from zero, so
  a naive copy overwrites. Episode ids are rewritten with the shard baked in, and the
  mapping is recorded so a segment in the merged corpus can be traced back to the job that
  produced it.

  Lose the split. `val_episodes` is per-shard and indexed per-shard; the merged manifest
  carries the renumbered set, or every downstream split would silently become train.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

# Fields that must agree across shards for the merge to mean anything. Anything that
# changes the physics, the excitation or the binary belongs here; anything that merely
# differs per run (host, timing, seed, counts) must not.
MUST_MATCH = ("terrain", "chrono_build", "command_ranges", "excitation", "versions",
              # A shard captured post-step, or by a different policy, is not the
              # same corpus however well its other fields agree.
              "row_capture", "policy_raw_order", "policy")

# Fields inside MUST_MATCH that manifests written before the fix recorded WRONGLY, and
# which therefore cannot be compared across those shards. Narrow and named on purpose:
# excluding a field from a comparison is a claim that its disagreement is not real, and
# that claim has to be specific enough to check.
#
# excitation.push.{enabled, events_per_episode} were written from collect.py's loop
# variable rather than its configured value, so each manifest recorded whichever episode
# came LAST -- push-free if that one was a long run, two pushes otherwise. Two shards run
# from the identical command then disagreed, and the merge correctly refused them. The
# configuration was the same; the record of it was not. The long-episode list, which IS
# recorded correctly, is what says which episodes carried no pushes.
#
# These are NOT rewritten in the shard manifests. Editing a recorded value to make it
# agree is the wrong-at-source repair this project has spent effort undoing; the shard
# keeps what it wrote and the merge says what it ignored and why.
UNRELIABLE_IN_OLD_MANIFESTS = {
    ("excitation", "push", "enabled"),
    ("excitation", "push", "events_per_episode"),
}


def _strip(value, path, drop):
    """A copy of `value` with any path in `drop` removed, for comparison only."""
    if not isinstance(value, dict):
        return value
    out = {}
    for k, v in value.items():
        p = path + (k,)
        if p in drop:
            continue
        out[k] = _strip(v, p, drop)
    return out


def _host_name(h):
    """The manifest records host as {"name", "platform"}; older ones may be a bare string.
    Assumed to be a string the first time, which failed only after the comparability
    check had passed -- a type guess about a field this script had never looked at."""
    if isinstance(h, dict):
        return str(h.get("name", ""))
    return str(h)


def load(shard: Path):
    m = shard / "manifest.json"
    if not m.exists():
        raise SystemExit(f"{shard} has no manifest.json; it is not a finished corpus")
    return json.loads(m.read_text())


def aggregate(parts):
    """Per-episode tallies summed across shards. `parts` is [(manifest, episode_offset,
    limit)], limit being the finished-episode count of a shard that died partway (None for
    a complete one). The merged manifest used to take these from shard 0 alone, so the
    v2 corpus of 23 shards recorded 50 episodes' family counts, 12 long episodes and one
    shard's failures under a header that said 1150 episodes."""
    fam, fails, checks, sidecar = {}, {"episodes": 0, "truncated": 0}, {}, {}
    long_eps, skipped, outputs = [], [], []
    short, wall = 0, 0.0
    for man, off, limit in parts:
        keep = (lambda e: True) if limit is None else (lambda e, n=limit: e < n)
        for k, v in (man.get("family_balance") or {}).items():
            fam[k] = fam.get(k, 0) + int(v)
        f = man.get("failures") or {}
        fails["episodes"] += int(f.get("episodes", 0))
        fails["truncated"] += int(f.get("truncated", 0))
        for k, v in (f.get("by_check") or {}).items():
            checks[k] = checks.get(k, 0) + int(v)
        for k, v in (man.get("sidecar_rows") or {}).items():
            sidecar[k] = sidecar.get(k, 0) + int(v)
        long_eps += [off + e for e in man.get("long_episodes", []) if keep(e)]
        skipped += [off + e if isinstance(e, int) else e
                    for e in man.get("skipped_episodes", [])
                    if not isinstance(e, int) or keep(e)]
        outputs += man.get("outputs", [])
        short += int(man.get("dropped_short_segments", 0))
        wall += float(man.get("wall_clock_s") or 0.0)
    fails["rate"] = round(fails["truncated"] / fails["episodes"], 4) if fails["episodes"] else None
    fails["by_check"] = checks
    return {"family_balance": fam, "long_episodes": sorted(long_eps),
            "failures": fails, "sidecar_rows": sidecar, "skipped_episodes": skipped,
            "dropped_short_segments": short, "outputs": outputs,
            # Summed node time, not elapsed time: the shards ran concurrently.
            "wall_clock_s": round(wall, 1), "wall_clock_is": "sum over shards"}


def check_comparable(mans):
    first_name, first = mans[0]
    for name, m in mans[1:]:
        for k in MUST_MATCH:
            a = _strip(first.get(k), (k,), UNRELIABLE_IN_OLD_MANIFESTS)
            b = _strip(m.get(k), (k,), UNRELIABLE_IN_OLD_MANIFESTS)
            if a != b:
                raise SystemExit(
                    f"shards {first_name} and {name} disagree on {k!r}, so they are not "
                    f"one corpus:\n  {first_name}: {json.dumps(a)[:200]}\n"
                    f"  {name}: {json.dumps(b)[:200]}\n"
                    f"Merging them would produce a corpus whose episodes were collected "
                    f"under different conditions, which is exactly the defect the "
                    f"per-corpus manifest exists to prevent.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--train-only", nargs="*", default=[],
                    help="shard directory names whose episodes all become TRAINING data. "
                         "For extending a corpus: the held-out set stays exactly what it "
                         "was, so a model trained on the larger corpus is selected and "
                         "scored against the same episodes as one trained on the smaller, "
                         "and the only thing that changed is how much training data there "
                         "is. Without this, more data also means a different validation "
                         "set, and the two effects cannot be separated.")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="accept shards whose manifest says complete=false (a run that "
                         "died partway); only their finished episodes are merged, and the "
                         "merged manifest lists them")
    a = ap.parse_args()

    shards = [Path(s) for s in a.shards]
    for s in shards:
        if not s.is_dir():
            raise SystemExit(f"{s} is not a directory")
    mans = [(s.name, load(s)) for s in shards]
    check_comparable(mans)
    # A shard still being collected, or one whose run died, writes complete=false. Merging
    # it by accident would take a partial corpus for a whole one -- and one still running
    # would keep adding files under the merge. Manifests from before the field existed were
    # written only at the end of a finished run, so a missing field means complete.
    incomplete = [(n, m.get("episodes_done"), m.get("episodes")) for n, m in mans
                  if m.get("complete") is False]
    if incomplete and not a.allow_incomplete:
        raise SystemExit(
            "incomplete shard(s): " + ", ".join(f"{n} ({d}/{e} episodes)" for n, d, e in
                                                 incomplete)
            + ". Either the run is still going or it died partway. Pass --allow-incomplete "
              "to merge only their finished episodes; the merged manifest will say so.")

    out = Path(a.out) / a.name
    if out.exists():
        raise SystemExit(f"{out} already exists; refusing to merge into it")
    for sub in ("episodes", "failures", "pushes"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    ep_off = 0
    val_eps, provenance, rows, segs, shard_parts = [], [], 0, 0, []
    for (sname, man), sdir in zip(mans, shards, strict=True):
        n_ep = int(man.get("episodes", 0))
        # For a shard that died partway, only episodes the manifest counts as finished. A
        # crash between writing an episode's files and updating the manifest leaves files
        # the record does not vouch for.
        limit = int(man["episodes_done"]) if man.get("complete") is False else None
        local_val = set() if sname in set(a.train_only) else \
            set(man.get("split", {}).get("val_episodes", []))
        if limit is not None:
            local_val = {v for v in local_val if v < limit}
        val_eps += [ep_off + v for v in sorted(local_val)]
        # NOT `parts`: the filename split below binds that name, and shadowing it here
        # silently emptied the aggregation input.
        shard_parts.append((man, ep_off, limit))
        moved = 0
        for sub in ("episodes", "failures", "pushes"):
            src = sdir / sub
            if not src.is_dir():
                continue
            for f in sorted(src.glob("*.csv")):
                # <corpus>_<ep:04d>[_s<k>].csv -> renumber the episode, keep the segment
                parts = f.stem.split("_")
                try:
                    ep_field = next(i for i, x in enumerate(parts)
                                    if x.isdigit() and len(x) == 4)
                except StopIteration:
                    raise SystemExit(f"cannot find an episode number in {f.name!r}")
                if limit is not None and int(parts[ep_field]) >= limit:
                    continue
                new_ep = ep_off + int(parts[ep_field])
                parts[ep_field] = f"{new_ep:04d}"
                parts[:ep_field] = [a.name]
                shutil.copy2(f, out / sub / ("_".join(parts) + f.suffix))
                moved += 1
        provenance.append({"shard": sname, "episode_offset": ep_off,
                           "episodes": n_ep, "files": moved,
                           "run_id": man.get("run_id"), "host": man.get("host"),
                           "seed": man.get("command", {}).get("seed")
                           if isinstance(man.get("command"), dict) else None,
                           "wall_clock_s": man.get("wall_clock_s"),
                           "complete": man.get("complete", True),
                           "episodes_done": man.get("episodes_done", n_ep)})
        rows += int(man.get("rows", 0))
        segs += int(man.get("segments", 0))
        ep_off += n_ep

    base = dict(mans[0][1])
    base.update(aggregate(shard_parts))
    base.update({
        "corpus": a.name,
        # The episode INDEX space, which partial shards leave gaps in; episodes_done is
        # how many actually finished, and is the number to quote.
        "episodes": ep_off,
        "episodes_done": sum(int(p["episodes_done"]) for p in provenance),
        "segments": segs,
        "rows": rows,
        "split": {"val_episodes": sorted(val_eps),
                  "val_fraction": mans[0][1].get("split", {}).get("val_fraction")},
        "merged_from": provenance,
        "train_only_shards": sorted(set(a.train_only)),
        "merge_ignored_fields": sorted(".".join(p) for p in UNRELIABLE_IN_OLD_MANIFESTS),
        "merge_ignored_reason": ("recorded from collect.py's per-episode loop variable "
                                 "before the fix, so each shard stored its last "
                                 "episode's push setting rather than its configuration"),
        # The per-shard hosts and run ids are kept above rather than collapsed, because a
        # merged corpus that cannot say which node produced a given episode cannot be
        # audited when one node turns out to have been wrong.
        "host": sorted({_host_name(p["host"]) for p in provenance if p.get("host")}),
        "run_id": None,
        "complete": True,
        "incomplete_shards": [{"shard": n, "episodes_done": d, "episodes": e}
                              for n, d, e in incomplete],
    })
    (out / "manifest.json").write_text(json.dumps(base, indent=2))

    print(f"merged {len(shards)} shards -> {out}")
    done = sum(int(p["episodes_done"]) for p in provenance)
    print(f"  {done} episodes finished (index space {ep_off}), {segs} segments, "
          f"{rows:,} rows")
    print(f"  val episodes: {len(val_eps)} of {ep_off} "
          f"({100 * len(val_eps) / max(ep_off, 1):.0f}%)")
    if a.train_only:
        print(f"  training-only shards: {len(set(a.train_only))} "
              f"(their episodes contribute no validation data, so the held-out set is "
              f"unchanged from the corpus being extended)")
    actual = len(list((out / 'episodes').glob('*.csv')))
    print(f"  segment files on disk: {actual}")
    if actual != segs:
        print(f"  WARNING: manifest says {segs} segments, {actual} files present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
