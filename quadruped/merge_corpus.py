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
MUST_MATCH = ("terrain", "chrono_build", "command_ranges", "excitation", "versions")

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


def load(shard: Path):
    m = shard / "manifest.json"
    if not m.exists():
        raise SystemExit(f"{shard} has no manifest.json; it is not a finished corpus")
    return json.loads(m.read_text())


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
    a = ap.parse_args()

    shards = [Path(s) for s in a.shards]
    for s in shards:
        if not s.is_dir():
            raise SystemExit(f"{s} is not a directory")
    mans = [(s.name, load(s)) for s in shards]
    check_comparable(mans)

    out = Path(a.out) / a.name
    if out.exists():
        raise SystemExit(f"{out} already exists; refusing to merge into it")
    for sub in ("episodes", "failures", "pushes"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    ep_off = 0
    val_eps, provenance, rows, segs = [], [], 0, 0
    for (sname, man), sdir in zip(mans, shards, strict=True):
        n_ep = int(man.get("episodes", 0))
        local_val = set(man.get("split", {}).get("val_episodes", []))
        val_eps += [ep_off + v for v in sorted(local_val)]
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
                           "wall_clock_s": man.get("wall_clock_s")})
        rows += int(man.get("rows", 0))
        segs += int(man.get("segments", 0))
        ep_off += n_ep

    base = dict(mans[0][1])
    base.update({
        "corpus": a.name,
        "episodes": ep_off,
        "segments": segs,
        "rows": rows,
        "split": {"val_episodes": sorted(val_eps),
                  "val_fraction": mans[0][1].get("split", {}).get("val_fraction")},
        "merged_from": provenance,
        "merge_ignored_fields": sorted(".".join(p) for p in UNRELIABLE_IN_OLD_MANIFESTS),
        "merge_ignored_reason": ("recorded from collect.py's per-episode loop variable "
                                 "before the fix, so each shard stored its last "
                                 "episode's push setting rather than its configuration"),
        # The per-shard hosts and run ids are kept above rather than collapsed, because a
        # merged corpus that cannot say which node produced a given episode cannot be
        # audited when one node turns out to have been wrong.
        "host": sorted({p["host"] for p in provenance if p.get("host")}),
        "run_id": None,
    })
    (out / "manifest.json").write_text(json.dumps(base, indent=2))

    print(f"merged {len(shards)} shards -> {out}")
    print(f"  {ep_off} episodes, {segs} segments, {rows:,} rows")
    print(f"  val episodes: {len(val_eps)} of {ep_off} "
          f"({100 * len(val_eps) / max(ep_off, 1):.0f}%)")
    actual = len(list((out / 'episodes').glob('*.csv')))
    print(f"  segment files on disk: {actual}")
    if actual != segs:
        print(f"  WARNING: manifest says {segs} segments, {actual} files present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
