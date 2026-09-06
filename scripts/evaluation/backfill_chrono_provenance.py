"""Establish, and then RECORD, which Chrono build produced each existing dataset.

WHY THIS IS URGENT RATHER THAN TIDY. Twenty datasets record a seed, which makes
their build recoverable: replay one episode under a known build and compare
physics columns. None records the build itself. That recoverability has an
expiry date nobody would think to announce -- the moment `chrono-build` is
rebuilt, replaying gives a different answer and there is no way to tell whether
the difference is the new build or something else. The datasets do not become
wrong, they become permanently ambiguous.

So the fingerprint is not a redundant second field beside the seed. It is a
hedge against build churn, and it costs one episode per dataset to buy.

REUSES THE VERDICT HARNESS'S REPLAY PATH rather than writing a second one. A
separate implementation of "replay one episode and compare" would be a second
thing to keep correct, and the divergence between the two would be invisible
until it mattered.

A FAILURE HERE IS A FINDING, NOT A CHORE. Bit-identical means the dataset was
produced by the build running now. Anything else means it was not, and that is
worth knowing about data underneath published results.

    "$NEDM_PY" scripts/evaluation/backfill_chrono_provenance.py --datasets-root ...
"""
from __future__ import annotations
import argparse, glob, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_go2_finetune_verdict import (episode_spec, run_arm, compare_replay,
                                      BASE_CKPT)


def current_build():
    import pychrono
    so = Path(pychrono.__file__).parent / "_core.so"
    return {"pychrono_path": str(so),
            "core_so_md5": hashlib.md5(so.read_bytes()).hexdigest(),
            "source_build": "chrono-build" in str(so)}


def one_episode(root: Path):
    """A single episode sidecar from this dataset, or None if it has no replayable spec."""
    # GLOB THE WAY THE HARNESS DOES. episode_spec reads the per-episode sidecar under
    # episodes/, not the run-level config; rglob("*.json") also matches
    # collector_config.resolved.json, which carries command_family but no episode_id.
    for j in sorted(root.glob("**/episodes/*.json")):
        # str(), NOT the Path: episode_spec does json_path.replace(".json", ".csv"),
        # which on a Path is Path.replace -- a RENAME, not a string substitution. It
        # raises rather than renaming, and a bare `except: continue` above turns that
        # into a silent skip, which is how this first reported 32/32 unreplayable.
        try:
            spec, why = episode_spec(str(j))
        except Exception:
            continue
        if spec:
            csv = j.parent / f"{j.stem}.csv"
            if not csv.exists():
                hit = glob.glob(str(j.parent / "*.csv"))
                if not hit:
                    continue
                csv = Path(hit[0])
            spec["csv"] = str(csv)
            return spec
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-root", default="/home/kyle/sbel-artifacts/datasets")
    ap.add_argument("--out", default="docs/state/provenance/go2_chrono_build_backfill.json")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    build = current_build()
    print(f"replaying against build {build['core_so_md5'][:8]}  "
          f"{'SOURCE' if build['source_build'] else 'CONDA'}\n")

    root = Path(a.datasets_root)
    names = a.only or sorted(p.name for p in root.iterdir() if p.is_dir())
    results = {}
    for name in names:
        spec = one_episode(root / name)
        if spec is None:
            results[name] = {"status": "no replayable episode",
                             "build": "unknown", "confidence": "unknown"}
            print(f"  {name:28s} SKIP  no replayable episode spec")
            continue
        if a.dry_run:
            print(f"  {name:28s} would replay {Path(spec['csv']).name}")
            continue
        tmp = tempfile.mkdtemp(prefix="chronoprov_")
        try:
            got = run_arm(spec, BASE_CKPT, tmp)
            if not got:
                ok, why = False, "replay produced no CSV"
            else:
                ok, why = compare_replay(spec["csv"], got)
            results[name] = {
                "status": "bit-identical" if ok else f"MISMATCH: {why}",
                "build": build["core_so_md5"] if ok else "ambiguous",
                "confidence": "verified" if ok else "ambiguous",
                "episode": Path(spec["csv"]).name,
                "evidence": ("replayed one episode under the build named in this file, "
                             "with checkpoint " + str(BASE_CKPT) +
                             ", and compared physics columns only"),
                # A MATCH IS CONCLUSIVE; A MISMATCH IS NOT. The episode sidecars record
                # the seed but NOT the checkpoint, so a replay that differs could mean a
                # different build OR a different policy. A match rules out both at once
                # and is therefore the only clean verdict this instrument can return.
                "mismatch_is_ambiguous": not ok,
            }
            print(f"  {name:28s} {'OK  bit-identical' if ok else 'MISMATCH  ' + str(why)}")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    if not a.dry_run:
        out = {"note": ("Backfilled Chrono build provenance. Establishes by replay what the "
                        "datasets never recorded, before a rebuild makes it unrecoverable."),
               "build_replayed_against": build, "datasets": results}
        Path(a.out).write_text(json.dumps(out, indent=1))
        n = sum(1 for v in results.values() if v["confidence"] == "verified")
        print(f"\n{n}/{len(results)} verified against {build['core_so_md5'][:8]} -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
