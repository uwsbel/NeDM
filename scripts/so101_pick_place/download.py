"""Download the Hub files of the SO101 pick-and-place release and check the size and SHA-256 of every file.

    python scripts/so101_pick_place/download.py [--groups nrd,teacher,...] [--dest DIR] [--source DIR] [--check-only]

The file list, the sizes and the SHA-256 values come from so101_pick_place/release_manifest.json (repository copy).
Files land in DEST/so101_pick_place/<path> (default DEST = artifacts of this checkout: the code reads its defaults
there, nedm.so101_policy.paths). A file that is already in place with the right size and SHA-256 is not fetched again.

Where the files come from:
- the Hub (harryzhang1018/NeDM, dataset), at the manifest's pinned "hf_revision" (huggingface_hub.snapshot_download);
- or a local folder with the layout of the Hub folder so101_pick_place/ (--source DIR, or the environment variable
  SO101_PP_HF_SOURCE): e.g. a release staging folder or an earlier download. Files are hard-linked when the folder is
  on the same file system, else copied. Every file is checked in the same way.
Without a pinned revision ("PENDING") and without --source, the script only checks the files that are in place.

Groups (folders of so101_pick_place/ on the Hub; sizes from the manifest are printed):
  nrd          nrd/                          the policy NRD c1_s62 and the base NRD c0_s62 (main-compatible best.pt,
                                             the original files in original/, the training records)
  nrd_test     nrd_test/                     the sealed scripted NRD test set (3,050 episodes)
  nrd_data     nrd_data/                     the 4 NRD training data parts (93,494 episodes) + the 7 PPO policies
  teacher      teacher/ without t1_expert_data/   teacher checkpoints T0, T2-T6 and their run records
  expert_data  teacher/t1_expert_data/       the 24,576 NRD expert episodes behind the BC teacher T2
  student      student/                      student checkpoints S0-S5 and their run records
  rl_inputs    rl_inputs/                    task sets, start banks, geometry, the task sets of the sim checks
  results      results/                      sim-check and robot result files, the NRD test results
  bundle       bundle/                       the folder installed on the robot computer, and its tarball
  recordings   robot/lerobot_recordings/     2 LeRobot teleop recordings (replay input, camera calibration input)
  robot        robot/                        robot logs, videos and the recordings
  scene_calib  scene_calib/                  calibration and scene files (the same bytes as configs/so101_pick_place/)
  provenance   provenance/                   code snapshots that are in no commit
  card         README.md                     the dataset card of the folder
  all          every file
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[2]
MANIFEST = CODE / "so101_pick_place" / "release_manifest.json"
GROUPS = {"nrd": ("nrd/",), "nrd_test": ("nrd_test/",), "nrd_data": ("nrd_data/",), "teacher": ("teacher/",),
          "expert_data": ("teacher/t1_expert_data/",), "student": ("student/",), "rl_inputs": ("rl_inputs/",),
          "results": ("results/",), "bundle": ("bundle/",), "recordings": ("robot/lerobot_recordings/",),
          "robot": ("robot/",), "scene_calib": ("scene_calib/",), "provenance": ("provenance/",),
          "card": ("README.md",), "all": ("",)}
EXCLUDE = {"teacher": ("teacher/t1_expert_data/",)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def select(files, prefix, groups):
    out = {}
    for g in groups:
        if g not in GROUPS:
            sys.exit(f"unknown group {g!r} (groups: {', '.join(GROUPS)})")
        for n, m in files.items():
            rel = n[len(prefix):]
            if rel.startswith(GROUPS[g]) and not rel.startswith(EXCLUDE.get(g, ("\0",))):
                out[n] = m
    return out


def place(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--groups", default="nrd,teacher,student,rl_inputs,results,bundle,recordings",
                   help="comma list (default: what reproduce.sh full needs; see above)")
    p.add_argument("--dest", default=str(CODE / "artifacts"), help="files land in DEST/so101_pick_place/")
    p.add_argument("--source", default=os.environ.get("SO101_PP_HF_SOURCE", ""),
                   help="local folder with the layout of the Hub folder so101_pick_place/ (instead of the Hub)")
    p.add_argument("--manifest", default=str(MANIFEST))
    p.add_argument("--check-only", action="store_true", help="only check the files in place")
    p.add_argument("--quiet", action="store_true", help="print only the summary and the bad files")
    a = p.parse_args(argv)
    manifest = json.loads(Path(a.manifest).read_text())
    prefix, dest = manifest["prefix"], Path(a.dest)
    files = select(manifest["files"], prefix, [g for g in a.groups.split(",") if g])
    for n in files:
        if not n.startswith(prefix) or n.startswith("/") or "\\" in n or ".." in n.split("/"):
            sys.exit(f"unsafe path in the manifest: {n}")
    total = sum(m["bytes"] for m in files.values())
    print(f"{len(files)} files, {total / 1e9:.3f} GB (groups {a.groups}) -> {dest / prefix}", flush=True)
    good = lambda n: ((dest / n).is_file() and (dest / n).stat().st_size == files[n]["bytes"]
                      and sha256(dest / n) == files[n]["sha256"])
    ok = {n: good(n) for n in sorted(files)}
    missing = [n for n in ok if not ok[n]]
    rev = manifest.get("hf_revision", "")
    pinned = re.fullmatch(r"[0-9a-f]{40}", rev) is not None
    if missing and not a.check_only:
        if a.source:
            src = Path(a.source)
            print(f"{len(missing)} file(s) from the local folder {src} (hard link or copy)", flush=True)
            for n in missing:
                f = src / n[len(prefix):]
                if f.is_file():
                    place(f, dest / n)
        elif pinned:
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                sys.exit(f"{len(missing)} file(s) must be downloaded, and the download needs huggingface_hub "
                         "(pip install huggingface_hub); or give --source DIR")
            print(f"download {len(missing)} file(s) from {manifest['repo_id']} at revision {rev}", flush=True)
            snapshot_download(repo_id=manifest["repo_id"], repo_type="dataset", revision=rev, allow_patterns=missing,
                              local_dir=str(dest))
        else:
            print(f"hf_revision is {rev!r} (no pinned Hub revision) and no --source: only the check of the files "
                  f"in {dest / prefix}", flush=True)
        ok.update({n: good(n) for n in missing})
    for n in ok:
        if not (a.quiet and ok[n]):
            print(f"{'ok ' if ok[n] else 'BAD'} {dest / n}", flush=True)
    bad = [n for n in ok if not ok[n]]
    print(f"{len(ok) - len(bad)} of {len(ok)} files OK (size and SHA-256)", flush=True)
    if bad:
        sys.exit(f"{len(bad)} file(s) missing or with another size or SHA-256")


if __name__ == "__main__":
    main()
