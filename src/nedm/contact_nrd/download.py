"""Download the contact NRD release (training data, test data, trained models) from Hugging Face.

    PYTHONPATH=src python -m nedm.contact_nrd.download --case all --part all
    PYTHONPATH=src python -m nedm.contact_nrd.download --case pool --part test models

Files land in <dest>/contact_nrd/<case>/<part>/ (default dest: artifacts/ of this checkout), from the Hub revision
pinned in contact_nrd/release_manifest.json, and are checked against the size and SHA256 listed there.
Needs huggingface_hub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST = REPO_ROOT / "contact_nrd" / "release_manifest.json"
CASES = ("bouncing_ball", "pool", "so101_push_t")
PARTS = ("train", "test", "models")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", nargs="+", default=["all"], choices=[*CASES, "all"])
    parser.add_argument("--part", nargs="+", default=["all"], choices=[*PARTS, "all"])
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "artifacts")
    args = parser.parse_args()
    if not MANIFEST.is_file():
        raise SystemExit(f"missing {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text())
    revision = manifest.get("hf_revision", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise SystemExit(f"{MANIFEST}: hf_revision must be a 40-character Hub commit, not {revision!r}")
    cases = CASES if "all" in args.case else args.case
    parts = PARTS if "all" in args.part else args.part
    folders = tuple(f"{manifest['prefix']}{c}/{p}/" for c in cases for p in parts)
    files = {name: meta for name, meta in manifest["files"].items() if name.startswith(folders)}
    for name in files:
        if name.startswith("/") or "\\" in name or ".." in name.split("/"):
            raise SystemExit(f"unsafe path in the manifest: {name}")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=manifest["repo_id"], repo_type="dataset", revision=revision,
                      allow_patterns=[f + "*" for f in folders], local_dir=str(args.dest))
    bad = []
    for name, meta in sorted(files.items()):
        path = args.dest / name
        ok = path.is_file() and path.stat().st_size == meta["bytes"] and sha256(path) == meta["sha256"]
        bad += [] if ok else [path]
        print(f"{'ok ' if ok else 'BAD'} {path}", flush=True)
    if bad:
        raise SystemExit(f"{len(bad)} file(s) missing or with another size or SHA256: delete them and run again")


if __name__ == "__main__":
    main()
