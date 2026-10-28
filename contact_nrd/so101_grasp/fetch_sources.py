"""Fetch the pinned MuJoCo Menagerie SO101 model (sparse, shallow) and verify every file against the manifest.

    python -I contact_nrd/so101_grasp/fetch_sources.py

The checkout goes to vendor/ (gitignored) and is never edited: the collector builds its scene around the
unchanged so101.xml. The fetched folder holds only data (MJCF, STL meshes, text); nothing in it is executed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parent


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main() -> None:
    manifest = json.loads((ROOT / "source_manifest.json").read_text())
    source = manifest["source"]
    target = ROOT / source["local_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix="menagerie-fetch-", dir=target.parent) as tmp:
            stage = Path(tmp) / "mujoco_menagerie"
            stage.mkdir()
            git("init", "--quiet", cwd=stage)
            git("remote", "add", "origin", source["url"], cwd=stage)
            git("sparse-checkout", "set", "--no-cone", *source["sparse_paths"], cwd=stage)
            git("fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", source["revision"], cwd=stage)
            git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=stage)
            stage.rename(target)
    if Path(git("rev-parse", "--show-toplevel", cwd=target)).resolve() != target.resolve():
        raise SystemExit(f"Expected a standalone upstream checkout: {target}")
    if git("rev-parse", "HEAD", cwd=target) != source["revision"]:
        raise SystemExit("Existing checkout has another revision; leave it intact and use a separate folder.")
    if git("status", "--porcelain", cwd=target):
        raise SystemExit("Existing upstream checkout has changes; leave it intact and use a separate folder.")
    present = sorted(str(p.relative_to(target)) for p in target.rglob("*") if p.is_file() and ".git" not in p.parts)
    expected = sorted(item["path"] for item in manifest["files"])
    if present != expected:
        raise SystemExit(f"Checkout files differ from the manifest: {sorted(set(present) ^ set(expected))}")
    for item in manifest["files"]:
        path = target / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise SystemExit(f"Source verification failed: {path}")
    if any(p.endswith(".py") for p in present):
        raise SystemExit("The fetched model holds Python files; it should hold data only.")
    print(f"Verified mujoco_menagerie {source['revision']} and {len(manifest['files'])} pinned files.")
    print(f"Model: {target / manifest['model']['path']}")


if __name__ == "__main__":
    main()
