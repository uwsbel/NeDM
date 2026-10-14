"""Fetch the HOMIE walking teacher (OpenHomie) at the pinned commit, check the pinned files and write its weights.

HOMIE (https://github.com/InternRobotics/OpenHomie, under its own license) is neither in this repository nor on the
Hugging Face Hub. The footstep study needs it in three places: deploy.onnx (the teacher inside the NRD and in
MuJoCo), g1.xml and its 49 meshes (the MuJoCo scene), and homie_weights.npz (the ONNX initializers as numpy arrays,
which the PPO runs and the MuJoCo evaluation load).

    PYTHONPATH=src python scripts/g1_footstep_rl/fetch_homie.py [--sources configs/g1_footstep_rl/homie_sources.json]

| Step | What |
|---|---|
| 1 | git fetch (depth 1) of source.revision from source.url into source.local_path (default artifacts/g1_footstep_rl/vendor/OpenHomie); an existing checkout must be that revision and clean |
| 2 | Size and SHA-256 of every file in files (71 files: README, licenses, deploy.onnx, g1.xml, the meshes, the deploy and training code the teacher contract was read from) |
| 3 | artifacts/g1_footstep_rl/homie/homie_weights.npz = np.savez(**nedm.g1_footstep_rl.homie.load_weights(deploy.onnx)); its SHA-256 must equal WEIGHTS_SHA256, the value both released PPO runs record in manifest.json |

If step 3 gives another SHA-256, the script compares the arrays with the ONNX initializers and prints both hashes: a
numpy other than 2.4 can write the same arrays as other bytes. The run manifests pin the bytes, so it then stops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCES = ROOT / "configs/g1_footstep_rl/homie_sources.json"
WEIGHTS = ROOT / "artifacts/g1_footstep_rl/homie/homie_weights.npz"
WEIGHTS_SHA256 = "9a4ade34676a519bd831a055e3afd43cf82ebbdca38474a7b87bf2ab1d3f9f41"


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(manifest: dict) -> Path:
    """Steps 1 and 2: the pinned checkout and its file checks. Returns the checkout folder."""
    source = manifest["source"]
    target = ROOT / source["local_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix="homie-fetch-", dir=target.parent) as tmp:
            stage = Path(tmp) / "OpenHomie"
            stage.mkdir()
            git("init", "--quiet", cwd=stage)
            git("remote", "add", "origin", source["url"], cwd=stage)
            git("fetch", "--quiet", "--depth", "1", "origin", source["revision"], cwd=stage)
            git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=stage)
            stage.rename(target)
    if Path(git("rev-parse", "--show-toplevel", cwd=target)).resolve() != target.resolve():
        raise SystemExit(f"expected a standalone OpenHomie checkout: {target}")
    if git("rev-parse", "HEAD", cwd=target) != source["revision"]:
        raise SystemExit(f"{target}: another revision is checked out; remove the folder or use another local_path")
    if git("status", "--porcelain", cwd=target):
        raise SystemExit(f"{target}: the checkout has changes; remove the folder or use another local_path")
    for item in manifest["files"]:
        path = target / item["path"]
        if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise SystemExit(f"source check failed: {path}")
    print(f"OpenHomie {source['revision']}: {len(manifest['files'])} pinned files checked in {target}")
    return target


def write_weights(onnx_path: Path, onnx_sha256: str) -> None:
    """Step 3: homie_weights.npz from the ONNX initializers, checked against the released runs."""
    import numpy as np
    from nedm.g1_footstep_rl import homie
    weights = homie.load_weights(onnx_path, sha256=onnx_sha256)
    WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    np.savez(WEIGHTS, **weights)
    digest = sha256(WEIGHTS)
    if digest == WEIGHTS_SHA256:
        print(f"wrote {WEIGHTS} (sha256 {digest}, as in the released run manifests)")
        return
    with np.load(WEIGHTS) as f:
        same = sorted(f.files) == sorted(weights) and all(np.array_equal(f[k], weights[k]) for k in weights)
    print(f"{WEIGHTS}: sha256 {digest}, expected {WEIGHTS_SHA256}; arrays "
          f"{'equal to' if same else 'DIFFERENT from'} the ONNX initializers (numpy {np.__version__})", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=SOURCES, help="pinned sources (url, revision, files)")
    ap.add_argument("--no-weights", action="store_true", help="steps 1 and 2 only (no torch needed)")
    a = ap.parse_args()
    manifest = json.loads(a.sources.read_text())
    target = fetch(manifest)
    if not a.no_weights:
        policy = manifest["policy"]["path"]
        onnx_sha = next(f["sha256"] for f in manifest["files"] if f["path"] == policy)
        write_weights(target / policy, onnx_sha)


if __name__ == "__main__":
    main()
