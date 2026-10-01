"""Fetch items of the traversing release from Hugging Face and restore them into this repo's layout.

    python scripts/traversing/release/download_traversing_data.py --list --all
    python scripts/traversing/release/download_traversing_data.py --milestone m2 --bundle models
    python scripts/traversing/release/download_traversing_data.py --items hmmwv_rigid_f104_pool_designed --verify-members
    python scripts/traversing/release/download_traversing_data.py --all --from-dir /data/trv_staging --dest /tmp/restore

Files land in ``--cache-dir`` (default ``<dest>/artifacts/hf_release/download``, a mirror of the Hub tree) and are
checked against the SHA256 in ``traversing/release_manifest.json`` (a cached file with another hash is downloaded
again with ``force_download``). They are then restored under ``--dest`` (default: this checkout): plain files to
``<restore_root>/<path>``, tar shards extracted in place (every member must be a regular file with a normalised
relative path under the item's restore root, listed in the item's ``traversing/<bundle>/<item>/index.csv.gz`` with the
same SHA256). Every target must resolve inside ``--dest``, is never written through a symlink, and is written via a new
randomly named temporary file next to it (no other existing path is ever removed). Existing files with other content
are left alone unless ``--overwrite`` is given. Two selected items may restore the same path only if each names the other
in ``shared_paths_with`` and the content is identical. Items named in a selected item's ``requires`` are added.

The default revision is ``hf_revision`` (a 40-hex Hub commit) of the pinned GitHub copy
``traversing/manifests/hf_release_manifest.json``. The manifest read from the Hub at that revision, or from
``--from-dir``, must list the same files and items as the pinned copy; with no pinned copy, another ``--revision``
or ``--unpinned`` it is used unchecked, with a warning. ``--from-dir`` reads an already staged or downloaded tree
instead of the Hub. Needs ``huggingface_hub`` only for Hub downloads.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[3]
PINNED = REPO_ROOT / "traversing" / "manifests" / "hf_release_manifest.json"
MANIFEST = "traversing/release_manifest.json"
REPO_ID = "harryzhang1018/NeDM"
COMMIT = re.compile(r"[0-9a-f]{40}")
CHUNK = 1 << 20
UMASK = os.umask(0o022)
os.umask(UMASK)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def safe_rel(name: str) -> PurePosixPath:
    """A normalised relative path without '..'; refuses anything that could land outside the destination."""
    path = PurePosixPath(name)
    if not name or name == "." or "\\" in name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise SystemExit(f"refusing unsafe path {name!r}")
    return path


def read_index(path: Path) -> list[dict]:
    with gzip.open(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def milestones(value) -> set[str]:
    """'M1 navigation, M4a unseen arenas' -> {'m1', 'm4a'}; a bare '2' -> {'m2'}."""
    text = " ".join(value) if isinstance(value, list) else str(value)
    found = set(re.findall(r"\bm\d+[a-z]?\b", text.lower()))
    return found or {f"m{text.strip().lower()}"}


def select(manifest: dict, args) -> list[str]:
    items = manifest["items"]
    unknown = [n for n in args.items or [] if n not in items]
    if unknown:
        raise SystemExit(f"unknown items: {unknown} (see --list --all)")
    names = [n for n in items if not args.items or n in args.items]
    if args.bundle:
        names = [n for n in names if items[n]["bundle"] in args.bundle]
    if args.milestone:
        wanted = {m.lower() if m.lower().startswith("m") else f"m{m.lower()}" for m in args.milestone}
        names = [n for n in names if any(t.startswith(w) for t in milestones(items[n]["milestone"]) for w in wanted)]
    for name in names:  # grows while iterating: required items (and theirs) come along
        for req in items[name].get("requires", []):
            if req not in items:
                raise SystemExit(f"{name} requires the unknown item {req}")
            if req not in names:
                print(f"{req}: added, {name} requires it", file=sys.stderr)
                names.append(req)
    return names


def load_pinned(path: Path) -> dict | None:
    """The pinned GitHub copy of the manifest (None if absent); its hf_revision must be a commit hash."""
    if not path.exists():
        return None
    pinned = json.loads(path.read_text())
    if not COMMIT.fullmatch(str(pinned.get("hf_revision", ""))):
        raise SystemExit(f"{path}: hf_revision must be a 40-hex Hub commit, not {pinned.get('hf_revision')!r}")
    return pinned


def same_release(a: dict, b: dict) -> bool:
    return a["files"] == b["files"] and a["items"] == b["items"]


def check_items(manifest: dict) -> None:
    """Item names, restore roots and file paths are safe relative paths; every item file is in ``files``."""
    for name, item in manifest["items"].items():
        safe_rel(name)
        if item["restore_root"]:
            safe_rel(item["restore_root"])
        prefix = f"traversing/{item['bundle']}/{name}/"
        for path in item["files"]:
            if not path.startswith(prefix) or path not in manifest["files"]:
                raise SystemExit(f"{name}: bad file entry {path!r}")
            safe_rel(path)


def load_manifest(args) -> dict:
    pinned = None if args.unpinned else load_pinned(args.pinned)
    if args.from_dir:
        manifest, source = json.loads((args.from_dir / MANIFEST).read_text()), str(args.from_dir)
    elif args.list and pinned and args.revision == pinned["hf_revision"]:
        manifest, source = pinned, str(args.pinned)
    else:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        path = hf_hub_download(args.repo_id, MANIFEST, repo_type="dataset", revision=args.revision,
                               local_dir=str(args.cache_dir), force_download=True)
        manifest, source = json.loads(Path(path).read_text()), f"{args.repo_id}@{args.revision}"
    if pinned is None:
        why = "--unpinned" if args.unpinned else f"no pinned manifest at {args.pinned}"
        print(f"WARNING: {why}; the manifest of {source} is not checked against the GitHub copy", file=sys.stderr)
    elif not args.from_dir and args.revision != pinned["hf_revision"]:
        print(f"WARNING: revision {args.revision} is not the pinned {pinned['hf_revision']}; its manifest is not "
              "checked against the GitHub copy", file=sys.stderr)
    elif not same_release(manifest, pinned):
        raise SystemExit(f"the manifest of {source} lists other files or items than the pinned {args.pinned}")
    check_items(manifest)
    return manifest


def fetch(hub_path: str, meta: dict, args) -> Path:
    """Local copy of one release file, downloaded if needed, with the manifest size and SHA256."""
    local = (args.from_dir or args.cache_dir) / safe_rel(hub_path)

    def good(p: Path) -> bool:
        return p.is_file() and p.stat().st_size == meta["bytes"] and sha256_file(p) == meta["sha256"]

    if not args.from_dir and not good(local):
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        local = Path(hf_hub_download(args.repo_id, hub_path, repo_type="dataset", revision=args.revision,
                                     local_dir=str(args.cache_dir), force_download=local.exists()))
    if not good(local):
        raise SystemExit(f"{hub_path}: missing or SHA256 differs from the manifest ({local})")
    return local


def place(stream, target: Path, sha: str, nbytes: int, args) -> str:
    """Write a stream to target via a new temporary file next to it (a random name, created exclusively; no other
    path is ever removed); checks size and SHA256. Returns 'wrote' or 'kept'."""
    if not target.parent.resolve().is_relative_to(args.dest):
        raise SystemExit(f"refusing {target}: it resolves outside {args.dest}")
    if target.is_symlink():
        raise SystemExit(f"refusing to write through the symlink {target}")
    if target.exists():
        if target.is_file() and target.stat().st_size == nbytes and sha256_file(target) == sha:
            return "kept"
        if not args.overwrite:
            raise SystemExit(f"{target} exists with other content (use --overwrite to replace it)")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=target.parent, prefix=".nedm-dl-")
    tmp = Path(name)
    try:
        h, size = hashlib.sha256(), 0
        with os.fdopen(fd, "wb") as out:
            while block := stream.read(CHUNK):
                h.update(block)
                size += len(block)
                out.write(block)
        if size != nbytes or h.hexdigest() != sha:
            raise SystemExit(f"{target}: content does not match the release record")
        os.chmod(tmp, 0o666 & ~UMASK)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return "wrote"


def extract_shard(shard: Path, name: str, rows: list[dict], args) -> dict:
    expected = {r["path"]: r for r in rows if r["shard"] == name}
    seen, counts = set(), {"wrote": 0, "kept": 0}
    with tarfile.open(shard, "r|gz") as tar:
        for member in tar:
            if not member.isreg():
                raise SystemExit(f"{shard}: refusing non-regular member {member.name!r} (type {member.type!r})")
            rel = safe_rel(member.name)
            row = expected.get(member.name)
            if row is None or member.name in seen:
                raise SystemExit(f"{shard}: member {member.name!r} is not in the index (or repeated)")
            seen.add(member.name)
            counts[place(tar.extractfile(member), args.dest / rel, row["sha256"], int(row["bytes"]), args)] += 1
    if len(seen) != len(expected):
        raise SystemExit(f"{shard}: {len(expected) - len(seen)} indexed members are missing from the archive")
    return counts


def index_path(manifest: dict, name: str) -> str | None:
    """The item's tar index: exactly traversing/<bundle>/<name>/index.csv.gz with kind 'index' (else None)."""
    path = f"traversing/{manifest['items'][name]['bundle']}/{name}/index.csv.gz"
    return path if path in manifest["items"][name]["files"] and manifest["files"][path]["kind"] == "index" else None


def item_index(manifest: dict, name: str, local: dict[str, Path]) -> list[dict]:
    """Rows of the item's index.csv.gz (if present locally); every path must lie under the item's restore root."""
    path = index_path(manifest, name)
    index = local.get(path) if path else None
    rows = read_index(index) if index else []
    root = manifest["items"][name]["restore_root"]
    for r in rows:
        if root and not str(safe_rel(r["path"])).startswith(f"{root}/"):
            raise SystemExit(f"{index}: {r['path']!r} lies outside the restore root {root}")
    return rows


def restore_targets(manifest: dict, name: str, local: dict[str, Path]) -> list[tuple[PurePosixPath, str, int]]:
    """(repo-relative path, sha256, bytes) of every file an item restores."""
    item, files = manifest["items"][name], manifest["files"]
    prefix = f"traversing/{item['bundle']}/{name}/"
    out = [(safe_rel(r["path"]), r["sha256"], int(r["bytes"])) for r in item_index(manifest, name, local)]
    for hub_path in item["files"]:
        if files[hub_path]["kind"] == "file":
            rel = PurePosixPath(item["restore_root"]) / safe_rel(hub_path[len(prefix):])
            out.append((rel, files[hub_path]["sha256"], files[hub_path]["bytes"]))
    return out


def nested(a: str, b: str) -> bool:
    return a == b or not a or not b or a.startswith(b + "/") or b.startswith(a + "/")


def check_shared(manifest: dict, targets: dict[str, list]) -> int:
    """Restore paths claimed by two items must be declared in shared_paths_with (both ways) with the same content."""
    items, bad = manifest["items"], []
    names = [a for a in targets if any(b != a and nested(items[a]["restore_root"], items[b]["restore_root"])
                                       for b in targets)]
    owner: dict = {}
    for name in names:
        for rel, sha, nbytes in targets[name]:
            first, *record = owner.setdefault(rel, (name, sha, nbytes))
            if first != name and not (name in items[first].get("shared_paths_with", [])
                                      and first in items[name].get("shared_paths_with", [])
                                      and record == [sha, nbytes]):
                bad.append(f"{rel} ({first}, {name})")
    for line in bad[:5]:
        print(f"  COLLISION {line}")
    if bad:
        print(f"{len(bad)} restore paths are claimed by two items without shared_paths_with or with other content")
    return len(bad)


def restore_item(manifest: dict, name: str, local: dict[str, Path], args) -> None:
    item, files = manifest["items"][name], manifest["files"]
    counts = {"wrote": 0, "kept": 0}
    rows = item_index(manifest, name, local)
    for hub_path in item["files"]:
        if files[hub_path]["kind"] == "tar_shard":
            for key, n in extract_shard(local[hub_path], PurePosixPath(hub_path).name, rows, args).items():
                counts[key] += n
    prefix = f"traversing/{item['bundle']}/{name}/"
    for hub_path in item["files"]:
        meta = files[hub_path]
        if meta["kind"] == "file":
            target = args.dest / item["restore_root"] / safe_rel(hub_path[len(prefix):])
            with open(local[hub_path], "rb") as src:
                counts[place(src, target, meta["sha256"], meta["bytes"], args)] += 1
    print(f"{name}: restored under {args.dest / item['restore_root']} ({counts['wrote']} written, "
          f"{counts['kept']} already present)", flush=True)


def verify_members(targets: list, name: str, dest: Path) -> int:
    bad = 0
    for rel, sha, nbytes in targets:
        path = dest / rel
        if not path.is_file() or path.stat().st_size != nbytes or sha256_file(path) != sha:
            print(f"  MISMATCH {path}")
            bad += 1
    print(f"{name}: {'all restored files match' if not bad else f'{bad} restored files do not match'}", flush=True)
    return bad


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", nargs="+", default=None, help="Item names.")
    parser.add_argument("--bundle", nargs="+", default=None, choices=["models", "evaluation", "processed", "raw", "assets"])
    parser.add_argument("--milestone", nargs="+", default=None, help="e.g. m2, m4a (m4 matches m4a and m4b).")
    parser.add_argument("--all", action="store_true", help="Every item (combine with --bundle/--milestone to filter).")
    parser.add_argument("--list", action="store_true", help="Print the selection and exit.")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default=None, help="Hub revision (default: the pinned commit, else main).")
    parser.add_argument("--pinned", type=Path, default=PINNED, help="Pinned manifest (default: the GitHub copy).")
    parser.add_argument("--unpinned", action="store_true", help="Do not compare with the pinned manifest (warns).")
    parser.add_argument("--dest", type=Path, default=REPO_ROOT, help="Repo root to restore into (default: this checkout).")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Download mirror (default: <dest>/artifacts/hf_release/download).")
    parser.add_argument("--from-dir", type=Path, default=None, help="Use an already staged/downloaded tree, no download.")
    parser.add_argument("--no-restore", action="store_true", help="Only download and check the release files.")
    parser.add_argument("--verify-members", action="store_true", help="Re-hash every restored file at the end.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing files whose content differs.")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel downloads / checks.")
    args = parser.parse_args(argv)
    if not (args.items or args.bundle or args.milestone or args.all):
        parser.error("choose --items, --bundle, --milestone or --all")
    if args.verify_members and args.no_restore:
        parser.error("--verify-members checks restored files; it cannot be combined with --no-restore")
    if args.revision is None:
        pinned = None if args.unpinned else load_pinned(args.pinned)
        args.revision = pinned["hf_revision"] if pinned else "main"
    args.dest = args.dest.resolve()
    args.cache_dir = (args.cache_dir or args.dest / "artifacts" / "hf_release" / "download").resolve()
    args.from_dir = args.from_dir.resolve() if args.from_dir else None
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = load_manifest(args)
    names = select(manifest, args)
    files = manifest["files"]
    if args.list:
        total = 0
        for n in names:
            it = manifest["items"][n]
            nbytes = sum(files[p]["bytes"] for p in it["files"])
            total += nbytes
            milestone = ", ".join(it["milestone"]) if isinstance(it["milestone"], list) else it["milestone"]
            print(f"{n:55s} {it['bundle']:10s} {it['packing']:5s} {len(it['files']):6d} files {nbytes:>16,} B  "
                  f"-> {it['restore_root'] or '.'}  [{milestone}]")
        print(f"{len(names)} items, {total:,} B to download (source: {args.from_dir or args.revision})")
        return 0
    local = {}
    with cf.ThreadPoolExecutor(args.jobs) as pool:
        for name in names:
            paths = manifest["items"][name]["files"]
            local[name] = dict(zip(paths, pool.map(lambda p: fetch(p, files[p], args), paths)))
            print(f"{name}: {len(paths)} release files present and checked", flush=True)
    if args.no_restore:
        return 0
    targets = {name: restore_targets(manifest, name, local[name]) for name in names}
    if check_shared(manifest, targets):
        raise SystemExit("refusing to restore: restore paths collide")
    bad = 0
    for name in names:
        restore_item(manifest, name, local[name], args)
        if args.verify_members:
            bad += verify_members(targets[name], name, args.dest)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
