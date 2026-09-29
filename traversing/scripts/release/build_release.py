"""Pack the traversing data and model release into a staging tree that mirrors the Hub repo.

    python traversing/scripts/release/build_release.py build --spec traversing/manifests/release_spec.json \
        --source-repo ~/NeDM-traverse_mppi --items m2_shared_history_model --staging /data/trv_staging --jobs 8
    python3.12 build_release.py build --spec manifests/release_spec.json \
        --items gator_f104_soil_collection_runs --staging $WORK/trv_staging --jobs 16   # on the cluster
    python traversing/scripts/release/build_release.py merge --staging /data/trv_staging \
        --spec traversing/manifests/release_spec.json
    python traversing/scripts/release/build_release.py pin --staging /data/trv_staging --revision <40-hex Hub commit>

``build`` resolves each item of the spec (``traversing/manifests/release_spec.json``) from its source root and base
directory: ``include`` globs (``*`` within one folder, ``**`` any number of folders), a ``file_list`` of paths, or an
``id_list`` of run folders with the ``per_run_files`` globs inside each (``per_run_required`` must exist). A pattern
that selects nothing, a missing listed file, a symlink or an unreadable folder in the selection stops the build.
``exclude`` fnmatch patterns drop members and ``known_sha256`` values must match. Id and file lists (``.txt`` or
``.txt.gz``, one normalised relative path per line) are read relative to ``--id-list-root`` (default: the spec's
``id_list_root`` relative to the spec's folder, else the spec's folder). It writes
``<staging>/traversing/<bundle>/<item>/``:

* packing ``files``: every source file as is, at its path under the base directory;
* packing ``tar``: deterministic ``part-NNNNN.tar.gz`` shards (about ``--shard-bytes`` of member bytes each, run
  folders never split, arcname = restore path, mtime/uid/gid 0, PAX, gzip level 6 with mtime 0 and no name; the
  manifest records the SHA256 of both the gzip file and the tar stream inside it), ``index.csv.gz`` (path, bytes,
  sha256, shard), ``ids.txt.gz`` for id-list items and ``episodes.csv.gz`` for items with ``"episodes": true``;

plus ``<staging>/_fragments/<item>.json`` and ``<item>.paths.csv.gz`` (restore path, bytes, sha256 of every member).
A fragment records ``spec_entry``: the sha256 of the canonical JSON of the item's spec entry, of its id/file list and
of this script. Items are packed under ``<staging>/_work/<item>/`` (copies pass through new random names in
``_work/_tmp/``) and replace their previous output only when complete; on an error the queued work is cancelled.
``merge --spec`` needs exactly one fragment per spec item, each built from the current spec entry, list and builder;
it assembles ``traversing/release_manifest.json`` and the ``traversing/README.md`` index from the fragments (carrying
the spec's ``path_remap``, how paths inside released task files map to restore paths) and refuses a restore path
claimed by two items unless each names the other in ``shared_paths_with`` and the content is identical. ``pin``
(needs ``huggingface_hub``) checks that the Hub's ``release_manifest.json`` at a commit equals the staged one and
writes ``traversing/manifests/hf_release_manifest.json`` (the manifest plus ``hf_revision``; a ``--repo-id`` other than
the manifest's, such as a scratch test repo, needs an explicit ``--out``). ``--dry-run`` only resolves and counts
(writes nothing). Root ``repo`` is ``--source-repo`` (the experiment checkout); other roots are the spec's cluster
directories, or their local copies for ``"machine": "local"``; ``--root KEY=PATH`` overrides one. ``build`` and
``merge`` use the standard library only.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import fnmatch
import glob
import gzip
import hashlib
import io
import json
import os
import re
import secrets
import shutil
import socket
import stat
import sys
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

SCHEMA = "nedm-traversing-release/1"
PREFIX = "traversing/"
BUNDLES = ("models", "evaluation", "processed", "raw", "assets")
INDEX_COLUMNS = ["path", "bytes", "sha256", "shard"]
EPISODE_COLUMNS = ["run_id", "collection", "vehicle", "world", "arena", "status", "elapsed_s", "n_files", "bytes",
                   "trajectory_sha256", "outcome_sha256", "shard"]
PINNED = Path(__file__).resolve().parents[2] / "manifests" / "hf_release_manifest.json"
COMMIT = re.compile(r"[0-9a-f]{40}")
ITEM_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # so the builder's own _work/_tmp folder is never an item
UMASK = os.umask(0o022)
os.umask(UMASK)
PRECHECK_BYTES = 64 << 20  # known_sha256 members up to this size are hashed before any packing starts
CHUNK = 1 << 20


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def canonical_sha256(obj) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def gzip_bytes(data: bytes) -> bytes:
    """Deterministic gzip: level 6, mtime 0, no file name (GzipFile: the same header on every Python version)."""
    buf = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buf, compresslevel=6, mtime=0) as gz:
        gz.write(data)
    return buf.getvalue()


def csv_gz(columns: list[str], rows: list[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return gzip_bytes(buf.getvalue().encode())


def read_csv_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def safe_rel(name: str, what: str = "path") -> str:
    """A normalised relative POSIX path (no absolute, '..', '.', empty parts, backslash or outer spaces)."""
    if (not name or name == "." or "\\" in name or name != name.strip() or PurePosixPath(name).is_absolute()
            or PurePosixPath(name).as_posix() != name or ".." in PurePosixPath(name).parts):
        raise SystemExit(f"unsafe or non-normalised {what} {name!r}")
    return name


class HashReader:
    """File reader that hashes what tarfile copies out of it."""

    def __init__(self, f):
        self.f = f
        self.h = hashlib.sha256()

    def read(self, n: int = -1) -> bytes:
        block = self.f.read(n)
        self.h.update(block)
        return block


class HashWriter:
    """Pass-through writer that counts and hashes what goes through it (tar stream, or gzip output)."""

    def __init__(self, out):
        self.out = out
        self.h = hashlib.sha256()
        self.n = 0

    def write(self, block) -> int:
        self.h.update(block)
        self.n += len(block)
        return self.out.write(block)

    def tell(self) -> int:
        return self.n

    def flush(self) -> None:
        self.out.flush()


# ---------------------------------------------------------------- resolving sources


def root_map(spec: dict) -> dict:
    """Cluster and local path of every named source root (spec key ``roots``, or ``root_map``)."""
    roots = spec.get("roots") or spec.get("root_map") or {}
    return {k: {"cluster": v["cluster"], "local": v["local"]} for k, v in roots.items() if k != "repo"}


def item_paths(spec: dict, item: dict, source_repo: Path, overrides: dict) -> tuple[Path, str]:
    """(absolute source base directory, repo-relative restore root) of one item."""
    src = item["source"]
    root, base = src.get("root", "repo"), src.get("base", "") or "."
    if root == "repo":
        top, local = source_repo, "."
    else:
        entry = root_map(spec)[root]
        local = entry["local"]
        top = source_repo / local if src["machine"] == "local" else Path(entry["cluster"])
    top = Path(overrides.get(root, top))
    restore = PurePosixPath(item.get("restore_root") or PurePosixPath(local, base)).as_posix()
    return top / base, "" if restore == "." else safe_rel(restore, f"{item['name']} restore_root")


def no_links(base: Path, parts) -> None:
    """Refuse a symlinked folder or file on the way from base to base/parts."""
    for k in range(1, len(parts) + 1):
        if os.path.islink(base.joinpath(*parts[:k])):
            raise SystemExit(f"symlink in the selection (symlinks are not released): {base.joinpath(*parts[:k])}")


def lstat_regular(path: Path) -> os.stat_result:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        raise SystemExit(f"selected file is missing: {path}") from None
    if stat.S_ISLNK(st.st_mode):
        raise SystemExit(f"symlink in the selection (symlinks are not released): {path}")
    if not stat.S_ISREG(st.st_mode):
        raise SystemExit(f"not a regular file: {path}")
    return st


def match(pat: tuple, parts: tuple, prefix: bool = False) -> bool:
    """Glob match of path parts ('*' within one name, '**' any number of folders). prefix: could a path below match?"""
    if not pat:
        return not parts and not prefix
    if pat[0] == "**":
        return prefix or any(match(pat[1:], parts[i:]) for i in range(len(parts) + 1))
    if not parts:
        return prefix
    return fnmatch.fnmatchcase(parts[0], pat[0]) and match(pat[1:], parts[1:], prefix)


def glob_files(base: Path, pattern: str) -> list[str]:
    """Regular files under base matching a pattern, sorted. A literal pattern gives its file, or nothing if absent.
    Fails on a symlink the pattern reaches (dangling or not, file or folder) and on an unreadable folder."""
    parts = PurePosixPath(safe_rel(pattern, "pattern")).parts
    k = next((i for i, p in enumerate(parts) if glob.has_magic(p)), len(parts))
    no_links(base, parts[:k])
    if k == len(parts):
        return [pattern] if os.path.lexists(base / pattern) and lstat_regular(base / pattern) else []
    top, rest, found = base.joinpath(*parts[:k]), parts[k:], []

    def fail(exc: OSError):
        raise SystemExit(f"cannot read {exc.filename}: {exc.strerror}")

    if not top.is_dir():
        return []
    for folder, dirs, files in os.walk(top, onerror=fail):
        here = PurePosixPath(os.path.relpath(folder, top)).parts if folder != str(top) else ()
        for name in files:
            if match(rest, here + (name,)):
                lstat_regular(Path(folder, name))
                found.append("/".join(parts[:k] + here + (name,)))
            elif os.path.islink(os.path.join(folder, name)) and match(rest, here + (name,), prefix=True):
                raise SystemExit(f"symlink in the selection (symlinks are not released): {folder}/{name}")
        for name in dirs:
            if os.path.islink(os.path.join(folder, name)) and match(rest, here + (name,), prefix=True):
                raise SystemExit(f"symlinked folder in the selection (not followed): {folder}/{name}")
        dirs[:] = [d for d in dirs if match(rest, here + (d,), prefix=True)]
    return sorted(found)


def stat_member(base: Path, rel: str, group: str) -> dict:
    no_links(base, PurePosixPath(rel).parts[:-1])
    src = base / rel
    st = lstat_regular(src)
    if st.st_size <= 200:
        with open(src, "rb") as f:
            if f.read(40).startswith(b"version https://git-lfs"):
                raise SystemExit(f"Git LFS pointer instead of content: {src}")
    return {"rel": rel, "src": str(src), "bytes": st.st_size, "group": group}


def read_json(path: Path) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def episode_fields(folder: Path, run_path: PurePosixPath, defaults: dict) -> dict:
    """Episode fields from the drive's outcome.json and case.json, else the item's episode_defaults, else empty."""
    out, case = read_json(folder / "outcome.json"), read_json(folder / "case.json")
    veh = out.get("vehicle") if isinstance(out.get("vehicle"), dict) else {}
    parent = run_path.parent
    terrain = out.get("terrain")  # soil collectors record terrain "crm"
    world = veh.get("world") or out.get("world") or ("soil" if terrain == "crm" else terrain)
    fields = {"run_id": run_path.name, "collection": (parent.parent if parent.name == "runs" else parent).name,
              "vehicle": veh.get("name"), "world": world, "arena": case.get("arena"), "status": out.get("status"),
              "elapsed_s": out.get("elapsed_s")}
    return {k: v if v not in (None, "") else defaults.get(k, "") for k, v in fields.items()}


def resolve_run(base: Path, run: str, item: dict) -> tuple[list[dict], dict, Counter]:
    """Members of one run folder (per_run_files globs; per_run_required must exist), its episode fields, hit counts."""
    no_links(base, PurePosixPath(run).parts)
    folder = base / run
    if not folder.is_dir():
        raise SystemExit(f"{item['name']}: run folder {folder} not found")
    required = item.get("per_run_required", [])
    hits, names = Counter(), set()
    for pattern in dict.fromkeys(item.get("per_run_files", ["**/*"]) + required):
        found = glob_files(folder, pattern)
        hits[pattern] += len(found)
        names.update(found)
    missing = [p for p in required if not hits[p]]
    if missing:
        raise SystemExit(f"{item['name']}: run {run} lacks {missing}")
    members = [stat_member(base, f"{run}/{n}", run) for n in sorted(names)]
    run_path = PurePosixPath(item["source"].get("base", "") or ".", run)
    episode = episode_fields(folder, run_path, item.get("episode_defaults", {})) if item.get("episodes") else {}
    return members, {"run": run, **episode}, hits


def read_list(path: Path, runs: bool) -> list[str]:
    """Lines of an id or file list (.txt or .txt.gz): normalised relative paths, no duplicates, no nested runs."""
    with (gzip.open(path, "rt") if path.suffix == ".gz" else open(path)) as f:
        lines = f.read().splitlines()
    for line in lines:
        safe_rel(line, f"line in {path}")
    if len(set(lines)) != len(lines):
        raise SystemExit(f"duplicate lines in {path}")
    listed = set(lines)
    nested = [x for x in lines if any(p.as_posix() in listed for p in PurePosixPath(x).parents)] if runs else []
    if nested:
        raise SystemExit(f"run folders inside other listed run folders in {path}: {nested[:3]}")
    return lines


def find_member(members, key: str, name: str) -> dict:
    """The member a known_sha256 key names: its path under the base, else a unique file name."""
    hits = [m for m in members if m["rel"] == key] or [m for m in members if PurePosixPath(m["rel"]).name == key]
    if len(hits) != 1:
        raise SystemExit(f"{name}: known_sha256 key {key} matches {len(hits)} members")
    return hits[0]


def plan_item(spec: dict, item: dict, args, pool: cf.Executor, list_dir: Path) -> dict:
    name = item["name"]
    base, restore = item_paths(spec, item, args.source_repo, args.root)
    if not base.is_dir():
        raise SystemExit(f"{name}: source base {base} is not a directory on this machine")
    listed = item.get("id_list") or item.get("file_list")
    patterns = item.get("include", [] if listed else ["**/*"])
    hits = dict(zip(patterns, pool.map(lambda p: glob_files(base, p), patterns)))
    empty = [p for p, found in hits.items() if not found]
    if empty:
        raise SystemExit(f"{name}: include patterns select no file (or the file is missing): {empty}")
    rels = {r for found in hits.values() for r in found}
    if item.get("file_list"):
        rels.update(read_list(list_dir / item["file_list"], runs=False))  # stat_member fails on a missing one
    ids = read_list(list_dir / item["id_list"], runs=True) if item.get("id_list") else []
    if "runs" in item and item["runs"] != len(ids):
        raise SystemExit(f"{name}: the id list has {len(ids)} runs, the spec says {item['runs']}")
    runs = set(ids)

    def group(rel: str) -> str:  # a file inside a listed run folder is part of that run's group
        return next((p.as_posix() for p in PurePosixPath(rel).parents if p.as_posix() in runs), rel)

    members = {m["rel"]: m for m in pool.map(lambda r: stat_member(base, r, group(r)), sorted(rels))}
    episodes, per_run = [], Counter()
    for run_members, episode, counts in pool.map(lambda run: resolve_run(base, run, item), ids):
        members.update((m["rel"], m) for m in run_members)
        episodes.append(episode)
        per_run.update(counts)
    unused = [p for p in item.get("per_run_files", []) if ids and not per_run[p]]
    if unused:
        raise SystemExit(f"{name}: per_run_files patterns select no file in any run: {unused}")
    for rel in [r for r in members if any(fnmatch.fnmatchcase(r, p) for p in item.get("exclude", []))]:
        del members[rel]
    if not members:
        raise SystemExit(f"{name}: no files selected under {base}")
    ordered = [members[r] for r in sorted(members)]
    for m in ordered:
        m["path"] = safe_rel(f"{restore}/{m['rel']}" if restore else m["rel"], "member path")
    for key in item.get("known_sha256", {}):
        find_member(ordered, key, name)
    checkpoints = {m["rel"] for m in ordered if m["rel"].endswith((".pt", ".pth"))}
    unrecorded = item.get("params_unrecorded", False)
    stray = (set(item.get("known_params", {})) | set(unrecorded if isinstance(unrecorded, list) else [])) - checkpoints
    if stray:
        raise SystemExit(f"{name}: known_params / params_unrecorded name no released checkpoint: {sorted(stray)}")
    return {"item": item, "base": base, "restore": restore, "members": ordered, "ids": ids, "hits": hits,
            "per_run": per_run, "episodes": sorted(episodes, key=lambda e: e["run"]),
            "fingerprint": spec_fingerprint(item, list_dir)}


def check_spec(spec: dict, names: list[str]) -> dict:
    by_name = {it["name"]: it for it in spec["items"]}
    unknown = [n for n in names if n not in by_name]
    if unknown or len(by_name) != len(spec["items"]) or len(set(names)) != len(names):
        twice = sorted(n for n, k in Counter(names).items() if k > 1)
        raise SystemExit(f"unknown or duplicate items: {unknown or twice}")
    for it in spec["items"]:
        name = it["name"]
        if not ITEM_NAME.fullmatch(name) or it["bundle"] not in BUNDLES or it["packing"] not in ("files", "tar"):
            raise SystemExit(f"{name}: bad name, bundle or packing")
        shared = it.get("shared_paths_with", [])
        if any(o not in by_name or name not in by_name[o].get("shared_paths_with", []) for o in shared):
            raise SystemExit(f"{name}: shared_paths_with must name existing items that name it back: {shared}")
        if any(o not in by_name or o == name for o in it.get("requires", [])):
            raise SystemExit(f"{name}: requires names unknown items: {it['requires']}")
        if set(it.get("episode_defaults", {})) - set(EPISODE_COLUMNS[1:7]):
            raise SystemExit(f"{name}: episode_defaults may only set {EPISODE_COLUMNS[1:7]}")
        if not all(isinstance(v, int) and v > 0 for v in it.get("known_params", {}).values()):
            raise SystemExit(f"{name}: known_params values must be positive integers")
    return by_name


# ---------------------------------------------------------------- packing


def split_shards(members: list[dict], limit: int) -> list[list[dict]]:
    """Fill shards in path order; a group (one run folder, or one file) never straddles two shards."""
    groups: list[list[dict]] = []
    seen = set()
    for m in members:
        if not groups or groups[-1][0]["group"] != m["group"]:
            if m["group"] in seen:
                raise SystemExit(f"group {m['group']} is not contiguous in path order")
            seen.add(m["group"])
            groups.append([])
        groups[-1].append(m)
    shards: list[list[dict]] = [[]]
    size = 0
    for group in groups:
        nbytes = sum(m["bytes"] for m in group)
        if shards[-1] and size + nbytes > limit:
            shards.append([])
            size = 0
        shards[-1].extend(group)
        size += nbytes
    return shards


def write_shard(path: Path, members: list[dict]) -> dict:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        out = HashWriter(f)
        with gzip.GzipFile(filename="", mode="wb", fileobj=out, compresslevel=6, mtime=0) as gz:
            stream = HashWriter(gz)  # the uncompressed tar stream: comparable across zlib versions
            with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for m in members:
                    info = tarfile.TarInfo(m["path"])
                    info.size, info.mtime, info.mode = m["bytes"], 0, 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with open(m["src"], "rb") as src:
                        reader = HashReader(src)
                        try:
                            tar.addfile(info, reader)
                        except OSError as exc:
                            raise SystemExit(f"{m['src']} changed or failed while packing: {exc}") from None
                        if src.read(1):
                            raise SystemExit(f"{m['src']} grew while packing (past {m['bytes']} B)")
                    m["sha256"] = reader.h.hexdigest()
    os.replace(tmp, path)
    print(f"  wrote {path.name} ({len(members)} members, {out.n:,} B)", flush=True)
    return {"bytes": out.n, "sha256": out.h.hexdigest(), "tar_bytes": stream.n, "tar_sha256": stream.h.hexdigest()}


def copy_member(m: dict, dest: Path, link: bool, tmp_dir: Path) -> None:
    """Copy (or hard-link) one source file under a new random name in the private tmp_dir (same filesystem), then
    move it into place. Nothing but that own temporary name is ever removed; a hard link is only read."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    if link:
        try:
            tmp = tmp_dir / f".ln-{secrets.token_hex(12)}"
            os.link(m["src"], tmp)  # fails if the name exists
        except OSError:
            tmp = None
    try:
        if tmp:
            m["sha256"], size = sha256_file(tmp), tmp.stat().st_size
        else:
            h, size = hashlib.sha256(), 0
            with open(m["src"], "rb") as src:
                fd, name = tempfile.mkstemp(dir=tmp_dir, prefix=".cp-")  # O_EXCL: a new file of this job only
                tmp = Path(name)
                with os.fdopen(fd, "wb") as out:
                    while block := src.read(CHUNK):
                        h.update(block)
                        size += len(block)
                        out.write(block)
            os.chmod(tmp, 0o666 & ~UMASK)
            m["sha256"] = h.hexdigest()
        if size != m["bytes"]:
            raise SystemExit(f"{m['src']} changed while staging ({m['bytes']} -> {size} B)")
    except BaseException:
        if tmp:
            tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dest)


def put_small(files: dict, work: Path, folder: str, name: str, data: bytes, item: str, kind: str) -> None:
    (work / name).write_bytes(data)
    files[f"{folder}/{name}"] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "item": item,
                                 "kind": kind}


def check_known(plan: dict) -> None:
    name = plan["item"]["name"]
    for key, want in plan["item"].get("known_sha256", {}).items():
        got = find_member(plan["members"], key, name)["sha256"]
        if got != want:
            raise SystemExit(f"{name}: {key} does not match its known sha256 {want} (got {got})")


def spec_fingerprint(item: dict, list_dir: Path) -> dict:
    """What a fragment was built from: the item's spec entry, its id/file list and the builder (sha256 each)."""
    lists = {item[k]: sha256_file(list_dir / item[k]) for k in ("id_list", "file_list") if item.get(k)}
    parts = {"entry_sha256": canonical_sha256(item), "lists_sha256": lists, "builder_sha256": tool_record()["sha256"]}
    return {**parts, "sha256": canonical_sha256(parts)}


def finish_item(plan: dict, spec: dict, work: Path, shard_info: dict) -> dict:
    """Index files and the fragment of one packed item (in its work folder); checks known_sha256."""
    item, name, members = plan["item"], plan["item"]["name"], plan["members"]
    check_known(plan)
    folder = f"{PREFIX}{item['bundle']}/{name}"
    files = {}
    if item["packing"] == "files":
        for m in members:
            files[f"{folder}/{m['rel']}"] = {"bytes": m["bytes"], "sha256": m["sha256"], "item": name, "kind": "file"}
    else:
        for shard, info in shard_info.items():
            files[f"{folder}/{shard}"] = {**info, "item": name, "kind": "tar_shard"}
        put_small(files, work, folder, "index.csv.gz", csv_gz(INDEX_COLUMNS, members), name, "index")
        if plan["ids"]:
            put_small(files, work, folder, "ids.txt.gz", gzip_bytes("".join(f"{i}\n" for i in plan["ids"]).encode()),
                      name, "index")
        if item.get("episodes"):
            by_run: dict[str, list[dict]] = {}
            for m in members:
                by_run.setdefault(m["group"], []).append(m)
            for e in plan["episodes"]:
                run = by_run.get(e["run"], [])
                sha = {PurePosixPath(m["rel"]).relative_to(e["run"]).as_posix(): m["sha256"] for m in run}
                e.update(n_files=len(run), bytes=sum(m["bytes"] for m in run), shard=run[0]["shard"] if run else "",
                         trajectory_sha256=sha.get("trajectory.npz", ""), outcome_sha256=sha.get("outcome.json", ""))
            put_small(files, work, folder, "episodes.csv.gz", csv_gz(EPISODE_COLUMNS, plan["episodes"]), name,
                      "episodes")
    src = item["source"]
    entry = {key: item.get(key, "") for key in ("bundle", "milestone", "description", "needed_for", "packing", "notes")}
    entry.update({k: item[k] for k in ("requires", "shared_paths_with", "known_params", "params_unrecorded",
                                       "param_buffers", "episode_defaults") if k in item})
    entry.update(restore_root=plan["restore"], members=len(members), member_bytes=sum(m["bytes"] for m in members),
                 source={"machine": src["machine"], "root": src.get("root", "repo"), "base": src.get("base", "")},
                 files=sorted(files))
    if plan["ids"]:
        entry["runs"] = len(plan["ids"])
    return {"schema": SCHEMA, "repo_id": spec.get("repo_id", "harryzhang1018/NeDM"),
            "source_commit": spec.get("source_commit", ""), "root_map": root_map(spec),
            "path_remap": spec.get("path_remap"), "tool": tool_record(),
            "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "host": socket.gethostname(),
            "name": name, "spec_entry": plan["fingerprint"], "item": entry, "files": files}


def commit_item(staging: Path, plan: dict, fragment: dict, work: Path) -> None:
    """Replace the item's previous output with the finished work folder; the fragment is written last."""
    item, name = plan["item"], plan["item"]["name"]
    frags = staging / "_fragments"
    frags.mkdir(parents=True, exist_ok=True)
    (frags / f"{name}.json").unlink(missing_ok=True)
    final = staging / PREFIX / item["bundle"] / name
    shutil.rmtree(final, ignore_errors=True)
    final.parent.mkdir(parents=True, exist_ok=True)
    work.rename(final)
    (frags / f"{name}.paths.csv.gz").write_bytes(csv_gz(["path", "bytes", "sha256"], plan["members"]))
    (frags / f"{name}.json").write_text(json.dumps(fragment, indent=1, sort_keys=True) + "\n")
    total = sum(f["bytes"] for f in fragment["files"].values())
    print(f"{name}: {len(plan['members'])} members, {fragment['item']['member_bytes']:,} B -> "
          f"{len(fragment['files'])} files, {total:,} B", flush=True)


def tool_record() -> dict:
    return {"script": "traversing/scripts/release/build_release.py", "sha256": sha256_file(Path(__file__))}


def build(args) -> int:
    spec = json.loads(args.spec.read_text())
    names = [it["name"] for it in spec["items"]] if args.items == ["all"] else args.items
    check_spec(spec, names)
    by_name = {it["name"]: it for it in spec["items"]}
    list_dir = list_root(args, spec)
    tmp_dir = args.staging / "_work" / "_tmp"
    pool = cf.ThreadPoolExecutor(args.jobs)
    plans, done = [], set()
    try:
        for name in names:
            plan = plan_item(spec, by_name[name], args, pool, list_dir)
            print(f"{name}: {len(plan['members'])} members, {sum(m['bytes'] for m in plan['members']):,} B "
                  f"from {plan['base']} -> {plan['restore']}", flush=True)
            if args.dry_run:
                counts = {**{p: len(h) for p, h in plan["hits"].items()}, **{f"per run {p}": n for p, n in
                                                                              plan["per_run"].items()}}
                print("  pattern counts:", counts)
                if plan["episodes"] and plan["item"].get("episodes"):
                    print("  first episode:", plan["episodes"][0])
            plans.append(plan)
        if args.dry_run:
            return 0
        for plan in plans:  # checksums of record of small files, before any packing
            for key, want in plan["item"].get("known_sha256", {}).items():
                m = find_member(plan["members"], key, plan["item"]["name"])
                if m["bytes"] <= PRECHECK_BYTES and sha256_file(m["src"]) != want:
                    raise SystemExit(f"{plan['item']['name']}: {key} does not match its known sha256 {want}")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        for plan in plans:  # every item's copies and shards go into one pool; items finish in order
            item = plan["item"]
            plan["work"] = work = args.staging / "_work" / item["name"]
            shutil.rmtree(work, ignore_errors=True)
            work.mkdir(parents=True)
            if item["packing"] == "files":
                plan["shards"] = {}
                plan["jobs"] = [pool.submit(copy_member, m, work / m["rel"], args.link, tmp_dir)
                                for m in plan["members"]]
                continue
            shards = {f"part-{k:05d}.tar.gz": group for k, group in
                      enumerate(split_shards(plan["members"], args.shard_bytes))}
            for shard, group in shards.items():
                for m in group:
                    m["shard"] = shard
            plan["shards"] = {s: pool.submit(write_shard, work / s, g) for s, g in shards.items()}
            plan["jobs"] = list(plan["shards"].values())
        for plan in plans:
            for job in plan["jobs"]:
                job.result()
            fragment = finish_item(plan, spec, plan["work"], {s: job.result() for s, job in plan["shards"].items()})
            commit_item(args.staging, plan, fragment, plan["work"])
            done.add(plan["item"]["name"])
    except BaseException:
        pool.shutdown(wait=True, cancel_futures=True)  # queued shards and copies are dropped, running ones finish
        for plan in plans:
            if plan["item"]["name"] not in done and "work" in plan:
                shutil.rmtree(plan["work"], ignore_errors=True)
        raise
    finally:
        pool.shutdown()
        for folder in (tmp_dir, tmp_dir.parent):  # left in place while another build still uses them
            try:
                folder.rmdir()
            except OSError:
                pass
    return 0


def list_root(args, spec: dict) -> Path:
    """Folder of the id/file lists: --id-list-root, else the spec's id_list_root relative to the spec's folder."""
    return args.id_list_root or args.spec.resolve().parent / spec.get("id_list_root", ".")


# ---------------------------------------------------------------- merge and pin


def nested(a: str, b: str) -> bool:
    return a == b or not a or not b or a.startswith(b + "/") or b.startswith(a + "/")


def check_collisions(staging: Path, items: dict) -> None:
    """A restore path in two items needs shared_paths_with in both and identical content."""
    names = sorted(items)
    bad = []
    for i, a in enumerate(names):
        others = [b for b in names[i + 1:] if nested(items[a]["restore_root"], items[b]["restore_root"])]
        if not others:
            continue
        rows = {r["path"]: (r["bytes"], r["sha256"]) for r in read_csv_gz(staging / "_fragments" / f"{a}.paths.csv.gz")}
        for b in others:
            allowed = b in items[a].get("shared_paths_with", []) and a in items[b].get("shared_paths_with", [])
            for r in read_csv_gz(staging / "_fragments" / f"{b}.paths.csv.gz"):
                if r["path"] in rows and not (allowed and rows[r["path"]] == (r["bytes"], r["sha256"])):
                    bad.append(f"{r['path']} ({a}, {b})")
    if bad:
        raise SystemExit(f"{len(bad)} restore paths are claimed by two items without shared_paths_with or with "
                         f"different content: {bad[:5]}")


def merge(args) -> int:
    frags = [json.loads(p.read_text()) for p in sorted((args.staging / "_fragments").glob("*.json"))]
    if not frags:
        raise SystemExit(f"no fragments in {args.staging / '_fragments'}")
    for key in ("schema", "repo_id", "source_commit", "root_map", "path_remap", "tool"):
        values = {json.dumps(f.get(key), sort_keys=True) for f in frags}
        if len(values) != 1:
            raise SystemExit(f"fragments disagree on {key}: {sorted(values)}")
    spec = json.loads(args.spec.read_text())
    by_name, list_dir = {it["name"]: it for it in spec["items"]}, list_root(args, spec)
    names = {f["name"] for f in frags}
    missing, extra = sorted(set(by_name) - names), sorted(names - set(by_name))
    if missing or extra:
        raise SystemExit(f"items without a fragment: {missing}; fragments of items not in the spec: {extra}")
    stale = [f["name"] for f in frags
             if (f.get("spec_entry") or {}).get("sha256") != spec_fingerprint(by_name[f["name"]], list_dir)["sha256"]]
    if stale:
        raise SystemExit(f"fragments built from another spec entry, id/file list or builder (rebuild them): {stale}")
    want = {"repo_id": spec.get("repo_id", "harryzhang1018/NeDM"), "source_commit": spec.get("source_commit", ""),
            "root_map": root_map(spec), "path_remap": spec.get("path_remap")}
    other = [key for key, value in want.items() if json.dumps(frags[0].get(key), sort_keys=True) !=
             json.dumps(value, sort_keys=True)]
    if other:
        raise SystemExit(f"the fragments' {other} differ from the spec")
    items, files = {}, {}
    for f in frags:
        clash = set(files) & set(f["files"])
        if clash:
            raise SystemExit(f"{f['name']}: files already claimed by another item: {sorted(clash)[:5]}")
        items[f["name"]] = f["item"]
        files.update(f["files"])
    check_collisions(args.staging, items)
    for path, meta in files.items():  # staged copies may already be deleted after upload
        local = args.staging / path
        if local.exists() and local.stat().st_size != meta["bytes"]:
            raise SystemExit(f"{local}: size differs from its fragment")
    head = frags[0]
    manifest = {"schema": SCHEMA, "repo_id": head["repo_id"], "prefix": PREFIX, "source_commit": head["source_commit"],
                "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": head["tool"],
                "root_map": head["root_map"], **({"path_remap": head["path_remap"]} if head.get("path_remap") else {}),
                "items": dict(sorted(items.items())), "files": dict(sorted(files.items()))}
    out = args.staging / PREFIX
    out.mkdir(parents=True, exist_ok=True)
    (out / "release_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (out / "README.md").write_text(readme(manifest))
    print(f"merged {len(items)} items, {len(files)} files, {sum(m['bytes'] for m in files.values()):,} B -> {out}")
    return 0


def pin(args) -> int:
    """Write the GitHub copy of the manifest once the Hub holds exactly the staged one at a commit."""
    staged = (args.staging / PREFIX / "release_manifest.json").read_bytes()
    manifest = json.loads(staged)
    if not COMMIT.fullmatch(args.revision):
        raise SystemExit(f"--revision must be a 40-hex Hub commit, not {args.revision!r}")
    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

    repo_id = args.repo_id or manifest["repo_id"]
    if repo_id != manifest["repo_id"] and args.out is None:  # a test repo's commit must never reach the GitHub copy
        raise SystemExit(f"--repo-id {repo_id} is not the manifest's {manifest['repo_id']}; pass --out")
    out = args.out or PINNED
    if HfApi().dataset_info(repo_id, revision=args.revision).sha != args.revision:
        raise SystemExit(f"{args.revision} is not a commit of {repo_id}")
    with tempfile.TemporaryDirectory() as tmp:
        hub = Path(hf_hub_download(repo_id, f"{PREFIX}release_manifest.json", repo_type="dataset",
                                   revision=args.revision, local_dir=tmp, force_download=True)).read_bytes()
    if hub != staged:
        raise SystemExit(f"the Hub's {PREFIX}release_manifest.json at {args.revision} differs from the staged one")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**manifest, "hf_revision": args.revision}, indent=1) + "\n")
    print(f"pinned {repo_id}@{args.revision} ({len(manifest['files'])} files) -> {out}")
    return 0


def readme(manifest: dict) -> str:
    lines = ["# NeDM traversing release", "",
             "Data and models of the traversing study (learned route-risk planning for off-road vehicles in Project",
             "Chrono). The full description is in Part B of the dataset card at the repository root; every file with",
             "its size and SHA256 is in `release_manifest.json`. Restore items into a NeDM checkout with",
             "`python traversing/scripts/release/download_traversing_data.py --list --all` and `--items NAME ...`.",
             f"Source commit: `{manifest['source_commit']}`.", "",
             "| item | bundle | milestone | packing | files | bytes | restores to | description |",
             "|---|---|---|---|---|---|---|---|"]
    for name, it in manifest["items"].items():
        nbytes = sum(manifest["files"][p]["bytes"] for p in it["files"])
        milestone = ", ".join(it["milestone"]) if isinstance(it["milestone"], list) else it["milestone"]
        lines.append(f"| `{name}` | {it['bundle']} | {milestone} | {it['packing']} | {len(it['files'])} | "
                     f"{nbytes:,} | `{it['restore_root'] or '.'}` | {it['description']} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="Pack items into the staging tree and write their fragments.")
    b.add_argument("--spec", type=Path, required=True)
    b.add_argument("--items", nargs="+", required=True, help="Item names, or 'all'.")
    b.add_argument("--staging", type=Path, required=True)
    b.add_argument("--jobs", type=int, default=4, help="Parallel shard, copy and scan workers.")
    b.add_argument("--shard-bytes", type=int, default=1 << 30, help="Target member bytes per tar shard.")
    b.add_argument("--source-repo", type=Path, default=None, help="Checkout holding the local sources (default: this one).")
    b.add_argument("--root", action="append", default=[], metavar="KEY=PATH", help="Override a source root.")
    b.add_argument("--id-list-root", type=Path, default=None,
                   help="Directory of the id/file lists (default: the spec's id_list_root, relative to the spec).")
    b.add_argument("--link", action="store_true", help="Hard-link 'files' items instead of copying when possible.")
    b.add_argument("--dry-run", action="store_true", help="Resolve and count only; write nothing.")
    m = sub.add_parser("merge", help="Assemble release_manifest.json and README.md from the fragments.")
    m.add_argument("--staging", type=Path, required=True)
    m.add_argument("--spec", type=Path, required=True, help="Exactly one current fragment per spec item.")
    m.add_argument("--id-list-root", type=Path, default=None, help="As for build.")
    p = sub.add_parser("pin", help="Check the Hub manifest at a commit equals the staged one; write the GitHub copy.")
    p.add_argument("--staging", type=Path, required=True)
    p.add_argument("--revision", required=True, help="The Hub commit (40 hex) of the final upload.")
    p.add_argument("--repo-id", default=None, help="Default: the manifest's repo_id (another one needs --out).")
    p.add_argument("--out", type=Path, default=None, help="Default: traversing/manifests/hf_release_manifest.json.")
    args = parser.parse_args(argv)
    if args.command == "build":
        args.root = dict(r.split("=", 1) for r in args.root)
        args.source_repo = args.source_repo or Path(__file__).resolve().parents[3]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return {"build": build, "merge": merge, "pin": pin}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
