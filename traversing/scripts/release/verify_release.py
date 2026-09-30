"""Check the traversing release on the Hub, in a local tree, and its models.

    python traversing/scripts/release/verify_release.py --hub                      # the pinned revision
    python traversing/scripts/release/verify_release.py --local /data/trv_staging  # staged or downloaded tree
    python traversing/scripts/release/verify_release.py --count-params net_s0.pt   # the known_params rule

The reference manifest is ``--manifest``, else the pinned GitHub copy ``traversing/manifests/hf_release_manifest.json``
(its ``hf_revision`` must be a 40-hex commit). ``--hub``: every file of the manifest exists under ``traversing/`` at
the revision with the same size and LFS/xet SHA256 (small non-LFS files are downloaded and hashed), the Hub's own
``release_manifest.json`` lists the same files and items, nothing else is under ``traversing/``, every path outside
``traversing/`` except ``README.md`` (and ``traversing/`` LFS rules the Hub appends to ``.gitattributes``) is unchanged
against ``--paper-revision`` (default: commit 8091c3b4..., which the
tag ``paper-v1`` pins; same blob id or LFS SHA256, none missing, none added), and every ``load_dataset`` config in the
YAML front matter of ``README.md`` at that commit is unchanged (added configs must read only ``traversing/`` files and
must not be the default).

``--local DIR``: the tree's own ``traversing/release_manifest.json`` must list the same files and items as the
reference manifest (with no pinned copy and no ``--manifest``, the tree's own manifest is used, with a note); every
manifest file present under DIR has the recorded size and SHA256, and each tar shard's uncompressed stream its
``tar_sha256`` (``--complete`` also requires all files); no restore path is claimed by two items unless both name each
other in ``shared_paths_with`` with identical content; every local-only record cited with a SHA256 in
``traversing/results/README.md`` and ``traversing/docs/evidence.md`` (tables, prose and bullets, and each
``*_sha256`` cell of the result tables) is in the release, as a plain file or as a tar member listed in an
``index.csv.gz``, at the cited path, and every 64-hex hash in the two documents is classified (local-only citation,
tracked record, or a repeat of one); and each model checkpoint (``.pt``/``.pth``) loads with torch
(``weights_only=True``, NumPy arrays and paths allowed; ``--unsafe-load`` allows a full unpickle only of files whose
size and SHA256 matched the manifest in this run) with the parameter count the item records in ``known_params`` (path
under the item -> count). The count is the number of elements of every tensor in the checkpoint's state dict, except
batch-norm ``running_mean``, ``running_var``, ``num_batches_tracked`` and the registered buffers the item names in
``param_buffers`` (matched on the last key component). The state dict is a pickled module's ``state_dict()``, else
the checkpoint itself if it is a dict of tensors, else the first of its keys ``state``, ``model``,
``model_state_dict``, ``state_dict`` that holds one, else the dict of tensors with the most elements inside it. A
checkpoint without a recorded count is UNVERIFIED and fails unless the item lists it in ``params_unrecorded`` (or sets
it to true). Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

from download_traversing_data import (CHUNK, COMMIT, MANIFEST, PINNED, REPO_ID, REPO_ROOT, check_shared, index_path,
                                      load_pinned, restore_targets, same_release, sha256_file)

EXTRA_OK = {MANIFEST, "traversing/README.md"}
DOCS = ("traversing/results/README.md", "traversing/docs/evidence.md")
SHA = re.compile(r"\b[0-9a-f]{64}\b")
TICKED = re.compile(r"`([^`]+)`")
BULLET = re.compile(r"\s*([-*]|\d+\.)\s")
CHECKPOINT = (".pt", ".pth")
BN_STATS = ("running_mean", "running_var", "num_batches_tracked")
STATE_KEYS = ("state", "model", "model_state_dict", "state_dict")
PAPER_V1 = "8091c3b4c9ac5dd270fd6a85eb95afe77652c6f2"  # the Hub commit the tag paper-v1 pins


def reference_manifest(args) -> dict | None:
    if args.manifest:
        return json.loads(args.manifest.read_text())
    return load_pinned(PINNED)


# ---------------------------------------------------------------- Hub


def blob(entry) -> tuple:
    lfs = entry.lfs
    sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
    return entry.blob_id, sha


def card_configs(repo_id: str, revision: str, folder: str) -> dict:
    """load_dataset configs (config_name -> entry) in the YAML front matter of the root README.md at a revision."""
    import yaml  # noqa: PLC0415  (a dependency of huggingface_hub)
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    text = Path(hf_hub_download(repo_id, "README.md", repo_type="dataset", revision=revision, local_dir=folder,
                                force_download=True)).read_text()
    match = re.match(r"---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
    meta = (yaml.safe_load(match.group(1)) if match else None) or {}
    return {c.get("config_name"): c for c in meta.get("configs") or []}


def data_paths(value) -> list[str]:
    """Every path string in a config's data_files (a string, a list, or split -> path(s) entries)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return data_paths(value.get("path", []))
    return [p for v in value or [] for p in data_paths(v)]


def gitattributes_only_added(repo_id: str, paper_rev: str, revision: str) -> bool:
    """The Hub appends LFS rules for large uploaded text files to .gitattributes. Accept that change only when every
    line of the paper's .gitattributes is kept, in order at the top, and every added line is a traversing/ rule."""
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    with tempfile.TemporaryDirectory() as tmp:
        old, new = (Path(hf_hub_download(repo_id, ".gitattributes", repo_type="dataset", revision=rev,
                                         local_dir=f"{tmp}/{i}")).read_text().splitlines()
                    for i, rev in enumerate((paper_rev, revision)))
    ok = new[:len(old)] == old and all(line.startswith("traversing/") for line in new[len(old):] if line.strip())
    if ok:
        print(f".gitattributes: paper lines kept, {len(new) - len(old)} traversing/ LFS rules added by the Hub")
    return ok


def check_paper(api, repo_id: str, remote: dict, paper_rev: str, revision: str) -> int:
    """Every path outside traversing/ except README.md is unchanged against the paper commit, and so is every
    load_dataset config in README.md's front matter (new configs may only read traversing/ and are never default)."""
    paper = {e.path: e for e in api.list_repo_tree(repo_id, recursive=True, repo_type="dataset", revision=paper_rev)
             if hasattr(e, "size") and not e.path.startswith("traversing/") and e.path != "README.md"}
    bad = 0
    for path, entry in sorted(paper.items()):
        now = remote.get(path)
        if now is None or not (blob(now)[0] == blob(entry)[0] or blob(now)[1] and blob(now)[1] == blob(entry)[1]):
            if now is not None and path == ".gitattributes" and gitattributes_only_added(repo_id, paper_rev, revision):
                continue
            print(f"PAPER {'MISSING' if now is None else 'CHANGED'}  {path}")
            bad += 1
    for path in sorted(p for p in remote if not p.startswith("traversing/") and p != "README.md" and p not in paper):
        print(f"PAPER ADDED  {path} (new files belong under traversing/)")
        bad += 1
    with tempfile.TemporaryDirectory() as tmp:
        old, new = card_configs(repo_id, paper_rev, f"{tmp}/paper"), card_configs(repo_id, revision, f"{tmp}/now")
    for name, entry in old.items():
        if new.get(name) != entry:
            print(f"PAPER CONFIG {'MISSING' if name not in new else 'CHANGED'}  {name} (README.md front matter)")
            bad += 1
    for name in (n for n in new if n not in old):
        if new[name].get("default") or not all(p.startswith("traversing/") for p in
                                                data_paths(new[name].get("data_files"))):
            print(f"CONFIG {name}: an added config must read only traversing/ files and must not be the default")
            bad += 1
    print(f"paper files: {len(paper)} files and {len(old)} README configs checked against {paper_rev}, {bad} problems")
    return bad


def check_hub(manifest: dict, repo_id: str, revision: str, paper_rev: str) -> int:
    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415

    api = HfApi()
    remote = {e.path: e for e in api.list_repo_tree(repo_id, recursive=True, repo_type="dataset", revision=revision)
              if hasattr(e, "size")}
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        for path, meta in manifest["files"].items():
            entry = remote.get(path)
            if entry is None:
                print(f"MISSING  {path}")
                bad += 1
                continue
            sha = blob(entry)[1]
            if sha is None:  # a small file stored in git: hash its content
                sha = sha256_file(Path(hf_hub_download(repo_id, path, repo_type="dataset", revision=revision,
                                                       local_dir=tmp, force_download=True)))
            if entry.size != meta["bytes"] or sha != meta["sha256"]:
                print(f"DIFFERS  {path}: Hub {entry.size} B {sha}, manifest {meta['bytes']} B {meta['sha256']}")
                bad += 1
        extra = sorted(p for p in remote if p.startswith("traversing/") and p not in manifest["files"]
                       and p not in EXTRA_OK)
        for path in extra:
            print(f"EXTRA    {path}")
        hub_manifest = json.loads(Path(hf_hub_download(repo_id, MANIFEST, repo_type="dataset", revision=revision,
                                                       local_dir=tmp, force_download=True)).read_text())
    if not same_release(hub_manifest, manifest):
        print(f"DIFFERS  {MANIFEST} on the Hub lists other files or items than the reference manifest")
        bad += 1
    print(f"hub {repo_id}@{revision}: {len(manifest['files'])} manifest files, {bad} problems, {len(extra)} extra")
    return bad + len(extra) + check_paper(api, repo_id, remote, paper_rev, revision)


# ---------------------------------------------------------------- local tree


def check_files(manifest: dict, tree: Path, complete: bool) -> tuple[int, set[str]]:
    """Problems, and the manifest paths whose size and SHA256 matched."""
    bad = missing = 0
    matched = set()
    for path, meta in manifest["files"].items():
        local = tree / path
        if not local.is_file():
            missing += 1
            continue
        if local.stat().st_size != meta["bytes"] or sha256_file(local) != meta["sha256"]:
            print(f"DIFFERS  {local}")
            bad += 1
        elif meta.get("tar_sha256"):
            h = hashlib.sha256()
            with gzip.open(local, "rb") as f:
                while block := f.read(CHUNK):
                    h.update(block)
            if h.hexdigest() != meta["tar_sha256"]:
                print(f"DIFFERS  {local} (uncompressed tar stream)")
                bad += 1
        else:
            matched.add(path)
    print(f"local {tree}: {len(manifest['files']) - missing} files checked, {bad} differ, {missing} not present")
    return bad + (missing if complete else 0), matched


def local_targets(manifest: dict, tree: Path) -> tuple[dict[str, list], list[str]]:
    """Restore targets of every item whose index (if any) is present locally, and the items without one."""
    targets, no_index = {}, []
    for name, item in manifest["items"].items():
        present = {p: tree / p for p in item["files"] if (tree / p).is_file()}
        index = index_path(manifest, name)
        if index and index not in present:
            no_index.append(name)
        targets[name] = restore_targets(manifest, name, present)
    return targets, no_index


def expand(path: str) -> list[str]:
    """`a/{x,y}.json` -> [a/x.json, a/y.json]."""
    m = re.search(r"\{([^{}]*)\}", path)
    return [path[:m.start()] + alt + path[m.end():] for alt in m.group(1).split(",")] if m else [path]


def with_prefix(prefix: str, path: str) -> str:
    return path if path.startswith("artifacts/") else prefix + path


def row_paths(cell: str, prefix: str, shas: list[str]) -> list[str | None]:
    """Cited path of each hash of a table row: one path, or one per hash in order (lists, braces); else unknown."""
    paths = [p for raw in TICKED.findall(cell) for p in expand(raw)]
    if "..." in cell or not paths or any("*" in p for p in paths) or len(paths) not in (1, len(shas)):
        return [None] * len(shas)
    if len(paths) == 1:
        return [with_prefix(prefix, paths[0]) if len(shas) == 1 else None] * len(shas)
    folder = str(PurePosixPath(paths[0]).parent)
    return [with_prefix(prefix, p if "/" in p or folder == "." else f"{folder}/{p}") for p in paths]


def blocks(text: str):
    """(first line number, 'row' or 'prose', text): table rows, and paragraphs/bullets with their continuation lines."""
    current = None
    for n, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("|"):
            if current:
                yield current
            current = None
            yield n, "row", line
        elif not line.strip() or line.lstrip().startswith("#") or BULLET.match(line):
            if current:
                yield current
            current = (n, "prose", line) if line.strip() else None
        elif current:
            current = (current[0], "prose", f"{current[2]} {line.strip()}")
        else:
            current = (n, "prose", line)
    if current:
        yield current


def citations(docs_root: Path) -> tuple[list[tuple[str, str, str | None]], list[tuple[str, str]]]:
    """Local-only records cited with a SHA256: (where, sha256, cited path or None); and the hashes in the two
    documents that are neither local-only citations, tracked records, nor repeats of one of those."""
    found, tracked, loose = [], set(), []
    for doc in DOCS:
        context, prefix, last_row, context_line = "", "", -1, -1
        for n, kind, text in blocks((docs_root / doc).read_text()):
            where, shas = f"{doc}:{n}", SHA.findall(text)
            if kind == "row" and 0 <= last_row < n - 1 and context_line < last_row:  # a new table, no prose before it
                context, prefix = "", ""
            if kind == "prose":
                context, context_line = text, n
                under = re.search(r"under `([^`]+)`", text)
                prefix = under.group(1) if under else ""
                if "local-only" in text.lower():
                    last = None
                    for token in TICKED.findall(text):
                        if SHA.fullmatch(token):
                            found.append((where, token, with_prefix(prefix, last) if last else None))
                        elif "/" in token or "." in token:
                            last = token
                    shas = [s for s in shas if f"`{s}`" not in text]  # bare hashes stay unclassified
                elif "tracked" in text.lower():
                    tracked.update(shas)
                    shas = []
                loose += [(where, s) for s in shas]
                continue
            last_row = n
            header = re.search(r"\(prefix `([^`]+)`\)", text)
            if header:
                prefix = header.group(1)
            if not shas:
                continue
            if re.search(r"\|\s*tracked\s*\|", text):
                tracked.update(shas)
            elif re.search(r"\|\s*local-only\s*\|", text) or "local-only" in context.lower():
                found += [(where, s, p) for s, p in zip(shas, row_paths(text.split("|")[1], prefix, shas))]
            else:
                loose += [(where, s) for s in shas]
    for table in sorted((docs_root / "traversing/results").glob("*.csv")):
        with open(table, newline="") as f:
            for n, row in enumerate(csv.DictReader(f), 2):
                found += [(f"{table.name}:{n}", v, None) for k, v in row.items() if k.endswith("_sha256") and v]
    known = tracked | {sha for _, sha, _ in found}
    return found, [(where, s) for where, s in loose if s not in known]


def check_citations(targets: dict[str, list], no_index: list[str], docs_root: Path, roots: dict[str, str],
                    complete: bool) -> int:
    """Every cited local-only record is in the release at its path. On a partial download, a record that would sit
    under an item whose index is not present is reported as unchecked instead (unless --complete)."""
    unindexed = [roots[name].rstrip("/") + "/" for name in no_index]
    by_sha = defaultdict(set)
    for rows in targets.values():
        for rel, sha, _ in rows:
            by_sha[sha].add(str(rel))
    cited, unclassified = citations(docs_root)
    bad = unchecked = 0
    for where, sha, path in cited:
        if sha not in by_sha and not complete and no_index and (not path or any(
                path.startswith(root) or root == "./" for root in unindexed)):
            unchecked += 1
            continue
        if sha not in by_sha:
            print(f"NOT RELEASED  {where}: {sha} {path or ''}")
            bad += 1
        elif path and path not in by_sha[sha]:
            print(f"OTHER PATH    {where}: {path} is released as {sorted(by_sha[sha])[:3]}")
            bad += 1
    for where, sha in unclassified:
        print(f"UNCLASSIFIED  {where}: {sha} is neither a local-only citation, a tracked record nor a repeat")
    note = f" (indexes not present locally for {len(no_index)} items: {no_index[:5]})" if no_index else ""
    if unchecked:
        note += f"; {unchecked} records under those items not checked (download them, or use --complete)"
    print(f"citations: {len(cited)} local-only SHA256 records, {bad} not found in the release, "
          f"{len(unclassified)} unclassified hashes{note}")
    return bad + len(unclassified)


# ---------------------------------------------------------------- models


def state_tensors(obj) -> dict | None:
    """The checkpoint's state dict: a pickled module's state_dict(), else the checkpoint itself if it is a dict of
    tensors, else the first of its keys state, model, model_state_dict, state_dict that holds one, else the dict of
    tensors with the most elements inside it."""
    if hasattr(obj, "state_dict") and hasattr(obj, "parameters"):
        return obj.state_dict()
    if not isinstance(obj, dict):
        return None
    if obj and all(hasattr(v, "numel") for v in obj.values()):
        return obj
    for key in STATE_KEYS:
        if key in obj and (found := state_tensors(obj[key])):
            return found
    found = [d for d in (state_tensors(v) for v in obj.values()) if d]
    return max(found, key=lambda d: sum(v.numel() for v in d.values()), default=None)


def param_count(obj, buffers=()) -> int | None:
    """Elements of every tensor of the state dict except batch-norm running statistics and the named buffers
    (last key component)."""
    tensors, skip = state_tensors(obj), set(BN_STATS) | set(buffers)
    return sum(v.numel() for k, v in tensors.items() if str(k).rsplit(".", 1)[-1] not in skip) if tensors else None


def load_checkpoint(torch, path: Path, unsafe: bool):
    """torch.load with weights_only=True, allowing NumPy arrays and paths (the release's checkpoints store both)."""
    import pathlib  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    core = getattr(np, "_core", None) or np.core
    allowed = [np.ndarray, np.dtype, core.multiarray._reconstruct, core.multiarray.scalar, pathlib.PosixPath,
               pathlib.WindowsPath] + [type(np.dtype(t)) for t in (np.float16, np.float32, np.float64, np.int8,
                                                                    np.int16, np.int32, np.int64, np.uint8, np.bool_)]
    try:
        with torch.serialization.safe_globals(allowed):
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001
        if not unsafe:
            raise
        return torch.load(path, map_location="cpu", weights_only=False)  # only files that matched their SHA256


def check_models(manifest: dict, tree: Path, unsafe: bool, matched: set[str]) -> int:
    """Parameter counts; --unsafe-load unpickles only checkpoints in matched (their size and SHA256 matched)."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        torch = None
    bad = checked = 0
    for name, item in manifest["items"].items():
        if item["bundle"] != "models":
            continue
        prefix = f"traversing/{item['bundle']}/{name}/"
        known, unrecorded = item.get("known_params", {}), item.get("params_unrecorded", False)
        for path in (p for p in item["files"] if p.endswith(CHECKPOINT) and (tree / p).is_file()):
            rel, checked = path[len(prefix):], checked + 1
            want = known.get(rel)
            marked = unrecorded is True or (isinstance(unrecorded, list) and rel in unrecorded)
            if torch is None:
                print(f"UNVERIFIED {name}/{rel}: torch is not installed (use --no-models to skip the model check)")
                bad += 1
                continue
            trusted = unsafe and path in matched
            try:
                n = param_count(load_checkpoint(torch, tree / path, trusted), item.get("param_buffers", []))
            except Exception as exc:  # noqa: BLE001
                hint = ("" if trusted else " (not unpickled: its SHA256 does not match the manifest)" if unsafe
                        else " (--unsafe-load allows a full unpickle)")
                print(f"BAD {name}/{rel}: does not load with weights_only={not trusted}{hint}: "
                      f"{(str(exc).splitlines() or [''])[0][:200]}")
                bad += 1
                continue
            if want is None:
                print(f"UNVERIFIED {name}/{rel}: {n} parameters, no recorded count"
                      f"{' (params_unrecorded)' if marked else ''}")
                bad += not marked
                continue
            note = " (computed; the training record states none)" if marked else ""
            print(f"{'ok' if n == want else 'BAD':3s} {name}/{rel}: {n} parameters, recorded {want}{note}")
            bad += n != want
    print(f"models: {checked} checkpoints checked, {bad} problems")
    return bad


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hub", action="store_true", help="Check the Hub copy at the revision.")
    parser.add_argument("--local", type=Path, default=None, help="Check a staged or downloaded tree.")
    parser.add_argument("--count-params", type=Path, nargs="+", default=None, help="Print the counts and exit.")
    parser.add_argument("--manifest", type=Path, default=None, help="Reference manifest (default: the pinned copy).")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default=None, help="Default: the manifest's hf_revision.")
    parser.add_argument("--paper-revision", default=PAPER_V1, help="The paper release commit (tag paper-v1) the files "
                        "outside traversing/ and the README configs must equal.")
    parser.add_argument("--docs-root", type=Path, default=REPO_ROOT, help="Checkout holding the cited documents.")
    parser.add_argument("--complete", action="store_true", help="With --local: every manifest file must be present.")
    parser.add_argument("--no-models", action="store_true", help="With --local: skip loading the models.")
    parser.add_argument("--param-buffers", nargs="+", default=[], help="With --count-params: buffers to leave out.")
    parser.add_argument("--unsafe-load", action="store_true",
                        help="Allow weights_only=False for checkpoints whose SHA256 matched the manifest.")
    args = parser.parse_args(argv)
    if not (args.hub or args.local or args.count_params):
        parser.error("choose --hub, --local DIR or --count-params")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.count_params:
        import torch  # noqa: PLC0415

        for path in args.count_params:
            print(f"{param_count(load_checkpoint(torch, path, args.unsafe_load), args.param_buffers)}\t{path}")
        return 0
    reference, bad = reference_manifest(args), 0
    if args.hub:
        if reference is None:
            raise SystemExit(f"no manifest at {PINNED}; pass --manifest")
        revision = args.revision or reference.get("hf_revision")
        if not args.revision and not COMMIT.fullmatch(str(revision)):
            raise SystemExit(f"the reference manifest's hf_revision must be a 40-hex Hub commit, not {revision!r}; "
                             "pass --revision")
        bad += check_hub(reference, args.repo_id, revision, args.paper_revision)
    if args.local:
        own = json.loads((args.local / MANIFEST).read_text()) if (args.local / MANIFEST).is_file() else None
        manifest = reference or own
        if manifest is None:
            raise SystemExit(f"no manifest: none pinned at {PINNED}, none at {args.local / MANIFEST}")
        if reference is None:
            print(f"NOTE: no pinned manifest ({PINNED}) and no --manifest: checking against the tree's own "
                  "manifest (before pinning)")
        elif own is not None and not same_release(own, reference):
            print(f"DIFFERS  {args.local / MANIFEST} lists other files or items than the reference manifest")
            bad += 1
        problems, matched = check_files(manifest, args.local, args.complete)
        bad += problems
        targets, no_index = local_targets(manifest, args.local)
        bad += check_shared(manifest, targets)
        roots = {name: item["restore_root"] for name, item in manifest["items"].items()}
        bad += check_citations(targets, no_index, args.docs_root, roots, args.complete)
        if not args.no_models:
            bad += check_models(manifest, args.local, args.unsafe_load, matched)
    print("verdict:", "PASS" if not bad else f"FAIL ({bad} problems)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
