"""Copy N episodes of one split (default splits == 1, the validation episodes; evenly spaced over the rows of that
split) of each NRD data part into a small data folder (unified_data.npz with every per-episode array sliced,
system.json copied). N = 0: every episode of the split. --split all: N episodes evenly spaced over all rows (smoke
training runs need train and validation rows).

    python -I subsets.py OUT N [--split 1|2|all] name=PART_DIR [name=PART_DIR ...]

Release checks of the SO101 pick-and-place study (so101_pick_place/README.md, "The NRD"; reproduce.sh steps nrd_val,
nrd_strip and the smoke of nrd_train). The release used: N = 64 of each of the 4 parts (original-vs-main check) and
N = 0 of production_plus (2,853 validation episodes, the validation columns of the README).
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out_root, n = Path(sys.argv[1]), int(sys.argv[2])
    args, split = sys.argv[3:], "1"
    if args[:1] == ["--split"]:
        split, args = args[1], args[2:]
    record = {}
    for item in args:
        name, src = item.split("=", 1)
        src = Path(src)
        out = out_root / name
        out.mkdir(parents=True, exist_ok=True)
        with np.load(src / "unified_data.npz") as p:
            files = list(p.files)
            splits = p["splits"]
            total = len(splits)
            val = np.arange(total) if split == "all" else np.flatnonzero(splits == int(split))
            pick = val if n == 0 or n >= len(val) else val[np.linspace(0, len(val) - 1, n).round().astype(np.int64)]
            assert len(np.unique(pick)) == len(pick), "not enough episodes"
            arrays = {}
            for k in files:                       # one array at a time (the large parts are several GB)
                a = p[k]
                if a.ndim == 0 or a.shape[0] != total:
                    raise ValueError(f"{src}: array {k} {a.shape} is not per episode")
                arrays[k] = np.ascontiguousarray(a[pick])
                del a
        np.savez(out / "unified_data.npz", **arrays)
        shutil.copy2(src / "system.json", out / "system.json")
        fam = arrays.get("family")
        families = [f["name"] for f in json.loads((src / "system.json").read_text()).get("families", [])]
        record[name] = dict(source=str(src), source_npz_sha256=sha256(src / "unified_data.npz"),
                            source_episodes=int(total), split=split, source_split_episodes=int(len(val)),
                            rows=pick.tolist(), subset_npz_sha256=sha256(out / "unified_data.npz"),
                            families={families[int(f)]: int(c) for f, c in zip(*np.unique(fam, return_counts=True))}
                            if fam is not None else None)
        print(name, total, "episodes,", len(val), f"in split {split}, picked", len(pick), record[name]["families"], flush=True)
    (out_root / "subsets.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
