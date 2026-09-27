"""Exact comparison of two run folders: every array of every .npz (np.array_equal, dtype and shape), JSON keys that differ."""
import json, sys
from pathlib import Path
import numpy as np

IGNORE = {"wall_s", "wall_s_including_finalization", "out", "rtf_sim_over_wall", "build_s", "collection_request_sha256",
          "request_sha256", "artifacts_sha256", "adapted_function_sha256", "case", "route"}


def flat(d, p=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from flat(v, f"{p}.{k}" if p else k)
    else:
        yield p, d


def main(a, b):
    a, b = Path(a), Path(b)
    rep = {"a": str(a), "b": str(b), "npz": {}, "json": {}}
    for f in sorted(a.glob("*.npz")):
        g = b / f.name
        if not g.exists():
            rep["npz"][f.name] = "missing in b"; continue
        za, zb = np.load(f, allow_pickle=False), np.load(g, allow_pickle=False)
        diff = [k for k in za.files if k not in zb.files or za[k].dtype != zb[k].dtype or za[k].shape != zb[k].shape
                or not np.array_equal(za[k], zb[k], equal_nan=za[k].dtype.kind == "f")]
        rep["npz"][f.name] = {"keys": len(za.files), "extra_in_b": sorted(set(zb.files) - set(za.files)), "different": diff,
                              "identical": not diff and set(za.files) == set(zb.files)}
    for f in sorted(a.glob("*.json")):
        g = b / f.name
        if not g.exists():
            rep["json"][f.name] = "missing in b"; continue
        fa, fb = dict(flat(json.loads(f.read_text()))), dict(flat(json.loads(g.read_text())))
        keys = sorted(set(fa) | set(fb))
        d = [k for k in keys if k.split(".")[-1] not in IGNORE and fa.get(k) != fb.get(k)]
        rep["json"][f.name] = d[:20]
    rep["files_only_in_b"] = sorted(p.name for p in b.iterdir() if not (a / p.name).exists())
    rep["files_only_in_a"] = sorted(p.name for p in a.iterdir() if not (b / p.name).exists())
    rep["trajectory_identical"] = rep["npz"].get("trajectory.npz", {}).get("identical", False)
    rep["all_npz_identical"] = all(isinstance(v, dict) and v["identical"] for v in rep["npz"].values())
    return rep


if __name__ == "__main__":
    r = main(sys.argv[1], sys.argv[2])
    print(json.dumps(r, indent=1))
