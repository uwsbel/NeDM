#!/usr/bin/env python3
"""Verifier's own exact comparison of two collector run folders (VERIFY_E2).

Every .npz: same key set, and for every key the same dtype, shape and bytes (np.array_equal with equal_nan plus a raw
byte comparison).  Every .json: parsed and compared recursively; the differing leaf paths are listed so wall-time-only
differences can be told apart from anything else.  Files present in only one folder are listed.
usage: cmp_runs.py A B [--out result.json]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def diff_json(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k} (only in {'B' if k not in a else 'A'})")
            else:
                out += diff_json(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path} (list length {len(a)} vs {len(b)})")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += diff_json(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(path)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out")
    x = p.parse_args()
    A, B = Path(x.a), Path(x.b)
    fa = {f.name for f in A.iterdir() if f.is_file()}
    fb = {f.name for f in B.iterdir() if f.is_file()}
    res = {"a": str(A), "b": str(B), "only_a": sorted(fa - fb), "only_b": sorted(fb - fa), "npz": {}, "json": {}}
    n_arrays = 0
    for name in sorted(fa & fb):
        if name.endswith(".npz"):
            za, zb = np.load(A / name, allow_pickle=False), np.load(B / name, allow_pickle=False)
            keys_a, keys_b = set(za.files), set(zb.files)
            bad = []
            for k in sorted(keys_a & keys_b):
                u, v = za[k], zb[k]
                n_arrays += 1
                same = (u.dtype == v.dtype and u.shape == v.shape and u.tobytes() == v.tobytes())
                if not same:
                    bad.append(k)
            res["npz"][name] = {"keys": len(keys_a & keys_b), "key_set_equal": keys_a == keys_b,
                                "differing_arrays": bad, "identical": keys_a == keys_b and not bad}
        elif name.endswith(".json"):
            ja, jb = json.loads((A / name).read_text()), json.loads((B / name).read_text())
            res["json"][name] = diff_json(ja, jb)
    res["arrays_compared"] = n_arrays
    res["all_npz_identical"] = all(v["identical"] for v in res["npz"].values()) and not res["only_a"] and not res["only_b"]
    js = json.dumps(res, indent=1)
    if x.out:
        Path(x.out).write_text(js + "\n")
    print(json.dumps({"all_npz_identical": res["all_npz_identical"], "arrays_compared": n_arrays,
                      "npz_files": len(res["npz"]), "only_a": res["only_a"], "only_b": res["only_b"],
                      "json_diff_paths": {k: v for k, v in res["json"].items() if v}}, indent=1))
    return 0 if res["all_npz_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
