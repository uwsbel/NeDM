"""Make a contact NRD checkpoint that origin/main's nedm.contact_nrd.model.load_model accepts, from a checkpoint of
the SO101 study's training code (commit f6f05c21c). The study trained with a training-only geometry loss: the
model_config key ``aux_geometry`` and two heads ``geo_collision`` / ``geo_contact`` (Linear 128 -> 12 each) that the
step never uses. This script removes exactly these: the key from the top-level ``model_config`` and the 4 tensors
``geo_collision.{weight,bias}``, ``geo_contact.{weight,bias}`` from ``model_state_dict``. Every other entry (also
the training record ``runtime``, ``validation``, ``sigma``, ``stage``, ``update``) is kept unchanged and in the same
order. Then it reloads the new file and checks every kept entry against the original bit for bit (tensors: dtype,
shape and raw bytes; other values: exact equality).

    python -I strip_ckpt.py SRC.pt DST.pt REPORT.json [EXPECTED.pt]

EXPECTED.pt (e.g. the released main-compatible best.pt): the new file is compared with it, by bytes and, if the bytes
differ, entry by entry as above. The bytes depend on the torch version (zip layout) and on the output file name (the
zip folder name inside the file): the release files were written by torch 2.12.0 under the name best.pt.

Release check of the SO101 pick-and-place study (so101_pick_place/README.md, "The NRD"); used by
scripts/so101_pick_place/reproduce.sh, step nrd_strip.
"""
import hashlib
import json
import math
import os
import sys
from collections import OrderedDict

import torch

GEO = ("geo_collision.", "geo_contact.")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(path):
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    return torch.load(path, map_location="cpu", weights_only=True)


def same(a, b, where, bad):
    """Exact equality; tensors by dtype, shape and raw bytes; floats NaN-aware; dict key order too."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        ok = (torch.is_tensor(a) and torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape
              and torch.equal(a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8)))
        if not ok:
            bad.append(where)
        return
    if type(a) is not type(b):
        bad.append(f"{where} (type {type(a).__name__} vs {type(b).__name__})")
        return
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            bad.append(f"{where} (keys/order)")
            return
        for k in a:
            same(a[k], b[k], f"{where}/{k}", bad)
        ma, mb = getattr(a, "_metadata", None), getattr(b, "_metadata", None)
        if ma is not None or mb is not None:
            same(dict(ma or {}), dict(mb or {}), f"{where}/_metadata", bad)
        return
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            bad.append(f"{where} (length)")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{where}[{i}]", bad)
        return
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        return
    if a != b:
        bad.append(where)


def main():
    src, dst, report = sys.argv[1:4]
    p = load(src)
    sd = p["model_state_dict"]
    removed = [k for k in sd if k.startswith(GEO)]
    assert sorted(removed) == sorted(["geo_collision.weight", "geo_collision.bias", "geo_contact.weight",
                                      "geo_contact.bias"]), removed
    assert p["model_config"].get("aux_geometry") is True, p["model_config"]
    new_sd = OrderedDict((k, v) for k, v in sd.items() if not k.startswith(GEO))
    meta = getattr(sd, "_metadata", None)
    if meta is not None:   # per-module version records of state_dict(); drop the two removed modules' entries
        new_sd._metadata = OrderedDict((k, v) for k, v in meta.items() if k not in ("geo_collision", "geo_contact"))
    new = {}
    for k, v in p.items():
        if k == "model_config":
            new[k] = {kk: vv for kk, vv in v.items() if kk != "aux_geometry"}
        elif k == "model_state_dict":
            new[k] = new_sd
        else:
            new[k] = v
    torch.save(new, dst)

    # check: reload both files from disk and compare everything that was kept
    a, b = load(src), load(dst)
    bad = []
    if list(a.keys()) != list(b.keys()):
        bad.append("top-level keys/order")
    for k in a:
        if k == "model_config":
            same({kk: vv for kk, vv in a[k].items() if kk != "aux_geometry"}, b[k], k, bad)
            if "aux_geometry" in b[k]:
                bad.append("model_config still has aux_geometry")
        elif k == "model_state_dict":
            kept = OrderedDict((kk, vv) for kk, vv in a[k].items() if not kk.startswith(GEO))
            ma = getattr(a[k], "_metadata", None)
            if ma is not None:
                kept._metadata = OrderedDict((kk, vv) for kk, vv in ma.items()
                                             if kk not in ("geo_collision", "geo_contact"))
            same(kept, b[k], k, bad)
            if any(kk.startswith(GEO) for kk in b[k]):
                bad.append("geo tensors still present")
        else:
            same(a[k], b[k], k, bad)
    out = dict(source=src, source_bytes=int(os.path.getsize(src)), source_sha256=sha256(src),
               released=dst, released_bytes=int(os.path.getsize(dst)), released_sha256=sha256(dst),
               removed_model_config_keys=["aux_geometry"], removed_tensors=removed,
               removed_parameters=int(sum(sd[k].numel() for k in removed)),
               kept_tensors=len(b["model_state_dict"]), kept_parameters=int(sum(t.numel() for t in b["model_state_dict"].values())),
               state_dict_metadata=meta is not None,
               unchanged_entries=[k for k in a if k not in ("model_config", "model_state_dict")],
               note="runtime (the training record) is unchanged: it still lists aux_geometry in runtime.model_config "
                    "and runtime.config, and runtime.parameters counts the removed heads",
               mismatches=bad, all_kept_entries_equal=not bad, torch=torch.__version__)
    if len(sys.argv) > 4:                     # compare with an expected file (the released main-compatible one)
        exp = sys.argv[4]
        out.update(expected=exp, expected_sha256=sha256(exp), bytes_equal_to_expected=sha256(exp) == out["released_sha256"])
        if not out["bytes_equal_to_expected"]:
            bad_exp = []
            same(load(exp), b, "expected", bad_exp)
            out["entries_equal_to_expected"] = not bad_exp
            out["mismatches_to_expected"] = bad_exp
            bad += [f"expected: {m}" for m in bad_exp]
    with open(report, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("unchanged_entries",)}, indent=1))
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
