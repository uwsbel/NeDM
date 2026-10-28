"""Memory-mapped reader of an uncompressed .npz (np.savez): the start banks read the first states of chosen episodes
of the NRD data (unified_data.npz, up to 4.2 GB) without loading the whole file.

Copy of ``npz_members`` of the study branch's ``nedm/contact_nrd/rollouts.py`` (commit 359910370 on the branch
so101-pick-place-study), unchanged; that module is not part of this release (``task.load_start_bank`` uses only this
function). Main's ``nedm.g1_footstep_rl.npz_rows.npz_members`` is the same function.
"""
from __future__ import annotations

import struct
import zipfile

import numpy as np


def npz_members(path):
    """{name: read-only np.memmap} of the arrays of an uncompressed .npz (np.savez): rows are read on access, not
    the whole file."""
    out = {}
    with zipfile.ZipFile(path) as z, open(path, "rb") as fh:
        for info in z.infolist():
            if not info.filename.endswith(".npy"):
                continue
            if info.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"{path}: {info.filename} is compressed; this reader needs np.savez (no compression)")
            fh.seek(info.header_offset)
            head = fh.read(30)
            if head[:4] != b"PK\x03\x04":
                raise ValueError(f"{path}: bad local header of {info.filename}")
            name_len, extra_len = struct.unpack("<HH", head[26:30])
            fh.seek(info.header_offset + 30 + name_len + extra_len)
            version = np.lib.format.read_magic(fh)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
            elif version == (2, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
            else:
                raise ValueError(f"{path}: {info.filename} has .npy format {version}")
            if dtype.hasobject:
                raise ValueError(f"{path}: {info.filename} holds Python objects")
            out[info.filename[:-4]] = np.memmap(path, dtype=dtype, mode="r", offset=fh.tell(), shape=shape,
                                                order="F" if fortran else "C")
    return out
