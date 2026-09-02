"""Episode-chunked compressed frame store for traversal collection (plan §6.1).

Schema v1, one directory per episode:

    ep_0007_spline/
      meta.json     schema/codec/chunking, layout manifest, route, camera
                    manifest, contact events, per-chunk byte index
      states.npz    float32 table (T, F) + field names: full capture_row set,
                    powertrain torque/speed, applied 20 Hz actions
      rgb.bin       concatenated compressed chunks, (chunk, H, W, 3) uint8
      depth.bin     concatenated compressed chunks, (chunk, H, W) uint16 mm

Frames are grouped into fixed-size temporal chunks (default 20 = 1 s at 20 Hz)
so a random training window touches at most two chunks. Within a chunk every
frame is stored as a wraparound difference against the chunk keyframe — the
arena background is static, so diffs are near-zero and zstd crushes them —
while random access inside the chunk stays O(1) (diff vs keyframe, not vs the
previous frame).

Depth is quantized to uint16 millimeters above ``DEPTH_OFFSET_M`` (ray depths
here live in ~85–120 m; WP0b measured 6–17 mm sensor error, so 1 mm
quantization is far below the noise floor). No-hit / out-of-range pixels store
``DEPTH_NO_HIT``.

Codec: zstandard when importable, zlib otherwise (recorded per episode; the
reader honors whatever the file says). Everything is numpy-only so the same
module serves collection (newton) and training (local) without torch.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import zstandard as _zstd
except ImportError:  # pragma: no cover - zlib fallback keeps stock envs working
    _zstd = None

SCHEMA_VERSION = 1
META_NAME = "meta.json"
STATES_NAME = "states.npz"
DEPTH_OFFSET_M = 80.0
DEPTH_NO_HIT = np.uint16(65535)
DEPTH_MAX_ENCODE_MM = 65534


def default_codec() -> str:
    return "zstd" if _zstd is not None else "zlib"


def _compress(payload: bytes, codec: str, level: int) -> bytes:
    if codec == "zstd":
        if _zstd is None:
            raise RuntimeError("episode was requested as zstd but zstandard is not importable")
        return _zstd.ZstdCompressor(level=level).compress(payload)
    if codec == "zlib":
        return zlib.compress(payload, level=min(level, 9))
    raise ValueError(f"unknown codec: {codec}")


def _decompress(blob: bytes, codec: str) -> bytes:
    if codec == "zstd":
        if _zstd is None:
            raise RuntimeError("episode is zstd-compressed but zstandard is not importable")
        return _zstd.ZstdDecompressor().decompress(blob)
    if codec == "zlib":
        return zlib.decompress(blob)
    raise ValueError(f"unknown codec: {codec}")


def encode_depth_mm(depth_m: np.ndarray) -> np.ndarray:
    """float32 ray-depth (m) -> uint16 mm above DEPTH_OFFSET_M, sentinel no-hit."""
    mm = np.round((np.asarray(depth_m, np.float64) - DEPTH_OFFSET_M) * 1000.0)
    invalid = ~np.isfinite(mm) | (mm < 0) | (mm > DEPTH_MAX_ENCODE_MM)
    out = np.where(invalid, float(DEPTH_NO_HIT), mm).astype(np.uint16)
    return out


def decode_depth_m(depth_mm: np.ndarray) -> np.ndarray:
    """uint16 mm -> float32 meters with NaN at the no-hit sentinel."""
    out = DEPTH_OFFSET_M + np.asarray(depth_mm, np.float32) / 1000.0
    out[np.asarray(depth_mm) == DEPTH_NO_HIT] = np.nan
    return out


def _encode_chunk(frames: np.ndarray, codec: str, level: int) -> bytes:
    """(C, ...) uint8/uint16 -> keyframe + wraparound diffs, compressed."""
    key = frames[:1]
    diff = frames[1:] - key  # unsigned wraparound (mod 2^bits), exactly invertible
    return _compress(key.tobytes() + diff.tobytes(), codec, level)


def _decode_chunk(blob: bytes, codec: str, n: int, frame_shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    raw = np.frombuffer(_decompress(blob, codec), dtype=dtype).reshape(n, *frame_shape)
    key = raw[:1]
    out = np.empty_like(raw)
    out[:1] = key
    out[1:] = raw[1:] + key  # unsigned wraparound undoes the diff
    return out


@dataclass
class _StreamWriter:
    path: Path
    frame_shape: tuple[int, ...]
    dtype: np.dtype
    codec: str
    level: int
    chunk_frames: int
    _handle: Any = None
    _buffer: list[np.ndarray] = field(default_factory=list)
    index: list[tuple[int, int, int]] = field(default_factory=list)  # (offset, nbytes, nframes)
    _offset: int = 0

    def append(self, frame: np.ndarray) -> None:
        frame = np.ascontiguousarray(frame, dtype=self.dtype)
        if frame.shape != self.frame_shape:
            raise ValueError(f"{self.path.name}: frame shape {frame.shape} != {self.frame_shape}")
        self._buffer.append(frame)
        if len(self._buffer) >= self.chunk_frames:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        if self._handle is None:
            self._handle = self.path.open("wb")
        blob = _encode_chunk(np.stack(self._buffer), self.codec, self.level)
        self._handle.write(blob)
        self.index.append((self._offset, len(blob), len(self._buffer)))
        self._offset += len(blob)
        self._buffer = []

    def close(self) -> int:
        self._flush()
        if self._handle is not None:
            self._handle.close()
        return self._offset


class EpisodeWriter:
    """Streams frames + state rows into one episode directory (bounded memory)."""

    def __init__(
        self,
        out_dir: Path,
        width: int,
        height: int,
        chunk_frames: int = 20,
        codec: str | None = None,
        level: int = 9,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_frames = chunk_frames
        self.codec = codec or default_codec()
        self.level = level
        self._rgb = _StreamWriter(
            self.out_dir / "rgb.bin", (height, width, 3), np.dtype(np.uint8),
            self.codec, level, chunk_frames,
        )
        self._depth = _StreamWriter(
            self.out_dir / "depth.bin", (height, width), np.dtype(np.uint16),
            self.codec, level, chunk_frames,
        )
        self._fields: list[str] | None = None
        self._rows: list[list[float]] = []
        self.frames = 0

    def append(self, rgb: np.ndarray, depth_m: np.ndarray, state_row: dict[str, float]) -> None:
        numeric = {k: v for k, v in state_row.items() if isinstance(v, (int, float, np.floating, np.integer))}
        if self._fields is None:
            self._fields = sorted(numeric)
        self._rgb.append(rgb)
        self._depth.append(encode_depth_mm(depth_m))
        self._rows.append([float(numeric.get(name, np.nan)) for name in self._fields])
        self.frames += 1

    def finalize(self, meta: dict[str, Any]) -> dict[str, Any]:
        rgb_bytes = self._rgb.close()
        depth_bytes = self._depth.close()
        table = np.asarray(self._rows, dtype=np.float32)
        np.savez_compressed(
            self.out_dir / STATES_NAME, fields=np.array(self._fields or []), table=table
        )
        h, w, _ = self._rgb.frame_shape
        raw_bytes = self.frames * (w * h * 3 + w * h * 2)
        full_meta = {
            "schema_version": SCHEMA_VERSION,
            "frames": self.frames,
            "width": w,
            "height": h,
            "chunk_frames": self.chunk_frames,
            "codec": self.codec,
            "codec_level": self.level,
            "delta_vs_keyframe": True,
            "depth_offset_m": DEPTH_OFFSET_M,
            "depth_no_hit": int(DEPTH_NO_HIT),
            "rgb_index": self._rgb.index,
            "depth_index": self._depth.index,
            "state_fields": self._fields or [],
            "bytes": {
                "rgb_bin": rgb_bytes,
                "depth_bin": depth_bytes,
                "states_npz": (self.out_dir / STATES_NAME).stat().st_size,
                "raw_frames": raw_bytes,
            },
            **meta,
        }
        with (self.out_dir / META_NAME).open("w", encoding="utf-8") as handle:
            json.dump(full_meta, handle, indent=1)
        return full_meta


class EpisodeReader:
    """Random-window access to one stored episode (tiny per-stream chunk cache)."""

    def __init__(self, ep_dir: Path, cache_chunks: int = 3) -> None:
        self.ep_dir = Path(ep_dir)
        with (self.ep_dir / META_NAME).open("r", encoding="utf-8") as handle:
            self.meta = json.load(handle)
        if self.meta["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"schema {self.meta['schema_version']} != {SCHEMA_VERSION}")
        self.frames = int(self.meta["frames"])
        self._cache_chunks = cache_chunks
        self._handles: dict[str, Any] = {}
        self._cache: dict[tuple[str, int], np.ndarray] = {}
        self._states: tuple[list[str], np.ndarray] | None = None

    def _read_chunk(self, stream: str, chunk_idx: int) -> np.ndarray:
        key = (stream, chunk_idx)
        if key in self._cache:
            return self._cache[key]
        offset, nbytes, nframes = self.meta[f"{stream}_index"][chunk_idx]
        if stream not in self._handles:
            self._handles[stream] = (self.ep_dir / f"{stream}.bin").open("rb")
        handle = self._handles[stream]
        handle.seek(offset)
        blob = handle.read(nbytes)
        h, w = int(self.meta["height"]), int(self.meta["width"])
        shape, dtype = ((h, w, 3), np.dtype(np.uint8)) if stream == "rgb" else ((h, w), np.dtype(np.uint16))
        chunk = _decode_chunk(blob, self.meta["codec"], nframes, shape, dtype)
        if len(self._cache) >= 2 * self._cache_chunks:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = chunk
        return chunk

    def _read_stream(self, stream: str, t0: int, n: int) -> np.ndarray:
        if t0 < 0 or t0 + n > self.frames:
            raise IndexError(f"window [{t0}, {t0 + n}) outside {self.frames} frames")
        cf = int(self.meta["chunk_frames"])
        parts = []
        t = t0
        while t < t0 + n:
            chunk_idx, within = divmod(t, cf)
            chunk = self._read_chunk(stream, chunk_idx)
            take = min(len(chunk) - within, t0 + n - t)
            parts.append(chunk[within : within + take])
            t += take
        return np.concatenate(parts) if len(parts) > 1 else parts[0].copy()

    def read_window(self, t0: int, n: int) -> dict[str, np.ndarray]:
        """{"rgb": (n,H,W,3) u8, "depth_mm": (n,H,W) u16, "states": (n,F) f32}"""
        fields, table = self.states()
        return {
            "rgb": self._read_stream("rgb", t0, n),
            "depth_mm": self._read_stream("depth", t0, n),
            "states": table[t0 : t0 + n],
            "state_fields": fields,
        }

    def states(self) -> tuple[list[str], np.ndarray]:
        if self._states is None:
            with np.load(self.ep_dir / STATES_NAME) as data:
                self._states = ([str(f) for f in data["fields"]], data["table"])
        return self._states

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._cache.clear()


def list_episodes(root: Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("ep_*/meta.json"))


def verify_episode(
    ep_dir: Path,
    raw_rgb: np.ndarray,
    raw_depth_m: np.ndarray,
    rng: np.random.Generator,
    n_windows: int = 3,
    window: int = 8,
) -> None:
    """Exact RGB + quantization-bounded depth roundtrip on random windows."""
    reader = EpisodeReader(ep_dir)
    try:
        for _ in range(n_windows):
            n = min(window, reader.frames)
            t0 = int(rng.integers(0, reader.frames - n + 1))
            got = reader.read_window(t0, n)
            if not np.array_equal(got["rgb"], raw_rgb[t0 : t0 + n]):
                raise AssertionError(f"{ep_dir}: RGB roundtrip mismatch at window {t0}")
            want = encode_depth_mm(raw_depth_m[t0 : t0 + n])
            if not np.array_equal(got["depth_mm"], want):
                raise AssertionError(f"{ep_dir}: depth roundtrip mismatch at window {t0}")
    finally:
        reader.close()
