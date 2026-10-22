"""NumPy arrays over HTTP: `.npy` records, one after the other.

A served forecast is too large for one response document (an 837-step
forecast is ~1 GB, several times that as JSON), so the forecast API streams
it as it is computed, a block of frames at a time
(`evaluation/rollout.py`'s `rollout_blocks`), each block an `.npy` record:
the format's own header (magic string, version, dtype, shape) followed by
the raw data. A reader needs nothing but NumPy and reads one record after
the other until the stream ends. The request's context is a single record.

Pickled (object) arrays are refused both ways: an `.npy` file with one is
code, not data.
"""

from __future__ import annotations

import io
from collections.abc import Iterable, Iterator

import numpy as np

MEDIA_TYPE = "application/x-npy"
STREAM_MEDIA_TYPE = "application/x-npy-stream"


def encode(array: np.ndarray) -> bytes:
    """`array` as one `.npy` record."""
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, np.ascontiguousarray(array), allow_pickle=False)
    return buffer.getvalue()


def decode(data: bytes) -> np.ndarray:
    """The array of the single `.npy` record `data`."""
    with io.BytesIO(data) as buffer:
        array = np.lib.format.read_array(buffer, allow_pickle=False)
        if buffer.read(1):
            raise ValueError("trailing bytes after the .npy record")
    return array


def read_arrays(chunks: Iterable[bytes]) -> Iterator[np.ndarray]:
    """The arrays of a stream of `.npy` records that arrives as `chunks` of
    any size; a stream that ends inside a record raises `ValueError`."""
    reader = io.BufferedReader(_ChunkReader(iter(chunks)))
    while reader.peek(1):
        yield np.lib.format.read_array(reader, allow_pickle=False)


class _ChunkReader(io.RawIOBase):
    """A readable file over an iterator of byte chunks."""

    def __init__(self, chunks: Iterator[bytes]):
        self._chunks = chunks
        self._pending = memoryview(b"")

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        while not self._pending:
            chunk = next(self._chunks, None)
            if chunk is None:
                return 0
            self._pending = memoryview(chunk)
        n = min(len(buffer), len(self._pending))
        buffer[:n] = self._pending[:n]
        self._pending = self._pending[n:]
        return n
