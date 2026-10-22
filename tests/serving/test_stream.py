import io

import numpy as np
import pytest

from mhd_surrogate.serving.stream import decode, encode, read_arrays


def arrays() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [
        rng.normal(size=(3, 2, 5, 4)).astype(np.float32),
        rng.normal(size=(1, 2, 5, 4)).astype(np.float32),
        np.arange(6, dtype=np.int64).reshape(2, 3),
        np.zeros((0, 2, 5, 4), dtype=np.float32),
    ]


def chunked(data: bytes, size: int) -> list[bytes]:
    return [data[i : i + size] for i in range(0, len(data), size)]


def test_an_encoded_array_decodes_to_itself():
    for array in arrays():
        decoded = decode(encode(array))
        assert decoded.dtype == array.dtype
        np.testing.assert_array_equal(decoded, array)


def test_encode_writes_a_view_as_its_values():
    array = np.arange(12, dtype=np.float32).reshape(3, 4).T  # not C-contiguous

    np.testing.assert_array_equal(decode(encode(array)), array)


def test_decode_refuses_trailing_bytes_garbage_and_pickles():
    with pytest.raises(ValueError, match="trailing bytes"):
        decode(encode(np.zeros(3)) + b"x")
    with pytest.raises(ValueError):
        decode(b"not an npy record at all")
    pickled = io.BytesIO()
    np.save(pickled, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(ValueError):
        decode(pickled.getvalue())


@pytest.mark.parametrize("size", [1, 7, 100, 10**6])
def test_read_arrays_reads_the_records_back_whatever_the_chunks(size):
    data = b"".join(encode(array) for array in arrays())
    chunks = chunked(data, size)
    chunks.insert(1, b"")  # an empty chunk is no end of stream

    read = list(read_arrays(chunks))

    assert len(read) == len(arrays())
    for got, expected in zip(read, arrays(), strict=True):
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)


def test_read_arrays_of_nothing_is_nothing():
    assert list(read_arrays([])) == []


def test_a_stream_ending_inside_a_record_raises():
    data = encode(np.ones((4, 2, 5, 4), dtype=np.float32))

    with pytest.raises(ValueError):
        list(read_arrays(chunked(data[:-10], 64)))
    with pytest.raises(ValueError):
        list(read_arrays([data[:20]]))  # inside the header
