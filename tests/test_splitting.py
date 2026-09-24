import pytest

from mhd_surrogate.splitting import trailing_split


def test_trailing_split_matches_committed_re16k_t400_0_split():
    s = trailing_split(1248, 0.2, buffer_steps=10)
    assert (s.train_start, s.train_end) == (0, 988)
    assert (s.test_start, s.test_end) == (998, 1248)
    assert s.test_end - s.test_start == 250


def test_trailing_split_buffer_zero_matches_unbuffered_split():
    assert trailing_split(1248, 0.2, buffer_steps=0).train_end == 998


def test_trailing_split_buffer_does_not_shrink_test_set():
    unbuffered = trailing_split(1248, 0.2, buffer_steps=0)
    buffered = trailing_split(1248, 0.2, buffer_steps=50)
    assert buffered.test_start == unbuffered.test_start
    assert buffered.test_end == unbuffered.test_end


@pytest.mark.parametrize("bad_fraction", [0.0, 1.0, -0.1, 1.5])
def test_trailing_split_rejects_invalid_test_fraction(bad_fraction):
    with pytest.raises(ValueError):
        trailing_split(100, bad_fraction)


def test_trailing_split_rejects_negative_buffer():
    with pytest.raises(ValueError):
        trailing_split(100, 0.2, buffer_steps=-1)


def test_trailing_split_rejects_buffer_that_empties_train():
    with pytest.raises(ValueError):
        trailing_split(100, 0.2, buffer_steps=100)


def test_trailing_split_rejects_too_few_steps():
    with pytest.raises(ValueError):
        trailing_split(1, 0.2)
