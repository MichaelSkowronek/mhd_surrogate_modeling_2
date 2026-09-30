import pytest

from mhd_surrogate.data.splitting import trailing_split


def test_trailing_split_matches_committed_re16k_t400_0_split():
    s = trailing_split(1248, val_steps=160, test_steps=400, buffer_steps=20)
    assert (s.train_start, s.train_end) == (0, 648)
    assert (s.val_start, s.val_end) == (668, 828)
    assert (s.test_start, s.test_end) == (848, 1248)


def test_trailing_split_buffer_zero_leaves_regions_adjacent():
    s = trailing_split(1000, val_steps=100, test_steps=200, buffer_steps=0)
    assert s.train_end == s.val_start
    assert s.val_end == s.test_start


def test_trailing_split_buffer_does_not_shrink_val_or_test():
    unbuffered = trailing_split(1000, val_steps=100, test_steps=200, buffer_steps=0)
    buffered = trailing_split(1000, val_steps=100, test_steps=200, buffer_steps=50)
    assert buffered.val_end - buffered.val_start == unbuffered.val_end - unbuffered.val_start
    assert buffered.test_end - buffered.test_start == unbuffered.test_end - unbuffered.test_start
    assert buffered.test_start == unbuffered.test_start
    assert buffered.test_end == unbuffered.test_end


def test_trailing_split_rejects_non_positive_val_steps():
    with pytest.raises(ValueError):
        trailing_split(1000, val_steps=0, test_steps=200)


def test_trailing_split_rejects_non_positive_test_steps():
    with pytest.raises(ValueError):
        trailing_split(1000, val_steps=100, test_steps=0)


def test_trailing_split_rejects_negative_buffer():
    with pytest.raises(ValueError):
        trailing_split(1000, val_steps=100, test_steps=200, buffer_steps=-1)


def test_trailing_split_rejects_sizes_that_empty_train():
    with pytest.raises(ValueError):
        trailing_split(100, val_steps=40, test_steps=40, buffer_steps=20)
