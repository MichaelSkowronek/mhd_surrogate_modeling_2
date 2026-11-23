"""The device (JAX, float32) reductions against their NumPy float64
references: the scoring's per-step quantities, diagnostics and ensemble
scores must not depend on where they are computed, nor on how a forecast
is cut into blocks."""

import numpy as np
import pytest

from mhd_surrogate.analysis.spectral import spectrum_sum
from mhd_surrogate.evaluation.diagnostics import (
    DeviceSeriesSummary,
    device_domain_mean,
    device_spectrum_sum,
    series_summary,
)
from mhd_surrogate.evaluation.ensemble import DeviceEnsembleAccumulator, EnsembleAccumulator
from mhd_surrogate.evaluation.metrics import (
    CrpsAccumulator,
    DeviceCrpsAccumulator,
    DeviceRmse,
    rmse_per_step,
)
from mhd_surrogate.evaluation.quantities import (
    device_energy_and_enstrophy,
    device_gradient,
    enstrophy,
    kinetic_energy,
    to_device,
)
from mhd_surrogate.evaluation.stability import DeviceQuantities, energy_and_enstrophy

DX, DY = 0.5, 0.25
SCALE = np.array([2.0, 0.5])


def random_series(n=11, shape=(2, 9, 6), seed=0, mean=0.0):
    rng = np.random.default_rng(seed)
    return (mean + rng.normal(size=(n, *shape))).astype(np.float32)


def blocks_of(series, sizes):
    """`series` cut into consecutive blocks of `sizes` steps, on the device."""
    edges = np.cumsum([0, *sizes])
    assert edges[-1] == len(series)
    return [(int(a), to_device(series[a:b])) for a, b in zip(edges[:-1], edges[1:])]


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_device_gradient_is_numpys_with_one_sided_edges(axis):
    f = random_series(n=5, shape=(4, 7))

    got = np.asarray(device_gradient(to_device(f), 0.3, axis))

    np.testing.assert_allclose(got, np.gradient(f.astype(np.float64), 0.3, axis=axis), rtol=1e-5)


def test_device_energy_and_enstrophy_of_analytic_flows():
    # A uniform flow (3, 4): energy 12.5. A solid body rotation
    # u_x = -y, u_y = x: vorticity 2, enstrophy 2, at every pixel.
    nx, ny = 8, 6
    x = np.arange(nx)[:, None] * DX * np.ones((1, ny))
    y = np.arange(ny)[None, :] * DY * np.ones((nx, 1))
    uniform = np.stack([np.full((nx, ny), 3.0), np.full((nx, ny), 4.0)])[None]
    rotation = np.stack([-y, x])[None]

    energy, _ = device_energy_and_enstrophy(to_device(uniform), DX, DY)
    _, ens = device_energy_and_enstrophy(to_device(rotation), DX, DY)

    assert np.asarray(energy) == pytest.approx([12.5])
    assert np.asarray(ens) == pytest.approx([2.0])


def test_device_quantities_are_the_float64_ones_in_any_blocks():
    series = random_series()
    expected = {
        "energy": kinetic_energy(series.astype(np.float64)),
        "enstrophy": enstrophy(series.astype(np.float64), DX, DY),
    }

    for sizes in ([11], [4, 4, 3], [1] * 11):
        quantities = DeviceQuantities(DX, DY)
        for _, block in blocks_of(series, sizes):
            quantities.add(block)
        got = quantities.series()

        for key in expected:
            np.testing.assert_allclose(got[key], expected[key], rtol=1e-6)
            assert got[key].dtype == np.float64
    assert energy_and_enstrophy(series, DX, DY).keys() == got.keys()


def test_device_quantities_need_frames():
    with pytest.raises(ValueError, match="no frames"):
        DeviceQuantities(DX, DY).series()


@pytest.mark.parametrize("shape", [(2, 8, 6), (2, 9, 7)])  # even and odd lengths
@pytest.mark.parametrize("axis", [2, 3])
def test_device_spectrum_sum_is_numpys(shape, axis):
    block = random_series(n=3, shape=shape)

    got = np.asarray(device_spectrum_sum(to_device(block), axis, 0.7))

    np.testing.assert_allclose(got, spectrum_sum(block.astype(np.float64), axis, 0.7), rtol=1e-5)


def test_the_device_domain_mean_resolves_a_near_constant_field():
    # The domain-mean u_x is ~1 and varies by ~0.4% over time: one float32
    # mean would round it to ~1e-7, the two parts resolve it far better.
    rng = np.random.default_rng(3)
    frames = (1.0 + 1e-3 * rng.normal(size=(4, 2, 300, 200))).astype(np.float32)
    exact = frames.astype(np.float64).mean(axis=(2, 3))

    reference, correction = device_domain_mean(to_device(frames))
    got = np.asarray(reference, np.float64) + np.asarray(correction, np.float64)

    assert np.abs(got - exact).max() < 1e-9


def test_a_device_summary_is_series_summary_in_any_blocks():
    series = random_series(n=12, mean=0.5)
    expected = series_summary(series, DX, DY)

    for sizes in ([12], [5, 5, 2]):
        summary = DeviceSeriesSummary(DX, DY)
        for _, block in blocks_of(series, sizes):
            summary.add(block)
        got = summary.summary()

        assert got.keys() == expected.keys()
        for key in expected:
            np.testing.assert_allclose(got[key], expected[key], rtol=1e-5, atol=1e-7)


def test_a_device_summary_needs_frames():
    with pytest.raises(ValueError, match="no frames"):
        DeviceSeriesSummary(DX, DY).summary()


def ensemble(n_members, n=10, seed=0):
    targets = random_series(n=n, seed=seed)
    members = [targets + random_series(n=n, seed=seed + 1 + k) for k in range(n_members)]
    return targets, members


def stream(accumulator, targets, members, sizes):
    for first, target in blocks_of(targets, sizes):
        stop = first + len(target)
        accumulator.add_block(first, target, (to_device(m[first:stop]) for m in members))
    return accumulator


@pytest.mark.parametrize("n_members", [1, 2, 3, 5])
@pytest.mark.parametrize("sizes", [[10], [3, 3, 4]])
def test_the_device_crps_is_the_numpy_one_from_the_same_cycle_of_pairs(n_members, sizes):
    targets, members = ensemble(n_members)
    expected = CrpsAccumulator(targets, SCALE)
    for member in members:
        expected.add(member)

    got = stream(DeviceCrpsAccumulator(len(targets), SCALE), targets, members, sizes)

    np.testing.assert_allclose(got.crps(), expected.crps(), rtol=1e-6)
    assert got.size == n_members


@pytest.mark.parametrize("n_members", [1, 2, 4])
def test_the_device_ensemble_scores_are_the_numpy_ones(n_members):
    targets, members = ensemble(n_members)
    expected = EnsembleAccumulator(targets, SCALE)
    for member in members:
        expected.add(member)
    expected = expected.scores()

    got = stream(DeviceEnsembleAccumulator(len(targets), SCALE), targets, members, [4, 6])
    got = got.scores()

    assert got.size == expected.size
    for name in ("crps", "spread", "ensemble_mean_rmse"):
        np.testing.assert_allclose(getattr(got, name), getattr(expected, name), rtol=1e-6)


def test_a_nan_member_makes_the_device_crps_nan_like_numpys():
    targets, members = ensemble(3)
    members[1] = members[1].copy()
    members[1][4] = np.nan

    got = stream(DeviceCrpsAccumulator(len(targets), SCALE), targets, members, [10])

    assert np.isnan(got.crps()[4]) and np.isfinite(np.delete(got.crps(), 4)).all()


def test_the_device_crps_refuses_blocks_of_other_sizes_missing_leads_and_no_members():
    targets, members = ensemble(2)
    accumulator = DeviceCrpsAccumulator(len(targets), SCALE)
    with pytest.raises(ValueError, match="no members"):
        accumulator.crps()

    accumulator.add_block(0, to_device(targets[:4]), (to_device(m[:4]) for m in members))
    with pytest.raises(ValueError, match="block of 1 members, expected 2"):
        accumulator.add_block(4, to_device(targets[4:]), [to_device(members[0][4:])])
    with pytest.raises(ValueError, match="block of 0 members"):
        DeviceCrpsAccumulator(len(targets), SCALE).add_block(0, to_device(targets), [])
    with pytest.raises(ValueError, match="lead times 5-10 were never added"):
        accumulator.crps()


def test_the_device_rmse_is_rmse_per_step():
    targets, (member,) = ensemble(1)
    rmse = DeviceRmse(len(targets), SCALE)
    for first, target in blocks_of(targets, [6, 4]):
        rmse.add(first, to_device(member[first : first + len(target)]), target)

    np.testing.assert_allclose(rmse.curve(), rmse_per_step(member, targets, SCALE), rtol=1e-6)
    with pytest.raises(ValueError, match="never added"):
        DeviceRmse(3, SCALE).curve()
