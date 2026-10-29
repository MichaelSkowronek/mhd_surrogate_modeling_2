import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mhd_surrogate.models.flow import (
    euler_sample,
    flow_matching_loss,
    interpolate,
    step_noise,
)

# A conditional Gaussian target: given the context c (window, r), each
# component of d is N(mu, sigma^2) with mu = the newest context entry.
SIGMA = jnp.array([0.5, 1.0, 2.0])


def exact_velocity(x, t, condition):
    """E[d - z | x_t = x, c] for that target: with a = 1 - t, b = t and
    V = a^2 + b^2 sigma^2 (the variance of x_t), it is
    mu + (b sigma^2 - a) (x - b mu) / V."""
    mu = condition[-1]
    a, b = 1.0 - t, t
    variance = a**2 + b**2 * SIGMA**2
    return mu + (b * SIGMA**2 - a) * (x - b * mu) / variance


def gaussian_batch(n, seed=0):
    """`n` (condition, target) pairs of that target, contexts of 2 steps."""
    rng = np.random.default_rng(seed)
    condition = rng.normal(size=(n, 2, 3)).astype(np.float32)
    target = condition[:, -1] + np.asarray(SIGMA) * rng.normal(size=(n, 3)).astype(np.float32)
    return jnp.asarray(condition), jnp.asarray(target)


def test_the_path_runs_from_the_noise_to_the_target():
    noise, target = jnp.array([1.0, -2.0]), jnp.array([3.0, 4.0])

    assert interpolate(noise, target, 0.0).tolist() == [1.0, -2.0]
    assert interpolate(noise, target, 1.0).tolist() == [3.0, 4.0]
    assert interpolate(noise, target, 0.25).tolist() == [1.5, -0.5]


def test_one_euler_step_of_the_exact_field_lands_on_the_conditional_mean():
    condition, _ = gaussian_batch(5)
    noise = jax.random.normal(jax.random.key(0), (5, 3))

    samples = jax.vmap(lambda c, z: euler_sample(exact_velocity, c, z, 1))(condition, noise)

    # Whatever the noise: x_0 = z says nothing about d.
    np.testing.assert_allclose(samples, condition[:, -1], atol=1e-6)


def test_many_euler_steps_of_the_exact_field_carry_the_noise_to_the_target():
    """In one dimension the flow's time-1 map is monotone and carries
    N(0, 1) to N(mu, sigma^2), so it is z -> mu + sigma z."""
    condition, _ = gaussian_batch(5)
    noise = jax.random.normal(jax.random.key(1), (5, 3))

    samples = jax.vmap(lambda c, z: euler_sample(exact_velocity, c, z, 400))(condition, noise)

    np.testing.assert_allclose(samples, condition[:, -1] + SIGMA * noise, atol=1e-2)


def test_the_exact_field_minimizes_the_loss():
    """E[d - z | x_t, c] is the least-squares fit of d - z: moving the
    field away from it in any direction g raises the loss."""
    condition, target = gaussian_batch(50_000)
    key = jax.random.key(2)

    def perturbed(eps):
        def velocity(x, t, c):
            return exact_velocity(x, t, c) + eps * (jnp.sin(3.0 * x) + t)

        return float(flow_matching_loss(velocity, condition, target, key))

    best = perturbed(0.0)
    assert best < perturbed(0.1) and best < perturbed(-0.1)
    assert best < 0.9 * perturbed(0.5)


def test_a_zero_field_scores_the_targets_mean_square_plus_one():
    condition, target = gaussian_batch(50_000, seed=3)

    loss = flow_matching_loss(
        lambda x, t, c: jnp.zeros_like(x), condition, target, jax.random.key(3)
    )

    assert float(loss) == pytest.approx(float(jnp.mean(target**2)) + 1.0, rel=0.01)


def test_the_loss_draws_its_noise_and_times_from_its_key():
    condition, target = gaussian_batch(64)

    def loss(seed):
        return float(flow_matching_loss(exact_velocity, condition, target, jax.random.key(seed)))

    assert loss(4) == loss(4)
    assert loss(4) != loss(5)


def test_step_noise_depends_on_the_key_and_the_step_alone():
    key = jax.random.key(6)

    noise = step_noise(key, 3, (1000,))

    np.testing.assert_array_equal(noise, step_noise(key, jnp.asarray(3), (1000,)))
    assert not np.allclose(noise, step_noise(key, 4, (1000,)))
    assert not np.allclose(noise, step_noise(jax.random.key(7), 3, (1000,)))
    assert float(jnp.std(noise)) == pytest.approx(1.0, abs=0.1)
