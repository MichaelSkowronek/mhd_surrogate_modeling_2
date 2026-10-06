import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from mhd_surrogate.models.unet import UNet, _norm


def small_unet(in_channels=3, out_channels=2, depth=2, seed=0):
    return UNet(in_channels, out_channels, base_channels=4, depth=depth, key=jax.random.key(seed))


def test_output_has_the_input_resolution_and_the_requested_channels():
    net = small_unet(depth=2)
    x = jax.random.normal(jax.random.key(1), (3, 8, 12))

    assert net(x).shape == (2, 8, 12)


def test_untrained_network_outputs_zeros():
    """The zero-initialized head makes a residual model start as persistence."""
    net = small_unet()
    x = jax.random.normal(jax.random.key(1), (3, 8, 8))

    assert np.all(np.asarray(net(x)) == 0.0)


def test_head_is_the_only_zero_layer_and_still_receives_gradient():
    net = small_unet()
    x = jax.random.normal(jax.random.key(1), (3, 8, 8))
    target = jax.random.normal(jax.random.key(2), (2, 8, 8))

    grads = eqx.filter_grad(lambda n: jnp.mean((n(x) - target) ** 2))(net)

    assert float(jnp.abs(grads.head.weight).sum()) > 0
    assert float(jnp.abs(net.encoders[0].conv1.weight).sum()) > 0


def test_every_level_reaches_the_output():
    """Perturbing the input at one pixel changes the output beyond the
    finest level's receptive field, through the coarse levels: the skips and
    upsampling are wired, not just the first block."""
    net = small_unet(depth=2)
    # Give the head non-zero weights, or every output is zero.
    net = eqx.tree_at(lambda n: n.head.weight, net, jnp.ones_like(net.head.weight))
    x = jnp.zeros((3, 16, 16))
    delta = net(x.at[:, 0, 0].set(1.0)) - net(x)

    assert float(jnp.abs(delta[:, 15, 15]).max()) > 0


def test_rejects_a_resolution_that_is_not_a_multiple_of_2_to_the_depth():
    with pytest.raises(ValueError, match="multiple of 2\\*\\*depth = 4"):
        small_unet(depth=2)(jnp.zeros((3, 8, 6)))


def test_rejects_zero_depth():
    with pytest.raises(ValueError, match="depth"):
        UNet(1, 1, depth=0, key=jax.random.key(0))


def test_batches_with_vmap():
    net = small_unet()
    x = jax.random.normal(jax.random.key(1), (5, 3, 8, 8))

    out = jax.vmap(net)(x)

    assert out.shape == (5, 2, 8, 8)
    np.testing.assert_allclose(out[2], net(x[2]), atol=1e-6)


@pytest.mark.parametrize("channels, groups", [(4, 4), (8, 8), (32, 8), (12, 6), (7, 7), (9, 3)])
def test_group_count_is_the_largest_divisor_up_to_eight(channels, groups):
    assert _norm(channels).groups == groups
