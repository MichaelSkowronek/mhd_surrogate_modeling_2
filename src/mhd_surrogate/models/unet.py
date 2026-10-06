"""U-Net: the convolutional network behind the first neural surrogate.

A U-Net (Ronneberger et al., 2015) is an encoder-decoder of convolution
blocks: the encoder halves the resolution `depth` times while doubling the
channels, the decoder upsamples back, and skip connections concatenate each
encoder level's features into the decoder level of the same resolution. The
coarse levels see large structures (the bottleneck's receptive field spans a
good part of the domain), the skips keep the fine ones, which is why it's the
standard first network for field-to-field prediction.

This module is the network alone: a map from (in_channels, H, W) to
(out_channels, H, W) on unbatched arrays (Equinox convention; `jax.vmap` for
a batch). H and W must be multiples of 2**depth; `pad_to_multiple` and
`crop` get the 1151 x 127 frames there and back. What goes in and comes out
(normalized frames, coordinates, a residual update) is the surrogate's
business, not the network's.

Choices:
- GroupNorm, not BatchNorm: no running statistics to carry between training
  and inference, and it works at the small batch sizes these ~0.6 MB-per-
  channel frames force.
- Max-pooling down, transposed convolution up, GELU activations.
- The output layer is zero-initialized, so an untrained network outputs
  zeros: used as a residual update, the model starts out as persistence (a
  sane forecast) rather than noise.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

MAX_GROUPS = 8


def _norm(channels: int) -> eqx.nn.GroupNorm:
    groups = next(g for g in range(min(MAX_GROUPS, channels), 0, -1) if channels % g == 0)
    return eqx.nn.GroupNorm(groups, channels)


class ConvBlock(eqx.Module):
    """Two (3x3 convolution, GroupNorm, GELU) layers."""

    conv1: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d
    norm2: eqx.nn.GroupNorm

    def __init__(self, in_channels: int, out_channels: int, *, key: jax.Array) -> None:
        k1, k2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, 3, padding=1, key=k1)
        self.norm1 = _norm(out_channels)
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, 3, padding=1, key=k2)
        self.norm2 = _norm(out_channels)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jax.nn.gelu(self.norm1(self.conv1(x)))
        return jax.nn.gelu(self.norm2(self.conv2(x)))


class UNet(eqx.Module):
    """`depth` down/up levels; level i has `base_channels * 2**i` channels."""

    encoders: list[ConvBlock]
    upsamplers: list[eqx.nn.ConvTranspose2d]
    decoders: list[ConvBlock]
    head: eqx.nn.Conv2d
    pool: eqx.nn.MaxPool2d
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        *,
        key: jax.Array,
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        widths = [base_channels * 2**i for i in range(depth + 1)]
        keys = iter(jax.random.split(key, 3 * depth + 2))
        self.depth = depth
        self.encoders = [ConvBlock(in_channels, widths[0], key=next(keys))] + [
            ConvBlock(widths[i - 1], widths[i], key=next(keys)) for i in range(1, depth + 1)
        ]
        # Decoder level i (coarse to fine): upsample level i + 1's features to
        # level i's width, concatenate the skip, convolve.
        self.upsamplers = [
            eqx.nn.ConvTranspose2d(widths[i + 1], widths[i], 2, stride=2, key=next(keys))
            for i in reversed(range(depth))
        ]
        self.decoders = [
            ConvBlock(2 * widths[i], widths[i], key=next(keys)) for i in reversed(range(depth))
        ]
        head = eqx.nn.Conv2d(widths[0], out_channels, 1, key=next(keys))
        self.head = eqx.tree_at(
            lambda h: (h.weight, h.bias),
            head,
            (jnp.zeros_like(head.weight), jnp.zeros_like(head.bias)),
        )
        self.pool = eqx.nn.MaxPool2d(2, stride=2)

    def __call__(self, x: jax.Array) -> jax.Array:
        multiple = 2**self.depth
        if x.shape[-2] % multiple or x.shape[-1] % multiple:
            raise ValueError(
                f"spatial shape {x.shape[-2:]} must be a multiple of 2**depth = {multiple}; "
                "pad it with pad_to_multiple"
            )
        skips = []
        for encoder in self.encoders[:-1]:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.encoders[-1](x)
        for upsample, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            x = decoder(jnp.concatenate([upsample(x), skip], axis=0))
        return self.head(x)


def padded_size(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def pad_to_multiple(x: jax.Array, multiple: int) -> jax.Array:
    """Pad the last two axes at their far end up to a multiple of `multiple`,
    repeating the edge values (at a no-slip wall that's its zero velocity;
    at the outlet, the last column)."""
    h, w = x.shape[-2:]
    pad = [(0, 0)] * (x.ndim - 2) + [
        (0, padded_size(h, multiple) - h),
        (0, padded_size(w, multiple) - w),
    ]
    return jnp.pad(x, pad, mode="edge")


def crop(x: jax.Array, shape: tuple[int, int]) -> jax.Array:
    """Undo `pad_to_multiple`: keep the first `shape` entries of the last two axes."""
    return x[..., : shape[0], : shape[1]]


def coordinate_channels(shape: tuple[int, int]) -> jax.Array:
    """(2, H, W) grid coordinates, each scaled to [-1, 1].

    Convolutions are translation-equivariant, but this flow isn't
    homogeneous (an inlet, an outlet, two walls): concatenated to the input,
    the coordinates let the network tell where it is ("CoordConv").
    """
    h, w = shape
    x, y = jnp.meshgrid(jnp.linspace(-1.0, 1.0, h), jnp.linspace(-1.0, 1.0, w), indexing="ij")
    return jnp.stack([x, y])
