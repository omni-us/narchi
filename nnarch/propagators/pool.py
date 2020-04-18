"""Propagator classes for pooling layers."""

from .conv import ConvPropagator


class PoolPropagator(ConvPropagator):
    """Propagator for pooling style blocks."""

    num_features_source = 'from_shape'


propagators = [
    PoolPropagator('MaxPool1d', conv_dims=1),
    PoolPropagator('MaxPool2d', conv_dims=2),
    PoolPropagator('MaxPool3d', conv_dims=3),
]
