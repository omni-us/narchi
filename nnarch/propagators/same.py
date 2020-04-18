"""Propagator classes that preserve the same shape."""

from .base import BasePropagator, get_shape, create_shape


class SameShapePropagator(BasePropagator):
    """Propagator for blocks in which the input and output shapes are the same."""

    num_input_blocks = 1


    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.
        """
        block._shape = create_shape(get_shape('out', from_blocks[0]))


propagators = [
    SameShapePropagator('Identity'),
    SameShapePropagator('Sigmoid'),
    SameShapePropagator('LogSigmoid'),
    SameShapePropagator('Tanh'),
    SameShapePropagator('ReLU'),
    SameShapePropagator('LeakyReLU'),
    SameShapePropagator('BatchNorm2d'),
]
