"""Propagator classes for other simple layers."""

from .base import BasePropagator, get_shape, create_shape


class LinearPropagator(BasePropagator):
    """Propagator for linear blocks."""

    out_features = True


    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.

        Raises:
            ValueError: When a valid out_features is expected but not found.
            NotImplementedError: If num_features_source is not one of {"from_shape", "out_features"}.
        """
        from_shape = get_shape('out', from_blocks[0])
        block._shape = create_shape(from_shape, from_shape[0:-1]+[block.out_features])


propagators = [
    LinearPropagator('Linear'),
]
