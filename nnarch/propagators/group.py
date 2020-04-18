"""Propagator classes for groups of blocks."""

from .base import BasePropagator, get_shape, create_shape


class SequentialPropagator(BasePropagator):
    """Propagator for a sequence of blocks."""

    num_input_blocks = 1
    requires_propagators = True


    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Calls the base class checks and makes sure that the input shape agrees
        with the convolution dimensions.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.

        Raises:
            ValueError: When conv_dims does not agree with from_block[0]._shape.
        """
        super().initial_checks(from_blocks, block)
        if not hasattr(block, 'blocks'):
            raise ValueError(block._class+' expected block to include attribute blocks, not found for block '+block._id+'.')
        if not isinstance(block.blocks, list) or len(block.blocks) < 1:
            raise ValueError(block._class+' expected block.blocks to be a list with at least one item, not found for block '+block._id+'.')


    def propagate(self, from_blocks, block, propagators):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.
            propagators (dict): Dictionary of propagators.

        Raises:
            ValueError: If no propagator found for some block.
        """
        from_block = from_blocks[0]
        for seq_block in block.blocks:
            if seq_block._class not in propagators:
                raise ValueError('No propagator found for block '+seq_block._id+' of type '+seq_block._class+' found while propagating '+block._class+' '+block._id+'.')
            propagator = propagators[seq_block._class]
            if propagator.requires_propagators:
                propagator([from_block], seq_block, propagators)
            else:
                propagator([from_block], seq_block)
            from_block = seq_block
        in_shape = get_shape('out', from_blocks[0])
        out_shape = get_shape('out', seq_block)
        block._shape = create_shape(in_shape, out_shape)


propagators = [
    SequentialPropagator('Sequential'),
]
