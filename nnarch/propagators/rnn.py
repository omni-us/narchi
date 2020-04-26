"""Propagator classes for recurrent blocks."""

from .base import BasePropagator, get_shape, create_shape, set_shape_dim


class RnnPropagator(BasePropagator):
    """Propagator for recurrent style blocks."""

    num_input_blocks = 1
    output_size_dims = 1


    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Calls the base class checks and makes sure that the input shape has two
        dimensions and that block includes a valid output_size attribute.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.

        Raises:
            ValueError: When block.output_size not valid.
            ValueError: When len(from_block[0]._shape) != 2.
        """
        super().initial_checks(from_blocks, block)
        shape_in = get_shape('out', from_blocks[0])
        if len(shape_in) != 2:
            raise ValueError(block._class+' blocks require input shape to have 2 dimensions, but got '+str(shape_in)+'.')


    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.

        Raises:
            ValueError: When bidirectional==True and output_size not even.
        """
        ## Set default values ##
        if not hasattr(block, 'bidirectional'):
            block.bidirectional = False

        ## Initialize block._shape ##
        from_shape = get_shape('out', from_blocks[0])
        output_size = block.output_size
        block._shape = create_shape(from_shape, ['<<auto>>', output_size])

        ## Set hidden size ##
        if block.bidirectional and output_size % 2 != 0:
            raise ValueError('For bidirectional '+block._class+' expected output_size to be even, but got '+str(output_size)+'.')
        block.hidden_size = output_size // (2 if block.bidirectional else 1)

        ## Propagate first dimension ##
        set_shape_dim('out', block, 0, from_shape[0])


propagators = [
    RnnPropagator('RNN'),
    RnnPropagator('LSTM'),
    RnnPropagator('GRU'),
]
