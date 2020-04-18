"""Propagator classes for reshaping."""

import numpy as np
from .base import BasePropagator, get_shape, create_shape


class PermutePropagator(BasePropagator):
    """Propagator for permutations."""

    num_input_blocks = 1


    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Calls the base class checks and makes sure that the dims attribute
        is valid and agrees with the input dimensions.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.

        Raises:
            ValueError: When block does not have a valid dim attribute that agrees with input dimensions.
        """
        super().initial_checks(from_blocks, block)
        if not hasattr(block, 'dims'):
            raise ValueError(block._class+' expected block to have a dims attribute, but not found for block '+block._id+'.')
        dims = sorted([x for dim in block.dims for x in ([dim] if isinstance(dim, int) else dim)])
        if dims != list(range(len(dims))):
            raise ValueError(block._class+' dim attribute expected to be list of ints or lists of ints '
                             'indicating complete unique dimension indexes, for block '+block._id+' got '+str(dims)+'.')
        shape_in = get_shape('out', from_blocks[0])
        if dims != list(range(len(shape_in))):
            raise ValueError('Number of dimensions indexes in dim attribute of block '+block._id+' does '
                             'not agree with the input dimensions from block '+from_blocks[0]._id+'.')


    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.
        """
        shape_in = get_shape('out', from_blocks[0])
        shape_out = []
        for dim in block.dims:
            if isinstance(dim, int):
                shape_out.append(shape_in[dim])
            else:
                if not all(isinstance(shape_in[x], int) for x in dim):
                    raise NotImplementedError(self.block_class+' does not currently support combination of variable dimensions.')
                shape_out.append(int(np.prod([shape_in[x] for x in dim])))
        block._shape = create_shape(shape_in, shape_out)


propagators = [
    PermutePropagator('Permute'),
]
