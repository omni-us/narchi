"""Propagator classes for concatenating."""

from .base import BasePropagator, get_shape, create_shape
from ..sympy import sum


class ConcatenatePropagator(BasePropagator):
    """Propagator for concatenating along a given dimension."""

    num_input_blocks = '>1'


    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Calls the base class checks and makes sure that the dim attribute
        is valid and agrees with the input dimensions.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.

        Raises:
            ValueError: When block does not have a valid dim attribute that agrees with input dimensions.
        """
        super().initial_checks(from_blocks, block)
        shape_0 = get_shape('out', from_blocks[0])
        dim = block.dim if block.dim >= 0 else len(shape_0)+block.dim
        if dim < 0 or dim >= len(shape_0):
            raise ValueError('Value of dim attribute ('+str(block.dim)+') in block[id='+block._id+'] does not '
                             'agree with the input dimensions coming from block[id='+from_blocks[0]._id+'].')
        for n in range(1, len(from_blocks)):
            shape_n = get_shape('out', from_blocks[n])
            if len(shape_0) != len(shape_n) or \
               any(shape_0[k] != shape_n[k] for k in range(len(shape_0)) if k != dim):
                raise ValueError(self.block_class+' expects all inputs to have the same shape except along '
                                 'the concatenating dimension, differs for block[id='+from_blocks[n]._id+'] '
                                 'connecting to block[id='+block._id+'], '+str(shape_0)+' vs. '+str(shape_n)+' .')


    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.
        """
        shape_in = list(get_shape('out', from_blocks[0]))
        shape_in[block.dim] = None
        shape_out = list(shape_in)
        shape_out[block.dim] = sum([get_shape('out', b)[block.dim] for b in from_blocks])
        block._shape = create_shape(shape_in, shape_out)
