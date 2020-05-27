"""Propagator classes for fixed output blocks."""

from .base import BasePropagator, get_shape, create_shape


class AddFixedPropagator(BasePropagator):
    """Propagator for blocks that adds fixed dimensions."""

    fixed_dims = 1


    def __init__(self, block_class, fixed_dims=1):
        """Initializer for AddFixedPropagator instance.

        Args:
            block_class (str): The name of the block class being propagated.
            fixed_dims (int): Number of fixed dimensions.

        Raises:
            ValueError: If fixed_dims not int > 0.
        """
        super().__init__(block_class)
        if not isinstance(fixed_dims, int) or not fixed_dims > 0:
            raise ValueError(type(self).__name__+' requires fixed_dims to be an int > 0.')
        self.fixed_dims = fixed_dims


    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.
        """
        from_shape = get_shape('out', from_blocks[0])
        if self.fixed_dims == 1:
            to_shape = from_shape + [block.output_feats]
        else:
            to_shape = from_shape + block.output_feats
        block._shape = create_shape(from_shape, to_shape)


class FixedOutputPropagator(BasePropagator):
    """Propagator for fixed output size blocks."""

    num_input_blocks = 1
    unfixed_dims = 'any'
    output_feats_dims = 1


    def __init__(self, block_class, unfixed_dims='any', fixed_dims=1):
        """Initializer for FixedOutputPropagator instance.

        Args:
            block_class (str): The name of the block class being propagated.
            fixed_dims (int): Number of fixed dimensions.
            unfixed_dims (int or str): Number of unfixed dimensions.

        Raises:
            ValueError: If fixed_dims not int > 0.
            ValueError: If unfixed_dims not "any" or int > 0.
        """
        super().__init__(block_class)
        if not ((isinstance(unfixed_dims, int) and unfixed_dims > 0) or unfixed_dims == 'any'):
            raise ValueError(type(self).__name__+' requires unfixed_dims to be "any" or an int > 0.')
        if not isinstance(fixed_dims, int) or not fixed_dims > 0:
            raise ValueError(type(self).__name__+' requires fixed_dims to be an int > 0.')
        self.unfixed_dims = unfixed_dims
        self.output_feats_dims = fixed_dims


    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Calls the base class checks and makes sure that the input shape has at
        least (fixed_dims+1) dimensions if unfixed_dims=="any" or exactly
        (fixed_dims+fixed_dims) dimensions if unfixed_dims is int.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.

        Raises:
            ValueError: When fixed_dims and unfixed_dims do not agree with from_block[0]._shape.
        """
        super().initial_checks(from_blocks, block)
        from_shape = get_shape('out', from_blocks[0])
        msg = (block._class+' propagator requires input shape to have %s %d dimensions, but '
               'block[id='+from_blocks[0]._id+'] -> block[id='+block._id+'] has '+str(len(from_shape))+'.')
        if self.unfixed_dims == 'any' and len(from_shape) < self.output_feats_dims:
            raise ValueError(msg % ('at least', self.output_feats_dims))
        if isinstance(self.unfixed_dims, int) and len(from_shape) != self.output_feats_dims+self.unfixed_dims:
            raise ValueError(msg % ('exactly', self.output_feats_dims+self.unfixed_dims))


    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.
        """
        from_shape = get_shape('out', from_blocks[0])
        if self.output_feats_dims == 1:
            to_shape = from_shape[0:-self.output_feats_dims] + [block.output_feats]
        else:
            to_shape = from_shape[0:-self.output_feats_dims] + block.output_feats
        block._shape = create_shape(from_shape, to_shape)
