"""Propagator classes for convolution blocks."""

from .base import BasePropagator, get_shape, create_shape, set_shape_dim, check_output_size_dims


class ConvPropagator(BasePropagator):
    """Propagator for convolution style blocks."""

    num_input_blocks = 1
    num_features_source = 'output_size'
    conv_dims = None


    def __init__(self, block_class, conv_dims):
        """Initializer for ConvPropagator instance.

        Args:
            block_class (str): The name of the block class being propagated.
            conv_dims (int): Number of dimensions for the convolution.

        Raises:
            ValueError: If conv_dims not int > 0.
        """
        super().__init__(block_class)
        valid_num_features_source = {'output_size', 'from_shape'}
        if self.num_features_source not in valid_num_features_source:
            raise ValueError(type(self).__name__+' only allows num_features_source to be one of '+str(valid_num_features_source)+'.')
        if not isinstance(conv_dims, int) or conv_dims < 1:
            raise ValueError(type(self).__name__+' only allows conv_dims to be an int > 0.')
        self.conv_dims = conv_dims


    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Calls the base class checks and makes sure that the input shape agrees
        with the convolution dimensions.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.

        Raises:
            ValueError: When conv_dims does not agree with from_block[0]._shape.
        """
        super().initial_checks(from_blocks, block)
        shape_in = get_shape('out', from_blocks[0])
        if len(shape_in)-1 != self.conv_dims:
            raise ValueError(block._class+' blocks require input shape to have '+str(self.conv_dims+1)+' dimensions, but got '+str(shape_in)+'.')


    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.

        Raises:
            ValueError: When block.output_size not valid.
            NotImplementedError: If num_features_source is not one of {"from_shape", "output_size"}.
        """
        ## Set default values ##
        if not hasattr(block, 'stride'):
            block.stride = 1
        if not hasattr(block, 'padding'):
            block.padding = 0

        ## Initialize block._shape ##
        auto_dims = ['<<auto>>' for n in range(self.conv_dims)]
        from_shape = get_shape('out', from_blocks[0])
        if self.num_features_source == 'from_shape':
            block._shape = create_shape(from_shape, [from_shape[0]]+auto_dims)
        elif self.num_features_source == 'output_size':
            check_output_size_dims(1, self.block_class, block)
            block._shape = create_shape(from_shape, [block.output_size]+auto_dims)

        ## Calculate and set <<auto>> output dimensions ##
        if not (block.kernel_size == block.stride or block.kernel_size//2 == block.padding):
            raise NotImplementedError('<<auto>> output dims of '+block._class+' only implemented for kernel_size==stride and kernel_size//2==padding.')
        for dim, val in enumerate(get_shape('out', block)):
            if val == '<<auto>>':
                in_dim = get_shape('in', block)[dim]
                if block.kernel_size == block.stride:
                    set_shape_dim('out', block, dim, in_dim, '/'+str(block.kernel_size))
                else:
                    fact = '/'+str(block.stride) if hasattr(block, 'stride') and block.stride > 1 else None
                    set_shape_dim('out', block, dim, in_dim, fact=fact)


class PoolPropagator(ConvPropagator):
    """Propagator for pooling style blocks."""

    num_features_source = 'from_shape'
