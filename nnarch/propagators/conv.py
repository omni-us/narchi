"""Propagator classes for convolution layers."""

from .base import BasePropagator, get_shape, create_shape, set_shape_dim


class ConvPropagator(BasePropagator):
    """Propagator for convolution style blocks."""

    num_input_blocks = 1
    num_features_source = 'out_features'
    conv_dims = None


    def __init__(self, block_class, conv_dims):
        """Initializer for ConvPropagator instance.

        Args:
            block_class (str): The name of the block class being propagated.
            conv_dims (int): Number of dimensions for the convolution.

        Raises:
            NotImplementedError: If conv_dims is not one of {1, 2, 3}.
        """
        super().__init__(block_class)
        if conv_dims not in {1, 2, 3}:
            raise NotImplementedError(type(self).__name__+' only allows conv_dims to be one of {1, 2, 3}.')
        self.conv_dims = conv_dims


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
        shape_in = get_shape('out', from_blocks[0])
        if len(shape_in)-1 != self.conv_dims:
            raise ValueError(block._class+' blocks require input shape to have '+str(self.conv_dims+1)+' dimensions, but got '+str(shape_in)+'.')


    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.

        Raises:
            ValueError: When a valid out_features is expected but not found.
            NotImplementedError: If num_features_source is not one of {"from_shape", "out_features"}.
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
        elif self.num_features_source == 'out_features':
            if not hasattr(block, 'out_features'):
                raise ValueError(self.block_class+' expected block to include an out_features attribute.')
            out_features = block.out_features
            if not isinstance(out_features, int) or out_features < 1:
                raise ValueError(self.block_class+' expected block.out_features to be an int larger than zero.')
            block._shape = create_shape(from_shape, [block.out_features]+auto_dims)
        else:
            raise NotImplementedError(type(self).__name__+' only accepts num_features_source to be one of {"from_shape", "out_features"} but is "'+self.num_features_source+'".')

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


propagators = [
    ConvPropagator('Conv1d', conv_dims=1),
    ConvPropagator('Conv2d', conv_dims=2),
    ConvPropagator('Conv3d', conv_dims=3),
]
