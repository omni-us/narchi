"""Propagator classes that preserve the same shape."""

from .base import BasePropagator, get_shape, create_shape


class SameShapePropagator(BasePropagator):
    """Propagator for blocks in which the input and output shapes are the same."""

    multi_input = None


    def __init__(self, block_class, multi_input=False):
        """Initializer for ConvPropagator instance.

        Args:
            block_class (str): The name of the block class being propagated.
            multi_input (bool): Whether propagator accepts more than one input.
        """
        super().__init__(block_class)
        self.multi_input = multi_input


    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Calls the base class checks and if multi-input makes sure that all
        inputs have the same shape and if not multi-input makes sure that there
        is only a single input block.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.

        Raises:
            ValueError: When multi_input==False and len(from_blocks) != 1.
        """
        super().initial_checks(from_blocks, block)
        if self.multi_input:
            if len(from_blocks) < 2:
                raise ValueError('block[id='+block._id+'] of type '+self.block_class+' requires more than one '
                                 'input block, but got '+str(len(from_blocks))+'.')
            shape = get_shape('out', from_blocks[0])
            if not all(shape == get_shape('out', b) for b in from_blocks[1:]):
                in_shapes = ', '.join('[id='+b._id+']='+str(get_shape('out', b)) for b in from_blocks)
                raise ValueError('block[id='+block._id+'] of type '+self.block_class+' requires all inputs to '
                                 'have the same shape, but got '+in_shapes+'.')
        else:
            if len(from_blocks) != 1:
                raise ValueError('block[id='+block._id+'] of type '+self.block_class+' only accepts one input '
                                 'block, but got '+str(len(from_blocks))+'.')


    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.
        """
        block._shape = create_shape(get_shape('out', from_blocks[0]))


propagators = [
    SameShapePropagator('Identity'),
    SameShapePropagator('Sigmoid'),
    SameShapePropagator('LogSigmoid'),
    SameShapePropagator('Softmax'),
    SameShapePropagator('LogSoftmax'),
    SameShapePropagator('Tanh'),
    SameShapePropagator('ReLU'),
    SameShapePropagator('LeakyReLU'),
    SameShapePropagator('Dropout'),
    SameShapePropagator('BatchNorm2d'),
    SameShapePropagator('Add', multi_input=True),
]
