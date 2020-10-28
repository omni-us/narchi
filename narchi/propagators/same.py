"""Propagator classes that preserve the same shape."""

from .base import BasePropagator, get_shape, create_shape


class SameShapePropagator(BasePropagator):
    """Propagator for blocks in which the input and output shapes are the same."""

    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Calls the base class checks and if multi-input makes sure that all
        inputs have the same shape and if not multi-input makes sure that there
        is only a single input block.

        Args:
            from_blocks (list[Namespace]): The input blocks.
            block (Namespace): The block to propagate its shapes.

        Raises:
            ValueError: When multi_input==False and len(from_blocks) != 1.
        """
        super().initial_checks(from_blocks, block)
        if self.num_input_blocks is not None:
            shape = get_shape('out', from_blocks[0])
            if not all(shape == get_shape('out', b) for b in from_blocks[1:]):
                in_shapes = ', '.join('[id='+b._id+']='+str(get_shape('out', b)) for b in from_blocks)
                raise ValueError(f'block[id={block._id}] of type {self.block_class} requires all inputs to '
                                 f'have the same shape, but got {in_shapes}.')
        else:
            if len(from_blocks) != 1:
                raise ValueError(f'block[id={block._id}] of type {self.block_class} only accepts one input '
                                 f'block, but got {len(from_blocks)}.')


    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[Namespace]): The input blocks.
            block (Namespace): The block to propagate its shapes.
        """
        block._shape = create_shape(get_shape('out', from_blocks[0]))


class SameShapesPropagator(SameShapePropagator):
    """Propagator for blocks that receive multiple inputs of the same shape and preserves this shape."""

    num_input_blocks = '>1'


class SameShapeConsumeDimPropagator(SameShapePropagator):
    """Propagator for blocks in which the output shape is the same as input except the last which is consumed."""

    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Calls the base class checks and makes sure that the input has more than
        one dimension.

        Args:
            from_blocks (list[Namespace]): The input blocks.
            block (Namespace): The block to propagate its shapes.

        Raises:
            ValueError: When len(input_shape) < 2.
        """
        super().initial_checks(from_blocks, block)
        if len(get_shape('out', from_blocks[0])) < 2:
            raise ValueError(f'block[id={block._id}] of type {self.block_class} requires input to have '
                             f'at least two dimensions.')

    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[Namespace]): The input blocks.
            block (Namespace): The block to propagate its shapes.
        """
        block._shape = create_shape(get_shape('out', from_blocks[0]), get_shape('out', from_blocks[0])[:-1])
