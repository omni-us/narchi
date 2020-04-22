"""Base propagator class and related functions."""

from jsonargparse import SimpleNamespace, dict_to_namespace
from copy import deepcopy
from ..sympy import variable_operate, is_valid_dim


def get_shape(key, shape):
    """Gets the shape list for a given key among {'in','out'}."""
    if isinstance(shape, SimpleNamespace) and hasattr(shape, '_shape'):
        shape = shape._shape
    if isinstance(shape, SimpleNamespace):
        shape = vars(shape)
    if isinstance(shape, list):
        return shape
    return shape[key]


def create_shape(shape_in, shape_out=None):
    """Creates a shape namespace with 'in' and 'out' attributes and copied shape arrays."""
    shape = {'in': deepcopy(shape_in),
             'out': shape_in if shape_out is None else shape_out}
    return dict_to_namespace(deepcopy(shape))


def set_shape_dim(key, shape, dim, val, fact=None):
    """Sets a value for a given dimension, shape and key ('in' or 'out')."""
    if fact is not None:
        assert fact[0] in {'/', '*'}, 'Expected factor to start with "/" or "*" but got '+fact
        val = variable_operate(val, '__input__'+fact)
    shape = get_shape(key, shape)
    shape[dim] = val


def shapes_agree(shape_from, shape_to):
    """Checks whether the output shape from a block agrees with input shape of another block."""
    return get_shape('out', shape_from) == get_shape('in', shape_to)


def shape_has_auto(shape):
    """Checks whether a shape has any <<auto>> values."""
    if isinstance(shape, str):
        shape = [shape]
    if any([x == '<<auto>>' for x in shape]):
        return True
    return False


def check_output_size_dims(output_size_dims, block_class, block):
    """Checks the output_size attribute of a block."""
    if output_size_dims in {1, 2, 3}:
        if not hasattr(block, 'output_size'):
            raise ValueError(block_class+' propagator expected block[id='+block._id+'] to include an output_size attribute.')
        if output_size_dims == 1 and not is_valid_dim(block.output_size):
            raise ValueError(block_class+' propagator expected block[id='+block._id+'] output_size to be a '
                             'variable or an int larger than zero.')
        if output_size_dims > 1 and (not isinstance(block.output_size, list) or not all(is_valid_dim(x) for x in block.output_size)):
            raise ValueError(block_class+' propagator expected block[id='+block._id+'] output_size to be a '
                             'list with '+str(output_size_dims)+' variables or ints larger than zero.')


class BasePropagator:
    """Base class for block shapes propagation."""

    block_class = None
    num_input_blocks = None
    output_size_dims = False
    requires_propagators = False


    def __init__(self, block_class):
        """Initializer for BasePropagator instance.

        Args:
            block_class (str): The name of the block class being propagated.
        """
        self.block_class = block_class


    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Extensions of this method in derived classes should always call this
        base method. This base method implements the following checks:

        - That the block class is the same as the one expected by the
          propagator.
        - That the input shapes don't contain any <<auto>> values.
        - If num_input_blocks is set and is an int, that there are exactly this
          number of input blocks.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.

        Raises:
            ValueError: When block._class != block_class.
            ValueError: When input shape contains <<auto>>.
            ValueError: When len(from_blocks) != num_input_blocks.
            NotImplementedError: When block._shape != '<<auto>>'.
        """
        if not hasattr(block, '_class'):
            raise ValueError(self.block_class+' propagator expected block to include a _class attribute.')

        if block._class != self.block_class:
            raise ValueError('Attempted to propagate a '+block._class+' block using a '+self.block_class+' propagator.')

        if not isinstance(from_blocks, list) or not all(isinstance(x, SimpleNamespace) for x in from_blocks):
            raise ValueError('Expected from_blocks to be of type list[SimpleNamespace], not so for block connecting to '+block._id+'.')

        for from_block in from_blocks:
            if not hasattr(from_block, '_shape'):
                raise ValueError(self.block_class+' propagator expected from_block to include a _shape attribute.')
            shape_in = get_shape('out', from_block)
            if len(shape_in) < 1:
                raise ValueError('Input block requires to have at least one dimension, zero'
                                 'found for '+from_block._id+' -> '+block._id+'.')
            if shape_has_auto(shape_in):
                raise ValueError('Input block not allowed to have <<auto>> values in shape, '
                                 'found for '+from_block._id+' -> '+block._id+'.')

        check_output_size_dims(self.output_size_dims, self.block_class, block)

        if isinstance(self.num_input_blocks, int):
            if len(from_blocks) != self.num_input_blocks:
                raise ValueError('Blocks of type '+self.block_class+' only accepts '+str(self.num_input_blocks)+' input blocks.')

        if hasattr(block, '_shape') and block._shape != '<<auto>>':
            raise NotImplementedError('Propagation only supported for blocks with shape set as <<auto>>.')


    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        This base method should be implemented by all derived classes.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError('This method should be implemented by derived classes.')


    def final_checks(self, from_blocks, block):
        """Method that checks for problems after shapes have been propagated.

        This base method implements checking the output shape don't contain
        <<auto>> values and if there is only a single from_block, that the
        connecting shapes agree. Extensions of this method in derived classes
        should always call this base one.

        Args: from_blocks (list[SimpleNamaspace]): The input blocks. block
            (SimpleNamaspace): The block to propagate its shapes.
        """
        if shape_has_auto(get_shape('out', block)):
            raise ValueError('Unexpectedly after propagation block has <<auto>> values '
                             'in output shape, found for block '+block._id+'.')

        if len(from_blocks) == 1 and not shapes_agree(from_blocks[0], block):
            raise ValueError('Shapes do not agree for block '+from_blocks[0]._id+' connecting to block '+block._id+'.')


    def __call__(self, from_blocks, block, propagators=None):
        """Propagates shapes to the given block.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.
        """
        self.initial_checks(from_blocks, block)
        if self.requires_propagators:
            self.propagate(from_blocks, block, propagators)  # pylint: disable=too-many-function-args
        else:
            self.propagate(from_blocks, block)
        self.final_checks(from_blocks, block)
