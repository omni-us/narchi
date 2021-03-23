"""Base propagator class and related functions."""

import re
import inspect
from jsonargparse import Namespace, dict_to_namespace, namespace_to_dict
from copy import deepcopy
from typing import List
from ..schemas import auto_tag, block_validator
from ..sympy import is_valid_dim


gt_regex = re.compile('^>[0-9]+$')


def get_shape(key, shape):
    """Gets the shape list for a given key among {'in','out'}."""
    if isinstance(shape, Namespace) and hasattr(shape, '_shape'):
        shape = shape._shape
    if isinstance(shape, Namespace):
        shape = vars(shape)
    if isinstance(shape, list):
        return shape
    return shape[key]


def create_shape(shape_in, shape_out=None):
    """Creates a shape namespace with 'in' and 'out' attributes and copied shape arrays."""
    shape = {'in': deepcopy(shape_in),
             'out': shape_in if shape_out is None else shape_out}
    return dict_to_namespace(deepcopy(shape))


def set_shape_dim(key, shape, dim, val):
    """Sets a value for a given dimension, shape and key ('in' or 'out')."""
    get_shape(key, shape)[dim] = val


def shapes_agree(shape_from, shape_to):
    """Checks whether the output shape from a block agrees with input shape of another block."""
    return get_shape('out', shape_from) == get_shape('in', shape_to)


def shape_has_auto(shape):
    """Checks whether a shape has any <<auto>> values."""
    if isinstance(shape, str):
        shape = [shape]
    if any([x == auto_tag for x in shape]):
        return True
    return False


def check_output_feats_dims(output_feats_dims, block_class, block):
    """Checks the output_feats attribute of a block."""
    if output_feats_dims in {1, 2, 3}:
        if not hasattr(block, 'output_feats'):
            raise ValueError(f'{block_class} propagator expected block[id={block._id}] to include an output_feats attribute.')
        if output_feats_dims == 1 and not is_valid_dim(block.output_feats):
            raise ValueError(f'{block_class} propagator expected block[id={block._id}] output_feats to be a '
                             f'variable or an int larger than zero.')
        if output_feats_dims > 1 and (not isinstance(block.output_feats, list) or not all(is_valid_dim(x) for x in block.output_feats)):
            raise ValueError(f'{block_class} propagator expected block[id={block._id}] output_feats to be a '
                             f'list with {output_feats_dims} variables or ints larger than zero.')


class BasePropagator:
    """Base class for block shapes propagation."""

    block_class = None
    num_input_blocks = None
    output_feats_dims = False


    def __init__(self, block_class):
        """Initializer for BasePropagator instance.

        Args:
            block_class (str): The name of the block class being propagated.
        """
        self.block_class = block_class


    def initial_checks(self, from_blocks: List[Namespace], block: Namespace):
        """Method that does some initial checks before propagation.

        Extensions of this method in derived classes should always call this
        base method. This base method implements the following checks:

        - That the block class is the same as the one expected by the
          propagator.
        - That the input shapes don't contain any <<auto>> values.
        - If num_input_blocks is set and is an int, that there are exactly this
          number of input blocks.

        Args:
            from_blocks: The input blocks.
            block: The block to propagate its shapes.

        Raises:
            ValueError: If block fails to validate against schema.
            ValueError: If block already has a _shape attribute.
            ValueError: If block._class != block_class.
            ValueError: If input shape not present, invalid or contains <<auto>>.
            ValueError: If output_feats required by class and not present or invalid.
            ValueError: If len(from_blocks) != num_input_blocks.
        """
        try:
            block_validator.validate(namespace_to_dict(block))
        except Exception as ex:
            block_id = block._id if hasattr(block, '_id') else 'None'
            raise ValueError(f'Validation failed for block[id={block_id}] :: {ex}') from ex

        if hasattr(block, '_shape'):
            raise ValueError(f'Propagation only supported for blocks without a _shape attribute, '
                             f'found {block._shape} in block[id={block._id}].')

        if block._class != self.block_class:
            raise ValueError(f'Attempted to propagate block[id={block._id}] of class {block._class} using '
                             f'a {self.block_class} propagator.')

        if not isinstance(from_blocks, list) or not all(isinstance(x, Namespace) for x in from_blocks):
            raise ValueError(f'Expected from_blocks to be of type list[Namespace], not so for blocks '
                             f'connecting to block[id={block._id}].')

        for from_block in from_blocks:
            if not hasattr(from_block, '_shape'):
                raise ValueError(f'{self.block_class} propagator expected from_block[id={from_block._id}] to '
                                 f'include a _shape attribute.')
            shape_in = get_shape('out', from_block)
            if len(shape_in) < 1:
                raise ValueError(f'Input block requires to have at least one dimension, zero'
                                 f'found for block[id={from_block._id}] -> block[id={block._id}].')
            if shape_has_auto(shape_in):
                raise ValueError(f'Input block not allowed to have {auto_tag} values in shape, '
                                 f'found for block[id={from_block._id}] -> block[id={block._id}].')

        check_output_feats_dims(self.output_feats_dims, self.block_class, block)

        if self.num_input_blocks is not None:
            invalid = True
            if isinstance(self.num_input_blocks, int) and len(from_blocks) == self.num_input_blocks:
                invalid = False
            elif gt_regex.match(str(self.num_input_blocks)) and len(from_blocks) > int(self.num_input_blocks[1:]):
                invalid = False
            if invalid:
                raise ValueError(f'Blocks of class {self.block_class} only accept {self.num_input_blocks} input '
                                 f'blocks, found {len(from_blocks)} for block[id={block._id}].')


    def propagate(self, from_blocks: List[Namespace], block: Namespace):
        """Method that propagates shapes to a block.

        This base method should be implemented by all derived classes.

        Args:
            from_blocks: The input blocks.
            block: The block to propagate its shapes.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError('This method should be implemented by derived classes.')


    def final_checks(self, from_blocks: List[Namespace], block: Namespace):
        """Method that checks for problems after shapes have been propagated.

        This base method implements checking the output shape don't contain
        <<auto>> values and if there is only a single from_block, that the
        connecting shapes agree. Extensions of this method in derived classes
        should always call this base one.

        Args:
            from_blocks: The input blocks.
            block: The block to propagate its shapes.
        """
        if shape_has_auto(get_shape('out', block)):
            raise ValueError(f'Unexpectedly after propagation block has {auto_tag} values '
                             f'in output shape, found for block[id={block._id}].')

        if len(from_blocks) == 1 and not shapes_agree(from_blocks[0], block):
            raise ValueError(f'Shapes do not agree for block[id={from_blocks[0]._id}] connecting to block[id={block._id}].')


    def __call__(
        self,
        from_blocks: List[Namespace],
        block: Namespace,
        propagators: dict = None,
        ext_vars: dict = {},
        cwd: str = None
    ):
        """Propagates shapes to the given block.

        Args:
            from_blocks: The input blocks.
            block: The block to propagate its shapes.
            propagators: Dictionary of propagators.
            ext_vars: Dictionary of external variables required to load jsonnet.
            cwd: Working directory to resolve relative paths.
        """
        self.initial_checks(from_blocks, block)
        func_param = {x.name for x in inspect.signature(self.propagate).parameters.values()}
        kwargs = {}
        if 'propagators' in func_param:
            kwargs['propagators'] = propagators
        if 'ext_vars' in func_param:
            kwargs['ext_vars'] = ext_vars
        if 'cwd' in func_param:
            kwargs['cwd'] = cwd
        self.propagate(from_blocks, block, **kwargs)
        self.final_checks(from_blocks, block)
