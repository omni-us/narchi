"""Propagator classes for reshaping."""

from jsonargparse import Namespace
from jsonargparse import namespace_to_dict as n2d
from .base import BasePropagator, get_shape, create_shape
from ..sympy import prod, divide
from ..schemas import auto_tag, reshape_validator


def check_reshape_spec(reshape_spec):
    """Checks that reshape_spec is valid according to schema, indexes range is valid and there is at most one <<auto>> in each unflatten."""
    reshape_validator.validate(reshape_spec)
    idxs = []
    if reshape_spec != 'flatten':
        for val in reshape_spec:
            if isinstance(val, (int, str)):
                idxs.append(val)
            elif isinstance(val, list):
                idxs.extend([x for x in val])
            else:
                idx = next(iter(val.keys()))
                idxs.append(int(idx))
                if sum([x == auto_tag for x in val[idx]]) > 1:
                    raise ValueError(f'At most one {auto_tag} is allowed in unflatten definition ({val[idx]}).')
        if sorted(idxs) != list(range(len(idxs))):
            raise ValueError(f'Invalid indexes range ({sorted(idxs)}) in reshape_spec.')
    return idxs


def norm_reshape_spec(reshape_spec):
    """Converts elements of a reshape_spec from Namespace to dict."""
    if isinstance(reshape_spec, str):
        return reshape_spec
    return [n2d(x) if isinstance(x, Namespace) else x for x in reshape_spec]


class ReshapePropagator(BasePropagator):
    """Propagator for reshapping which could involve any of: permute, flatten and unflatten."""

    num_input_blocks = 1


    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Calls the base class checks and makes sure that the reshape_spec
        attribute is valid and agrees with the input dimensions.

        Args:
            from_blocks (list[Namespace]): The input blocks.
            block (Namespace): The block to propagate its shapes.

        Raises:
            ValueError: When block does not have a valid reshape_spec attribute that agrees with input dimensions.
        """
        super().initial_checks(from_blocks, block)
        if block.reshape_spec != 'flatten':
            reshape_spec = norm_reshape_spec(block.reshape_spec)
            try:
                idxs = check_reshape_spec(reshape_spec)
            except Exception as ex:
                raise ValueError(f'Invalid reshape_spec attribute in block[id={block._id}] :: {ex}')
            shape_in = get_shape('out', from_blocks[0])
            if len(idxs) != len(shape_in):
                raise ValueError(f'Number of dimensions indexes in reshape_spec attribute of block[id={block._id}] does '
                                 f'not agree with the input dimensions coming from block[id={from_blocks[0]._id}].')


    def propagate(self, from_blocks, block):
        """Method that propagates shapes to a block.

        Args:
            from_blocks (list[Namespace]): The input blocks.
            block (Namespace): The block to propagate its shapes.
        """
        shape_in = get_shape('out', from_blocks[0])
        shape_out = []
        if block.reshape_spec == 'flatten':
            reshape_spec = [[n for n in range(len(shape_in))]]
        else:
            reshape_spec = norm_reshape_spec(block.reshape_spec)
        for val in reshape_spec:
            if isinstance(val, int):
                shape_out.append(shape_in[val])
            elif isinstance(val, list):
                shape_out.append(prod([shape_in[x] for x in val]))
            elif isinstance(val, dict):
                idx = next(iter(val.keys()))
                in_dim = shape_in[int(idx)]
                dims = val[idx]
                if any(x == auto_tag for x in dims):
                    auto_idx = dims.index(auto_tag)
                    nonauto = prod([x for x in dims if x != auto_tag])
                    dims[auto_idx] = divide(in_dim, nonauto)
                shape_out.extend(dims)
        block._shape = create_shape(shape_in, shape_out)
