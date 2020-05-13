"""Propagator classes for groups of blocks."""

import re
import inspect
from collections import OrderedDict
from .base import BasePropagator, get_shape, create_shape
from ..graph import parse_graph
from ..schemas import id_separator


def get_blocks_dict(blocks):
    """Function that creates a dictionary of blocks using _id as keys.

    Args:
        blocks (list[dict]): List of blocks objects.

    Returns:
        dict[str, dict]: Dictionary of blocks.
    """
    blocks_dict = {}
    for block in blocks:
        if block._id in blocks_dict:
            raise ValueError('Duplicate block id: '+block._id+'.')
        blocks_dict[block._id] = block
    return blocks_dict


def add_ids_prefix(block, io_blocks, skip_io=True):
    """Adds to block id a prefix consisting of parent id and separator as defined in propagated schema."""
    prefix = block._id + id_separator
    for num, subblock in enumerate(block.blocks):
        if hasattr(block, '_class') and block._class == 'Sequential' and not hasattr(subblock, '_id'):
            subblock._id = prefix + str(num)
        else:
            subblock._id = prefix + subblock._id
    if hasattr(block, 'input'):
        block.input = prefix + block.input
    if hasattr(block, 'output'):
        block.output = prefix + block.output
    if hasattr(block, 'inputs') and not skip_io:
        for node in block.inputs:
            node._id = prefix + node._id
    if hasattr(block, 'outputs') and not skip_io:
        for node in block.outputs:
            node._id = prefix + node._id
    if hasattr(block, 'graph'):
        skip_ids = set() if skip_io else {b._id for b in io_blocks}
        re_nodes = re.compile(' +-> +')
        for num, graph_line in enumerate(block.graph):
            nodes = re_nodes.split(graph_line)
            nodes = [n if n in skip_ids else prefix+n for n in nodes]
            block.graph[num] = ' -> '.join(nodes)


def propagate_shapes(blocks_dict, topological_predecessors, propagators, ext_vars, cwd, skip_ids=None):
    """Function that propagates shapes in blocks based on a connections mapping.

    Args:
        blocks_dict (dict[str, dict]): Dictionary of blocks.
        topological_predecessors (OrderedDict[str, list[str]]): Mapping of block IDs to its input blocks IDs.
        propagators (dict): Dictionary of propagators.
        skip_ids (set): Blocks that should be skipped in propagation.
        ext_vars (dict): Dictionary of external variables required to load jsonnet.
        cwd (str): Working directory to resolve relative paths.

    Raises:
        ValueError: If there graph references an undefined block.
        ValueError: If no propagator found for some block.
    """
    if skip_ids is None:
        skip_ids = set()

    for node_to, nodes_from in topological_predecessors.items():
        if node_to in skip_ids:
            continue
        from_blocks = [blocks_dict[n] for n in nodes_from]
        if node_to not in blocks_dict:
            block_ids = {k for k in blocks_dict.keys()}
            raise ValueError('Graph references block[id='+node_to+'] which is not found among ids='+str(block_ids)+'.')
        block = blocks_dict[node_to]
        if block._class not in propagators:
            raise ValueError('No propagator found for block[id='+block._id+'] of type '+block._class+'.')
        propagator = propagators[block._class]
        func_param = {x.name for x in inspect.signature(propagator).parameters.values()}
        kwargs = {}
        if 'propagators' in func_param:
            kwargs['propagators'] = propagators
        if 'ext_vars' in func_param:
            kwargs['ext_vars'] = ext_vars
        if 'cwd' in func_param:
            kwargs['cwd'] = cwd
        propagator(from_blocks, block, **kwargs)

    return blocks_dict


class SequentialPropagator(BasePropagator):
    """Propagator for a sequence of blocks."""

    num_input_blocks = 1


    def propagate(self, from_blocks, block, propagators, ext_vars, cwd=None):
        """Method that propagates shapes in the given block.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.
            propagators (dict): Dictionary of propagators.
            ext_vars (dict): Dictionary of external variables required to load jsonnet.
            cwd (str): Working directory to resolve relative paths.

        Raises:
            ValueError: If there are multiple blocks with the same id.
            ValueError: If no propagator found for some block.
        """
        add_ids_prefix(block, from_blocks)
        blocks = get_blocks_dict(from_blocks + block.blocks)
        topological_predecessors = parse_graph(from_blocks, block)
        try:
            propagate_shapes(blocks,
                             topological_predecessors,
                             propagators=propagators,
                             ext_vars=ext_vars,
                             cwd=cwd)
        except Exception as ex:
            raise type(ex)('block[id='+block._id+']: '+str(ex))
        in_shape = get_shape('out', from_blocks[0])
        out_shape = get_shape('out', block.blocks[-1])
        block._shape = create_shape(in_shape, out_shape)


class GroupPropagator(SequentialPropagator):
    """Propagator for a sequence of blocks."""

    def propagate(self, from_blocks, block, propagators, ext_vars, cwd=None):
        """Method that propagates shapes in the given block.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.
            propagators (dict): Dictionary of propagators.
            ext_vars (dict): Dictionary of external variables required to load jsonnet.
            cwd (str): Working directory to resolve relative paths.

        Raises:
            ValueError: If there are multiple blocks with the same id.
            ValueError: If there graph references an undefined block.
            ValueError: If no propagator found for some block.
        """
        add_ids_prefix(block, from_blocks)
        blocks = get_blocks_dict(from_blocks + block.blocks)
        topological_predecessors = parse_graph(from_blocks, block)
        try:
            propagate_shapes(blocks,
                             topological_predecessors,
                             propagators=propagators,
                             ext_vars=ext_vars,
                             cwd=cwd)
        except Exception as ex:
            raise type(ex)('block[id='+block._id+']: '+str(ex))
        in_shape = get_shape('out', from_blocks[0])
        out_shape = get_shape('out', next(x for x in block.blocks if x._id==block.output))
        block._shape = create_shape(in_shape, out_shape)
