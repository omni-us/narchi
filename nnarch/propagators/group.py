"""Propagator classes for groups of blocks."""

import inspect
from collections import OrderedDict
from .base import BasePropagator, get_shape, create_shape
from ..graph import parse_graph


def propagate_shapes(blocks, topological_predecessors, propagators, ext_vars, cwd, skip_ids=None):
    """Function that propagates shapes in blocks based on a connections mapping.

    Args:
        blocks (list[dict]): List of blocks to propagate.
        topological_predecessors (OrderedDict[str, list[str]]): Mapping of block IDs to its input blocks IDs.
        propagators (dict): Dictionary of propagators.
        skip_ids (set): Blocks that should be skipped in propagation.
        ext_vars (dict): Dictionary of external variables required to load jsonnet.
        cwd (str): Working directory to resolve relative paths.

    Raises:
        KeyError: If there are blocks with same id.
        ValueError: If no propagator found for some block.
    """
    blocks_dict = {}
    for block in blocks:
        if block._id in blocks:
            raise KeyError('Duplicate block id: '+block._id+'.')
        blocks_dict[block._id] = block

    if skip_ids is None:
        skip_ids = set()

    for node_to, nodes_from in topological_predecessors.items():
        if node_to in skip_ids:
            continue
        from_blocks = [blocks_dict[n] for n in nodes_from]
        if node_to not in blocks_dict:
            block_ids = {k for k in blocks_dict.keys()}
            raise KeyError('Graph references block[id='+node_to+'] which is not found among ids='+str(block_ids)+'.')
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


    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Calls the base class checks and makes sure that the block includes a
        blocks attribute with at least one item.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.

        Raises:
            ValueError: When blocks attribute is missing or not list with at least one item.
        """
        super().initial_checks(from_blocks, block)
        if not hasattr(block, 'blocks'):
            raise ValueError(block._class+' expected block to include a blocks attribute, not found in block[id='+block._id+'].')
        if not isinstance(block.blocks, list) or len(block.blocks) < 1:
            raise ValueError(block._class+' expected block.blocks to be a list with at least one item, not so in block[id='+block._id+'].')


    def propagate(self, from_blocks, block, propagators, ext_vars, cwd=None):
        """Method that propagates shapes in the given block.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.
            propagators (dict): Dictionary of propagators.
            ext_vars (dict): Dictionary of external variables required to load jsonnet.
            cwd (str): Working directory to resolve relative paths.

        Raises:
            ValueError: If no propagator found for some block.
        """
        topological_predecessors = OrderedDict()
        prev_block_id = from_blocks[0]._id
        for num, seq_block in enumerate(block.blocks):
            if not hasattr(seq_block, '_id'):
                seq_block._id = block._id+'_'+str(num)
            topological_predecessors[seq_block._id] = [prev_block_id]
            prev_block_id = seq_block._id
        propagate_shapes(from_blocks + block.blocks,
                         topological_predecessors,
                         propagators=propagators,
                         ext_vars=ext_vars,
                         cwd=cwd)
        in_shape = get_shape('out', from_blocks[0])
        out_shape = get_shape('out', block.blocks[-1])
        block._shape = create_shape(in_shape, out_shape)


class GroupPropagator(SequentialPropagator):
    """Propagator for a sequence of blocks."""

    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Calls the base class checks and makes sure that the block includes a
        graph attribute with at least one item.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.

        Raises:
            ValueError: When blocks attribute is missing or not list with at least one item.
        """
        super().initial_checks(from_blocks, block)
        if not hasattr(block, 'graph'):
            raise ValueError(block._class+' expected block to include a graph attribute, not found in block[id='+block._id+'].')
        if not hasattr(block, 'input') or not isinstance(block.input, str):
            raise ValueError(block._class+' expected block to include an input attribute, not found in block[id='+block._id+'].')
        if not hasattr(block, 'output') or not isinstance(block.output, str):
            raise ValueError(block._class+' expected block to include an output attribute, not found in block[id='+block._id+'].')


    def propagate(self, from_blocks, block, propagators, ext_vars, cwd=None):
        """Method that propagates shapes in the given block.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.
            propagators (dict): Dictionary of propagators.
            ext_vars (dict): Dictionary of external variables required to load jsonnet.
            cwd (str): Working directory to resolve relative paths.

        Raises:
            ValueError: If no propagator found for some block.
        """
        topological_predecessors = parse_graph(from_blocks, block)
        propagate_shapes(from_blocks + block.blocks,
                         topological_predecessors,
                         propagators=propagators,
                         ext_vars=ext_vars,
                         cwd=cwd)
        in_shape = get_shape('out', from_blocks[0])
        out_shape = get_shape('out', next(x for x in block.blocks if x._id==block.output))
        block._shape = create_shape(in_shape, out_shape)


propagators = [
    SequentialPropagator('Sequential'),
    GroupPropagator('Group'),
]
