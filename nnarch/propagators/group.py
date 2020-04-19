"""Propagator classes for groups of blocks."""

from collections import OrderedDict
from .base import BasePropagator, get_shape, create_shape
from ..graph import parse_graph


def propagate_shapes(blocks, in_nodes, propagators, skip_ids=None):
    """Function that propagates shapes in blocks based on a connections mapping.

    Arguments:
        blocks (list[dict]): List of blocks to propagate.
        in_nodes (OrderedDict[str, list[str]]): Mapping of block ID to its input blocks IDs.
        propagators (dict): Dictionary of propagators.
        skip_ids (set): Blocks that should be skipped in propagation.

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

    for node_to, nodes_from in in_nodes.items():
        if node_to in skip_ids:
            continue
        from_blocks = [blocks_dict[n] for n in nodes_from]
        block = blocks_dict[node_to]
        if block._class not in propagators:
            raise ValueError('No propagator found for block '+block._id+' of type '+block._class+'.')
        propagator = propagators[block._class]
        if propagator.requires_propagators:
            propagator(from_blocks, block, propagators)
        else:
            propagator(from_blocks, block)


class SequentialPropagator(BasePropagator):
    """Propagator for a sequence of blocks."""

    num_input_blocks = 1
    requires_propagators = True


    def initial_checks(self, from_blocks, block):
        """Method that does some initial checks before propagation.

        Calls the base class checks and makes sure that the block includes a
        blocks attribute with at least one item.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.

        Raises:
            ValueError: When blocks attribute is missing or not list with at least one item.
        """
        super().initial_checks(from_blocks, block)
        if not hasattr(block, 'blocks'):
            raise ValueError(block._class+' expected block to include a blocks attribute, not found in block '+block._id+'.')
        if not isinstance(block.blocks, list) or len(block.blocks) < 1:
            raise ValueError(block._class+' expected block.blocks to be a list with at least one item, not so in block '+block._id+'.')


    def propagate(self, from_blocks, block, propagators):
        """Method that propagates shapes in the given block.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.
            propagators (dict): Dictionary of propagators.

        Raises:
            ValueError: If no propagator found for some block.
        """
        in_nodes = OrderedDict()
        prev_block_id = from_blocks[0]._id
        for num, seq_block in enumerate(block.blocks):
            if not hasattr(seq_block, '_id'):
                seq_block._id = block._id+'.'+str(num)
            in_nodes[seq_block._id] = [prev_block_id]
            prev_block_id = seq_block._id
        propagate_shapes(from_blocks + block.blocks, in_nodes, propagators)
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
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.

        Raises:
            ValueError: When blocks attribute is missing or not list with at least one item.
        """
        super().initial_checks(from_blocks, block)
        if not hasattr(block, 'graph'):
            raise ValueError(block._class+' expected block to include a graph attribute, not found in block '+block._id+'.')
        if not isinstance(block.blocks, list) or len(block.blocks) < 1:
            raise ValueError(block._class+' expected block.blocks to be a list with at least one item, not so in block '+block._id+'.')
        if not hasattr(block, 'input') or not isinstance(block.input, str):
            raise ValueError(block._class+' expected block to include an input attribute, not found in block '+block._id+'.')
        if not hasattr(block, 'output') or not isinstance(block.output, str):
            raise ValueError(block._class+' expected block to include an output attribute, not found in block '+block._id+'.')


    def propagate(self, from_blocks, block, propagators):
        """Method that propagates shapes in the given block.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.
            propagators (dict): Dictionary of propagators.

        Raises:
            ValueError: If no propagator found for some block.
        """
        from_id = from_blocks[0]._id
        graph = [from_id+' -> '+block.input] + block.graph
        in_nodes = parse_graph(graph, from_id)
        propagate_shapes(from_blocks + block.blocks, in_nodes, propagators)
        in_shape = get_shape('out', from_blocks[0])
        out_shape = get_shape('out', block.blocks[block.output])
        block._shape = create_shape(in_shape, out_shape)


propagators = [
    SequentialPropagator('Sequential'),
    GroupPropagator('Group'),
]
