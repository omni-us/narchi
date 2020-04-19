"""Functions and classes related to neural network module architectures."""

from collections import OrderedDict
from jsonargparse import SimpleNamespace, ActionJsonnet, Path, namespace_to_dict
from pygraphviz import AGraph
from networkx.drawing.nx_agraph import from_agraph
from networkx.algorithms.dag import is_directed_acyclic_graph
from networkx.algorithms.traversal.edgebfs import edge_bfs
from .schema import nnarch_validator
from .propagators.base import BasePropagator, get_shape, create_shape, shapes_agree


def get_nodes_with_inputs(graph, source):
    """Traverses a graph creating an OrderedDict of nodes with respective inputs."""
    in_nodes = OrderedDict()
    for node_from, node_to, _ in edge_bfs(graph, source=source):
        if node_to not in in_nodes:
            in_nodes[node_to] = []
        in_nodes[node_to].append(node_from)
    return in_nodes


def load_module_architecture(architecture, ext_vars={}, propagators={}):
    """Loads a neural network module architecture.

    Args:
        architecture (SimpleNamespace or str): A parsed architecture namespace object or path to a jsonnet architecture file.
        ext_vars (dict or SimpleNamespace): Dictionary of external variables required to load the jsonnet.

    Returns:
        dict: A dictionary with elements:
            1) architecture: an architecture object with all shapes propagated.
            2) blocks: an ordered dictionary (as defined in architecture) of the network blocks including inputs and outputs.
            3) in_nodes: an ordered dictionary (by graph traversal) of network block IDs mapping to its inputs.
    """
    ## Load file or snippet or make copy of object ##
    if isinstance(architecture, (str, Path)):
        architecture = ActionJsonnet(schema=None).parse(architecture, ext_vars=ext_vars)

    ## Validate input ##
    try:
        nnarch_validator.validate(namespace_to_dict(architecture))
    except Exception as ex:
        raise type(ex)('Input architecture does not validate against nnarch schema :: '+str(ex))

    ## Check if supported ##
    if len(architecture.inputs) != 1:
        raise NotImplementedError('Architectures with more than one input not yet implemented.')
    if len(architecture.outputs) != 1:
        raise NotImplementedError('Architectures with more than one output not yet implemented.')

    ## Parse graph ##
    try:
        graph = from_agraph(AGraph('\n'.join(['digraph {']+architecture.graph+['}'])))
    except Exception as ex:
        raise ValueError('Problems parsing architecture graph: '+str(ex))
    if not is_directed_acyclic_graph(graph):
        raise ValueError('Expected architecture graph to be directed and acyclic.')

    ## Traverse graph ##
    try:
        in_nodes = get_nodes_with_inputs(graph, source=architecture.inputs[0]._id)
    except Exception as ex:
        raise ValueError('Problems traversing architecture graph: '+str(ex))
    if len(in_nodes) != graph.number_of_nodes()-len(architecture.inputs):
        raise ValueError('Graph traversal does not include all nodes: '+str(in_nodes))
    if next(reversed(in_nodes)) != architecture.outputs[0]._id:
        raise ValueError('Expected output node to be the last in the graph.')

    ## Create dictionary of blocks ##
    blocks = OrderedDict()
    input_node = architecture.inputs[0]._id
    blocks[input_node] = architecture.inputs[0]
    for block in architecture.blocks:
        if block._id in blocks:
            raise KeyError('Duplicate block id: '+block._id+'.')
        blocks[block._id] = block
    output_node = architecture.outputs[0]._id
    blocks[output_node] = architecture.outputs[0]

    ## Propagate output features to pre-output block ##
    pre_output_block = blocks[in_nodes[output_node][0]]
    output_block = blocks[output_node]
    if (not hasattr(pre_output_block, '_shape') or pre_output_block._shape == '<<auto>>') and not hasattr(pre_output_block, 'out_features'):
        pre_output_block.out_features = output_block._shape[-1]

    ## Propagate shapes for the architecture blocks ##
    for node_to, nodes_from in in_nodes.items():
        if node_to != architecture.outputs[0]._id:
            from_blocks = [blocks[n] for n in nodes_from]
            block = blocks[node_to]
            if block._class not in propagators:
                raise ValueError('No propagator found for block '+block._id+' of type '+block._class+'.')
            propagator = propagators[block._class]
            if propagator.requires_propagators:
                propagator(from_blocks, block, propagators)
            else:
                propagator(from_blocks, block)

    ## Automatic output dimensions ##
    for dim, val in enumerate(output_block._shape):
        if val == '<<auto>>':
            output_block._shape[dim] = get_shape('out', pre_output_block)[dim]

    ## Check that output shape agrees ##
    if not shapes_agree(pre_output_block, output_block):
        raise ValueError('Output shape does not agree: '+str(pre_output_block._shape['out'])+' vs. '+str(output_block._shape))

    ## Validate result ##
    try:
        nnarch_validator.validate(namespace_to_dict(architecture))
    except Exception as ex:
        raise type(ex)('Propagated architecture does not validate against nnarch schema :: '+str(ex))

    ## Set propagated shape ##
    in_shape = architecture.inputs[0]._shape
    out_shape = architecture.outputs[0]._shape
    architecture._shape = create_shape(in_shape, out_shape)

    return SimpleNamespace(**{
        'architecture': architecture,
        'blocks': blocks,
        'in_nodes': in_nodes,
    })


class ModulePropagator(BasePropagator):
    """Propagator for complete modules."""

    num_input_blocks = 1
    requires_propagators = True


    def propagate(self, from_blocks, block, propagators):
        """Method that propagates shapes through a module.

        Args:
            from_blocks (list[SimpleNamaspace]): The input blocks.
            block (SimpleNamaspace): The block to propagate its shapes.
            propagators (dict): Dictionary of propagators.

        Raises:
            ValueError: If no propagator found for some block.
        """
        module = load_module_architecture(block.path, propagators=propagators)
        block._shape = module.architecture._shape


propagators = [
    ModulePropagator('Module'),
]
