"""Functions and classes related to neural network module architectures."""

from jsonargparse import SimpleNamespace, ActionJsonnet, Path, namespace_to_dict
from .schema import nnarch_validator
from .graph import parse_graph
from .propagators.base import BasePropagator, get_shape, create_shape, shapes_agree
from .propagators.group import propagate_shapes


def load_module_architecture(architecture, ext_vars={}, propagators={}):
    """Loads a neural network module architecture.

    Args:
        architecture (SimpleNamespace or str): A parsed architecture namespace object or path to a jsonnet architecture file.
        ext_vars (dict or SimpleNamespace): Dictionary of external variables required to load the jsonnet.

    Returns:
        dict: A dictionary with elements:
            1) architecture: an architecture object with all shapes propagated.
            2) blocks: an ordered dictionary (as defined in architecture) of the network blocks including inputs and outputs.
            3) topological_predecessors: an ordered dictionary (by graph traversal) of network block IDs mapping to its inputs.
    """
    ## Load jsonnet file or snippet ##
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

    ## Parse graph getting node mapping in topological order ##
    topological_predecessors = parse_graph(architecture.inputs, architecture)
    if next(reversed(topological_predecessors)) != architecture.outputs[0]._id:
        raise ValueError('Expected output node to be the last in the graph.')

    ## Propagate output features to pre-output block ##
    output_block = architecture.outputs[0]
    pre_output_block_id = next(v[0] for k, v in topological_predecessors.items() if k == output_block._id)
    pre_output_block = next(b for b in architecture.blocks if b._id == pre_output_block_id)
    if (not hasattr(pre_output_block, '_shape') or pre_output_block._shape == '<<auto>>') and not hasattr(pre_output_block, 'out_features'):
        pre_output_block.out_features = output_block._shape[-1]

    ## Propagate shapes for the architecture blocks ##
    blocks = propagate_shapes(architecture.inputs + architecture.blocks, topological_predecessors, propagators, skip_ids={output_block._id})

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
        'topological_predecessors': topological_predecessors,
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
