"""Functions and classes related to neural network module architectures."""

import json
from jsonargparse import SimpleNamespace, ActionJsonnet, Path, namespace_to_dict
from .schema import nnarch_validator
from .graph import parse_graph
from .propagators.base import BasePropagator, get_shape, create_shape, shapes_agree
from .propagators.group import propagate_shapes


class ModuleArchitecture:
    """Class for instantiating a ModuleArchitecture objects."""

    architecture = None
    ext_vars = None
    propagators = None
    propagated = False

    def __init__(self, architecture, ext_vars={}, propagators={}, propagate=True, validate=True):
        """Initializer for ModuleArchitecture class.

        Args:
            architecture (str or Path): Path to a jsonnet architecture file or jsonnet content.
            ext_vars (dict): Dictionary of external variables required to load the jsonnet.
            propagators (dict): Dictionary of propagators.
            propagate (bool): Whether to propagate dimensions on initialization.
        """
        self.ext_vars = ext_vars
        self.propagators = propagators

        ## Load jsonnet file or snippet ##
        if isinstance(architecture, (str, Path)):
            architecture = ActionJsonnet(schema=None).parse(architecture, ext_vars=ext_vars)
        if not isinstance(architecture, SimpleNamespace):
            raise ValueError(type(self).__name__+' expected architecture to be either a path or a namespace.')
        self.architecture = architecture

        ## Validate input ##
        if validate:
            self.validate('Input')

        ## Propagate shapes ##
        if propagate:
            self.propagate()


    def validate(self, source=''):
        """Validates the architecture against the nnarch schema."""
        try:
            nnarch_validator.validate(namespace_to_dict(self.architecture))
        except Exception as ex:
            raise type(ex)(source+' architecture failed to validate against nnarch schema :: '+str(ex))


    def propagate(self, validate=True):
        """Propagates the shapes of the neural network module architecture."""
        architecture = self.architecture
        ext_vars = self.ext_vars
        propagators = self.propagators

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
        if (not hasattr(pre_output_block, '_shape') or pre_output_block._shape == '<<auto>>') and not hasattr(pre_output_block, 'output_size'):
            pre_output_block.output_size = output_block._shape[-1]

        ## Propagate shapes for the architecture blocks ##
        blocks = propagate_shapes(architecture.inputs + architecture.blocks, topological_predecessors, propagators, skip_ids={output_block._id})

        ## Automatic output dimensions ##
        for dim, val in enumerate(output_block._shape):
            if val == '<<auto>>':
                output_block._shape[dim] = get_shape('out', pre_output_block)[dim]

        ## Check that output shape agrees ##
        if not shapes_agree(pre_output_block, output_block):
            raise ValueError('Output shape does not agree: '+str(pre_output_block._shape.out)+' vs. '+str(output_block._shape))

        ## Validate result ##
        if validate:
            self.validate('Propagated')

        ## Set propagated shape ##
        in_shape = architecture.inputs[0]._shape
        out_shape = architecture.outputs[0]._shape
        architecture._shape = create_shape(in_shape, out_shape)

        ## Update properties ##
        self.blocks = blocks
        self.topological_predecessors = topological_predecessors
        self.propagated = True


    def write_json(self, json_path):
        """Writes the current state of the architecture in json format to the given path."""
        with open(json_path, 'w') as f:
            f.write(json.dumps(namespace_to_dict(self.architecture), indent=2, sort_keys=True))


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
        module = ModuleArchitecture(block.path, propagators=propagators)
        block._shape = module.architecture._shape


propagators = [
    ModulePropagator('Module'),
]
