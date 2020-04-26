"""Functions and classes related to neural network module architectures."""

import os
import json
from copy import deepcopy
from jsonargparse import SimpleNamespace, ActionJsonnet, Path, namespace_to_dict, config_read_mode
from .schema import nnarch_validator
from .graph import parse_graph
from .propagators.base import BasePropagator, get_shape, create_shape, shapes_agree
from .propagators.group import propagate_shapes


class ModuleArchitecture:
    """Class for instantiating a ModuleArchitecture objects."""

    path = None
    cwd = None
    architecture = None
    ext_vars = None
    propagators = None
    propagated = False

    def __init__(self, architecture, ext_vars=None, cwd=None, propagators={}, propagate=True, validate=True):
        """Initializer for ModuleArchitecture class.

        Args:
            architecture (str or Path): Path to a jsonnet architecture file or jsonnet content.
            ext_vars (SimpleNamespace): External variables required to load jsonnet.
            cwd (str or None): Working directory to resolve relative paths.
            propagators (dict): Dictionary of propagators.
            propagate (bool): Whether to propagate dimensions on initialization.
            validate (bool): Whether to validate against nnarch schema.
        """
        self.ext_vars = ext_vars
        self.propagators = propagators

        ## Load jsonnet file or snippet ##
        if isinstance(architecture, (str, Path)):
            path = Path(architecture, cwd=cwd) if isinstance(architecture, str) else architecture
            if cwd is None:
                cwd = os.path.dirname(path())
            self.path = Path(architecture, mode=config_read_mode, cwd=cwd)
            self.cwd = cwd
            architecture = ActionJsonnet(schema=None).parse(self.path, ext_vars=ext_vars)
            if not hasattr(architecture, '_id'):
                architecture._id = os.path.splitext(os.path.basename(self.path()))[0]
        if not isinstance(architecture, SimpleNamespace):
            raise ValueError(type(self).__name__+' expected architecture to be either a path or a namespace.')
        self.architecture = architecture

        ## Validate input ##
        if validate:
            self.validate('Input')

        ## Propagate shapes ##
        if propagate:
            self.propagate(propagators, ext_vars, cwd)


    def validate(self, source=''):
        """Validates the architecture against the nnarch schema."""
        try:
            nnarch_validator.validate(namespace_to_dict(self.architecture))
        except Exception as ex:
            raise type(ex)(source+' architecture failed to validate against nnarch schema :: '+str(ex))


    def propagate(self, propagators=None, ext_vars=None, cwd=None, validate=True):
        """Propagates the shapes of the neural network module architecture.

        Args:
            propagators (dict or None): Dictionary of propagators. Set None to use the ones provided at init.
            ext_vars (SimpleNamespace or None): External variables required to load jsonnet. Set None to use the ones provided at init.
            cwd (str or None): Working directory to resolve relative paths. Set None to use the one provided at init.
            validate (bool): Whether to validate against the nnarch schema.
        """
        if self.propagated:
            raise RuntimeError('Not possible to propagate an already propagated '+type(self).__name__+'.')

        architecture = self.architecture
        if propagators is None:
            propagators = self.propagators
        if ext_vars is None:
            ext_vars = self.ext_vars
        if cwd is None:
            cwd = self.cwd

        ## Check if supported ##
        if len(architecture.inputs) != 1:
            raise NotImplementedError('Architectures with more than one input not yet implemented.')
        if len(architecture.outputs) != 1:
            raise NotImplementedError('Architectures with more than one output not yet implemented.')

        ## Parse graph getting node mapping in topological order ##
        topological_predecessors = parse_graph(architecture.inputs, architecture)
        if next(reversed(topological_predecessors)) != architecture.outputs[0]._id:
            raise ValueError('In module[id='+architecture._id+'] expected output node '+architecture.outputs[0]._id+' to be the last in the graph.')

        ## Get output and pre-output blocks ##
        output_block = architecture.outputs[0]
        pre_output_block_id = next(v[0] for k, v in topological_predecessors.items() if k == output_block._id)
        try:
            pre_output_block = next(b for b in architecture.blocks if b._id == pre_output_block_id)
        except StopIteration:
            block_ids = {b._id for b in architecture.blocks}
            raise ValueError('In module[id='+architecture._id+'] pre-output block[id='+pre_output_block_id+'] not found among ids='+str(block_ids)+'.')

        ## Propagate shapes for the architecture blocks ##
        blocks = propagate_shapes(architecture.inputs + architecture.blocks,
                                  topological_predecessors,
                                  propagators=propagators,
                                  ext_vars=ext_vars,
                                  cwd=cwd,
                                  skip_ids={output_block._id})

        ## Automatic output dimensions ##
        for dim, val in enumerate(output_block._shape):
            if val == '<<auto>>':
                output_block._shape[dim] = get_shape('out', pre_output_block)[dim]

        ## Check that output shape agrees ##
        if not shapes_agree(pre_output_block, output_block):
            raise ValueError('In module[id='+architecture._id+'] output shape does not agree: '+str(pre_output_block._shape.out)+' vs. '+str(output_block._shape))

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


    def propagate(self, from_blocks, block, propagators, ext_vars, cwd):
        """Method that propagates shapes through a module.

        Args:
            from_blocks (list[SimpleNamespace]): The input blocks.
            block (SimpleNamespace): The block to propagate its shapes.
            propagators (dict): Dictionary of propagators.
            ext_vars (SimpleNamespace): External variables required to load jsonnet.
            cwd (str): Working directory to resolve relative paths.

        Raises:
            ValueError: If no propagator found for some block.
        """
        block_ext_vars = deepcopy(ext_vars)
        if ext_vars is None:
            block_ext_vars = SimpleNamespace()
        elif isinstance(ext_vars, dict):
            block_ext_vars = SimpleNamespace(**block_ext_vars)
        if hasattr(block, 'ext_vars'):
            vars(block_ext_vars).update(vars(block.ext_vars))
        module = ModuleArchitecture(block.path, propagators=propagators, ext_vars=block_ext_vars, cwd=cwd)
        block._shape = module.architecture._shape
        block.architecture = module.architecture


propagators = [
    ModulePropagator('Module'),
]
