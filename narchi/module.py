"""Classes related to neural network module architectures."""

import os
import json
from copy import deepcopy
from jsonargparse import ArgumentParser, SimpleNamespace, Path, config_read_mode, namespace_to_dict, \
                         ActionConfigFile, ActionJsonnet, ActionJsonnetExtVars, ActionPath
from .schemas import narchi_validator, propagated_validator
from .graph import parse_graph
from .propagators.base import BasePropagator, get_shape, create_shape, shapes_agree
from .propagators.group import get_blocks_dict, propagate_shapes, add_ids_prefix
from .instantiators.common import import_object
from . import __version__


class ModuleArchitecture:
    """Class for instantiating ModuleArchitecture objects."""

    path = None
    jsonnet = None
    architecture = None
    propagators = None
    blocks = None
    topological_predecessors = None


    @staticmethod
    def get_config_parser():
        """Returns a ModuleArchitecture configuration parser."""
        parser = ArgumentParser(
            description=ModuleArchitecture.__doc__,
            version=__version__)
        parser.add_argument('--cfg',
            action=ActionConfigFile,
            help='Path to a configuration file.')

        # loading options #
        group_load = parser.add_argument_group('Loading related options')
        group_load.add_argument('--validate',
            default=True,
            type=bool,
            help='Whether to validate architecture against narchi schema.')
        group_load.add_argument('--propagate',
            default=True,
            type=bool,
            help='Whether to propagate shapes in architecture.')
        group_load.add_argument('--propagated',
            default=False,
            type=bool,
            help='Whether architecture has already been propagated.')
        group_load.add_argument('--propagators',
            choices=[None, 'default'],
            help='Whether to set or not the default propagators.')
        group_load.add_argument('--ext_vars',
            action=ActionJsonnetExtVars(),
            help='External variables required to load jsonnet.')
        group_load.add_argument('--cwd',
            help='Current working directory to load inner referenced files. Default None uses '
                 'directory of main architecture file.')
        group_load.add_argument('--parent_id',
            default='',
            help='Identifier of parent module.')

        # output options #
        group_out = parser.add_argument_group('Output related options')
        group_out.add_argument('--overwrite',
            default=False,
            type=bool,
            help='Whether to overwrite existing files.')
        group_out.add_argument('--outdir',
            default='.',
            action=ActionPath(mode='dw'),
            help='Directory where to write output files.')
        group_out.add_argument('--save_json',
            default=False,
            type=bool,
            help='Whether to write the architecture (up to the last successful step: jsonnet load, '
                 'schema validation, parsing) in json format to the output directory.')

        return parser


    def __init__(self, architecture=None, cfg=None, parser=None):
        """Initializer for ModuleArchitecture class.

        Args:
            architecture (str or Path or None): Path to a jsonnet architecture file.
            cfg (str or dict or SimpleNamespace): Path to config file or config object.
            parser (jsonargparse.ArgumentParser): Parser object in case it is an extension of get_config_parser().
        """
        if parser is None:
            parser = self.get_config_parser()
        self.parser = parser
        self.apply_config(cfg)

        if architecture is not None:
            self.load_architecture(architecture)


    def apply_config(self, cfg):
        """Applies a configuration to the ModuleArchitecture instance.

        Args:
            cfg (str or dict or SimpleNamespace): Path to config file or config object.
        """
        if cfg is None:
            self.cfg = self.parser.get_defaults()
        elif isinstance(cfg, (str, Path)):
            self.cfg_file = cfg
            self.cfg = self.parser.parse_path(cfg)
        elif isinstance(cfg, SimpleNamespace):
            self.parser.check_config(cfg)
            self.cfg = cfg
        elif isinstance(cfg, dict):
            cfg = dict(cfg)
            if 'propagators' in cfg and isinstance(cfg['propagators'], dict):
                self.propagators = cfg.pop('propagators')
            if not hasattr(self, 'cfg'):
                self.cfg = self.parser.parse_object(cfg)
            else:
                self.cfg = self.parser.parse_object(cfg, cfg_base=self.cfg, defaults=False)
        else:
            raise ValueError('Unexpected configuration object: '+str(cfg))

        if self.cfg.propagators == 'default':
            self.propagators = import_object('narchi.blocks.propagators')


    def load_architecture(self, architecture):
        """Loads an architecture file.

        Args:
            architecture (str or Path or None): Path to a jsonnet architecture file.
        """
        self.path = None
        self.jsonnet = None
        self.architecture = None
        self.blocks = None
        self.topological_predecessors = None

        ## Load jsonnet file or snippet ##
        if isinstance(architecture, (str, Path)):
            self.path = Path(architecture, mode=config_read_mode, cwd=self.cfg.cwd)
            self.cfg.cwd = os.path.dirname(self.path())
            self.jsonnet = self.path.get_content()
            architecture = ActionJsonnet(schema=None).parse(self.path, ext_vars=self.cfg.ext_vars)
            if not hasattr(architecture, '_id'):
                architecture._id = os.path.splitext(os.path.basename(self.path()))[0]
        if not isinstance(architecture, SimpleNamespace):
            raise ValueError(type(self).__name__+' expected architecture to be either a path or a namespace.')
        self.architecture = architecture

        ## Validate prior to propagation ##
        self.validate()

        ## Create dictionary of blocks ##
        if all(hasattr(architecture, x) for x in ['inputs', 'outputs', 'blocks']):
            if self.cfg.parent_id:
                architecture._id = self.cfg.parent_id
                add_ids_prefix(architecture, architecture.inputs+architecture.outputs, skip_io=False)
            self.blocks = get_blocks_dict(architecture.inputs + architecture.blocks)

        ## Propagate shapes ##
        if self.cfg.propagate:
            if not self.cfg.propagated:
                self.propagate()
            elif self.topological_predecessors is None:
                self.topological_predecessors = parse_graph(architecture.inputs, architecture)

    def validate(self):
        """Validates the architecture against the narchi or propagated schema."""
        if not self.cfg.validate:
            return
        try:
            if self.cfg.propagated:
                propagated_validator.validate(namespace_to_dict(self.architecture))
            else:
                narchi_validator.validate(namespace_to_dict(self.architecture))
        except Exception as ex:
            self.write_json_outdir()
            source = 'Propagated' if self.cfg.propagated else 'Pre-propagated'
            raise type(ex)(source+' architecture failed to validate against schema :: '+str(ex))


    def propagate(self):
        """Propagates the shapes of the neural network module architecture."""
        if self.cfg.propagated:
            raise RuntimeError('Not possible to propagate an already propagated '+type(self).__name__+'.')
        if self.propagators is None:
            raise RuntimeError('No propagators configured.')

        architecture = self.architecture

        ## Parse graph getting node mapping in topological order ##
        topological_predecessors = parse_graph(architecture.inputs, architecture)
        output_ids = {b._id for b in architecture.outputs}
        if next(reversed(topological_predecessors)) not in output_ids:
            raise ValueError('In module[id='+architecture._id+'] expected one of output nodes '+str(output_ids)+' to be the last in the graph.')

        ## Propagate shapes for the architecture blocks ##
        try:
            propagate_shapes(self.blocks,
                            topological_predecessors,
                            propagators=self.propagators,
                            ext_vars=self.cfg.ext_vars,
                            cwd=self.cfg.cwd,
                            skip_ids=output_ids)
        except Exception as ex:
            self.write_json_outdir()
            raise ex

        for output_block in architecture.outputs:
            ## Get pre-output blocks ##
            pre_output_block_id = next(v[0] for k, v in topological_predecessors.items() if k == output_block._id)
            try:
                pre_output_block = next(b for b in architecture.blocks if b._id == pre_output_block_id)
            except StopIteration:
                block_ids = {b._id for b in architecture.blocks}
                raise ValueError('In module[id='+architecture._id+'] pre-output block[id='+pre_output_block_id+'] not found among ids='+str(block_ids)+'.')

            ## Automatic output dimensions ##
            for dim, val in enumerate(output_block._shape):
                if val == '<<auto>>':
                    output_block._shape[dim] = get_shape('out', pre_output_block)[dim]

            ## Check that output shape agrees ##
            if not shapes_agree(pre_output_block, output_block):
                self.write_json_outdir()
                raise ValueError('In module[id='+architecture._id+'] pre-output block[id='+pre_output_block._id+'] and output '
                                'shape do not agree: '+str(pre_output_block._shape.out)+' vs. '+str(output_block._shape)+'.')

        ## Update properties ##
        self.topological_predecessors = topological_predecessors
        self.cfg.propagated = True

        ## Set propagated shape ##
        in_shape = architecture.inputs[0]._shape
        out_shape = architecture.outputs[0]._shape
        architecture._shape = create_shape(in_shape, out_shape)

        ## Validate result ##
        self.validate()

        ## Write json file if requested ##
        self.write_json_outdir()


    def write_json(self, json_path):
        """Writes the current state of the architecture in json format to the given path."""
        with open(json_path if isinstance(json_path, str) else json_path(), 'w') as f:
            architecture = namespace_to_dict(self.architecture)
            f.write(json.dumps(architecture,
                               indent=2,
                               sort_keys=True,
                               ensure_ascii=False))


    def _check_overwrite(self, path):
        """Raises IOError if overwrite not set and path already exists."""
        if not self.cfg.overwrite and os.path.isfile(path):
            raise IOError('Refusing to overwrite existing file: '+path)


    def write_json_outdir(self):
        """Writes the current state of the architecture in to the configured output directory."""
        if not self.cfg.save_json or self.cfg.outdir is None or not hasattr(self, 'architecture'):
            return
        outdir = self.cfg.outdir if isinstance(self.cfg.outdir, str) else self.cfg.outdir()
        out_path = os.path.join(outdir, self.architecture._id + '.json')
        self._check_overwrite(out_path)
        self.write_json(out_path)


class ModulePropagator(BasePropagator):
    """Propagator for complete modules."""

    num_input_blocks = 1


    def propagate(self, from_blocks, block, propagators=None, ext_vars={}, cwd=None):
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
        if hasattr(block, '_ext_vars'):
            vars(block_ext_vars).update(vars(block._ext_vars))
        cfg = {'ext_vars':    block_ext_vars,
               'cwd':         cwd,
               'parent_id':   block._id,
               'propagators': propagators}
        module = ModuleArchitecture(block._path, cfg=cfg)
        block._shape = module.architecture._shape
        delattr(module.architecture, '_shape')
        block.architecture = module.architecture
