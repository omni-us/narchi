import re
import itertools
import textwrap
from jsonargparse import ArgumentParser, ActionJsonSchema, ActionJsonnetExtVars, ActionOperators
from pygraphviz import AGraph
from .propagators.base import get_shape
from .module import ModuleArchitecture
from .register import propagators as default_propagators
from .sympy import sympify_variable
from . import __version__


class ModuleArchitectureRenderer(ModuleArchitecture):
    """Class for instantiating a ModuleArchitectureRenderer objects useful for rendering module architectures diagram."""

    @staticmethod
    def get_config_parser():
        """Returns a ModuleArchitectureRenderer configuration parser."""
        parser = ArgumentParser(
            description=ModuleArchitectureRenderer.__doc__,
            version=__version__)
        parser.add_argument('--ext_vars',
            action=ActionJsonnetExtVars(),
            help='External variables required to load jsonnet.')
        parser.add_argument('--propagate',
            default=True,
            type=bool,
            help='Whether to propagate shapes of architecture.')
        parser.add_argument('--validate',
            default=True,
            type=bool,
            help='Whether to validate architecture against nnarch schema.')
        parser.add_argument('--block_attrs',
            default={'Default': 'shape=box',
                     'Input':   'shape=box, style=rounded, penwidth=1.5',
                     'Output':  'shape=box, style=rounded, peripheries=2',
                     'Reshape': 'shape=hexagon',
                     'Add':     'shape=diamond'},
            action=ActionJsonSchema(schema={'type': 'object', 'items': {'type': 'string'}}),
            help='Attributes for block nodes.')
        parser.add_argument('--subgraphs_depth',
            default=3,
            action=ActionOperators(expr=('>=', 0)),
            help='Maximum depth for subgraphs. Set to 0 for unlimited.')
        parser.add_argument('--layout_prog',
            choices=['dot', 'neato', 'twopi', 'circo', 'fdp'],
            default='dot',
            help='The graphviz layout method to use.')
        return parser


    def __init__(self, architecture, propagators=None, cfg=None, parser=None):
        """Initializer for ModuleArchitectureRenderer class.

        Args:
            architecture (str or Path): Path to a jsonnet architecture file or jsonnet content.
            propagators (dict): Dictionary of propagators. If None default propagators are used.
            cfg (str or SimpleNamespace or None): Path to configuration file or an already parsed namespace \
                                                  object. If None default values are used.
            parser (ArgumentParser): Parser object to check config for the cases in which the parser is \
                                     an extension of ModuleArchitectureRenderer.get_config_parser.
        """
        if parser is None:
            parser = self.get_config_parser()
        if cfg is None:
            cfg = parser.get_defaults()
        elif isinstance(cfg, str):
            self.cfg_file = cfg
            cfg = parser.parse_path(cfg)
        self.parser = parser

        self.apply_config(cfg)

        super().__init__(architecture,
                         ext_vars=cfg.ext_vars,
                         propagators=default_propagators if propagators is None else propagators,
                         propagate=cfg.propagate,
                         validate=cfg.validate)


    def apply_config(self, cfg):
        self.parser.check_config(cfg)

        self.cfg = cfg

        block_attrs = {}
        for block, attrs in vars(cfg.block_attrs).items():
            attrs_dict = {}
            for a, v in [x.split('=') for x in re.split(', *', attrs)]:
                attrs_dict[a] = v
            block_attrs[block] = attrs_dict
        if 'Default' not in block_attrs:
            block_attrs['Default'] = {'shape': 'box'}
        self.block_attrs = block_attrs


    @staticmethod
    def set_architecture_description(graph, architecture):
        if hasattr(architecture, '_description'):
            description = '<BR />'.join(textwrap.wrap(architecture._description, width=100))
            graph.graph_attr['label'] = '<'+description+'>'
            graph.graph_attr['labelloc'] = 't'
            graph.graph_attr['labeljust'] = 'l'


    @staticmethod
    def set_node_description(graph, node):
        description = node._id
        if hasattr(node, '_description'):
            description = '<BR />'.join(textwrap.wrap(node._description, width=50))
            description = '<'+node._id+'<FONT POINT-SIZE="6"><BR />'+description+'</FONT>>'
        graph.get_node(node._id).attr['label'] = description


    @staticmethod
    def set_edge_shape(graph, blocks, node_from, node_to, submodule=False):
        if submodule and node_from not in blocks:
            module_from, index = node_from.rsplit('.', 1)
            block_parent = blocks[module_from]
            block_from = block_parent.blocks[int(index)]
        else:
            block_from = blocks[node_from]
        if hasattr(block_from, '_shape'):
            shape = get_shape('out', block_from)
            shape = ' x '.join(str(sympify_variable(d)) for d in shape)
            graph.get_edge(node_from, node_to).attr['label'] = ' '+shape


    @staticmethod
    def set_block_properties(graph, node):
        exclude = {'output_size', 'in_features'}
        props = ''
        if hasattr(node, '_id'):
            props += '<BR />id: '+node._id
        for k, v in vars(node).items():
            if not k.startswith('_') and k not in exclude:
                if node._class in {'Sequential', 'Group'} and k == 'blocks':
                    props += '<BR />'+k+': '+str(len(v))
                else:
                    props += '<BR />'+k+': '+str(v)
        if props != '':
            label = '<'+node._class+'<FONT POINT-SIZE="6">'+props+'</FONT>>'
        else:
            label = node._class
        graph.get_node(node._id).attr['label'] = label


    def create_graph(self):
        architecture = self.architecture
        blocks = self.blocks
        block_attrs = self.block_attrs

        ## Create raw graph ##
        graph = AGraph('\n'.join(['digraph {']+architecture.graph+['}']))

        ## Add architecture description ##
        self.set_architecture_description(graph, architecture)

        ## Set attributes of blocks ##
        for node in architecture.inputs:
            attrs = block_attrs['Input'] if 'Input' in block_attrs else block_attrs['Default']
            for a, v in attrs.items():
                graph.get_node(node._id).attr[a] = v
        for node in architecture.outputs:
            attrs = block_attrs['Output'] if 'Output' in block_attrs else block_attrs['Default']
            for a, v in attrs.items():
                graph.get_node(node._id).attr[a] = v
        for node in architecture.blocks:
            attrs = block_attrs[node._class] if node._class in block_attrs else block_attrs['Default']
            for a, v in attrs.items():
                graph.get_node(node._id).attr[a] = v

        ## Add input/output descriptions ##
        for node in itertools.chain(architecture.inputs, architecture.outputs):
            self.set_node_description(graph, node)

        ## Add tensor shapes to edges ##
        for node_from, node_to in graph.edges():
            self.set_edge_shape(graph, blocks, node_from, node_to)

        ## Relabel nodes with properties if required ##
        for node in architecture.blocks:
            self.set_block_properties(graph, node)

        return graph


    def render(self, out_file=None, cfg=None):
        """Renders the architecture diagram optionally writing to the given file path.

        Args:
            out_file (str or None): Path where to write the rendered diagram with a valid extension \
                                    for pygraphviz to determine the type.
            cfg (SimpleNamespace): Configuration to apply before rendering.

        Returns:
            AGraph: pygraphviz graph object.
        """
        if cfg is not None:
            self.apply_config(cfg)
        graph = self.create_graph()
        graph.layout(prog=self.cfg.layout_prog)
        if out_file is not None:
            graph.draw(out_file)
        return graph
