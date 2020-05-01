import re
import itertools
import textwrap
from jsonargparse import ArgumentParser, ActionJsonSchema, ActionJsonnetExtVars, ActionOperators, SimpleNamespace, namespace_to_dict
from pygraphviz import AGraph
from .propagators.base import get_shape
from .module import ModuleArchitecture
from .propagators.group import get_blocks_dict
from .graph import parse_graph
from .register import propagators as default_propagators
from .schema import id_separator
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
                     'Nested':  'shape=box, style=dashed',
                     'Reshape': 'shape=hexagon',
                     'Add':     'shape=diamond'},
            action=ActionJsonSchema(schema={'type': 'object', 'items': {'type': 'string'}}),
            help='Attributes for block nodes.')
        parser.add_argument('--nested_depth',
            default=3,
            action=ActionOperators(expr=('>=', 0)),
            help='Maximum depth for nested subblocks to render. Set to 0 for unlimited.')
        parser.add_argument('--full_ids',
            default=False,
            type=bool,
            help='Whether block IDs should include parent(s) prefix.')
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
            description = architecture._description
            description = '<BR />'.join(textwrap.wrap(description, width=100))
            graph.graph_attr['label'] = '<'+description+'>'
            graph.graph_attr['labelloc'] = 't'
            graph.graph_attr['labeljust'] = 'l'


    @staticmethod
    def set_node_description(graph, node, full_ids=False):
        node_id = node._id if full_ids else node._id.split(id_separator)[-1]
        description = node_id
        if hasattr(node, '_description'):
            description = '<BR />'.join(textwrap.wrap(node._description, width=50))
            description = '<'+node_id+'<FONT POINT-SIZE="6"><BR />'+description+'</FONT>>'
        graph.get_node(node._id).attr['label'] = description


    @staticmethod
    def set_edge_label(graph, blocks, node_from, node_to, subblock=False):
        block_from = blocks[node_from]
        if hasattr(block_from, '_shape'):
            shape = get_shape('out', block_from)
            shape = ' x '.join(str(sympify_variable(d)) for d in shape)
            graph.get_edge(node_from, node_to).attr['label'] = ' '+shape


    @staticmethod
    def set_block_label(graph, block, graph_attr=False, full_ids=False):
        exclude = {'output_size', 'graph', 'input', 'output', 'architecture'}
        name = block._class
        if hasattr(block, '_name'):
            name = block._name
        props = ''
        if hasattr(block, '_id'):
            block_id = block._id if full_ids else block._id.split(id_separator)[-1]
            props += '<BR />id: '+block_id
        for k, v in vars(block).items():
            if not k.startswith('_') and k not in exclude:
                if block._class in {'Sequential', 'Group'} and k == 'blocks':
                    props += '<BR />'+k+': '+str(len(v))
                else:
                    if isinstance(v, SimpleNamespace):
                        v = namespace_to_dict(v)
                    props += '<BR />'+k+': '+str(v)
        if props != '':
            label = '<'+name+'<FONT POINT-SIZE="6">'+props+'</FONT>>'
        else:
            label = name
        if graph_attr:
            graph.graph_attr['label'] = label
        else:
            graph.get_node(block._id).attr['label'] = label


    def set_block_attrs(self, graph, blocks, block_class=None, graph_attr=False):
        block_attrs = self.block_attrs
        for block in blocks:
            attrs_class = block._class if block_class is None else block_class
            attrs = block_attrs[attrs_class] if attrs_class in block_attrs else block_attrs['Default']
            for a, v in attrs.items():
                if graph_attr:
                    graph.graph_attr[a] = v
                else:
                    graph.get_node(block._id).attr[a] = v


    def create_graph(self):
        architecture = self.architecture
        blocks = self.blocks

        ## Create raw graph ##
        graph = AGraph('\n'.join(['digraph {']+architecture.graph+['}']))

        ## Add architecture description ##
        self.set_architecture_description(graph, architecture)

        ## Set attributes of blocks ##
        self.set_block_attrs(graph, architecture.inputs, block_class='Input')
        self.set_block_attrs(graph, architecture.outputs, block_class='Output')
        self.set_block_attrs(graph, architecture.blocks)

        ## Add input/output descriptions ##
        for node in itertools.chain(architecture.inputs, architecture.outputs):
            self.set_node_description(graph, node)

        ## Add tensor shapes to edges ##
        for node_from, node_to in graph.edges():
            self.set_edge_label(graph, blocks, node_from, node_to)

        ## Set block properties ##
        for block in architecture.blocks:
            self.set_block_label(graph, block)

        ## Create subgraphs ##
        self.add_subgraphs(graph, architecture.blocks, blocks, depth=2)

        return graph


    def add_subgraphs(self, graph, blocks, blocks_dict, depth):
        if depth > self.cfg.nested_depth and not self.cfg.nested_depth == 0:
            return
        full_ids = self.cfg.full_ids
        subblocks_dict = dict(blocks_dict)
        for block in [b for b in blocks if b._class in {'Sequential', 'Group', 'Module'}]:
            ## Remove edges and node ##
            edges = graph.edges(block._id)
            edges_from = [(u, v) for u, v in edges if v == block._id]
            edges_to = [(u, v) for u, v in edges if u == block._id]
            for edge in edges:
                graph.remove_edge(*edge)
            graph.remove_node(block._id)
            ## Create subgraph cluster ##
            subgraph = graph.add_subgraph(name='cluster_'+block._id, labeljust='r', labelloc='t')
            self.set_block_label(subgraph, block, graph_attr=True, full_ids=full_ids)
            self.set_block_attrs(subgraph, [block], block_class='Nested', graph_attr=True)
            if block._class == 'Module':
                subblocks_dict.update(get_blocks_dict(block.architecture.inputs+block.architecture.outputs))
                input_id = block.architecture.inputs[0]._id
                subgraph.add_node(input_id)
                self.set_node_description(graph, subblocks_dict[input_id], full_ids=full_ids)
                self.set_block_attrs(graph, [subblocks_dict[input_id]], block_class='Input')
                graph.add_edge(edges_from[0][0], input_id)
                self.set_edge_label(graph, subblocks_dict, edges_from[0][0], input_id, subblock=True)
                output_id = block.architecture.outputs[0]._id
                subgraph.add_node(output_id)
                self.set_node_description(graph, subblocks_dict[output_id], full_ids=full_ids)
                self.set_block_attrs(graph, [subblocks_dict[output_id]], block_class='Output')
                graph.add_edge(output_id, edges_to[0][1])
                self.set_edge_label(graph, subblocks_dict, output_id, edges_to[0][1], subblock=True)
                block = block.architecture
                edges_from[0] = (input_id, edges_from[0][1])
                edges_to = []
            subblocks_dict.update(get_blocks_dict(block.blocks))
            ## Add subblocks nodes and edges ##
            blocks_from = [subblocks_dict[edges_from[0][0]]]
            topological_predecessors = parse_graph(blocks_from, block)
            for subblock_id, prev_ids in topological_predecessors.items():
                subgraph.add_node(subblock_id)
                subblock = subblocks_dict[subblock_id]
                if hasattr(subblock, '_class'):
                    self.set_block_label(subgraph, subblock, full_ids=full_ids)
                for node_id_prev in prev_ids:
                    graph.add_edge(node_id_prev, subblock_id)
                    self.set_edge_label(graph, subblocks_dict, node_id_prev, subblock_id, subblock=True)
            self.set_block_attrs(graph, block.blocks)
            ## Add final edges ##
            for u, v in edges_to:
                graph.add_edge(subblock_id, v)
                self.set_edge_label(graph, subblocks_dict, subblock_id, v, subblock=True)
            ## Add subgraphs ##
            self.add_subgraphs(graph, block.blocks, subblocks_dict, depth=depth+1)


    def render(self, out_render=None, out_gv=None, out_json=None, cfg=None):
        """Renders the architecture diagram optionally writing to the given file path.

        Args:
            out_render (str or Path or None): Path where to write the rendered diagram with a valid \
                                              extension for pygraphviz to determine the type.
            out_gv (str or Path or None): Path where to write the graphviz source in dot format.
            out_json (str or Path or None): Path where to write the architecture in json format.
            cfg (SimpleNamespace): Configuration to apply before rendering.

        Returns:
            AGraph: pygraphviz graph object.
        """
        if cfg is not None:
            self.apply_config(cfg)
        graph = self.create_graph()
        if out_json:
            self.write_json(out_json)
        if out_gv:
            graph.write(out_gv if isinstance(out_gv, str) else out_gv())
        graph.layout(prog=self.cfg.layout_prog)
        if out_render is not None:
            graph.draw(out_render if isinstance(out_render, str) else out_render())
        return graph
