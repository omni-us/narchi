import os
import re
import itertools
import textwrap
from jsonargparse import ActionJsonSchema, ActionOperators, SimpleNamespace, namespace_to_dict
from pygraphviz import AGraph
from .propagators.base import get_shape
from .module import ModuleArchitecture
from .propagators.group import get_blocks_dict
from .graph import parse_graph
from .schema import id_separator
from .sympy import sympify_variable


class ModuleArchitectureRenderer(ModuleArchitecture):
    """Class for instantiating a ModuleArchitectureRenderer objects useful for rendering module architectures diagram."""

    @staticmethod
    def get_config_parser():
        """Returns a ModuleArchitectureRenderer configuration parser."""
        parser = ModuleArchitecture.get_config_parser()
        parser.description = ModuleArchitectureRenderer.__doc__

        # render options #
        group_render = parser.add_argument_group('Rendering related options')
        group_render.add_argument('--save_pdf',
            default=False,
            type=bool,
            help='Whether write rendered pdf file to output directory.')
        group_render.add_argument('--save_gv',
            default=False,
            type=bool,
            help='Whether write graphviz file to output directory.')
        group_render.add_argument('--block_attrs',
            default={'Default': 'shape=box',
                     'Input':   'shape=box, style=rounded, penwidth=1.5',
                     'Output':  'shape=box, style=rounded, peripheries=2',
                     'Nested':  'shape=box, style=dashed',
                     'Reshape': 'shape=hexagon',
                     'Add':     'shape=diamond'},
            action=ActionJsonSchema(schema={'type': 'object', 'items': {'type': 'string'}}),
            help='Attributes for block nodes.')
        group_render.add_argument('--nested_depth',
            default=3,
            action=ActionOperators(expr=('>=', 0)),
            help='Maximum depth for nested subblocks to render. Set to 0 for unlimited.')
        group_render.add_argument('--full_ids',
            default=False,
            type=bool,
            help='Whether when rendering block IDs should include parent prefix.')
        group_render.add_argument('--layout_prog',
            choices=['dot', 'neato', 'twopi', 'circo', 'fdp'],
            default='dot',
            help='The graphviz layout method to use.')

        return parser


    def apply_config(self, cfg):
        """Applies a configuration to the ModuleArchitectureRenderer instance.

        Args:
            cfg (str or dict or SimpleNamespace): Path to config file or config object.
        """
        super().apply_config(cfg)

        if hasattr(cfg, 'block_attrs'):  # @todo support also dict
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
    def _set_architecture_description(graph, architecture):
        """Sets the architecture description to a graph as a label."""
        if hasattr(architecture, '_description'):
            description = architecture._description
            description = '<BR />'.join(textwrap.wrap(description, width=100))
            graph.graph_attr['label'] = '<'+description+'>'
            graph.graph_attr['labelloc'] = 't'
            graph.graph_attr['labeljust'] = 'l'


    @staticmethod
    def _set_node_description(graph, node, full_ids=False):
        """Sets a node description as a label."""
        node_id = node._id if full_ids else node._id.split(id_separator)[-1]
        description = node_id
        if hasattr(node, '_description'):
            description = '<BR />'.join(textwrap.wrap(node._description, width=50))
            description = '<'+node_id+'<FONT POINT-SIZE="6"><BR />'+description+'</FONT>>'
        graph.get_node(node._id).attr['label'] = description


    @staticmethod
    def _set_edge_label(graph, blocks, node_from, node_to, subblock=False):
        """Sets the shape dimensions to an edge as its label."""
        block_from = blocks[node_from]
        if hasattr(block_from, '_shape'):
            shape = get_shape('out', block_from)
            shape = ' x '.join(str(sympify_variable(d)) for d in shape)
            graph.get_edge(node_from, node_to).attr['label'] = ' '+shape


    @staticmethod
    def _set_block_label(graph, block, graph_attr=False, full_ids=False):
        """Sets a block's label including its id and properties."""
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


    def _set_block_attrs(self, graph, blocks, block_class=None, graph_attr=False):
        """Sets graph style attributes to a block."""
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
        """Creates a pygraphviz graph of the architecture using the current configuration."""
        architecture = self.architecture
        blocks = self.blocks

        ## Create raw graph ##
        graph = AGraph('\n'.join(['digraph {']+architecture.graph+['}']))

        ## Add architecture description ##
        self._set_architecture_description(graph, architecture)

        ## Set attributes of blocks ##
        self._set_block_attrs(graph, architecture.inputs, block_class='Input')
        self._set_block_attrs(graph, architecture.outputs, block_class='Output')
        self._set_block_attrs(graph, architecture.blocks)

        ## Add input/output descriptions ##
        for node in itertools.chain(architecture.inputs, architecture.outputs):
            self._set_node_description(graph, node)

        ## Add tensor shapes to edges ##
        for node_from, node_to in graph.edges():
            self._set_edge_label(graph, blocks, node_from, node_to)

        ## Set block properties ##
        for block in architecture.blocks:
            self._set_block_label(graph, block)

        ## Create subgraphs ##
        self._add_subgraphs(graph, architecture.blocks, blocks, depth=2)

        return graph


    def _add_subgraphs(self, graph, blocks, blocks_dict, depth, parent_graph=None):
        """Adds subgraphs to a graph if the depth is not higher that configured value."""
        if depth > self.cfg.nested_depth and not self.cfg.nested_depth == 0:
            return
        if parent_graph is None:
            parent_graph = graph
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
            subgraph = parent_graph.add_subgraph(name='cluster_'+block._id, labeljust='r', labelloc='t')
            self._set_block_label(subgraph, block, graph_attr=True, full_ids=full_ids)
            self._set_block_attrs(subgraph, [block], block_class='Nested', graph_attr=True)
            ## Handle Module ##
            if block._class == 'Module':
                subblocks_dict.update(get_blocks_dict(block.architecture.inputs+block.architecture.outputs))
                input_id = block.architecture.inputs[0]._id
                subgraph.add_node(input_id)
                self._set_node_description(graph, subblocks_dict[input_id], full_ids=full_ids)
                self._set_block_attrs(graph, [subblocks_dict[input_id]], block_class='Input')
                graph.add_edge(edges_from[0][0], input_id)
                self._set_edge_label(graph, subblocks_dict, edges_from[0][0], input_id, subblock=True)
                output_id = block.architecture.outputs[0]._id
                subgraph.add_node(output_id)
                self._set_node_description(graph, subblocks_dict[output_id], full_ids=full_ids)
                self._set_block_attrs(graph, [subblocks_dict[output_id]], block_class='Output')
                graph.add_edge(output_id, edges_to[0][1])
                self._set_edge_label(graph, subblocks_dict, output_id, edges_to[0][1], subblock=True)
                block = block.architecture
                edges_from[0] = (input_id, edges_from[0][1])
                edges_to = []
            ## Add subblocks nodes and edges ##
            subblocks_dict.update(get_blocks_dict(block.blocks))
            blocks_from = [subblocks_dict[edges_from[0][0]]]
            topological_predecessors = parse_graph(blocks_from, block)
            for subblock_id, prev_ids in topological_predecessors.items():
                subgraph.add_node(subblock_id)
                subblock = subblocks_dict[subblock_id]
                if hasattr(subblock, '_class'):
                    self._set_block_label(subgraph, subblock, full_ids=full_ids)
                for node_id_prev in prev_ids:
                    graph.add_edge(node_id_prev, subblock_id)
                    self._set_edge_label(graph, subblocks_dict, node_id_prev, subblock_id, subblock=True)
            self._set_block_attrs(graph, block.blocks)
            ## Add final edges ##
            for u, v in edges_to:
                graph.add_edge(subblock_id, v)
                self._set_edge_label(graph, subblocks_dict, subblock_id, v, subblock=True)
            ## Add subgraphs ##
            self._add_subgraphs(graph, block.blocks, subblocks_dict, depth=depth+1, parent_graph=subgraph)


    def render(self, architecture=None, out_render=None, cfg=None):
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
        if architecture is not None:
            self.load_architecture(architecture)
        graph = self.create_graph()
        if self.cfg.save_gv:
            out_gv = os.path.join(self.cfg.outdir(), self.architecture._id + '.gv')
            graph.write(out_gv)
        graph.layout(prog=self.cfg.layout_prog)
        if self.cfg.save_pdf:
            out_pdf = os.path.join(self.cfg.outdir(), self.architecture._id + '.pdf')
            graph.draw(out_pdf)
        if out_render is not None:
            graph.draw(out_render if isinstance(out_render, str) else out_render())
        return graph