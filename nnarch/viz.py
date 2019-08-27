
import itertools
import textwrap
from jsonargparse import ArgumentParser, ActionYesNo
from pygraphviz import AGraph
from .parse import get_shape
from . import __version__


def draw_graph(graph, out_file, layout_prog='dot'):
    """Draws a graph to the given file.

    Args:
        graph (AGraph): The graph to draw.
        out_file (str): Path where to write the drawn image with a valid extension for pygraphviz to determine the type.
        layout_prog (str): The graphviz layout method to use.
    """
    graph.layout(prog=layout_prog)
    graph.draw(out_file)


class CreateArchitectureGraph:
    """Class for instantiating a CreateArchitectureGraph object useful for creating graphs for an architecture."""

    @staticmethod
    def get_config_parser():
        """Returns a CreateArchitectureGraph configuration parser."""
        parser = ArgumentParser(description=__doc__, version=__version__)
        parser.add_argument('--distinguish_nodes',
            default=True,
            action=ActionYesNo,
            help='Whether to have different shapes for types of graph nodes.')
        parser.add_argument('--edge_shapes',
            default=True,
            action=ActionYesNo,
            help='Whether to include tensor shapes next to the edges.')
        parser.add_argument('--properties',
            default=True,
            action=ActionYesNo,
            help='Whether to include properties in nodes.')
        parser.add_argument('--subgraphs',
            default=True,
            action=ActionYesNo,
            help='Whether to create cluster subgraphs for sequential nodes.')
        parser.add_argument('--descriptions',
            default=True,
            action=ActionYesNo,
            help='Whether to include input/output descriptions.')
        return parser

    def __init__(self, cfg=None, parser=None):
        """Initializer for CreateArchitectureGraph class.

        Args:
            cfg (str or SimpleNamespace or None): Path to configuration file or an already parsed namespace object. If None default values are used.
            parser (ArgumentParser): Parser object to check config for the cases in which the parser is an extension of CreateArchitectureGraph.get_config_parser.
        """
        if parser is None:
            parser = self.get_config_parser()
        if cfg is None:
            cfg = parser.get_defaults()
        elif isinstance(cfg, str):
            self.cfg_file = cfg
            cfg = parser.parse_path(cfg)
        else:
            parser.check_config(cfg, skip_none=True)

        self.cfg = cfg
        self.parser = parser

    def __call__(self, architecture, blocks):
        """Creates an AGraph for the given architecture and blocks.

        Args:
            architecture (SimpleNamespace): A parsed architecture namespace object.
            blocks (OrderedDict): The dictionary of architecture blocks.

        Returns:
            AGraph: The created graph for the architecture.
        """
        cfg = self.cfg

        ## Create raw graph ##
        graph = AGraph('\n'.join(['digraph {']+architecture.graph+['}']))

        ## Helper functions ##
        def props_label(node):
            exclude = {'out_features', 'in_features'} if cfg.edge_shapes else set()
            props = ''
            if hasattr(node, '_id'):
                props += '<BR />id: '+node._id
            for k, v in vars(node).items():
                if not k.startswith('_') and k not in exclude:
                    if node._class == 'Sequential' and k == 'modules':
                        props += '<BR />'+k+': '+str(len(v))
                    else:
                        props += '<BR />'+k+': '+str(v)
            if props != '':
                label = '<'+node._class+'<FONT POINT-SIZE="6">'+props+'</FONT>>'
            else:
                label = node._class
            return label

        def desc_label(node):
            if not hasattr(node, '_description'):
                return node._id
            desc = '<BR />'.join(textwrap.wrap(node._description, width=50))
            return '<'+node._id+'<FONT POINT-SIZE="6"><BR />'+desc+'</FONT>>'

        def set_edge_shape(node_from, node_to, submodule=False):
            if submodule and not node_from in blocks:
                module_from, index = node_from.rsplit('.', 1)
                block_parent = blocks[module_from]
                block_from = block_parent.modules[int(index)]
            else:
                block_from = blocks[node_from]
            shape = get_shape('out', block_from)
            shape = ' x '.join([x.replace('<<variable','#').replace('>>','') if str(x).startswith('<<variable') else str(x) for x in shape])
            graph.get_edge(node_from, node_to).attr['label'] = ' '+shape  # pylint: disable=no-member

        ## Add architecture description ##
        if hasattr(architecture, '_description'):
            desc = '<BR />'.join(textwrap.wrap(architecture._description, width=100))
            graph.graph_attr['label'] = '<'+desc+'>'
            graph.graph_attr['labelloc'] = 't'
            graph.graph_attr['labeljust'] = 'l'

        ## Change node shapes for different types ##
        if cfg.distinguish_nodes:
            for node in architecture.inputs:
                graph.get_node(node._id).attr['shape'] = 'invhouse'
            for node in architecture.outputs:
                graph.get_node(node._id).attr['shape'] = 'house'
            for node in architecture.blocks:
                graph.get_node(node._id).attr['shape'] = 'hexagon' if node._class.startswith('Reshape') else 'box'

        ## Add input/output node descriptions ##
        if cfg.descriptions:
            for node in itertools.chain(architecture.inputs, architecture.outputs):
                graph.get_node(node._id).attr['label'] = desc_label(node)

        ## Add tensor shapes to edges ##
        if cfg.edge_shapes:
            for node_from, node_to in graph.edges():
                set_edge_shape(node_from, node_to)

        ## Relabel nodes with properties if required ##
        for node in architecture.blocks:
            label = props_label(node) if cfg.properties else node._class
            graph.get_node(node._id).attr['label'] = label

        ## Replace sequentials with cluster subgraphs ##
        if cfg.subgraphs:
            for block in architecture.blocks:
                if block._class == 'Sequential':
                    ## Remove edges and node ##
                    edges = graph.edges(block._id)
                    for edge in edges:
                        graph.remove_edge(*edge)
                    edges_to = [(u, v) for u, v in edges if u == block._id]
                    edges_from = [(u, v) for u, v in edges if v == block._id]
                    graph.remove_node(block._id)
                    ## Create subgraph cluster ##
                    label = props_label(block) if cfg.properties else block._class
                    subgraph = graph.add_subgraph(name='cluster_'+block._id, label=label, labeljust='r', labelloc='t')
                    ## Add sequential nodes and edges ##
                    for num, module in enumerate(block.modules):
                        name = block._id+'.'+str(num)
                        label = props_label(module) if cfg.properties else module._class
                        subgraph.add_node(name, label=label)
                        if cfg.distinguish_nodes:
                            subgraph.get_node(name).attr['shape'] = 'hexagon' if block._class.startswith('Reshape') else 'box'
                        if num > 0:
                            subgraph.add_edge(name_prev, name)
                            if cfg.edge_shapes:
                                set_edge_shape(name_prev, name, submodule=True)
                        if num == 0:
                            for u, v in edges_from:
                                graph.add_edge(u, name)
                                if cfg.edge_shapes:
                                    set_edge_shape(u, name, submodule=True)
                        name_prev = name
                    ## Add final edges ##
                    for u, v in edges_to:
                        graph.add_edge(name, v)
                        if cfg.edge_shapes:
                            set_edge_shape(name, v, submodule=True)

        return graph
