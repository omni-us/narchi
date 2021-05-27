"""Functions related to the parsing of graphs."""

from collections import OrderedDict
from jsonargparse import dict_to_namespace, Namespace
from networkx import DiGraph
from networkx.algorithms.dag import is_directed_acyclic_graph, topological_sort
from typing import Dict, List


def digraph_from_graph_list(graph_list):
    edges = []
    for line in graph_list:
        nodes = line.split(' -> ')
        for num in range(1, len(nodes)):
            edges.append((nodes[num-1], nodes[num]))
    graph = DiGraph()
    graph.add_edges_from(edges)
    return graph


def parse_graph(from_blocks: List[Namespace], block: Namespace) -> Dict[str, List[str]]:
    """Parses a graph of a block.

    Args:
        from_blocks: The input blocks.
        block: The block to parse its graph.

    Returns:
        Dictionary in topological order mapping node IDs to its respective input nodes IDs.

    Raises:
        ValueError: If there are problems parsing the graph.
        ValueError: If the graph is not directed and acyclic.
        ValueError: If topological sort does not include all nodes.
    """
    if isinstance(block, dict):
        block = dict_to_namespace(block)
    if any(isinstance(x, dict) for x in from_blocks):
        from_blocks = [dict_to_namespace(x) for x in from_blocks]

    ## Get graph list ##
    if hasattr(block, '_class') and block._class == 'Sequential':
        graph_list = [from_blocks[0]._id + ' -> ' + ' -> '.join([b._id for b in block.blocks])]
    else:
        graph_list = block.graph
        if hasattr(block, 'input') and isinstance(block.input, str):
            graph_list = [from_blocks[0]._id+' -> '+block.input] + graph_list

    ## Parse graph ##
    try:
        graph = digraph_from_graph_list(graph_list)
    except Exception as ex:
        raise ValueError(f'Problems parsing graph for block[id={block._id}]: {ex}') from ex
    if not is_directed_acyclic_graph(graph):
        raise ValueError(f'Expected graph to be directed and acyclic for block[id={block._id}], graph={graph_list}.')

    ## Create topologically ordered dict mapping all nodes to its inputs ##
    topological_predecessors = OrderedDict()
    for node in topological_sort(graph):
        predecessors = [n for n in graph.predecessors(node)]
        if len(predecessors) > 0:
            topological_predecessors[node] = predecessors

    nodes_blocks = {b._id for b in block.blocks}
    nodes_topological = {k for k in topological_predecessors.keys()}
    missing = nodes_blocks - nodes_topological
    if len(missing) > 0:
        raise ValueError(f'Graph in block[id={block._id}] does not reference all of its blocks: missing={missing}.')

    return topological_predecessors
