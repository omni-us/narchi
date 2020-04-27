"""Functions related to the parsing of graphs."""

from collections import OrderedDict
from pygraphviz import AGraph
from networkx.drawing.nx_agraph import from_agraph
from networkx.algorithms.dag import is_directed_acyclic_graph, topological_sort


def parse_graph(from_blocks, block):
    """Parses a graph of a block.

    Args:
        from_blocks (list[SimpleNamespace]): The input blocks.
        block (SimpleNamespace): The block to parse its graph.

    Returns:
        OrderedDict[str, list[str]]: Dictionary in topological order mapping \
                                     node IDs to its respective input nodes IDs.

    Raises:
        ValueError: If there are problems parsing the graph.
        ValueError: If the graph is not directed and acyclic.
        ValueError: If topological sort does not include all nodes.
    """
    ## Parse graph ##
    graph_list = block.graph
    if hasattr(block, 'input') and isinstance(block.input, str):
        graph_list = [from_blocks[0]._id+' -> '+block.input] + block.graph
    try:
        graph = from_agraph(AGraph('\n'.join(['digraph {']+graph_list+['}'])))
    except Exception as ex:
        raise ValueError('Problems parsing graph for block[id='+block._id+']: '+str(ex))
    if not is_directed_acyclic_graph(graph):
        raise ValueError('Expected graph to be directed and acyclic for block[id='+block._id+'].')

    ## Create topologically ordered dict mapping all nodes to its inputs ##
    try:
        all_nodes = set()
        topological_predecessors = OrderedDict()
        for node in topological_sort(graph):
            all_nodes.add(node)
            predecessors = [n for n in graph.predecessors(node)]
            if len(predecessors) > 0:
                topological_predecessors[node] = predecessors
    except Exception as ex:
        raise ValueError('Topological sorting failed for block[id='+block._id+'] :: '+str(ex))
    missing = all_nodes - {k for k in topological_predecessors.keys()} - {b._id for b in from_blocks}
    if len(missing) > 0:
        raise ValueError('Topological sort does not include all nodes for block[id='+block._id+']: missing='+str(missing)+'.')

    return topological_predecessors
