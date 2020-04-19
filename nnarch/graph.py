"""Functions related to the parsing of graphs."""

from collections import OrderedDict
from pygraphviz import AGraph
from networkx.drawing.nx_agraph import from_agraph
from networkx.algorithms.dag import is_directed_acyclic_graph
from networkx.algorithms.traversal.edgebfs import edge_bfs


def get_nodes_with_inputs(graph, source):
    """Traverses a graph creating an OrderedDict of nodes with respective inputs."""
    in_nodes = OrderedDict()
    for node_from, node_to, _ in edge_bfs(graph, source=source):
        if node_to not in in_nodes:
            in_nodes[node_to] = []
        in_nodes[node_to].append(node_from)
    return in_nodes


def parse_graph(graph_attr, input_node):
    """Parses a graph attribute.

    Arguments:
        graph_attr (list[str]): Graph attribute, a list of strings in graphviz format.
        input_node (list[dict]): ID of the input node of the graph.

    Returns:
        OrderedDict[str, list[str]]: Dictionary mapping node IDs to its
            respective input nodes IDs ordered by traversal.

    Raises:
        ValueError: If there are problems parsing or traversing the graph.
        ValueError: If the graph is not directed and acyclic.
        ValueError: If graph traversal does not include all nodes.
    """
    ## Parse graph ##
    try:
        graph = from_agraph(AGraph('\n'.join(['digraph {']+graph_attr+['}'])))
    except Exception as ex:
        raise ValueError('Problems parsing graph: '+str(ex))
    if not is_directed_acyclic_graph(graph):
        raise ValueError('Expected graph to be directed and acyclic.')

    ## Traverse graph ##
    try:
        in_nodes = get_nodes_with_inputs(graph, source=input_node)
    except Exception as ex:
        raise ValueError('Problems traversing graph: '+str(ex))
    if len(in_nodes) != graph.number_of_nodes()-1:
        raise ValueError('Graph traversal does not include all nodes: '+str(in_nodes))

    return in_nodes
