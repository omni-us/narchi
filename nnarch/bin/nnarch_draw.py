#!/usr/bin/env python
"""Command line tool for drawing to a file a diagram of a neural network architecture."""

from yamlargparse import ActionJsonSchema, ActionJsonnet, namespace_to_dict
from nnarch.parse import parse_architecture
from nnarch.viz import CreateArchitectureGraph, draw_graph


def get_parser():
    """Returns the argument parser object for the command line tool."""
    parser = CreateArchitectureGraph.get_config_parser()
    parser.error_handler = 'usage_and_exit_error_handler'
    parser.description = __doc__
    parser.add_argument('--const',
        action=ActionJsonSchema(schema={'type': 'object'}),
        default={},
        help='Path to or string containing a json defining constants required by the architecture.')
    parser.add_argument('--dot',
        help='Set to also write dot file to given path.')
    parser.add_argument('--layout_prog',
        choices=['neato', 'dot', 'twopi', 'circo', 'fdp'],
        default='dot',
        help='The graphviz layout method to use.')
    parser.add_argument('input',
        action=ActionJsonnet(schema=None),
        help='Path to a neural network architecture file in jsonnet nnarch format.')
    parser.add_argument('output',
        help='Path where to write the architecture diagram (with a valid extension for pygraphviz draw).')
    return parser


## Main block called only when run from command line ##
if __name__ == '__main__':
    ## Parse arguments ##
    parser = get_parser()
    cfg = parser.parse_args()
    ## Parse architecture ##
    architecture, blocks, _ = parse_architecture(cfg.input, const=namespace_to_dict(cfg.const))
    ## Create graph ##
    graph = CreateArchitectureGraph(cfg=cfg, parser=parser)(architecture, blocks)
    ## Write output ##
    if cfg.dot is not None:
        graph.write(cfg.dot)
    draw_graph(graph, cfg.output, layout_prog=cfg.layout_prog)
