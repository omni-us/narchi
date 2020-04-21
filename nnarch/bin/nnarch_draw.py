#!/usr/bin/env python3
"""Command line tool for drawing to a file a diagram of a neural network module architecture."""

import sys
from jsonargparse import ActionJsonnetExtVars, ActionJsonSchema, ActionPath
from nnarch.module import load_module_architecture
from nnarch.register import propagators
from nnarch.viz import CreateArchitectureGraph, draw_graph


def get_parser():
    """Returns the argument parser object for the command line tool."""
    parser = CreateArchitectureGraph.get_config_parser()
    parser.error_handler = 'usage_and_exit_error_handler'
    parser.description = __doc__
    parser.add_argument('--ext_vars',
        action=ActionJsonnetExtVars,
        help='Path to or string containing a json defining external variables required to load the jsonnet.')
    parser.add_argument('--save_gv',
        action=ActionPath(mode='fc'),
        help='Set to also write graphviz file to given path.')
    parser.add_argument('--layout_prog',
        choices=['neato', 'dot', 'twopi', 'circo', 'fdp'],
        default='dot',
        help='The graphviz layout method to use.')
    parser.add_argument('jsonnet_path',
        action=ActionPath(mode='fr'),
        help='Path to a neural network module architecture file in jsonnet nnarch format.')
    parser.add_argument('output',
        action=ActionPath(mode='fc'),
        help='Path where to write the architecture diagram (with a valid extension for pygraphviz draw).')
    return parser


def main(argv=None):
    """Main execution function."""

    ## Parse arguments ##
    parser = get_parser()
    cfg = parser.parse_args(sys.argv[1:] if argv is None else argv)

    ## Load architecture ##
    module = load_module_architecture(cfg.jsonnet_path(), ext_vars=cfg.ext_vars, propagators=propagators)

    ## Create graph ##
    graph = CreateArchitectureGraph(cfg=cfg, parser=parser)(module)

    ## Write output ##
    if cfg.save_gv is not None:
        graph.write(cfg.save_gv())
    draw_graph(graph, cfg.output(), layout_prog=cfg.layout_prog)


## Main block called only when run from command line ##
if __name__ == '__main__':
    main()
