#!/usr/bin/env python3
"""Command line tool for rendering to a file a diagram of a neural network module architecture."""

import sys
from jsonargparse import ActionPath, ActionConfigFile
from nnarch.render import ModuleArchitectureRenderer


def get_parser():
    """Returns the argument parser object for the command line tool."""
    parser = ModuleArchitectureRenderer.get_config_parser()
    parser.error_handler = 'usage_and_exit_error_handler'
    parser.description = __doc__
    parser.add_argument('jsonnet_path',
        action=ActionPath(mode='fr'),
        help='Path to a neural network module architecture file in jsonnet nnarch format.')
    parser.add_argument('output',
        action=ActionPath(mode='fc'),
        help='Path where to write the architecture diagram (with a valid extension for pygraphviz draw).')
    parser.add_argument('--save_gv',
        action=ActionPath(mode='fc'),
        help='Set to also write graphviz file to given path.')
    parser.add_argument('--cfg',
        action=ActionConfigFile,
        help='Path to a configuration file.')
    return parser


def nnarch_render(argv=None):
    """Main execution function."""

    ## Parse arguments ##
    parser = get_parser()
    cfg = parser.parse_args(sys.argv[1:] if argv is None else argv)

    ## Instantiate render object ##
    renderer = ModuleArchitectureRenderer(cfg.jsonnet_path(), cfg=cfg, parser=parser)

    ## Render architecture diagram to file ##
    graph = renderer.render(cfg.output())

    ## Save graphviz file if requested ##
    if cfg.save_gv is not None:
        graph.write(cfg.save_gv())


## Main block called only when run from command line ##
if __name__ == '__main__':
    nnarch_render()
