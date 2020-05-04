#!/usr/bin/env python3
"""General command line tool for nnarch package functionalities."""

import sys
from jsonargparse import ArgumentParser, ActionPath
from nnarch.render import ModuleArchitecture, ModuleArchitectureRenderer
from nnarch.schema import schema_as_str
from nnarch import __version__


def get_parser():
    """Returns the argument parser object for the command line tool."""
    ## validate parser ##
    parser_validate = ModuleArchitecture.get_config_parser()
    parser_validate.description = 'Command for checking the validity of neural network module architecture files.'
    parser_validate.set_defaults(propagators='default')
    parser_validate.add_argument('jsonnet_paths',
        action=ActionPath(mode='fr'),
        nargs='+',
        help='Path(s) to neural network module architecture file(s) in jsonnet nnarch format.')

    ## render parser ##
    parser_render = ModuleArchitectureRenderer.get_config_parser()
    parser_render.description = 'Command for rendering a neural network module architecture file.'
    parser_render.set_defaults(propagators='default')
    parser_render.add_argument('jsonnet_path',
        action=ActionPath(mode='fr'),
        help='Path to a neural network module architecture file in jsonnet nnarch format.')
    parser_render.add_argument('out_file',
        nargs='?',
        action=ActionPath(mode='fc'),
        help='Path where to write the architecture diagram (with a valid extension for pygraphviz draw). If '
             'unset a pdf is saved to the output directory.')

    ## schema parser ##
    parser_schema = ArgumentParser(
        description='Prints a schema as a pretty json.')
    parser_schema.add_argument('schema',
        nargs='?',
        default='nnarch',
        choices=['nnarch', 'propagated', 'reshape', 'block'],
        help='Which of the available schemas to print.')

    ## global parser ##
    parser = ArgumentParser(
        error_handler='usage_and_exit_error_handler',
        description=__doc__,
        version=__version__)
    parser.parser_validate = parser_validate
    parser.parser_render = parser_render
    parser.parser_schema = parser_schema

    subcommands = parser.add_subcommands()
    subcommands.add_subcommand('validate', parser_validate)
    subcommands.add_subcommand('render', parser_render)
    subcommands.add_subcommand('schema', parser_schema)

    return parser


def get_validate_parser():
    return get_parser().parser_validate


def get_render_parser():
    return get_parser().parser_render


def get_schema_parser():
    return get_parser().parser_schema


def nnarch_cli(argv=None):
    """Main execution function."""

    ## Parse arguments ##
    parser = get_parser()
    cfg = parser.parse_args(sys.argv[1:] if argv is None else argv)

    ## Schema subcommand ##
    if cfg.subcommand == 'schema':
        print(schema_as_str(cfg.schema.schema))

    ## Validate subcommand ##
    elif cfg.subcommand == 'validate':
        module = ModuleArchitecture(cfg=cfg.validate, parser=parser.parser_validate)
        for jsonnet_path in cfg.validate.jsonnet_paths:
            module.load_architecture(jsonnet_path)

    ## Render subcommand ##
    elif cfg.subcommand == 'render':
        if cfg.render.out_file is None:
            cfg.render.save_pdf = True
        module = ModuleArchitectureRenderer(cfg=cfg.render, parser=parser.parser_render)
        module.render(architecture=cfg.render.jsonnet_path, out_render=cfg.render.out_file)


## Main block called only when run from command line ##
if __name__ == '__main__':
    nnarch_cli()
