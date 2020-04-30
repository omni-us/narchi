#!/usr/bin/env python3
"""Command line tool for validating neural network module architecture jsonnet files."""

import os
import sys
import traceback
from jsonargparse import ArgumentParser, ParserError, ActionPath, ActionJsonnetExtVars, ActionYesNo
from nnarch.module import ModuleArchitecture
from nnarch.register import propagators
from nnarch.schema import schema_as_str
from nnarch import __version__


def get_parser():
    """Returns the argument parser object for the command line tool."""
    parser = ArgumentParser(
        version=__version__,
        logger={'name': os.path.basename(__file__), 'level': 'WARNING'},
        description=__doc__)
    parser.add_argument('jsonnet_path',
        nargs='*',
        action=ActionPath(mode='fr'),
        help='Path to a neural network module architecture file in jsonnet nnarch format.')
    parser.add_argument('--propagate',
        action=ActionYesNo,
        default=False,
        help='Whether to also load the module architecture to verify it can be propagated.')
    parser.add_argument('--ext_vars',
        action=ActionJsonnetExtVars(),
        help='Path to or string containing a json defining external variables required to load the jsonnet.')
    parser.add_argument('--save_json',
        action=ActionPath(mode='fc'),
        help='Save the parsed file (up to the last successful step: jsonnet load, schema validation, parsing) to the given path in json format.')
    parser.add_argument('--schema',
        action='store_true',
        help='Print the nnarch schema and exit.')
    return parser


def nnarch_validate(argv=None, sys_exit=True):
    """Main execution function."""

    module = cfg = None

    ## Function to exit when there are problems ##
    def error_exit(ex, msg=None):
        if hasattr(cfg, 'save_json') and all(x is not None for x in [module, cfg.save_json]):
            parser.logger.warning('Saving current state of json to '+cfg.save_json(absolute=False))
            module.write_json(cfg.save_json())
        if msg is not None and sys_exit:
            parser.logger.error(msg)
        if sys_exit:
            sys.exit(1)
        else:
            raise ex

    ## Parse arguments ##
    parser = get_parser()
    try:
        cfg = parser.parse_args(sys.argv[1:] if argv is None else argv)
    except Exception as ex:
        error_exit(ex)

    ## Print schema and exit if requested ##
    if cfg.schema:
        print(schema_as_str())
        sys.exit(0)

    ## Check that one jsonnet file is provided ##
    if len(cfg.jsonnet_path) != 1:
        msg = 'A single jsonnet file path must be provided.'
        error_exit(ParserError(msg), msg)

    ## Load jsonnet file ##
    jsonnet_path = cfg.jsonnet_path[0]
    path_relative = jsonnet_path(absolute=False)
    try:
        module = ModuleArchitecture(jsonnet_path,
                                    ext_vars=cfg.ext_vars,
                                    propagators=propagators,
                                    propagate=False,
                                    validate=False)
    except Exception as ex:
        error_exit(ex, 'Failed to load jsonnet file "'+path_relative+'" :: '+traceback.format_exc())

    ## Validate jsonnet against schema ##
    try:
        module.validate()
    except Exception as ex:
        error_exit(ex, 'Architecture file "'+path_relative+'" does not validate against nnarch schema :: '+traceback.format_exc())

    ## Propagate shapes of module architecture ##
    if cfg.propagate:
        try:
            module.propagate()
        except Exception as ex:
            error_exit(ex, 'Architecture file "'+path_relative+'" failed to propagate :: '+traceback.format_exc())

    ## Write final json ##
    if cfg.save_json is not None:
        module.write_json(cfg.save_json)


## Main block called only when run from command line ##
if __name__ == '__main__':
    nnarch_validate()
