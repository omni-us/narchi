#!/usr/bin/env python3
"""Command line tool for validating neural network module architecture jsonnet files."""

import os
import sys
import json
from jsonargparse import ArgumentParser, ActionPath, ActionJsonnet, ActionJsonnetExtVars, ActionYesNo, namespace_to_dict
from nnarch.schema import nnarch_validator
from nnarch.module import load_module_architecture
from nnarch.register import propagators
from nnarch import __version__


def get_parser():
    """Returns the argument parser object for the command line tool."""
    parser = ArgumentParser(
        version=__version__,
        logger={'name': os.path.basename(__file__), 'level': 'ERROR'},
        description=__doc__)
    parser.add_argument('--propagate',
        action=ActionYesNo,
        default=False,
        help='Whether to also load the module architecture to verify it can be propagated.')
    parser.add_argument('--ext_vars',
        action=ActionJsonnetExtVars,
        help='Path to or string containing a json defining external variables required to load the jsonnet.')
    parser.add_argument('--save_json',
        action=ActionPath(mode='fc'),
        help='Save the parsed file (up to the last successful step: jsonnet load, schema validation, parsing) to the given path in json format.')
    parser.add_argument('jsonnet_path',
        action=ActionPath(mode='fr'),
        help='Path to a neural network module architecture file in jsonnet nnarch format.')
    return parser


def write_json(architecture, cfg):
    """Writes the architecture json to disk if requested."""
    if cfg.save_json is not None:
        with open(cfg.save_json(), 'w') as f:
            f.write(json.dumps(namespace_to_dict(architecture), indent=2, sort_keys=True))


def main(argv=None):
    """Main execution function."""

    ## Helper function ##
    def error_exit(msg=None):
        if msg is not None:
            parser.logger.error(msg)
        sys.exit(1)

    ## Parse arguments ##
    parser = get_parser()
    try:
        cfg = parser.parse_args(sys.argv[1:] if argv is None else argv)
    except:
        error_exit()
    jsonnet_path = cfg.jsonnet_path(absolute=False)

    ## Load jsonnet file ##
    try:
        architecture = ActionJsonnet(schema=None).parse(cfg.jsonnet_path, ext_vars=cfg.ext_vars)
    except Exception as ex:
        error_exit('Failed to load jsonnet file "'+jsonnet_path+'" :: '+str(ex))

    ## Validate jsonnet against schema ##
    try:
        nnarch_validator.validate(namespace_to_dict(architecture))
    except Exception as ex:
        write_json(architecture, cfg)
        error_exit('Architecture file "'+jsonnet_path+'" does not validate against nnarch schema :: '+str(ex))

    ## Propagate shapes of module architecture ##
    if cfg.propagate:
        try:
            load_module_architecture(architecture, propagators=propagators)
        except Exception as ex:
            write_json(architecture, cfg)
            error_exit('Architecture file "'+jsonnet_path+'" failed to load :: '+str(ex))

    ## Write final json ##
    write_json(architecture, cfg)


## Main block called only when run from command line ##
if __name__ == '__main__':
    main()
