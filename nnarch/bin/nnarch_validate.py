#!/usr/bin/env python3
"""Command line tool for validating neural network architecture jsonnet files."""

import json
from jsonargparse import ArgumentParser, ActionPath, ActionJsonnet, ActionJsonnetExtVars, ActionYesNo, namespace_to_dict
from nnarch.schema import nnarch_validator
from nnarch.module import load_module_architecture
from nnarch.propagators.register import registered_propagators as propagators
from nnarch import __version__


def get_parser():
    """Returns the argument parser object for the command line tool."""
    parser = ArgumentParser(
        version=__version__,
        error_handler='usage_and_exit_error_handler',
        description=__doc__)
    parser.add_argument('--load',
        action=ActionYesNo,
        default=False,
        help='Whether to also load the module architecture to verify it can be propagated.')
    parser.add_argument('--ext_vars',
        action=ActionJsonnetExtVars,
        help='Path to or string containing a json defining external variables required to load the jsonnet.')
    parser.add_argument('--save_json',
        action=ActionPath(mode='fc'),
        help='Save the parsed file (up to the last successful step: jsonnet load, schema validation, parsing) to the given path in json format.')
    parser.add_argument('input',
        action=ActionPath(mode='fr'),
        help='Path to a neural network module architecture file in jsonnet nnarch format.')
    return parser


def write_json(architecture, cfg):
    if cfg.save_json is not None:
        with open(cfg.save_json(), 'w') as f:
            f.write(json.dumps(namespace_to_dict(architecture), indent=2, sort_keys=True))


## Main block called only when run from command line ##
if __name__ == '__main__':
    ## Parse arguments ##
    cfg = get_parser().parse_args()
    jsonnet_path = cfg.input(absolute=False)

    ## Load jsonnet file ##
    try:
        architecture = ActionJsonnet(schema=None).parse(cfg.input, ext_vars=cfg.ext_vars)
    except Exception as ex:
        raise type(ex)('Failed to load jsonnet file "'+jsonnet_path+'" :: '+str(ex))

    ## Validate jsonnet against schema ##
    try:
        nnarch_validator.validate(namespace_to_dict(architecture))
    except Exception as ex:
        write_json(architecture, cfg)
        raise type(ex)('Architecture file "'+jsonnet_path+'" does not validate against nnarch schema :: '+str(ex))

    ## Load module architecture ##
    if cfg.load:
        try:
            module = load_module_architecture(architecture, propagators=propagators)
            architecture = module.architecture
        except Exception as ex:
            write_json(architecture, cfg)
            raise type(ex)('Architecture file "'+jsonnet_path+'" failed to load :: '+str(ex))

    ## Write final json ##
    write_json(architecture, cfg)
