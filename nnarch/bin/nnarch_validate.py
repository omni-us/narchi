#!/usr/bin/env python
"""Command line tool for validating neural network architecture jsonnet files."""

import json
from yamlargparse import ArgumentParser, ActionPath, ActionJsonnet, ActionJsonSchema, ActionYesNo, namespace_to_dict
from nnarch.parse import parse_architecture, nnarch_validator


def get_parser():
    """Returns the argument parser object for the command line tool."""
    parser = ArgumentParser(
        error_handler='usage_and_exit_error_handler',
        description=__doc__)
    parser.add_argument('--parse_architecture',
        action=ActionYesNo,
        default=False,
        help='Whether to also parse the loaded architecture to verify that completion.')
    parser.add_argument('--const',
        action=ActionJsonSchema(schema={'type': 'object'}),
        default={},
        help='Path to or string containing a json defining constants required by the architecture.')
    parser.add_argument('--output',
        help='Set to save the parsed file to given path in json format.')
    parser.add_argument('input',
        action=ActionPath(mode='fr'),
        help='Path to a neural network architecture file in jsonnet nnarch format.')
    return parser


def write_output(architecture, cfg):
    if cfg.output is not None:
        with open(cfg.output, 'w') as f:
            f.write(json.dumps(namespace_to_dict(architecture), indent=2, sort_keys=True))


## Main block called only when run from command line ##
if __name__ == '__main__':
    ## Parse arguments ##
    cfg = get_parser().parse_args()
    jsonnet_path = cfg.input(absolute=False)

    ## Load jsonnet file ##
    try:
        architecture = ActionJsonnet(schema=None).parse(jsonnet_path)
    except Exception as ex:
        raise type(ex)('Failed to load jsonnet file "'+jsonnet_path+'" :: '+str(ex))

    ## Validate jsonnet against schema ##
    try:
        nnarch_validator.validate(namespace_to_dict(architecture))
    except Exception as ex:
        write_output(architecture, cfg)
        raise type(ex)('Architecture file "'+jsonnet_path+'" does not validate against nnarch schema :: '+str(ex))

    ## Parse architecture ##
    if cfg.parse_architecture:
        try:
            architecture, _, _ = parse_architecture(architecture, const=namespace_to_dict(cfg.const))
        except Exception as ex:
            write_output(architecture, cfg)
            raise type(ex)('Architecture file "'+jsonnet_path+'" failed to parse :: '+str(ex))

    ## Write output ##
    write_output(architecture, cfg)
