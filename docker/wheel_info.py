#!/usr/bin/env python3
"""Command line tool for extracting information from a wheel package."""

# requirements: jsonargparse wheel-inspect

import os
import re
from jsonargparse import ArgumentParser, ActionPath
from wheel_inspect import inspect_wheel


def get_parser():
    parser = ArgumentParser(
        #logger=os.path.basename(__file__),
        error_handler='usage_and_exit_error_handler',
        description=__doc__)
    parser.add_argument('action',
        choices=['project', 'version', 'requirements'],
        help='Type of information to extract from wheel.')
    parser.add_argument('wheel_file',
        action=ActionPath(mode='fr'),
        help='Path to wheel file.')
    parser.add_argument('--extras_require',
        default=[],
        nargs='+',
        help='Names of extras_requires to include in requirements list.')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    cfg = parser.parse_args()
    wheel_info = inspect_wheel(cfg.wheel_file())
    if cfg.action == 'project':
        print(wheel_info['project'])
    elif cfg.action == 'version':
        print(wheel_info['version'].replace('_', '-'))
    elif cfg.action == 'requirements':
        reqs = {}  # type: ignore
        for v in cfg.extras_require:
            if v not in wheel_info['dist_info']['metadata']['provides_extra']:
                parser.logger.warning('extras_require "'+v+'" not defined in wheel package.')
                continue
        re_extra = re.compile('^extra == "(.+)"$')
        for require_info in wheel_info['dist_info']['metadata']['requires_dist']:
            if require_info['marker'] is None or re_extra.sub('\\1', require_info['marker']) in cfg.extras_require:
                if require_info['name'] not in reqs:
                    print(require_info['name']+require_info['specifier'])
                    reqs[require_info['name']] = require_info['specifier']
                elif reqs[require_info['name']] != require_info['specifier']:
                    raise RuntimeError('conflicting requirements for '+require_info['name']+': '+reqs[require_info['name']]+' vs. '+require_info['specifier'])
