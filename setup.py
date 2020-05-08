#!/usr/bin/env python3

from setuptools import setup, Command
from glob import glob


NAME = next(filter(lambda x: x.startswith('name = '), open('setup.cfg').readlines())).strip().split()[-1]
NAME_TESTS = next(filter(lambda x: x.startswith('test_suite = '), open('setup.cfg').readlines())).strip().split()[-1]
CMDCLASS = {}


## test_coverage target ##
class CoverageCommand(Command):
    description = 'run test coverage and generate html report'
    user_options = []  # type: ignore
    def initialize_options(self): pass
    def finalize_options(self): pass
    def run(self):
        __import__(NAME_TESTS+'.__main__').__main__.run_test_coverage()

CMDCLASS['test_coverage'] = CoverageCommand


## build_sphinx target ##
try:
    from sphinx.setup_command import BuildDoc
    CMDCLASS['build_sphinx'] = BuildDoc

except Exception:
    print('warning: sphinx package not found, build_sphinx target will not be available.')


## Run setuptools setup ##
setup(package_data={NAME_TESTS+'.data': ['*.jsonnet']},
      cmdclass=CMDCLASS)
