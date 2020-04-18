#!/usr/bin/env python3

from setuptools import setup, Command
from glob import glob
import sys


NAME = next(filter(lambda x: x.startswith('name = '), open('setup.cfg').readlines())).strip().split()[-1]
NAME_TESTS = next(filter(lambda x: x.startswith('test_suite = '), open('setup.cfg').readlines())).strip().split()[-1]
CMDCLASS = {}


## test_coverage target ##
try:
    import coverage

    class CoverageCommand(Command):
        description = 'print test coverage report'
        user_options = []  # type: ignore
        def initialize_options(self): pass
        def finalize_options(self): pass
        def run(self):
            cov = coverage.Coverage()
            cov.start()
            rc = __import__(NAME_TESTS+'.__main__').__main__.run_tests().wasSuccessful()
            if not rc:
                sys.exit(not rc)
            cov.stop()
            cov.save()
            cov.report()
            cov.html_report(directory='htmlcov')
            print('\nSaved html report to htmlcov directory.')

    CMDCLASS['test_coverage'] = CoverageCommand

except Exception:
    print('warning: coverage package not found, test_coverage target will not be available.')


## build_sphinx target ##
try:
    from sphinx.setup_command import BuildDoc
    CMDCLASS['build_sphinx'] = BuildDoc

except Exception:
    print('warning: sphinx package not found, build_sphinx target will not be available.')


## Run setuptools setup ##
setup(version=__import__(NAME+'.__init__').__version__,
      scripts=[x for x in glob(NAME+'/bin/*.py') if not x.endswith('__.py')],
      package_data={NAME_TESTS+'.data': ['*.jsonnet']},
      cmdclass=CMDCLASS)
