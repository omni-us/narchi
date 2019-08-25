#!/usr/bin/env python

from setuptools import setup, find_packages, Command
import subprocess
from glob import glob
import re

try:
    from sphinx.setup_command import BuildDoc
except ImportError:
    BuildDoc = False
    print('warning: sphinx not found, build_sphinx target will not be available.')


NAME = 'nnarch'
AUTHOR = 'Mauricio Villegas'
AUTHOR_EMAIL = 'mauricio@omnius.com'
DESCRIPTION = 'nnarch package'
LONG_DESCRIPTION = re.sub(':class:|:func:|:ref:', '', open('README.rst').read())


class CoverageCommand(Command):
    description = 'print test coverage report'
    user_options = []  # type: ignore
    def initialize_options(self): pass
    def finalize_options(self): pass
    def run(self):
        subprocess.check_call(['python', '-m', 'coverage', 'run', '--source', NAME, 'setup.py', 'test'])
        subprocess.check_call(['python', '-m', 'coverage', 'report', '-m'])


CMDCLASS = {'test_coverage': CoverageCommand}
if BuildDoc:
    CMDCLASS['build_sphinx'] = BuildDoc


def get_runtime_requirements():
    """Returns a list of required packages filtered to include only the ones necessary at runtime."""
    with open('requirements.txt') as f:
        requirements = [x.strip() for x in f.readlines()]
    regex = re.compile('^(coverage|pylint|pycodestyle|mypy|sphinx|autodocsumm)', re.IGNORECASE)
    return [x for x in requirements if not regex.match(x)]


setup(name=NAME,
      version=__import__(NAME).__version__,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      packages=find_packages(),
      scripts=[x for x in glob(NAME+'/bin/*.py') if not x.endswith('__.py')],
      python_requires='>=3.5',
      install_requires=get_runtime_requirements(),
      test_suite=NAME+'_tests',
      cmdclass=CMDCLASS,
      command_options={
          'build_sphinx': {
              'project': ('setup.py', NAME),
              'version': ('setup.py', 'local build'),
              'release': ('setup.py', 'local build'),
              'build_dir': ('setup.py', 'sphinx/_build'),
              'source_dir': ('setup.py', 'sphinx')}})
