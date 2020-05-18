"""Run all unit tests in package."""

import os
import sys
import unittest


testing_package = os.path.basename(os.path.dirname(os.path.realpath(__file__)))


def run_tests():
    tests = unittest.defaultTestLoader.discover(testing_package, pattern='*_tests.py')
    if not unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful():
        sys.exit(True)


def run_test_coverage():
    try:
        import coverage
    except:
        print('error: coverage package not found, run_test_coverage requires it.')
        sys.exit(True)
    package_source = os.path.dirname(__import__(testing_package.replace('_tests', '')).__file__)
    cwd = os.getcwd()
    os.chdir(os.path.dirname(__file__))
    cov = coverage.Coverage(source=[package_source], data_file=os.path.join(cwd, '.coverage'))
    cov.start()
    run_tests()
    cov.stop()
    cov.save()
    cov.report()
    if 'xml' in sys.argv:
        outfile = os.path.join(cwd, sys.argv[sys.argv.index('xml')+1])
        cov.xml_report(outfile=outfile)
        print('\nSaved coverage report to '+outfile+'.')
    else:
        cov.html_report(directory='htmlcov')
        print('\nSaved html coverage report to htmlcov directory.')
    os.chdir(cwd)


if __name__ == '__main__':
    if 'coverage' in sys.argv:
        run_test_coverage()
    else:
        run_tests()
