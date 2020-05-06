"""Run all unit tests in package."""

import os
import sys
import unittest


def run_tests():
    package = os.path.basename(os.path.dirname(os.path.realpath(__file__)))
    tests = unittest.defaultTestLoader.discover(package, pattern='*_tests.py')
    if not unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful():
        sys.exit(True)


def run_test_coverage():
    try:
        import coverage
    except:
        print('error: coverage package not found, run_test_coverage requires it.')
        sys.exit(True)
    cov = coverage.Coverage()
    cov.start()
    run_tests()
    cov.stop()
    cov.save()
    cov.report()
    cov.html_report(directory='htmlcov')
    print('\nSaved html coverage report to htmlcov directory.')


if __name__ == '__main__':
    if 'coverage' in sys.argv[1:]:
        run_test_coverage()
    else:
        run_tests()
