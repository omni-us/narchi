#!/usr/bin/env python3
"""Unit tests for modules."""

import os
import unittest
from nnarch.module import load_module_architecture, ModulePropagator
from nnarch.propagators.register import registered_propagators as propagators


data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')


class ModuleTests(unittest.TestCase):
    """Tests for the BasePropagator class."""

    def test_load_module(self):
        laia_jsonnet = os.path.join(data_dir, 'laia.jsonnet')
        ext_vars = {'num_symbols': 68}
        module = load_module_architecture(laia_jsonnet, ext_vars=ext_vars, propagators=propagators)
        expected = [[16, 32, '<<variable:W/2>>'],
                    [16, 16, '<<variable:W/4>>'],
                    [32, 16, '<<variable:W/4>>'],
                    [32, 8, '<<variable:W/8>>'],
                    ['<<variable:W/8>>', 256],
                    ['<<variable:W/8>>', 512],
                    ['<<variable:W/8>>', 68]]
        self.assertEqual(expected, [b._shape.out for b in module.architecture.blocks])


if __name__ == '__main__':
    unittest.main(verbosity=2)
