#!/usr/bin/env python3
"""Unit tests for instantiators."""

import unittest
from nnarch.instantiators.pytorch import StandardModule
from nnarch_tests.module_tests import laia_jsonnet, laia_ext_vars

try:
    import torch
    from torch import rand
except:
    torch = False


@unittest.skipIf(not torch, 'torch package is required')
class PytorchStandardModuleTests(unittest.TestCase):
    """Tests for the pytorch.StandardModule class."""

    def test_laia(self):
        cfg = {'ext_vars': laia_ext_vars, 'propagators': 'default'}
        module = StandardModule(laia_jsonnet, cfg=cfg)
        image = rand(1, 3, 64, 128)
        symbprob = module(image=image)
        self.assertEqual(symbprob.shape[0], 1)
        self.assertEqual(symbprob.shape[1], 128/8)
        self.assertEqual(symbprob.shape[2], laia_ext_vars['num_symbols'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
