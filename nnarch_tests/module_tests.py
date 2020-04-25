#!/usr/bin/env python3
"""Unit tests for modules."""

import os
import json
import shutil
import tempfile
import unittest
from jsonargparse import dict_to_namespace
from jsonschema.exceptions import ValidationError
from nnarch.module import ModuleArchitecture, ModulePropagator
from nnarch.register import propagators
from nnarch.bin.nnarch_validate import main as nnarch_validate


data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

laia_jsonnet = os.path.join(data_dir, 'laia.jsonnet')
laia_ext_vars = {'num_symbols': 68}
laia_shapes = [[16, 32, '<<variable:W/2>>'],
               [16, 16, '<<variable:W/4>>'],
               [32, 16, '<<variable:W/4>>'],
               [32, 8, '<<variable:W/8>>'],
               ['<<variable:W/8>>', 256],
               ['<<variable:W/8>>', 512],
               ['<<variable:W/8>>', 68]]


class ModuleTests(unittest.TestCase):
    """Tests for the BasePropagator class."""

    def test_module_architecture_init(self):
        kwargs = {'ext_vars': laia_ext_vars, 'propagators': propagators}

        module = ModuleArchitecture(laia_jsonnet, **kwargs)
        self.assertEqual(laia_shapes, [b._shape.out for b in module.architecture.blocks])

        kwargs['propagate'] = False
        module = ModuleArchitecture(laia_jsonnet, **kwargs)
        module.architecture.outputs[0]._shape[0] = '<<auto>>'
        module.propagate()
        self.assertEqual(laia_shapes, [b._shape.out for b in module.architecture.blocks])

        self.assertRaises(ValueError, lambda: ModuleArchitecture({}, **kwargs))

        module = ModuleArchitecture(laia_jsonnet, **kwargs)
        module.architecture.outputs[0]._shape[0] = '<<variable:W/16>>'
        self.assertRaises(ValueError, lambda: module.propagate())

        module = ModuleArchitecture(laia_jsonnet, **kwargs)
        module.architecture.graph[0] = 'image -- symbprob'
        self.assertRaises(ValidationError, lambda: module.validate())


    def test_validate_cli(self):
        tmpdir = tempfile.mkdtemp(prefix='_nnarch_test_')
        out_json = os.path.join(tmpdir, 'laia.json')
        args = ['--save_json', out_json, '--ext_vars', json.dumps(laia_ext_vars), '--propagate', laia_jsonnet]
        nnarch_validate(args)
        with open(out_json) as f:
            architecture = dict_to_namespace(json.loads(f.read()))
        self.assertEqual(laia_shapes, [b._shape.out for b in architecture.blocks])
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    unittest.main(verbosity=2)
