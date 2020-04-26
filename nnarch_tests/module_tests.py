#!/usr/bin/env python3
"""Unit tests for modules."""

import os
import json
import shutil
import tempfile
import unittest
from jsonargparse import dict_to_namespace, ParserError
from jsonschema.exceptions import ValidationError
from nnarch.module import ModuleArchitecture, ModulePropagator
from nnarch.register import propagators
from nnarch.bin.nnarch_validate import nnarch_validate


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
        self.assertEqual('laia', module.architecture._id)
        self.assertEqual(laia_shapes, [b._shape.out for b in module.architecture.blocks])
        self.assertRaises(RuntimeError, lambda: module.propagate())

        self.assertRaises(ValueError, lambda: ModuleArchitecture({}, **kwargs))

        module = ModuleArchitecture(laia_jsonnet, **kwargs)
        module.architecture.outputs[0]._shape[0] = '<<variable:W/16>>'
        self.assertRaises(ValueError, lambda: module.propagate())

        module = ModuleArchitecture(laia_jsonnet, **kwargs)
        module.architecture.graph[0] = 'image -- symbprob'
        self.assertRaises(ValidationError, lambda: module.validate())


    def test_nested_modules(self):
        nested1_jsonnet = os.path.join(data_dir, 'nested1.jsonnet')
        nested2_jsonnet = os.path.join(data_dir, 'nested2.jsonnet')
        nested3_jsonnet = os.path.join(data_dir, 'nested3.jsonnet')
        kwargs = {'propagators': propagators, 'cwd': data_dir}

        kwargs['ext_vars'] = {'input_size': 64, 'output_size': 16}
        module = ModuleArchitecture(nested3_jsonnet, **kwargs)
        self.assertEqual({'in': [64], 'out': [16]}, vars(module.architecture._shape))

        kwargs['ext_vars'] = {'input_size': 128, 'nested3_size': 64, 'output_size': 16}
        module = ModuleArchitecture(nested2_jsonnet, **kwargs)
        self.assertEqual({'in': [128], 'out': [16]}, vars(module.architecture._shape))

        kwargs['ext_vars'] = {'hidden_size': 64, 'output_size': 16}
        module = ModuleArchitecture(nested1_jsonnet, **kwargs)
        self.assertEqual({'in': [128], 'out': [16]}, vars(module.architecture._shape))


    def test_validate_cli(self):
        tmpdir = tempfile.mkdtemp(prefix='_nnarch_test_')

        out_json = os.path.join(tmpdir, 'out.json')
        args = ['--save_json', out_json, '--ext_vars', json.dumps(laia_ext_vars), '--propagate', laia_jsonnet]
        nnarch_validate(args)
        with open(out_json) as f:
            architecture = dict_to_namespace(json.loads(f.read()))
        self.assertEqual(laia_shapes, [b._shape.out for b in architecture.blocks])

        self.assertRaises(ParserError, lambda: nnarch_validate([], sys_exit=False))

        with self.assertLogs('nnarch_validate.py'):
            self.assertRaises(ParserError, lambda: nnarch_validate(['--abc'], sys_exit=False))

        bad_jsonnet = os.path.join(tmpdir, 'bad.jsonnet')
        with open(bad_jsonnet, 'w') as f:
            f.write('{}')
        self.assertRaises(ValidationError, lambda: nnarch_validate([bad_jsonnet], sys_exit=False))

        with open(bad_jsonnet, 'w') as f:
            f.write('local local;')
        self.assertRaises(ParserError, lambda: nnarch_validate([bad_jsonnet], sys_exit=False))

        with self.assertLogs('nnarch_validate.py'):
            with open(laia_jsonnet) as f:
                laia = f.read()
            with open(bad_jsonnet, 'w') as f:
                f.write(laia.replace('<<variable:W/8>>', '<<variable:W/4>>'))
            args = ['--save_json', out_json, '--ext_vars', json.dumps(laia_ext_vars), '--propagate', bad_jsonnet]
            self.assertRaises(ValueError, lambda: nnarch_validate(args, sys_exit=False))

        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    unittest.main(verbosity=2)
