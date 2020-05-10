#!/usr/bin/env python3
"""Unit tests for modules."""

import os
import unittest
from jsonargparse import ParserError
from jsonschema.exceptions import ValidationError
from narchi.module import ModuleArchitecture, ModulePropagator
from narchi.register import propagators


data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

resnet_jsonnet = os.path.join(data_dir, 'resnet.jsonnet')
resnet_ext_vars = {'num_blocks': [2, 2, 2, 2]}

laia_jsonnet = os.path.join(data_dir, 'laia.jsonnet')
laia_ext_vars = {'num_symbols': 68}
laia_shapes = [[16, 32, '<<variable:W/2>>'],
               [16, 16, '<<variable:W/4>>'],
               [32, 16, '<<variable:W/4>>'],
               [32, 8, '<<variable:W/8>>'],
               ['<<variable:W/8>>', 256],
               ['<<variable:W/8>>', 512],
               ['<<variable:W/8>>', 68]]

nested1_jsonnet = os.path.join(data_dir, 'nested1.jsonnet')
nested2_jsonnet = os.path.join(data_dir, 'nested2.jsonnet')
nested3_jsonnet = os.path.join(data_dir, 'nested3.jsonnet')
nested1_ext_vars = {'hidden_size': 64, 'output_size': 16}
nested2_ext_vars = {'input_size': 128, 'nested3_size': 64, 'output_size': 16}
nested3_ext_vars = {'input_size': 64, 'output_size': 16}


class ModuleTests(unittest.TestCase):
    """Tests for the ModuleArchitecture class."""

    def test_module_architecture_init(self):
        cfg = {'ext_vars': laia_ext_vars, 'propagators': propagators}

        module = ModuleArchitecture(laia_jsonnet, cfg=cfg)
        self.assertEqual(laia_shapes, [b._shape.out for b in module.architecture.blocks])

        cfg['propagate'] = False
        module = ModuleArchitecture(laia_jsonnet, cfg=cfg)
        module.architecture.outputs[0]._shape[0] = '<<auto>>'
        module.propagate()
        self.assertEqual('laia', module.architecture._id)
        self.assertEqual(laia_shapes, [b._shape.out for b in module.architecture.blocks])
        self.assertRaises(RuntimeError, lambda: module.propagate())

        self.assertRaises(ValueError, lambda: ModuleArchitecture({}, cfg=cfg))

        module = ModuleArchitecture(laia_jsonnet, cfg=cfg)
        module.architecture.outputs[0]._shape[0] = '<<variable:W/16>>'
        self.assertRaises(ValueError, lambda: module.propagate())

        module = ModuleArchitecture(laia_jsonnet, cfg=cfg)
        module.architecture.graph[0] = 'image -- symbprob'
        self.assertRaises(ValidationError, lambda: module.validate())

        self.assertRaises(RuntimeError, lambda: ModuleArchitecture(laia_jsonnet, cfg={'ext_vars': laia_ext_vars}))

        self.assertRaises(ParserError, lambda: ModuleArchitecture(laia_jsonnet, cfg={'propagators': propagators}))
        self.assertRaises(ParserError, lambda: ModuleArchitecture(laia_jsonnet, cfg=None))


    def test_nested_modules(self):
        cfg = {'propagators': propagators, 'cwd': data_dir}

        cfg['ext_vars'] = nested3_ext_vars
        module = ModuleArchitecture(nested3_jsonnet, cfg=cfg)
        self.assertEqual({'in': [64], 'out': [16]}, vars(module.architecture._shape))

        cfg['ext_vars'] = nested2_ext_vars
        module = ModuleArchitecture(nested2_jsonnet, cfg=cfg)
        self.assertEqual({'in': [128], 'out': [16]}, vars(module.architecture._shape))

        cfg['ext_vars'] = nested1_ext_vars
        module = ModuleArchitecture(nested1_jsonnet, cfg=cfg)
        self.assertEqual({'in': [128], 'out': [16]}, vars(module.architecture._shape))


if __name__ == '__main__':
    unittest.main(verbosity=2)
