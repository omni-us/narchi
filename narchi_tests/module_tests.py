#!/usr/bin/env python3
"""Unit tests for modules."""

import os
import shutil
import tempfile
import unittest
from jsonargparse import ParserError
from jsonschema.exceptions import ValidationError
from narchi.module import ModuleArchitecture, ModulePropagator
from narchi.blocks import propagators


data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')

resnet_jsonnet = os.path.join(data_dir, 'resnet.jsonnet')
resnet_ext_vars = {'num_blocks': [2, 2, 2, 2]}

text_image_jsonnet = os.path.join(data_dir, 'text_image_classification.jsonnet')
text_image_ext_vars = {'vocabulary_size': 5000}

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
nested2_jsonnet = os.path.join(data_dir, 'nested/nested/nested2.jsonnet')
nested3_jsonnet = os.path.join(data_dir, 'nested/nested3.jsonnet')
nested1_ext_vars = {'hidden_size': 64, 'output_feats': 16}
nested2_ext_vars = {'input_size': 128, 'nested3_size': 64, 'output_feats': 16}
nested3_ext_vars = {'input_size': 64, 'output_feats': 16}


class ModuleTests(unittest.TestCase):
    """Tests for the ModuleArchitecture class."""

    def test_module_architecture_init(self):
        tmpdir = tempfile.mkdtemp(prefix='_narchi_test_')

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

        with open(os.path.join(tmpdir, 'laia.jsonnet'), 'w') as f:
            f.write(module.jsonnet.replace('fc -> symbprob', 'symbprob -> fc'))
        module = ModuleArchitecture(os.path.join(tmpdir, 'laia.jsonnet'), cfg=cfg)
        self.assertRaises(ValueError, lambda: module.propagate())

        module = ModuleArchitecture(laia_jsonnet, cfg=cfg)
        module.architecture.outputs[0]._shape[0] = '<<variable:W/16>>'
        self.assertRaises(ValueError, lambda: module.propagate())

        module = ModuleArchitecture(laia_jsonnet, cfg=cfg)
        module.architecture.graph[0] = 'image -- symbprob'
        self.assertRaises(ValidationError, lambda: module.validate())

        self.assertRaises(RuntimeError, lambda: ModuleArchitecture(laia_jsonnet, cfg={'ext_vars': laia_ext_vars}))

        self.assertRaises(ParserError, lambda: ModuleArchitecture(laia_jsonnet, cfg={'propagators': propagators}))
        self.assertRaises(ParserError, lambda: ModuleArchitecture(laia_jsonnet, cfg=None))
        self.assertRaises(ValueError, lambda: ModuleArchitecture(laia_jsonnet, cfg=False))
        self.assertRaises(TypeError, lambda: ModuleArchitecture(laia_jsonnet, cfg='/tmp'))

        cfg.update({'outdir': tmpdir, 'save_json': True})
        module = ModuleArchitecture(laia_jsonnet, cfg=cfg)
        module.architecture.blocks[0]._class = 'Unk'
        self.assertRaises(ValueError, lambda: module.propagate())
        assert os.path.isfile(os.path.join(tmpdir, 'laia.json'))
        self.assertRaises(IOError, lambda: ModuleArchitecture(laia_jsonnet, cfg=cfg).propagate())

        shutil.rmtree(tmpdir)


    def test_multi_input_output(self):
        cfg = {'ext_vars': text_image_ext_vars, 'propagators': propagators}

        module = ModuleArchitecture(text_image_jsonnet, cfg=cfg)
        self.assertEqual([i._shape for i in module.architecture.inputs], [['<<variable:L>>'], [3, '<<variable:H>>', '<<variable:W>>']])
        self.assertEqual([o._shape for o in module.architecture.outputs], [[16], [3]])

        del cfg['ext_vars']
        self.assertRaises(ParserError, lambda: ModuleArchitecture(text_image_jsonnet, cfg=cfg))


    def test_nested_modules(self):
        cfg = {'propagators': propagators}

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
