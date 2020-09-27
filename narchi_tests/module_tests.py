#!/usr/bin/env python3
"""Unit tests for modules."""

import os
import shutil
import tempfile
import unittest
from jsonargparse import ParserError
from jsonschema.exceptions import ValidationError
from narchi.module import ModuleArchitecture
from narchi.blocks import propagators
from narchi.schemas import auto_tag
from narchi_tests.data import *


class ModuleArchitectureTests(unittest.TestCase):
    """Tests for the ModuleArchitecture class."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='_narchi_test_')
        self.laia_out_json = os.path.join(self.tmpdir, 'laia.json')


    def tearDown(self):
        shutil.rmtree(self.tmpdir)


    def test_failed_class_init(self):
        self.assertRaises(ValueError, lambda: ModuleArchitecture({}, cfg=laia_cfg))
        self.assertRaises(ParserError, lambda: ModuleArchitecture(laia_jsonnet, cfg=None))
        self.assertRaises(ValueError, lambda: ModuleArchitecture(laia_jsonnet, cfg=False))
        self.assertRaises(TypeError, lambda: ModuleArchitecture(laia_jsonnet, cfg='/tmp'))


    def test_propagation_success_failure(self):
        cfg = dict(laia_cfg)
        cfg['propagate'] = False

        module = ModuleArchitecture(laia_jsonnet, cfg=cfg)
        module.propagate()
        self.assertEqual(laia_shapes, [b._shape.out for b in module.architecture.blocks])

        module = ModuleArchitecture(laia_jsonnet, cfg=cfg)
        module.architecture.outputs[0]._shape[0] = auto_tag
        module.propagate()
        self.assertEqual('laia', module.architecture._id)
        self.assertEqual(laia_shapes, [b._shape.out for b in module.architecture.blocks])
        self.assertRaises(RuntimeError, lambda: module.propagate())

        orig_jsonnet = module.jsonnet
        tmp_laia_jsonnet = os.path.join(self.tmpdir, 'laia.jsonnet')
        with open(tmp_laia_jsonnet, 'w') as f:
            f.write(orig_jsonnet.replace('fc -> logits', 'logits -> fc'))
        module = ModuleArchitecture(tmp_laia_jsonnet, cfg=cfg)
        self.assertRaises(ValueError, lambda: module.propagate())

        with open(tmp_laia_jsonnet, 'w') as f:
            f.write(orig_jsonnet.replace('fc -> logits', 'fc').replace('logits', 'fc'))
        self.assertRaises(ValueError, lambda: ModuleArchitecture(tmp_laia_jsonnet, cfg=cfg))

        module = ModuleArchitecture(laia_jsonnet, cfg=cfg)
        module.architecture.outputs[0]._shape[0] = '<<variable:W/16>>'
        self.assertRaises(ValueError, lambda: module.propagate())


    def test_validation_fail(self):
        cfg = dict(laia_cfg)
        cfg['propagate'] = False
        module = ModuleArchitecture(laia_jsonnet, cfg=cfg)
        module.architecture.graph[0] = 'image -- logits'
        self.assertRaises(ValidationError, lambda: module.validate())


    def test_save_json_write_overwrite(self):
        cfg = dict(laia_cfg)
        cfg.update({'outdir': self.tmpdir, 'save_json': True})
        ModuleArchitecture(laia_jsonnet, cfg=cfg)
        self.assertTrue(os.path.isfile(self.laia_out_json))
        self.assertRaises(IOError, lambda: ModuleArchitecture(laia_jsonnet, cfg=cfg))
        cfg['overwrite'] = True
        ModuleArchitecture(laia_jsonnet, cfg=cfg)
        os.remove(self.laia_out_json)


    def test_custom_block_undefined_defined(self):
        cfg = dict(laia_cfg)
        cfg.update({'outdir': self.tmpdir, 'save_json': True, 'propagate': False})
        module = ModuleArchitecture(laia_jsonnet, cfg=cfg)
        custom_block = module.architecture.blocks[0]
        custom_propagators = dict(propagators)
        custom_propagators['Custom'] = propagators[custom_block._class]
        custom_block._class = 'Custom'
        self.assertRaises(ValueError, lambda: module.propagate())
        self.assertTrue(os.path.isfile(self.laia_out_json))
        os.remove(self.laia_out_json)
        cfg.update({'save_json': False, 'propagators': custom_propagators})
        ModuleArchitecture(laia_jsonnet, cfg=cfg).propagate()


    def test_multi_input_output(self):
        cfg = dict(text_image_cfg)

        module = ModuleArchitecture(text_image_jsonnet, cfg=cfg)
        self.assertEqual([i._shape for i in module.architecture.inputs], [['<<variable:L>>'], [3, '<<variable:H>>', '<<variable:W>>']])
        self.assertEqual([o._shape for o in module.architecture.outputs], [[16], [3]])

        del cfg['ext_vars']
        self.assertRaises(ParserError, lambda: ModuleArchitecture(text_image_jsonnet, cfg=cfg))


    def test_nested_modules(self):
        cfg = {'ext_vars': nested3_ext_vars}
        module = ModuleArchitecture(nested3_jsonnet, cfg=cfg)
        self.assertEqual({'in': [64], 'out': [16]}, vars(module.architecture._shape))

        cfg = {'ext_vars': nested2_ext_vars}
        module = ModuleArchitecture(nested2_jsonnet, cfg=cfg)
        self.assertEqual({'in': [128], 'out': [16]}, vars(module.architecture._shape))

        cfg = {'ext_vars': nested1_ext_vars}
        module = ModuleArchitecture(nested1_jsonnet, cfg=cfg)
        self.assertEqual({'in': [128], 'out': [16]}, vars(module.architecture._shape))


if __name__ == '__main__':
    unittest.main(verbosity=2)
