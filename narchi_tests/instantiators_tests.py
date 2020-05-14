#!/usr/bin/env python3
"""Unit tests for instantiators."""

# pylint: disable=no-member

import os
import shutil
import tempfile
import unittest
from narchi.instantiators.common import import_object
from narchi_tests.module_tests import data_dir, resnet_jsonnet, resnet_ext_vars, laia_jsonnet, laia_ext_vars

try:
    import torch
    from narchi.instantiators.pytorch import StandardModule, Reshape
except:
    torch = False


@unittest.skipIf(not torch, 'torch package is required')
class PytorchTests(unittest.TestCase):
    """Tests for pytorch instantiator."""

    def test_reshape(self):
        ## successes ##
        examples = [
            {
                'in_shape': [1, 16, 8, 4, 2],
                'reshape_spec': [3, 2, 0, 1],
                'expected': [1, 2, 4, 16, 8],
            },
            {
                'in_shape': [2, 12, 8, 7, 2],
                'reshape_spec': [[3, 2], [0, 1]],
                'expected': [2, 14, 96],
            },
            {
                'in_shape': [7, 4608],
                'reshape_spec': [{'0': [3, 48, '<<auto>>']}],
                'expected': [7, 3, 48, 32],
            },
            {
                'in_shape': [1, 4608, 9],
                'reshape_spec': [1, {'0': [3, '<<auto>>', 32]}],
                'expected': [1, 9, 3, 48, 32],
            },
        ]
        for example in examples:
            reshape = Reshape(example['reshape_spec'])
            x = torch.rand(*example['in_shape'])
            y = reshape(x)
            self.assertEqual(list(y.shape), example['expected'])

        ## failures ##
        examples = [
            {
                'in_shape': [1, 4608, 9],
                'reshape_spec': [{'0': [3, 48, '<<auto>>']}],
            },
        ]
        for example in examples:
            reshape = Reshape(example['reshape_spec'])
            x = torch.rand(*example['in_shape'])
            self.assertRaises(RuntimeError, lambda: reshape(x))


    def test_resnet(self):
        cfg = {'ext_vars': resnet_ext_vars, 'propagators': 'default'}

        with torch.no_grad():
            module = StandardModule(resnet_jsonnet, cfg=cfg)
            module.eval()
            image = torch.rand(1, 3, 256, 256)
            classprob = module(image=image)
            self.assertEqual(list(classprob.shape), [1, 1000])

            imagenet_jsonnet = os.path.join(data_dir, 'imagenet_classifier.jsonnet')
            module2 = StandardModule(imagenet_jsonnet, cfg=cfg)
            module2.eval()
            module2.state_dict_prop = {'classifier.'+k: v for k, v in module.state_dict().items()}
            classprob2 = module2(image=image)
            self.assertTrue(torch.all(classprob.eq(classprob2)))

            torchvision = [
                {
                    'pth': 'resnet18-5c106cde.pth',
                    'class': 'torchvision.models.resnet.resnet18',
                    'num_blocks': [2, 2, 2, 2],
                },
                {
                    'pth': 'resnet34-333f7ec4.pth',
                    'class': 'torchvision.models.resnet.resnet34',
                    'num_blocks': [3, 4, 6, 3],
                },
            ]
            for num in range(len(torchvision)):
                state_dict_path = os.path.join(data_dir, torchvision[num]['pth'])
                if os.path.isfile(state_dict_path):
                    cfg['ext_vars']['num_blocks'] = torchvision[num]['num_blocks']
                    module = StandardModule(resnet_jsonnet, cfg=cfg, state_dict=state_dict_path)
                    module.eval()
                    classprob = module(image=image)
                    torchvision_class = import_object(torchvision[num]['class'])
                    module2 = torchvision_class()
                    module2.eval()
                    module2.load_state_dict(torch.load(state_dict_path))
                    classprob2 = module2(image)
                    self.assertTrue(torch.all(classprob.eq(classprob2)))


    def test_laia(self):
        tmpdir = tempfile.mkdtemp(prefix='_narchi_test_')
        cfg = {'ext_vars': laia_ext_vars, 'propagators': 'default'}
        state_dict_path = os.path.join(tmpdir, 'laia.pth')

        with torch.no_grad():
            module = StandardModule(laia_jsonnet, cfg=cfg)
            module.eval()
            image = torch.rand(1, 3, 64, 128)
            symbprob = module(image=image)
            self.assertEqual(len(symbprob.shape), 3)
            self.assertEqual(symbprob.shape[0], 1)
            self.assertEqual(symbprob.shape[1], 128/8)
            self.assertEqual(symbprob.shape[2], laia_ext_vars['num_symbols'])

            state_dict = module.state_dict_prop
            torch.save(state_dict, state_dict_path)
            module2 = StandardModule(laia_jsonnet, cfg=cfg, state_dict=state_dict_path)
            module2.eval()
            state_dict2 = module2.state_dict_prop
            self.assertEqual(state_dict.keys(), state_dict2.keys())
            for key in state_dict.keys():
                comp = torch.all(state_dict[key].eq(state_dict2[key]))
                self.assertTrue(comp, 'Reloaded state_dict differs for '+key+'.')
            symbprob2 = module2(image=image)
            comp = torch.all(symbprob.eq(symbprob2))
            self.assertTrue(comp, 'Output from module reloading state_dict differs.')

            self.assertRaises(RuntimeError, lambda: module(image))
            self.assertRaises(RuntimeError, lambda: module(value=image))
            self.assertRaises(RuntimeError, lambda: module(image=torch.rand(64, 128)))

        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    unittest.main(verbosity=2)
