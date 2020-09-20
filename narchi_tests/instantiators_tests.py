#!/usr/bin/env python3
"""Unit tests for instantiators."""

# pylint: disable=no-member

import os
import shutil
import tempfile
import unittest
from narchi.instantiators.common import import_object
from narchi_tests.data import *

try:
    import torch
    import numpy as np
    from torch.nn.utils.rnn import pad_packed_sequence
except:
    torch = False

if torch:
    from narchi.instantiators.pytorch import BaseModule, StandardModule, Reshape, standard_pytorch_blocks_mappings
    from narchi.instantiators.pytorch_packed import PackedModule, packed_pytorch_blocks_mappings


@unittest.skipIf(not torch, 'torch package is required')
class PytorchTests(unittest.TestCase):
    """Tests for pytorch instantiator."""

    def test_base_module(self):
        cfg = {'ext_vars': resnet_ext_vars, 'propagators': 'default'}
        mappings = dict(standard_pytorch_blocks_mappings)
        del mappings['Conv2d']

        class TestModule(BaseModule):
            blocks_mappings = mappings

        with torch.no_grad():
            self.assertRaises(NotImplementedError, lambda: TestModule(resnet_jsonnet, cfg=cfg))


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
            logits = module(image=image)
            self.assertEqual(list(logits.shape), [1, 1000])

            imagenet_jsonnet = os.path.join(data_dir, 'imagenet_classifier.jsonnet')
            module2 = StandardModule(imagenet_jsonnet, cfg=cfg)
            module2.eval()
            module2.state_dict_prop = {'classifier.'+k: v for k, v in module.state_dict().items()}
            logits2 = module2(image=image)
            self.assertTrue(torch.all(logits.eq(logits2)))

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
                    logits = module(image=image)
                    torchvision_class = import_object(torchvision[num]['class'])
                    module2 = torchvision_class()
                    module2.eval()
                    module2.load_state_dict(torch.load(state_dict_path))
                    logits2 = module2(image)
                    self.assertTrue(torch.all(logits.eq(logits2)))


    def test_squeezenet(self):
        with torch.no_grad():
            module = StandardModule(squeezenet_jsonnet)
            module.eval()
            image = torch.rand(1, 3, 256, 256)
            logits = module(image=image)
            self.assertEqual(list(logits.shape), [1, 1000])


    def test_text_image_classification(self):
        cfg = {'ext_vars': text_image_ext_vars, 'propagators': 'default'}

        with torch.no_grad():
            module = StandardModule(text_image_jsonnet, cfg=cfg)
            module.eval()
            text = torch.randint(0, 100, (1, 512))
            image = torch.rand(1, 3, 256, 256)
            logits = module(text=text, image=image)
            self.assertEqual([list(p.shape) for p in logits], [[1, 16], [1, 3]])


    def test_laia(self):
        tmpdir = tempfile.mkdtemp(prefix='_narchi_test_')
        cfg = {'ext_vars': laia_ext_vars, 'propagators': 'default'}
        state_dict_path = os.path.join(tmpdir, 'laia.pth')

        with torch.no_grad():
            module = StandardModule(laia_jsonnet, cfg=cfg)
            module.eval()
            image = torch.rand(1, 3, 64, 128)
            logits = module(image=image)
            self.assertEqual(len(logits.shape), 3)
            self.assertEqual(logits.shape[0], 1)
            self.assertEqual(logits.shape[1], 128/8)
            self.assertEqual(logits.shape[2], laia_ext_vars['num_symbols'])

            state_dict = module.state_dict_prop
            torch.save(state_dict, state_dict_path)
            module2 = StandardModule(laia_jsonnet, cfg=cfg, state_dict=state_dict_path)
            module2.eval()
            state_dict2 = module2.state_dict_prop
            self.assertEqual(state_dict.keys(), state_dict2.keys())
            for key in state_dict.keys():
                comp = torch.all(state_dict[key].eq(state_dict2[key]))
                self.assertTrue(comp, 'Reloaded state_dict differs for '+key+'.')
            logits2 = module2(image=image)
            comp = torch.all(logits.eq(logits2))
            self.assertTrue(comp, 'Output from module reloading state_dict differs.')

            self.assertRaises(RuntimeError, lambda: module(image))
            self.assertRaises(RuntimeError, lambda: module(value=image))
            self.assertRaises(RuntimeError, lambda: module(image=torch.rand(64, 128)))

        shutil.rmtree(tmpdir)


    def test_laia_packed(self):
        cfg = {'ext_vars': laia_ext_vars, 'propagators': 'default'}

        with torch.no_grad():
            module = PackedModule(laia_jsonnet, cfg=cfg, gap_size=8, length_fact=8)
            module.eval()
            module2 = StandardModule(laia_jsonnet, cfg=cfg, state_dict=module.state_dict())
            module2.eval()
            #images = [torch.rand(3, 64, 129), torch.rand(3, 64, 91), torch.rand(3, 64, 65)]
            images = [torch.rand(3, 64, 128), torch.rand(3, 64, 96), torch.rand(3, 64, 64)]
            logits_packed = module(image=images)
            logits, lengths = pad_packed_sequence(logits_packed, batch_first=True)

            for num, image in enumerate(images):
                with self.subTest(f'image {num}'):
                    logits2 = module2(image=image.unsqueeze(0)).numpy()
                    self.assertEqual(lengths[num], logits2.shape[1])
                    np.testing.assert_almost_equal(logits2[0,:,:], logits[num,:lengths[num],:].numpy())

            self.assertFalse(hasattr(module, 'intermediate_outputs'))
            module = PackedModule(laia_jsonnet, cfg=cfg, gap_size=8, length_fact=8, debug=True)
            module.eval()
            module(image=images)
            self.assertTrue(hasattr(module, 'intermediate_outputs'))
            intermediate_outputs = list(module.intermediate_outputs.keys())
            self.assertEqual(intermediate_outputs, ['conv1', 'conv2', 'conv3', 'conv4', 'to_1d', 's3blstm', 'fc'])

            class TestModule(BaseModule):
                blocks_mappings = packed_pytorch_blocks_mappings

            module2 = TestModule(laia_jsonnet, cfg=cfg)
            module2.eval()
            module2(image=torch.rand(1, 3, 64, 96))

            unsorted_images = [images[n] for n in [1, 2, 0]]
            self.assertRaises(ValueError, lambda: module(image=unsorted_images))


if __name__ == '__main__':
    unittest.main(verbosity=2)
