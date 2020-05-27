#!/usr/bin/env python3
"""Unit tests for propagation classes."""

import unittest
from copy import deepcopy
from jsonargparse import dict_to_namespace as d2n
from narchi.blocks import propagators, register_propagator
from narchi.propagators.base import BasePropagator, get_shape, create_shape
from narchi.propagators.concat import ConcatenatePropagator
from narchi.propagators.conv import ConvPropagator, PoolPropagator
from narchi.propagators.fixed import AddFixedPropagator, FixedOutputPropagator
from narchi.propagators.group import SequentialPropagator
from narchi.propagators.reshape import ReshapePropagator
from narchi.propagators.rnn import RnnPropagator
from narchi.propagators.same import SameShapePropagator, SameShapesPropagator
from narchi.graph import parse_graph


class BasePropagatorTests(unittest.TestCase):
    """Tests for the BasePropagator class."""

    def test_propagators_dict(self):
        self.assertRaises(ValueError, lambda: register_propagator(FixedOutputPropagator('Linear')))
        self.assertRaises(ValueError, lambda: register_propagator(FixedOutputPropagator('Default')))
        register_propagator(FixedOutputPropagator('Linear'), replace=True)
        for propagator in propagators.values():
            self.assertTrue(isinstance(propagator, BasePropagator))


    def test_base_propagator(self):
        propagator = BasePropagator('Base')

        examples = [
            {
                'from': [{}],
                'to': d2n({'_id': 'b2', '_class': 'Base'}),
            },
            {
                'from': [d2n({'_id': 'b1'})],
                'to': d2n({'_id': 'b2', '_class': 'Base'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5]}})],
                'to': d2n({}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5]}})],
                'to': d2n({'_id': 'b2', '_class': 'Base2'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5]}})],
                'to': d2n({'_id': 'b2', '_class': 'Base', '_shape': {'in': [5], 'out': [5]}}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<auto>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Base'}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator.initial_checks(example['from'], example['to']))

        self.assertRaises(NotImplementedError, lambda: propagator.propagate(None, None))


    def test_bad_subclass(self):
        class BadPropagator(BasePropagator):
            def propagate(self, from_blocks, block):
                shape_in = get_shape('out', from_blocks[0])
                if any(isinstance(x, int) for x in shape_in):
                    block._shape = create_shape(shape_in, ['<<auto>>'])
                else:
                    block._shape = create_shape(['<<variable:V>>'], shape_in)

        propagator = BadPropagator('Bad')

        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [16]}})],
                'to': d2n({'_id': 'b2', '_class': 'Bad'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:Z>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Bad'}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


class AddFixedPropagatorTests(unittest.TestCase):
    """Tests for the AddFixedPropagator class."""

    def test_initializer(self):
        self.assertRaises(ValueError, lambda: AddFixedPropagator('Test', fixed_dims=0))


    def test_embedding(self):
        propagator = propagators['Embedding']

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [21]}})],
                'to': d2n({'_id': 'b2', '_class': 'Embedding',
                           'output_feats': 17}),
                'expected': [21, 17],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:L>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Embedding',
                           'output_feats': 25}),
                'expected': [3, '<<variable:L>>', 25],
            },
        ]
        for example in examples:
            from_block = example['from']
            block = example['to']
            propagator(from_block, block)
            self.assertEqual(block._shape.out, example['expected'])


class FixedOutputPropagatorTests(unittest.TestCase):
    """Tests for the FixedOutputPropagator class."""

    def test_initializer(self):
        self.assertRaises(ValueError, lambda: FixedOutputPropagator('Test', fixed_dims=0))
        self.assertRaises(ValueError, lambda: FixedOutputPropagator('Test', unfixed_dims=0))

        propagator = FixedOutputPropagator('Fixed', fixed_dims=2)
        example = {
            'from': [d2n({'_id': 'b1', '_shape': {'out': [480]}})],
            'to': d2n({'_id': 'b2', '_class': 'Fixed',
                       'output_feats': [1, 480]}),
        }
        self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


    def test_linear_propagations(self):
        propagator = FixedOutputPropagator('Linear')

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [21]}})],
                'to': d2n({'_id': 'b2', '_class': 'Linear',
                           'output_feats': 17}),
                'expected': [17],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7, 16]}})],
                'to': d2n({'_id': 'b2', '_class': 'Linear',
                           'output_feats': 9}),
                'expected': [7, 9],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:2*X>>', 24, '<<variable:Y>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Linear',
                           'output_feats': 6}),
                'expected': [3, '<<variable:2*X>>', 24, 6],
            },
        ]
        for example in examples:
            from_block = example['from']
            block = example['to']
            propagator(from_block, block)
            self.assertEqual(block._shape.out, example['expected'])

        ## failures ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7, 16]}})],
                'to': d2n({'_id': 'b2', '_class': 'Linear'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': []}})],
                'to': d2n({'_id': 'b2', '_class': 'Linear',
                           'output_feats': 17}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7, 16]}}),
                         d2n({'_id': 'b2', '_shape': {'out': [7, 16]}})],
                'to': d2n({'_id': 'b3', '_class': 'Linear',
                           'output_feats': 17}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


    def test_adaptiveavgpool_propagations(self):
        propagator = propagators['AdaptiveAvgPool2d']

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, 480, 640]}})],
                'to': d2n({'_id': 'b2', '_class': 'AdaptiveAvgPool2d',
                           'output_feats': [1, 1]}),
                'expected': [3, 1, 1],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:C>>', '<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'AdaptiveAvgPool2d',
                           'output_feats': ['<<variable:VG>>', '<<variable:HG>>']}),
                'expected': ['<<variable:C>>', '<<variable:VG>>', '<<variable:HG>>'],
            },
        ]
        for example in examples:
            from_block = example['from']
            block = example['to']
            propagator(from_block, block)
            self.assertEqual(block._shape.out, example['expected'])

        ## failures ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, 480, 640]}})],
                'to': d2n({'_id': 'b2', '_class': 'AdaptiveAvgPool2d', 'output_feats': [0, 0]}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'AdaptiveAvgPool2d', 'output_feats': ['<<variable:VG>>', '<<variable:HG>>']}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


class SameShapePropagatorsTests(unittest.TestCase):
    """Tests for propagator classes that preserve shape."""

    def test_same_shape_propagations(self):
        propagator = SameShapePropagator('SameShape')

        ## successes ##
        examples = [
            [5],
            ['<<variable:A>>', 3],
            [7, 3, '<<variable:A>>'],
        ]
        for example in examples:
            from_block = d2n({'_id': 'b1', '_shape': {'out': example}})
            block = d2n({'_id': 'b2', '_class': 'SameShape'})
            propagator([from_block], block)
            self.assertEqual(from_block._shape.out, block._shape.out)
            self.assertEqual(vars(block._shape)['in'], block._shape.out)

        ## failures ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5]}})],
                'to': d2n({'_id': 'b2', '_class': 'SameShape2'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': '<<auto>>'}})],
                'to': d2n({'_id': 'b2', '_class': 'SameShape'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5]}}),
                         d2n({'_id': 'b2', '_shape': {'out': [7]}})],
                'to': d2n({'_id': 'b3', '_class': 'SameShape'}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


    def test_same_shapes_propagations(self):
        propagator = SameShapesPropagator('SameShapes')

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:4*X>>']}}),
                         d2n({'_id': 'b2', '_shape': {'out': ['<<variable:4*X>>']}})],
                'to': d2n({'_id': 'b3', '_class': 'SameShapes'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [26, '<<variable:X>>']}}),
                         d2n({'_id': 'b2', '_shape': {'out': [26, '<<variable:X>>']}}),
                         d2n({'_id': 'b3', '_shape': {'out': [26, '<<variable:X>>']}})],
                'to': d2n({'_id': 'b4', '_class': 'SameShapes'}),
            },
        ]
        for example in examples:
            propagator(example['from'], example['to'])
            for from_block in example['from']:
                self.assertEqual(from_block._shape.out, example['to']._shape.out)
            self.assertEqual(vars(example['to']._shape)['in'], example['to']._shape.out)

        ## failures ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5]}})],
                'to': d2n({'_id': 'b3', '_class': 'SameShapes'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:4*X>>']}}),
                         d2n({'_id': 'b2', '_shape': {'out': ['<<variable:X>>']}})],
                'to': d2n({'_id': 'b3', '_class': 'SameShapes'}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


class ConcatPropagatorsTests(unittest.TestCase):
    """Tests for the concatenation propagation classes."""

    def test_concatenate_propagations(self):
        propagator = propagators['Concatenate']

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, 48, 32]}}),
                         d2n({'_id': 'b2', '_shape': {'out': [4, 48, 32]}}),
                         d2n({'_id': 'b3', '_shape': {'out': [5, 48, 32]}})],
                'to': d2n({'_id': 'b4', '_class': 'Concatenate', 'dim': 0}),
                'expected': [12, 48, 32],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:C>>', '<<variable:H>>', '<<variable:W>>']}}),
                         d2n({'_id': 'b2', '_shape': {'out': ['<<variable:C+2>>', '<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b4', '_class': 'Concatenate', 'dim': -3}),
                'expected': ['<<variable:2*C+2>>', '<<variable:H>>', '<<variable:W>>'],
            },
        ]
        for example in examples:
            from_blocks = example['from']
            block = example['to']
            propagator(from_blocks, block)
            self.assertEqual(block._shape.out, example['expected'])

        ## failures ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, 48, 32]}}),
                         d2n({'_id': 'b2', '_shape': {'out': [4, 48, 32]}}),
                         d2n({'_id': 'b3', '_shape': {'out': [5, 48, 32]}})],
                'to': d2n({'_id': 'b4', '_class': 'Concatenate'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, 48, 32]}})],
                'to': d2n({'_id': 'b4', '_class': 'Concatenate', 'dim': 0}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, 48, 32]}}),
                         d2n({'_id': 'b2', '_shape': {'out': [4, 32, 48]}}),
                         d2n({'_id': 'b3', '_shape': {'out': [5, 48, 32]}})],
                'to': d2n({'_id': 'b4', '_class': 'Concatenate', 'dim': 0}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


class ConvPropagatorTests(unittest.TestCase):
    """Tests for the ConvPropagator class."""

    def test_init(self):
        self.assertRaises(ValueError, lambda: ConvPropagator('Bad', conv_dims=0))


    def test_bad_subclass(self):
        class BadPropagator(ConvPropagator):
            num_features_source = 'out_features'
        self.assertRaises(ValueError, lambda: BadPropagator('Bad', conv_dims=1))


    def test_conv_1d_propagations(self):
        propagator = ConvPropagator('Conv1d', conv_dims=1)

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7, 16]}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv1d',
                           'output_feats': 9, 'kernel_size': 3, 'padding': 1}),
                'expected': [9, 16],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:L>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv1d',
                           'output_feats': 6, 'kernel_size': 5, 'padding': 2}),
                'expected': [6, '<<variable:L>>'],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5, '<<variable:L>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv1d',
                           'output_feats': 2, 'kernel_size': 2, 'stride': 2}),
                'expected': [2, '<<variable:L/2>>'],
            },
        ]
        for example in examples:
            from_block = example['from']
            block = example['to']
            propagator(from_block, block)
            self.assertEqual(block._shape.out, example['expected'])

        ## failures ##
        examples = [
            {
                'from': [d2n({'_id': 'b1'})],
                'to': d2n({'_id': 'b2', '_class': 'Conv1d',
                           'output_feats': 9, 'kernel_size': 3, 'padding': 1}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7]}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv1d',
                           'output_feats': 9, 'kernel_size': 3, 'padding': 1}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:L>>', 16]}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv1d',
                           'output_feats': 6, 'kernel_size': 5, 'padding': 2}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5, '<<L>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv1d',
                           'output_feats': 2, 'kernel_size': 2, 'stride': 2}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


    def test_conv_2d_propagations(self):
        propagator = ConvPropagator('Conv2d', conv_dims=2)

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7, 16, 128]}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv2d',
                           'output_feats': 9, 'kernel_size': 3, 'padding': 1}),
                'expected': [9, 16, 128],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, 24, '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv2d',
                           'output_feats': 6, 'kernel_size': 5, 'padding': 2}),
                'expected': [6, 24, '<<variable:W>>'],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5, '<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv2d',
                           'output_feats': 2, 'kernel_size': 3, 'stride': 3}),
                'expected': [2, '<<variable:H/3>>', '<<variable:W/3>>'],
            },
        ]
        for example in examples:
            from_block = example['from']
            block = example['to']
            propagator(from_block, block)
            self.assertEqual(block._shape.out, example['expected'])

        ## failures ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7, 16, 128]}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv2d',
                           'output_feats': 0, 'kernel_size': 3, 'padding': 1}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7, 16, 128]}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv2d',
                           'kernel_size': 3, 'padding': 1}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [32, 3, 24, '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv2d',
                           'output_feats': 6, 'kernel_size': 5, 'padding': 2}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5]}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv2d',
                           'output_feats': 2, 'kernel_size': 3, 'stride': 3}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


class PoolPropagatorTests(unittest.TestCase):
    """Tests for the PoolPropagator class."""

    def test_pool_1d_propagations(self):
        propagator = PoolPropagator('Pool1d', conv_dims=1)
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7, 16]}})],
                'to': d2n({'_id': 'b2', '_class': 'Pool1d',
                           'kernel_size': 3, 'padding': 1}),
                'expected': [7, 16],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:L>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Pool1d',
                           'kernel_size': 5, 'padding': 2}),
                'expected': [3, '<<variable:L>>'],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5, '<<variable:L>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Pool1d',
                           'kernel_size': 2, 'stride': 2}),
                'expected': [5, '<<variable:L/2>>'],
            },
        ]
        for example in examples:
            from_block = example['from']
            block = example['to']
            propagator(from_block, block)
            self.assertEqual(block._shape.out, example['expected'])

    def test_pool_2d_propagations(self):
        propagator = PoolPropagator('Pool2d', conv_dims=2)
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7, 16, 128]}})],
                'to': d2n({'_id': 'b2', '_class': 'Pool2d',
                           'kernel_size': 3, 'padding': 1}),
                'expected': [7, 16, 128],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, 24, '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Pool2d',
                           'kernel_size': 5, 'padding': 2}),
                'expected': [3, 24, '<<variable:W>>'],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:C>>', '<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Pool2d',
                           'kernel_size': 3, 'stride': 3}),
                'expected': ['<<variable:C>>', '<<variable:H/3>>', '<<variable:W/3>>'],
            },
        ]
        for example in examples:
            from_block = example['from']
            block = example['to']
            propagator(from_block, block)
            self.assertEqual(block._shape.out, example['expected'])


class RnnPropagatorTests(unittest.TestCase):
    """Tests for the RnnPropagator class."""

    def test_rnn_propagations(self):
        propagator = RnnPropagator('RNN')

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [128, 16]}})],
                'to': d2n({'_id': 'b2', '_class': 'RNN',
                           'output_feats': 8}),
                'expected': [128, 8],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:L>>', 14]}})],
                'to': d2n({'_id': 'b2', '_class': 'RNN',
                           'output_feats': 12, 'bidirectional': True}),
                'expected': ['<<variable:L>>', 12],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:L>>', '<<variable:F>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'RNN',
                           'output_feats': 7}),
                'expected': ['<<variable:L>>', 7],
            },
        ]
        for example in examples:
            from_block = example['from']
            block = example['to']
            propagator(from_block, block)
            self.assertEqual(block._shape.out, example['expected'])
            self.assertEqual(block.output_feats/(2 if block.bidirectional else 1), block.hidden_size)

        ## failures ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, 128, 16]}})],
                'to': d2n({'_id': 'b2', '_class': 'RNN',
                           'output_feats': 8}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:L>>', 14]}})],
                'to': d2n({'_id': 'b2', '_class': 'RNN',
                           'bidirectional': True}),
                'expected': ['<<variable:L>>', 12],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:L>>', 14]}})],
                'to': d2n({'_id': 'b2', '_class': 'RNN',
                           'output_feats': 11, 'bidirectional': True}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:L>>', '<<variable:F>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'RNN',
                           'output_feats': '7'}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


class ReshapePropagatorTests(unittest.TestCase):
    """Tests for the ReshapePropagator class."""

    def test_reshape_propagations(self):
        propagator = ReshapePropagator('Reshape')

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [32, 8, '<<variable:W/8>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Reshape', 'reshape_spec': [2, [0, 1]]}),
                'expected': ['<<variable:W/8>>', 256],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:H/2>>', 8, '<<variable:W/8>>', 2]}})],
                'to': d2n({'_id': 'b2', '_class': 'Reshape', 'reshape_spec': [[3, 2], [0, 1]]}),
                'expected': ['<<variable:W/4>>', '<<variable:4*H>>'],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [4608]}})],
                'to': d2n({'_id': 'b2', '_class': 'Reshape', 'reshape_spec': [{'0': [3, 48, '<<auto>>']}]}),
                'expected': [3, 48, 32],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:C*H*W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Reshape', 'reshape_spec': [{'0': ['<<variable:C>>', '<<auto>>', '<<variable:W>>']}]}),
                'expected': ['<<variable:C>>', '<<variable:H>>', '<<variable:W>>'],
            },
        ]
        for example in examples:
            from_block = example['from']
            block = example['to']
            propagator(from_block, block)
            self.assertEqual(block._shape.out, example['expected'])

        ## failures ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [32, 8, '<<variable:W/8>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Reshape'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [32, 8, '<<variable:W/8>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Reshape', 'reshape_spec': []}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [10, 32, 8, '<<variable:W/8>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Reshape', 'reshape_spec': [2, [0, 1]]}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [32, 8, '<<variable:W/8>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Reshape', 'reshape_spec': [3, [0, 1]]}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [4608]}})],
                'to': d2n({'_id': 'b2', '_class': 'Reshape', 'reshape_spec': [{'0': [3, 49, '<<auto>>']}]}),
                'expected': [3, 48, 32],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [4608]}})],
                'to': d2n({'_id': 'b2', '_class': 'Reshape', 'reshape_spec': [{'0': [3, '<<auto>>', '<<auto>>']}]}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


class GroupPropagatorTests(unittest.TestCase):
    """Tests for the group propagation classes."""

    def test_sequential_propagations(self):
        propagator = propagators['Sequential']

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Sequential',
                           'blocks': [
                                {'_id': 'i1', '_class': 'Conv2d', 'output_feats': 8, 'kernel_size': 7, 'padding': 3},
                                {'_id': 'i2', '_class': 'BatchNorm2d'},
                                {'_id': 'i3', '_class': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
                                {'_id': 'i4', '_class': 'Linear', 'output_feats': 12}]}),
                'expected': [8, '<<variable:H/2>>', 12],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b1', '_class': 'Sequential',
                           'blocks': [
                                {'_id': 'i1', '_class': 'Conv2d', 'output_feats': 8, 'kernel_size': 7, 'padding': 3},
                                {'_id': 'i2', '_class': 'BatchNorm2d'},
                                {'_id': 'i3', '_class': 'Sequential',
                                 'blocks': [
                                      {'_id': 'i4', '_class': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
                                      {'_id': 'i5', '_class': 'Linear', 'output_feats': 12}]}]}),
                'expected': [8, '<<variable:H/2>>', 12],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b1', '_class': 'Sequential',
                           'blocks': [
                                {'_class': 'Conv2d', 'output_feats': 8, 'kernel_size': 7, 'padding': 3},
                                {'_class': 'BatchNorm2d'},
                                {'_class': 'Sequential',
                                 'blocks': [
                                      {'_class': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
                                      {'_class': 'Linear', 'output_feats': 12}]}]}),
                'expected': [8, '<<variable:H/2>>', 12],
            },
        ]
        for example in examples:
            from_block = example['from']
            block = example['to']
            propagator(from_block, block, propagators)
            self.assertEqual(block._shape.out, example['expected'])

        ## failures ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Sequential'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [10, 32, 8, '<<variable:W/8>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Sequential', 'blocks': []}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, 48]}})],
                'to': d2n({'_id': 'b2', '_class': 'Sequential',
                           'blocks': [
                                {'_id': 's1', '_class': 'Linear', 'output_feats': 12},
                                {'_id': 's2', '_class': 'Unregistered', 'output_feats': 5}]}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, 48]}})],
                'to': d2n({'_id': 'b2', '_class': 'Sequential',
                           'blocks': [
                                {'_id': 's1', '_class': 'Linear', 'output_feats': 12},
                                {'_id': 's1', '_class': 'Linear', 'output_feats': 5}]}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to'], propagators))


    def test_group_propagations(self):
        propagator = propagators['Group']

        base_example = {
            'from': [d2n({'_id': 'b1', '_shape': {'out': [16, '<<variable:H>>', '<<variable:W>>']}})],
            'to': d2n({'_id': 'b2', '_class': 'Group', 'input': 'in', 'output': 'add',
                       'graph': [
                           'in -> conv -> add',
                           'in -> add'],
                       'blocks': [
                           {'_id': 'in', '_class': 'Identity'},
                           {'_id': 'conv', '_class': 'Conv2d', 'output_feats': 16, 'kernel_size': 3, 'padding': 1},
                           {'_id': 'add', '_class': 'Add'}]}),
            'expected': [16, '<<variable:H>>', '<<variable:W>>'],
        }

        nested_example = deepcopy(base_example)
        nested_example['to'].blocks[1] = deepcopy(base_example['to'])
        nested_example['to'].blocks[1]._id = 'conv'

        ## successes ##
        examples = [
            deepcopy(base_example),
            nested_example,
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [16, '<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Group', 'input': 'in', 'output': 'relu',
                           'graph': [
                               'in -> conv -> bn -> add -> relu',
                               'in -> add'],
                           'blocks': [
                               {'_id': 'in', '_class': 'Identity'},
                               {'_id': 'conv', '_class': 'Conv2d', 'output_feats': 16, 'kernel_size': 3, 'padding': 1},
                               {'_id': 'bn', '_class': 'BatchNorm2d'},
                               {'_id': 'add', '_class': 'Add'},
                               {'_id': 'relu', '_class': 'ReLU'}]}),
                'expected': [16, '<<variable:H>>', '<<variable:W>>'],
            },
        ]
        for example in examples:
            from_block = example['from']
            block = example['to']
            propagator(from_block, block, propagators)
            self.assertEqual(block._shape.out, example['expected'])

        ## failures ##
        example = deepcopy(base_example)
        delattr(example['to'], 'input')
        self.assertRaises(ValueError, lambda: propagator(example['from'], example['to'], propagators))

        example = deepcopy(base_example)
        delattr(example['to'], 'output')
        self.assertRaises(ValueError, lambda: propagator(example['from'], example['to'], propagators))

        example = deepcopy(base_example)
        delattr(example['to'], 'graph')
        self.assertRaises(ValueError, lambda: propagator(example['from'], example['to'], propagators))

        example = deepcopy(base_example)
        delattr(example['to'], 'blocks')
        self.assertRaises(ValueError, lambda: propagator(example['from'], example['to'], propagators))

        example = deepcopy(base_example)
        example['to'].blocks[1]._id = 'in'
        self.assertRaises(ValueError, lambda: propagator(example['from'], example['to'], propagators))

        example = deepcopy(base_example)
        example['to'].graph[0] = 'conv -> add'
        self.assertRaises(ValueError, lambda: propagator(example['from'], example['to'], propagators))

        example = deepcopy(base_example)
        example['to'].graph[1] = 'in -> unk -> add'
        self.assertRaises(ValueError, lambda: propagator(example['from'], example['to'], propagators))

        example = deepcopy(base_example)
        example['to'].graph[0] = '---'
        self.assertRaises(ValueError, lambda: propagator(example['from'], example['to'], propagators))
        self.assertRaises(ValueError, lambda: parse_graph(example['from'], example['to']))  # ResourceWarning: unclosed file <_io.BufferedRandom name=3> context = None

        example = deepcopy(base_example)
        example['to'].graph[1] = 'add -> in'
        self.assertRaises(ValueError, lambda: propagator(example['from'], example['to'], propagators))


if __name__ == '__main__':
    unittest.main(verbosity=2)
