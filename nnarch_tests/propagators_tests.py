#!/usr/bin/env python3
"""Unit tests for propagation classes."""

import unittest
from jsonargparse import dict_to_namespace as d2n
from nnarch.register import propagators, register_propagator
from nnarch.propagators.base import BasePropagator
from nnarch.propagators.simple import LinearPropagator
from nnarch.propagators.same import SameShapePropagator
from nnarch.propagators.conv import ConvPropagator
from nnarch.propagators.pool import PoolPropagator
from nnarch.propagators.rnn import RnnPropagator
from nnarch.propagators.reshape import PermutePropagator
from nnarch.propagators.group import SequentialPropagator


class BasePropagatorTests(unittest.TestCase):
    """Tests for the BasePropagator class."""

    def test_propagators_dict(self):
        self.assertRaises(ValueError, lambda: register_propagator(LinearPropagator('Linear')))
        register_propagator(LinearPropagator('Linear'), replace=True)
        for propagator in propagators.values():
            self.assertTrue(isinstance(propagator, BasePropagator))

    def test_base_propagator(self):
        propagator = BasePropagator('Base')

        examples = [
            {
                'from': [d2n({})],
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
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<auto>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Base'}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator.initial_checks(example['from'], example['to']))

        self.assertRaises(NotImplementedError, lambda: propagator.propagate(None, None))


class LinearPropagatorTests(unittest.TestCase):
    """Tests for the LinearPropagator class."""

    def test_linear_propagations(self):
        propagator = LinearPropagator('Linear')

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [21]}})],
                'to': d2n({'_id': 'b2', '_class': 'Linear',
                           'out_features': 17}),
                'expected': [17],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7, 16]}})],
                'to': d2n({'_id': 'b2', '_class': 'Linear',
                           'out_features': 9}),
                'expected': [7, 9],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:2*X>>', 24, '<<variable:Y>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Linear',
                           'out_features': 6}),
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
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


class SameShapePropagatorTests(unittest.TestCase):
    """Tests for the SameShapePropagator class."""

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


    def test_multi_input_same_shape_propagations(self):
        propagator = SameShapePropagator('SameShape', multi_input=True)

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:4*X>>']}}),
                         d2n({'_id': 'b2', '_shape': {'out': ['<<variable:4*X>>']}})],
                'to': d2n({'_id': 'b3', '_class': 'SameShape'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [26, '<<variable:X>>']}}),
                         d2n({'_id': 'b2', '_shape': {'out': [26, '<<variable:X>>']}}),
                         d2n({'_id': 'b3', '_shape': {'out': [26, '<<variable:X>>']}})],
                'to': d2n({'_id': 'b4', '_class': 'SameShape'}),
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
                'to': d2n({'_id': 'b3', '_class': 'SameShape'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:4*X>>']}}),
                         d2n({'_id': 'b2', '_shape': {'out': ['<<variable:X>>']}})],
                'to': d2n({'_id': 'b3', '_class': 'SameShape'}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


class ConvPropagatorTests(unittest.TestCase):
    """Tests for the ConvPropagator class."""

    def test_conv_1d_propagations(self):
        propagator = ConvPropagator('Conv1d', conv_dims=1)

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7, 16]}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv1d',
                           'out_features': 9, 'kernel_size': 3, 'padding': 1}),
                'expected': [9, 16],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:L>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv1d',
                           'out_features': 6, 'kernel_size': 5, 'padding': 2}),
                'expected': [6, '<<variable:L>>'],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5, '<<variable:L>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv1d',
                           'out_features': 2, 'kernel_size': 2, 'stride': 2}),
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
                           'out_features': 9, 'kernel_size': 3, 'padding': 1}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7]}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv1d',
                           'out_features': 9, 'kernel_size': 3, 'padding': 1}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:L>>', 16]}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv1d',
                           'out_features': 6, 'kernel_size': 5, 'padding': 2}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5, '<<L>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv1d',
                           'out_features': 2, 'kernel_size': 2, 'stride': 2}),
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
                           'out_features': 9, 'kernel_size': 3, 'padding': 1}),
                'expected': [9, 16, 128],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, 24, '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv2d',
                           'out_features': 6, 'kernel_size': 5, 'padding': 2}),
                'expected': [6, 24, '<<variable:W>>'],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5, '<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv2d',
                           'out_features': 2, 'kernel_size': 3, 'stride': 3}),
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
                           'out_features': 0, 'kernel_size': 3, 'padding': 1}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [7, 16, 128]}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv2d',
                           'kernel_size': 3, 'padding': 1}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [32, 3, 24, '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv2d',
                           'out_features': 6, 'kernel_size': 5, 'padding': 2}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [5]}})],
                'to': d2n({'_id': 'b2', '_class': 'Conv2d',
                           'out_features': 2, 'kernel_size': 3, 'stride': 3}),
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
                           'out_features': 8}),
                'expected': [128, 8],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:L>>', 14]}})],
                'to': d2n({'_id': 'b2', '_class': 'RNN',
                           'out_features': 12, 'bidirectional': True}),
                'expected': ['<<variable:L>>', 12],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:L>>', '<<variable:F>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'RNN',
                           'out_features': 7}),
                'expected': ['<<variable:L>>', 7],
            },
        ]
        for example in examples:
            from_block = example['from']
            block = example['to']
            propagator(from_block, block)
            self.assertEqual(block._shape.out, example['expected'])
            self.assertEqual(block.out_features/(2 if block.bidirectional else 1), block.hidden_size)

        ## failures ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, 128, 16]}})],
                'to': d2n({'_id': 'b2', '_class': 'RNN',
                           'out_features': 8}),
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
                           'out_features': 11, 'bidirectional': True}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:L>>', '<<variable:F>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'RNN',
                           'out_features': '7'}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to']))


class ReshapePropagatorTests(unittest.TestCase):
    """Tests for the PermutePropagator class."""

    def test_permute_propagations(self):
        propagator = PermutePropagator('Permute')

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [32, 8, '<<variable:W/8>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Permute', 'dims': [2, [0, 1]]}),
                'expected': ['<<variable:W/8>>', 256],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': ['<<variable:H/2>>', 8, '<<variable:W/8>>', 2]}})],
                'to': d2n({'_id': 'b2', '_class': 'Permute', 'dims': [[3, 2], [0, 1]]}),
                'expected': ['<<variable:W/4>>', '<<variable:4*H>>'],
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
                'to': d2n({'_id': 'b2', '_class': 'Permute'}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [10, 32, 8, '<<variable:W/8>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Permute', 'dims': [2, [0, 1]]}),
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [32, 8, '<<variable:W/8>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Permute', 'dims': [3, [0, 1]]}),
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
                                {'_id': 'i1', '_class': 'Conv2d', 'out_features': 8, 'kernel_size': 7, 'padding': 3},
                                {'_id': 'i2', '_class': 'BatchNorm2d'},
                                {'_id': 'i3', '_class': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
                                {'_id': 'i4', '_class': 'Linear', 'out_features': 12}]}),
                'expected': [8, '<<variable:H/2>>', 12],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b1', '_class': 'Sequential',
                           'blocks': [
                                {'_id': 'i1', '_class': 'Conv2d', 'out_features': 8, 'kernel_size': 7, 'padding': 3},
                                {'_id': 'i2', '_class': 'BatchNorm2d'},
                                {'_id': 'i3', '_class': 'Sequential',
                                 'blocks': [
                                      {'_id': 'i4', '_class': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
                                      {'_id': 'i5', '_class': 'Linear', 'out_features': 12}]}]}),
                'expected': [8, '<<variable:H/2>>', 12],
            },
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [3, '<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b1', '_class': 'Sequential',
                           'blocks': [
                                {'_class': 'Conv2d', 'out_features': 8, 'kernel_size': 7, 'padding': 3},
                                {'_class': 'BatchNorm2d'},
                                {'_class': 'Sequential',
                                 'blocks': [
                                      {'_class': 'MaxPool2d', 'kernel_size': 2, 'stride': 2},
                                      {'_class': 'Linear', 'out_features': 12}]}]}),
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
                                {'_id': 's1', '_class': 'Linear', 'out_features': 12},
                                {'_id': 's2', '_class': 'Unregistered', 'out_features': 5}]}),
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: propagator(example['from'], example['to'], propagators))


    def test_group_propagations(self):
        propagator = propagators['Group']

        ## successes ##
        examples = [
            {
                'from': [d2n({'_id': 'b1', '_shape': {'out': [16, '<<variable:H>>', '<<variable:W>>']}})],
                'to': d2n({'_id': 'b2', '_class': 'Group', 'input': 'in', 'output': 'relu',
                           'graph': [
                               'in -> conv -> bn -> add -> relu',
                               'in -> add'],
                           'blocks': [
                               {'_id': 'in', '_class': 'Identity'},
                               {'_id': 'conv', '_class': 'Conv2d', 'out_features': 16, 'kernel_size': 3, 'padding': 1},
                               {'_id': 'bn', '_class': 'BatchNorm2d'},
                               {'_id': 'add', '_class': 'Add'},
                               {'_id': 'relu2', '_class': 'ReLU'}]}),
                'expected': [8, '<<variable:H/2>>', 12],
            },
        ]
        for example in examples:
            from_block = example['from']
            block = example['to']
            #propagator(from_block, block, propagators)
            #self.assertEqual(block._shape.out, example['expected'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
