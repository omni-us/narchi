#!/usr/bin/env python3
"""Unit tests for symbolic operations."""

import os
import unittest
from narchi.sympy import variable_operate, prod


class SympyTests(unittest.TestCase):
    """Tests for symbolic operation functions."""

    def test_variable_operate(self):
        ## successes ##
        examples = [
            {
                'input': 12,
                'operation': '__input__/3',
                'expected': 4,
            },
            {
                'input': '<<variable:X>>',
                'operation': 16,
                'expected': 16,
            },
            {
                'input': '<<variable:X>>',
                'operation': '<<variable:T>>',
                'expected': '<<variable:T>>',
            },
            {
                'input': '<<variable:X>>',
                'operation': '__input__/5',
                'expected': '<<variable:X/5>>',
            },
            {
                'input': '<<variable:X>>',
                'operation': '(__input__+7)+2*(__input__+2)',
                'expected': '<<variable:3*X+11>>',
            },
            {
                'input': '<<variable:8*X+4>>',
                'operation': '__input__/2',
                'expected': '<<variable:4*X+2>>',
            },
            {
                'input': '<<variable:X/12+7*Y>>',
                'operation': '4*__input__',
                'expected': '<<variable:X/3+28*Y>>',
            },
            {
                'input': '<<variable:X>>',
                'operation': '2*__input__+Y',
                'expected': '<<variable:2*X+Y>>',
            },
        ]
        for example in examples:
            result = variable_operate(example['input'], example['operation'])
            self.assertEqual(result, example['expected'])

        ## failures ##
        examples = [
            {
                'input': '<<X>>',
                'operation': '__input__/3',
            },
            {
                'input': '<<variable:__input__>>',
                'operation': '__input__+5',
            },
            {
                'input': '<<variable:X>>',
                'operation': 3.25,
            },
            {
                'input': 7.6,
                'operation': '2*__input__',
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: variable_operate(example['input'], example['operation']))


    def test_prod(self):
        ## successes ##
        examples = [
            {
                'values': [2, 3, 4],
                'expected': 24,
            },
            {
                'values': ['<<variable:X>>', '<<variable:2*Y>>', 3],
                'expected': '<<variable:6*X*Y>>',
            },
        ]
        for example in examples:
            result = prod(example['values'])
            self.assertEqual(result, example['expected'])

        ## failures ##
        examples = [
            {
                'values': [],
            },
            {
                'values': ['<<variable:X>>', 2.5],
            },
        ]
        for example in examples:
            self.assertRaises(ValueError, lambda: prod(example['values']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
