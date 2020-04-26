#!/usr/bin/env python3
"""Unit tests for the json schema."""

import json
import unittest
from nnarch.schema import jsonvalidator, block_schema, nnarch_schema, schema_as_str


class SchemaTests(unittest.TestCase):
    """Tests for the json schema."""

    def test_nnarch_schema(self):
        jsonvalidator.check_schema(nnarch_schema)
        self.assertRegex(nnarch_schema['$id'], '.*nnarch/[0-9]+.[0-9]+/schema.json$')
        self.assertEqual(nnarch_schema, json.loads(schema_as_str()))


    def test_block_schema(self):
        jsonvalidator.check_schema(block_schema)


if __name__ == '__main__':
    unittest.main(verbosity=2)
