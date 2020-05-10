#!/usr/bin/env python3
"""Unit tests for the json schema."""

import json
import unittest
from narchi.schemas import jsonvalidator, block_schema, narchi_schema, schema_as_str


class SchemaTests(unittest.TestCase):
    """Tests for the json schema."""

    def test_narchi_schema(self):
        jsonvalidator.check_schema(narchi_schema)
        self.assertRegex(narchi_schema['$id'], '.*narchi/[0-9]+.[0-9]+/schema.json$')
        self.assertEqual(narchi_schema, json.loads(schema_as_str()))


    def test_block_schema(self):
        jsonvalidator.check_schema(block_schema)


if __name__ == '__main__':
    unittest.main(verbosity=2)
