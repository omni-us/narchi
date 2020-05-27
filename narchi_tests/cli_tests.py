#!/usr/bin/env python3
"""Unit tests for the command line interface."""

import os
import io
import json
import shutil
import tempfile
import unittest
import contextlib
from jsonargparse import dict_to_namespace
from narchi.bin.narchi_cli import narchi_cli, get_validate_parser, get_render_parser, get_schema_parser
from narchi.schemas import id_separator
from narchi_tests.module_tests import data_dir, laia_jsonnet, laia_ext_vars, laia_shapes, nested1_jsonnet, nested1_ext_vars


class CliTests(unittest.TestCase):
    """Tests for narchi_cli.py."""

    def test_validate(self):
        tmpdir = tempfile.mkdtemp(prefix='_narchi_test_')

        args = ['validate', '--validate=false', '--propagate=false', '--ext_vars', json.dumps(laia_ext_vars), laia_jsonnet]
        narchi_cli(args)

        out_json = os.path.join(tmpdir, 'laia.json')
        args = ['validate', '--save_json=true', '--outdir', tmpdir, '--ext_vars', json.dumps(laia_ext_vars), laia_jsonnet]
        narchi_cli(args)
        with open(out_json) as f:
            architecture = dict_to_namespace(json.loads(f.read()))
        self.assertEqual(laia_shapes, [b._shape.out for b in architecture.blocks])

        self.assertRaises(IOError, lambda: narchi_cli(['--stack_trace=true'] + args))

        shutil.rmtree(tmpdir)


    def test_render(self):
        tmpdir = tempfile.mkdtemp(prefix='_narchi_test_')

        pdf_file = os.path.join(tmpdir, 'laia.pdf')
        args = ['render', '--ext_vars', json.dumps(laia_ext_vars), laia_jsonnet, pdf_file]
        narchi_cli(args)
        assert os.path.isfile(pdf_file)

        gv_file = os.path.join(tmpdir, 'nested1.gv')
        args = ['render', '--overwrite=true', '--save_gv=true', '--outdir', tmpdir, '--ext_vars', json.dumps(nested1_ext_vars), nested1_jsonnet]
        for depth in [1, 2, 3]:
            narchi_cli(args+['--nested_depth', str(depth)])
            with open(gv_file) as f:
                gv = f.read()
                self.assertEqual(gv.count('subgraph'), depth-1)

        shutil.rmtree(tmpdir)


    def test_schema(self):
        with io.StringIO() as buf:
            with contextlib.redirect_stdout(buf):
                narchi_cli(['schema'])
            schema = buf.getvalue()
            self.assertIn('"$id"', schema)
            self.assertIn('/json/narchi/', schema)
            self.assertNotIn(id_separator, schema)

        with io.StringIO() as buf:
            with contextlib.redirect_stdout(buf):
                narchi_cli(['schema', 'propagated'])
            schema = buf.getvalue()
            self.assertIn(id_separator, schema)


    def test_get_subcommand_parsers(self):
        get_validate_parser()
        get_render_parser()
        get_schema_parser()


if __name__ == '__main__':
    unittest.main(verbosity=2)
