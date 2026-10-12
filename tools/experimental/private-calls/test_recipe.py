"""Small portability/identity guards; no compiler or console invoked."""
import importlib.util
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('recipe', Path(__file__).with_name('recipe.py'))
recipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recipe)


class RecipeTests(unittest.TestCase):
    def test_optimized_python_refused(self):
        result = subprocess.run([sys.executable, "-O", str(Path(recipe.__file__)), "--help"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Assertions are required", result.stderr)

    def test_source_manifest(self):
        pins, identity = recipe.package_identity()
        self.assertEqual(pins['llvm']['commit'], '3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff')
        self.assertEqual(identity['backend/buffer-call-proof-v4.patch'],
                         '40ae297525b4cd9ddad9f4f3acc39a905b4f5ed5d4694643f739802d010a033b')
        self.assertEqual(identity['specialization/PrivateSpecialize.cpp'],
                         '37385c7108503176180e3e4bbb6ebde518f14dadedc797ae1b39dcec779eb627')

    def test_existing_output_refused_before_tools(self):
        with tempfile.TemporaryDirectory(prefix='private recipe ') as directory:
            args = type('Args', (), {'output': Path(directory)})()
            with patch.object(recipe, 'tool_identity', side_effect=AssertionError('must not inspect tools')):
                with self.assertRaisesRegex(ValueError, 'new directory'):
                    recipe.prepare(args)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_changed_prepared_artifact_refused(self):
        with tempfile.TemporaryDirectory(prefix='private recipe ') as directory:
            root = Path(directory)
            source = root / 'source with spaces.ll'
            source.write_text('original')
            identity = recipe.package_identity()[1]
            recipe.write_json(root / 'prepared.json', {'package': identity,
                              'artifacts': {source.name: recipe.sha(source)}, 'tools': {}})
            source.write_text('changed')
            with patch.object(recipe, 'tool_identity', side_effect=AssertionError('must not inspect tools')):
                with self.assertRaisesRegex(ValueError, 'Prepared input changed'):
                    recipe.require_prepared(root)

    def test_command_keeps_spaces_and_refuses_rerun(self):
        with tempfile.TemporaryDirectory(prefix='private recipe ') as directory:
            root = Path(directory)
            command = ['/not an actual tool', 'input with spaces.ll']
            with patch.object(recipe.subprocess, 'run') as run:
                run.return_value.returncode = 0
                recipe.run(root, 'example', command)
                self.assertEqual(run.call_args.args[0], command)
                self.assertFalse(run.call_args.kwargs.get('shell', False))
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    recipe.run(root, 'example', command)
                self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main()
