"""Resource preparation must preserve strict identities through reuse and reversal."""
import difflib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import prepare_public_hip_resources as resources


class ResourcesTest(unittest.TestCase):
    def test_exact_apply_reuse_reverse_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            ports=root/'ports/clr';ports.mkdir(parents=True)
            before='before\n';after='after\n'
            diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/input',tofile='b/input'))
            digest=lambda value:hashlib.sha256(value.encode()).hexdigest()
            spec={'files':{'input':{'before':digest(before),'after':digest(after)}},'patch_sha256':digest(diff)}
            (ports/'ps5-public-resources.json').write_text(json.dumps(spec))
            patch_file=ports/'ps5-public-resources.patch';patch_file.write_text(diff)
            target=source/'input';target.write_text(before)
            with patch.object(resources,'ROOT',root):
                self.assertFalse(resources.prepared(source))
                resources.prepare(source);self.assertEqual(target.read_text(),after)
                resources.prepare(source);self.assertTrue(resources.prepared(source))
                resources.prepare(source,True);self.assertEqual(target.read_text(),before)
                resources.prepare(source,True);resources.prepare(source)
                target.write_text('unknown\n')
                for reverse in (False,True):
                    with self.assertRaisesRegex(ValueError,'prerequisites'):resources.prepare(source,reverse)
                    self.assertEqual(target.read_text(),'unknown\n')
                patch_file.write_text(diff+'\n')
                with self.assertRaisesRegex(ValueError,'patch changed'):resources.prepared(source)


if __name__=='__main__':unittest.main()
