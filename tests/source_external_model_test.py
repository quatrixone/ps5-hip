"""Persistent-model transfer policy, using actual bounded streaming with tiny fixtures."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import ftplib
import hashlib
import json
import tempfile
import unittest
from unittest.mock import patch
import external_model
import source_console as c

DATA = b'locked model data'
REMOTE = '/user/homebrew/ps5-hip/models/tinyllama-q4_0.gguf'

class FTP:
    def __init__(self):
        self.files = {}; self.dirs = {'/'}; self.uploads = []; self.renames = []
        self.partial = False; self.corrupt = False; self.before_rename = None
    def retrlines(self, command, callback):
        parent = command[5:].rstrip('/') or '/'
        for name in sorted(self.dirs | set(self.files)):
            if name != '/' and str(Path(name).parent) == parent:
                kind = 'd' if name in self.dirs else '-'
                callback(kind + 'rwxr-xr-x 1 root root 0 Jan 1 00:00 ' + Path(name).name)
    def mkd(self, path): self.dirs.add(path)
    def storbinary(self, command, stream, blocksize):
        path = command[5:]; self.uploads.append(path); self.files[path] = b''
        while chunk := stream.read(blocksize):
            self.files[path] += chunk
            if self.partial:
                self.files[path] = chunk[:3]
                raise ftplib.error_perm('552 no space')
        if self.corrupt: self.files[path] = b'x' + self.files[path][1:]
        if self.before_rename: self.before_rename()
    def retrbinary(self, command, callback, blocksize): callback(self.files[command[5:]])
    def rename(self, old, new):
        self.renames.append((old, new)); self.files[new] = self.files.pop(old)

class ExternalModelTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        source = Path(self.temp.name) / 'model.gguf'; source.write_bytes(DATA)
        self.patch_size = patch.object(external_model, 'MODEL_BYTES', len(DATA)); self.patch_size.start(); self.addCleanup(self.patch_size.stop)
        self.patch_sha = patch.object(external_model, 'MODEL_SHA', hashlib.sha256(DATA).hexdigest()); self.patch_sha.start(); self.addCleanup(self.patch_sha.stop)
        self.o = c.Control.__new__(c.Control); self.o.phase = 'deploy'; self.o.fixture = 'llm_client'
        self.o.manifest = {'external_model': {'path': REMOTE, 'bytes': len(DATA), 'sha256': hashlib.sha256(DATA).hexdigest()}, 'llm_client': {'phase': 'model-gate', 'model': {'asset': str(source)}}}
        self.o.ftp = FTP(); self.o.record = {}; self.o.token = 'test'; self.o.idle = lambda: None
    def test_missing_uploads_and_verifies_then_reuses(self):
        self.o.provision_external_model()
        self.assertEqual(self.o.ftp.files[REMOTE], DATA)
        self.assertTrue(self.o.record['external_model']['complete'])
        self.assertEqual(len(self.o.ftp.uploads), 1)
        self.assertTrue(all(r['complete'] for r in self.o.record['external_model_transfers']))
        self.o.provision_external_model()
        self.assertEqual(self.o.record['external_model']['action'], 'reuse')
        self.assertEqual(len(self.o.ftp.uploads), 1)
    def test_corrupt_existing_is_never_overwritten(self):
        self.o.ensure_directory(str(Path(REMOTE).parent), 'model'); self.o.ftp.files[REMOTE] = b'x' * len(DATA)
        with self.assertRaisesRegex(ValueError, 'identity mismatch'): self.o.provision_external_model()
        self.assertEqual(self.o.ftp.files[REMOTE], b'x' * len(DATA))
        self.assertEqual(self.o.ftp.uploads, []); self.assertEqual(self.o.ftp.renames, [])
        self.assertFalse(self.o.record['external_model']['complete'])
    def test_partial_or_corrupt_stage_is_retained_not_promoted(self):
        for failure in ('partial', 'corrupt'):
            with self.subTest(failure=failure):
                self.o.ftp = FTP(); self.o.record = {}; setattr(self.o.ftp, failure, True)
                with self.assertRaises((ftplib.error_perm, ValueError)): self.o.provision_external_model()
                self.assertNotIn(REMOTE, self.o.ftp.files); self.assertEqual(self.o.ftp.renames, [])
                self.assertFalse(self.o.record['external_model_transfers'][-1]['complete'])
                self.assertIn(self.o.record['external_model']['temporary'], self.o.ftp.files)
    def test_destination_appearing_during_upload_prevents_promotion(self):
        self.o.ftp.before_rename = lambda: self.o.ftp.files.update({REMOTE: b'another writer'})
        with self.assertRaisesRegex(RuntimeError, 'appeared'): self.o.provision_external_model()
        self.assertEqual(self.o.ftp.files[REMOTE], b'another writer'); self.assertEqual(self.o.ftp.renames, [])
    def test_prelaunch_verification_rejects_changed_or_missing_model(self):
        self.o.provision_external_model(); self.o.phase = 'execute'
        self.o.ftp.files[REMOTE] = b'x' * len(DATA)
        with self.assertRaises(ValueError): self.o.verify_external_model()
        del self.o.ftp.files[REMOTE]
        with self.assertRaisesRegex(RuntimeError, 'missing'): self.o.verify_external_model()
    def test_legacy_is_noop(self):
        del self.o.manifest['external_model']; self.o.provision_external_model(); self.o.verify_external_model()
        self.assertEqual(self.o.ftp.uploads, []); self.assertEqual(self.o.record, {})
    def test_directory_in_place_of_model_rejected(self):
        self.o.ensure_directory(str(Path(REMOTE).parent), 'model'); self.o.ftp.dirs.add(REMOTE)
        with self.assertRaisesRegex(RuntimeError, 'regular file'): self.o.provision_external_model()
        self.assertEqual(self.o.ftp.uploads, [])

    def test_changed_local_source_never_uploads(self):
        Path(self.o.manifest['llm_client']['model']['asset']).write_bytes(b'wrong model')
        with self.assertRaisesRegex(ValueError, 'local identity mismatch'): self.o.provision_external_model()
        self.assertEqual(self.o.ftp.uploads, [])

    def test_external_candidate_uses_small_package_cap(self):
        for size in (c.package_limit('llm_client'), c.package_limit('llm_client') + 1):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                root = Path(directory); candidate = root / 'candidate'; candidate.mkdir()
                package = root / 'pkg'; package.write_bytes(b'x')
                manifest = {**self.o.manifest, 'fixture': 'llm_client', 'results_dir': '/data/test',
                            'host': c.HOST, 'title_id': c.TITLE, 'package': 'pkg', 'files': {}}
                mp = candidate / 'manifest.json'; mp.write_text(json.dumps(manifest))
                (candidate / 'review.json').write_text(json.dumps({'approved': True, 'manifest_sha256': c.sha(mp.read_bytes())}))
                original_stat = Path.stat
                def stat(path, *args, **kwargs):
                    result = original_stat(path, *args, **kwargs)
                    if path != package: return result
                    class Sized:
                        st_size = size
                    return Sized()
                with patch.object(c, 'ROOT', root), patch.object(Path, 'stat', stat), patch.object(c.Control, 'authenticate_llm_client'):
                    if size == c.package_limit('llm_client'):
                        c.Control(candidate, 'deploy')
                    else:
                        with self.assertRaisesRegex(RuntimeError, 'candidate package size'): c.Control(candidate, 'deploy')

if __name__ == '__main__': unittest.main()
