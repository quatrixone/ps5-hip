"""The execution entry point must refuse before lock, FTP or socket access."""
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
import importlib.util


class InertEntryTests(TestCase):
    def test_entry_points_refuse_without_network(self):
        base = Path(__file__).resolve().parent
        for kind in ('execution',):
            with self.subTest(kind=kind):
                path = base / kind / 'run.py'
                spec = importlib.util.spec_from_file_location('inert_' + kind, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                with patch.object(module.ftplib, 'FTP') as ftp, \
                     patch('socket.create_connection') as network:
                    with self.assertRaisesRegex(RuntimeError, '^INERT: public HIP'):
                        module.main()
                    ftp.assert_not_called()
                    network.assert_not_called()
