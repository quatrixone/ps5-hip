import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from prepare_llm_hip_interface import adapt,checked_header_files,clean_client,VENDOR,OLD,NEW

class Interface(unittest.TestCase):
 def test_exact_guard_only(self):
  data=('before\n'+OLD+'\nafter HIP_VERSION\n').encode()
  self.assertEqual(adapt(data),data.replace(OLD.encode(),NEW.encode()))
  for bad in (b'',data+data,adapt(data)):
   with self.assertRaises(ValueError):adapt(bad)
 def test_header_hash_size_missing_and_path(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder);p=root/'a';p.write_bytes(b'official')
   spec={'files':{'a':{'bytes':8,'sha256':hashlib.sha256(b'official').hexdigest()}}}
   self.assertEqual(checked_header_files(root,spec),{'a':p})
   p.write_bytes(b'changed!')
   with self.assertRaises(ValueError):checked_header_files(root,spec)
   p.write_bytes(b'short')
   with self.assertRaises(ValueError):checked_header_files(root,spec)
   p.unlink()
   with self.assertRaises(ValueError):checked_header_files(root,spec)
   with self.assertRaises(ValueError):checked_header_files(root,{'files':{'../x':{}}})
 def test_clean_locked_git_object(self):
  with tempfile.TemporaryDirectory() as folder:
   root=Path(folder).resolve()
   def git(*args):return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL)
   git('init');p=root/VENDOR;p.parent.mkdir(parents=True);p.write_bytes(b'original')
   git('add','.');git('-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m','fixture')
   revision=git('rev-parse','HEAD').decode().strip()
   self.assertEqual(clean_client(root,revision),b'original')
   with self.assertRaises(ValueError):clean_client(root,'0'*40)
   p.write_bytes(b'changed')
   with self.assertRaises(ValueError):clean_client(root,revision)

if __name__=='__main__':unittest.main()
