from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import hashlib, tempfile, unittest
from unittest.mock import patch
from pathlib import Path
from source_model_streaming import CHUNK, stage_upload, readback
class FTP:
 def __init__(self, mutate=lambda b:b): self.data=b'';self.mutate=mutate;self.maximum=0;self.renames=[]
 def storbinary(self, command, stream, blocksize):
  assert blocksize==CHUNK
  while True:
   b=stream.read(blocksize)
   if not b:break
   self.maximum=max(self.maximum,len(b));self.data+=b
 def retrbinary(self, command, callback, blocksize):
  for start in range(0,len(self.mutate(self.data)),blocksize):callback(self.mutate(self.data)[start:start+blocksize])
 def rename(self,*args):self.renames.append(args)
class Test(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'image';self.data=b'abcd'*(CHUNK+13);self.path.write_bytes(self.data);self.digest=hashlib.sha256(self.data).hexdigest()
 def run_upload(self, ftp):
  records=[]
  try:return stage_upload(ftp,self.path,'/temporary',len(self.data),self.digest,records,30)
  finally:self.assertFalse(ftp.renames);self.assertLessEqual(ftp.maximum,CHUNK)
 def test_upload_and_readback_share_deadline(self):
  now=[0.0]
  class SlowFTP(FTP):
   def storbinary(inner,*args,**kwargs):
    super().storbinary(*args,**kwargs);now[0]=50.0
   def retrbinary(inner,*args,**kwargs):
    now[0]=100.0;super().retrbinary(*args,**kwargs)
  ftp=SlowFTP();records=[]
  with patch('source_model_streaming.time.monotonic',side_effect=lambda:now[0]):
   with self.assertRaisesRegex(ValueError,'readback deadline'):
    stage_upload(ftp,self.path,'/temporary',len(self.data),self.digest,records,60)
  self.assertFalse(ftp.renames);self.assertFalse(records[-1]['complete'])
 def test_success(self):self.assertTrue(self.run_upload(FTP())['complete'])
 def test_partial(self):
  with self.assertRaisesRegex(ValueError,'identity'):self.run_upload(FTP(lambda b:b[:-1]))
 def test_oversize(self):
  with self.assertRaisesRegex(ValueError,'oversize'):self.run_upload(FTP(lambda b:b+b'x'))
 def test_wrong_hash(self):
  with self.assertRaisesRegex(ValueError,'identity'):self.run_upload(FTP(lambda b:b'x'+b[1:]))
 def test_local_change_prevents_upload(self):
  self.path.write_bytes(b'wrong');ftp=FTP()
  with self.assertRaises(ValueError):self.run_upload(ftp)
  self.assertEqual(ftp.data,b'')
 def test_backup_exclusive(self):
  ftp=FTP();ftp.data=self.data;dest=Path(self.tmp.name)/'backup';records=[]
  readback(ftp,'/original',len(self.data),self.digest,records,30,dest);self.assertEqual(dest.read_bytes(),self.data)
  with self.assertRaises(FileExistsError):readback(ftp,'/original',len(self.data),self.digest,records,30,dest)
 def test_failure_preserves_partial_evidence(self):
  ftp=FTP(lambda b:b[:-1]);ftp.data=self.data;dest=Path(self.tmp.name)/'partial';records=[]
  with self.assertRaises(ValueError):readback(ftp,'/original',len(self.data),self.digest,records,30,dest)
  self.assertEqual(dest.stat().st_size,len(self.data)-1);self.assertFalse(records[-1]['complete'])
if __name__=='__main__':unittest.main()
