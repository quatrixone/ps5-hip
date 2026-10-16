from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import errno,ftplib,hashlib,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import source_console as c
import source_model_streaming
from source_model_callsites_test import FTP
class PartialFTP(FTP):
 def storbinary(self,command,stream,blocksize):
  self.files[command[5:]]=stream.read(3)
  raise ftplib.error_perm('552 No space left on device')
class LateFTP(FTP):
 def __init__(self,old,clock):super().__init__(old);self.clock=clock
 def storbinary(self,command,stream,blocksize):
  super().storbinary(command,stream,blocksize);self.clock[0]=31
class BrokenSink:
 def write(self,block):raise OSError(errno.ENOSPC,'No space left on device')
 def close(self):pass
class Edges(unittest.TestCase):
 def control(self,root,ftp):
  (root/'pkg').write_bytes(b'package');o=c.Control.__new__(c.Control);o.fixture='llm_client';o.manifest={'llm_client':{'phase':'model-gate'},'baseline_log_sha256':None,'baseline_image_sha256':hashlib.sha256(b'baseline').hexdigest(),'baseline_transfer':{'bytes':8,'deadline_seconds':30},'package':'pkg','model_transfer':{'bytes':7,'sha256':hashlib.sha256(b'package').hexdigest(),'deadline_seconds':30}};o.out=root;o.record={};o.token='test';o.baseline_log=o.log='same';o.ftp=ftp
  o.idle=lambda:None;o.send=lambda x:b'';o.ensure_results_directory=lambda:None;o.baseline_log_read=lambda:None;o.listing=lambda x:[];o.verify=lambda:None;o.absent=lambda x:None
  return o
 def run_case(self,ftp,clock=None,local_full=False):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);o=self.control(root,ftp);original_open=Path.open
   def opened(path,*args,**kwargs):
    if local_full and path.name=='old.ffpkg':return BrokenSink()
    return original_open(path,*args,**kwargs)
   with patch.object(c,'ROOT',root),patch.object(c,'snapshot',lambda x:None),patch.object(c,'model_contract',lambda x:None),patch.object(Path,'open',opened),patch.object(source_model_streaming.time,'monotonic',lambda:clock[0] if clock else 0):
    with self.assertRaises((ValueError,OSError,ftplib.error_perm)):o.deploy_model()
   self.assertEqual(ftp.files[c.REMOTE],b'baseline');self.assertEqual(ftp.renames,[]);self.assertFalse(o.record.get('promoted',False))
   return o.record
 def test_remote_disk_full_partial_stor(self):
  ftp=PartialFTP(b'baseline');r=self.run_case(ftp)
  self.assertEqual(ftp.files['/data/homebrew/.PPSA99014-source-test.upload'],b'pac');self.assertFalse(r['ftp_transfers'][-1]['complete']);self.assertIn('552',r['ftp_transfers'][-1]['error'])
 def test_upload_wallclock_even_after_successful_stor(self):
  clock=[0];r=self.run_case(LateFTP(b'baseline',clock),clock)
  self.assertIn('upload deadline',r['ftp_transfers'][-1]['error'])
 def test_local_backup_disk_full_prevents_upload(self):
  ftp=FTP(b'baseline');r=self.run_case(ftp,local_full=True)
  self.assertEqual(set(ftp.files),{c.REMOTE});self.assertIn('No space',r['ftp_reads'][-1]['error'])
 def test_readback_wallclock_not_only_idle(self):
  clock=[0]
  class Slow(FTP):
   def retrbinary(self,command,callback,blocksize):
    for b in self.files[command[5:]]:
     clock[0]+=6;callback(bytes([b]))
  ftp=Slow(b'baseline');r=self.run_case(ftp,clock)
  self.assertIn('readback deadline',r['ftp_reads'][-1]['error'])
if __name__=='__main__':unittest.main()
