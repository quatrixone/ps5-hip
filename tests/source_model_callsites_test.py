from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import unittest,tempfile,hashlib,json
from pathlib import Path
from unittest.mock import patch
import source_console as c
import source_model_freeze as f
class FTP:
 def __init__(self,old):self.files={c.REMOTE:old};self.renames=[];self.corrupt=False
 def retrbinary(self,command,callback,blocksize):
  data=self.files[command[5:]]
  if self.corrupt and command[5:]!=c.REMOTE:data=b'x'+data[1:]
  for i in range(0,len(data),blocksize):callback(data[i:i+blocksize])
 def storbinary(self,command,stream,blocksize):
  data=b''
  while True:
   block=stream.read(blocksize)
   if not block:break
   data+=block
  self.files[command[5:]]=data
 def rename(self,*args):self.renames.append(args)
class Calls(unittest.TestCase):
 def test_actual_deploy_mismatch_never_renames(self):
  for which in ('baseline','temporary'):
   with self.subTest(which=which),tempfile.TemporaryDirectory() as t:
    root=Path(t);(root/'pkg').write_bytes(b'package');old=b'baseline'
    obj=c.Control.__new__(c.Control);obj.fixture='llm_client';obj.manifest={'llm_client':{'phase':'model-gate'},'baseline_log_sha256':None,'baseline_image_sha256':hashlib.sha256(old).hexdigest(),'baseline_transfer':{'bytes':len(old),'deadline_seconds':30},'package':'pkg','model_transfer':{'bytes':7,'sha256':hashlib.sha256(b'package').hexdigest(),'deadline_seconds':30}};obj.out=root;obj.record={};obj.token='test';obj.baseline_log=obj.log='same';obj.ftp=FTP(old if which=='temporary' else b'wrong');obj.ftp.corrupt=which=='temporary'
    obj.idle=lambda:None;obj.send=lambda x:b'';obj.ensure_results_directory=lambda:None;obj.baseline_log_read=lambda:None;obj.listing=lambda x:[];obj.verify=lambda:None;obj.absent=lambda x:None
    # Isolate transport callsites with tiny payloads; authentication tested separately fail-closed.
    with patch.object(c,'ROOT',root),patch.object(c,'snapshot',lambda x:None),patch.object(c,'model_contract',lambda x:None):
     with self.assertRaises(ValueError):obj.deploy_model()
    self.assertEqual(obj.ftp.renames,[]);self.assertFalse(obj.record.get('promoted',False))
 def test_legacy_deploy_keeps_original_path(self):
  obj=c.Control.__new__(c.Control);obj.fixture='bootstrap';obj.manifest={};obj.idle=lambda:(_ for _ in ()).throw(RuntimeError('legacy idle reached'));obj.deploy_model=lambda:self.fail('model route used')
  with self.assertRaisesRegex(RuntimeError,'legacy idle reached'):obj.deploy()
if __name__=='__main__':unittest.main()
