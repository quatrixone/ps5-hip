from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import unittest,tempfile
from pathlib import Path
from unittest.mock import patch
import source_console as c
import source_model_storage_edges_test
from source_model_callsites_test import FTP
class Accumulated(FTP):
 def __init__(self,clock):super().__init__(b'baseline');self.clock=clock
 def retrbinary(self,*a,**kw):super().retrbinary(*a,**kw);self.clock[0]+=600
 def storbinary(self,*a,**kw):super().storbinary(*a,**kw);self.clock[0]+=600
class Budget(unittest.TestCase):
 def test_accumulated_transfers_prevent_promotion(self):
  clock=[0];ftp=Accumulated(clock)
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);o=source_model_storage_edges_test.Edges().control(root,ftp)
   o.manifest['baseline_transfer']['deadline_seconds']=1800;o.manifest['model_transfer']['deadline_seconds']=1800
   with patch.object(c,'ROOT',root),patch.object(c,'snapshot',lambda x:None),patch.object(c,'model_contract',lambda x:None),patch.object(c.time,'monotonic',lambda:clock[0]):
    with self.assertRaisesRegex(RuntimeError,'whole model phase'):o.deploy_model()
   self.assertEqual(clock[0],1800);self.assertEqual(ftp.renames,[]);self.assertEqual(ftp.files[c.REMOTE],b'baseline');self.assertFalse(o.record.get('promoted',False))
 def test_expired_execute_cannot_launch(self):
  o=c.Control.__new__(c.Control);o.fixture='llm_client';o.manifest={'llm_client':{'phase':'model-gate'}};o.phase_deadline=1;o.send=lambda name:self.fail('network/launch invoked')
  with patch.object(c.time,'monotonic',return_value=2),self.assertRaisesRegex(RuntimeError,'whole model phase'):o.execute()
 def test_proxy_blocks_mutation_after_expiry(self):
  calls=[];o=c.Control.__new__(c.Control);o.fixture='llm_client';o.manifest={'llm_client':{'phase':'model-gate'}};o.phase_deadline=1
  class Remote:
   def rename(self,*args):calls.append(args)
  ftp=c.PhaseFTP(Remote(),o.phase_remaining)
  with patch.object(c.time,'monotonic',return_value=2),self.assertRaises(RuntimeError):ftp.rename('temp','installed')
  self.assertEqual(calls,[])
 def test_remaining_never_restarts(self):
  o=c.Control.__new__(c.Control);o.fixture='llm_client';o.manifest={'llm_client':{'phase':'model-gate'}}
  with patch.object(c.time,'monotonic',return_value=10):self.assertEqual(o.phase_remaining(),1800)
  with patch.object(c.time,'monotonic',return_value=610):self.assertEqual(o.phase_remaining(),1200)
  with patch.object(c.time,'monotonic',return_value=1810),self.assertRaises(RuntimeError):o.phase_remaining()
if __name__=='__main__':unittest.main()
