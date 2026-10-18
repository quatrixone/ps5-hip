from pathlib import Path
import sys,json,unittest,types
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import source_console_finalization_test as b
c=b.c
PROFILES=(('model-prefix','smoke'),('model-layer','smoke'),('model-full','smoke'),('model-full','generation'),('model-full','corpus'))
class Calls(unittest.TestCase):
 setUp=b.Tests.setUp
 save=b.Tests.save
 def prepare(self,phase,profile,mutation=None,checkpoint=None):
  obj=self.obj;obj.record={'accepted':False};obj.manifest['llm_client']['phase']=phase
  kind='full' if phase=='model-full' else 'layer' if phase=='model-layer' else 'prefix'
  obj.manifest['model_'+kind]={'reference':'results/candidate/full-reference','profile':profile}
  obj.manifest.update(baseline_log_sha256=None,baseline_image_sha256='1'*64,baseline_transfer={'bytes':8,'deadline_seconds':30},model_transfer={'bytes':8,'sha256':c.sha(b'original'),'deadline_seconds':30})
  self.save();obj.baseline_log=obj.log='same';self.events=[]
  obj.idle=lambda:self.events.append('idle');obj.send=lambda name:self.events.append(name) or b''
  obj.provision_external_model=lambda:None;obj.model_readback=lambda *a,**k:None
  obj.ensure_results_directory=lambda:None;obj.baseline_log_read=lambda:None;obj.preserve_baseline_log=lambda x:None
  obj.fresh_log=lambda:None;obj.absent=lambda name:None;obj.installed=lambda:None
  def mutate():
   if mutation:(self.root/mutation).write_bytes(b'CHANGED!')
  def listing(name):
   if checkpoint==1:mutate()
   return []
  obj.listing=listing
  def history(name):
   if name=='release':
    if checkpoint==2:mutate()
    return ('[IMG][LVD] unmount complete: source='+c.REMOTE+' mount='+c.MOUNT+' unit=1').encode()
   return b''
  obj.history_read=history
  def upload(*a,**k):self.events.append('upload')
  obj.model_upload=upload
  class FTP(b.FTP):
   def rename(ftp,source,target):self.events.append('promote' if target==c.REMOTE else 'preserve')
  obj.ftp=FTP();self.ftp_type=FTP
  original=obj.verify
  def verify(*args,**kwargs):
   self.events.append(('verify',kwargs.get('execution_only',False)));return original(*args,**kwargs)
  obj.verify=verify
  self.addCleanup(patch.stopall);patch.object(c,'snapshot',lambda raw:None).start();patch.object(c,'model_contract',lambda m:None).start();patch.object(c.time,'sleep',lambda n:None).start()
  obj.deploy=types.MethodType(c.Control.deploy_model,obj)
 def test_pending_receipt_rejected_by_actual_execute(self):
  self.assertEqual(self.obj.run(defer_prefix_authentication=True),2)
  self.obj.idle=Mock(side_effect=AssertionError('idle reached'));self.obj.installed=Mock(side_effect=AssertionError('installed reached'));self.obj.send=Mock(side_effect=AssertionError('send reached'))
  with patch.object(c.socket,'create_connection',side_effect=AssertionError('network reached')) as network,self.assertRaisesRegex(RuntimeError,'fresh deployment not complete'):
   c.Control.execute(self.obj)
  self.obj.idle.assert_not_called();self.obj.installed.assert_not_called();self.obj.send.assert_not_called();network.assert_not_called()

def success(phase,profile):
 def test(self):
  self.prepare(phase,profile);self.obj.deploy_model()
  scoped=phase!='model-gate';checks=[i for i,e in enumerate(self.events) if e==('verify',scoped)]
  self.assertEqual(len(checks),2);self.assertLess(checks[0],self.events.index('upload'));self.assertLess(self.events.index('upload'),checks[1]);self.assertLess(checks[1],self.events.index('promote'));self.assertTrue(self.obj.record['accepted'])
 return test

def mutation(phase,profile,checkpoint,kind):
 def test(self):
  names={'package':self.paths[0],'controller':self.paths[1],'oracle':'results/candidate/full-reference/raw/result','validator':'tools/source_console.py','producer':'.deps/sdk/compiler'}
  self.prepare(phase,profile,names[kind],checkpoint)
  with self.assertRaisesRegex(RuntimeError,'frozen input changed'):self.obj.deploy_model()
  self.assertNotIn('upload' if checkpoint==1 else 'promote',self.events);self.assertFalse(self.obj.record.get('accepted'))
  self.assertEqual(sum(isinstance(e,tuple) and e[0]=='verify' for e in self.events),checkpoint)
 return test

def producer_final(phase,profile):
 def test(self):
  self.prepare(phase,profile,'.deps/sdk/compiler',1)
  # Run the actual deploy_model through actual Control.run; only remote operations are mocked.
  with patch.object(c.ftplib,'FTP',self.ftp_type):self.assertEqual(self.obj.run(defer_prefix_authentication=True),2)
  self.assertIn('promote',self.events);self.assertFalse(self.lock.exists());self.assertTrue(self.secondary.exists());self.assertFalse(self.obj.record['accepted'])
  self.secondary.unlink();self.obj.record['secondary_lock_released']=True
  self.assertEqual(self.obj.finish_prefix_authentication(),1);self.assertFalse(self.obj.record['accepted']);self.assertTrue(self.obj.record['promoted']);self.assertFalse(self.secondary.exists());self.obj.authenticate_llm_client.assert_not_called()
 return test
for phase,profile in PROFILES+(('model-gate','smoke'),):
 label=phase.replace('-','_')+'_'+profile
 setattr(Calls,'test_success_'+label,success(phase,profile))
 for checkpoint in (1,2):
  for kind in (('producer',) if phase=='model-gate' else ('package','controller','oracle','validator')):
   setattr(Calls,f'test_mutation_{label}_{checkpoint}_{kind}',mutation(phase,profile,checkpoint,kind))
 if phase!='model-gate':setattr(Calls,'test_producer_final_'+label,producer_final(phase,profile))
if __name__=='__main__':unittest.main()
