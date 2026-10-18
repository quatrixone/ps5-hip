from pathlib import Path
import hashlib,json,sys,tempfile,types,unittest
from unittest.mock import patch,Mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import source_console as c
class Socket:
 def __init__(self,hook=None):self.sent=[];self.hook=hook
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def settimeout(self,*args):pass
 def sendall(self,blob):self.sent.append(blob);self.hook and self.hook()
 def recv(self,*args):return b''

class Authentication(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.candidate=self.root/'results/candidate';self.candidate.mkdir(parents=True);self.out=self.candidate/'execute';self.out.mkdir()
  self.obj=c.Control.__new__(c.Control);self.obj.candidate=self.candidate;self.obj.out=self.out;self.obj.fixture='llm_client';self.obj.token='owned';self.obj.defer_prefix_authentication=False;self.obj.record={'accepted':False,'prefix_authentication_pending':True,'lock_released':True,'healthy_after':True}
  self.obj.manifest={'llm_client':{'phase':'model-prefix'},'files':{}}
  for name in ('snapshot','post_snapshot','launch','close'):
   p=self.candidate/(name+'.elf');p.write_bytes(name.encode());relative=str(p.relative_to(self.root));self.obj.manifest[name]=relative;self.obj.manifest['files'][relative]=c.sha(p.read_bytes())
  self.save();self.obj.owned_lock=Mock();self.obj.phase_remaining=Mock(return_value=1800);self.obj.cleanup_remaining=Mock(return_value=30)
  self.obj.verify=Mock();self.obj.authenticate_llm_client=Mock();self.addCleanup(patch.stopall);patch.object(c,'ROOT',self.root).start()
 def save(self):
  raw=json.dumps(self.obj.manifest).encode();(self.candidate/'manifest.json').write_bytes(raw);self.obj.manifest_sha=c.sha(raw)
 def test_payload_checks_do_not_rehash_unrelated_closure(self):
  for name in ('snapshot','post_snapshot','launch','close'):
   sock=Socket()
   with patch.object(c.socket,'create_connection',return_value=sock):self.assertEqual(self.obj.send(name),b'')
   self.assertEqual(sock.sent,[name.encode()]);self.assertTrue((self.out/(name+'-consumed')).exists())
  self.obj.verify.assert_not_called();self.assertEqual(self.obj.owned_lock.call_count,8)
 def test_manifest_tamper_before_network(self):
  (self.candidate/'manifest.json').write_bytes(b'changed')
  with patch.object(c.socket,'create_connection') as connect,self.assertRaisesRegex(RuntimeError,'manifest changed'):self.obj.send('launch')
  connect.assert_not_called();self.assertFalse((self.out/'launch-consumed').exists())
 def test_payload_tamper_before_network(self):
  (self.root/self.obj.manifest['launch']).write_bytes(b'changed')
  with patch.object(c.socket,'create_connection') as connect,self.assertRaisesRegex(RuntimeError,'payload differs'):self.obj.send('launch')
  connect.assert_not_called();self.assertFalse((self.out/'launch-consumed').exists())
 def test_exact_checked_bytes_transmitted(self):
  path=self.root/self.obj.manifest['launch'];sock=Socket()
  def connect(*args,**kwargs):path.write_bytes(b'later modification');return sock
  with patch.object(c.socket,'create_connection',side_effect=connect):self.obj.send('launch')
  self.assertEqual(sock.sent,[b'launch'])
 def test_lock_checked_again_immediately_before_send(self):
  sock=Socket();self.obj.owned_lock.side_effect=[None,RuntimeError('lock changed')]
  with patch.object(c.socket,'create_connection',return_value=sock),self.assertRaisesRegex(RuntimeError,'lock changed'):self.obj.send('launch')
  self.assertEqual(sock.sent,[])
 def test_unknown_or_escaping_path(self):
  with self.assertRaisesRegex(RuntimeError,'unknown'):self.obj.frozen_controller_payload('package')
  self.obj.manifest['launch']='outside.elf';(self.root/'outside.elf').write_bytes(b'launch');self.save()
  with self.assertRaisesRegex(RuntimeError,'outside candidate'):self.obj.frozen_controller_payload('launch')
 def test_read_substitution_rejected(self):
  original=Path.read_bytes
  def substituted(path):return b'changed during read' if path.name=='launch.elf' else original(path)
  with patch.object(Path,'read_bytes',substituted),self.assertRaisesRegex(RuntimeError,'payload differs'):self.obj.frozen_controller_payload('launch')
 def test_full_final_authentication_required(self):
  self.obj.record['accepted']=True;self.obj.final_prefix_authentication()
  self.obj.verify.assert_called_once_with();self.obj.authenticate_llm_client.assert_called_once_with();self.assertTrue(self.obj.record['accepted'])
  self.assertEqual(self.obj.record['prefix_final_authentication']['status'],'PASS')
 def test_final_source_or_receipt_change_never_passes(self):
  for target in ('verify','authenticate_llm_client'):
   self.obj.record={'accepted':False,'prefix_authentication_pending':True,'lock_released':True,'healthy_after':True};setattr(self.obj,target,Mock(side_effect=ValueError('changed')))
   with self.assertRaisesRegex(ValueError,'changed'):self.obj.final_prefix_authentication()
   self.assertFalse(self.obj.record['accepted']);self.assertNotIn('prefix_final_authentication',self.obj.record)
   setattr(self.obj,target,Mock())
 def test_final_budget_exhaustion_rejects(self):
  self.obj.record['accepted']=True;self.obj.phase_remaining.side_effect=[1,RuntimeError('deadline')]
  with self.assertRaisesRegex(RuntimeError,'deadline'):self.obj.final_prefix_authentication()
  self.assertFalse(self.obj.record['accepted'])
 def test_unaccepted_phase_cannot_be_promoted(self):
  self.obj.record['accepted']=False;self.obj.record['prefix_authentication_pending']=False
  with self.assertRaisesRegex(RuntimeError,'has not passed'):self.obj.final_prefix_authentication()
  self.assertFalse(self.obj.record['accepted']);self.obj.verify.assert_not_called()
 def test_legacy_send_still_full_verifies(self):
  self.obj.manifest['llm_client']['phase']='model-gate';self.save();sock=Socket()
  with patch.object(c.socket,'create_connection',return_value=sock):self.obj.send('launch')
  self.obj.verify.assert_called_once_with()
 def test_prefix_imports_are_primed(self):
  names=('source_llm_prefix_log','verify_llm_prefix_readbacks','build_llm_prefix_reference','public_hip_prefix_assets')
  modules={n:types.ModuleType(n) for n in names}
  with patch.dict(sys.modules,modules):self.obj.preload_prefix_validators()
 def test_run_final_gate_failure_retains_rejection_and_unlocks(self):
  class FTP:
   def connect(self,*args,**kwargs):pass
   def login(self,*args):pass
   def sendcmd(self,*args):return '226 SELF transfer mode disabled'
   def close(self):pass
  self.obj.phase='deploy';self.obj.owned=False;self.obj.ftp=None;self.obj.health=lambda:True
  self.obj.deploy=lambda:self.obj.record.update(accepted=True)
  self.obj.authenticate_llm_client.side_effect=ValueError('final receipt changed')
  lock=self.root/'lock.txt'
  with patch.object(c,'LOCK',lock),patch.object(c.ftplib,'FTP',FTP):self.assertEqual(self.obj.run(),1)
  self.assertFalse(self.obj.record['accepted']);self.assertTrue(self.obj.record['lock_released']);self.assertFalse(lock.exists())

class FrozenVerification(unittest.TestCase):
 setUp=Authentication.setUp
 save=Authentication.save
 def verify(self):c.Control.verify(self.obj)
 def test_every_file_read_again_on_every_call(self):
  with patch.object(c,'stream_digest',wraps=c.stream_digest) as digest:
   self.verify();self.verify()
  self.assertEqual(digest.call_count,2*len(self.obj.manifest['files']))
 def test_same_size_change_with_restored_mtime_rejected(self):
  import os
  self.verify();path=self.root/self.obj.manifest['launch'];before=path.stat()
  path.write_bytes(b'LAUNCH');os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
  with self.assertRaisesRegex(RuntimeError,'frozen input changed'):self.verify()
 def test_missing_file_propagates(self):
  (self.root/self.obj.manifest['launch']).unlink()
  with self.assertRaises(FileNotFoundError):self.verify()
 def test_manifest_checked_before_any_file_read(self):
  (self.candidate/'manifest.json').write_bytes(b'changed')
  with patch.object(c,'stream_digest') as digest,self.assertRaisesRegex(RuntimeError,'manifest changed'):self.verify()
  digest.assert_not_called()
 def test_outside_symlink_rejected_even_with_matching_bytes(self):
  with tempfile.TemporaryDirectory() as other:
   target=Path(other)/'launch';target.write_bytes(b'launch')
   path=self.root/self.obj.manifest['launch'];path.unlink();path.symlink_to(target)
   with self.assertRaisesRegex(RuntimeError,'frozen input changed'):self.verify()
 def test_read_error_propagates(self):
  with patch.object(c,'stream_digest',side_effect=OSError('read failed')),self.assertRaisesRegex(OSError,'read failed'):self.verify()
 def test_workers_overlap_but_are_joined_on_failure(self):
  import threading
  for index in range(4):
   path=self.candidate/str(index);path.write_bytes(b'extra')
   self.obj.manifest['files'][str(path.relative_to(self.root))]=c.sha(b'extra')
  self.save()
  barrier=threading.Barrier(4);lock=threading.Lock();active=0;peak=0
  def read(path):
   nonlocal active,peak
   with lock:active+=1;peak=max(peak,active)
   try:
    barrier.wait(timeout=5)
    raise OSError('worker failed')
   finally:
    with lock:active-=1
  with patch.object(c,'stream_digest',side_effect=read),self.assertRaisesRegex(OSError,'worker failed'):self.verify()
  self.assertEqual(peak,4);self.assertEqual(active,0)

if __name__=='__main__':unittest.main()
