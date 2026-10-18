from pathlib import Path
import sys,json,tempfile,unittest,os
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import source_console as c
class FTP:
 def connect(self,*a,**k):pass
 def login(self,*a):pass
 def sendcmd(self,*a):return '226 SELF transfer mode disabled'
 def close(self):pass
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.candidate=self.root/'results/candidate';self.candidate.mkdir(parents=True)
  self.obj=c.Control.__new__(c.Control);self.obj.candidate=self.candidate;self.obj.out=self.candidate/'deploy';self.obj.out.mkdir()
  self.obj.fixture='llm_client';self.obj.phase='deploy';self.obj.owned=False;self.obj.ftp=None;self.obj.token='ours';self.obj.record={'accepted':False,'promoted':True}
  self.obj.manifest={'llm_client':{'phase':'model-full'},'files':{}}
  self.paths=['results/candidate/package/pkg','results/candidate/controllers/launch.elf','tools/source_console.py','ports/driver.cpp','.deps/sdk/compiler','build/old/object','results/original/raw']
  for name in self.paths:
   p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original');self.obj.manifest['files'][name]=c.sha(b'original')
  self.obj.manifest.update(package=self.paths[0],app='results/candidate/package/app',launch=self.paths[1],close=self.paths[1],snapshot=self.paths[1],post_snapshot=self.paths[1],package_files={'config':c.sha(b'original')},model_full={'reference':'results/candidate/full-reference'})
  for name in ('package/app/config','full-reference/receipt.json','full-reference/raw/result','build-receipt.json','llm-client-native-audit.json'):
   p=self.candidate/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original');self.obj.manifest['files'][str(p.relative_to(self.root))]=c.sha(b'original')
  raw=json.dumps(self.obj.manifest).encode();(self.candidate/'manifest.json').write_bytes(raw);self.obj.manifest_sha=c.sha(raw)
  self.obj.health=Mock(return_value=True);self.obj.phase_remaining=Mock(return_value=4200)
  self.obj.deploy=lambda:self.obj.record.update(accepted=True)
  self.obj.authenticate_llm_client=Mock()
  self.lock=self.root/'primary';self.secondary=self.root/'secondary';self.secondary.write_text('ours-secondary')
  self.addCleanup(patch.stopall);patch.object(c,'ROOT',self.root).start();patch.object(c,'LOCK',self.lock).start();patch.object(c.ftplib,'FTP',FTP).start()
 def test_execution_scope(self):
  with patch.object(c,'stream_digest',wraps=c.stream_digest) as digest:self.obj.verify(execution_only=True)
  self.assertEqual({str(call.args[0].relative_to(self.root)) for call in digest.call_args_list},{name for name in self.obj.manifest['files'] if name.startswith('results/candidate/') or name in self.paths[2:4]})
 def test_critical_same_size_restored_mtime_rejects(self):
  for name in (name for name in self.obj.manifest['files'] if name.startswith('results/candidate/') or name in self.paths[2:4]):
   p=self.root/name;before=p.stat();p.write_bytes(b'CHANGED!');os.utime(p,ns=(before.st_atime_ns,before.st_mtime_ns))
   with self.assertRaisesRegex(RuntimeError,'frozen input changed'):self.obj.verify(execution_only=True)
   p.write_bytes(b'original')
 def test_manifest_rejected_before_reads(self):
  (self.candidate/'manifest.json').write_bytes(b'changed')
  with patch.object(c,'stream_digest') as digest,self.assertRaisesRegex(RuntimeError,'manifest changed'):self.obj.verify(execution_only=True)
  digest.assert_not_called()
 def save(self):
  raw=json.dumps(self.obj.manifest).encode();(self.candidate/'manifest.json').write_bytes(raw);self.obj.manifest_sha=c.sha(raw)
 def test_missing_manifest_critical_member(self):
  for name in tuple(self.obj.manifest['files']):
   if not name.startswith('results/candidate/'):continue
   digest=self.obj.manifest['files'].pop(name);self.save()
   with self.assertRaisesRegex(RuntimeError,'missing from manifest'):self.obj.verify(execution_only=True)
   self.obj.manifest['files'][name]=digest;self.save()
 def test_forged_critical_location(self):
  for location in ('build/old','results/other'):
   directory=self.root/location;directory.mkdir(parents=True,exist_ok=True)
   for key in ('package','launch','close','snapshot','post_snapshot','app'):
    original=self.obj.manifest[key];self.obj.manifest[key]=location;self.save()
    with self.assertRaisesRegex(RuntimeError,'outside candidate'):self.obj.verify(execution_only=True)
    self.obj.manifest[key]=original;self.save()
   original=self.obj.manifest['model_full']['reference'];self.obj.manifest['model_full']['reference']=location;self.save()
   with self.assertRaisesRegex(RuntimeError,'outside candidate'):self.obj.verify(execution_only=True)
   self.obj.manifest['model_full']['reference']=original;self.save()
 def test_final_admission_is_not_replayed(self):
  self.obj.authenticate_llm_client.side_effect=ValueError('producer changed');self.assertEqual(self.finish(),1)
  self.assertEqual(self.obj.finish_prefix_authentication(),1);self.obj.authenticate_llm_client.assert_called_once()
 def test_candidate_escape_rejected(self):
  p=self.root/self.paths[0];p.unlink();p.symlink_to('/etc/hostname')
  with self.assertRaisesRegex(RuntimeError,'outside candidate'):self.obj.verify(execution_only=True)
 def test_pending_is_unaccepted_and_no_final_work_under_primary(self):
  self.obj.verify=Mock()
  self.assertEqual(self.obj.run(defer_prefix_authentication=True),2)
  self.assertFalse(self.lock.exists());self.assertTrue(self.secondary.exists())
  self.obj.verify.assert_not_called();self.obj.authenticate_llm_client.assert_not_called()
  receipt=json.loads((self.obj.out/'receipt.json').read_text());self.assertFalse(receipt['accepted']);self.assertTrue(receipt['prefix_authentication_pending'])
  with self.assertRaisesRegex(RuntimeError,'secondary lock'):self.obj.final_prefix_authentication()
 def finish(self):
  self.assertEqual(self.obj.run(defer_prefix_authentication=True),2);self.secondary.unlink();self.obj.record['secondary_lock_released']=True
  return self.obj.finish_prefix_authentication()
 def test_full_final_success_after_both_released(self):
  original=self.obj.verify
  def verify():
   self.assertFalse(self.lock.exists());self.assertFalse(self.secondary.exists());original()
  self.obj.verify=verify
  self.assertEqual(self.finish(),0);self.assertTrue(self.obj.record['accepted']);self.obj.authenticate_llm_client.assert_called_once()
 def test_full_final_sdk_mutation_rejects_and_preserves_recovery(self):
  def deploy():
   self.obj.record['accepted']=True;p=self.root/'.deps/sdk/compiler';before=p.stat();p.write_bytes(b'CHANGED!');os.utime(p,ns=(before.st_atime_ns,before.st_mtime_ns))
  self.obj.deploy=deploy
  self.assertEqual(self.finish(),1);self.assertFalse(self.obj.record['accepted']);self.assertTrue(self.obj.record['promoted']);self.assertFalse(self.lock.exists());self.assertFalse(self.secondary.exists())
 def test_deep_final_failure_rejects(self):
  self.obj.authenticate_llm_client.side_effect=ValueError('producer changed')
  self.assertEqual(self.finish(),1);self.assertFalse(self.obj.record['accepted'])
 def test_cleanup_failure_never_finishes(self):
  self.obj.health.side_effect=[True,RuntimeError('unhealthy')]
  self.assertEqual(self.obj.run(defer_prefix_authentication=True),1);self.obj.authenticate_llm_client.assert_not_called();self.assertFalse(self.obj.record['accepted'])
 def test_other_owner_lock_retained(self):
  def deploy():self.obj.record['accepted']=True;self.lock.write_text('someone-else')
  self.obj.deploy=deploy
  self.assertEqual(self.obj.run(defer_prefix_authentication=True),1);self.assertEqual(self.lock.read_text(),'someone-else');self.obj.authenticate_llm_client.assert_not_called()
 def test_default_run_finalizes_after_primary_release(self):
  self.secondary.unlink();original=self.obj.verify
  def verify():self.assertFalse(self.lock.exists());original()
  self.obj.verify=verify
  self.assertEqual(self.obj.run(),0);self.assertTrue(self.obj.record['accepted'])
if __name__=='__main__':unittest.main()
