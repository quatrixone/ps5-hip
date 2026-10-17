import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json
spec=importlib.util.spec_from_file_location('native_build',Path(__file__).resolve().parents[1]/'tools/build_llm_native_client.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
class NativeBuild(unittest.TestCase):
 def test_scheduler_pair_only_changed_and_allocation_checked(self):
  source=b.ROOT/'.deps/sources/llama-client/ggml/src/ggml-backend.cpp'
  original=source.read_text();generated=b.scheduler.render(original)
  allocation='    sched->context_buffer = (char *) ggml_aligned_malloc(sched->context_buffer_size);\n    GGML_ASSERT(sched->context_buffer != NULL);'
  release='    ggml_aligned_free(sched->context_buffer, sched->context_buffer_size);'
  self.assertEqual(generated.count(allocation),1);self.assertEqual(generated.count(release),1)
  self.assertEqual(generated.replace(allocation,b.scheduler.ALLOCATE).replace(release,b.scheduler.RELEASE),original)
  self.assertNotIn(b.scheduler.ALLOCATE,generated);self.assertNotIn(b.scheduler.RELEASE,generated)
 def test_scheduler_upstream_drift_and_duplicate_rejected(self):
  source=(b.ROOT/'.deps/sources/llama-client/ggml/src/ggml-backend.cpp').read_text()
  for token in (b.scheduler.ALLOCATE,b.scheduler.RELEASE):
   for changed in (source.replace(token,''),source+'\n'+token):
    with self.subTest(token=token),self.assertRaisesRegex(ValueError,'pair changed'):b.scheduler.render(changed)
  with self.assertRaisesRegex(ValueError,'pair changed'):b.scheduler.render(b.scheduler.render(source))
 def test_scheduler_generated_source_provenance_and_tampering(self):
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);source=folder/'upstream.cpp';generated=folder/b.scheduler.GENERATED_NAME
   source.write_text(b.scheduler.ALLOCATE+'\n'+b.scheduler.RELEASE+'\n')
   original=source.read_bytes();generated.write_text(b.scheduler.render(source.read_text()))
   record=b.scheduler.provenance(source,generated)
   self.assertEqual(record,{'classification':'native-scheduler-aligned-context','source':str(source),'source_sha256':b.sha(source),'generated':str(generated),'generated_sha256':b.sha(generated)})
   self.assertEqual(source.read_bytes(),original)
   generated.write_text(generated.read_text().replace('GGML_ASSERT(sched->context_buffer != NULL);',''))
   with self.assertRaisesRegex(ValueError,'source differs'):b.scheduler.provenance(source,generated)
 def test_scheduler_generated_symlink_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);source=folder/'upstream.cpp';target=folder/'target.cpp';generated=folder/b.scheduler.GENERATED_NAME
   source.write_text(b.scheduler.ALLOCATE+'\n'+b.scheduler.RELEASE+'\n');target.write_text(b.scheduler.render(source.read_text()));generated.symlink_to(target)
   with self.assertRaisesRegex(ValueError,'source differs'):b.scheduler.provenance(source,generated)
 def test_actual_locked_selection_has_unique_complete_source_units(self):
  units,models=b.selection(b.ROOT/'.deps/sources/llama-client')
  self.assertEqual(len(units),59);self.assertEqual(len(set(units)),59);self.assertEqual(len(models),161)
  self.assertEqual(len(units)+(len(models)+15)//16+1,71)
  self.assertIn('ggml-backend-reg.cpp',[p.name for p in units]);self.assertEqual(sum(p.name=='ggml-backend-reg.cpp' for p in units),1)
 def test_identity_drift_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory)/'source';p.write_bytes(b'old');inputs={};b.bind(inputs,p);b.bind(inputs,p);p.write_bytes(b'new')
   with self.assertRaisesRegex(ValueError,'input changed'):b.bind(inputs,p)
 def test_fresh_scoped_output_required(self):
  with tempfile.TemporaryDirectory() as directory:
   with self.assertRaisesRegex(ValueError,'fresh'):b.build(Path(directory)/'outside')
  with self.assertRaisesRegex(ValueError,'fresh'):b.build(b.ROOT/'results')
 def test_cmake_path_injection_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   client=Path(directory);(client/'src').mkdir();(client/'src/CMakeLists.txt').write_text('set(LLAMA_CORE_SOURCES ../outside.cpp)')
   with self.assertRaisesRegex(ValueError,'unexpected core'):b.selection(client)
 def test_app_receipt_admission(self):
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);r={'schema':1,'status':'PASS','classification':'source-only-native-client-payload-link','native_executed':False,'gpu_executed':False}
   for key,value in [('schema',0),('status','FAIL'),('classification','prototype'),('native_executed',True),('gpu_executed',True)]:
    (folder/'receipt.json').write_text(json.dumps({**r,key:value}))
    with self.subTest(key=key),self.assertRaisesRegex(ValueError,'unsupported'),patch.object(b,'source') as fetch:
     b.app_objects(folder)
    fetch.assert_not_called()
 def test_app_locked_revision_rejection(self):
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);r={'schema':1,'status':'PASS','classification':'source-only-native-client-payload-link','native_executed':False,'gpu_executed':False,'client_revision':'wrong','sdk_revision':'wrong'}
   (folder/'receipt.json').write_text(json.dumps(r))
   with self.assertRaisesRegex(ValueError,'locked revision'),patch.object(b,'source') as fetch:b.app_objects(folder)
   fetch.assert_not_called()
 def test_artifact_bytes_size_inventory_and_paths(self):
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);p=folder/'0.o';p.write_bytes(b'object');receipt=folder/'receipt.json';receipt.write_text('{}')
   a={'0.o':{'sha256':b.sha(p),'bytes':6}};b.verify_artifacts(folder,a,receipt)
   for broken in ({'0.o':{'sha256':'0'*64,'bytes':6}},{'0.o':{'sha256':b.sha(p),'bytes':5}},{str(p):a['0.o']},{'../0.o':a['0.o']}):
    with self.subTest(broken=broken),self.assertRaises(ValueError):b.verify_artifacts(folder,broken,receipt)
   (folder/'extra.o').write_bytes(b'extra')
   with self.assertRaisesRegex(ValueError,'inventory'):b.verify_artifacts(folder,a,receipt)
 def test_artifact_symlink_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);p=folder/'target';p.write_bytes(b'x');link=folder/'0.o';link.symlink_to(p)
   with self.assertRaisesRegex(ValueError,'unsafe'):b.verify_artifacts(folder,{'0.o':{'sha256':b.sha(p),'bytes':1}},folder/'receipt.json')
 def test_command_success_inventory_and_arguments(self):
  row={'label':'compile-0','argv':['compiler','-c','source','-o','0.o'],'exit_code':0};expected={'compile-0':row['argv']};artifacts={'compile-0.stdout':{},'compile-0.stderr':{}}
  b.verify_commands([row],expected,[row],artifacts)
  for bad in ([],[row,row],[{**row,'exit_code':1}],[{**row,'timed_out':True}],[{**row,'argv':['compiler','other']}],[{**row,'label':'other'}]):
   with self.subTest(bad=bad),self.assertRaises(ValueError):b.verify_commands(bad,expected,bad,artifacts)
  with self.assertRaisesRegex(ValueError,'record'):b.verify_commands([row],expected,[],artifacts)
  with self.assertRaisesRegex(ValueError,'output'):b.verify_commands([row],expected,[row],{})


class PreparedGenerated(unittest.TestCase):
 def fixture(self, folder):
  texts={'llama-version.h':'llama\n','ggml-version.h':'ggml\n',b.scheduler.GENERATED_NAME:'scheduler\n','unity-0.cpp':'unity\n'}
  for name,text in texts.items():(folder/name).write_bytes(text.encode())
  return texts
 def test_verified_inputs_do_not_require_or_rewrite_old_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);texts=self.fixture(folder);(folder/'receipt.json').write_text('stale receipt')
   self.assertEqual(b.verify_generated(folder,texts),folder)
   self.assertEqual((folder/'receipt.json').read_text(),'stale receipt')
 def test_same_size_content_missing_and_header_shadow_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);texts=self.fixture(folder)
   for name,text in texts.items():
    (folder/name).write_bytes(b'X'+text.encode()[1:])
    with self.subTest(name=name),self.assertRaisesRegex(ValueError,'differs'):b.verify_generated(folder,texts)
    (folder/name).write_bytes(text.encode())
   (folder/'extra.h').write_text('shadow')
   with self.assertRaisesRegex(ValueError,'inventory'):b.verify_generated(folder,texts)
   (folder/'extra.h').unlink();(folder/'unity-0.cpp').unlink()
   with self.assertRaisesRegex(ValueError,'differs'):b.verify_generated(folder,texts)
 def test_nested_header_directory_and_symlinks_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);texts=self.fixture(folder);(folder/'nested').mkdir()
   with self.assertRaisesRegex(ValueError,'inventory'):b.verify_generated(folder,texts)
   (folder/'nested').rmdir();target=folder/'target';target.write_text(texts['unity-0.cpp']);(folder/'unity-0.cpp').unlink();(folder/'unity-0.cpp').symlink_to(target)
   with self.assertRaisesRegex(ValueError,'differs'):b.verify_generated(folder,texts)
 def test_reused_flags_and_generated_paths_stay_old_fresh_loader_separate(self):
  client=b.ROOT/'.deps/sources/llama-client';_,models=b.selection(client)
  texts=b.generated_texts(client,models,'revision')
  self.assertNotIn(b.loader.GENERATED_NAME,texts)
  self.assertEqual(len(texts),14)
  self.assertEqual(texts['unity-0.cpp'],'\n'.join('#include "'+str(p)+'"' for p in models[:16])+'\n')
  old=Path('/old');fresh=Path('/fresh')
  self.assertEqual(b.compile_flags(old,client)[7],'-I/old')
  self.assertNotEqual(b.compile_flags(old,client),b.compile_flags(fresh,client))
 def test_unexpected_version_bytes_and_directory_alias_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);folder=root/'inputs';folder.mkdir();texts=self.fixture(folder)
   (folder/'ggml-version.h').write_bytes(b'ggml\r\n')
   with self.assertRaisesRegex(ValueError,'differs'):b.verify_generated(folder,texts)
   (folder/'ggml-version.h').write_bytes(texts['ggml-version.h'].encode());alias=root/'alias';alias.symlink_to(folder,target_is_directory=True)
   with self.assertRaisesRegex(ValueError,'unsafe'):b.verify_generated(alias,texts)

if __name__=='__main__':unittest.main()
