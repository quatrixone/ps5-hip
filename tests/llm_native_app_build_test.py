import importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('driver',Path(__file__).resolve().parents[1]/'tools/build_llm_native_app.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
class Driver(unittest.TestCase):
 def test_phase_exact(self):
  with tempfile.TemporaryDirectory() as name:
   p=Path(name)/'phase';d.write_phase(p);self.assertEqual(p.read_bytes(),b'vocab\n')
 def test_full_phase_requires_its_own_fixture(self):
  with tempfile.TemporaryDirectory() as name:
   p=Path(name)/'phase';d.write_phase(p,'model-full');self.assertEqual(p.read_bytes(),b'model-full\n')
  d.validate_phase_arguments('model-full','model','source','cpu',full_fixture='full')
  for phase in ('vocab','model-gate','model-prefix','model-layer'):
   with self.assertRaisesRegex(ValueError,'full fixture'):
    d.validate_phase_arguments(phase,'model','source','cpu',full_fixture='full')
  with self.assertRaisesRegex(ValueError,'full fixture'):d.validate_phase_arguments('model-full','model','source','cpu')
  with self.assertRaisesRegex(ValueError,'prerequisites'):d.validate_phase_arguments('model-full',None,'source','cpu',full_fixture='full')
 def test_outputs_never_treated_as_input(self):
  with tempfile.TemporaryDirectory() as name:
   folder=Path(name);a=folder/'input';o=folder/'output';dep=folder/'out.d'
   for p in (a,o,dep):p.write_bytes(b'x')
   self.assertEqual(d.file_inputs(['compiler','--in',str(a),'--out',str(o),'-MF',str(dep)]),[a])
   self.assertEqual(d.file_inputs(['compiler','-c',str(a),'-o',str(o)]),[a])
 def receipt(self,folder):
  return dict(schema=1,status='PASS',classification='upstream-vocabulary-only-gguf-tokenization',tensor_count=0,metadata_fields=45,tokenization_cases=32,original_and_vocabulary_tokens_bitwise_equal=True,native_executed=False,gpu_executed=False,inputs={},artifacts={'tokenize':'unused'})
 def test_missing_vocabulary_identities_rejected(self):
  with tempfile.TemporaryDirectory() as name:
   folder=Path(name);r=self.receipt(folder);(folder/'receipt.json').write_text(json.dumps(r))
   with self.assertRaisesRegex(ValueError,'missing vocabulary identities'):d.vocab_inputs(folder)
 def test_stale_and_omitted_vocabulary_sources_rejected(self):
  with tempfile.TemporaryDirectory() as name:
   folder=Path(name);p=folder/'input';p.write_bytes(b'current');r=self.receipt(folder)
   r['inputs']={str(p):'0'*64};(folder/'receipt.json').write_text(json.dumps(r))
   with self.assertRaisesRegex(ValueError,'stale vocabulary input'):d.vocab_inputs(folder)
   r['inputs']={str(p):d.b.sha(p)};(folder/'receipt.json').write_text(json.dumps(r))
   with self.assertRaisesRegex(ValueError,'missing vocabulary source'):d.vocab_inputs(folder)
 def test_rehashed_corrupt_asset_does_not_match_evidence(self):
  with tempfile.TemporaryDirectory() as name:
   folder=Path(name);asset=folder/'vocab.gguf';asset.write_bytes(b'reviewed');evidence=folder/'evidence.json'
   spec=dict(status='PASS_PRODUCTION_VOCABULARY_FIXTURE',tensor_count=0,metadata_fields=45,tokenization_cases=32,vocab_bytes=8,asset_sha256=d.b.sha(asset));evidence.write_text(json.dumps(spec))
   d.validate_vocab_asset(asset,evidence)
   asset.write_bytes(b'changed!');self.assertNotEqual(d.b.sha(asset),spec['asset_sha256'])
   # A new self-consistent receipt cannot replace the independently bound evidence.
   forged={'artifacts':{'vocab.gguf':d.b.sha(asset)}};self.assertEqual(forged['artifacts']['vocab.gguf'],d.b.sha(asset))
   with self.assertRaisesRegex(ValueError,'reviewed bytes'):d.validate_vocab_asset(asset,evidence)
 def test_host_link_plan_resolves_implicit_and_script_inputs(self):
  with tempfile.TemporaryDirectory() as name:
   folder=Path(name);ld=folder/'ld';obj=folder/'unit.o';shared=folder/'libx.so.1';archive=folder/'libextra.a'
   for p in (ld,obj,shared):p.write_bytes(b'\x7fELFrest')
   archive.write_bytes(b'!<arch>\n');script=folder/'libx.so';script.write_text('/* source */ GROUP ( libx.so.1 AS_NEEDED ( -lextra ) )')
   plan=' '.join(json.dumps(str(p)) for p in [ld,'-L'+str(folder),obj,'-lx','-o',folder/'not-created'])
   self.assertEqual(set(d.host_link_inputs(plan)),{ld,obj,script,shared,archive})
   script.write_text('GROUP ( missing.so )')
   with self.assertRaisesRegex(ValueError,'unresolved'):d.host_link_inputs(plan)
   script.write_text('SEARCH_DIR ( /unexpected )')
   with self.assertRaisesRegex(ValueError,'unsupported'):d.host_link_inputs(plan)
 def test_host_link_plan_rejects_extra_commands(self):
  with self.assertRaisesRegex(ValueError,'unexpected'):d.host_link_inputs('"/bin/true"\n"/bin/false"')
 def test_package_scratch_receipt_survives_deletion_without_skipping_other_inputs(self):
  with tempfile.TemporaryDirectory() as name:
   out=Path(name);scratch=out/'package/pack-known/PPSA99014.ffpkg';scratch.parent.mkdir(parents=True);scratch.write_bytes(b'package')
   final=out/'package/PPSA99014.ffpkg';final.write_bytes(scratch.read_bytes());other=out/'tool';other.write_bytes(b'tool')
   persistent,temporary=d.command_file_inputs(out,'fsck',['dotnet',str(other),str(scratch)],'PPSA99014')
   self.assertEqual(persistent,[other]);self.assertEqual(temporary,[scratch]);digest=d.b.sha(scratch);scratch.unlink();self.assertEqual(digest,d.b.sha(final))
   persistent,temporary=d.command_file_inputs(out,'package-extract',['dotnet',str(final)],'PPSA99014');self.assertEqual(persistent,[final]);self.assertFalse(temporary)
   scratch.write_bytes(b'package');persistent,temporary=d.command_file_inputs(out,'unrelated',['tool',str(scratch)],'PPSA99014');self.assertEqual(persistent,[scratch]);self.assertFalse(temporary)
class RunnerSourceClosure(unittest.TestCase):
 def test_recorded_runner_restored_before_final_authentication(self):
  import ast,sys
  import build_llm_prefix_reference as reference
  # The loader above must register its module for the ownership walk to match a real CLI run.
  with patch.dict(sys.modules,{'driver':d}):
   original=set(reference.sources())
   with patch.object(d.b,'run',d.build):
    instrumented=set(reference.sources())
    self.assertIn(Path(d.__file__).resolve(),instrumented-original)
   self.assertEqual(set(reference.sources()),original)
  node=next(n for n in ast.parse(Path(d.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='build')
  guarded=next(n for n in node.body if isinstance(n,ast.Try))
  restore=next(i for i,n in enumerate(guarded.body) if isinstance(n,ast.Assign) and ast.unparse(n)=='b.run = original_run')
  audit=next(i for i,n in enumerate(guarded.body) if isinstance(n,ast.Expr) and ast.unparse(n).startswith("merge(ar['inputs'])"))
  final_auth=next(i for i,n in enumerate(guarded.body) if isinstance(n,ast.If) and 'app prerequisite changed' in ast.unparse(n))
  self.assertLess(audit,restore);self.assertLess(restore,final_auth)

class PreparedSource(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.old=self.root/'build/old';self.new=self.root/'build/new'
  for folder in (self.old,self.new):
   (folder/'source-base/source').mkdir(parents=True);(folder/'source-public').mkdir()
   (folder/'source-base/receipt.json').write_text(json.dumps({'classification':'exact-combined-native-clr-gpu-source'}))
   (folder/'source-public/unit.cpp').write_text('int x;')
   (folder/'source-public/public-hip-preparation.json').write_text(json.dumps({'classification':'exact-public-hip-patch-roundtrip','inputs':{str(folder/'source-base/source/unit.cpp'):'a'*64}}))
  self.mock=patch.object(d,'ROOT',self.root);self.mock.start();self.addCleanup(self.mock.stop)
 def select(self):return d.select_prepared_source(self.new,self.new/'source-public',self.old)
 def test_identical_tree_preserves_exact_compiler_paths(self):
  before={str(p):p.read_bytes() for p in self.old.rglob('*') if p.is_file()}
  selected=self.select();self.assertEqual(selected,self.old/'source-public')
  with patch.object(Path,'is_dir',return_value=True):
   self.assertEqual(d.b.compiler_flags(selected),d.b.compiler_flags(self.old/'source-public'))
   self.assertNotEqual(d.b.compiler_flags(selected),d.b.compiler_flags(self.new/'source-public'))
  self.assertEqual(before,{str(p):p.read_bytes() for p in self.old.rglob('*') if p.is_file()})
  self.assertEqual(d.select_prepared_source(self.new,self.new/'source-public',None),self.new/'source-public')
 def test_content_and_new_file_rejected(self):
  p=self.old/'source-public/unit.cpp';p.write_text('changed')
  with self.assertRaisesRegex(ValueError,'content'):self.select()
  p.write_text('int x;');(self.old/'source-public/extra.h').touch()
  with self.assertRaisesRegex(ValueError,'inventory'):self.select()
 def test_missing_and_changed_receipt_rejected(self):
  p=self.old/'source-base/receipt.json';p.unlink()
  with self.assertRaisesRegex(ValueError,'receipt missing'):self.select()
  p.write_text(json.dumps({'classification':'exact-combined-native-clr-gpu-source','extra':True}))
  with self.assertRaisesRegex(ValueError,'base identity'):self.select()
 def test_symlink_rejected(self):
  (self.old/'source-public/alias').symlink_to(self.old/'source-public/unit.cpp')
  with self.assertRaisesRegex(ValueError,'symlink'):self.select()
 def test_outside_and_alias_path_rejected(self):
  with self.assertRaisesRegex(ValueError,'distinct'):d.select_prepared_source(self.new,self.new/'source-public',self.root/'outside')
  with self.assertRaisesRegex(ValueError,'distinct'):d.select_prepared_source(self.new,self.new/'source-public',self.new)
  alias=self.root/'build/alias';alias.symlink_to(self.old,target_is_directory=True)
  with self.assertRaisesRegex(ValueError,'ambiguous'):d.select_prepared_source(self.new,self.new/'source-public',alias)
 def test_unexpected_or_colliding_receipt_reference_rejected(self):
  p=self.old/'source-public/public-hip-preparation.json';data=json.loads(p.read_text())
  data['inputs'][str(self.old/'unexpected')]='a'*64;p.write_text(json.dumps(data))
  with self.assertRaisesRegex(ValueError,'unexpected'):self.select()
  del data['inputs'][str(self.old/'unexpected')];data['inputs'][str(self.new/'source-base/source/unit.cpp')]='a'*64;p.write_text(json.dumps(data))
  with self.assertRaisesRegex(ValueError,'ambiguous'):self.select()

class PreparedSourceParents(unittest.TestCase):
 setUp=PreparedSource.setUp
 def test_base_receipt_and_parent_symlinks_rejected(self):
  for folder in (self.old,self.new):
   for target in (folder/'source-base',folder/'source-base/receipt.json'):
    with self.subTest(path=str(target)):
     escaped=self.root/'escaped';target.rename(escaped);target.symlink_to(escaped,target_is_directory=escaped.is_dir())
     try:
      with self.assertRaisesRegex(ValueError,'base receipt'):d.select_prepared_source(self.new,self.new/'source-public',self.old)
     finally:target.unlink();escaped.rename(target)

class RelativeCommandInputs(unittest.TestCase):
 def test_relative_operand_uses_command_cwd(self):
  with tempfile.TemporaryDirectory() as tmp:
   folder=Path(tmp);p=folder/'unit.cpp';p.write_text('source')
   self.assertEqual(d.file_inputs(['compiler','-c','unit.cpp','-o','unit.o'],cwd=folder),[p])
   self.assertEqual(d.command_file_inputs(folder,'compile',['compiler','unit.cpp'],'title',cwd=folder),([p],[]))
 def test_recorded_runner_forwards_and_records_cwd(self):
  import ast
  tree=ast.parse(Path(d.__file__).read_text())
  runner=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='recorded_run')
  self.assertIn('cwd',[a.arg for a in runner.args.args])
  text=ast.unparse(runner)
  self.assertIn("'cwd': str(ROOT if cwd is None else cwd)",text)
  self.assertIn("original_run(out, label, argv, **{} if cwd is None else {'cwd': cwd})",text)

if __name__ == '__main__':
    unittest.main()
