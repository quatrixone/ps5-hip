import importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('driver',Path(__file__).resolve().parents[1]/'tools/build_llm_native_app.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
class Driver(unittest.TestCase):
 def test_phase_exact(self):
  with tempfile.TemporaryDirectory() as name:
   p=Path(name)/'phase';d.write_phase(p);self.assertEqual(p.read_bytes(),b'vocab\n')
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
if __name__=='__main__':unittest.main()
