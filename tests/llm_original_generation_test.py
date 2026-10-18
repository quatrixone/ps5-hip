"""Original-source CLI routing; no model, compiler or dependency build."""
import importlib.util,pathlib,sys,unittest,json,tempfile,types,hashlib,contextlib
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
spec=importlib.util.spec_from_file_location('original_generation_builder',ROOT/'tools/build_llm_generation_reference.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
class OriginalGeneration(unittest.TestCase):
 def test_default_and_explicit_routes(self):
  output=ROOT/'results/test';cpu=ROOT/'build/cpu';model=ROOT/'oracle.gguf';source=ROOT/'originals'
  default=builder.command_plan(output,cpu,model)
  self.assertEqual(default['capture'],list(map(str,[output/'trace',model,output/'raw'])))
  self.assertNotIn('-DPS5HIP_ORIGINAL_MODEL=1',default['compile'])
  original=builder.command_plan(output,cpu,model,original_source=source)
  self.assertEqual(original['capture'],list(map(str,[output/'trace',source,output/'raw','--original-source'])))
  self.assertNotIn(str(model),original['capture'])
  self.assertIn('-DPS5HIP_ORIGINAL_MODEL=1',original['compile'])
  self.assertIn('-DPS5HIP_ORIGINAL_MODEL=1',original['headers'])
 def test_original_corpus_rejected(self):
  with self.assertRaisesRegex(ValueError,'short'):
   builder.command_plan(ROOT/'out',ROOT/'cpu',ROOT/'model','corpus',ROOT/'source')
 def test_source_lifetime_and_complete_gate(self):
  source=(ROOT/'tests/llm_generation_reference.cpp').read_text()
  self.assertLess(source.index('direct_tiny::Metadata metadata'),source.index('model(nullptr,llama_model_free)'))
  self.assertIn('callback->complete()',source)
  self.assertIn('llama_model_load_from_file(argv[1],mp)',source)

 def test_relabelled_original_cannot_be_oracle(self):
  with tempfile.TemporaryDirectory() as tmp:
   folder=pathlib.Path(tmp)
   (folder/'receipt.json').write_text(json.dumps(dict(classification='successful-greedy-generation-cpu-reference',original_source='/originals')))
   with self.assertRaisesRegex(ValueError,'not an independent GGUF oracle'):
    builder.authenticate_generation_reference(folder)

 def exercise_build(self,mutation=None):
  with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
   root=pathlib.Path(tmp);cpu=root/'build/cpu';original=root/'originals';reference=root/'results/reference';output=root/'results/result';model=root/'oracle.gguf'
   def put(name,text='fixture'):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);return p
   sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
   names=['tools/prepare_llm_native_loader.py','tools/llm_user_loader.cmake','tools/build_llm_generation_numeric_test.py','tools/build_llm_executed_graph.py','tests/llm_original_generation_test.py','tests/llm_generation_reference.cpp','llm-inputs.json','.deps/sources/llama-client/src/llama-model-loader.cpp']
   for name in names:put(name)
   put('oracle.gguf');put('build/cpu/llama-model-loader-user.cpp','rendered');put('originals/config.json');put('originals/model.safetensors')
   lock={'metadata':{'config.json':{'bytes':7,'sha256':sha(original/'config.json')}},'weights':{'file_bytes':7,'expected_file_sha256':sha(original/'model.safetensors')}}
   put('models/tinyllama-1.1b-chat-v1.0.json',json.dumps(lock))
   cpu_receipt={'identity':{'inputs':{n:sha(root/n) for n in names[:2]}}}
   if mutation=='unpatched':cpu_receipt['identity']['inputs'].clear()
   put('build/cpu/receipt.json',json.dumps(cpu_receipt))
   if mutation=='source':put('originals/model.safetensors','changed')
   oracle={'cpu_build':str(cpu if mutation!='oracle' else root/'wrong'),'model':{'path':str(model)}}
   compare=types.ModuleType('build_llm_generation_numeric_test');compare.compare_generation=lambda *a,**k:([{'pass_budget':mutation!='numeric'}],dict.fromkeys(range(384)),16)
   renderer=types.ModuleType('prepare_llm_native_loader');renderer.render_user=lambda s:'rendered'
   stack.enter_context(patch.dict(sys.modules,{'prepare_llm_native_loader':renderer,'build_llm_generation_numeric_test':compare}))
   stack.enter_context(patch.object(builder,'ROOT',root));stack.enter_context(patch.object(builder,'SOURCES',[root/n for n in ('tools/build_llm_executed_graph.py','tests/llm_generation_reference.cpp')]))
   stack.enter_context(patch.object(builder,'authenticate_generation_reference',return_value=(oracle,{})))
   stack.enter_context(patch.object(builder,'validate',return_value={'requests':16,'raw_files':384}))
   stack.enter_context(patch.object(builder.provenance,'cpu_inputs',return_value={str(model):sha(model)}))
   real_sha=builder.provenance.sha
   stack.enter_context(patch.object(builder.provenance,'sha',side_effect=lambda p:sha(p) if pathlib.Path(p).is_relative_to(root) else 'tool-identity'))
   stack.enter_context(patch.object(builder.provenance,'dependencies',return_value=[]));stack.enter_context(patch.object(builder.provenance,'runtime_inputs',return_value={}))
   calls=[]
   def recheck(inputs):
    calls.append(dict(inputs))
    if mutation=='drift' and len(calls)==2:raise ValueError('final input drift')
   stack.enter_context(patch.object(builder.provenance,'recheck',side_effect=recheck))
   def execute(argv,**kwargs):
    argv=list(map(str,argv))
    if '-o' in argv:pathlib.Path(argv[argv.index('-o')+1]).write_bytes(b'binary')
    stdout=builder.LINE.encode() if argv[0]==str(output/'trace') else b''
    return types.SimpleNamespace(returncode=0,stdout=stdout,stderr=b'')
   stack.enter_context(patch.object(builder.subprocess,'run',side_effect=execute))
   builder.build(cpu,model,output,original_source=original,reference=reference)
   data=json.loads((output/'receipt.json').read_text())
   self.assertEqual(data['classification'],'original-format-host-short-generation-compared')
   self.assertEqual(data['commands'][-1]['argv'],[str(output/'trace'),str(original),str(output/'raw'),'--original-source'])
   self.assertIn(str(original/'model.safetensors'),data['inputs'])
   for name in ('tools/build_llm_generation_numeric_test.py','tools/build_llm_executed_graph.py'):
    self.assertIn(str(root/name),data['inputs'])
   self.assertEqual(len(calls),2)
 def test_original_build_accepts_bound_route(self):self.exercise_build()
 def test_original_build_rejects_bad_admission_or_comparison(self):
  for kind in ('unpatched','source','oracle','numeric','drift'):
   with self.subTest(kind=kind),self.assertRaises(ValueError):self.exercise_build(kind)
if __name__=='__main__':unittest.main()
