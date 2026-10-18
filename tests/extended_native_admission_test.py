import pathlib,sys,tempfile,unittest,subprocess,json,shutil
from unittest.mock import patch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tests')]
import public_hip_full_assets as assets
import build_public_hip as builder
import audit_public_hip_llm_client_native as auditor
import build_llm_generation_reference as generation
from public_hip_full_assets_test import FullAssets
from llm_client_native_audit_test import Full
class Tests(unittest.TestCase):
 def test_exact_writer_inventory_and_rejections(self):
  source=r'''
#include "extended_title_output.h"
#include <cassert>
#include <string>
int main(){using namespace ps5hip_generation;unsigned files=0;size_t bytes=0;
for(const auto& c:extended_cases)for(unsigned run=0;run<2;++run)for(unsigned pos=0;pos<c.prompt_count+c.generated-1;++pos)for(unsigned stage=0;stage<24;++stage){
 if(stage<23&&!extended_checkpoint(c,pos))continue;std::string suffix=stage<22?"layer-"+std::to_string(stage):stage==22?"norm":"logits";
 std::string n=std::string(c.id)+"/run-"+std::to_string(run)+"-pos-"+std::to_string(pos)+"-"+suffix+".f32le";size_t size=stage==23?128000:8192;
 assert(extended_output_size(n.c_str(),size));assert(!extended_output_size(n.c_str(),size-1));++files;bytes+=size;
}
assert(files==2096&&bytes==158064640); assert(extended_output_size("extended.json",65536)); assert(!extended_output_size("extended.json",65537)); assert(extended_output_size("short64/generation.json",1048576));
for(auto n:{"extended.json","short64/generation.json","short64/steps.jsonl","eos/generation.json","eos/steps.jsonl","boundary/generation.json","boundary/steps.jsonl"}){assert(extended_output_size(n,1));assert(!extended_output_size(n,0));assert(!extended_output_size(n,1048577));}
for(auto n:{"../extended.json","other/generation.json","boundary/../generation.json","short64/run-2-pos-0-logits.f32le","boundary/run-0-pos-511-logits.f32le","boundary/run-0-pos-999999999-logits.f32le","boundary/run-0-pos-0255-logits.f32le","boundary/run-0-pos-002-logits.f32le","boundary/run-0-pos-2-layer-0.f32le","eos/run-0-pos-9-logits.f32le","short64/run-0-pos-68-logits.f32le","short64/run-0-pos-2-layer-0.f32le","boundary/run-0-pos-510-logits.f32le/extra"})assert(!extended_output_limit(n));
}
'''
  with tempfile.TemporaryDirectory() as folder:
   folder=pathlib.Path(folder);cpp=folder/'writer.cpp';cpp.write_text(source);binary=folder/'writer'
   subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-misleading-indentation','-fsanitize=address,undefined','-fno-sanitize-recover=all','-I'+str(ROOT/'ports/ggml'),'-I'+str(ROOT/'ports/ggml'),str(cpp),'-o',str(binary)],check=True)
   subprocess.run([str(binary)],check=True,timeout=10)
 def test_extended_exact_abi_and_source_binding(self):
  body=Full.smoke_body().replace('$0x1','$0x4');auditor.full_profile_operand(body,'run','extended')
  for value in ('0x1','0x2','0x3'):
   with self.assertRaises(RuntimeError):auditor.full_profile_operand(body.replace('0x4',value),'run','extended')
  for name in ('ports/ggml/extended_generation.h','ports/ggml/extended_generation_cases.def','ports/ggml/extended_title_output.h'):self.assertIn(name,auditor.FULL_SOURCES)
 def test_unqualified_deadline_rejects_before_compile(self):
  with patch.object(builder,'compile_bound_unit') as compile,self.assertRaisesRegex(ValueError,'watchdog not qualified'):
   builder.compile_units(pathlib.Path('/nonexistent'),pathlib.Path('/nonexistent'),fixture='llm_client',client_phase='model-full',full_profile='extended')
  compile.assert_not_called()
  title=(ROOT/'tools/public_hip_llm_client_console_title.cpp').read_text()
  self.assertIn('#error extended native watchdog requires measured corpus timing',title)
  self.assertIn('NativeFullProfile::extended',title)
 def test_asset_branch_inventory_and_profile(self):
  fixture=FullAssets();fixture.setUp()
  try:
   shutil.rmtree(fixture.reference/'raw');cases=generation.extended_cases();files={}
   for c in cases:
    prefix=c['id']+'/'
    for run in range(2):
     for pos in range(c['prompt_count']+c['generated']-1):
      for stage in generation.STAGES if pos in c['checkpoints'] else ('logits',):files[prefix+f'run-{run}-pos-{pos}-{stage}.f32le']=b'x'
    for n in ('generation.json','steps.jsonl'):files[prefix+n]=b'{}'
   files['extended.json']=b'{}';self.assertEqual(len(files),2103)
   for name,data in files.items():fixture.write('reference/raw/'+name,data)
   model=fixture.root/'model.gguf';inputs={str(p):assets.sha(p) for p in [model,fixture.reference/'receipt.json',*[p for p in (fixture.reference/'raw').rglob('*') if p.is_file()]]}
   data=dict(fixture.refdata,classification='successful-greedy-generation-cpu-reference',profile='extended',requests=1176,raw_files=2096)
   with patch.object(generation,'authenticate_generation_reference',return_value=(data,inputs)),patch.object(generation,'validate_extended',return_value={'requests':1176,'raw_files':2096}):
    result=assets.assemble(fixture.parents,fixture.reference,'extended');self.assertEqual(len(result['raw_files']),2103)
    data['profile']='corpus'
    with self.assertRaisesRegex(ValueError,'extended reference profile'):assets.assemble(fixture.parents,fixture.reference,'extended')
    data['profile']='extended';fixture.write('reference/raw/extra',b'x')
    with self.assertRaisesRegex(ValueError,'extended raw contract'):assets.assemble(fixture.parents,fixture.reference,'extended')
  finally:fixture.doCleanups()
if __name__=='__main__':unittest.main()
