#!/usr/bin/env python3
"""Saved actual-object/model proof and rejection tests for the matmul client."""
import copy,hashlib,json,pathlib,sys,unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from probe_hip_matmul import MODES,SHAPES,TERMINAL,check_profile,validate_model,shader
from probe_geometry_lds import oracle_cases

def check(output):
    report=json.loads((output/'receipt.json').read_text())
    assert report['classification']=='genuine-tiled-matmul-runtime-offline-pass'
    assert report['hardware_validated'] is False and report['native_linked'] is False
    for name,digest in report['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in report['inputs'].items():assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==digest,name
    for name,digest in report['artifacts'].items():assert hashlib.sha256((output/name).read_bytes()).hexdigest()==digest,name
    for tool in report['tools'].values():assert hashlib.sha256(pathlib.Path(tool['path']).read_bytes()).hexdigest()==tool['sha256']
    assert [(c['host'],c['optimization']) for c in report['cases']]==[('linux','O1'),('linux','O2'),('ps5','O1'),('ps5','O2'),('linux-model','O2')]
    for case in report['cases']:
        check_profile(case);assert case['lifecycle']==dict(modules=1,functions=1,atexit=1,unregister=1,ordinary_launch_stub=True)
    assert [m['mode'] for m in report['models']]==list(MODES)
    for model in report['models']:
        stem='model-'+model['mode'];validate_model((output/(stem+'.stdout')).read_text()+(output/(stem+'.stderr')).read_text(),model['mode'],model['exit_code'])
    assert json.loads((output/'oracle-cases.json').read_text())==oracle_cases()
    print('Matmul saved proof: five genuine objects, '+str(len(MODES))+' runtime models; no GPU claim')

class Boundaries(unittest.TestCase):
    def good(self):
        text='VECTOR_FUNCTION bound=true\n'
        for i,shape in enumerate(SHAPES):
            for round in range(2):
                text+=f'RUNTIME_COMPLETE sequence={i*2+round+1} kernel=tiled_matmul blocking=true\n'
                text+=f'MATMUL_COMPLETE shape={i} round={round} m={shape[0]} n={shape[1]} k={shape[2]} exact=true guards=true inputs=unchanged\n'
        return text+'MATMUL_CLIENT_PASS cases=4 rounds=2 dispatches=8 values=982 exact_fp32_integer_oracle=true\nVECTOR_UNREGISTER after_main=true active=0\nVECTOR_FINAL registered=1 removed=1 completed=8 active=0 terminal=0 main_finished=1 hardware_validated=false\nMATMUL_MODEL_FINAL live=0 backends=1 submits=8 releases=44 finalized=1\n'
    def test_exact_success_and_mutations(self):
        text=self.good();validate_model(text,'normal',0)
        mutations=[text+text,text.replace('values=982','values=983'),text.replace('dispatches=8','dispatches=7'),text.replace('live=0','live=1'),text.replace('finalized=1','finalized=0'),text.replace('inputs=unchanged','inputs=changed'),text.replace('guards=true','guards=false'),text.replace('shape=1 round=0','shape=1 round=1'),text.replace('m=13 n=21','m=21 n=13'),text.replace('RUNTIME_COMPLETE sequence=8','RUNTIME_COMPLETE sequence=7'),text.replace('VECTOR_UNREGISTER after_main=true active=0\n',''),text+'runtime error:']
        for bad in mutations:
            with self.assertRaises(RuntimeError):validate_model(bad,'normal',0)
    def test_terminal_ownership(self):
        for mode in TERMINAL:
            text='VECTOR_FUNCTION bound=true\n'+f'MATMUL_TERMINAL live={4 if mode=="cleanup" else 7} unresolved={0 if mode=="cleanup" else 1} submits=1\n'
            validate_model(text,mode,30)
            for bad in [text.replace('live=','live=9'),text+'MATMUL_MODEL_FINAL',text+'MATMUL_CLIENT_PASS',text+text]:
                with self.assertRaises(RuntimeError):validate_model(bad,mode,30)
    def test_oracle_shapes_and_range(self):
        cases=oracle_cases();self.assertEqual([(c['m'],c['n'],c['k']) for c in cases],list(SHAPES))
        self.assertEqual(sum(c['logical_outputs']*2 for c in cases),982)
        for c in cases:
            self.assertEqual(c['logical_outputs']+c['inactive_threads'],c['groups'][0]*c['groups'][1]*64)
            self.assertLessEqual(6*c['k'],114)
            self.assertNotEqual(c['rounds'][0]['output_bits'],c['rounds'][1]['output_bits'])
    def test_resource_mutations(self):
        good={'descriptors':{'tiled_matmul':dict(properties=0x409,rsrc2=0x98c,group_bytes=512,kernarg_bytes=296,private_bytes=0,rsrc3=0,preload=0)},'isa':{'tiled_matmul':dict(opcodes=['s_barrier','ds_read_b32','ds_write_b32'],flat_scratch_setup=False)}}
        check_profile(good)
        for key in good['descriptors']['tiled_matmul']:
            bad=copy.deepcopy(good);bad['descriptors']['tiled_matmul'][key]^=1
            with self.assertRaises(RuntimeError):check_profile(bad)
        for op in ('flat_load_dword','scratch_store_dword','s_swappc_b64'):
            bad=copy.deepcopy(good);bad['isa']['tiled_matmul']['opcodes'].append(op)
            with self.assertRaises(RuntimeError):check_profile(bad)
    def test_shader_extraction(self):
        body='extern "C" __global__ __launch_bounds__(64)\nvoid tiled_matmul() {\n}\n'
        self.assertEqual(shader('headers\n'+body+'int main(){}'),body)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--check':check(pathlib.Path(sys.argv[2]).resolve())
    else:unittest.main()
