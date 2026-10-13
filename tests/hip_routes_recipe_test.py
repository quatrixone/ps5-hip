"""Offline transcript/provenance boundaries, never GPU arithmetic validation."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
UNCERTAIN={'scratch-submit','scratch-kick','scratch-timeout','scratch-throw','scratch-unknown','scratch-postflush'}
REJECT={'init','missing','corrupt','input','guard','stale-second-round','wrong-route-output','scratch-map'}
MODES={'normal','preflush','partial','late-partial','scratch-cleanup'}|UNCERTAIN|REJECT

def require(ok,message):
    if not ok:raise ValueError(message)

def validate_model(log,mode,exit_code):
    require(mode in MODES,'unknown model mode')
    require(not re.search(r'AddressSanitizer|UndefinedBehaviorSanitizer|runtime error:|LeakSanitizer',log),'sanitizer diagnostic')
    completions=re.findall(r'^ROUTES_COMPLETE round=(\d) route=(\d) blocking=true full_guards=true inputs=unchanged$',log,re.M)
    expected=[(str(round),str(route)) for round in (1,2) for route in range(4)]
    require(completions==expected[:len(completions)],'route completion order/duplicates')
    if mode in UNCERTAIN|{'scratch-cleanup'}:
        require(exit_code==30 and len(completions)==2,'terminal exit boundary')
        matches=re.findall(r'^ROUTES_CLIENT_TERMINAL_PASS mode=(\S+) unresolved=(\d) live=(\d+) submits=(\d+) releases=(\d+) ownership_retained=true$',log,re.M)
        require(len(matches)==1,'missing/duplicate terminal ownership observation')
        name,unresolved,live,submits,releases=matches[0]
        require(name==mode and int(unresolved)==(mode!='scratch-cleanup') and int(live)==(4 if mode=='scratch-cleanup' else 8) and int(submits)==3,'wrong terminal retained ownership')
        require('ROUTES_CLIENT_PASS' not in log and 'ROUTES_MODEL_FINAL' not in log,'terminal path performed normal cleanup')
        return
    finals=re.findall(r'^ROUTES_MODEL_FINAL allocations=(\d+) backends=(\d+) submits=(\d+) releases=(\d+) finalized=(\d) model_only=true$',log,re.M)
    require(len(finals)==1,'missing/duplicate final observation')
    allocations,backends,submits,releases,finalized=map(int,finals[0])
    require(allocations==0,'model leaked resolved allocation')
    if mode in ('normal','preflush'):
        require(exit_code==0 and completions==expected and (backends,submits,finalized)==(1,8,1),'normal completion/finalization')
        require(log.count('ROUTES_CLIENT_PASS sequences=2 dispatches=8 changed_inputs=true full_guards=true hardware_validated=false')==1,'normal client success proof')
    elif mode=='late-partial':
        require(exit_code==0 and completions==expected and (backends,submits,finalized)==(1,8,0),'late registration failure incorrectly finalized')
        require(log.count('ROUTES_CLIENT_LATE_PARTIAL_PASS completed=8 stale_tokens=true finalized_required=false')==1,'late partial latch proof')
        require('ROUTES_CLIENT_PASS' not in log,'late partial falsely passed')
    elif mode=='partial':
        require(exit_code==0 and not completions and (backends,submits,finalized)==(0,0,0),'partial constructor did native work')
        require(log.count('ROUTES_CLIENT_PARTIAL_PASS no_gpu=true')==1,'partial latch proof')
    else:
        require(mode in REJECT and exit_code==20 and finalized==0 and 'ROUTES_CLIENT_REJECT' in log and 'ROUTES_CLIENT_PASS' not in log,'clean failure falsely passed')
        rejected=re.findall(r'^ROUTES_CLIENT_REJECT mode=(\S+) reason=[^\n]* terminal=(\d)$',log,re.M)
        require(rejected==[(mode,'0')],'wrong clean failure mode/lifetime')
        expected_submit,expected_complete={'init':(0,0),'scratch-map':(2,2),'stale-second-round':(5,4),'wrong-route-output':(2,1)}.get(mode,(1,0))
        require(backends==1 and submits==expected_submit and len(completions)==expected_complete,'wrong clean failure injection boundary')

def check_output(output):
    receipt=json.loads((output/'receipt.json').read_text())
    require(receipt['classification']=='genuine-single-TU-four-route-offline-proof-pass' and receipt['hardware_validated'] is False,'wrong proof classification')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for group in ('source_sha256','runtime_sources'):
        for name,digest in receipt.get(group,{}).items():require(sha(ROOT/name)==digest,'source changed: '+name)
    for name,digest in receipt['inputs'].items():require(sha(Path(name))==digest,'compiler/native input changed: '+name)
    for tool in receipt['tools'].values():require(sha(Path(tool['path']))==tool['sha256'],'compiler tool changed')
    for name,digest in receipt['artifacts'].items():
        relative=Path(name);require(not relative.is_absolute() and '..' not in relative.parts,'artifact escapes output')
        require(sha(output/relative)==digest,'proof artifact changed: '+name)
    require({case['host'] for case in receipt['cases']}=={'linux','ps5','linux-model'},'compiler host inventory')
    require({case['mode'] for case in receipt['models']}==MODES,'model inventory incomplete')
    for item in receipt['models']:
        mode=item['mode'];validate_model((output/('model-'+mode+'.stdout')).read_text()+(output/('model-'+mode+'.stderr')).read_text(),mode,item['exit_code'])
    require(receipt['native']['HIP_imports']==0 and receipt['native']['native_executed'] is False,'native boundary missing')
    require(receipt['native']['audits']['raw_pie'] and receipt['native']['audits']['converted_eboot'],'linked lifecycle audits missing')
    return receipt

class Tests(unittest.TestCase):
    def test_wrapper_model_redzone_is_not_production_abi(self):
        sys.path.insert(0,str(ROOT/'tools'))
        from probe_hip_routes import audit_wrapper
        def report(size):
            return f'  Symbol {{\n    Name: __hip_fatbin_wrapper (1)\n    Value: 0x0\n    Size: {size}\n    Section: .hipFatBinSegment (0x1)\n  }}\n  Section (2) .rela.hipFatBinSegment {{\n    0x8 R_X86_64_64 .hip_fatbin 0x0\n  }}'
        for host,size in [('linux',24),('ps5',24),('linux-model',64)]:
            raw=bytes.fromhex('4650494801000000')+bytes(size-8)
            self.assertEqual(audit_wrapper(raw,report(size),host)['payload_bytes'],24)
            for mutated in [raw[:-1],raw[:-1]+b'\x01',b'\x00'+raw[1:]]:
                with self.assertRaises(RuntimeError):audit_wrapper(mutated,report(size),host)
            for old,new in [('0x8 R_X86_64_64','0x10 R_X86_64_64'),('.hip_fatbin 0x0','.hip_fatbin 0x8'),('Value: 0x0','Value: 0x8')]:
                with self.assertRaises(RuntimeError):audit_wrapper(raw,report(size).replace(old,new),host)
        with self.assertRaises(RuntimeError):audit_wrapper(bytes.fromhex('4650494801000000')+bytes(56),report(64),'ps5')

    def test_constructor_relocation_binding(self):
        sys.path.insert(0,str(ROOT/'tools'))
        from probe_hip_routes import audit_registration,NAMES
        ctor='0000 <__hip_module_ctor>:\n R_X86_64_PLT32 __hipRegisterFatBinary-0x4\n'
        for name in NAMES:ctor+=f' R_X86_64_REX_GOTPCRELX {name}-0x4\n R_X86_64_PLT32 __hipRegisterFunction-0x4\n'
        ctor+=' R_X86_64_PLT32 atexit-0x4\n0001 <__hip_module_dtor>:\n R_X86_64_PLT32 __hipUnregisterFatBinary-0x4\n'
        self.assertEqual(audit_registration(ctor)['register_function_calls'],4)
        for old,new in [('routes_spill','unknown_token'),('__hipRegisterFunction','wrong_symbol'),('__hipUnregisterFatBinary','wrong_unregister')]:
            with self.assertRaises(RuntimeError):audit_registration(ctor.replace(old,new))
    def normal(self):
        return ''.join(f'ROUTES_COMPLETE round={round} route={route} blocking=true full_guards=true inputs=unchanged\n' for round in (1,2) for route in range(4))+\
            'ROUTES_CLIENT_PASS sequences=2 dispatches=8 changed_inputs=true full_guards=true hardware_validated=false\nROUTES_MODEL_FINAL allocations=0 backends=1 submits=8 releases=37 finalized=1 model_only=true\n'
    def terminal(self,mode='scratch-timeout'):
        return ''.join(f'ROUTES_COMPLETE round=1 route={route} blocking=true full_guards=true inputs=unchanged\n' for route in range(2))+\
            f'ROUTES_CLIENT_TERMINAL_PASS mode={mode} unresolved={int(mode!="scratch-cleanup")} live={4 if mode=="scratch-cleanup" else 8} submits=3 releases=6 ownership_retained=true\n'
    def test_normal_exact_sequence(self):
        validate_model(self.normal(),'normal',0)
        for old,new in [('round=2 route=0','round=1 route=0'),('finalized=1','finalized=0'),('allocations=0','allocations=1'),('dispatches=8','dispatches=7')]:
            with self.assertRaises(ValueError):validate_model(self.normal().replace(old,new),'normal',0)
    def test_uncertain_retains_and_completed_cleanup_distinct(self):
        validate_model(self.terminal(),'scratch-timeout',30);validate_model(self.terminal('scratch-cleanup'),'scratch-cleanup',30)
        for old,new in [('unresolved=1','unresolved=0'),('live=8','live=3'),('submits=3','submits=4'),('ownership_retained=true','ownership_retained=false')]:
            with self.assertRaises(ValueError):validate_model(self.terminal().replace(old,new),'scratch-timeout',30)
    def test_terminal_cannot_cleanup_or_success(self):
        for extra in ('ROUTES_CLIENT_PASS','ROUTES_MODEL_FINAL allocations=0','AddressSanitizer error'):
            with self.assertRaises(ValueError):validate_model(self.terminal()+extra,'scratch-timeout',30)
    def test_late_partial_cannot_finalize(self):
        log=self.normal().replace('ROUTES_CLIENT_PASS sequences=2 dispatches=8 changed_inputs=true full_guards=true hardware_validated=false','ROUTES_CLIENT_LATE_PARTIAL_PASS completed=8 stale_tokens=true finalized_required=false').replace('finalized=1','finalized=0')
        validate_model(log,'late-partial',0)
        with self.assertRaises(ValueError):validate_model(log.replace('finalized=0','finalized=1'),'late-partial',0)
    def test_unknown_missing_duplicate(self):
        for log,mode,code in [('', 'normal',0),(self.normal(),'unknown',0),(self.normal()+self.normal(),'normal',0),(self.normal(),'normal',20)]:
            with self.assertRaises(ValueError):validate_model(log,mode,code)
    def test_clean_failure_exact_mode_and_prefix(self):
        log='ROUTES_CLIENT_REJECT mode=corrupt reason=routes client assertion terminal=0\nROUTES_MODEL_FINAL allocations=0 backends=1 submits=1 releases=8 finalized=0 model_only=true\n'
        validate_model(log,'corrupt',20)
        for old,new in [('mode=corrupt','mode=missing'),('terminal=0','terminal=1'),('submits=1','submits=0')]:
            with self.assertRaises(ValueError):validate_model(log.replace(old,new),'corrupt',20)
        with self.assertRaises(ValueError):validate_model(log,'stale-second-round',20)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--check',type=Path);args,remaining=parser.parse_known_args()
    if args.check:check_output(args.check);print('Saved four-route compiler/model/native proof verified; no GPU execution.')
    else:unittest.main(argv=[__file__,*remaining])
