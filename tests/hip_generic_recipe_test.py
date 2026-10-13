#!/usr/bin/env python3
"""Strict saved generic proof checks and independent transcript mutations."""
import argparse, hashlib, json, pathlib, sys, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from probe_hip_generic import validate_model, check_profiles

def check(output):
    receipt=json.loads((output/'receipt.json').read_text())
    assert receipt['classification']=='bounded-generic-runtime-offline-pass'
    for name,digest in receipt['sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for name,digest in receipt['inputs'].items():assert hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest()==digest,name
    for name,digest in receipt['artifacts'].items():assert hashlib.sha256((output/name).read_bytes()).hexdigest()==digest,name
    assert len(receipt['cases'])==6
    for case in receipt['cases']:check_profiles(case,case['kernels'])
    assert len(receipt['models'])==24
    for model in receipt['models']:
        stem=model['variant']+'-'+model['mode'];text=(output/(stem+'.stdout')).read_text()+(output/(stem+'.stderr')).read_text()
        validate_model(text,model['mode'],model['exit_code'],3 if model['variant']=='model-extra' else 2)
    assert receipt['second_module_registration']['register_module_calls']==1
    if receipt['native']:assert receipt['native']['HIP_imports']==0 and receipt['native']['native_executed'] is False
    print('Generic saved proof: six combined objects, 24 model modes, exact artifacts')

class TranscriptTests(unittest.TestCase):
    def good(self,count=3):
        return 'VECTOR_FUNCTION bound=true\n'*2+f'GENERIC_CLIENT_PASS launches={count} variable_sizes=true full_guards=true unused_function_allowed=true\nGENERIC_MODEL_FINAL live=0 backends={int(count>0)} submits={count} releases=18 finalized=1\n'
    def test_variable_success(self):
        for mode,n in [('zero',0),('one',1),('three',3),('seven',7)]:validate_model(self.good(n),mode,0,2)
    def test_missing_or_false_closure(self):
        text=self.good()
        for altered in [text.replace('live=0','live=1'),text.replace('finalized=1','finalized=0'),text.replace('submits=3','submits=4'),text.replace('launches=3','launches=7'),text+'AddressSanitizer',text+text,text.replace('GENERIC_CLIENT_PASS','WRONG')]:
            with self.assertRaises(RuntimeError):validate_model(altered,'three',0,2)
    def test_terminal_classification(self):
        for mode in ('submit','kick','timeout','throw','unknown','cleanup'):
            text=f'GENERIC_TERMINAL live={3 if mode=="cleanup" else 6} unresolved={0 if mode=="cleanup" else 1} submits=1\n'
            validate_model(text,mode,30,2)
            for bad in [text.replace('live=','live=9'),text+'GENERIC_CLIENT_PASS',text+'GENERIC_MODEL_FINAL']:
                with self.assertRaises(RuntimeError):validate_model(bad,mode,30,2)
    def test_clean_failure_not_application_pass(self):
        text='VECTOR_FUNCTION bound=true\n'*2+'GENERIC_CLIENT_FAILURE clean=1 terminal=0\nGENERIC_MODEL_FINAL live=0 backends=1 submits=1 releases=6 finalized=1\n'
        validate_model(text,'corrupt',20,2)
        with self.assertRaises(RuntimeError):validate_model(text+'GENERIC_CLIENT_PASS','corrupt',20,2)
    def test_partial_never_finalized(self):
        text='GENERIC_PARTIAL no_gpu=true\nGENERIC_MODEL_FINAL live=0 backends=0 submits=0 releases=0 finalized=0\n'
        for mode in ('partial','second-module'):
            validate_model(text,mode,0,2)
            with self.assertRaises(RuntimeError):validate_model(text.replace('finalized=0','finalized=1'),mode,0,2)

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--check':check(pathlib.Path(sys.argv[2]).resolve())
    else:unittest.main()
