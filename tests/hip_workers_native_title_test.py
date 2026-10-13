#!/usr/bin/env python3
"""Worker profile boundaries and actual Linux-linked forwarding fixture; no GPU."""
import argparse,copy,hashlib,json,pathlib,sys,tempfile,subprocess,unittest
from unittest import mock
ROOT=pathlib.Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'tests'))
import native_vector_title as profile
import build_native_vector_title as builder
p=argparse.ArgumentParser(add_help=False);p.add_argument('--compiler',type=pathlib.Path);a,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
sha=lambda b:hashlib.sha256(b).hexdigest()
class WorkersProfile(unittest.TestCase):
    def compiler(self):
        if not a.compiler:self.skipTest('actual compiler directory not supplied')
        return (a.compiler/'receipt.json').read_bytes(),(a.compiler/'ps5/threads.o').read_bytes()
    def test_profile_scope(self):
        p=profile.profile_for('hip_workers_main');self.assertEqual(p['actors'],[0,1,2,1,0]);self.assertEqual(p['kernarg_bytes'],272);self.assertEqual(p['initialization_actor'],'main')
        self.assertEqual(builder.compiler_object(pathlib.Path('x'),'hip_workers_main'),pathlib.Path('x/ps5/threads.o'))
        self.assertIn('workers-agc-disassembly.stdout',profile.provenance_files('hip_workers_main'))
        with self.assertRaises(ValueError):profile.profile_for('hip_workers_init')
    def test_actual_compiler(self):
        raw,obj=self.compiler();profile.validate_workers_compiler(raw,obj,ROOT)
        for r,o in ((raw+b' ',obj),(raw,obj+b' ')):
            with self.assertRaises(ValueError):profile.validate_workers_compiler(r,o,ROOT)
    def test_semantic_negatives(self):
        raw,obj=self.compiler();base=json.loads(raw)
        for mode in ('scope','source','descriptor','wrapper','registration','artifact','unchanged'):
            r=copy.deepcopy(base);c=next(c for c in r['cases'] if c['host']=='ps5')
            if mode=='scope':r['native_linked']=True
            elif mode=='source':r['sources']['tests/hip_threads.hip']='0'*64
            elif mode=='descriptor':c['audit']['descriptors']['thread_words']['private_bytes']=4
            elif mode=='wrapper':c['wrapper']['payload_bytes']=32
            elif mode=='registration':c['lifecycle']['register_function_calls']=2
            elif mode=='unchanged':r['kernel_source_unchanged']=False
            else:r['artifacts']['ps5/threads.bundle']='0'*64
            changed=json.dumps(r).encode()
            with mock.patch.dict(profile.WORKERS_PROFILE,compiler_receipt_sha256=sha(changed)):
                with self.assertRaises(ValueError):profile.validate_workers_compiler(changed,obj,ROOT)
    def test_actual_host_link_redirection(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=pathlib.Path(tmp)
            (t/'imports.cpp').write_text('extern "C" {int sceAgcInit(unsigned){return 0;} int sceAgcDriverSubmitDcb(void*){return 0;} int sceAgcSuspendPoint(){return 0;}}')
            (t/'fixture.cpp').write_text('''extern "C" {int sceAgcInit(unsigned);int sceAgcDriverSubmitDcb(void*);int sceAgcSuspendPoint();
int init() asm("_ZN6ps5hip10AgcBackendC2Ev");int submit() asm("_ZN6ps5hip10AgcBackend6submitEPjj");int kick() asm("_ZN6ps5hip10AgcBackend4kickEv");
int init(){return sceAgcInit(8);}int submit(){return sceAgcDriverSubmitDcb(nullptr);}int kick(){return sceAgcSuspendPoint();}}
namespace ps5hip_workers {unsigned actor() noexcept{return 0;}}int main(){return init()+submit()+kick();}''')
            # Actual wrapper implementation and SDK signatures, with harmless Linux providers.
            sdk=ROOT.parent.parent/'.deps/sdk-source-95c08f27386fc698f6bbe21dde3030140a41d10b/platform/include'
            def run(*cmd):return subprocess.check_output(list(map(str,cmd)),stderr=subprocess.STDOUT,text=True)
            run('clang++-18','-shared','-fPIC',t/'imports.cpp','-o',t/'libmock.so')
            run('clang++-18','-std=c++17','-O0','-fPIE','-pie','-I',sdk,'-I',ROOT/'tools',t/'fixture.cpp',ROOT/'tools/hip_workers_agc.cpp','-Wl,--wrap=sceAgcInit,--wrap=sceAgcDriverSubmitDcb,--wrap=sceAgcSuspendPoint',t/'libmock.so','-o',t/'fixture')
            dis=run('llvm-objdump-18','-d','--no-show-raw-insn',t/'fixture');result=profile.audit_workers_agc(dis,t/'fixture');self.assertEqual(len(result['calls']),3)
            for name in ('sceAgcInit','sceAgcDriverSubmitDcb','sceAgcSuspendPoint'):
                # Removing the actual redirection from the disassembly must fail.
                with self.assertRaises(ValueError):profile.audit_workers_agc(dis.replace('<__wrap_'+name+'>','<not_wrapped>'),t/'fixture')
if __name__=='__main__':unittest.main()
