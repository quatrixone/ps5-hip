"""Compile the genuine StatCO digest body with fault-injected extraction."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_public_hip_static_module import prepare

PREAMBLE = r'''
#include <cassert>
#include <memory>
#include <map>
#include <new>
using hipError_t=int;
constexpr int hipSuccess=0, hipErrorOutOfMemory=2;
#define IHIP_RETURN_ONFAIL(expr) do { auto e=(expr); if(e!=hipSuccess)return e; } while(0)
namespace amd { struct ScopedLock { explicit ScopedLock(int&) {} }; }
int g_devices=1, live=0, attempts=0, result=0, fault=0;
struct FatBinaryInfo {
 FatBinaryInfo(const char*,const void*) { if(fault==1)throw std::bad_alloc(); ++live; }
 ~FatBinaryInfo(){ --live; }
 hipError_t ExtractFatBinary(int) { ++attempts; if(fault==2)throw std::bad_alloc(); return result; }
};
#define HIP_ERROR_PRINT(...) ((void)0)
struct StatCO { int sclock_=0; std::map<const void*,FatBinaryInfo*> modules_; hipError_t digestFatBinary(const void*,FatBinaryInfo*&); FatBinaryInfo** addFatBinary(const void*,bool); };
'''
DRIVER = r'''
int main(){
 StatCO co; FatBinaryInfo* p=nullptr;
 result=7;
 assert(co.digestFatBinary(nullptr,p)==7);
 // This retry incorrectly returns success in upstream, caching the failed object.
 const int retry=co.digestFatBinary(nullptr,p);
 if(retry!=7){ delete p; return 23; }
 assert(!p && live==0 && attempts==2);
 for(int error: {3,4,8}) { result=error; assert(co.digestFatBinary(nullptr,p)==error); assert(!p&&live==0); }
 for(int f: {1,2}) { fault=f; assert(co.digestFatBinary(nullptr,p)==hipErrorOutOfMemory); assert(!p&&live==0); }
 fault=0; result=0; assert(co.digestFatBinary(nullptr,p)==0 && p && live==1);
 auto* old=p; int count=attempts; result=7; fault=1;
 assert(co.digestFatBinary(nullptr,p)==0 && p==old && attempts==count && live==1);
 delete p; assert(live==0);
 fault=0; result=7;
 auto** late=co.addFatBinary(nullptr,true); assert(late && !*late && live==0);
 result=0; assert(co.digestFatBinary(nullptr,*late)==0 && *late && live==1);
 delete *late; assert(live==0);
}
'''

class StaticModuleTest(unittest.TestCase):
    def test_transaction_and_original_false_success(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)
            target=source/'hipamd/src/hip_code_object.cpp'
            target.parent.mkdir(parents=True)
            target.write_bytes((ROOT/'.deps/sources/clr/hipamd/src/hip_code_object.cpp').read_bytes())
            patch=(ROOT/'ports/clr/ps5-public-hip.patch').read_text()
            marker='--- a/hipamd/src/hip_code_object.cpp\n'
            section=marker+patch.split(marker)[1].split('--- a/',1)[0]
            subprocess.run(['patch','--batch','--fuzz=0','-p1'],input=section,cwd=source,
                           text=True,check=True,capture_output=True)
            global_file=source/'hipamd/src/hip_global.cpp'
            global_file.write_bytes((ROOT/'.deps/sources/clr/hipamd/src/hip_global.cpp').read_bytes())
            original=target.read_text()
            prepare(source); prepare(source)
            fixed=target.read_text()
            for label,text,expected in [('original',original,23),('fixed',fixed,0)]:
                body=text.split('hipError_t StatCO::digestFatBinary(')[1].split('\n}\n',1)[0]
                unit=source/(label+'.cpp'); binary=source/label
                late=text.split('FatBinaryInfo** StatCO::addFatBinary(')[1].split('\n}\n',1)[0]
                unit.write_text(PREAMBLE+'hipError_t StatCO::digestFatBinary('+body+'\n}\n'+'FatBinaryInfo** StatCO::addFatBinary('+late+'\n}\n'+DRIVER)
                subprocess.run(['clang++-18','-std=c++20','-DWITH_PS5_DEVICE','-O1','-g',
                                '-fsanitize=address,undefined',unit,'-o',binary],check=True,capture_output=True)
                run=subprocess.run([binary],capture_output=True,env=dict(os.environ,
                                   ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1'))
                self.assertEqual(run.returncode,expected,run.stderr.decode())
            consumer_preamble = r"""
#include <cassert>
#include <vector>
using hipError_t=int; using hipModule_t=void*; using hipFunction_t=void*;
constexpr int hipSuccess=0,hipErrorNoBinaryForGpu=3,hipErrorInvalidDeviceFunction=4,hipErrorInvalidSymbol=5;
#define guarantee(x,...) assert(x)
#define IHIP_RETURN_ONFAIL(expr) do {auto e=(expr);if(e)return e;}while(0)
std::vector<int> g_devices(1); int calls=0;
struct FatBinaryInfo {int BuildProgram(int){++calls;return 0;} int GetModule(int,void**p){*p=this;return 0;}};
struct DeviceFunc {DeviceFunc(int,void*){} void* asHipFunction(){return this;}};
struct DeviceVar {DeviceVar(int,void*,int){}};
struct Function {FatBinaryInfo** modules_=nullptr;std::vector<DeviceFunc*> dFunc_{nullptr};int name_=0;int getStatFunc(void**,int);};
struct Var {FatBinaryInfo** modules_=nullptr;std::vector<DeviceVar*> dVar_{nullptr};int name_=0;int getStatDeviceVar(DeviceVar**,int);};
"""
            consumers=global_file.read_text()
            bodies=''
            for signature in ('hipError_t Function::getStatFunc(', 'hipError_t Var::getStatDeviceVar('):
                bodies+=signature+consumers.split(signature)[1].split('\n}\n',1)[0]+'\n}\n'
            driver=r"""
int main(){Function f;Var v;void* out=nullptr;DeviceVar* var=nullptr;
assert(f.getStatFunc(&out,0)==hipErrorInvalidDeviceFunction);
assert(v.getStatDeviceVar(&var,0)==hipErrorInvalidSymbol);
FatBinaryInfo* module=nullptr;f.modules_=&module;v.modules_=&module;
assert(f.getStatFunc(&out,0)==hipErrorInvalidDeviceFunction);
assert(v.getStatDeviceVar(&var,0)==hipErrorInvalidSymbol);
assert(!out&&!var&&calls==0);
FatBinaryInfo good;module=&good;
assert(f.getStatFunc(&out,0)==0&&out);assert(v.getStatDeviceVar(&var,0)==0&&var);assert(calls==2);
delete f.dFunc_[0];delete v.dVar_[0];}
"""
            unit=source/'consumers.cpp';binary=source/'consumers'
            unit.write_text(consumer_preamble+bodies+driver)
            subprocess.run(['clang++-18','-std=c++20','-DWITH_PS5_DEVICE','-O1','-g',
                            '-fsanitize=address,undefined',unit,'-o',binary],check=True,capture_output=True)
            subprocess.run([binary],check=True,capture_output=True,env=dict(os.environ,
                           ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1'))
            target.write_text(fixed+'\n// drift\n')
            with self.assertRaises(ValueError): prepare(source)

if __name__=='__main__': unittest.main()
