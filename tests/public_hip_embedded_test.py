"""Fault-inject the genuine prepared embedded-image and static lifecycle bodies."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from prepare_public_hip_aot import prepare as prepare_aot
from prepare_public_hip_static_module import prepare as prepare_static
from prepare_public_hip_embedded import prepare
import public_hip_aot_test as aot

def source_tree(folder):
    names=['hip_fatbin.cpp','hip_fatbin.hpp','hip_code_object.cpp','hip_code_object.hpp','hip_platform.cpp','hip_platform.hpp','hip_global.cpp']
    target=folder/'hipamd/src';target.mkdir(parents=True)
    for name in names:
        (target/name).write_bytes((ROOT/'.deps/sources/clr/hipamd/src'/name).read_bytes())
    patch=(ROOT/'ports/clr/ps5-public-hip.patch').read_text()
    for name in ['hip_code_object.cpp','hip_fatbin.cpp']:
        marker='--- a/hipamd/src/'+name+'\n'
        section=marker+patch.split(marker)[1].split('--- a/',1)[0]
        subprocess.run(['patch','--batch','--fuzz=0','-p1'],input=section,cwd=folder,text=True,check=True,capture_output=True)
    prepare_aot(folder);prepare_static(folder);prepare(folder);prepare(folder)
    return target

def body(text,signature):
    return signature+text.split(signature)[1].split('\n}\n',1)[0]+'\n}\n'

class EmbeddedTest(unittest.TestCase):
    def compile_run(self,folder,label,text,extra=()):
        unit=folder/(label+'.cpp');binary=folder/label;unit.write_text(text)
        p=subprocess.run(['clang++-18','-std=c++20','-DWITH_PS5_DEVICE','-O1','-g',
            '-fsanitize=address,undefined','-I'+str(ROOT/'include'),'-I'+str(ROOT/'ports/clr'),unit,*extra,'-o',binary],capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        p=subprocess.run([binary],capture_output=True,text=True,env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1'))
        self.assertEqual(p.returncode,0,p.stderr)

    def test_prepared_extraction_and_static_lifecycle(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);source=source_tree(folder)
            text=(source/'hip_fatbin.cpp').read_text()
            extract=body(text,'hipError_t FatBinaryInfo::ExtractFatBinary(')
            for original,replacement in {'fopen':'open','fclose':'close','fseek':'seek','ftell':'tell','fread':'read','fgetc':'getc','ferror':'error'}.items():
                extract=extract.replace('std::'+original,'fault_'+replacement)
            preamble=aot.PREAMBLE.replace(' bool ps5_extract_attempted_=false;',' bool ps5_static_image_=false;\n bool ps5_extract_attempted_=false;')
            preamble += r'''
namespace ps5hip {
struct BundledCodeObject {std::vector<unsigned char> bytes;};
int reads=0;
BundledCodeObject read_linked_host_module(const void* p){++reads;if(p!=reinterpret_cast<void*>(123))throw ObjectError{};return {std::vector<unsigned char>(128)};}
}
'''
            driver=aot.DRIVER.replace(' assert(opened==closed&&!infos&&!programs);',r'''
 {FatBinaryInfo f;f.fname_.clear();f.ps5_static_image_=true;f.image_=reinterpret_cast<void*>(1);
  assert(f.ExtractFatBinary(devices)==hipErrorInvalidImage);assert(!f.fatbin_dev_info_[0]);}
 int open_before=opened;
 {FatBinaryInfo f;f.fname_.clear();f.ps5_static_image_=true;f.image_=reinterpret_cast<void*>(123);
  assert(f.ExtractFatBinary(devices)==0);assert(f.image_==f.ps5_image_storage_.data());assert(f.fatbin_dev_info_[0]->image==f.image_);}
 assert(opened==open_before&&ps5hip::reads==2);
 assert(opened==closed&&!infos&&!programs);
''')
            self.compile_run(folder,'extract',preamble+extract+driver)
            preamble=r'''
#include <cassert>
#include <memory>
#include <new>
#include <unordered_map>
#include <vector>
using hipError_t=int;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipErrorOutOfMemory=2;
#define IHIP_RETURN_ONFAIL(expr) do {auto e=(expr);if(e)return e;}while(0)
#define HIP_ERROR_PRINT(...) ((void)0)
#define DevLogPrintfError(...) ((void)0)
namespace amd {struct ScopedLock {explicit ScopedLock(int&) {}};}
int live=0,extracts=0,result=0;
struct Device {int deviceId(){return 0;}};std::vector<Device*> g_devices;
struct FatBinaryInfo {
 bool is_static=false;
 FatBinaryInfo(const char*,const void*){++live;}~FatBinaryInfo(){--live;}
 void setPs5StaticImage(){is_static=true;}
 int ExtractFatBinary(const std::vector<Device*>&){assert(is_static);++extracts;return result;}
};
struct DeviceVar {void* device_ptr(){return nullptr;}};
struct Var {FatBinaryInfo** moduleInfo(){return nullptr;}int getStatDeviceVar(DeviceVar**,int){return 0;}};
int ihipFree(void*){return 0;}
struct Function {FatBinaryInfo** module;explicit Function(FatBinaryInfo** p):module(p){} FatBinaryInfo** moduleInfo(){return module;}};
struct StatCO {
 int sclock_=0;
 std::unordered_map<const void*,FatBinaryInfo*> modules_;
 FatBinaryInfo** ps5_retired_[32]{};size_t ps5_retired_count_=0;
 std::unordered_map<const void*,Var*> vars_;
 std::vector<Var*> managedVars_;
 std::unordered_map<const void*,Function*> functions_;
 int digestFatBinary(const void*,FatBinaryInfo*&);FatBinaryInfo** addFatBinary(const void*,bool);
 bool ps5RegisteredModule(FatBinaryInfo**);int removeFatBinary(FatBinaryInfo**);
 int registerStatFunction(const void*,Function*);
};
'''
            text=(source/'hip_code_object.cpp').read_text();functions=''
            for sig in ['hipError_t StatCO::digestFatBinary(','bool StatCO::ps5RegisteredModule(','FatBinaryInfo** StatCO::addFatBinary(','hipError_t StatCO::removeFatBinary(','hipError_t StatCO::registerStatFunction(']:
                functions+=body(text,sig)
            driver=r'''
int main(){
 StatCO c;int a=0,b=0,token=0;
 assert(!c.ps5RegisteredModule(nullptr));assert(c.removeFatBinary(reinterpret_cast<FatBinaryInfo**>(1))==hipErrorInvalidValue);
 auto** early=c.addFatBinary(&a,false);assert(early&&!*early&&c.ps5RegisteredModule(early));
 assert(!c.addFatBinary(&a,false));
 auto* fn=new Function(early);assert(c.registerStatFunction(&token,fn)==0);
 auto* duplicate=new Function(early);assert(c.registerStatFunction(&token,duplicate)==hipErrorInvalidValue);delete duplicate;
 result=7;assert(c.digestFatBinary(&a,*early)==7&&!*early&&live==0);
 assert(c.digestFatBinary(&a,*early)==7&&!*early&&live==0);
 result=0;assert(c.digestFatBinary(&a,*early)==0&&*early&&live==1);
 assert(c.removeFatBinary(early)==0&&!*early&&live==0&&c.functions_.empty());
 assert(!c.ps5RegisteredModule(early)&&c.removeFatBinary(early)==hipErrorInvalidValue);
 assert(!c.addFatBinary(&a,false));
 auto* stale=new Function(early);assert(c.registerStatFunction(&token,stale)==hipErrorInvalidValue);delete stale;
 result=7;auto** late=c.addFatBinary(&b,true);assert(late&&late!=early&&!*late&&live==0);
 assert(c.removeFatBinary(late)==0);
 int keys[32]{};
 for(int i=0;i<30;++i){auto** h=c.addFatBinary(keys+i,false);assert(h);assert(c.removeFatBinary(h)==0);}
 assert(!c.addFatBinary(keys+30,false)&&c.ps5_retired_count_==32);
}
'''
            self.compile_run(folder,'lifecycle',preamble+functions+driver)

            preamble=r"""
#include <cassert>
#include <string>
#include <vector>
using hipError_t=int;using hipModule_t=void*;using hipFunction_t=void*;using cl_program=void*;
constexpr int hipSuccess=0,hipErrorNoBinaryForGpu=3,hipErrorInvalidDeviceFunction=4,CL_DEVICE_TYPE_GPU=1;
#define guarantee(x,...) assert(x)
#define IHIP_RETURN_ONFAIL(expr) do {auto e=(expr);if(e)return e;}while(0)
namespace device {struct Kernel {struct WorkGroupInfo {int localMemSize_=0,privateMemSize_=0,availableLDSSize_=128,size_=64,usedVGPRs_=8;} info;WorkGroupInfo* workGroupInfo(){return &info;}};}
namespace amd {
struct Device {static const std::vector<Device*>& getDevices(int,bool){static Device d;static std::vector<Device*> ds{&d};return ds;}};
struct Program {const void* findSymbol(const char* name){return std::string(name)=="present"?this:nullptr;}};
struct Kernel {device::Kernel k;device::Kernel* getDeviceKernel(Device&){return &k;}struct Sig {int version(){return 5;}};Sig signature(){return {};}};
}
amd::Program program;amd::Program* as_amd(cl_program p){return static_cast<amd::Program*>(p);}
std::vector<int> g_devices(1);int created=0;
struct FatBinaryInfo {int BuildProgram(int){return 0;}int GetModule(int,void**p){*p=&program;return 0;}};
struct DeviceFunc {amd::Kernel k;DeviceFunc(const std::string& name,void* p){assert(as_amd(p)->findSymbol(name.c_str()));++created;}void* asHipFunction(){return this;}amd::Kernel* kernel(){return &k;}};
struct hipFuncAttributes {int sharedSizeBytes=-1,binaryVersion=-1,cacheModeCA=-1,constSizeBytes=-1,localSizeBytes=-1,maxDynamicSharedSizeBytes=-1,maxThreadsPerBlock=-1,numRegs=-1,preferredShmemCarveout=-1,ptxVersion=-1;};
struct Function {FatBinaryInfo** modules_;std::vector<DeviceFunc*> dFunc_{nullptr};std::string name_;int getStatFunc(void**,int);int getStatFuncAttr(hipFuncAttributes*,int);~Function(){delete dFunc_[0];}};
"""
            text=(source/'hip_global.cpp').read_text()
            methods=body(text,'hipError_t Function::getStatFunc(')+body(text,'hipError_t Function::getStatFuncAttr(')
            driver=r"""
int main(){FatBinaryInfo module;auto* ptr=&module;void* out=nullptr;hipFuncAttributes attr;
 {Function f{&ptr,{nullptr},"absent"};assert(f.getStatFunc(&out,0)==hipErrorInvalidDeviceFunction);assert(!out&&!created);assert(f.getStatFuncAttr(&attr,0)==hipErrorInvalidDeviceFunction);assert(attr.binaryVersion==-1&&!created);}
 {Function f{&ptr,{nullptr},"present"};assert(f.getStatFunc(&out,0)==0&&out&&created==1);assert(f.getStatFuncAttr(&attr,0)==0&&attr.binaryVersion==5&&created==1);}
}
"""
            self.compile_run(folder,'static-symbol-lookup',preamble+methods+driver)
            original=(source/'hip_platform.cpp').read_text()
            # Compile the actual unregister wrapper with a failing synchronization provider.
            preamble=r'''
#include <cassert>
using hipError_t=int;constexpr int hipSuccess=0,hipErrorInvalidValue=1;
#define HIP_ERROR_PRINT(...) ((void)0)
#define LogPrintfError(...) ((void)0)
#define guarantee(v,...) assert(v)
namespace hip {struct FatBinaryInfo {};}
int syncs=0,removes=0,status=0;hip::FatBinaryInfo* slot=nullptr;
int hipDeviceSynchronize(){++syncs;return status;}
struct PlatformState {static PlatformState& instance(){static PlatformState p;return p;}bool ps5RegisteredModule(hip::FatBinaryInfo** p){return p==&slot;}int removeFatBinary(hip::FatBinaryInfo**){++removes;return 0;}};
'''
            driver=r'''
int main(){__hipUnregisterFatBinary(nullptr);assert(!syncs&&!removes);status=7;__hipUnregisterFatBinary(&slot);assert(syncs==1&&!removes);status=0;__hipUnregisterFatBinary(&slot);assert(syncs==2&&removes==1);}
'''
            self.compile_run(folder,'unregister',preamble+body(original,'void __hipUnregisterFatBinary(hip::FatBinaryInfo** modules) {').replace('(hip::FatBinaryInfo** modules) {','(hip::FatBinaryInfo** modules) {',1)+driver)

            preamble=r"""
#include <cassert>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <vector>
using hipError_t=int;constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipErrorOutOfMemory=2,hipErrorInvalidImage=3;
using hipFunction_t=void*;struct uint3{};struct dim3{};
#define HIP_ERROR_PRINT(...) ((void)0)
#define HIP_INIT_VOID() ((void)0)
#define guarantee(v,...) assert(v)
std::vector<int> g_devices{0};int names=0,live=0,gets=0,registered=0,parse=0;
namespace ps5hip {struct ObjectError{};int read_linked_host_module(const void* p){++parse;if(p!=reinterpret_cast<void*>(123))throw ObjectError{};return 0;}std::string linked_host_name(const char* p){++names;if(!p)throw ObjectError{};return "kernel";}}
namespace hip {struct FatBinaryInfo{};struct Function{Function(const std::string&,FatBinaryInfo**){++live;}~Function(){--live;}};}
hip::FatBinaryInfo* slot=nullptr;hip::Function* retained=nullptr;
struct PlatformState {
 static PlatformState& instance(){static PlatformState p;return p;}
 bool ps5RegisteredModule(hip::FatBinaryInfo** p){return p==&slot;}
 hip::FatBinaryInfo** addFatBinary(const void*){return &slot;}
 int registerStatFunction(const void*,hip::Function* f){if(retained)return 1;retained=f;++registered;return 0;}
 int getStatFunc(void**,const void*,int){++gets;return 1;}
};
"""
            driver=r"""
int main(){
 setenv("HIP_ENABLE_DEFERRED_LOADING","0",1);int token;
 assert(!__hipRegisterFatBinary(reinterpret_cast<void*>(1))&&parse==1);
 assert(__hipRegisterFatBinary(reinterpret_cast<void*>(123))==reinterpret_cast<void**>(&slot));
 auto reg=[&](hip::FatBinaryInfo** h,const char* name){__hipRegisterFunction(h,&token,nullptr,name,0,nullptr,nullptr,nullptr,nullptr,nullptr);};
 reg(nullptr,"kernel");assert(!live&&!names&&!gets);
 reg(&slot,nullptr);assert(!live&&names==1&&!gets);
 reg(&slot,"kernel");assert(live==1&&registered==1&&gets==1);
 reg(&slot,"kernel");assert(live==1&&registered==1&&gets==1);
 delete retained;assert(!live);
}
"""
            registration=body(original,'void** __hipRegisterFatBinary(')+body(original,'void __hipRegisterFunction(hip::FatBinaryInfo** modules,')
            self.compile_run(folder,'registration',preamble+registration+driver)
            (source/'hip_platform.cpp').write_text(original+'\n// drift\n')
            with self.assertRaises(ValueError):prepare(folder)

    def test_missing_linker_extents_reject_without_read(self):
        with tempfile.TemporaryDirectory() as d:
            text=r'''
#include "ps5-embedded.h"
#include <cassert>
int main(){for(auto* p:{static_cast<const void*>(nullptr),reinterpret_cast<const void*>(1)}){bool rejected=false;try{ps5hip::read_linked_host_module(p);}catch(const ps5hip::ObjectError&){rejected=true;}assert(rejected);}bool rejected=false;try{ps5hip::linked_host_name(reinterpret_cast<const char*>(1));}catch(const ps5hip::ObjectError&){rejected=true;}assert(rejected);}
'''
            self.compile_run(Path(d),'missing',text,[ROOT/'ports/clr/ps5-embedded.cpp',ROOT/'src/host_registration.cpp',ROOT/'src/fat_binary.cpp',ROOT/'src/code_object.cpp'])

if __name__=='__main__':unittest.main()
