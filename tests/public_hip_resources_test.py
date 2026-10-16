"""Resource preparation must preserve strict identities through reuse and reversal."""
import difflib
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import prepare_public_hip_resources as resources


class ResourcesTest(unittest.TestCase):
    def test_exact_apply_reuse_reverse_and_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';source.mkdir()
            ports=root/'ports/clr';ports.mkdir(parents=True)
            before='before\n';after='after\n'
            diff=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/input',tofile='b/input'))
            digest=lambda value:hashlib.sha256(value.encode()).hexdigest()
            spec={'files':{'input':{'before':digest(before),'after':digest(after)}},'patch_sha256':digest(diff)}
            (ports/'ps5-public-resources.json').write_text(json.dumps(spec))
            patch_file=ports/'ps5-public-resources.patch';patch_file.write_text(diff)
            target=source/'input';target.write_text(before)
            with patch.object(resources,'ROOT',root):
                self.assertFalse(resources.prepared(source))
                resources.prepare(source);self.assertEqual(target.read_text(),after)
                resources.prepare(source);self.assertTrue(resources.prepared(source))
                resources.prepare(source,True);self.assertEqual(target.read_text(),before)
                resources.prepare(source,True);resources.prepare(source)
                target.write_text('unknown\n')
                for reverse in (False,True):
                    with self.assertRaisesRegex(ValueError,'prerequisites'):resources.prepare(source,reverse)
                    self.assertEqual(target.read_text(),'unknown\n')
                patch_file.write_text(diff+'\n')
                with self.assertRaisesRegex(ValueError,'patch changed'):resources.prepared(source)


    def test_real_patch_preflight_shapes_and_effective_dynamic_capacity(self):
        # Compile the actual added/context patch statements, not a Python reimplementation.
        diff=(resources.ROOT/'ports/clr/ps5-public-resources.patch').read_text()
        after='\n'.join(line[1:] for line in diff.splitlines()
                        if line.startswith(('+',' ')) and not line.startswith('+++'))
        early=after[after.index('  const bool ps5Linear'):]
        early=early[:early.index('#endif')]
        late=after[after.index('  if (uint64_t(blockDimX)'):]
        late=late[:late.index('#endif')]
        program=r"""
#include <cassert>
#include <cstdint>
struct Info { uint64_t size_,availableLDSSize_,localMemSize_; };
constexpr int hipErrorInvalidValue=1;
int validate(uint32_t blockDimX,uint32_t blockDimY,uint32_t blockDimZ,
             uint32_t globalWorkSizeX,uint32_t globalWorkSizeY,uint32_t globalWorkSizeZ,
             uint32_t sharedMemBytes,const Info* ps5Info) {
 // Genuine upstream checks zero dimensions before the patched code.
 if (!blockDimX||!blockDimY||!blockDimZ||!globalWorkSizeX||!globalWorkSizeY||!globalWorkSizeZ) return 1;
"""+early+late+r"""
 return 0;
}
int main(){
 const Info initial{64,0,0},q8{256,0,0},q4{128,384,384},rms{1024,65536,0},bad{128,0,384};
 assert(!validate(64,1,1,128,1,1,0,&initial));
 assert(!validate(256,1,1,5632,1,1,0,&q8));
 assert(!validate(32,4,1,160,4,1,0,&q4));
 assert(!validate(32,4,1,160,8,1,0,&q4)); // Q4 supports grid Y > 1.
 assert(!validate(1024,1,1,1024,1,1,128,&rms));
 assert(!validate(1024,1,1,1024,1,1,65536,&rms));
 assert(validate(64,1,1,64,1,1,0,&q8));
 assert(validate(256,1,1,256,1,1,0,&initial));
 assert(validate(128,1,1,128,1,1,0,&q4));
 assert(validate(32,2,1,160,2,1,0,&q4));
 assert(validate(32,8,1,160,8,1,0,&q4));
 assert(validate(32,4,2,160,4,2,0,&q4));
 assert(validate(32,4,1,160,4,2,0,&q4));
 assert(validate(32,4,1,0xffffffe0u,4,1,0,&q4));
 assert(validate(32,4,1,159,4,1,0,&q4));
 assert(validate(32,4,1,160,5,1,0,&q4));
 assert(validate(256,1,1,256,2,1,0,&q8));
 for (unsigned shared:{1u,384u,65537u}) assert(validate(32,4,1,160,4,1,shared,&q4));
 assert(validate(32,4,1,160,4,1,0,&bad));
 assert(validate(1024,1,1,1024,1,1,0,&rms));
 assert(validate(1024,1,1,1024,1,1,65537,&rms));
 assert(validate(0,4,1,160,4,1,0,&q4));
}
"""
        program='#include <initializer_list>\n'+program
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'preflight.cpp';binary=Path(directory)/'preflight'
            source.write_text(program)
            compiler=shutil.which('clang++-18') or shutil.which('c++')
            self.assertIsNotNone(compiler)
            subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror',str(source),'-o',str(binary)],check=True,capture_output=True)
            subprocess.run([str(binary)],check=True,capture_output=True)


    def test_actual_adapter_pointer_preflight_and_error_publication(self):
        source=(resources.ROOT/'ports/clr/ps5-device.cpp').read_text()
        start=source.index('    bool validateKernel(')
        method=source[start:source.index('#include "ps5-device-unsupported.inc"',start)]
        diff=(resources.ROOT/'ports/clr/ps5-public-resources.patch').read_text()
        after='\n'.join(line[1:] for line in diff.splitlines()
                        if line.startswith(('+',' ')) and not line.startswith('+++'))
        start=after.index('  // Reject non-owned pointers synchronously')
        hook=after[start:after.index('#endif',start)]
        self.assertIn('#if defined(WITH_PS5_DEVICE)\n'+hook,after)
        program=r"""
#include <cassert>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <string>
constexpr int T_POINTER=1, hipErrorInvalidValue=1;
struct ObjectError:std::runtime_error{using std::runtime_error::runtime_error;};
struct Desc{size_t offset_,size_;int type_;};
struct Arg{size_t offset,size;std::string kind;};
struct Signature{std::vector<Desc> args{{0,8,T_POINTER}};size_t bytes=8;
 size_t numParameters()const{return args.size();}size_t paramsSize()const{return bytes;}
 const Desc& at(size_t i)const{return args.at(i);}};
struct DeviceKernel{virtual ~DeviceKernel()=default;};
struct Kernel:DeviceKernel{struct{std::vector<Arg> arguments{{0,8,"global_buffer"}};}native;
 Signature sig;const Signature& signature()const{return sig;}};
namespace amd{
struct MemObjMap{static inline bool live=true;static inline int checks=0;static void* FindMemObj(const void* p){++checks;auto v=reinterpret_cast<uintptr_t>(p);return live&&v>=0x10000&&v<0x10020?reinterpret_cast<void*>(1):nullptr;}};
namespace device{struct VirtualDevice{const void* owner;const int& device()const{return *static_cast<const int*>(owner);}};}
struct Params{alignas(8) uint8_t bytes[8]{};bool okay=true;bool check()const{return okay;}const void* values()const{return bytes;}};
struct Kernel{::Kernel selected;Params params;bool present=true;
 const DeviceKernel* getDeviceKernel(const auto&)const{return present?&selected:nullptr;}
 const Params& parameters()const{return params;}};
struct NDRangeKernelCommand{enum{CooperativeGroups=1};};
}
struct State{struct{bool terminal=false;}observed;bool locked=false;int operations=0;
 struct Data{State* state;uintptr_t base=0x10000;size_t size=32;bool live=true;int checks=0;
  int resolve(const void* p,size_t n){assert(state->locked&&n==1);++checks;auto v=reinterpret_cast<uintptr_t>(p);
   if(!live||v<base||v-base>=size)throw ObjectError("not owned");return 1;}}data{this};
 template<class F>int operation(F f){++operations;locked=true;try{f();locked=false;return 0;}catch(...){locked=false;return -1;}}};
struct Base{virtual bool validateKernel(const amd::Kernel&,const amd::device::VirtualDevice*,bool)=0;};
struct Device:Base{State* state_;
"""+method+r"""
};
struct HipDevice{Device* d;std::vector<Device*> devices(){return {d};}};
int preflight(Device& dev,amd::Kernel* kernel){
 HipDevice hd{&dev};HipDevice* g_devices[]={&hd};int deviceId=0;uint32_t params=0;
"""+hook+r"""
 return 0;
}
int main(){State state;state.locked=true;Device d;d.state_=&state;amd::Kernel k;
 auto pointer=[&](uintptr_t v){std::memcpy(k.params.bytes,&v,8);};
 for(uintptr_t v:{uintptr_t(0),uintptr_t(0x10000),uintptr_t(0x10001),uintptr_t(0x1001f)}){pointer(v);assert(!preflight(d,&k));}
 for(uintptr_t v:{uintptr_t(1),uintptr_t(0xffff),uintptr_t(0x10020),uintptr_t(0x20000)}){pointer(v);assert(preflight(d,&k)==hipErrorInvalidValue);assert(!state.observed.terminal&&state.locked&&state.operations==0);}
 pointer(0x10000);amd::MemObjMap::live=false;assert(preflight(d,&k)==hipErrorInvalidValue);amd::MemObjMap::live=true;assert(!preflight(d,&k));
 int checks=amd::MemObjMap::checks;k.selected.sig.bytes=7;assert(preflight(d,&k)==hipErrorInvalidValue);assert(amd::MemObjMap::checks==checks);k.selected.sig.bytes=8;
 k.selected.sig.args[0].offset_=1;assert(preflight(d,&k)==hipErrorInvalidValue);k.selected.sig.args[0].offset_=0;
 k.selected.sig.args[0].type_=0;assert(preflight(d,&k)==hipErrorInvalidValue);k.selected.sig.args[0].type_=T_POINTER;
 k.params.okay=false;assert(preflight(d,&k)==hipErrorInvalidValue);k.params.okay=true;
 k.present=false;assert(preflight(d,&k)==hipErrorInvalidValue);k.present=true;
 assert(!d.validateKernel(k,nullptr,true));state.observed.terminal=true;assert(!d.validateKernel(k,nullptr,false));state.observed.terminal=false;
 amd::device::VirtualDevice wrong{&state};assert(!d.validateKernel(k,&wrong,false));
 amd::device::VirtualDevice same{&d};assert(d.validateKernel(k,&same,false));
 assert(!preflight(d,&k));assert(state.operations==0&&state.locked);
}
"""
        # VirtualDevice::device is only used for identity; match its address type to Device.
        program=program.replace('namespace amd{','struct Device;\nnamespace amd{',1).replace('const int& device()const{return *static_cast<const int*>(owner);}','const Device& device()const{return *static_cast<const Device*>(owner);}')
        with tempfile.TemporaryDirectory() as directory:
            cpp=Path(directory)/'pointer.cpp';binary=Path(directory)/'pointer';cpp.write_text(program)
            compiler=shutil.which('clang++-18') or shutil.which('c++')
            subprocess.run([compiler,'-std=c++20','-Wall','-Wextra','-Werror',str(cpp),'-o',str(binary)],check=True,capture_output=True)
            subprocess.run([str(binary)],check=True,capture_output=True)


if __name__=='__main__':unittest.main()
