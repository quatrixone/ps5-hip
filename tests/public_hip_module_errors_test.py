"""Exercise the actual upstream extraction function with injected extractor results.

This validates error propagation/ownership under sanitizers, not ELF parsing or GPU execution.
"""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_public_hip_module import prepare, BEFORE, AFTER
from prepare_public_hip_memory import digest

PREAMBLE = r'''
#include <cassert>
#include <cstddef>
#include <string>
#include <vector>
#include <utility>
using hipError_t=int;
using address=char*;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipErrorOutOfMemory=2,
 hipErrorInvalidImage=200,hipErrorNoBinaryForGpu=209,hipErrorInvalidKernelFile=218,
 hipErrorFileNotFound=301;
static bool HIP_USE_RUNTIME_UNBUNDLER=true,open_ok=true;
static size_t file_size=128;
static int extracted=0,programs=0,infos=0,comgr_calls=0,elf_calls=0;
static char image[128];
static std::vector<std::pair<const void*,size_t>> outputs;
#define LogPrintfError(...) ((void)0)
namespace amd {
struct Context {};
struct Program {explicit Program(Context&) {++programs;} ~Program(){--programs;}};
struct Os {static bool GetFileHandle(const char*,int*,size_t* size){*size=file_size;return open_ok;}};
}
namespace hip {
struct ISA {std::string isaName(){return "gfx1013";}};
struct Backend {ISA isa(){return {};}};
struct Device {
 int id; Backend backend; amd::Context context;
 std::vector<Backend*> devices(){return {&backend};}
 int deviceId(){return id;} amd::Context* asContext(){return &context;}
};
}
struct CodeObject {
 static int ExtractCodeObjectFromFile(int,size_t,const void** out,const std::vector<std::string>&,
                                     std::vector<std::pair<const void*,size_t>>& entries){
   *out=image;entries=outputs;return extracted;
 }
 static int ExtractCodeObjectFromMemory(const void*,const std::vector<std::string>&,
                                       std::vector<std::pair<const void*,size_t>>& entries,std::string&){
   entries=outputs;return extracted;
 }
 static size_t ElfSize(const void*){++elf_calls;return sizeof(image);}
};
struct FatBinaryDeviceInfo {
 amd::Program* program_=nullptr;
 FatBinaryDeviceInfo(const void*,size_t,size_t){++infos;}
 ~FatBinaryDeviceInfo(){delete program_;--infos;}
};
struct FatBinaryInfo {
 std::string fname_,uri_; int fdesc_=0; size_t fsize_=0; const void* image_=image;
 FatBinaryDeviceInfo* fatbin_dev_info_[2]={};
 ~FatBinaryInfo(){for(auto* p:fatbin_dev_info_)delete p;}
 int ExtractFatBinaryUsingCOMGR(const std::vector<hip::Device*>&){++comgr_calls;return 801;}
 int ExtractFatBinary(const std::vector<hip::Device*>&);
};
'''

DRIVER = r'''
int main(){
 hip::Device a{0},b{1};std::vector<hip::Device*> devices{&a,&b};
 for(bool file:{false,true}){
  for(int error:{hipErrorInvalidValue,hipErrorInvalidImage,hipErrorOutOfMemory,801,999}){
   FatBinaryInfo f;if(file)f.fname_="module.hsaco";
   extracted=error;outputs.clear();
   assert(f.ExtractFatBinary(devices)==error);assert(!programs&&!infos&&!elf_calls);
  }
  for(int status:{hipSuccess,hipErrorNoBinaryForGpu}){
   for(auto entries:std::vector<std::vector<std::pair<const void*,size_t>>>{
       {},{{image,16}},{{image,16},{image,16},{image,16}},
       {{image,0},{image,16}},{{nullptr,16},{image,16}},{{image,16},{nullptr,16}}}){
    FatBinaryInfo f;if(file)f.fname_="module.hsaco";extracted=status;outputs=entries;
    assert(f.ExtractFatBinary(devices)==hipErrorInvalidImage);assert(!programs&&!infos);
   }
  }
  {
   FatBinaryInfo f;if(file)f.fname_="module.hsaco";extracted=hipSuccess;
   outputs={{nullptr,0},{image,16}};
   assert(f.ExtractFatBinary(devices)==hipErrorInvalidImage);assert(!programs&&!infos);
  }
  {
   FatBinaryInfo f;if(file)f.fname_="module.hsaco";extracted=hipSuccess;
   outputs={{image,16},{image+16,16}};
   assert(f.ExtractFatBinary(devices)==hipSuccess);assert(programs==2&&infos==2);
  }
  {
   FatBinaryInfo f;if(file)f.fname_="module.hsaco";extracted=hipErrorNoBinaryForGpu;
   outputs={{nullptr,0},{image+16,16}};
   assert(f.ExtractFatBinary(devices)==hipErrorNoBinaryForGpu);assert(programs==1&&infos==1);
  }
  assert(!programs&&!infos);
  {
   FatBinaryInfo f;if(file)f.fname_="module.hsaco";extracted=hipErrorNoBinaryForGpu;
   outputs={{nullptr,0},{nullptr,0}};
   assert(f.ExtractFatBinary(devices)==hipErrorNoBinaryForGpu);assert(!programs&&!infos);
  }
 }
 {
  FatBinaryInfo f;extracted=hipErrorInvalidKernelFile;
  assert(f.ExtractFatBinary(devices)==0);assert(programs==2&&infos==2&&elf_calls==2);
 }
 {
  FatBinaryInfo f;HIP_USE_RUNTIME_UNBUNDLER=false;
  assert(f.ExtractFatBinary(devices)==801);assert(comgr_calls==1&&!infos&&!programs);
  HIP_USE_RUNTIME_UNBUNDLER=true;f.image_=nullptr;
  assert(f.ExtractFatBinary(devices)==hipErrorInvalidValue);
  f.fname_="missing";open_ok=false;
  assert(f.ExtractFatBinary(devices)==hipErrorFileNotFound);
  open_ok=true;file_size=0;
  assert(f.ExtractFatBinary(devices)==hipErrorInvalidImage);
 }
 assert(!infos&&!programs);
}
'''


class ModuleErrorsTest(unittest.TestCase):
    def test_actual_extractor(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            target = source / 'hipamd/src/hip_fatbin.cpp'
            target.parent.mkdir(parents=True)
            target.write_bytes((ROOT / '.deps/sources/clr/hipamd/src/hip_fatbin.cpp').read_bytes())
            # Reproduce the exact public admission prerequisite from pinned source.
            patch = (ROOT / 'ports/clr/ps5-public-hip.patch').read_text()
            section = '--- a/hipamd/src/hip_fatbin.cpp\n' + patch.split('--- a/hipamd/src/hip_fatbin.cpp\n')[1].split('--- a/', 1)[0]
            subprocess.run(['patch', '--batch', '--fuzz=0', '-p1'], input=section,
                           cwd=source, text=True, check=True, capture_output=True)
            self.assertEqual(digest(target), BEFORE)
            prepare(source)
            self.assertEqual(digest(target), AFTER)
            prepare(source)
            body = target.read_text().split('hipError_t FatBinaryInfo::ExtractFatBinary(')[1].split('\n}\n', 1)[0]
            unit = source / 'module.cpp'
            unit.write_text(PREAMBLE + 'hipError_t FatBinaryInfo::ExtractFatBinary(' + body + '\n}\n' + DRIVER)
            env = dict(os.environ, ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
            for enabled in (True, False):
                binary = source / ('patched' if enabled else 'upstream')
                argv = ['clang++-18', '-std=c++20', '-O1', '-g', '-fsanitize=address,undefined', str(unit), '-o', str(binary)]
                if enabled: argv.append('-DWITH_PS5_DEVICE')
                subprocess.run(argv, check=True, capture_output=True)
                result = subprocess.run([str(binary)], env=env, capture_output=True)
                if enabled: self.assertEqual(result.returncode, 0, result.stderr.decode())
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(b'null pointer', result.stderr)
            target.write_text(target.read_text() + '// drift\n')
            with self.assertRaises(ValueError): prepare(source)


if __name__ == '__main__':
    unittest.main()
