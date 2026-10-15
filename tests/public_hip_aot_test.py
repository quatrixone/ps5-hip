"""Fault-inject the actual AOT extractor; full runtime tests cover real ELF parsing."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_public_hip_aot import prepare

PREAMBLE = r'''
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <string>
#include <string_view>
#include <vector>
static int fail_new=-1, fault=0, opened=0, closed=0, infos=0, programs=0;
void* operator new(size_t n){
 if(fail_new==0){fail_new=-1;throw std::bad_alloc();}
 if(fail_new>0)--fail_new;
 if(void* p=std::malloc(n?n:1))return p;
 throw std::bad_alloc();
}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,size_t) noexcept {std::free(p);}
static std::FILE* fault_open(const char*,const char*){
 if(fault==1)return nullptr;
 auto* f=std::tmpfile();assert(f);++opened;
 unsigned char bytes[128]={};assert(std::fwrite(bytes,1,sizeof(bytes),f)==sizeof(bytes));return f;
}
static int fault_close(std::FILE* f){++closed;int r=std::fclose(f);return fault==8?-1:r;}
static int fault_seek(std::FILE* f,long n,int origin){
 return (fault==2&&origin==SEEK_END)||(fault==3&&origin==SEEK_SET)?-1:std::fseek(f,n,origin);
}
static long fault_tell(std::FILE* f){return fault==4?-1:std::ftell(f);}
static size_t fault_read(void* p,size_t s,size_t n,std::FILE* f){return fault==5?0:std::fread(p,s,n,f);}
static int fault_getc(std::FILE* f){return fault==6?0:std::fgetc(f);}
static int fault_error(std::FILE* f){return fault==7?1:std::ferror(f);}
using hipError_t=int;
constexpr int hipSuccess=0,hipErrorInvalidValue=1,hipErrorOutOfMemory=2,
 hipErrorInvalidDevice=101,hipErrorInvalidImage=200,hipErrorNoBinaryForGpu=209,
 hipErrorFileNotFound=301,hipErrorNotSupported=801;
namespace ps5hip {
struct ObjectError {};
struct Object {std::string_view target;};
static Object inspect_code_object(const void*,size_t){
 if(fault==9)throw ObjectError{};
 return {fault==10?"gfx1030":"gfx1013"};
}
}
namespace amd {
struct Context {};
struct Program {explicit Program(Context&){++programs;} ~Program(){--programs;}};
}
namespace hip {
struct Device {int id;amd::Context context;int deviceId(){return id;} amd::Context* asContext(){return &context;}};
}
struct FatBinaryDeviceInfo {
 const unsigned char* image;amd::Program* program_=nullptr;
 FatBinaryDeviceInfo(const void* p,size_t n,size_t):image(static_cast<const unsigned char*>(p)){assert(n==128);++infos;}
 ~FatBinaryDeviceInfo(){assert(image[0]==0);delete program_;--infos;}
};
struct FatBinaryInfo {
 bool ps5_extract_attempted_=false;
 std::vector<unsigned char> ps5_image_storage_;
 std::string fname_="fixture";
 const void* image_=nullptr;
 std::vector<FatBinaryDeviceInfo*> fatbin_dev_info_{nullptr};
 ~FatBinaryInfo(){for(auto* p:fatbin_dev_info_)delete p;}
 hipError_t ExtractFatBinary(const std::vector<hip::Device*>&);
};
'''

DRIVER = r'''
int main(){
 hip::Device a{0},bad{-1},outside{1};std::vector<hip::Device*> devices{&a};
 for(int mode=0;mode<=10;++mode){
  fault=mode;
  {
   FatBinaryInfo f;
   const int expected=mode==0?hipSuccess:mode==1?hipErrorFileNotFound:mode==10?hipErrorNoBinaryForGpu:hipErrorInvalidImage;
   assert(f.ExtractFatBinary(devices)==expected);
   assert(opened==closed);
   assert((f.fatbin_dev_info_[0]!=nullptr)==(mode==0));
   const void* image=f.image_;
   assert(f.ExtractFatBinary(devices)==hipErrorInvalidValue&&f.image_==image);
  }
  assert(!infos&&!programs);
 }
 fault=0;
 for(int allocation=0;allocation<3;++allocation){
  {FatBinaryInfo f;fail_new=allocation;
   assert(f.ExtractFatBinary(devices)==hipErrorOutOfMemory);
   assert(fail_new==-1&&!f.fatbin_dev_info_[0]&&!infos&&!programs&&opened==closed);
   assert(f.ExtractFatBinary(devices)==hipErrorInvalidValue);
  }
 }
 for(auto invalid:std::vector<std::vector<hip::Device*>>{{},{nullptr},{&a,&a},{&bad},{&outside}}){
  FatBinaryInfo f;int before=opened;
  assert(f.ExtractFatBinary(invalid)==hipErrorInvalidDevice&&opened==before);
 }
 {FatBinaryInfo f;f.fname_.clear();f.image_=reinterpret_cast<void*>(1);
  assert(f.ExtractFatBinary(devices)==hipErrorNotSupported);}
 assert(opened==closed&&!infos&&!programs);
}
'''


class AotTest(unittest.TestCase):
    def test_actual_extractor_faults(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            folder = source / 'hipamd/src'
            folder.mkdir(parents=True)
            for name in ('hip_fatbin.cpp', 'hip_fatbin.hpp'):
                (folder / name).write_bytes((ROOT / '.deps/sources/clr/hipamd/src' / name).read_bytes())
            patch = (ROOT / 'ports/clr/ps5-public-hip.patch').read_text()
            section = '--- a/hipamd/src/hip_fatbin.cpp\n' + patch.split('--- a/hipamd/src/hip_fatbin.cpp\n')[1].split('--- a/', 1)[0]
            subprocess.run(['patch', '--batch', '--fuzz=0', '-p1'], input=section,
                           cwd=source, text=True, check=True, capture_output=True)
            prepare(source)
            prepare(source)
            text = (folder / 'hip_fatbin.cpp').read_text()
            body = text.split('hipError_t FatBinaryInfo::ExtractFatBinary(')[1].split('\n}\n', 1)[0]
            for original, replacement in {'fopen': 'open', 'fclose': 'close', 'fseek': 'seek',
                                          'ftell': 'tell', 'fread': 'read', 'fgetc': 'getc',
                                          'ferror': 'error'}.items():
                body = body.replace('std::' + original, 'fault_' + replacement)
            unit = source / 'aot.cpp'
            unit.write_text(PREAMBLE + 'hipError_t FatBinaryInfo::ExtractFatBinary(' + body + '\n}\n' + DRIVER)
            binary = source / 'aot'
            subprocess.run(['clang++-18', '-std=c++20', '-DWITH_PS5_DEVICE', '-O1', '-g',
                            '-fsanitize=address,undefined', str(unit), '-o', str(binary)],
                           check=True, capture_output=True)
            result = subprocess.run([binary], capture_output=True, env=dict(os.environ,
                                    ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1'))
            self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == '__main__':
    unittest.main()
