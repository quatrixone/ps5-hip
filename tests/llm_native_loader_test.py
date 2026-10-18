"""Exercise generated staging under constrained heap and sanitizer exception paths."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import prepare_llm_native_loader as producer
ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.deps/sources/llama-client/src/llama-model-loader.cpp'
HARNESS = r'''
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>
#include <sys/mman.h>
#include <sys/resource.h>
constexpr size_t bytes = 36864000;
static unsigned allocations, releases, live, stage;
static int fail;
static void * owned;
void * ggml_aligned_malloc(size_t n) {
 assert(n==bytes && !live);
 if(fail==1)return nullptr;
 owned=mmap(nullptr,n,PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0);
 assert(owned!=MAP_FAILED);++allocations;++live;return owned;
}
void ggml_aligned_free(void * p,size_t n) {
 assert(n==bytes&&p==owned&&live==1);assert(munmap(p,n)==0);
 owned=nullptr;--live;++releases;
}
struct File {
 void seek(size_t off,int origin){assert(off==42&&origin==SEEK_SET);++stage;if(fail==2)throw std::runtime_error("seek");}
 void read_raw(void * p,size_t n){assert(stage==1&&p==owned&&n==bytes);++stage;if(fail==3)throw std::runtime_error("read");memset(p,0x5a,n);}
};
struct Tensor {int type=7;};
void ggml_backend_tensor_set(Tensor *,const void * p,size_t off,size_t n){assert(stage==2&&p==owned&&off==0&&n==bytes);++stage;if(fail==4)throw std::runtime_error("upload");assert(static_cast<const unsigned char *>(p)[n-1]==0x5a);}
bool ggml_validate_row_data(int type,const void * p,size_t n){assert(stage==3&&type==7&&p==owned&&n==bytes);++stage;if(fail==5)throw std::runtime_error("validate");return fail!=6;}
const char * ggml_get_name(Tensor *){return "test";}
const char * format(const char *,const char *){return "invalid data";}
void exercise() {
 File f;File * file=&f;struct Weight {size_t offs=42;} w;auto * weight=&w;
 Tensor tensor;auto * cur=&tensor;const size_t n_size=bytes;const bool check_tensors=true;
 @BLOCK@
}
int main(int argc,char **) {
 if(argc>1){
  rlimit limit{16*1024*1024,16*1024*1024};assert(setrlimit(RLIMIT_DATA,&limit)==0);
  bool rejected=false;try{std::vector<unsigned char> ordinary(bytes);assert(ordinary.size()==bytes);}catch(const std::bad_alloc&){rejected=true;}assert(rejected);
 }
 for(fail=0;fail<=6;++fail){
  stage=0;bool caught=false;
  try{exercise();}catch(const std::bad_alloc&){assert(fail==1);caught=true;}catch(const std::runtime_error&){assert(fail>=2);caught=true;}
  assert(caught==(fail!=0));assert(!live&&!owned&&allocations==releases);
  assert(stage==(fail==1?0:fail==2?1:fail==3?2:fail==4?3:4));
 }
 assert(allocations==6&&releases==6);puts("PASS: 36,864,000-byte staging, success and six failure paths, balanced owners");
}
'''


class NativeLoader(unittest.TestCase):
    def test_render_is_narrow(self):
        original = SOURCE.read_text(); generated = producer.render(original)
        restored = generated.replace(producer.REPLACEMENT, producer.ALLOCATE).replace('\n#include "ggml-impl.h"\n#include <new>', '')
        self.assertEqual(restored.replace(producer.USER_GUARD, ''), original)
        for changed in (original.replace(producer.ALLOCATE, ''), original + producer.ALLOCATE,
                        original.replace('read_buf(n_size)', 'read_buf(n_size + 1)'),
                        original + '\n#include <new>\n'):
            with self.assertRaises(ValueError): producer.render(changed)

    def test_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/producer.GENERATED_NAME;target.write_text(producer.render(SOURCE.read_text()))
            self.assertEqual(producer.provenance(SOURCE,target)['classification'],'native-loader-aligned-staging')
            target.write_text(target.read_text()+'\n')
            with self.assertRaises(ValueError):producer.provenance(SOURCE,target)
            target.unlink();target.symlink_to(SOURCE)
            with self.assertRaises(ValueError):producer.provenance(SOURCE,target)

    def test_actual_generated_block(self):
        original=SOURCE.read_text(); generated=producer.render(original)
        start=generated.index('                    // Tensor staging must')
        end=generated.index('\n                }',start)
        block=generated[start:end]
        self.assertIn('ggml_backend_tensor_set(cur, read_buf.data(), 0, n_size);',block)
        self.assertIn('ggml_validate_row_data(cur->type, read_buf.data(), n_size)',block)
        with tempfile.TemporaryDirectory(prefix='ps5hip-loader-') as tmp:
            src=Path(tmp)/'test.cpp';src.write_text(HARNESS.replace('@BLOCK@',block))
            for mode in ('heap','sanitizer'):
                binary=Path(tmp)/mode
                flags=['-fsanitize=address,undefined','-fno-omit-frame-pointer'] if mode=='sanitizer' else []
                subprocess.run(['clang++-18','-std=c++17','-O1','-g','-Wall','-Wextra','-Werror',*flags,str(src),'-o',str(binary)],check=True,timeout=60)
                env={**os.environ,'ASAN_OPTIONS':'detect_leaks=1:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'}
                result=subprocess.run([str(binary),*(['heap'] if mode=='heap' else [])],env=env,capture_output=True,text=True,timeout=30)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                self.assertEqual(result.stderr,'');self.assertIn('balanced owners',result.stdout)
        self.assertEqual(SOURCE.read_text(),original)

if __name__=='__main__':unittest.main()
