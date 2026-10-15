"""Execute the exact native-title dump helper against real files and syscall controls."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RmsnormDumpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('clang++-18')
        if not compiler:
            raise RuntimeError('clang++-18 required for native helper regression')
        source = (ROOT/'tools/public_hip_rmsnorm_console_title.cpp').read_text()
        start = source.index('void dump_exclusive(')
        end = source.index('\ndouble absolute(', start)
        helper = source[start:end]
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.binary = cls.directory/'test'
        prefix = r'''
#include <string>
#include <vector>
#include <cstdint>
#include <cerrno>
#include <stdexcept>
#include <fstream>
#include <iterator>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
static std::string mode;
static unsigned writes=0, syncs=0, closes=0;
static int flags_seen=0, permissions_seen=0;
[[noreturn]] void ps5hip_clr_gpu_fail(const char* reason) { throw std::runtime_error(reason); }
int controlled_open(const char* path,int flags,int permissions) {
    flags_seen=flags;permissions_seen=permissions;return ::open(path,flags,permissions);
}
ssize_t controlled_write(int fd,const void* data,size_t size) {
    ++writes;
    if (mode=="eintr-bound" || (mode=="eintr-recover" && writes<=8)) {errno=EINTR;return -1;}
    if (mode=="zero") return 0;
    if (mode=="error" && writes>1) {errno=EIO;return -1;}
    const size_t chunk=(mode=="partial" || mode=="error") && size>3?3:size;
    return ::write(fd,data,chunk);
}
int controlled_fsync(int fd) {++syncs;if(mode=="fsync") {errno=EIO;return -1;}return ::fsync(fd);}
int controlled_close(int fd) {++closes;const int result=::close(fd);if(mode=="close") {errno=EIO;return -1;}return result;}
#define open controlled_open
#define write controlled_write
#define fsync controlled_fsync
#define close controlled_close
'''
        suffix = r'''
#undef open
#undef write
#undef fsync
#undef close
void check(bool ok) {if(!ok)throw std::runtime_error("assertion");}
int main(int argc,char** argv) {
    try {
        check(argc==3);mode=argv[1];const std::string path=argv[2];
        const std::vector<uint8_t> data={0,1,2,3,4,5,6,7,8,9};
        if(mode=="exclusive") {std::ofstream file(path,std::ios::binary);file<<"preserve";}
        bool failed=false;std::string reason;
        try {dump_exclusive(path,data.data(),data.size());}
        catch(const std::runtime_error& error) {failed=true;reason=error.what();}
        check(flags_seen==(O_WRONLY|O_CREAT|O_EXCL) && permissions_seen==0600);
        std::ifstream file(path,std::ios::binary);check(bool(file));
        const std::vector<uint8_t> actual{std::istreambuf_iterator<char>(file),{}};
        if(mode=="exclusive") {
            check(failed && reason=="raw-dump-open" && writes==0 && syncs==0 && closes==0);
            check(std::string(actual.begin(),actual.end())=="preserve");
        } else {
            check(syncs==1 && closes==1);
            if(mode=="partial" || mode=="eintr-recover" || mode=="success") {
                check(!failed && actual==data);
                if(mode=="partial")check(writes==4);
                if(mode=="eintr-recover")check(writes==9);
            } else {
                check(failed && reason=="raw-dump-write-sync-close");
                if(mode=="eintr-bound")check(writes==9 && actual.empty());
                if(mode=="zero")check(writes==1 && actual.empty());
                if(mode=="error")check(writes==2 && actual==std::vector<uint8_t>({0,1,2}));
                if(mode=="fsync" || mode=="close")check(actual==data);
            }
            struct stat state{};check(::stat(path.c_str(),&state)==0 && (state.st_mode&0777)==0600);
        }
    } catch(const std::exception& error) {return 1;}
}
'''
        generated = cls.directory/'test.cpp'
        generated.write_text(prefix+helper+suffix)
        subprocess.run([compiler,'-std=c++17','-Wall','-Wextra','-Werror',
                        '-fsanitize=address,undefined',str(generated),'-o',str(cls.binary)],check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_helper_syscall_and_persistence_controls(self):
        modes=['success','partial','eintr-recover','eintr-bound','zero','error','fsync','close','exclusive']
        for mode in modes:
            with self.subTest(mode=mode):
                subprocess.run([self.binary,mode,self.directory/(mode+'.raw')],check=True,timeout=10)


if __name__ == '__main__':
    unittest.main()
