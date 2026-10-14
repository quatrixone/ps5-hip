"""Apply pinned CLR patches and exercise extracted native/Linux startup functions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PORT = ROOT / "ports/clr"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def body(text, signature):
    start = text.index(signature)
    cursor = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[cursor] == "{") - (text[cursor] == "}")
        cursor += 1
    return text[start:cursor]


def run(argv, **kwargs):
    subprocess.run(argv, check=True, **kwargs)


AFFINITY = r'''
#include <cassert>
#include <cstdlib>
#include <cstddef>
#undef __linux__
#define PS5HIP_CLR_PLATFORM 1
struct NativeCpuSet { unsigned char bytes[32]; };
using pthread_t=unsigned long;
static int mode,getcalls,setcalls,threadcalls;
static bool AMD_CPU_AFFINITY=true;
static constexpr int _SC_PAGESIZE=47,_SC_NPROCESSORS_CONF=57;
static long mock_sysconf(int n){return n==47?(mode==1?-1:16384):(mode==2?-1:16);}
static pthread_t mock_self(){return 7;}
static int mock_get(pthread_t,size_t n,NativeCpuSet* m){++getcalls;assert(n==16);for(auto v:m->bytes)assert(v==0);if(mode==3)return 34;m->bytes[0]=63;return 0;}
static int mock_set(pthread_t,size_t n,const NativeCpuSet* m){++setcalls;assert(n==16);for(int i=16;i<32;++i)assert(m->bytes[i]==0);return mode==5?22:0;}
#define sysconf mock_sysconf
#define pthread_self mock_self
#define pthread_getaffinity_np mock_get
#define pthread_setaffinity_np mock_set
#define ClPrint(...) ((void)0)
static void fatal(const char*){throw 1;}
struct Thread {static bool init(){++threadcalls;return mode!=4;}};
struct Os {static size_t pageSize_;static long processorCount_;struct ThreadAffinityMask{NativeCpuSet mask_{};};static bool init();static void setThreadAffinity(const void*,const ThreadAffinityMask&);static bool setThreadAffinityToMainThread();};
size_t Os::pageSize_;long Os::processorCount_;
static NativeCpuSet nativeMask_{};
static int (*pthread_setaffinity_fptr)(pthread_t,size_t,const NativeCpuSet*);
'''
AFFINITY_MAIN = r'''
int main(int argc,char** argv){assert(argc==2);mode=std::atoi(argv[1]);bool ready=Os::init();
if(mode>=1&&mode<=4){assert(!ready);assert(threadcalls==(mode==4));assert(getcalls==(mode>=3));return 0;}
assert(ready&&threadcalls==1&&getcalls==1&&Os::pageSize_==16384&&Os::processorCount_==16);
for(int i=16;i<32;++i)assert(nativeMask_.bytes[i]==0);
Os::ThreadAffinityMask mask;mask.mask_.bytes[0]=3;
if(mode==6){mask.mask_.bytes[17]=1;try{Os::setThreadAffinity((void*)7,mask);assert(false);}catch(int){}assert(setcalls==0);nativeMask_.bytes[18]=1;assert(!Os::setThreadAffinityToMainThread()&&setcalls==0);return 0;}
if(mode==5){try{Os::setThreadAffinity((void*)7,mask);assert(false);}catch(int){}assert(!Os::setThreadAffinityToMainThread());assert(setcalls==2);return 0;}
Os::setThreadAffinity((void*)7,mask);assert(Os::setThreadAffinityToMainThread());assert(setcalls==2);}
'''
PATH_MODEL = r'''
#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <libgen.h>
#ifdef TEST_NATIVE
#undef __linux__
#endif
constexpr unsigned FILE_PATH_MAX_LENGTH=256;
static int calls;static bool success;
static long test_readlink(const char* p,char* out,unsigned n){++calls;assert(std::string(p)=="/proc/self/exe"&&n==256);if(!success)return -1;std::strcpy(out,"/tmp/application");return 16;}
#define readlink test_readlink
struct Os {static void getAppPathAndFileName(std::string&,std::string&);};
'''
PATH_MAIN = r'''
int main(){for(bool value:{false,true}){success=value;std::string a="old",b="old";int before=calls;Os::getAppPathAndFileName(a,b);
#ifdef TEST_NATIVE
assert(a.empty()&&b.empty()&&calls==before);
#else
assert(calls==before+1);if(success){assert(a=="application"&&b=="/tmp/application");}else{assert(a.empty()&&b.empty());}
#endif
}}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True,
                        help="Clean pinned ROCm/clr source; never modified")
    parser.add_argument("--compiler", default="clang++-18")
    args = parser.parse_args()
    base = json.loads((PORT / "manifest.json").read_text())
    adapter = json.loads((PORT / "adapter-manifest.json").read_text())
    require(adapter["upstream_revision"] == base["upstream_revision"], "revision mismatch")
    require(digest(PORT / "ps5-platform.patch") == base["patch_sha256"], "base patch changed")
    require(adapter["base_patch_sha256"] == base["patch_sha256"], "adapter base mismatch")
    for name, expected in adapter["patches"].items():
        require(digest(PORT / name) == expected, "patch changed: " + name)
    with tempfile.TemporaryDirectory(prefix="ps5hip-clr-startup-") as temp:
        work = Path(temp)
        for name in set(base["files"]) | set(adapter["files"]):
            source = args.upstream / name
            expected = (base["files"][name]["upstream_sha256"] if name in base["files"]
                        else adapter["files"][name]["base_sha256"])
            require(digest(source) == expected, "upstream changed: " + name)
            target = work / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for name in ["ps5-platform.patch", *adapter["patch_order"]]:
            run(["git", "apply", "--check", str(PORT / name)], cwd=work)
            run(["git", "apply", str(PORT / name)], cwd=work)
        expected = {name: value["adapted_sha256"] for name, value in base["files"].items()}
        expected.update({name: value["adapted_sha256"] for name, value in adapter["files"].items()})
        for name, sha in expected.items():
            require(digest(work / name) == sha, "patch roundtrip mismatch: " + name)
        text = (work / "rocclr/os/os_posix.cpp").read_text()
        start = text.index("// PS5 libkernel scePthread")
        end = text.index("typedef int (*pthread_setaffinity_fn)", start)
        affinity = AFFINITY + text[start:end] + "\n".join(body(text, name) for name in (
            "bool Os::init()", "void Os::setThreadAffinity(", "bool Os::setThreadAffinityToMainThread()")) + AFFINITY_MAIN
        path = PATH_MODEL + body(text, "void Os::getAppPathAndFileName(") + PATH_MAIN
        for name, model, defines, modes in [
            ("affinity", affinity, [], [str(i) for i in range(7)]),
            ("native-path", path, ["-DPS5HIP_CLR_PLATFORM", "-DTEST_NATIVE"], [None]),
            ("linux-platform-path", path, ["-DPS5HIP_CLR_PLATFORM"], [None]),
            ("linux-path", path, [], [None]),
        ]:
            source, binary = work / (name + ".cpp"), work / name
            source.write_text(model)
            run([args.compiler, "-std=c++20", "-fsanitize=address,undefined",
                 "-fno-sanitize-recover=all", *defines, str(source), "-o", str(binary)])
            for mode in modes:
                run([str(binary)] + ([] if mode is None else [mode]))
    print("PASS: all patch hashes/roundtrip, seven native affinity modes, three path profiles")


if __name__ == "__main__":
    main()
