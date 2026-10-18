"""Compile the production watchdog; exercise deadlines without wall-clock sleeps."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = r"""
#include "watchdog.h"
#include <pthread.h>
#include <time.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
static const char* mode;
static uint64_t tick=1000000000ULL, deadline;
static int creates=0, joins=0, sleeps=0;
static void* (*entry)(void*);
extern "C" [[noreturn]] void ps5hip_clr_gpu_fail(const char* why) {
 std::printf("%s creates=%d joins=%d sleeps=%d\n",why,creates,joins,sleeps);
 std::exit(77);
}
extern "C" int __wrap_clock_gettime(clockid_t id, timespec* t) {
 if(id!=CLOCK_MONOTONIC)std::abort();
 if(!std::strcmp(mode,"clock-error"))return -1;
 t->tv_sec=tick/1000000000ULL;t->tv_nsec=tick%1000000000ULL;return 0;
}
extern "C" int __wrap_pthread_create(pthread_t* thread,const pthread_attr_t*,void*(*fn)(void*),void*) {
 ++creates;*thread=pthread_t(1);entry=fn;
 return !std::strcmp(mode,"create-error") ? EAGAIN : 0;
}
extern "C" int __wrap_pthread_join(pthread_t,void**) {
 ++joins;
 if(!std::strcmp(mode,"join-error"))return EINVAL;
 if(!std::strcmp(mode,"late-join"))tick=deadline;
 if(!std::strcmp(mode,"join-regression"))tick=0;
 entry(nullptr); // The real worker must observe finish's stop flag and return.
 return 0;
}
extern "C" int __wrap_nanosleep(const timespec* t,timespec*) {
 if(t->tv_sec || t->tv_nsec!=10000000)std::abort();
 if(++sleeps>2)std::abort();
 tick=deadline;
 if(!std::strcmp(mode,"sleep-error")){errno=EINVAL;return -1;}
 if(!std::strcmp(mode,"interrupted")){errno=EINTR;return -1;}
 return 0;
}
int main(int argc,char**argv) {
 if(argc!=3)return 2;mode=argv[1];unsigned ms=std::strtoul(argv[2],nullptr,10);
 deadline=tick+uint64_t(ms)*1000000;
 if(!std::strcmp(mode,"missing-start")){ps5hip_clr_gpu::watchdog_finish();return 3;}
 ps5hip_clr_gpu::watchdog_start(ms);
 if(!std::strcmp(mode,"repeat-start"))ps5hip_clr_gpu::watchdog_start(ms);
 if(!std::strcmp(mode,"timeout")){tick=deadline;entry(nullptr);}
 if(!std::strcmp(mode,"missing-finish")||!std::strcmp(mode,"interrupted")||!std::strcmp(mode,"sleep-error"))entry(nullptr);
 if(!std::strcmp(mode,"worker-regression")){tick=0;entry(nullptr);}
 if(!std::strcmp(mode,"late-finish"))tick=deadline;
 else if(!std::strcmp(mode,"finish-regression"))tick=0;
 else tick=deadline-1;
 ps5hip_clr_gpu::watchdog_finish();
 if(!std::strcmp(mode,"repeat-finish"))ps5hip_clr_gpu::watchdog_finish();
 if(creates!=1||joins!=1||sleeps!=0)return 4;
 std::puts("PASS");
}
"""

class WatchdogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('clang++-18')
        if not compiler:
            raise RuntimeError('clang++-18 required for actual watchdog regression')
        cls.temp = tempfile.TemporaryDirectory()
        folder = Path(cls.temp.name)
        source = folder/'harness.cpp'
        source.write_text(HARNESS)
        cls.binary = folder/'watchdog-test'
        subprocess.run([compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror',
                        '-pthread', '-I'+str(ROOT/'tools/clr_gpu'), str(source),
                        str(ROOT/'tools/clr_gpu/watchdog.cpp'),
                        *['-Wl,--wrap='+name for name in ('clock_gettime', 'nanosleep',
                                                         'pthread_create', 'pthread_join')],
                        '-o', str(cls.binary)], check=True, capture_output=True, text=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def check(self, mode, milliseconds, expected, creates=1, joins=0, sleeps=0):
        result = subprocess.run([str(self.binary), mode, str(milliseconds)],
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 0 if expected == 'PASS' else 77, result)
        suffix = '' if expected == 'PASS' else f' creates={creates} joins={joins} sleeps={sleeps}'
        self.assertEqual(result.stdout, expected+suffix+'\n')
        self.assertEqual(result.stderr, '')

    def test_valid_deadlines_finish_one_nanosecond_before_limit(self):
        for milliseconds in (1, 15000, 30000, 60000, 120000, 1800000):
            with self.subTest(milliseconds=milliseconds):
                self.check('valid', milliseconds, 'PASS')

    def test_invalid_state_never_creates_an_extra_worker(self):
        for milliseconds in (0, 120001, 1799999, 1800001, 4294967295):
            with self.subTest(milliseconds=milliseconds):
                self.check('valid', milliseconds, 'watchdog-state', creates=0)
        self.check('missing-start', 120000, 'watchdog-state', creates=0)
        self.check('repeat-start', 120000, 'watchdog-state')
        self.check('repeat-finish', 120000, 'watchdog-state', joins=1)

    def test_timeout_and_late_cleanup_remain_strict(self):
        for milliseconds in (30000, 60000, 120000, 1800000):
            with self.subTest(milliseconds=milliseconds):
                self.check('timeout', milliseconds, 'watchdog-timeout')
                self.check('missing-finish', milliseconds, 'watchdog-timeout', sleeps=1)
                self.check('interrupted', milliseconds, 'watchdog-timeout', sleeps=1)
                self.check('late-finish', milliseconds, 'watchdog-late-finish')
                self.check('late-join', milliseconds, 'watchdog-late-join', joins=1)

    def test_clock_thread_and_sleep_errors_preserved(self):
        self.check('clock-error', 120000, 'watchdog-clock', creates=0)
        self.check('create-error', 120000, 'watchdog-create')
        self.check('join-error', 120000, 'watchdog-join', joins=1)
        self.check('sleep-error', 120000, 'watchdog-sleep', sleeps=1)
        self.check('worker-regression', 120000, 'watchdog-clock-regressed')
        self.check('finish-regression', 120000, 'watchdog-clock-regressed')
        self.check('join-regression', 120000, 'watchdog-clock-regressed', joins=1)

if __name__ == '__main__':
    unittest.main()
