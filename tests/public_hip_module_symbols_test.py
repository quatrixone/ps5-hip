"""Actual AOT function enumeration under sanitizers; no module/GPU execution claim."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_public_hip_symbols import prepare, BEFORE, AFTER
from prepare_public_hip_memory import digest

HARNESS = r'''
#include <cassert>
#include <unordered_map>
#include <algorithm>
#include <string>
#include <vector>
struct Kernel {};
struct Program {
 bool loaded=false;
 std::unordered_map<std::string,Kernel*> table;
 bool isCodeObjectLoaded()const{return loaded;}
 const auto& kernels()const{return table;}
 bool getGlobalFuncFromCodeObj(std::vector<std::string>*) const;
};
'''
DRIVER = r'''
int main(){
 Program p;Kernel a,b;std::vector<std::string> names{"prior"};
 assert(!p.getGlobalFuncFromCodeObj(&names));assert(names==std::vector<std::string>{"prior"});
 p.table={{"a_kernel",&a}};
 assert(!p.getGlobalFuncFromCodeObj(&names));assert(names==std::vector<std::string>{"prior"});
 p.loaded=true;
 assert(!p.getGlobalFuncFromCodeObj(nullptr));
 assert(p.getGlobalFuncFromCodeObj(&names));assert(names==std::vector<std::string>{"a_kernel"});
 p.table.emplace("b_kernel",&b);
 assert(p.getGlobalFuncFromCodeObj(&names));
 auto sorted=names;std::sort(sorted.begin(),sorted.end());
 assert((sorted==std::vector<std::string>{"a_kernel","b_kernel"}));
 // Repeated discovery replaces the output atomically and does not duplicate names.
 assert(p.getGlobalFuncFromCodeObj(&names));assert(names.size()==2);
 auto prior=names;
 p.table["z_invalid"]=nullptr;
 assert(!p.getGlobalFuncFromCodeObj(&names));assert(names==prior);
 p.table.erase("z_invalid");p.table[""]=&a;
 assert(!p.getGlobalFuncFromCodeObj(&names));assert(names==prior);
 p.table.clear();assert(!p.getGlobalFuncFromCodeObj(&names));assert(names==prior);
}
'''


class ModuleSymbolsTest(unittest.TestCase):
    def test_actual_enumeration(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            target = source / 'rocclr/device/devprogram.cpp'
            target.parent.mkdir(parents=True)
            target.write_bytes((ROOT / '.deps/sources/clr/rocclr/device/devprogram.cpp').read_bytes())
            patch = (ROOT / 'ports/clr/ps5-device-aot.patch').read_text().split('--- a/rocclr/platform/')[0]
            subprocess.run(['patch','--batch','--fuzz=0','-p1'], input=patch,
                           text=True,cwd=source,check=True,capture_output=True)
            self.assertEqual(digest(target), BEFORE)
            prepare(source)
            self.assertEqual(digest(target), AFTER)
            prepare(source)
            body = target.read_text().split('bool Program::getGlobalFuncFromCodeObj(')[1].split('\n}\n',1)[0]
            unit = source / 'symbols.cpp'
            unit.write_text(HARNESS+'bool Program::getGlobalFuncFromCodeObj('+body+'\n}\n'+DRIVER)
            for enabled in (True,False):
                binary = source / ('patched' if enabled else 'upstream')
                argv=['clang++-18','-std=c++20','-O1','-g','-fsanitize=address,undefined',str(unit),'-o',str(binary)]
                if enabled: argv.append('-DWITH_PS5_DEVICE')
                subprocess.run(argv,check=True,capture_output=True)
                result=subprocess.run([str(binary)],capture_output=True,
                                      env=dict(os.environ,ASAN_OPTIONS='detect_leaks=1:halt_on_error=1',UBSAN_OPTIONS='halt_on_error=1'))
                if enabled:self.assertEqual(result.returncode,0,result.stderr.decode())
                else:self.assertNotEqual(result.returncode,0,'upstream false success was not detected')
            target.write_text(target.read_text()+'// drift\n')
            with self.assertRaises(ValueError):prepare(source)


if __name__ == '__main__':
    unittest.main()
