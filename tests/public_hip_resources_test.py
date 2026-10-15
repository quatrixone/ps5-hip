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


if __name__=='__main__':unittest.main()
