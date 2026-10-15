"""Compile genuine HIP aggregates and observe their real compiler host stubs.

The observer is test instrumentation, not a HIP runtime or GPU execution test.
Set PS5HIP_AGGREGATE_OUTPUT to a fresh directory to retain commands and artifacts.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def metadata(text):
    result = {}
    for block in text.split('  - .args:')[1:]:
        name = re.search(r'^    \.name:\s*(\S+)', block, re.M).group(1)
        args = block.split('    .group_segment_fixed_size:', 1)[0]
        fields = [tuple((int(off), int(size), kind)) for off, size, kind in re.findall(
            r'\.offset:\s*(\d+)\s+\.size:\s*(\d+)\s+\.value_kind:\s*(\S+)', args)]
        result[name] = dict(arguments=fields, size=int(re.search(r'\.kernarg_segment_size:\s*(\d+)', block).group(1)),
                            alignment=int(re.search(r'\.kernarg_segment_align:\s*(\d+)', block).group(1)))
    return result


def compiler_flags():
    inc = ['-I'+str(ROOT/p) for p in ('.deps/sources/hip-headers/include', '.deps/sources/clr/hipamd/include', 'ports/hip')]
    flags = ['-x', 'hip', '-nogpuinc', '-nogpulib', '-B/usr/lib/llvm-18/bin',
             '--offload-arch=gfx1013', '-mcode-object-version=5', '-mno-wavefrontsize64',
             '-D__HIP_PLATFORM_AMD__', '-std=c++17', '-O2', '-fPIC', *inc,
             '-include', '__clang_hip_runtime_wrapper.h', '-c', str(ROOT/'tests/hip_aggregate_fixture.hip')]
    return inc, flags


class HipAggregateCompilerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        requested = os.environ.get('PS5HIP_AGGREGATE_OUTPUT')
        cls.temporary = None
        if requested:
            cls.output = Path(requested).resolve()
            cls.output.mkdir(parents=True, exist_ok=False)
        else:
            cls.temporary = tempfile.TemporaryDirectory(prefix='ps5hip-aggregate-')
            cls.output = Path(cls.temporary.name)
        cls.commands = []
        cls.inputs = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            ROOT/'tests/hip_aggregate_fixture.h', ROOT/'tests/hip_aggregate_fixture.hip',
            ROOT/'tests/hip_aggregate_stub_observer.cpp', Path(__file__).resolve())}
        inc, flags = compiler_flags()
        for name in ('clang++-18','ld.lld-18','llvm-readobj-18','llvm-nm-18','clang-offload-bundler-18'):
            found=shutil.which(name)
            if not found: raise RuntimeError('missing compiler tool: '+name)
            cls.inputs[found]=hashlib.sha256(Path(found).read_bytes()).hexdigest()
        for p in (Path(sys.executable),ROOT/'tools/ps5-clangxx',ROOT/'.deps/sdk/bin/prospero-clang++',ROOT/'.deps/sdk/bin/clang++'):
            cls.inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        for target in ('linux','ps5'):
            compiler='clang++-18' if target=='linux' else str(ROOT/'tools/ps5-clangxx')
            for side in ('host','device'):
                dep=cls.output/(target+'-'+side+'.d')
                extra=[] if target=='linux' else ['-nostdlib','-Xarch_device','-fno-short-wchar']
                cls.run_command(target+'-'+side+'-dependencies',[compiler,*flags,*extra,'--cuda-'+side+'-only','-M','-MF',str(dep)])
                for name in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):
                    p=Path(name).resolve();cls.inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        cls.run_command('compiler-version', ['clang++-18', '--version'])
        cls.run_command('linux-object', ['clang++-18', *flags, '-o', str(cls.output/'fixture.o')])
        cls.run_command('device', ['clang++-18', *flags, '--cuda-device-only', '--no-gpu-bundle-output',
                                   '-o', str(cls.output/'fixture.hsaco')])
        cls.run_command('ps5-object', [str(ROOT/'tools/ps5-clangxx'), *flags, '-nostdlib',
                                       '-Xarch_device', '-fno-short-wchar', '-o', str(cls.output/'ps5-fixture.o')])
        cls.run_command('observer', ['clang++-18', '-std=c++17', '-D__HIP_PLATFORM_AMD__', *inc,
                                     str(ROOT/'tests/hip_aggregate_stub_observer.cpp'), str(cls.output/'fixture.o'),
                                     '-o', str(cls.output/'observer')])
        cls.observed = cls.run_command('observe', [str(cls.output/'observer')])
        cls.notes = cls.run_command('notes', ['llvm-readobj-18', '--notes', str(cls.output/'fixture.hsaco')])
        cls.undefined = cls.run_command('undefined', ['llvm-nm-18', '--undefined-only', str(cls.output/'fixture.hsaco')])
        for name, digest in cls.inputs.items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('fixture source changed during compilation: '+name)
        (cls.output/'inputs.json').write_text(json.dumps(cls.inputs, indent=2)+'\n')

    @classmethod
    def run_command(cls, label, argv):
        try:
            p = subprocess.run(argv, cwd=ROOT, text=True, capture_output=True, timeout=120)
        except subprocess.TimeoutExpired as error:
            cls.commands.append(dict(label=label, argv=argv, timed_out=True))
            (cls.output/'commands.json').write_text(json.dumps(cls.commands, indent=2)+'\n')
            raise RuntimeError(label+' exceeded 120 seconds') from error
        (cls.output/(label+'.stdout')).write_text(p.stdout)
        (cls.output/(label+'.stderr')).write_text(p.stderr)
        cls.commands.append(dict(label=label, argv=argv, exit_code=p.returncode))
        (cls.output/'commands.json').write_text(json.dumps(cls.commands, indent=2)+'\n')
        if p.returncode:
            raise RuntimeError(label+': '+p.stderr[-4000:])
        return p.stdout

    @classmethod
    def tearDownClass(cls):
        if cls.temporary:
            cls.temporary.cleanup()

    def test_real_device_metadata_and_alignment_gaps(self):
        kernels = metadata(self.notes)
        self.assertEqual(set(kernels), {'aggregate_shapes', 'pointer_aggregate_shape'})
        self.assertEqual(kernels['aggregate_shapes'], dict(arguments=[
            (0, 8, 'global_buffer'), (8, 4, 'by_value'), (12, 12, 'by_value'),
            (32, 16, 'by_value'), (48, 128, 'by_value'), (176, 4, 'by_value')], size=180, alignment=16))
        # The 8-byte inter-argument gap [24,32) is not argument data. The 128-byte
        # object is aligned to 8 bytes, not to its size (offset 48 % 128 != 0).
        self.assertEqual(kernels['pointer_aggregate_shape'], dict(arguments=[
            (0, 8, 'global_buffer'), (8, 16, 'by_value')], size=24, alignment=8))
        self.assertFalse(self.undefined.strip())
        self.assertIn('amdgcn-amd-amdhsa--gfx1013', self.notes)

    def test_real_host_stub_member_bytes_and_padding_observation(self):
        self.assertIn('compiler aggregate stub observation PASS; no GPU execution', self.observed)
        rows = {(int(call), int(arg)): (int(size), bytes.fromhex(data)) for call, arg, size, data
                in re.findall(r'^arg (\d+) (\d+) (\d+) ([0-9a-f]+)$', self.observed, re.M)}
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0, 2][1].hex(), '4433221188776655ccbbaa99')
        self.assertEqual(rows[0, 3][1].hex(), '8070605040302010d4c3b2a1f6e50718')
        self.assertEqual(rows[0, 4][0], 128)
        # All non-padding bytes were independently checked by the observer.
        # Padding [12,16) is retained as observed evidence, not a portable value.
        self.assertEqual(len(rows[0, 4][1][12:16]), 4)
        self.assertEqual(rows[1, 1][1][:8], rows[1, 0][1])
        self.assertEqual(int.from_bytes(rows[1, 1][1][8:], 'little'), 37)


def write_receipt():
    cls=HipAggregateCompilerTest
    for name,digest in cls.inputs.items():
        path=Path(name) if Path(name).is_absolute() else ROOT/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest: raise RuntimeError('fixture input changed: '+name)
    record=dict(classification='public-hip-genuine-aggregate-compiler-fixture',inputs=cls.inputs,
                artifacts={str(p.relative_to(cls.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in cls.output.rglob('*') if p.is_file()},
                tests_passed=2,gpu_executed=False)
    (cls.output/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')


if __name__ == '__main__':
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(HipAggregateCompilerTest))
    if result.wasSuccessful() and os.environ.get('PS5HIP_AGGREGATE_OUTPUT'): write_receipt()
    sys.exit(0 if result.wasSuccessful() else 1)
