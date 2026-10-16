"""Exercise backend accounting with real GGML callbacks and injected HIP results."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CASES = ('normal', 'in-flight', 'concurrent', 'no-return', 'bounds', 'init-failure',
         'count-failure', 'allocation-failure', 'buffer-init-failure', 'cleanup-failure',
         'h2d-failure', 'd2h-failure', 'd2d-failure', 'fill-failure', 'clear-failure',
         'sync-failure', 'free-failure')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-build', type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    ggml = ROOT / '.deps/sources/llama-client/ggml'
    lib = args.cpu_build.resolve() / 'cmake/bin'
    sources = [ROOT / 'ports/ggml/ps5_backend.cpp', ROOT / 'ports/ggml/ps5_backend.h',
               Path(__file__), Path(__file__).with_suffix('.cpp'), lib / 'libggml-base.so',
               *[ROOT/'ports/ggml'/n for n in ('model_prefix.h','prefix_calls.h','prefix_dispatch.h','prefix_qualification.h')]]
    def identities():
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    before = identities()
    with tempfile.TemporaryDirectory(prefix='ps5hip-call-coverage-') as temporary:
        executable = Path(temporary) / 'test'
        subprocess.run(['c++', '-std=c++17', '-O0', '-g', '-UNDEBUG', '-pthread',
            '-D__HIP_PLATFORM_AMD__', '-I' + str(ROOT / 'ports/ggml'),
            '-I' + str(ggml / 'include'), '-I' + str(ggml / 'src'),
            '-I' + str(ROOT / '.deps/sources/hip-headers/include'),
            '-I' + str(ROOT / '.deps/sources/clr/hipamd/include'), '-I' + str(ROOT / 'ports/hip'),
            str(sources[0]), str(Path(__file__).with_suffix('.cpp')),
            '-Wl,--wrap=ggml_backend_buffer_init', '-L' + str(lib), '-Wl,-rpath,' + str(lib),
            '-lggml-base', '-o', str(executable)], check=True, timeout=120)
        for case in CASES:
            result = subprocess.run([executable, case], capture_output=True, text=True, timeout=30)
            if result.returncode:
                raise RuntimeError(f'{case}: {result.returncode}\n{result.stdout}\n{result.stderr}')
    assert identities() == before, 'test inputs changed'
    print(json.dumps(dict(status='PASS', cases=CASES, inputs=before,
                          hip_transport='mock', native_executed=False), indent=2))


if __name__ == '__main__':
    main()
