"""Run the real pinned scheduler/ggml context lifecycle with allocation interception."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import resource
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_llm_native_scheduler import ALLOCATE, render

HARNESS = r'''
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-impl.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
static size_t watched_size;
static int allocations, releases, initializations, borrowed, fallback;
static bool fail_allocation;
static std::map<void *, size_t> owned;
extern "C" void * real_ggml_aligned_malloc(size_t);
extern "C" void real_ggml_aligned_free(void *, size_t);
extern "C" void * ggml_aligned_malloc(size_t size) {

    if (size == watched_size && fail_allocation) {
        puts("INJECT_ALIGNED_FAILURE_BEFORE_CONTEXT"); fflush(stdout);
        return nullptr;
    }
    void * p = real_ggml_aligned_malloc(size);
    if (size == watched_size) {
        assert(p && !owned.count(p));
        owned[p] = size; ++allocations;
    }
    return p;
}
extern "C" void ggml_aligned_free(void * p, size_t size) {

    if (size == watched_size) {
        assert(p && owned.count(p) && owned.at(p) == size);
        owned.erase(p); ++releases;
    }
    real_ggml_aligned_free(p, size);
}
static ggml_context * observed_init(ggml_init_params params) {
    ++initializations;
    assert(!fail_allocation);
    auto ctx = ggml_init(params);
    assert(ctx && ggml_get_mem_size(ctx) == watched_size);
    void * buffer = ggml_get_mem_buffer(ctx);
    assert(owned.count(buffer));
    if (params.mem_buffer) {
        assert(buffer == params.mem_buffer); ++borrowed;
    } else {
        ++fallback;
    }
    return ctx;
}
#include "scheduler-under-test.cpp"
int main(int argc, char ** argv) {
    ggml_set_abort_callback([](const char * message) {
        assert(initializations == 0); fprintf(stderr, "%s\n", message);
    });
    assert(argc >= 2);
    const size_t capacity = strtoul(argv[1], nullptr, 10);
    assert(capacity == 64 || capacity == 1608);
    fail_allocation = argc == 3 && !strcmp(argv[2], "fail");
    auto backend = ggml_backend_cpu_init();
    assert(backend);
    auto ctx = ggml_init({1024*1024, nullptr, false});
    assert(ctx);
    auto x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
    auto y = ggml_add(ctx, x, x);
    auto graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, y);
    for (int cycle = 0; cycle < 5; ++cycle) {
        assert(owned.empty());
        auto sched = ggml_backend_sched_new(&backend, nullptr, 1, capacity, false, true);
        assert(sched && !fail_allocation);
        for (int repeat = 0; repeat < 3; ++repeat) {
            ggml_backend_sched_split_graph(sched, graph);
            assert(owned.size() == 1);
            const auto pointer = owned.begin()->first;
            const int freed = releases;
            ggml_backend_sched_reset(sched);
            assert(releases == freed && owned.count(pointer));
        }
        ggml_backend_sched_free(sched);
        assert(owned.empty() && allocations == releases);
    }
    assert(initializations == 15);
#ifdef ORIGINAL
    assert(fallback == 15 && borrowed == 0 && allocations == 15 && releases == 15);
#else
    assert(fallback == 0 && borrowed == 15 && allocations == 5 && releases == 5);
#endif
    ggml_free(ctx);
    ggml_backend_free(backend);
    printf("{\"graph_capacity\":%zu,\"context_buffer_bytes\":%zu,\"allocations\":%d,\"releases\":%d,\"context_initializations\":%d,\"borrowed\":%d,\"fallback\":%d}\n",
        capacity, watched_size, allocations, releases, initializations, borrowed, fallback);
}
'''


def run(cpu_build):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    source = ROOT / '.deps/sources/llama-client/ggml/src/ggml-backend.cpp'
    original = source.read_text()
    ggml = source.parent.parent
    lib = cpu_build.resolve() / 'cmake/bin'
    evidence = {'classification': 'host-real-scheduler-lifecycle', 'native_executed': False,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'ggml_c_sha256': hashlib.sha256((ggml / 'src/ggml.c').read_bytes()).hexdigest(),
                'generated_source_sha256': hashlib.sha256(render(original).encode()).hexdigest(),
                'libraries': {name: hashlib.sha256((lib / name).read_bytes()).hexdigest()
                              for name in ('libggml-base.so', 'libggml-cpu.so')},
                'compilers': {name: subprocess.check_output([name, '--version'], text=True).splitlines()[0]
                              for name in ('cc', 'c++')}, 'cases': {}}
    # Build generated copies only: the pinned upstream checkout remains untouched.
    with tempfile.TemporaryDirectory(prefix='ps5hip-scheduler-') as directory:
        work = Path(directory)
        (work / 'test.cpp').write_text(HARNESS)
        original_core = (ggml / 'src/ggml.c').read_text()
        core = original_core
        # Rename definitions only, so real ggml_init/free calls cross our observers.
        for signature in ('void * ggml_aligned_malloc(size_t size) {', 'void ggml_aligned_free(void * ptr, size_t size) {'):
            assert core.count(signature) == 1
            core = core.replace(signature, signature.replace('ggml_aligned_', 'real_ggml_aligned_'))
        (work / 'ggml-observed.c').write_text(core)
        subprocess.run(['cc', '-std=c11', '-D_GNU_SOURCE', '-O0', '-g',
                        '-I' + str(ggml / 'include'), '-I' + str(ggml / 'src'),
                        '-I' + str(cpu_build.resolve() / 'cmake/ggml/src'),
                        '-c', str(work / 'ggml-observed.c'), '-o', str(work / 'ggml.o')], check=True, timeout=120)
        for mode in ('original', 'paired'):
            text = original if mode == 'original' else render(original)
            allocation = ALLOCATE if mode == 'original' else '    sched->context_buffer = (char *) ggml_aligned_malloc(sched->context_buffer_size);'
            assert text.count(allocation) == 1
            text = text.replace(allocation, '    watched_size = sched->context_buffer_size;\n' +
                                ('    sched->context_buffer = nullptr;' if mode == 'original' else allocation))
            init = '    sched->ctx = ggml_init(params);'
            assert text.count(init) == 1
            text = text.replace(init, '    sched->ctx = observed_init(params);')
            (work / 'scheduler-under-test.cpp').write_text(text)
            executable = work / mode
            command = ['c++', '-std=c++17', '-O0', '-g', '-UNDEBUG', '-Wl,--export-dynamic',
                       '-I' + str(ggml / 'include'), '-I' + str(ggml / 'src'), str(work / 'test.cpp'), str(work / 'ggml.o'),
                       '-L' + str(lib), '-Wl,-rpath,' + str(lib), '-lggml-cpu', '-lggml-base', '-ldl',
                       '-o', str(executable)]
            if mode == 'original':
                command.insert(1, '-DORIGINAL')
            subprocess.run(command, check=True, timeout=120)
            for capacity in (64, 1608):
                result = subprocess.run([str(executable), str(capacity)], capture_output=True, text=True, timeout=30)
                if result.returncode:
                    raise RuntimeError(f'{mode}: {result.returncode}\n{result.stdout}\n{result.stderr}')
                evidence['cases'][f'{mode}_{capacity}'] = json.loads(result.stdout)
                if mode == 'paired':
                    failure = subprocess.run([str(executable), str(capacity), 'fail'], capture_output=True, text=True, timeout=30)
                    assert failure.returncode == -signal.SIGABRT, failure.stderr
                    assert failure.stdout.strip() == 'INJECT_ALIGNED_FAILURE_BEFORE_CONTEXT', failure.stdout
                    assert 'sched->context_buffer != NULL' in failure.stderr, failure.stderr
                    evidence['cases'][f'paired_allocation_failure_{capacity}'] = {'signal': 'SIGABRT', 'before_context_init': True}
    assert source.read_text() == original
    assert (ggml / 'src/ggml.c').read_text() == original_core
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-build', type=Path, default=ROOT / 'build/llm-cpu-reference')
    run(parser.parse_args().cpu_build)
