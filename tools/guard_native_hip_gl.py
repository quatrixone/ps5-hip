"""Exclude unused GL helpers when PS5HIP_CLR_NO_GRAPHICS rejects interop."""
from pathlib import Path
import argparse
import hashlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    path = args.source / 'hipamd/src/hip_gl.cpp'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '834e3783ba2b42d6ff14a4ca0ae3167cc73ce2801485f1b22f7b63aba0da9f7a'
    text = path.read_text()
    edits = [
        ('#include "platform/interop_gl.hpp"\n#include "cl_common.hpp"\n#include <GL/gl.h>\n#include <GL/glext.h>\n',
         '#if !defined(PS5HIP_CLR_NO_GRAPHICS)\n#include "platform/interop_gl.hpp"\n#include "cl_common.hpp"\n#include <GL/gl.h>\n#include <GL/glext.h>\n#endif\n'),
        ('namespace amd {\nstatic std::once_flag interopOnce;\n}\n',
         '#if !defined(PS5HIP_CLR_NO_GRAPHICS)\nnamespace amd {\nstatic std::once_flag interopOnce;\n}\n#endif\n'),
        ('namespace hip {\n// Sets up GL context association',
         'namespace hip {\n#if !defined(PS5HIP_CLR_NO_GRAPHICS)\n// Sets up GL context association'),
        ('// NOTE: This method cooresponds to OpenCL functionality in clGetGLContextInfoKHR()',
         '#endif\n// NOTE: This method cooresponds to OpenCL functionality in clGetGLContextInfoKHR()'),
        ('static inline void clearGLErrors(const amd::Context& amdContext) {',
         '#if !defined(PS5HIP_CLR_NO_GRAPHICS)\nstatic inline void clearGLErrors(const amd::Context& amdContext) {'),
        ('hipError_t hipGraphicsSubResourceGetMappedArray(',
         '#endif\nhipError_t hipGraphicsSubResourceGetMappedArray('),
    ]
    for before, after in edits:
        assert text.count(before) == 1, before
        text = text.replace(before, after)
    path.write_text(text)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == 'ead07db2ba6fd19cdabbfdaf8b7458cfdfc90062a4d5c8c31d1bc85e81bc51cc'
    print(digest)


if __name__ == '__main__':
    main()
