"""Derive native-only tensor staging with paired direct-memory-backed allocation."""
import hashlib
from pathlib import Path

GENERATED_NAME = 'llama-model-loader-native.cpp'
ALLOCATE = '                    std::vector<no_init<uint8_t>> read_buf(n_size);'
REPLACEMENT = '''                    // Tensor staging must use the bounded native aligned owner, not libc heap.
                    struct native_read_buffer {
                        size_t size;
                        void * pointer;
                        explicit native_read_buffer(size_t n) : size(n), pointer(ggml_aligned_malloc(n)) {
                            if (!pointer) throw std::bad_alloc();
                        }
                        native_read_buffer(const native_read_buffer &) = delete;
                        native_read_buffer & operator=(const native_read_buffer &) = delete;
                        ~native_read_buffer() { ggml_aligned_free(pointer, size); }
                        void * data() const { return pointer; }
                    } read_buf(n_size);'''


USER_ANCHOR = """    if (files.empty()) {
        if (flags & TENSOR_SKIP_IF_VIRTUAL) {
            return nullptr;
        }
        ggml_type type = GGML_TYPE_F32;
        const int64_t tid = gguf_find_tensor(metadata, tn.str().c_str());"""
USER_GUARD = """
        // User metadata declares supplied tensors; absent optional weights stay absent.
        if (tid == -1 && (flags & TENSOR_NOT_REQUIRED)) {
            return nullptr;
        }"""


def render_user(text):
    if text.count(USER_ANCHOR) != 1 or USER_GUARD in text:
        raise ValueError('upstream user loader optional branch changed')
    return text.replace(USER_ANCHOR, USER_ANCHOR + USER_GUARD)


def render(text):
    text = render_user(text)
    anchor = '#include "ggml-alloc.h"'
    if text.count(ALLOCATE) != 1 or text.count(anchor) != 1:
        raise ValueError('upstream loader staging allocation changed')
    if '#include "ggml-impl.h"' in text or '#include <new>' in text or 'native_read_buffer' in text:
        raise ValueError('upstream loader staging includes changed')
    return text.replace(anchor, anchor + '\n#include "ggml-impl.h"\n#include <new>').replace(ALLOCATE, REPLACEMENT)


def provenance(source, generated):
    source, generated = Path(source), Path(generated)
    if generated.is_symlink() or generated.read_text() != render(source.read_text()):
        raise ValueError('generated native loader source differs')
    return {'classification': 'native-loader-aligned-staging',
            'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'generated': str(generated), 'generated_sha256': hashlib.sha256(generated.read_bytes()).hexdigest()}
