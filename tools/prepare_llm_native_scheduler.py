"""Derive the native scheduler's paired aligned allocation without editing upstream."""
import hashlib
from pathlib import Path

GENERATED_NAME = 'ggml-backend-native.cpp'
ALLOCATE = '    sched->context_buffer = (char *) malloc(sched->context_buffer_size);'
RELEASE = '    free(sched->context_buffer);'


def render(text):
    if text.count(ALLOCATE) != 1 or text.count(RELEASE) != 1:
        raise ValueError('upstream scheduler allocation pair changed')
    return text.replace(ALLOCATE,
        '    sched->context_buffer = (char *) ggml_aligned_malloc(sched->context_buffer_size);\n'
        '    GGML_ASSERT(sched->context_buffer != NULL);').replace(RELEASE,
        '    ggml_aligned_free(sched->context_buffer, sched->context_buffer_size);')


def provenance(source, generated):
    source, generated = Path(source), Path(generated)
    if generated.is_symlink() or generated.read_text() != render(source.read_text()):
        raise ValueError('generated native scheduler source differs')
    return {'classification': 'native-scheduler-aligned-context',
            'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'generated': str(generated), 'generated_sha256': hashlib.sha256(generated.read_bytes()).hexdigest()}
