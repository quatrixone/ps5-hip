"""Bounded static audit of the vocabulary-phase native client application."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'dependencies.json').is_file())
sys.path.insert(0,str(ROOT/"tools"))
sys.path.insert(0,str(ROOT/'tests'))
from audit_public_hip_rmsnorm_native import (audit_text,load_regions,require,symbol_values,blocks,section,
    import_slots,bound_calls,symbol_calls,ordered_calls)
from public_hip_imports_test import verify as verify_imports
from native_import_libraries_test import read_imports,nid
WRAPPER='__wrap_ggml_backend_sched_graph_compute_async'
SUPPORTS='_ZN2m0L8supportsEP19ggml_backend_devicePK11ggml_tensor'
DEVICE='_ZN2m013device_handleEv'
# Reviewed LLVM18 O0 native wrapper: every nonempty graph reaches the false
# supports predicate or an earlier rejection. Only the loop exit can forward.
# Preserve all instruction operands/branch edges except relocated RIP addresses.
WRAPPER_TEMPLATE='pushq %rbp\nmovq %rsp, %rbp\nsubq $0x60, %rsp\nmovq %rdi, -0x10(%rbp)\nmovq %rsi, -0x18(%rbp)\nmovq -0x18(%rbp), %rax\nmovq %rax, RIP\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovl $0x0, -0x1c(%rbp)\nmovl -0x1c(%rbp), %eax\nmovl %eax, -0x34(%rbp)\nmovq -0x18(%rbp), %rdi\ncall\nmovl %eax, %ecx\nmovl -0x34(%rbp), %eax\ncmpl %ecx, %eax\njge 142\nmovq -0x18(%rbp), %rdi\nmovl -0x1c(%rbp), %esi\ncall\nmovq %rax, -0x28(%rbp)\nmovq -0x10(%rbp), %rdi\nmovq -0x28(%rbp), %rsi\ncall\nmovq %rax, -0x30(%rbp)\ncmpq $0x0, -0x30(%rbp)\nje b8\nmovq -0x30(%rbp), %rdi\ncall\nmovq %rax, -0x40(%rbp)\ncall\nmovq %rax, %rcx\nmovq -0x40(%rbp), %rax\ncmpq %rcx, %rax\njne b8\ncall\nmovq %rax, %rdi\nmovq -0x28(%rbp), %rsi\ncall\ntestb $0x1, %al\njne 12f\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovl -0x1c(%rbp), %eax\nmovl %eax, -0x4c(%rbp)\nmovq -0x28(%rbp), %rdi\ncall\nmovq %rax, -0x48(%rbp)\ncmpq $0x0, -0x30(%rbp)\nje f9\nmovq -0x30(%rbp), %rdi\ncall\nmovq %rax, -0x58(%rbp)\njmp 109\nleaq RIP, %rax\nmovq %rax, -0x58(%rbp)\njmp 109\nmovq -0x48(%rbp), %rdx\nmovl -0x4c(%rbp), %esi\nmovq -0x58(%rbp), %rcx\nleaq RIP, %rdi\nmovb $0x0, %al\ncall\nmovl $0xffffffff, -0x4(%rbp)\njmp 162\njmp 134\nmovl -0x1c(%rbp), %eax\naddl $0x1, %eax\nmovl %eax, -0x1c(%rbp)\njmp 31\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovq -0x10(%rbp), %rdi\nmovq -0x18(%rbp), %rsi\ncall\nmovl %eax, -0x4(%rbp)\nmovl -0x4(%rbp), %eax\naddq $0x60, %rsp\npopq %rbp\nretq\n'
LAYER_HOOK='_ZN2m0L20qualify_actual_layerEP11ggml_cgraph'
LAYER_WRAPPER_TEMPLATE='pushq %rbp\nmovq %rsp, %rbp\nsubq $0x60, %rsp\nmovq %rdi, -0x10(%rbp)\nmovq %rsi, -0x18(%rbp)\nmovq -0x18(%rbp), %rax\nmovq %rax, RIP\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\ncmpq $0x0, RIP\nje 41\nmovq -0x18(%rbp), %rdi\ncall\ncmpq $0x0, RIP\nje 58\nmovq -0x18(%rbp), %rdi\ncall\nmovl $0x0, -0x1c(%rbp)\nmovl -0x1c(%rbp), %eax\nmovl %eax, -0x34(%rbp)\nmovq -0x18(%rbp), %rdi\ncall\nmovl %eax, %ecx\nmovl -0x34(%rbp), %eax\ncmpl %ecx, %eax\njge 170\nmovq -0x18(%rbp), %rdi\nmovl -0x1c(%rbp), %esi\ncall\nmovq %rax, -0x28(%rbp)\nmovq -0x10(%rbp), %rdi\nmovq -0x28(%rbp), %rsi\ncall\nmovq %rax, -0x30(%rbp)\ncmpq $0x0, -0x30(%rbp)\nje e6\nmovq -0x30(%rbp), %rdi\ncall\nmovq %rax, -0x40(%rbp)\ncall\nmovq %rax, %rcx\nmovq -0x40(%rbp), %rax\ncmpq %rcx, %rax\njne e6\ncall\nmovq %rax, %rdi\nmovq -0x28(%rbp), %rsi\ncall\ntestb $0x1, %al\njne 15d\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovl -0x1c(%rbp), %eax\nmovl %eax, -0x4c(%rbp)\nmovq -0x28(%rbp), %rdi\ncall\nmovq %rax, -0x48(%rbp)\ncmpq $0x0, -0x30(%rbp)\nje 127\nmovq -0x30(%rbp), %rdi\ncall\nmovq %rax, -0x58(%rbp)\njmp 137\nleaq RIP, %rax\nmovq %rax, -0x58(%rbp)\njmp 137\nmovq -0x48(%rbp), %rdx\nmovl -0x4c(%rbp), %esi\nmovq -0x58(%rbp), %rcx\nleaq RIP, %rdi\nmovb $0x0, %al\ncall\nmovl $0xffffffff, -0x4(%rbp)\njmp 190\njmp 162\nmovl -0x1c(%rbp), %eax\naddl $0x1, %eax\nmovl %eax, -0x1c(%rbp)\njmp 5f\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovq -0x10(%rbp), %rdi\nmovq -0x18(%rbp), %rsi\ncall\nmovl %eax, -0x4(%rbp)\nmovl -0x4(%rbp), %eax\naddq $0x60, %rsp\npopq %rbp\nretq\n'
LAYER_SOURCES=['ports/ggml/native_layer.h','ports/ggml/layer_qualification.h','ports/ggml/layer_dispatch.h','ports/ggml/model_layer.h']
PREFIX_HOOK='_ZN2m0L21qualify_actual_prefixEP11ggml_cgraph'
PREFIX_WRAPPER_TEMPLATE='pushq %rbp\nmovq %rsp, %rbp\nsubq $0x60, %rsp\nmovq %rdi, -0x10(%rbp)\nmovq %rsi, -0x18(%rbp)\nmovq -0x18(%rbp), %rax\nmovq %rax, RIP\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\ncmpq $0x0, RIP\nje 41\nmovq -0x18(%rbp), %rdi\ncall\nmovl $0x0, -0x1c(%rbp)\nmovl -0x1c(%rbp), %eax\nmovl %eax, -0x34(%rbp)\nmovq -0x18(%rbp), %rdi\ncall\nmovl %eax, %ecx\nmovl -0x34(%rbp), %eax\ncmpl %ecx, %eax\njge 159\nmovq -0x18(%rbp), %rdi\nmovl -0x1c(%rbp), %esi\ncall\nmovq %rax, -0x28(%rbp)\nmovq -0x10(%rbp), %rdi\nmovq -0x28(%rbp), %rsi\ncall\nmovq %rax, -0x30(%rbp)\ncmpq $0x0, -0x30(%rbp)\nje cf\nmovq -0x30(%rbp), %rdi\ncall\nmovq %rax, -0x40(%rbp)\ncall\nmovq %rax, %rcx\nmovq -0x40(%rbp), %rax\ncmpq %rcx, %rax\njne cf\ncall\nmovq %rax, %rdi\nmovq -0x28(%rbp), %rsi\ncall\ntestb $0x1, %al\njne 146\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovl -0x1c(%rbp), %eax\nmovl %eax, -0x4c(%rbp)\nmovq -0x28(%rbp), %rdi\ncall\nmovq %rax, -0x48(%rbp)\ncmpq $0x0, -0x30(%rbp)\nje 110\nmovq -0x30(%rbp), %rdi\ncall\nmovq %rax, -0x58(%rbp)\njmp 120\nleaq RIP, %rax\nmovq %rax, -0x58(%rbp)\njmp 120\nmovq -0x48(%rbp), %rdx\nmovl -0x4c(%rbp), %esi\nmovq -0x58(%rbp), %rcx\nleaq RIP, %rdi\nmovb $0x0, %al\ncall\nmovl $0xffffffff, -0x4(%rbp)\njmp 179\njmp 14b\nmovl -0x1c(%rbp), %eax\naddl $0x1, %eax\nmovl %eax, -0x1c(%rbp)\njmp 48\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovq -0x10(%rbp), %rdi\nmovq -0x18(%rbp), %rsi\ncall\nmovl %eax, -0x4(%rbp)\nmovl -0x4(%rbp), %eax\naddq $0x60, %rsp\npopq %rbp\nretq\n'
PREFIX_SOURCES=['ports/ggml/native_prefix.h','ports/ggml/prefix_qualification.h','ports/ggml/prefix_dispatch.h','ports/ggml/prefix_calls.h','ports/ggml/model_prefix.h']
PREFIX_DISPATCH_EDGES={'_ZN13ps5hip_prefix8Dispatch10initializeERKNSt3__15arrayIPKcLm5EEE': ['_ZN13ps5hip_prefix8Dispatch2okE10hipError_tPKc', '_ZN13ps5hip_prefix8Dispatch4callIZNS0_10initializeERKNSt3__15arrayIPKcLm5EEEEUlvE0_EEvNS_4CallEmS5_T_', '_ZN13ps5hip_prefix8Dispatch4callIZNS0_10initializeERKNSt3__15arrayIPKcLm5EEEEUlvE1_EEvNS_4CallEmS5_T_', '_ZN13ps5hip_prefix8Dispatch4callIZNS0_10initializeERKNSt3__15arrayIPKcLm5EEEEUlvE_EEvNS_4CallEmS5_T_'], '_ZN13ps5hip_prefix8Dispatch8shutdownEv': ['_ZN13ps5hip_prefix8Dispatch4callIZNS0_8shutdownEvEUlvE0_EEvNS_4CallEmPKcT_', '_ZN13ps5hip_prefix8Dispatch4callIZNS0_8shutdownEvEUlvE1_EEvNS_4CallEmPKcT_', '_ZN13ps5hip_prefix8Dispatch4callIZNS0_8shutdownEvEUlvE_EEvNS_4CallEmPKcT_'], '_ZN13ps5hip_prefix8Dispatch13execute_graphEP11ggml_cgraph': ['_ZN13ps5hip_prefix8Dispatch4callIZNS0_13execute_graphEP11ggml_cgraphEUlvE_EEvNS_4CallEmPKcT_', '_ZN13ps5hip_prefix8Dispatch7executeEP11ggml_tensor', '_ZN13ps5hip_prefix8Dispatch8disjointEPK11ggml_tensorS3_', '_ZNK13ps5hip_prefix8Dispatch5admitEPK11ggml_tensor'], '_ZNK13ps5hip_prefix8Dispatch5admitEPK11ggml_tensor': ['_ZN13ps5hip_prefix8Dispatch8disjointEPK11ggml_tensorS3_', '_ZNK13ps5hip_prefix8Dispatch5validEPK11ggml_tensor9ggml_typeii'], '_ZN13ps5hip_prefix8Dispatch4callIZNS0_13execute_graphEP11ggml_cgraphEUlvE_EEvNS_4CallEmPKcT_': ['_ZN13ps5hip_prefix8Dispatch2okE10hipError_tPKc', '_ZZN13ps5hip_prefix8Dispatch13execute_graphEP11ggml_cgraphENKUlvE_clEv'], '_ZN13ps5hip_prefix8Dispatch7executeEP11ggml_tensor': ['_ZN13ps5hip_prefix8Dispatch4callIZNS0_7executeEP11ggml_tensorEUlvE_EEvNS_4CallEmPKcT_', '_ZN13ps5hip_prefix8Dispatch6launchEjjjjjjPPv', '_ZNK13ps5hip_prefix8Dispatch5admitEPK11ggml_tensor', '_ZNK13ps5hip_prefix8Dispatch5validEPK11ggml_tensor9ggml_typeii'], '_ZZN13ps5hip_prefix8Dispatch13execute_graphEP11ggml_cgraphENKUlvE_clEv': ['hipMemcpy'], '_ZN13ps5hip_prefix8Dispatch4callIZNS0_7executeEP11ggml_tensorEUlvE_EEvNS_4CallEmPKcT_': ['_ZN13ps5hip_prefix8Dispatch2okE10hipError_tPKc', '_ZZN13ps5hip_prefix8Dispatch7executeEP11ggml_tensorENKUlvE_clEv'], '_ZN13ps5hip_prefix8Dispatch6launchEjjjjjjPPv': ['_ZN13ps5hip_prefix8Dispatch4callIZNS0_6launchEjjjjjjPPvEUlvE0_EEvNS_4CallEmPKcT_', '_ZN13ps5hip_prefix8Dispatch4callIZNS0_6launchEjjjjjjPPvEUlvE_EEvNS_4CallEmPKcT_'], '_ZZN13ps5hip_prefix8Dispatch7executeEP11ggml_tensorENKUlvE_clEv': ['hipMemcpy'], '_ZN13ps5hip_prefix8Dispatch4callIZNS0_6launchEjjjjjjPPvEUlvE_EEvNS_4CallEmPKcT_': ['_ZN13ps5hip_prefix8Dispatch2okE10hipError_tPKc', '_ZZN13ps5hip_prefix8Dispatch6launchEjjjjjjPPvENKUlvE_clEv'], '_ZN13ps5hip_prefix8Dispatch4callIZNS0_6launchEjjjjjjPPvEUlvE0_EEvNS_4CallEmPKcT_': ['_ZN13ps5hip_prefix8Dispatch2okE10hipError_tPKc', '_ZZN13ps5hip_prefix8Dispatch6launchEjjjjjjPPvENKUlvE0_clEv'], '_ZZN13ps5hip_prefix8Dispatch6launchEjjjjjjPPvENKUlvE_clEv': ['hipModuleLaunchKernel'], '_ZZN13ps5hip_prefix8Dispatch6launchEjjjjjjPPvENKUlvE0_clEv': ['hipDeviceSynchronize'], '_ZN13ps5hip_prefix8Dispatch4callIZNS0_10initializeERKNSt3__15arrayIPKcLm5EEEEUlvE_EEvNS_4CallEmS5_T_': ['_ZN13ps5hip_prefix8Dispatch2okE10hipError_tPKc', '_ZZN13ps5hip_prefix8Dispatch10initializeERKNSt3__15arrayIPKcLm5EEEENKUlvE_clEv'], '_ZN13ps5hip_prefix8Dispatch4callIZNS0_10initializeERKNSt3__15arrayIPKcLm5EEEEUlvE0_EEvNS_4CallEmS5_T_': ['_ZN13ps5hip_prefix8Dispatch2okE10hipError_tPKc', '_ZZN13ps5hip_prefix8Dispatch10initializeERKNSt3__15arrayIPKcLm5EEEENKUlvE0_clEv'], '_ZN13ps5hip_prefix8Dispatch4callIZNS0_10initializeERKNSt3__15arrayIPKcLm5EEEEUlvE1_EEvNS_4CallEmS5_T_': ['_ZN13ps5hip_prefix8Dispatch2okE10hipError_tPKc', '_ZZN13ps5hip_prefix8Dispatch10initializeERKNSt3__15arrayIPKcLm5EEEENKUlvE1_clEv'], '_ZZN13ps5hip_prefix8Dispatch10initializeERKNSt3__15arrayIPKcLm5EEEENKUlvE_clEv': ['hipModuleLoad'], '_ZZN13ps5hip_prefix8Dispatch10initializeERKNSt3__15arrayIPKcLm5EEEENKUlvE0_clEv': ['hipModuleGetFunction'], '_ZZN13ps5hip_prefix8Dispatch10initializeERKNSt3__15arrayIPKcLm5EEEENKUlvE1_clEv': ['hipMalloc'], '_ZN13ps5hip_prefix8Dispatch4callIZNS0_8shutdownEvEUlvE_EEvNS_4CallEmPKcT_': ['_ZN13ps5hip_prefix8Dispatch2okE10hipError_tPKc', '_ZZN13ps5hip_prefix8Dispatch8shutdownEvENKUlvE_clEv'], '_ZN13ps5hip_prefix8Dispatch4callIZNS0_8shutdownEvEUlvE0_EEvNS_4CallEmPKcT_': ['_ZN13ps5hip_prefix8Dispatch2okE10hipError_tPKc', '_ZZN13ps5hip_prefix8Dispatch8shutdownEvENKUlvE0_clEv'], '_ZN13ps5hip_prefix8Dispatch4callIZNS0_8shutdownEvEUlvE1_EEvNS_4CallEmPKcT_': ['_ZN13ps5hip_prefix8Dispatch2okE10hipError_tPKc', '_ZZN13ps5hip_prefix8Dispatch8shutdownEvENKUlvE1_clEv'], '_ZZN13ps5hip_prefix8Dispatch8shutdownEvENKUlvE_clEv': ['hipDeviceSynchronize'], '_ZZN13ps5hip_prefix8Dispatch8shutdownEvENKUlvE0_clEv': ['hipFree'], '_ZZN13ps5hip_prefix8Dispatch8shutdownEvENKUlvE1_clEv': ['hipModuleUnload']}
SOURCES=['tools/audit_public_hip_llm_client_native.py','tests/llm_client_native_audit_test.py',
 'tools/audit_public_hip_rmsnorm_native.py','tools/audit_public_hip_embedded_native.py',
 'tools/probe_host_registration.py','tools/native_multi_query.py','tests/fat_binary_fixture_test.py',
 'tests/public_hip_imports_test.py','tests/native_import_libraries_test.py','tools/build_native_vector_title.py',
 'tools/public_hip_llm_client_console_title.cpp','tools/public_hip_app_crt.cpp',
 'ports/ggml/ps5_backend.cpp','ports/ggml/ps5_backend.h','ports/ggml/model_gate.h',
 'tools/source_public_hip_log.py','tests/native_call_transcript_test.py']
ALLOCATOR_SOURCES=['ports/ggml/native_host_allocator.cpp','ports/ggml/native_host_allocator.h',
 'tests/llm_native_host_allocator_test.py','src/native_memory.cpp','include/ps5hip/native_memory.h',
 'tools/prepare_llm_native_scheduler.py']

# Reviewed current LLVM18 O0 full-capable wrapper; RIP targets and calls are checked separately.
FULL_WRAPPER_TEMPLATE='pushq %rbp\nmovq %rsp, %rbp\nsubq $0xc0, %rsp\nmovq %rdi, -0x10(%rbp)\nmovq %rsi, -0x18(%rbp)\nmovq -0x18(%rbp), %rax\nmovq %rax, RIP\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\ntestb $0x1, RIP\nje 334\nmovb RIP, %al\nxorb $-0x1, %al\nmovzbl %al, %edi\nandl $0x1, %edi\ncall\nmovq $0x0, -0x20(%rbp)\nmovl $0x0, -0x24(%rbp)\nmovl -0x24(%rbp), %eax\nmovl %eax, -0x6c(%rbp)\nmovq -0x18(%rbp), %rdi\ncall\nmovl %eax, %ecx\nmovl -0x6c(%rbp), %eax\ncmpl %ecx, %eax\njge 1a4\nmovq -0x18(%rbp), %rdi\nmovl -0x24(%rbp), %esi\ncall\nmovq %rax, -0x30(%rbp)\nmovq -0x30(%rbp), %rax\nmovl 0x94(%rax), %eax\nandl $0x10, %eax\ncmpl $0x0, %eax\njne a5\njmp 196\nmovq -0x10(%rbp), %rdi\nmovq -0x30(%rbp), %rsi\ncall\nmovq %rax, -0x38(%rbp)\ncmpq $0x0, -0x38(%rbp)\nje f6\nleaq RIP, %rdi\nleaq -0x38(%rbp), %rsi\ncall\ncmpq $0x0, %rax\nje f6\ncmpq $0x0, -0x20(%rbp)\nje 18e\nmovq -0x20(%rbp), %rax\ncmpq -0x38(%rbp), %rax\nje 18e\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovq RIP, %rax\nmovq (%rax), %rax\nmovq %rax, -0x88(%rbp)\nmovl -0x24(%rbp), %eax\nmovl %eax, -0x7c(%rbp)\nmovq -0x30(%rbp), %rax\naddq $0x100, %rax\nmovq %rax, -0x78(%rbp)\ncmpq $0x0, -0x38(%rbp)\nje 14b\nmovq -0x38(%rbp), %rdi\ncall\nmovq %rax, -0x90(%rbp)\njmp 15e\nleaq RIP, %rax\nmovq %rax, -0x90(%rbp)\njmp 15e\nmovq -0x78(%rbp), %rcx\nmovl -0x7c(%rbp), %edx\nmovq -0x88(%rbp), %rdi\nmovq -0x90(%rbp), %r8\nleaq RIP, %rsi\nmovb $0x0, %al\ncall\nmovl $0xffffffff, -0x4(%rbp)\njmp 4bb\nmovq -0x38(%rbp), %rax\nmovq %rax, -0x20(%rbp)\nmovl -0x24(%rbp), %eax\naddl $0x1, %eax\nmovl %eax, -0x24(%rbp)\njmp 5c\ncmpq $0x0, -0x20(%rbp)\njne 1ca\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovl $0xffffffff, -0x4(%rbp)\njmp 4bb\nleaq RIP, %rdi\nleaq RIP, %rax\nleaq -0x20(%rbp), %rsi\ncall\nmovq (%rax), %rax\nmovq %rax, -0x40(%rbp)\nmovq -0x40(%rbp), %rdi\nmovq -0x18(%rbp), %rsi\ncall\nmovb %al, -0x91(%rbp)\njmp 1fd\nmovb -0x91(%rbp), %al\ntestb $0x1, %al\njne 257\njmp 210\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovl $0xffffffff, -0x4(%rbp)\njmp 4bb\nmovq %rax, %rcx\nmovl %edx, %eax\nmovq %rcx, -0x48(%rbp)\nmovl %eax, -0x4c(%rbp)\nmovq -0x48(%rbp), %rdi\ncall\nleaq RIP, %rdi\nleaq RIP, %rax\ncall\njmp 255\nud2\nmovb $0x1, RIP\nmovl RIP, %eax\nincl %eax\nmovl %eax, RIP\nmovq -0x10(%rbp), %rdi\nmovq -0x18(%rbp), %rsi\nleaq RIP, %rax\ncall\nmovl %eax, -0x98(%rbp)\njmp 288\nmovl -0x98(%rbp), %eax\nmovl %eax, -0x50(%rbp)\nmovq -0x40(%rbp), %rdi\nmovl -0x50(%rbp), %eax\ntestl %eax, %eax\nsete %al\nmovzbl %al, %esi\ncall\nmovb %al, -0x99(%rbp)\njmp 2b0\nmovb -0x99(%rbp), %al\nandb $0x1, %al\nmovb %al, -0x51(%rbp)\nmovb $0x0, RIP\ntestb $0x1, -0x51(%rbp)\njne 2e7\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovl $0xffffffff, -0x4(%rbp)\njmp 4bb\nmovq RIP, %rax\naddq $0x1, %rax\nmovq %rax, RIP\nmovq RIP, %rax\naddq $0x31e, %rax\nmovq %rax, RIP\nmovl $0x0, -0x4(%rbp)\njmp 4bb\nmovq %rax, %rcx\nmovl %edx, %eax\nmovq %rcx, -0x48(%rbp)\nmovl %eax, -0x4c(%rbp)\ncall\njmp 32f\njmp 4c7\ncmpq $0x0, RIP\nje 34b\nmovq -0x18(%rbp), %rdi\ncall\ncmpq $0x0, RIP\nje 362\nmovq -0x18(%rbp), %rdi\ncall\nmovl $0x0, -0x58(%rbp)\nmovl -0x58(%rbp), %eax\nmovl %eax, -0xa0(%rbp)\nmovq -0x18(%rbp), %rdi\ncall\nmovl %eax, %ecx\nmovl -0xa0(%rbp), %eax\ncmpl %ecx, %eax\njge 49b\nmovq -0x18(%rbp), %rdi\nmovl -0x58(%rbp), %esi\ncall\nmovq %rax, -0x60(%rbp)\nmovq -0x10(%rbp), %rdi\nmovq -0x60(%rbp), %rsi\ncall\nmovq %rax, -0x68(%rbp)\ncmpq $0x0, -0x68(%rbp)\nje 3fc\nmovq -0x68(%rbp), %rdi\ncall\nmovq %rax, -0xa8(%rbp)\ncall\nmovq %rax, %rcx\nmovq -0xa8(%rbp), %rax\ncmpq %rcx, %rax\njne 3fc\ncall\nmovq %rax, %rdi\nmovq -0x60(%rbp), %rsi\ncall\ntestb $0x1, %al\njne 488\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovl -0x58(%rbp), %eax\nmovl %eax, -0xb4(%rbp)\nmovq -0x60(%rbp), %rdi\ncall\nmovq %rax, -0xb0(%rbp)\ncmpq $0x0, -0x68(%rbp)\nje 446\nmovq -0x68(%rbp), %rdi\ncall\nmovq %rax, -0xc0(%rbp)\njmp 459\nleaq RIP, %rax\nmovq %rax, -0xc0(%rbp)\njmp 459\nmovq -0xb0(%rbp), %rdx\nmovl -0xb4(%rbp), %esi\nmovq -0xc0(%rbp), %rcx\nleaq RIP, %rdi\nmovb $0x0, %al\ncall\nmovl $0xffffffff, -0x4(%rbp)\njmp 4bb\njmp 48d\nmovl -0x58(%rbp), %eax\naddl $0x1, %eax\nmovl %eax, -0x58(%rbp)\njmp 369\nmovl RIP, %eax\naddl $0x1, %eax\nmovl %eax, RIP\nmovq -0x10(%rbp), %rdi\nmovq -0x18(%rbp), %rsi\ncall\nmovl %eax, -0x4(%rbp)\nmovl -0x4(%rbp), %eax\naddq $0xc0, %rsp\npopq %rbp\nretq\nmovq -0x48(%rbp), %rdi\ncall\nud2\nmovq %rax, %rdi\ncall\nud2\n'
FULL_SUPPORT_TEMPLATE='movq %rdi, -0x10(%rsp)\nmovq %rsi, -0x18(%rsp)\ntestb $0x1, RIP\nje 23\ncmpq $0x0, -0x18(%rsp)\njne 2d\nmovb $0x0, -0x1(%rsp)\njmp 68\nmovq -0x18(%rsp), %rax\nmovl 0x50(%rax), %eax\nmovq %rax, -0x20(%rsp)\nsubq $0x64, %rax\nja 63\nmovq -0x20(%rsp), %rax\nleaq RIP, %rcx\nmovslq (%rcx,%rax,4), %rax\naddq %rcx, %rax\njmpq *%rax\nmovb $0x1, -0x1(%rsp)\njmp 68\nmovb $0x0, -0x1(%rsp)\nmovb -0x1(%rsp), %al\nandb $0x1, %al\nmovzbl %al, %eax\nretq\n'
FULL_WRAPPER_CALLS=['_ZN2m07requireEb', 'ggml_graph_n_nodes', 'ggml_graph_node', 'ggml_backend_sched_get_tensor_backend', '_ZNKSt3__13mapIP12ggml_backendPN11ps5hip_full8DispatchENS_4lessIS2_EENS_9allocatorINS_4pairIKS2_S5_EEEEE5countB8ne180100ERSA_', 'ggml_backend_name', 'fprintf', '_ZNSt3__13mapIP12ggml_backendPN11ps5hip_full8DispatchENS_4lessIS2_EENS_9allocatorINS_4pairIKS2_S5_EEEEE2atERSA_', '_ZN11ps5hip_full8Dispatch5beginEP11ggml_cgraph', '__cxa_begin_catch', 'ps5hip_llm_fail', 'ggml_backend_sched_graph_compute_async', '_ZN11ps5hip_full8Dispatch6finishEb', '__cxa_end_catch', '_ZN2m0L21qualify_actual_prefixEP11ggml_cgraph', '_ZN2m0L20qualify_actual_layerEP11ggml_cgraph', 'ggml_graph_n_nodes', 'ggml_graph_node', 'ggml_backend_sched_get_tensor_backend', 'ggml_backend_get_device', '_ZN2m013device_handleEv', '_ZN2m013device_handleEv', '_ZN2m0L8supportsEP19ggml_backend_devicePK11ggml_tensor', 'ggml_op_desc', 'ggml_backend_name', 'printf', 'ggml_backend_sched_graph_compute_async', '_Unwind_Resume', '__clang_call_terminate']
FULL_GLOBAL_TARGETS={23: ('_ZN2m0L10last_graphE', 0), 30: ('_ZN2m0L10gate_callsE', 0), 39: ('_ZN2m0L10gate_callsE', 0), 45: ('_ZN2m0L12full_enabledE', 0), 58: ('_ZN2m0L11full_activeE', 0), 194: ('_ZN2m0L13full_backendsE', 0), 246: ('_ZN2m0L8rejectedE', 0), 255: ('_ZN2m0L8rejectedE', 0), 431: ('_ZN2m0L8rejectedE', 0), 440: ('_ZN2m0L8rejectedE', 0), 458: ('_ZN2m0L13full_backendsE', 0), 528: ('_ZN2m0L8rejectedE', 0), 537: ('_ZN2m0L8rejectedE', 0), 599: ('_ZN2m0L11full_activeE', 0), 606: ('_ZN2m0L23scheduled_compute_callsE', 0), 614: ('_ZN2m0L23scheduled_compute_callsE', 0), 699: ('_ZN2m0L11full_activeE', 0), 716: ('_ZN2m0L8rejectedE', 0), 725: ('_ZN2m0L8rejectedE', 0), 743: ('_ZN2m0L10full_statsE', 0), 754: ('_ZN2m0L10full_statsE', 0), 761: ('_ZN2m0L10full_statsE', 8), 774: ('_ZN2m0L10full_statsE', 8), 820: ('_ZN2m0L12prefix_scopeE', 0), 843: ('_ZN2m0L11layer_scopeE', 0), 1020: ('_ZN2m0L8rejectedE', 0), 1029: ('_ZN2m0L8rejectedE', 0), 1179: ('_ZN2m0L23scheduled_compute_callsE', 0), 1188: ('_ZN2m0L23scheduled_compute_callsE', 0)}
FULL_OPS=frozenset((0,1,2,7,25,29,35,36,37,38,40,42,46,48,100))
FULL_SOURCES=["ports/ggml/generation_cases.def","ports/ggml/native_full.h","ports/ggml/generation.h","ports/ggml/full_qualification.h","ports/ggml/full_dispatch.h","ports/ggml/full_graph.h","tests/llm_native_full_test.cpp","tests/native_full_title_test.py"]
LOADER_SOURCES=['tools/prepare_llm_native_loader.py','tests/llm_native_loader_test.py']

# LLVM18 O0 native local RAII owner: preserve size/pointer ABI and null-throw control flow.
LOADER_TEMPLATES={'constructor': 'pushq %rbp\nmovq %rsp, %rbp\nsubq $0x20, %rsp\nmovq %rdi, -0x8(%rbp)\nmovq %rsi, -0x10(%rbp)\nmovq -0x8(%rbp), %rax\nmovq %rax, -0x18(%rbp)\nmovq -0x10(%rbp), %rcx\nmovq %rcx, (%rax)\nmovq -0x10(%rbp), %rdi\ncall\nmovq %rax, %rcx\nmovq -0x18(%rbp), %rax\nmovq %rcx, 0x8(%rax)\ncmpq $0x0, 0x8(%rax)\njne 72\nmovl $0x8, %edi\ncall\nmovq %rax, %rdi\nmovq %rdi, -0x20(%rbp)\nleaq RIP, %rax\ncall\nmovq -0x20(%rbp), %rdi\nleaq RIP, %rsi\nleaq RIP, %rdx\ncall\nud2\naddq $0x20, %rsp\npopq %rbp\nretq\n', 'destructor': 'pushq %rbp\nmovq %rsp, %rbp\nsubq $0x10, %rsp\nmovq %rdi, -0x8(%rbp)\nmovq -0x8(%rbp), %rax\nmovq (%rax), %rsi\nmovq 0x8(%rax), %rdi\nleaq RIP, %rax\ncall\njmp 25\naddq $0x10, %rsp\npopq %rbp\nretq\nmovq %rax, %rdi\ncall\nud2\n', 'data': 'movq %rdi, -0x8(%rsp)\nmovq -0x8(%rsp), %rax\nmovq 0x8(%rax), %rax\nretq\n'}

def loader_roles(values):
 # The native staging owner is local to load_all_data, hence its _ZZ names.
 def unique(fragment):
  found=[n for n in values if fragment in n]
  require(len(found)==1,'missing/ambiguous native loader symbol '+fragment)
  return found[0]
 load=[n for n in values if n.startswith('_ZN18llama_model_loader13load_all_data')]
 require(len(load)==1,'missing/ambiguous loader entry')
 return dict(load=load[0],constructor=unique('18native_read_bufferC2'),
             destructor=unique('18native_read_bufferD2'),data=unique('18native_read_buffer4data'))

def loader_contract(bodies,roles):
 """Bound the new synchronous non-host tensor staging pair, not libc globally."""
 load=bodies[roles['load']]
 require(len(symbol_calls(load,roles['constructor']))==1,'loader staging constructor missing/duplicated')
 require(len(symbol_calls(load,roles['destructor']))==2,'loader normal/unwind release missing/duplicated')
 require(len(symbol_calls(load,roles['data']))==3,'loader staging data uses differ')
 upload=symbol_calls(load,'ggml_backend_tensor_set')
 require(upload,'loader tensor upload missing')
 start=symbol_calls(load,roles['constructor'])[0];uses=symbol_calls(load,roles['data']);end=symbol_calls(load,roles['destructor'])[0]
 require(start<uses[0]<uses[1] and any(uses[1]<p<uses[2] for p in upload) and uses[2]<end,
         'loader staging/upload/release call order differs')
 for role,target in [('constructor','__wrap_ggml_aligned_malloc'),('destructor','__wrap_ggml_aligned_free')]:
  require(len(symbol_calls(bodies[roles[role]],target))==1,'loader wrapped allocator edge missing: '+role)
 for role in roles:
  for target in ('ggml_aligned_malloc','ggml_aligned_free','malloc','calloc','realloc','free'):
   require(not symbol_calls(bodies[roles[role]],target),'loader original allocator bypass: '+role)
 require(not re.search(r'<[^>]*vector[^>]*no_init[^>]*>',load),'original loader staging vector retained')

def loader_machine_contract(bodies,roles,bases):
 loader_contract(bodies,roles)
 for role,expected in LOADER_TEMPLATES.items():
  actual=normalized_wrapper(bodies[roles[role]],bases[role])
  # Linker GOT relaxation may retain mov or produce lea; the call binding
  # independently requires the wrapped allocator, including indirect calls.
  actual=actual.replace('movq RIP,','leaq RIP,')
  require(actual==expected,'native loader owner operands/control flow differ: '+role)
 ctor=bodies[roles['constructor']]
 ordered_calls(ctor,['__wrap_ggml_aligned_malloc','__cxa_allocate_exception','__cxa_throw'])
 require(len(symbol_calls(bodies[roles['destructor']],'__clang_call_terminate'))==1,
         'native loader destructor exception edge differs')
 lines=[line for line in bodies[roles['load']].splitlines() if re.match(r'^\s*[0-9a-f]+:',line)]
 for role in ('constructor','destructor','data'):
  for i,line in enumerate(lines):
   if '<'+roles[role]+'>' not in line:continue
   require(i>0 and re.search(r'leaq\s+-0x2b0\(%rbp\),\s*%rdi\s*$',lines[i-1]),
           'native loader staging owner argument differs: '+role)
   if role=='constructor':
    require(i>1 and re.search(r'movq\s+-0x1a0\(%rbp\),\s*%rsi\s*$',lines[i-2]),
            'native loader staging size argument differs')

def threadpool_indirect_malloc(body,target):
 # Exact observed LLVM18 O0 prefix: RIP-relative wrapper address, pool size,
 # then indirect call. Resolve the LEA using its actual next instruction address.
 instructions=[]
 for line in body.splitlines():
  m=re.match(r'^\s*([0-9a-f]+):\s+(.*)$',line)
  if m:instructions.append((int(m[1],16),m[2].split('#')[0].strip()))
 hits=[]
 for i,(address,text) in enumerate(instructions):
  if text not in ('callq\t*%rax','callq *%rax'):continue
  require(i>=2,'unbound threadpool indirect allocator')
  lea=re.fullmatch(r'leaq\s+(-?0x[0-9a-f]+)\(%rip\),\s*%rax',instructions[i-2][1])
  require(lea is not None and re.fullmatch(r'movl\s+\$0x100,\s*%edi',instructions[i-1][1]) is not None,
          'threadpool allocator register flow differs')
  require(instructions[i-1][0]+int(lea[1],16)==target,'threadpool allocator LEA target differs')
  hits.append(address)
 require(len(hits)==1,'threadpool first allocation call missing/duplicated')
 return hits

def allocator_contract(bodies,roles):
 """All selected GGML aligned pairs, not ordinary libc allocations."""
 malloc='__wrap_ggml_aligned_malloc';free='__wrap_ggml_aligned_free'
 for caller,expected,count in [
  (roles['buffer_alloc'],malloc,1),(roles['buffer_free'],free,1),
  ('ggml_init',malloc,1),('ggml_free',free,1),
  ('ggml_backend_sched_new',malloc,1),('ggml_backend_sched_free',free,1),
  ('ggml_threadpool_new_impl',malloc,1),('ggml_threadpool_free',free,2)]:
  calls=symbol_calls(bodies[caller],expected)
  if caller=='ggml_threadpool_new_impl':
   calls=list(re.finditer(r'\bcallq?\s+0x[0-9a-f]+\s+<'+re.escape(expected)+r'>',bodies[caller]))
  require(len(calls)==count,'allocator routing missing/duplicated: '+caller)
  for original in ('ggml_aligned_malloc','ggml_aligned_free'):
   require(not symbol_calls(bodies[caller],original),'original allocator pair bypass: '+caller)
 threadpool_indirect_malloc(bodies['ggml_threadpool_new_impl'],roles['malloc_address'])
 for caller in ('main',roles['finalized']):
  require(symbol_calls(bodies[caller],roles['stats']),'allocator observation missing: '+caller)
 for caller,targets in [(malloc,roles['constructors']),(free,roles['destructors']),
                         (free,[roles['close']]),(malloc,[roles['cpu']]),(free,[roles['cpu']])]:
  require(any(symbol_calls(bodies[caller],target) for target in targets),'NativeMemory ownership call missing: '+caller)
 for caller in (malloc,free):
  require(symbol_calls(bodies[caller],'ps5hip_llm_fail'),'allocator fatal path missing: '+caller)
  for target in ('ggml_aligned_malloc','ggml_aligned_free'):
   require(not symbol_calls(bodies[caller],target),'allocator unexpectedly falls back to libc pair')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalized_wrapper(body,base):
 lines=[]
 for line in body.splitlines():
  m=re.match(r'^\s*([0-9a-f]+):\s+(\w+)\s*(.*)$',line)
  if not m:continue
  _,op,arg=m.groups();arg=arg.split('#')[0].strip()
  if op == 'addr32':
   # lld preserves a six-byte GOTPCRELX call footprint as 67 e8 rel32.
   # Only a direct call is admitted; the existing exact callee/order audit remains mandatory.
   require(re.fullmatch(r'callq?\s+0x[0-9a-f]+\s+<[^<>\s]+>',arg) is not None,'unsupported addr32 instruction')
   value='call'
  elif op.startswith('call'):value='call'
  elif op.startswith('j'):
   target=re.match(r'0x([0-9a-f]+)',arg);require(target is not None,'unresolved wrapper branch')
   value=op+' '+format(int(target[1],16)-base,'x')
  else:value=op+' '+re.sub(r'(?:-?0x[0-9a-f]+)?\(%rip\)','RIP',arg)
  lines.append(value.rstrip())
 return '\n'.join(lines)+'\n'
def wrapper_contract(body,base,support_body,call_positions):
 actual=normalized_wrapper(body,base)
 require(actual in (WRAPPER_TEMPLATE,PREFIX_WRAPPER_TEMPLATE,LAYER_WRAPPER_TEMPLATE),'wrapper instruction/control-flow contract differs')
 require(normalized_wrapper(support_body,0)=='movq %rdi, -0x8(%rsp)\nmovq %rsi, -0x10(%rsp)\nxorl %eax, %eax\nandb $0x1, %al\nmovzbl %al, %eax\nretq\n','zero-op supports predicate changed')
 expected=['ggml_graph_n_nodes','ggml_graph_node','ggml_backend_sched_get_tensor_backend',
  'ggml_backend_get_device',DEVICE,DEVICE,SUPPORTS,'ggml_op_desc','ggml_backend_name','printf','ggml_backend_sched_graph_compute_async']
 if actual==PREFIX_WRAPPER_TEMPLATE:expected.insert(0,PREFIX_HOOK)
 if actual==LAYER_WRAPPER_TEMPLATE:expected[:0]=[PREFIX_HOOK,LAYER_HOOK]
 require([name for _,name in sorted(call_positions)]==expected,'wrapper call targets/order differ')
def bounded_body(body,address,size):
 require(size>0,'missing function extent')
 lines=[]
 for line in body.splitlines(keepends=True):
  match=re.match(r'^\s*([0-9a-f]+):',line)
  if match is None or address<=int(match[1],16)<address+size:lines.append(line)
 return ''.join(lines)

def load_layout(data):
 require(len(data)>=64 and data[:6]==b'\x7fELF\x02\x01','not ELF64LE')
 off=struct.unpack_from('<Q',data,32)[0];ents,count=struct.unpack_from('<HH',data,54)
 require(ents==56 and count>0 and off<=len(data) and count<= (len(data)-off)//ents,'invalid program headers')
 segments=[]
 for i in range(count):
  typ,flags,offset,va,_,filesz,memsz,align=struct.unpack_from('<IIQQQQQQ',data,off+i*ents)
  if typ==1:
   require(flags&3!=3,'writable executable segment');require(filesz<=memsz and offset<=len(data) and filesz<=len(data)-offset,'load range invalid')
   segments.append(dict(flags=flags,file_size=filesz,memory_size=memsz,address=va))
 require(segments,'missing load segments');return segments

def lifecycle_imports(imports):
 expected=('libSceLibcInternal','libSceLibcInternal')
 for name in ('atexit','_init_env','printf'):
  require(imports.get(nid(name))==expected,'lifecycle import owner differs: '+name)
 return {name:expected for name in ('atexit','_init_env','printf')}

def phase_contract(phase,model_gate,chain,bodies):
 require(phase in ('vocab','model-gate','model-prefix','model-layer','model-full'),'unknown client phase')
 if phase=='vocab':
  require(model_gate is None,'model gate retained in vocabulary phase')
  return
 require(model_gate is not None and len(chain)==7,'model gate caller chain missing')
 for caller,callee in zip(chain,chain[1:]):
  require(symbol_calls(bodies[caller],callee),'missing actual model call '+caller+' -> '+callee)
 context=bodies[chain[-2]]
 require(not symbol_calls(context,'ggml_backend_sched_graph_compute_async'),'unwrapped context compute call')

def prefix_roles(match):
 return {role:match(fragment) for role,fragment in {
  'run':'run_native_prefix','configure':'configure_prefix','disable':'disable_prefix',
  'select':'ps5hip_prefix6select','materialize':'ps5hip_prefix11materialize',
  'graph':'_ZN2m0L5graphE','readback':'NativePrefixSink8readback','writer':'prefix_write',
  'scope_constructor':'ActualPrefixQualificationC1','scope_destructor':'ActualPrefixQualificationD1',
  'backend_init':'_ZN2m0L4initE','owned':'prefix_owned','fail':'prefix_fail'}.items()}

def prefix_writer_import_calls(text,slots,reader):
 found=bound_calls(text,slots,reader)
 rows=list(re.finditer(r'^\s*([0-9a-f]+):\s*(.*?)\s*$',text,re.M))
 for i,row in enumerate(rows[:-1]):
  load=re.fullmatch(r'movq\s+(-?0x[0-9a-f]+)\(%rip\),\s*%(rax|rcx)\s+#\s+0x([0-9a-f]+)(?:\s+.*)?',row[2])
  if not load:continue
  address=int(rows[i+1][1],16)+int(load[1],16)
  if address not in slots:continue
  require(int(load[3],16)==address,'prefix import GOT comment differs')
  register=load[2];family='(?:rax|eax|ax|al|ah)' if register=='rax' else '(?:rcx|ecx|cx|cl|ch)'
  for following in rows[i+1:i+6]:
   if re.fullmatch(r'callq?\s+\*%'+register,following[2]):found.append(following.start());break
   if re.search(r'%'+family+r'(?![a-z0-9])|\b(?:callq?|j[a-z]+|loop[a-z]*|retq?)\b',following[2]):break
 return sorted(found)


def prefix_scope_aliases(text,roles,values,scope="ActualPrefixQualification"):
 # LLVM may print either ABI name for identical complete/base ctor/dtor bodies.
 for role,first,second in [('scope_constructor','C1','C2'),('scope_destructor','D1','D2')]:
  canonical=roles[role];alias=canonical.replace(scope+first,scope+second)
  require(alias!=canonical and len(values.get(canonical,()))==1 and
          values.get(alias)==values[canonical],'prefix scope aliases differ: '+canonical)
  address=next(iter(values[canonical]))
  pattern=r'((?:addr32 )?callq?\s+0x([0-9a-f]+)\s+)<('+re.escape(canonical)+'|'+re.escape(alias)+r')>'
  def normalize(match):
   require(int(match[2],16)==address,'prefix scope call address differs: '+match[3])
   return match[1]+'<'+canonical+'>'
  text=re.sub(pattern,normalize,text)
  lea=r'(\bleaq\s+(-?0x[0-9a-f]+)\(%rip\),\s*%[a-z0-9]+\s+#\s+0x([0-9a-f]+)\s+)<('+re.escape(canonical)+'|'+re.escape(alias)+r')>([^\n]*\n\s*([0-9a-f]+):)'
  def normalize_lea(match):
   require(int(match[3],16)==address and int(match[6],16)+int(match[2],16)==address,
           'prefix scope LEA address differs: '+match[4])
   return match[1]+'<'+canonical+'>'+match[5]
  text=re.sub(lea,normalize_lea,text)
 return text


def prefix_contract(bodies,roles):
 edges=dict(PREFIX_DISPATCH_EDGES)
 edges.update({
  'main':[roles['run']],
  roles['run']:[roles['configure'],'llama_model_load_from_file','llama_init_from_model','llama_process',roles['disable'],'ggml_backend_free',roles['scope_constructor'],roles['scope_destructor']],
  roles['configure']:['_ZN13ps5hip_prefix8Dispatch10initializeERKNSt3__15arrayIPKcLm5EEE'],
  roles['disable']:['_ZN13ps5hip_prefix8Dispatch8shutdownEv'],
  PREFIX_HOOK:[roles['select'],roles['materialize'],'ggml_backend_alloc_ctx_tensors','ggml_backend_tensor_get','ggml_backend_tensor_set','ggml_backend_graph_compute','ggml_backend_buffer_free','ggml_free'],
  roles['graph']:['_ZN13ps5hip_prefix8Dispatch13execute_graphEP11ggml_cgraph'],
  roles['readback']:['ggml_backend_tensor_get'],roles['fail']:['ps5hip_llm_fail']})
 for caller,targets in edges.items():
  for callee in targets:require(symbol_calls(bodies[caller],callee),'prefix call missing: '+caller+' -> '+callee)
 ordered_calls(bodies[PREFIX_HOOK],[roles['select'],roles['materialize'],'ggml_backend_alloc_ctx_tensors','ggml_backend_tensor_get','ggml_backend_tensor_set','ggml_backend_graph_compute','ggml_backend_buffer_free','ggml_free'])
 ordered_calls(bodies[roles['run']],[roles['configure'],'llama_model_load_from_file','llama_init_from_model','llama_process',roles['disable'],'ggml_backend_free'])
 for caller in [PREFIX_HOOK,roles['graph'],*PREFIX_DISPATCH_EDGES]:
  for bypass in ('ggml_backend_cpu_init','ggml_backend_sched_graph_compute_async','__real_ggml_backend_sched_graph_compute_async'):
   require(not symbol_calls(bodies[caller],bypass),'prefix CPU/scheduler fallback call')

def layer_roles(match):
 return {role:match(fragment) for role,fragment in {
  'run':'run_native_layer','configure':'configure_layer','disable':'disable_layer','reset':'reset_layer_cache',
  'select':'ps5hip_layer6select','materialize':'ps5hip_layer11materialize','weights':'ps5hip_layer12make_weights','cache':'ps5hip_layer10make_cache',
  'graph':'_ZN2m0L5graphE','readback':'NativeLayerSink8readback','dump':'NativeLayerSink4dump','writer':'prefix_write',
  'scope_constructor':'ActualLayerQualificationC1','scope_destructor':'ActualLayerQualificationD1',
  'backend_init':'_ZN2m0L4initE','owned':'layer_owned','fail':'layer_fail',
  'initialize':'_ZN12ps5hip_layer8Dispatch10initialize','shutdown':'_ZN12ps5hip_layer8Dispatch8shutdown',
  'execute':'_ZN12ps5hip_layer8Dispatch13execute_graph','preflight':'_ZN12ps5hip_layer8Dispatch9preflight',
  'launch':'_ZN12ps5hip_layer8Dispatch6launch','read':'_ZN12ps5hip_layer8Dispatch4read',
  'dispatch_reset':'_ZN12ps5hip_layer8Dispatch5reset'}.items()}

def layer_edges(r):
 return {'main':[r['run']],
  r['run']:[r['configure'],'llama_model_load_from_file','llama_init_from_model',r['reset'],'llama_process',r['disable'],'ggml_backend_free',r['scope_constructor'],r['scope_destructor']],
  r['configure']:[r['initialize']],r['disable']:[r['shutdown'],'ggml_backend_buffer_free','ggml_free'],
  r['reset']:['ggml_backend_tensor_memset',r['dispatch_reset']],
  LAYER_HOOK:[r['select'],r['weights'],r['cache'],r['materialize'],'ggml_backend_alloc_buffer','ggml_backend_tensor_alloc','ggml_backend_alloc_ctx_tensors','ggml_backend_tensor_get','ggml_backend_tensor_set','ggml_backend_tensor_memset','ggml_backend_graph_compute','ggml_backend_buffer_free','ggml_free'],
  r['graph']:[r['execute']],r['execute']:[r['preflight'],r['launch']],r['preflight']:[r['select'],r['read']],
  r['readback']:[r['dump']],r['dump']:['ggml_backend_tensor_get'],r['fail']:['ps5hip_llm_fail']}

def layer_contract(bodies,r,dispatch_names):
 edges=layer_edges(r)
 node=[n for n in dispatch_names if n.startswith('_ZN12ps5hip_layer8Dispatch12execute_nodeE')]
 if node:
  require(len(node)==1,'ambiguous layer node dispatcher')
  edges[r['execute']]=[r['preflight'],node[0]];edges[node[0]]=[r['launch']]
 for caller,targets in edges.items():
  for callee in targets:require(symbol_calls(bodies[caller],callee),'layer call missing: '+caller+' -> '+callee)
 ordered_calls(bodies[r['run']],[r['configure'],'llama_model_load_from_file','llama_init_from_model','llama_process',r['disable'],'ggml_backend_free'])
 ordered_calls(bodies[LAYER_HOOK],[r['select'],r['weights'],r['cache'],r['materialize'],'ggml_backend_graph_compute','ggml_backend_buffer_free','ggml_free'])
 # Counter-recording template lambdas must form a connected public HIP path.
 def reachable(start,target):
  pending=[start];seen=set()
  while pending:
   caller=pending.pop()
   if caller in seen:continue
   seen.add(caller)
   if symbol_calls(bodies[caller],target):return True
   pending.extend(n for n in dispatch_names if n not in seen and symbol_calls(bodies[caller],n))
  return False
 for role,apis in [('initialize',('hipModuleLoad','hipModuleGetFunction','hipMalloc')),('shutdown',('hipFree','hipModuleUnload','hipDeviceSynchronize')),('launch',('hipModuleLaunchKernel','hipDeviceSynchronize')),('read',('hipMemcpy',))]:
  for api in apis:require(reachable(r[role],api),'layer public HIP call path missing: '+role+' -> '+api)
 for caller in [LAYER_HOOK,r['graph'],*dispatch_names]:
  for bypass in ('ggml_backend_cpu_init','ggml_backend_sched_graph_compute_async','__real_ggml_backend_sched_graph_compute_async'):
   require(not symbol_calls(bodies[caller],bypass),'layer CPU/scheduler fallback call')

def full_normalized(body,base):
 # Only supports' reviewed jump-table branch has an indirect jump.
 marker='jmpq *%rax'
 transformed=re.sub(r'jmpq\s+\*%rax', 'full_indirect_jump',body)
 return normalized_wrapper(transformed,base).replace('full_indirect_jump',marker)

def rip_references(body):
 rows=list(re.finditer(r'^\s*([0-9a-f]+):\s+(.*)$',body,re.M));found=[]
 for i,row in enumerate(rows[:-1]):
  m=re.search(r'(-?0x[0-9a-f]+)?\(%rip\).*?#\s*0x([0-9a-f]+)',row[2])
  if m:
   address=int(rows[i+1][1],16)+int(m[1] or '0',16)
   require(address==int(m[2],16),'RIP reference comment/address differs')
   found.append((int(row[1],16),address,row[2]))
 return found

def full_support_contract(body,base,reader,enabled):
 require(full_normalized(body,base)==FULL_SUPPORT_TEMPLATE,'full support predicate instructions differ')
 refs=rip_references(body)
 require(len(refs)==2 and refs[0][1]==enabled and refs[0][0]==base+0xa,'full support mode flag differs')
 table=refs[1][1];raw=reader(table,101*4)[0]
 require(len(raw)==404,'full support table truncated')
 offsets=struct.unpack('<101i',raw)
 for op,offset in enumerate(offsets):
  require(table+offset==base+(0x59 if op in FULL_OPS else 0x63),'full support opcode table differs: '+str(op))
 return table

def full_wrapper_contract(body,base,calls,enabled):
 require(full_normalized(body,base)==FULL_WRAPPER_TEMPLATE,'full wrapper instruction/control-flow contract differs')
 require([n for _,n in sorted(calls)]==FULL_WRAPPER_CALLS,'full wrapper calls/order differ')
 require(any(at==base+0x2d and target==enabled for at,target,_ in rip_references(body)),'full wrapper mode flag differs')

def full_global_bindings(body,base,bindings,one):
 references={at:target for at,target,_ in rip_references(body)}
 for offset,(name,addend) in bindings.items():
  require(references.get(base+offset)==one(name)+addend,'full wrapper global binding differs: '+name)

def full_mode_store(body,enabled,value):
 stores=[text for _,target,text in rip_references(body) if target==enabled and re.match(r'movb\s+\$',text)]
 require(len(stores)==1 and re.match(r'movb\s+\$'+hex(value)+r',',stores[0]),'full mode flag store differs')

def zero_initialized_flag(reader,headers,address):
 segments=[h for h in headers if h[0]==1 and h[3]<=address<h[3]+h[6]]
 require(len(segments)==1 and segments[0][1]&2 and not segments[0][1]&1,'full flag not in writable nonexecutable memory')
 h=segments[0]
 # ELF guarantees zero-filled memory after p_filesz; stored bytes must be zero too.
 if address-h[3]<h[5]:require(reader(address,1)[0]==b'\0','full mode flag initially nonzero')

def full_roles(match):
 return {role:match(fragment) for role,fragment in {
  'run':'run_native_full','enable':'enable_full','disable':'disable_full','reset':'reset_full',
  'context_reset':'unique_ptrI13llama_contextPFvPS1_EE5reset','model_reset':'unique_ptrI11llama_modelPFvPS1_EE5reset',
  'readback':'NativeFullSink8callback','writer':'prefix_write','graph':'_ZN2m0L5graphE',
  'backend_init':'_ZN2m0L4initE','release':'_ZN2m0L7releaseE',
  'begin':'_ZN11ps5hip_full8Dispatch5beginE','finish':'_ZN11ps5hip_full8Dispatch6finishE',
  'execute':'_ZN11ps5hip_full8Dispatch13execute_graphE','initialize':'_ZN11ps5hip_full8Dispatch10initializeE',
  'shutdown':'_ZN11ps5hip_full8Dispatch8shutdownE','controls':'_ZN11ps5hip_full8Dispatch17validate_controlsE',
  'select':'_ZN11ps5hip_full6selectE','node':'_ZN12ps5hip_layer8Dispatch12execute_nodeE',
  'launch':'_ZN12ps5hip_layer8Dispatch6launchE','read':'_ZN12ps5hip_layer8Dispatch4readE',
  'dispatch_reset':'_ZN11ps5hip_full8Dispatch5resetE'}.items()}

def full_edges(r):
 return {'main':[r['run']],
  r['run']:[r['enable'],'llama_model_load_from_file','llama_init_from_model','llama_synchronize','llama_memory_clear',r['reset'],'llama_process',r['context_reset'],r['model_reset'],r['disable'],'llama_backend_free'],
  r['backend_init']:[r['initialize']],r['release']:[r['shutdown']],
  r['reset']:[r['dispatch_reset']],r['graph']:[r['execute']],
  r['begin']:[r['select']],r['execute']:[r['controls'],r['node']],
  r['node']:[r['launch']],r['controls']:[r['read']],r['readback']:['ggml_backend_tensor_get'],
  WRAPPER:[r['begin'],'ggml_backend_sched_graph_compute_async',r['finish']]}

def full_profile_operand(body,run,profile="smoke"):
 require(profile in ("smoke","generation","corpus"),"unsupported full audit profile")
 operand={"smoke":"0x1","generation":"0x2","corpus":"0x3"}[profile]
 positions=symbol_calls(body,run);require(len(positions)==1,'full helper call missing/duplicated')
 before=body[:positions[0]]
 instructions=[]
 for line in before.splitlines():
  m=re.match(r'^\s*[0-9a-f]+:\s+(.*)$',line)
  if m:instructions.append(re.sub(r'\s+',' ',m[1].split('#')[0].strip()))
 window=instructions[-14:]
 first=next((i for i,s in enumerate(window) if s=='movq %rsp, %rax'),None)
 require(first is not None and window[first:first+3]==['movq %rsp, %rax','movq %rcx, (%rax)','movl $'+operand+', 0x8(%rax)'],'full title does not pass '+profile+' ABI operand')
 # No later stack/rax rewrite may replace the reviewed seventh source argument.
 require(not any(re.search(r', (?:0x8\(%r(?:ax|sp)\)|%rax|%rsp)$',s) for s in window[first+3:]),'full profile operand overwritten')

def full_contract(bodies,r,dispatch_names,profile="smoke"):
 for caller,targets in full_edges(r).items():
  for callee in targets:require(symbol_calls(bodies[caller],callee),'full call missing: '+caller+' -> '+callee)
 ordered_calls(bodies[r['run']],[r['enable'],'llama_model_load_from_file','llama_init_from_model','llama_synchronize','llama_memory_clear',r['reset'],'llama_process',r['context_reset'],r['model_reset'],r['disable'],'llama_backend_free'])
 full_profile_operand(bodies['main'],r['run'],profile)
 # Audit all connected dispatcher helpers, including counter-recording lambdas.
 def reachable(start,target):
  pending=[start];seen=set()
  while pending:
   caller=pending.pop()
   if caller in seen:continue
   seen.add(caller)
   if symbol_calls(bodies[caller],target):return True
   pending.extend(n for n in dispatch_names if n not in seen and symbol_calls(bodies[caller],n))
  return False
 for role,apis in [('initialize',('hipModuleLoad','hipModuleGetFunction','hipMalloc')),('shutdown',('hipFree','hipModuleUnload','hipDeviceSynchronize')),('launch',('hipModuleLaunchKernel','hipDeviceSynchronize')),('read',('hipMemcpy',))]:
  for api in apis:require(reachable(r[role],api),'full public HIP path missing: '+role+' -> '+api)
 for caller in [r['graph'],*dispatch_names]:
  for bypass in ('ggml_backend_cpu_init','ggml_backend_sched_graph_compute_async','__real_ggml_backend_sched_graph_compute_async'):
   require(not symbol_calls(bodies[caller],bypass),'full CPU/scheduler fallback call')

def generation_roles(values,function_addresses):
 def select(predicate,role):
  found=[n for n,v in values.items() if len(v)==1 and v[0] in function_addresses and predicate(n)]
  require(len(found)==1,'missing/ambiguous generation '+role)
  return found[0]
 return {role:select(predicate,role) for role,predicate in {
  'helper':lambda n:n.startswith('_ZN2m019run_full_generationE'),
  'sampler':lambda n:n.startswith('_ZN17ps5hip_generation8generateI'),
  'process':lambda n:n.startswith('_ZZN17ps5hip_generation8generateI') and n.endswith('_clEi'),
  'decode':lambda n:n.startswith('_ZZN2m019run_full_generationE') and n.endswith('_clEii'),
  'eog':lambda n:n.startswith('_ZZN2m019run_full_generationE') and n.endswith('_clEi'),
  'greedy':lambda n:n.startswith('_ZN17ps5hip_generation6greedyE')}.items()}


def generation_edges(full,r):
 return {full['run']:[r['helper']],
  r['helper']:['llama_model_get_vocab','llama_tokenize','llama_synchronize','llama_memory_clear',full['reset'],r['sampler'],'llama_token_to_piece'],
  r['sampler']:[r['process'],r['eog']],r['process']:[r['decode'],r['greedy']],
  r['decode']:['llama_batch_ext_init','llama_batch_ext_add_token','llama_batch_ext_set_pos','llama_batch_ext_set_output_logits','llama_process','llama_batch_ext_free','llama_get_logits_ith'],
  r['eog']:['llama_vocab_is_eog']}


def generation_contract(bodies,full,r,scope):
 edges=generation_edges(full,r)
 for caller,targets in edges.items():
  for callee in targets:require(symbol_calls(bodies[caller],callee),'generation call missing: '+caller+' -> '+callee)
 ordered_calls(bodies[r['helper']],['llama_model_get_vocab','llama_tokenize','llama_synchronize','llama_memory_clear',full['reset'],r['sampler'],'llama_token_to_piece'])
 ordered_calls(bodies[r['process']],[r['decode'],r['greedy']])
 ordered_calls(bodies[r['decode']],edges[r['decode']])
 require(len(symbol_calls(bodies[r['sampler']],r['process']))==2,'generation prompt/feedback process callsites differ')
 require(len(symbol_calls(bodies[r['sampler']],r['eog']))==1,'generation EOS callsite differs')
 for caller in scope:
  for bypass in ('ggml_backend_cpu_init','ggml_backend_sched_graph_compute_async','__real_ggml_backend_sched_graph_compute_async'):
   require(not symbol_calls(bodies[caller],bypass),'generation CPU/scheduler fallback call')
 return edges


def audit(raw,converted,extracted,phase="vocab",full_profile="smoke"):
 require(full_profile in ('smoke','generation','corpus') and (full_profile=='smoke' or phase=='model-full'),'invalid full audit profile')
 paths=[Path(p).resolve() for p in (raw,converted,extracted)]
 tool=Path(shutil.which('llvm-objdump-18') or '');require(tool.is_file(),'objdump unavailable')
 source_paths=[ROOT/n for n in SOURCES+(ALLOCATOR_SOURCES if phase in ('model-gate','model-prefix','model-layer','model-full') else [])]
 identities={str(p):sha(p) for p in [*paths,tool.resolve(),*source_paths]}
 data=paths[0].read_bytes();values=symbol_values(data)
 if PREFIX_HOOK in values:
  for p in (ROOT/n for n in PREFIX_SOURCES):identities[str(p)]=sha(p)
 if LAYER_HOOK in values:
  for p in (ROOT/n for n in LAYER_SOURCES):identities[str(p)]=sha(p)
 if phase=='model-full':
  for p in (ROOT/n for n in FULL_SOURCES+LOADER_SOURCES):identities[str(p)]=sha(p)
 def one(n):
  require(n in values and len(values[n])==1,'missing/ambiguous symbol '+n);return values[n][0]
 def match(fragment):
  names=[n for n in values if fragment in n and not n.startswith('_ZZ')];require(len(names)==1,'ambiguous symbol '+fragment);return names[0]
 finalized=match('finalizedEv');vocab=match('vocabulary_check');storage=match('storage_and_gate_check');gate=match('release_compute');registration=match('registrationEv')
 names=['_start','_init','main','ps5_fp_ieee','ps5hip_install_shell_exit_probe','ps5hip_install_vector_title_finalizer',finalized,vocab,storage,gate,registration,'llama_model_load_from_file','llama_tokenize','llama_model_free','hipInit','hipMalloc','hipMemcpy','hipFree','hipDeviceSynchronize','ggml_backend_sched_graph_compute','ggml_backend_graph_compute',WRAPPER,SUPPORTS,DEVICE]
 snapshot_names=[n for n in values if 'storage_call_snapshot' in n and not n.startswith('_ZZ')]
 require(len(snapshot_names)<=1,'ambiguous storage snapshot')
 call_snapshot=snapshot_names[0] if snapshot_names else None
 require(phase!='vocab' or call_snapshot is not None,'vocabulary storage snapshot missing')
 if call_snapshot is not None:
  hip_statistics=match('hip_call_statistics');names += [call_snapshot,hip_statistics]
 model_names=[n for n in values if 'reject_actual_model' in n and not n.startswith('_ZZ')]
 require(len(model_names)<=1,'ambiguous model gate')
 model_gate=model_names[0] if model_names else None
 chain=[]
 allocator_roles=None
 if phase in ('model-gate','model-prefix','model-layer','model-full'):
  if phase=='model-full':model_gate=match('run_native_full')
  require(model_gate is not None,'model phase code not compiled')
  chain=['main',model_gate,'llama_process',match('_ZN13llama_context6decode'),match('_ZN13llama_context14process_ubatch'),match('_ZN13llama_context13graph_compute'),WRAPPER]
  names+=list(dict.fromkeys(n for n in chain if n not in names))
  allocator_roles=dict(buffer_alloc=match('ggml_backend_cpu_buffer_type_alloc_buffer'),
   buffer_free=match('ggml_backend_cpu_buffer_free_buffer'),finalized=finalized,
   stats=match('native_host_allocator_stats'),close=match('NativeMemory5closeEv'),
   cpu=match('NativeMemory3cpuEv'),malloc_address=one('__wrap_ggml_aligned_malloc'))
  for role,fragments in [('constructors',('NativeMemoryC1Emmb','NativeMemoryC2Emmb')),
                         ('destructors',('NativeMemoryD1Ev','NativeMemoryD2Ev'))]:
   aliases=[n for n in values if any(fragment in n for fragment in fragments)]
   require(aliases and len({one(n) for n in aliases})==1,'missing/ambiguous NativeMemory '+role)
   allocator_roles[role]=aliases
  allocator_names=['__wrap_ggml_aligned_malloc','__wrap_ggml_aligned_free','ggml_init','ggml_free',
   'ggml_backend_sched_new','ggml_backend_sched_free',
   'ggml_threadpool_new_impl','ggml_threadpool_free','ps5hip_llm_fail',allocator_roles['buffer_alloc'],
   allocator_roles['buffer_free'],allocator_roles['stats'],allocator_roles['close'],allocator_roles['cpu'],
   *allocator_roles['constructors'],*allocator_roles['destructors']]
  names+=list(dict.fromkeys(n for n in allocator_names if n not in names))
  memory_snapshot=match('model_memory_snapshot')
  names.append(memory_snapshot)
 prefix=None
 if PREFIX_HOOK in values:names.append(PREFIX_HOOK)
 if phase=='model-prefix':
  require(PREFIX_HOOK in values,'prefix wrapper hook missing')
  prefix=prefix_roles(match)
  prefix_chain=['main',prefix['run'],*chain[2:]]
  prefix_names=[*prefix.values(),*PREFIX_DISPATCH_EDGES,*[n for targets in PREFIX_DISPATCH_EDGES.values() for n in targets],
   'llama_init_from_model','ggml_backend_alloc_ctx_tensors','ggml_backend_tensor_get','ggml_backend_tensor_set','ggml_backend_buffer_free','ggml_free',
   '_ZN13ps5hip_prefix8Dispatch13execute_graphEP11ggml_cgraph']
  names+=list(dict.fromkeys(n for n in prefix_names if n not in names))
 function_addresses=set();function_table=section(data,b'.symtab')
 for at in range(0,len(function_table),24):
  _,info,_,index,value,size=struct.unpack_from('<IBBHQQ',function_table,at)
  if index and info&15==2 and size:function_addresses.add(value)
 layer=None;layer_dispatch_names=[]
 if LAYER_HOOK in values:names.append(LAYER_HOOK)
 if phase=='model-layer':
  require(LAYER_HOOK in values,'layer wrapper hook missing');layer=layer_roles(match)
  layer_chain=['main',layer['run'],*chain[2:]]
  layer_dispatch_names=[n for n in values if ('ps5hip_layer8Dispatch' in n or 'ps5hip_prefix8Dispatch' in n) and (n.startswith('_ZN') or n.startswith('_ZZ')) and len(values[n])==1 and one(n) in function_addresses]
  layer_names=[*layer.values(),*layer_dispatch_names,*[n for targets in layer_edges(layer).values() for n in targets]]
  names+=list(dict.fromkeys(n for n in layer_names if n not in names))
 full=None;full_dispatch_names=[];loader=None;generation=None;generation_names=[];generation_calls=None
 if phase=='model-full':
  loader=loader_roles(values)
  names+=list(dict.fromkeys(n for n in loader.values() if n not in names))
  full=full_roles(match)
  full_dispatch_names=[n for n in values if any(p in n for p in ('ps5hip_full8Dispatch','ps5hip_layer8Dispatch','ps5hip_prefix8Dispatch')) and (n.startswith('_ZN') or n.startswith('_ZZ')) and len(values[n])==1 and one(n) in function_addresses]
  if full_profile in ('generation','corpus'):
   generation=generation_roles(values,function_addresses)
   generation_names=[n for n in values if ('run_full_generation' in n or 'ps5hip_generation' in n) and len(values[n])==1 and one(n) in function_addresses]
   names+=list(dict.fromkeys(n for n in [*generation_names,*[n for targets in generation_edges(full,generation).values() for n in targets]] if n not in names))
  full_names=[*full.values(),'llama_free',*full_dispatch_names,*[n for targets in full_edges(full).values() for n in targets]]
  names+=list(dict.fromkeys(n for n in full_names if n not in names))
 # Current full-capable wrapper is shared by all phases; older reviewed images remain supported.
 full_capable='_ZN2m0L12full_enabledE' in values
 if full_capable:
  names+=list(dict.fromkeys(n for n in FULL_WRAPPER_CALLS if n not in names and n in values))
 text=subprocess.check_output([str(tool),'-d','--no-show-raw-insn','--print-imm-hex','--disassemble-symbols='+','.join(names),str(paths[0])],text=True)
 sizes={};symtab=section(data,b'.symtab')
 for at in range(0,len(symtab),24):
  _,info,_,index,value,size=struct.unpack_from('<IBBHQQ',symtab,at)
  if index and info&15==2 and size:sizes[value]=max(sizes.get(value,0),size)
 bodies={a:t for a,n,t in blocks(text)}
 def body(n):return bounded_body(bodies[one(n)],one(n),sizes.get(one(n),0))
 phase_contract(phase,model_gate,chain,{n:body(n) for n in chain})
 if prefix is not None:
  phase_contract(phase,prefix['run'],prefix_chain,{n:body(n) for n in prefix_chain})
  prefix_contract({n:prefix_scope_aliases(body(n),prefix,values) for n in names},prefix)
 if layer is not None:
  phase_contract(phase,layer['run'],layer_chain,{n:body(n) for n in layer_chain})
  layer_contract({n:prefix_scope_aliases(body(n),layer,values,'ActualLayerQualification') for n in names},layer,layer_dispatch_names)
 if full is not None:full_contract({n:body(n) for n in names},full,full_dispatch_names,full_profile)
 if generation is not None:generation_calls=generation_contract({n:body(n) for n in names},full,generation,generation_names)
 if loader is not None:loader_machine_contract({n:body(n) for n in loader.values()},loader,{k:one(n) for k,n in loader.items()})
 if allocator_roles is not None:allocator_contract({n:body(n) for n in names},allocator_roles)
 positions=ordered_calls(body('_start'),['ps5_fp_ieee','ps5hip_install_shell_exit_probe','ps5hip_install_vector_title_finalizer','_init','main'])
 reader=load_regions(paths[0])[0];slots={n:import_slots(data,n) for n in ['atexit','_init_env','printf']}
 if phase in ('model-gate','model-prefix','model-layer','model-full'):
  for query in ('sceKernelGetDirectMemorySize','sceKernelAvailableDirectMemorySize'):
   require(len(bound_calls(body(memory_snapshot),import_slots(data,query),reader))==1,
           'memory measurement query missing/duplicated: '+query)
 env=bound_calls(body('_start'),slots['_init_env'],reader);require(len(env)==1 and positions[0]<env[0]<positions[1],'startup env ordering')
 require(len(bound_calls(body('ps5hip_install_vector_title_finalizer'),slots['atexit'],reader))==1,'atexit callback')
 require(one(finalized) in {int(s,16) for s in re.findall(r'0x([0-9a-f]+)',body('ps5hip_install_vector_title_finalizer'))},'finalizer identity')
 ordered_calls(body('main'),[vocab,storage]);ordered_calls(body(vocab),['llama_model_load_from_file','llama_tokenize','llama_model_free'])
 if call_snapshot is not None:
  ordered_calls(body(storage),[call_snapshot,gate])
  require(len(symbol_calls(body(call_snapshot),hip_statistics))==1,'storage HIP snapshot missing/duplicated')
 require(symbol_calls(body(storage),gate),'storage missing gate');require(symbol_calls(body(gate),'ggml_backend_sched_graph_compute'),'gate scheduler boundary missing')
 wrapper=body(WRAPPER);calls=[]
 for name in ['ggml_graph_n_nodes','ggml_graph_node','ggml_backend_sched_get_tensor_backend','ggml_backend_get_device',DEVICE,SUPPORTS,'ggml_op_desc','ggml_backend_name','ggml_backend_sched_graph_compute_async']:
  calls.extend((pos,name) for pos in symbol_calls(wrapper,name))
 if PREFIX_HOOK in values:calls.extend((pos,PREFIX_HOOK) for pos in symbol_calls(wrapper,PREFIX_HOOK))
 if LAYER_HOOK in values:calls.extend((pos,LAYER_HOOK) for pos in symbol_calls(wrapper,LAYER_HOOK))
 calls.extend((pos,'printf') for pos in bound_calls(wrapper,slots['printf'],reader))
 full_support_table=None
 if full_capable:
  calls=[]
  for name in set(FULL_WRAPPER_CALLS):
   if name in ('fprintf','printf'):found=bound_calls(wrapper,import_slots(data,name),reader)
   else:found=symbol_calls(wrapper,name)
   calls.extend((pos,name) for pos in found)
  full_wrapper_contract(wrapper,one(WRAPPER),calls,one('_ZN2m0L12full_enabledE'))
  full_global_bindings(wrapper,one(WRAPPER),FULL_GLOBAL_TARGETS,one)
  full_support_table=full_support_contract(body(SUPPORTS),one(SUPPORTS),reader,one('_ZN2m0L12full_enabledE'))
  if phase!='model-full':require(not any('enable_full' in n for n in values),'full mode enable retained in legacy phase')
 else:
  require(phase!='model-full','full wrapper missing')
  wrapper_contract(wrapper,one(WRAPPER),body(SUPPORTS),calls)
  if phase=='model-prefix':require(normalized_wrapper(wrapper,one(WRAPPER))in (PREFIX_WRAPPER_TEMPLATE,LAYER_WRAPPER_TEMPLATE),'prefix wrapper not retained')
  if phase=='model-layer':require(normalized_wrapper(wrapper,one(WRAPPER))==LAYER_WRAPPER_TEMPLATE,'layer wrapper not retained')
 if full is not None:
  full_mode_store(body(full['enable']),one('_ZN2m0L12full_enabledE'),1)
  full_mode_store(body(full['disable']),one('_ZN2m0L12full_enabledE'),0)
  for caller,target in [('main',full['writer']),(full['run'],full['readback']),(full['backend_init'],full['graph']),(full['run'],'llama_free'),(full['run'],'llama_model_free')]:
   require(one(target) in {int(v,16) for v in re.findall(r'0x([0-9a-f]+)',body(caller))},'full callback address missing: '+target)
  for name in ('open','write','fsync','close'):
   require(len(prefix_writer_import_calls(body(full['writer']),import_slots(data,name),reader))==1,'full raw writer import missing/duplicated: '+name)
 if layer is not None:
  for caller,target in [('main',layer['writer']),(layer['run'],layer['readback']),(layer['backend_init'],layer['graph'])]:
   require(one(target) in {int(v,16) for v in re.findall(r'0x([0-9a-f]+)',body(caller))},'layer callback address missing: '+target)
  for name in ('open','write','fsync','close'):
   require(len(prefix_writer_import_calls(body(layer['writer']),import_slots(data,name),reader))==1,'layer raw writer import missing/duplicated: '+name)
 if prefix is not None:
  # Selected callback address references are checked; complete indirect dataflow remains source/test evidence.
  for caller,target in [('main',prefix['writer']),(prefix['run'],prefix['readback']),(prefix['backend_init'],prefix['graph'])]:
   require(one(target) in {int(v,16) for v in re.findall(r'0x([0-9a-f]+)',body(caller))},'prefix callback address missing: '+target)
  for name in ('open','write','fsync','close'):
   require(len(prefix_writer_import_calls(body(prefix['writer']),import_slots(data,name),reader))==1,'prefix raw writer import missing/duplicated: '+name)
 audit_text([load_regions(p)[0] for p in paths],{one(n) for n in names},sizes)
 if full_support_table is not None:
  original=reader(full_support_table,404)[0]
  require(all(load_regions(p)[0](full_support_table,404)[0]==original for p in paths[1:]),'converted/extracted support table differs')
  for p in paths:
   flag_reader,headers,_=load_regions(p)
   for name in ('_ZN2m0L12full_enabledE','_ZN2m0L11full_activeE'):
    zero_initialized_flag(flag_reader,headers,one(name))
 layouts=[dict(file=str(p),segments=load_layout(p.read_bytes())) for p in paths]
 imports=[{**verify_imports(p),**lifecycle_imports(read_imports(p,verify_hash=True)[2])} for p in paths[1:]]
 require(imports[0]==imports[1],'converted/extracted import owners differ')
 require(all(sha(p)==h for p,h in identities.items()),'audit inputs changed')
 return dict(status='PASS',classification='offline-public-hip-llm-client-native-image-audit',phase=phase,
  inputs=identities,audited_functions=names,audited_function_count=len(names),layouts=layouts,imports=imports[0],
   actual_full_call_chain_audited=(phase=='model-full'),native_loader_staging_audited=(loader is not None),full_profile=(full_profile if phase=='model-full' else None),actual_generation_call_chain_audited=(generation is not None),generation_direct_edges=generation_calls,generation_feedback_dataflow_proven=False,full_callback_dataflow_proven=False,
   actual_layer_call_chain_audited=(phase=='model-layer'),layer_callback_dataflow_proven=False,
  actual_prefix_call_chain_audited=(phase=='model-prefix'),prefix_callback_dataflow_proven=False,
   native_allocator_routing_audited=(phase in ('model-gate','model-prefix','model-layer','model-full')),
   wrapper_retained=True,wrapper_no_bypass=True,actual_client_call_chain_audited=(phase in ('model-gate','model-prefix','model-layer','model-full')),actual_client_gate_executed=False,native_executed=False,gpu_executed=False,
  limits=['Prefix selected direct-call edges and callback addresses are checked; ownership, original-leaf staging, preflight/topology, indirect callback routing and exclusive-write flags are additionally source/test evidence, not exhaustive machine-code proofs.',
  'Selected GGML context, CPU-buffer, scheduler context-buffer and threadpool aligned pairs are checked in model mode; ordinary libc allocations are not intercepted and native ownership is unproven.',
  'Actual model call chains are checked for model phases; static presence is not execution evidence.',
   'Legacy reviewed wrappers reject nonempty graphs. Full-capable wrapper additionally checks exact active-node ownership, begin/real scheduler/finish control flow, mode/global targets and all 101 support-table entries; codegen changes require review.',
   'Full profile argument and, for generation, selected tokenizer/reset/sampler/decode/logit/EOS call edges are checked. Direct cleanup/reset edges and selected callback addresses are checked; complete indirect callback/unique_ptr dataflow and runtime tensor ownership additionally require source/host/native evidence.',
  'Static preservation/import/lifecycle audit only; no native loading, tokenization, computation or teardown execution proof.'])
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__)
 for n in ['raw','converted','extracted','output']:parser.add_argument('--'+n,type=Path,required=True)
 parser.add_argument('--phase',choices=('vocab','model-gate','model-prefix','model-layer','model-full'),default='vocab')
 parser.add_argument('--full-profile',choices=('smoke','generation','corpus'),default='smoke')
 args=parser.parse_args()
 if args.output.exists():parser.error('fresh output required')
 report=audit(args.raw,args.converted,args.extracted,args.phase,args.full_profile);args.output.parent.mkdir(parents=True,exist_ok=True)
 with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
