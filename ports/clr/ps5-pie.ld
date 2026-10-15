/*
 * ps5-native-app-boilerplate - Intermediate PIE linker layout.
 * Copyright (C) 2026 BlackBearReloaded
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Keeps code, immutable data, RELRO, and writable data in distinct address
 * ranges. The native converter replaces LLVM's generic dynamic metadata.
 */

OUTPUT_FORMAT("elf64-x86-64")
OUTPUT_ARCH(i386:x86-64)

PHDRS {
    text PT_LOAD FLAGS (0x5);
    ro PT_LOAD FLAGS (0x4);
    relro PT_LOAD FLAGS (0x6);
    data PT_LOAD FLAGS (0x6);
    dynamic PT_DYNAMIC FLAGS (0x6);
}

SECTIONS {
    .text 0 : ALIGN(CONSTANT(MAXPAGESIZE)) {
        QUAD(0xcccccccccccccccc)
        QUAD(0xcccccccccccccccc)
        *(.text .text.*)
        *(.plt .plt.*)
    } : text

    .eh_frame_hdr : ALIGN(CONSTANT(MAXPAGESIZE)) {
        KEEP(*(.eh_frame_hdr))
    } : ro
    .eh_frame : { KEEP(*(.eh_frame)) } : ro
    .gcc_except_table : { *(.gcc_except_table .gcc_except_table.*) } : ro
    .rodata : {
        HIDDEN(__ps5hip_names_begin = .);
        *(.rodata .rodata.*)
        HIDDEN(__ps5hip_names_end = .);
        . = ALIGN(4096);
        HIDDEN(__ps5hip_bundle_begin = .);
        . = ALIGN(4096);
        HIDDEN(multi_a_bundle_begin = .);
        KEEP(*multi-a.o(.hip_fatbin))
        HIDDEN(multi_a_bundle_end = .);
        . = ALIGN(4096);
        HIDDEN(multi_b_bundle_begin = .);
        KEEP(*multi-b.o(.hip_fatbin))
        HIDDEN(multi_b_bundle_end = .);
        HIDDEN(__ps5hip_bundle_end = .);
        . = ALIGN(8);
        HIDDEN(__ps5hip_module_spans_begin = .);
        QUAD(multi_a_wrapper_begin-__ps5hip_wrappers_begin);
        QUAD(multi_a_wrapper_end-__ps5hip_wrappers_begin);
        QUAD(multi_a_bundle_begin-__ps5hip_bundle_begin);
        QUAD(multi_a_bundle_end-__ps5hip_bundle_begin);
        QUAD(multi_b_wrapper_begin-__ps5hip_wrappers_begin);
        QUAD(multi_b_wrapper_end-__ps5hip_wrappers_begin);
        QUAD(multi_b_bundle_begin-__ps5hip_bundle_begin);
        QUAD(multi_b_bundle_end-__ps5hip_bundle_begin);
        HIDDEN(__ps5hip_module_spans_end = .);
    } : ro

    . = ALIGN(CONSTANT(MAXPAGESIZE));
    .data.rel.ro : {
        HIDDEN(__ps5hip_relro_begin = .);
        *(.data.rel.ro .data.rel.ro.*)
        . = ALIGN(8);
        HIDDEN(__ps5hip_wrappers_begin = .);
        HIDDEN(multi_a_wrapper_begin = .);
        KEEP(*multi-a.o(.hipFatBinSegment))
        HIDDEN(multi_a_wrapper_end = .);
        HIDDEN(multi_b_wrapper_begin = .);
        KEEP(*multi-b.o(.hipFatBinSegment))
        HIDDEN(multi_b_wrapper_end = .);
        HIDDEN(__ps5hip_wrappers_end = .);
        HIDDEN(__ps5hip_relro_end = .);
    } : relro
    .got : { *(.got) } : relro
    .got.plt : { *(.got.plt) } : relro
    .preinit_array : {
        PROVIDE_HIDDEN(__preinit_array_start = .);
        KEEP(*(.preinit_array))
        PROVIDE_HIDDEN(__preinit_array_end = .);
    } : relro
    .init_array : {
        PROVIDE_HIDDEN(__init_array_start = .);
        KEEP(*(SORT_BY_INIT_PRIORITY(.init_array.*) SORT_BY_INIT_PRIORITY(.ctors.*)))
        KEEP(*(.init_array .ctors))
        PROVIDE_HIDDEN(__init_array_end = .);
    } : relro
    .fini_array : {
        PROVIDE_HIDDEN(__fini_array_start = .);
        KEEP(*(SORT_BY_INIT_PRIORITY(.fini_array.*) SORT_BY_INIT_PRIORITY(.dtors.*)))
        KEEP(*(.fini_array .dtors))
        PROVIDE_HIDDEN(__fini_array_end = .);
    } : relro

    /* Kept only long enough for the converter to read and rewrite them. */
    .dynsym : { *(.dynsym) } : relro
    .gnu.hash : { *(.gnu.hash) } : relro
    .dynstr : { *(.dynstr) } : relro
    .rela.dyn : { *(.rela.dyn) } : relro
    .rela.plt : { *(.rela.plt) } : relro

    . = ALIGN(CONSTANT(MAXPAGESIZE));
    .dynamic : {
        *(.dynamic)
    } : data : dynamic
    .data : { *(.data .data.*) } : data
    .tdata : { *(.tdata .tdata.*) } : data
    .tbss : { *(.tbss .tbss.*) } : data
    .bss : {
        *(.bss .bss.*)
        *(COMMON)
    } : data
}

/* Audited optional provider absence for this CPU-only profile. */
HIDDEN(__dlopen = 0);
HIDDEN(__dlsym = 0);
HIDDEN(__dladdr = 0);
HIDDEN(__dlclose = 0);
HIDDEN(__dlerror = 0);
HIDDEN(kernel_mprotect = 0);
