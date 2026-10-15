"""Generate exact per-object bounds for compiler HIP registration."""
import re


def _parts(object_names):
    names = list(object_names)
    if (not 1 <= len(names) <= 32 or len(set(names)) != len(names)
            or any(not re.fullmatch(r'[A-Za-z0-9_][A-Za-z0-9_.-]*\.o', n) for n in names)):
        raise ValueError('expected 1..32 unique plain object basenames')
    rw = '        *(.data.rel.ro .data.rel.ro.*)'
    read = ['    .rodata : {', '        HIDDEN(__ps5hip_names_begin = .);',
            '        *(.rodata .rodata.*)', '        HIDDEN(__ps5hip_names_end = .);',
            '        . = ALIGN(4096);', '        HIDDEN(__ps5hip_bundle_begin = .);']
    write = ['        HIDDEN(__ps5hip_relro_begin = .);', rw,
             '        . = ALIGN(8);', '        HIDDEN(__ps5hip_wrappers_begin = .);']
    checks = []
    for i, name in enumerate(names):
        for lines, kind, section in ((read, 'bundle', '.hip_fatbin'),
                                     (write, 'wrapper', '.hipFatBinSegment')):
            if kind == 'bundle':
                lines.append('        . = ALIGN(4096);')
            lines += [f'        HIDDEN(__ps5hip_m{i}_{kind}_begin = .);',
                      f'        KEEP({name}({section}))',
                      f'        KEEP(*/{name}({section}))',
                      f'        HIDDEN(__ps5hip_m{i}_{kind}_end = .);']
        width = f'__ps5hip_m{i}_wrapper_end - __ps5hip_m{i}_wrapper_begin'
        size = f'__ps5hip_m{i}_bundle_end - __ps5hip_m{i}_bundle_begin'
        checks += [f'ASSERT({width} == 24, "missing or invalid HIP wrapper {i}")',
                   f'ASSERT({size} >= 32 && {size} <= 67108864, "invalid HIP bundle {i}")']
    read += ['        HIDDEN(__ps5hip_bundle_end = .);', '        . = ALIGN(8);',
             '        HIDDEN(__ps5hip_module_spans_begin = .);']
    for i in range(len(names)):
        for kind, base in (('wrapper', 'wrappers'), ('bundle', 'bundle')):
            for edge in ('begin', 'end'):
                read.append(f'        QUAD(__ps5hip_m{i}_{kind}_{edge} - __ps5hip_{base}_begin);')
    read += ['        HIDDEN(__ps5hip_module_spans_end = .);', '    } : ro',
             '    .ps5hip_unlisted : { KEEP(*(.hip_fatbin .hipFatBinSegment)) } : ro']
    write += ['        HIDDEN(__ps5hip_wrappers_end = .);',
              '        HIDDEN(__ps5hip_relro_end = .);']
    # Catch-all must follow both explicit selectors, or it consumes the wrappers.
    catchall = read.pop()
    checks += ['ASSERT(SIZEOF(.ps5hip_unlisted) == 0, "unlisted HIP object")',
               'ASSERT(__ps5hip_bundle_end - __ps5hip_bundle_begin <= 67108864, "HIP bundle budget exceeded")']
    return read, write, catchall, checks


def patch_linker(source, object_names):
    ro = '    .rodata : { *(.rodata .rodata.*) } : ro'
    rw = '        *(.data.rel.ro .data.rel.ro.*)'
    if source.count(ro) != 1 or source.count(rw) != 1 or '__ps5hip_' in source:
        raise ValueError('expected clean native linker layout')
    read, write, catchall, checks = _parts(object_names)
    output = source.replace(ro, '\n'.join(read)).replace(rw, '\n'.join(write))
    end = '    .got : { *(.got) } : relro'
    if output.count(end) != 1:
        raise ValueError('expected native GOT boundary')
    return output.replace(end, catchall + '\n' + end) + '\n' + '\n'.join(checks) + '\n'


def linux_linker_fragment(object_names):
    read, write, catchall, checks = _parts(object_names)
    # Preserve the system layout and loader; compiler names already live in .rodata.
    bundle = ['    .ps5hip_bundles : {', *read[4:-1], '    }']
    wrapper = ['    .data.rel.ro : {', *write, '    }']
    return ('HIDDEN(__ps5hip_names_begin = ADDR(.rodata));\n'
            'HIDDEN(__ps5hip_names_end = ADDR(.rodata) + SIZEOF(.rodata));\n'
            'SECTIONS {\n' + '\n'.join(bundle) + '\n} INSERT AFTER .rodata;\n'
            'SECTIONS {\n' + '\n'.join(wrapper) + '\n'
            + catchall.removesuffix(' : ro') + '\n} INSERT BEFORE .dynamic;\n'
            + '\n'.join(checks) + '\n')
