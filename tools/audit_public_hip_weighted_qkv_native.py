"""Static startup, import and text-preservation audit for the weighted RMSNorm/shared-Q8/QKV title."""
import argparse
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
from audit_public_hip_rmsnorm_native import (sha, import_slots, bound_calls, symbol_calls,
                                            ordered_calls)
from audit_public_hip_embedded_native import audit_text
from probe_host_registration import load_regions, require
from native_multi_query import symbol_values, blocks
from fat_binary_fixture_test import section

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ('hipModuleLoad','hipModuleGetFunction','hipMalloc','hipModuleLaunchKernel',
          'hipDeviceSynchronize','hipMemcpy','hipFree','hipModuleUnload')
IMPORTS = ('atexit','_init_env','open','write','fsync','close')
SOURCES = ('tools/audit_public_hip_weighted_qkv_native.py','tests/public_hip_weighted_qkv_native_audit_test.py',
           'tools/audit_public_hip_rmsnorm_native.py','tools/audit_public_hip_embedded_native.py',
           'tools/probe_host_registration.py','tools/native_multi_query.py',
           'tests/fat_binary_fixture_test.py','tests/native_import_libraries_test.py',
           'tools/build_native_vector_title.py','tools/public_hip_weighted_qkv_console_title.cpp',
           'tools/public_hip_app_crt.cpp','tools/clr_gpu/watchdog.cpp','tools/clr_gpu/watchdog.h')


def validate_calls(values, bodies, slots, reader):
    def one(name):
        require(name in values and len(values[name]) == 1,'missing or ambiguous symbol: '+name)
        return values[name][0]
    def body(name):
        address=one(name)
        require(address in bodies,'missing disassembly: '+name)
        return bodies[address]
    def matching(fragment):
        names=[name for name in values if fragment in name]
        require(len(names)==1,'missing or ambiguous helper: '+fragment)
        return names[0]
    dump=matching('dump_exclusive');finalized=matching('finalizedEv')
    start=matching('watchdog_start');finish=matching('watchdog_finish')
    finite=matching('finite_floats');quantized=matching('q8_domain');seed=matching('nan_seed')
    names=['_start','_init','main','ps5_fp_ieee','ps5hip_install_shell_exit_probe',
           'ps5hip_install_vector_title_finalizer',dump,finalized,start,finish,finite,quantized,seed,*PUBLIC]
    positions=ordered_calls(body('_start'),names[3:6]+['_init','main'])
    environment=bound_calls(body('_start'),slots['_init_env'],reader)
    require(len(environment)==1 and positions[0]<environment[0]<positions[1],
            'runtime environment startup order changed')
    main=body('main')
    for name in PUBLIC:require(symbol_calls(main,name),'main missing public call: '+name)
    require(len(symbol_calls(main,'hipModuleLaunchKernel'))==3,'weighted RMSNorm, Q8 and Q4 launch callsites required')
    require(len(symbol_calls(main,finite))==4,'input/norm/weighted/projection finite checks required')
    require(len(symbol_calls(main,quantized))==1,'Q8 domain check required')
    require(len(symbol_calls(main,seed))==3,'weighted/initial-output/projection-output sentinels required')
    require(len(symbol_calls(main,'hipMemcpy'))==4,'initial/reset/projection-weight/readback transfer callsites required')
    require(len(symbol_calls(main,dump))==1,'one looped raw dump callsite required')
    lifecycle=ordered_calls(main,[start,'hipModuleLoad','hipModuleUnload',finish])
    finalizer=body('ps5hip_install_vector_title_finalizer')
    require(len(bound_calls(finalizer,slots['atexit'],reader))==1,'actual atexit registration required')
    require(one(finalized) in {int(x,16) for x in re.findall(r'0x([0-9a-f]+)',finalizer)},
            'wrong finalizer callback')
    dump_body=body(dump)
    for name in ('open','write','fsync','close'):
        require(len(bound_calls(dump_body,slots[name],reader))==1,'dump missing actual import: '+name)
    require(bound_calls(dump_body,slots['fsync'],reader)[0]<bound_calls(dump_body,slots['close'],reader)[0],
            'dump close precedes sync')
    return dict(functions={one(name) for name in names},dump_helper=dump,finalizer_callback=finalized,
                watchdog_helpers=[start,finish],watchdog_main_order=lifecycle)


def audit(raw,converted,extracted):
    paths=[Path(p).resolve() for p in (raw,converted,extracted)]
    objdump=Path(shutil.which('llvm-objdump-18') or '')
    require(objdump.is_file(),'LLVM18 objdump unavailable')
    identities={str(p):sha(p) for p in [*paths,objdump.resolve(),*(ROOT/n for n in SOURCES)]}
    data=paths[0].read_bytes()
    disassembly=subprocess.check_output([str(objdump),'-d','--no-show-raw-insn','--print-imm-hex',str(paths[0])],text=True)
    bodies={address:text for address,_,text in blocks(disassembly)}
    slots={name:import_slots(data,name) for name in IMPORTS}
    result=validate_calls(symbol_values(data),bodies,slots,load_regions(paths[0])[0])
    sizes={};symtab=section(data,b'.symtab')
    for at in range(0,len(symtab),24):
        _,info,_,index,value,size=struct.unpack_from('<IBBHQQ',symtab,at)
        if index and info&15==2 and size:sizes[value]=max(sizes.get(value,0),size)
    functions=result.pop('functions')
    audit_text([load_regions(path)[0] for path in paths],functions,sizes)
    require(all(sha(path)==digest for path,digest in identities.items()),'audit input changed')
    return dict(schema=1,status='PASS',classification='offline-public-hip-weighted-qkv-native-image-audit',
                inputs=identities,audited_function_count=len(functions),public_calls=list(PUBLIC),
                startup_order=['IEEE','runtime environment','shell exit','title finalizer','constructors','main'],
                imports={name:sorted(addresses) for name,addresses in slots.items()},
                native_executed=False,**result,
                limits=['Static image/callsite/text preservation only; no numerical or GPU execution proof.',
                        'One dump callsite is looped; 30 runtime files require independent execution verification.',
                        'Dynamic module title; no compiler registration or launch-stub claim.',
                        'Callsite/text preservation does not independently prove buffer alias/lifetime semantics or forbid identical-byte reupload; source and raw execution review remain required.',
                        'Transport-complete transcript is not final numerical acceptance; external chained readback verification required.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('raw','converted','extracted','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('fresh output required')
    result=audit(args.raw,args.converted,args.extracted)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
