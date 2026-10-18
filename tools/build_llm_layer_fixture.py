"""Build bounded source-local layer kernels using an authenticated existing RMS fixture."""
import argparse
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
from build_llm_weighted_rmsnorm_fixture import original_inputs, discover_tool
from build_llm_rmsnorm_fixture import CONSTANTS
from build_llm_q4_fixture import ROOT, sha, bind
from qualify_public_hip_rmsnorm import checked_inputs, checked_tools
from probe_hip_routes import audit_object

CLASSIFICATION='source-only-bounded-layer-fixture'
DENSE_CLASSIFICATION='source-only-bounded-dense-f16-layer-fixture'
SYMBOLS=('layer_copy','layer_to_f16','layer_add','layer_mul','layer_silu','layer_rope','layer_set_rows','layer_scores','layer_softmax','layer_values','layer_quantize_q8_cpu','layer_q4_cpu','layer_rms_cpu','layer_silu_cpu','layer_softmax_cpu','layer_rope_cpu')
SOURCES=('kernels/llm_layer.hip','kernels/llm_cpu_math.h','tests/llm_cpu_math_test.cpp','tests/llm_cpu_math_test.py','tools/build_llm_layer_fixture.py','tests/llm_layer_fixture_test.py','tests/llm_layer_admission.cpp','tests/llm_layer_kernel_test.cpp','src/code_object.cpp','include/ps5hip/code_object.h','tools/build_llm_weighted_rmsnorm_fixture.py','tools/build_llm_q4_fixture.py','tools/build_llm_rmsnorm_fixture.py','tools/qualify_public_hip_rmsnorm.py','tools/probe_hip_routes.py','tools/build_canaries.py','tests/fat_binary_fixture_test.py')
TOOLS=('clang++-18','llvm-config-18','ld.lld-18','llvm-readobj-18','llvm-objdump-18','llvm-nm-18')
LABELS=('compile','notes','isa','undefined','admission-build','admission')
REQUIRED=('layer.hsaco','notes.stdout','isa.stdout','undefined.stdout','admission.stdout','commands.json')

def selected_symbols(profile):
    if profile not in ('default','dense-f16'):raise ValueError('unknown layer kernel profile')
    return SYMBOLS+('layer_dense_f16',) if profile=='dense-f16' else SYMBOLS

def contracts(folder,profile='default'):
    symbols=selected_symbols(profile)
    folder=Path(folder)
    if (folder/'undefined.stdout').read_text().strip():raise ValueError('undefined device symbols')
    isa=(folder/'isa.stdout').read_text()
    q4_parts=isa.split('<layer_q4_cpu>:',1)
    if len(q4_parts)!=2:raise ValueError('layer Q4 symbol missing')
    q4_isa=re.split(r'\n[0-9a-f]+ <[^>]+>:',q4_parts[1],maxsplit=1)[0]
    if len(re.findall(r'\bglobal_load_ubyte\b',q4_isa))!=16 or re.search(r'\b(?:flat_|scratch_|s_swappc|s_setpc)',isa):raise ValueError('layer ISA load/call contract changed')
    audit=audit_object((folder/'layer.hsaco').read_bytes(),(folder/'notes.stdout').read_text(),(folder/'isa.stdout').read_text(),symbols)
    rows=(folder/'admission.stdout').read_text().splitlines()
    if len(rows)!=len(symbols) or {r.split()[0] for r in rows}!=set(symbols) or any(not r.endswith('rejection=') for r in rows):raise ValueError('Initial admission failed')
    expected_sizes={n:28 for n in symbols}
    expected_sizes.update(layer_copy=20,layer_to_f16=20,layer_silu=20,layer_rope=24,layer_set_rows=32,layer_quantize_q8_cpu=20,layer_q4_cpu=32,layer_rms_cpu=16,layer_silu_cpu=20,layer_softmax_cpu=28,layer_rope_cpu=24)
    if profile=='dense-f16':
        expected_sizes['layer_dense_f16']=32
        parts=isa.split('<layer_dense_f16>:',1)
        if len(parts)!=2:raise ValueError('dense ISA symbol missing')
        dense=re.split(r'\n[0-9a-f]+ <[^>]+>:',parts[1],maxsplit=1)[0]
        # Observed gfx1013 lowering: 64-bit row offset and 32 independent F32 FMAs.
        if not all(op in dense for op in ('v_mad_u64_u32','v_lshlrev_b64','v_add_co_u32','v_add_co_ci_u32')) or len(re.findall(r'\bv_fma_mix_f32\b',dense))!=32 or len(re.findall(r'\bv_add_f32(?:_e32|_e64)?\b',dense))!=31:raise ValueError('dense address/reduction ISA changed')
    for name,d in audit['descriptors'].items():
        if d['kernarg_bytes']!=expected_sizes[name] or d['rsrc2']!=0x8c or d['properties']!=0x409 or any(d[n] for n in ('group_bytes','private_bytes','rsrc3','preload')):raise ValueError('unexpected Initial descriptor')
    return audit

def fixture_inputs(folder,profile='default'):
    selected_symbols(profile)
    folder=Path(folder).resolve();receipt=folder/'receipt.json';report=json.loads(receipt.read_text())
    if report.get('classification')!=(DENSE_CLASSIFICATION if profile=='dense-f16' else CLASSIFICATION) or report.get('status')!='PASS' or report.get('native_executed') is not False:raise ValueError('invalid layer fixture')
    if report.get('kernel_profile','default')!=profile:raise ValueError('layer kernel profile mismatch')
    original=Path(report['original_fixture']).resolve()
    if original==folder:raise ValueError('recursive prerequisite')
    parent=original_inputs(original)
    inputs=checked_inputs(report,'inputs',[ROOT/n for n in SOURCES])
    if any(inputs.get(n)!=d for n,d in parent.items()):raise ValueError('missing prerequisite closure')
    for n,d in checked_tools(report,'tools',TOOLS).items():
        if inputs.get(n)!=d:raise ValueError('missing tool identity')
    artifacts=report.get('artifacts',{})
    if not set(REQUIRED)<=artifacts.keys():raise ValueError('required artifact missing')
    for n,row in artifacts.items():
        p=(folder/n).resolve()
        if Path(n).is_absolute() or not p.is_relative_to(folder) or p==receipt or sha(p)!=row['sha256'] or p.stat().st_size!=row['bytes']:raise ValueError('artifact identity changed')
        inputs[str(p)]=row['sha256']
    commands=report.get('commands',[])
    if [c.get('label') for c in commands]!=list(LABELS) or any(c.get('exit_code')!=0 for c in commands) or json.loads((folder/'commands.json').read_text())!=commands:raise ValueError('invalid build commands')
    compile_args=commands[0]['argv'];admission_args=commands[-1]['argv']
    if ('-DPS5HIP_ENABLE_DENSE_F16=1' in compile_args)!=(profile=='dense-f16') or admission_args!=[str(folder/'admission'),str(folder/'layer.hsaco')]+(['--dense-f16'] if profile=='dense-f16' else []):raise ValueError('layer profile commands changed')
    if contracts(folder,profile)!=report.get('profile'):raise ValueError('object contract changed')
    inputs[str(receipt)]=sha(receipt)
    return inputs

def build(prerequisite,output,profile='default'):
    selected_symbols(profile)
    prerequisite,output=Path(prerequisite).resolve(),Path(output).resolve()
    if not any(output.is_relative_to(ROOT/n) for n in ('build','results')):raise ValueError('output outside build/results')
    parent=original_inputs(prerequisite);inputs=dict(parent)
    for n in SOURCES:bind(inputs,ROOT/n)
    bind(inputs,sys.executable)
    tools={n:discover_tool(n) for n in TOOLS}
    for row in tools.values():bind(inputs,row['path'])
    cxx=shutil.which('g++')
    if not cxx:raise ValueError('g++ required for actual admission')
    bind(inputs,cxx)
    output.mkdir(parents=True,exist_ok=False);commands=[]
    def run(label,argv):
        argv=list(map(str,argv));p=subprocess.run(argv,cwd=ROOT,capture_output=True,timeout=180)
        commands.append(dict(label=label,argv=argv,exit_code=p.returncode))
        for suffix,data in [('stdout',p.stdout),('stderr',p.stderr)]:(output/(label+'.'+suffix)).write_bytes(data)
        (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if p.returncode:raise RuntimeError(label+' failed; logs retained')
    bindir=subprocess.check_output([tools['llvm-config-18']['path'],'--bindir'],text=True).strip()
    order=['ockl-workitem-only','mul_hi','ocml',*(n for n in CONSTANTS if n!='oclc_isa_version_1030')]
    for n in order:
        path=prerequisite/'libs'/(n+'.bc')
        if inputs.get(str(path))!=sha(path):raise ValueError('library absent from prerequisite')
    flags=[tools['clang++-18']['path'],'-x','hip','-nogpuinc','--offload-arch=gfx1013','-mcode-object-version=5','-mno-wavefrontsize64','-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','--cuda-device-only','--no-gpu-bundle-output','-B'+bindir,'--rocm-device-lib-path='+str(prerequisite/'libs')]
    flags+=['--hip-device-lib='+n+'.bc' for n in order]
    if profile=='dense-f16':flags+=['-DPS5HIP_ENABLE_DENSE_F16=1']
    run('compile',flags+['-c',ROOT/'kernels/llm_layer.hip','-o',output/'layer.hsaco'])
    run('notes',[tools['llvm-readobj-18']['path'],'--notes',output/'layer.hsaco'])
    run('isa',[tools['llvm-objdump-18']['path'],'-d','--mcpu=gfx1013',output/'layer.hsaco'])
    run('undefined',[tools['llvm-nm-18']['path'],'--undefined-only',output/'layer.hsaco'])
    run('admission-build',[cxx,'-std=c++17','-O1','-I'+str(ROOT/'include'),ROOT/'tests/llm_layer_admission.cpp',ROOT/'src/code_object.cpp','-o',output/'admission'])
    run('admission',[output/'admission',output/'layer.hsaco']+(['--dense-f16'] if profile=='dense-f16' else []))
    audit=contracts(output,profile)
    if original_inputs(prerequisite)!=parent:raise ValueError('prerequisite changed')
    for n,d in inputs.items():
        if sha(n)!=d:raise ValueError('input changed: '+n)
    report=dict(classification=DENSE_CLASSIFICATION if profile=='dense-f16' else CLASSIFICATION,status='PASS',native_executed=False,gpu_executed=False,original_fixture=str(prerequisite),inputs=inputs,tools=tools,commands=commands,profile=audit,artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in output.iterdir() if p.is_file()},limits=['Source-local kernels; no native numerical validation.','Explicit F16 Q/P operands and F32 accumulation; differs from upstream HIP half-PV accumulation.'])
    if profile=='dense-f16':report['kernel_profile']=profile
    (output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');fixture_inputs(output,profile);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--rmsnorm-fixture',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--profile',choices=('default','dense-f16'),default='default');a=p.parse_args();build(a.rmsnorm_fixture,a.output,a.profile);print('PASS layer fixture: '+str(a.output))
