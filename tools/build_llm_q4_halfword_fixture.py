"""Authenticated offline Q4 halfword-load diagnostic; never a GPU qualification."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from build_llm_q4_fixture import ROOT, PROFILES, sha, bind, metadata_contract, descriptor_contract
from probe_hip_routes import audit_object
from qualify_public_hip_rmsnorm import checked_inputs, checked_tools

CLASSIFICATION = 'diagnostic-q4-halfword-fixture'
SOURCES = ('tools/build_llm_q4_halfword_fixture.py', 'tests/llm_q4_halfword_fixture_test.py',
           'tools/qualify_public_hip_q4.py', 'tools/build_llm_q4_fixture.py',
           'tools/probe_hip_routes.py', 'tools/qualify_public_hip_rmsnorm.py')
TOOLS = ('llc-18', 'ld.lld-18', 'llvm-readobj-18', 'llvm-objdump-18', 'llvm-nm-18')
REQUIRED = ('q8.hsaco', 'q4.hsaco', 'q4.ll', 'q4.o', 'q4-notes.stdout', 'q4-isa.stdout', 'q4-undefined.stdout')

def transform(original):
    needle = 'load i16, ptr addrspace(1)'
    if original.count(needle) != 4 or 'load volatile i16' in original:
        raise ValueError('expected exactly four original halfword loads')
    return original.replace(needle, 'load volatile i16, ptr addrspace(1)')

def original_inputs(folder):
    folder = Path(folder).resolve()
    report = json.loads((folder/'receipt.json').read_text())
    if report.get('classification') != 'source-only-selected-upstream-q4-q8-fixture':
        raise ValueError('diagnostic parent prohibited; original source fixture required')
    from qualify_public_hip_q4 import fixture_inputs as validate_original
    return validate_original(folder)

def contracts(folder, original):
    folder, original = Path(folder), Path(original)
    if (folder/'q4.ll').read_text() != transform((original/'q4.ll').read_text()):
        raise ValueError('diagnostic transform changed')
    if sha(folder/'q8.hsaco') != sha(original/'q8.hsaco'):
        raise ValueError('original Q8 changed')
    notes = (folder/'q4-notes.stdout').read_text()
    isa = (folder/'q4-isa.stdout').read_text()
    metadata = metadata_contract(notes, PROFILES['q4'])
    if metadata != metadata_contract((original/'q4-notes.stdout').read_text(), PROFILES['q4']):
        raise ValueError('original metadata changed')
    audit = audit_object((folder/'q4.hsaco').read_bytes(), notes, isa, [PROFILES['q4']['symbol']])
    descriptor = audit['descriptors'][PROFILES['q4']['symbol']]
    descriptor_contract(descriptor, dict(PROFILES['q4'], rsrc1=0x60af00c4))
    original_report = json.loads((original/'receipt.json').read_text())
    descriptor_contract(original_report['profiles']['q4']['descriptor'], PROFILES['q4'])
    if (folder/'q4-undefined.stdout').read_text().strip():
        raise ValueError('undefined diagnostic symbols')
    loads = [line.strip() for line in isa.splitlines() if 'global_load_' in line]
    if sum('global_load_ushort' in line for line in loads) != 5 or sum('global_load_ushort' in line and 'glc dlc' in line for line in loads) != 4:
        raise ValueError('halfword load ISA changed')
    return dict(metadata=metadata, descriptor=descriptor, global_loads=loads, undefined_symbols=[])

def fixture_inputs(folder):
    folder = Path(folder).resolve(); receipt = folder/'receipt.json'
    report = json.loads(receipt.read_text())
    if report.get('classification') != CLASSIFICATION or report.get('status') != 'PASS' or report.get('gpu_executed') is not False or report.get('native_executed') is not False:
        raise ValueError('unqualified halfword diagnostic')
    original = Path(report['original_fixture']).resolve()
    if original == folder: raise ValueError('recursive diagnostic parent')
    parent = original_inputs(original)
    inputs = checked_inputs(report, 'inputs', [ROOT/name for name in SOURCES])
    if any(inputs.get(name) != digest for name,digest in parent.items()):
        raise ValueError('original fixture closure missing')
    tools = checked_tools(report, 'tools', TOOLS)
    for name,digest in tools.items():
        if inputs.get(name) != digest: raise ValueError('tool closure missing')
    artifacts = report.get('artifacts', {})
    if not set(REQUIRED) <= artifacts.keys(): raise ValueError('required diagnostic artifact missing')
    for name,row in artifacts.items():
        path = (folder/name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(folder) or path == receipt:
            raise ValueError('invalid artifact path')
        if sha(path) != row.get('sha256') or path.stat().st_size != row.get('bytes'):
            raise ValueError('diagnostic artifact changed')
        if str(path) in inputs and inputs[str(path)] != row['sha256']: raise ValueError('identity collision')
        inputs[str(path)] = row['sha256']
    if report.get('profile') != contracts(folder,original): raise ValueError('diagnostic profile changed')
    commands = report.get('commands', [])
    if [c.get('label') for c in commands] != ['compile','link','notes','isa','undefined'] or any(c.get('exit_code') != 0 for c in commands):
        raise ValueError('incomplete diagnostic commands')
    inputs[str(receipt)] = sha(receipt)
    return inputs

def discover_tool(name):
    found = shutil.which(name)
    if not found: raise ValueError('missing tool: '+name)
    # Preserve argv[0]: the multicall lld driver selects its flavour by this name.
    path = Path(found).absolute()
    version = subprocess.check_output([str(path),'--version'],text=True)
    if '18.1.8' not in version: raise ValueError('wrong tool version')
    return dict(path=str(path),sha256=sha(path),version=version)

def build(original, output):
    original, output = Path(original).resolve(), Path(output).resolve()
    if not any(output.is_relative_to(ROOT/name) for name in ('build','results')):
        raise ValueError('output must stay below build/ or results/')
    inputs = original_inputs(original)
    for name in SOURCES: bind(inputs,ROOT/name)
    bind(inputs,sys.executable)
    tools = {}
    for name in TOOLS:
        row = discover_tool(name)
        bind(inputs,row['path']); tools[name] = row
    output.mkdir(parents=True,exist_ok=False)
    (output/'q4.ll').write_text(transform((original/'q4.ll').read_text()))
    shutil.copyfile(original/'q8.hsaco', output/'q8.hsaco')
    commands = []
    def run(label, argv):
        argv = list(map(str,argv))
        try:
            result = subprocess.run(argv,cwd=ROOT,capture_output=True,timeout=180)
        except subprocess.TimeoutExpired as error:
            for suffix,data in [('stdout',error.stdout),('stderr',error.stderr)]: (output/(label+'.'+suffix)).write_bytes(data or b'')
            commands.append(dict(label=label,argv=argv,timed_out=True))
            (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            raise
        for suffix,data in [('stdout',result.stdout),('stderr',result.stderr)]: (output/('q4-'+label+'.'+suffix)).write_bytes(data)
        commands.append(dict(label=label,argv=argv,exit_code=result.returncode))
        (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if result.returncode: raise RuntimeError(label+' failed; logs preserved')
    run('compile',[tools['llc-18']['path'],'-mtriple=amdgcn-amd-amdhsa','-mcpu=gfx1013','-filetype=obj',output/'q4.ll','-o',output/'q4.o'])
    run('link',[tools['ld.lld-18']['path'],'-shared',output/'q4.o','-o',output/'q4.hsaco'])
    run('notes',[tools['llvm-readobj-18']['path'],'--notes',output/'q4.hsaco'])
    run('isa',[tools['llvm-objdump-18']['path'],'--disassemble','--mcpu=gfx1013',output/'q4.hsaco'])
    run('undefined',[tools['llvm-nm-18']['path'],'--undefined-only',output/'q4.hsaco'])
    profile = contracts(output,original)
    for name,digest in inputs.items():
        if sha(name) != digest: raise ValueError('input changed during build: '+name)
    report = dict(classification=CLASSIFICATION,status='PASS',gpu_executed=False,native_executed=False,
                  original_fixture=str(original),inputs=inputs,tools=tools,commands=commands,profile=profile,
                  artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in output.iterdir() if p.is_file()},
                  limits=['Diagnostic IR differs from upstream; volatile changes load width, cache flags and scheduling.', 'No numerical tolerance change or GPU qualification.'])
    (output/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    fixture_inputs(output)
    return report

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-fixture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.original_fixture,args.output)
    print('PASS offline halfword diagnostic fixture: '+str(args.output))
