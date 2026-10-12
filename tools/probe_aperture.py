from pathlib import Path
import hashlib,json,re,shutil,subprocess,sys
root=Path(__file__).resolve().parents[1];out=root/'build/aperture-probes'
sys.path.insert(0,str(root/'tools'))
from build_canaries import require_device_linker,require_tool_versions
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
fp=lambda p:{'bytes':p.stat().st_size,'sha256':sha(p)}
def audit_output_isa(isa, opt):
    # Exact pinned-compiler instruction contract: 24*lane + [0,24) stores,
    # six output words in base.lo/base.hi/limit.lo/limit.hi/lane/xor order.
    # No general ISA emulator or hardware execution claim.
    actual = [' '.join(line.split('//')[0].split()) for line in isa.splitlines() if '// ' in line]
    while actual and actual[-1] == 's_code_end':
        actual.pop()
    expected = ISA_CONTRACTS[opt].splitlines()
    if actual != expected:
        raise ValueError('aperture bounded output ISA contract changed')

ISA_CONTRACTS = {
'O1': 's_load_dwordx2 s[0:1], s[4:5], null\nv_mov_b32_e32 v2, v0\ns_mov_b64 s[2:3], src_private_base\nv_mov_b32_e32 v5, s2\nv_mov_b32_e32 v6, s3\ns_mov_b64 s[4:5], src_private_limit\nv_mul_u32_u24_e32 v0, 6, v2\nv_mov_b32_e32 v1, s5\nv_xor_b32_e32 v3, 0x6b18d3a5, v2\nv_lshlrev_b32_e32 v4, 2, v0\nv_mov_b32_e32 v0, s4\nv_or_b32_e32 v7, 4, v4\ns_waitcnt lgkmcnt(0)\nglobal_store_dword v4, v5, s[0:1]\nglobal_store_dword v7, v6, s[0:1]\nglobal_store_dwordx4 v4, v[0:3], s[0:1] offset:8\ns_endpgm',
'O2': 's_load_dwordx2 s[0:1], s[4:5], null\nv_mul_u32_u24_e32 v1, 6, v0\ns_mov_b64 s[2:3], src_private_base\ns_mov_b64 s[4:5], src_private_limit\nv_mov_b32_e32 v2, s2\nv_mov_b32_e32 v3, s3\nv_mov_b32_e32 v4, s4\nv_mov_b32_e32 v5, s5\nv_lshlrev_b32_e32 v6, 2, v1\nv_xor_b32_e32 v1, 0x6b18d3a5, v0\ns_waitcnt lgkmcnt(0)\nglobal_store_dwordx4 v6, v[2:5], s[0:1]\nglobal_store_dwordx2 v6, v[0:1], s[0:1] offset:16\ns_endpgm',
}

def main():
    out.mkdir(parents=True,exist_ok=False)
    paths={n:Path(shutil.which(n+'-18')).resolve() for n in ('clang++','ld.lld','llvm-readobj','llvm-objdump')}
    for n in ('clang++','ld.lld'):
        for line in subprocess.check_output(['ldd',str(paths[n])],text=True).splitlines():
            m=re.search(r'=> (/\S+) ',line)
            if m and any(s in m[1] for s in ('libLLVM','libclang-cpp')):paths[Path(m[1]).name]=Path(m[1]).resolve()
    identities={n:{'resolved':str(p),**fp(p)} for n,p in paths.items()}
    versions={n:subprocess.check_output([shutil.which(n+'-18'),'--version'],text=True) for n in ('clang++','ld.lld','llvm-readobj','llvm-objdump')}
    require_tool_versions(versions,'18.1.8')
    (out/'tools-before.json').write_text(json.dumps({'identities':identities,'versions':versions},indent=2)+'\n')
    cases=[]
    for target in ('gfx1013','gfx1030'):
        for opt in ('O1','O2'):
            d=out/(target+'-'+opt);d.mkdir()
            base=[shutil.which('clang++-18'),'-x','hip','--cuda-device-only','--no-gpu-bundle-output','-B'+str(paths['ld.lld'].parent),'-nogpuinc','-nogpulib','--offload-arch='+target,'-mcode-object-version=5','-mno-wavefrontsize64','-'+opt,'-std=c++17','-fno-fast-math']
            commands=[]
            for name,extra in [('aperture.hsaco',[]),('aperture.s',['-S']),('aperture.ll',['-S','-emit-llvm'])]:
                cmd=base+extra+['-c',str(root/'kernels/private_aperture.hip'),'-o',str(d/name)];commands.append(cmd)
                if name.endswith('.hsaco'):
                    trace=subprocess.run(cmd+['-###'],capture_output=True,text=True,check=True).stderr
                    (d/'driver.txt').write_text(trace);require_device_linker(trace,str(paths['ld.lld']))
                r=subprocess.run(cmd,capture_output=True,text=True);(d/(name+'.log')).write_text(r.stdout+r.stderr);r.check_returncode()
            for name,cmd in [('metadata',[str(paths['llvm-readobj']),'--notes','--symbols','--relocations',str(d/'aperture.hsaco')]),('isa',[str(paths['llvm-objdump']),'-d','--mcpu='+target,str(d/'aperture.hsaco')])]:
                commands.append(cmd);(d/(name+'.txt')).write_text(subprocess.check_output(cmd,text=True))
            isa=(d/'isa.txt').read_text();asm=(d/'aperture.s').read_text();meta=(d/'metadata.txt').read_text()
            assert 'Relocations [\n]' in meta
            assert len(re.findall(r's_mov_b64 s\[\d+:\d+\], src_private_base',isa))==1
            assert len(re.findall(r's_mov_b64 s\[\d+:\d+\], src_private_limit',isa))==1
            forbidden=re.findall(r'^\s*((?:flat_|scratch_|buffer_|ds_|s_setreg|s_getreg|s_swappc|s_setpc)\w*)\b',isa,re.M)
            assert not forbidden,forbidden
            assert 'global_store' in isa and 'global_load' not in isa
            m=re.search(r'\.amdhsa_kernel private_aperture\n(.*?)\.end_amdhsa_kernel',asm,re.S);assert m
            desc={k:int(v,0) for k,v in re.findall(r'^\s*\.amdhsa_(\w+)\s+(\w+)',m[1],re.M)}
            assert desc['private_segment_fixed_size']==0 and desc['group_segment_fixed_size']==0 and desc['kernarg_size']==8
            assert desc['user_sgpr_flat_scratch_init']==0 and desc['uses_dynamic_stack']==0
            assert desc['user_sgpr_count']==6 and desc['wavefront_size32']==1
            assert 'hidden_' not in meta and all(int(v)==0 for v in re.findall(r'\.(?:sgpr|vgpr)_spill_count:\s+(\d+)',meta))
            audit_output_isa(isa,opt)
            (d/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
            cases.append({'target':target,'optimization':opt,'descriptor':desc,'files':{p.name:fp(p) for p in sorted(d.iterdir())},'aperture_read_instructions':[s.strip() for s in isa.splitlines() if 'src_private_' in s],'no_private_access_or_register_setup':True})
            print(target,opt,fp(d/'aperture.hsaco'),flush=True)
    after={n:{'resolved':str(p),**fp(p)} for n,p in paths.items()};assert after==identities
    (out/'tools-after.json').write_text(json.dumps(after,indent=2)+'\n')
    receipt={'schema':1,'classification':'offline read-only private aperture register diagnostic','hardware_validated':False,'native_linked':False,'console_access':False,'production_changed':False,'tool_versions':versions,'source_sha256':sha(root/'kernels/private_aperture.hip'),'cases':cases,'inputs':{p.name:fp(p) for p in (root/'kernels/private_aperture.hip',Path(__file__),out/'tools-before.json',out/'tools-after.json')},'launch_proposal':{'groups':[1,1,1],'threads':[64,1,1],'output_words':384,'arguments':'one global output pointer; no input pointer or hidden arguments','per_lane_output':['base.low32','base.high32','limit.low32','limit.high32','lane','lane xor 0x6b18d3a5'],'validation':'fresh poisoned guarded output; exact lane and xor witnesses for all64 lanes; all prefix and full rounded suffix unchanged; report raw aperture words per lane and flag any disagreement within/across the two waves; do not hardcode aperture addresses'},'limits':['No private pointer creation, private load/store, flat scratch setup, or scratch ring allocation is emitted. Global output writes are intentional.','Only __builtin_amdgcn_workitem_id_x is used. Aperture reads are explicit 64-bit s_mov_b64 inline assembly, not an s_getreg builtin; SRC_PRIVATE operands are not HW_REG selectors.','Transport validation does not establish valid aperture routing. Zero/inconsistent/colliding values require diagnosis, not guessed setup or a private access retry.','Even plausible stable aperture values do not qualify FLAT_SCR_LO/HI behavior, scratch backing, SPI offsets, private load/store, nested calls, or PS5 compute support.','No native image or console run is authorized by this receipt.'], 'recipe_sha256':sha(Path(__file__))}
    (out/'comparison.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('receipt',fp(out/'comparison.json'))


def verified_aperture_object(root, target, version):
    directory = root / 'build/aperture-probes'
    report = json.loads((directory / 'comparison.json').read_text())
    if report['recipe_sha256'] != sha(root / 'tools/probe_aperture.py') or report['source_sha256'] != sha(root / 'kernels/private_aperture.hip'):
        raise ValueError('aperture source or recipe changed')
    require_tool_versions(report['tool_versions'], version)
    profiles = [p for p in report['cases'] if p['target'] == target and p['optimization'] == 'O2']
    if len(profiles) != 1:
        raise ValueError('missing or duplicate aperture O2 target')
    path = directory / (target + '-O2')
    for name in ('aperture.hsaco', 'driver.txt'):
        if fp(path / name) != profiles[0]['files'][name]:
            raise ValueError('aperture artifact or driver trace changed')
    linker = shutil.which('ld.lld-18')
    if not linker:
        raise ValueError('pinned device linker unavailable')
    require_device_linker((path / 'driver.txt').read_text(), linker)
    return (path / 'aperture.hsaco').read_bytes(), report

if __name__ == '__main__':
    main()
