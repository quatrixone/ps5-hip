#!/usr/bin/env python3
"""Isolated one-member LLVM relink and natural-exp cutoff experiment; no GPU."""
import argparse,difflib,hashlib,json,os,pathlib,re,shlex,shutil,struct,subprocess
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
MEMBER='AMDGPUISelLowering.cpp.o'
OBJECT='lib/Target/AMDGPU/CMakeFiles/LLVMAMDGPUCodeGen.dir/'+MEMBER
ARCHIVE='lib/libLLVMAMDGPUCodeGen.a'
OLD='-0x1.9d1da0p+6f'
NEW='-0x1.9fe368p+6f'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def require(yes,why):
    if not yes:raise ValueError(why)
def dependencies(path):
    text=path.read_text().replace('\\\n',' ')
    return [pathlib.Path(x) for x in shlex.split(text.split(':',1)[1])]
def patch_source(text):
    require(text.count(OLD)==1,'natural-exp cutoff uniqueness')
    return text.replace(OLD,NEW)
def function(text,name):
    m=re.search(r'^'+re.escape(name)+r':.*?^\.Lfunc_end\d+:',text,re.M|re.S)
    require(m is not None,'assembly function missing '+name)
    return m[0]
def audit_matrix(before,noop,after):
    require(before==noop,'no-op relink changed compiler output')
    expected=before
    for name in ('exp10_scalar','exp_fast','unrelated_control'):
        require(function(before,name)==function(after,name),'unrelated lowering changed '+name)
    for name in ('exp_scalar','exp_vector'):
        a,b=function(before,name),function(after,name)
        require('0xc2ce8ed0' in a and '0xc2cff1b4' in b,'exact comparison missing '+name)
        require(a.replace('0xc2ce8ed0','0xc2cff1b4')==b,'change beyond cutoff '+name)
        require('v_exp_f32' in b and 'v_ldexp_f32' in b,'EXP/FLDEXP path missing')
        expected=expected.replace(a,b,1)
    require(expected==after,'assembly changed outside the two natural-exp function literals')
    require('.amdhsa_float_denorm_mode_32 3' in after,'IEEE denormal descriptor missing')
    return {'no_op_exact':True,'only_natural_exp_literal_changed':True,'exp10_fast_unrelated_exact':True}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=pathlib.Path,required=True)
    p.add_argument('--reference-root',type=pathlib.Path,required=True)
    p.add_argument('--output',type=pathlib.Path,required=True)
    args=p.parse_args();baseline=args.baseline.resolve();ref=args.reference_root.resolve();out=args.output.resolve()
    require(not out.exists(),'fresh output required');require(out.is_relative_to(ROOT/'results'),'output must be isolated results')
    out.mkdir(parents=True);inputs={};commands=[];report={'schema':1,'hardware_validated':False,'oracle_changed':False}
    def bind(path):
        path=pathlib.Path(path).resolve();require(path.is_file(),'input missing '+str(path))
        digest=sha(path);require(str(path) not in inputs or inputs[str(path)]==digest,'input changed '+str(path));inputs[str(path)]=digest
        return path
    def run(label,argv,cwd=out,timeout=600):
        argv=list(map(str,argv));print('START '+label,flush=True)
        z=subprocess.run(argv,cwd=cwd,capture_output=True,timeout=timeout)
        (out/(label+'.stdout')).write_bytes(z.stdout);(out/(label+'.stderr')).write_bytes(z.stderr)
        commands.append({'label':label,'argv':argv,'cwd':str(cwd),'exit':z.returncode})
        (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        require(z.returncode==0,label+' failed: '+z.stderr.decode(errors='replace')[-2000:]);return z.stdout.decode()
    try:
        source=bind(baseline.parent/'llvm-project/llvm/lib/Target/AMDGPU/AMDGPUISelLowering.cpp')
        for name in ('build.ninja','CMakeFiles/rules.ninja','CMakeCache.txt',OBJECT,ARCHIVE,'bin/llc'):bind(baseline/name)
        for name in ('probe.py','certificate.c','exp-controls.ll'):bind(HERE/name)
        oracle=bind(ref/'.local/g3-math-current/tools/math_oracle_contract.json')
        require(sha(oracle)=='373c94b8ad8dac7326401ad54a1299d750298a5922c1a226d0d4816c854430b8','frozen oracle mismatch')
        bind(ref/'results/math-underflow-offline-v1/receipt.json')
        tools={n:pathlib.Path(shutil.which(n)) for n in ('clang-18','clang++-18','ld.lld-18','llvm-ar-18','llvm-readobj-18','llvm-objdump-18','ninja')}
        # LLVM driver personality is selected by argv[0]; bind resolved bytes
        # while invoking the original clang++/ld.lld symlink name.
        for path in tools.values():bind(path)
        for n,t in tools.items():run(n+'-version',[t,'--version'])
        # Independent certificate uses existing pinned header/runtime, no regenerated oracle.
        mpfr=bind(ref/'results/ocml-numerical-design-v1/mpfr.h');require(sha(mpfr)=='588c2bd606cba060afb49cd024f9119e996089a341a99d1dc7e0367622335a13','MPFR header mismatch')
        ml=bind('/usr/lib/x86_64-linux-gnu/libmpfr.so.6');gl=bind('/usr/lib/x86_64-linux-gnu/libgmp.so.10')
        cert=out/'certificate';dep=out/'certificate.d'
        cc=[tools['clang-18'],'-std=c11','-O2','-fno-fast-math','-Wall','-Wextra','-Werror','-I'+str(mpfr.parent),'-MD','-MF',dep,HERE/'certificate.c',ml,gl,'-o',cert]
        # Capture preprocessor inputs before compiling the certificate.
        pre=cc.copy();pre.remove('-MD');pre.remove(ml);pre.remove(gl);pre[pre.index('-o')+1]=out/'unused-preprocess';pre+=['-M']
        run('certificate-dependencies',pre)
        for path in dependencies(dep):bind(path)
        run('certificate-compile',cc);rows=[json.loads(x) for x in run('certificate-run',[cert]).splitlines()]
        require(rows[0]['first_above_bits']=='c2cff1b4','cutoff certificate mismatch')
        (out/'certificate.json').write_text(json.dumps(rows,indent=2)+'\n')
        report['certificate']=rows
        # Read Ninja commands only; bypass shared ccache and never run a build target.
        raw_compile=run('read-compile-command',[tools['ninja'],'-t','commands',OBJECT],baseline).splitlines()[-1]
        tokens=shlex.split(raw_compile);start=tokens.index('/usr/bin/clang++-18');compile_command=tokens[start:]
        raw_link=run('read-link-command',[tools['ninja'],'-t','commands','bin/llc'],baseline).splitlines()[-1]
        require(raw_link.startswith(': && ') and raw_link.endswith(' && :'),'unexpected link shell structure')
        link_command=[x.replace('\\$ORIGIN','$ORIGIN') for x in shlex.split(raw_link[5:-5])]
        require(link_command.count(ARCHIVE)==1,'archive selection ambiguity')
        for token in link_command[1:]:
            path=pathlib.Path(token)
            if token.endswith(('.a','.o','.so')):bind(path if path.is_absolute() else baseline/path)
        # Bind loaded baseline runtime libraries as the explicit non-hermetic boundary.
        loaded=run('baseline-loaded-libraries',['ldd',baseline/'bin/llc'])
        for path in re.findall(r'(/[A-Za-z0-9_./+-]+)',loaded):
            if pathlib.Path(path).is_file():bind(path)
        members=run('baseline-archive-members',[tools['llvm-ar-18'],'t',baseline/ARCHIVE]).splitlines()
        require(len(members)==len(set(members)) and members.count(MEMBER)==1,'archive member ambiguity')
        def member_hashes(path):
            return {n:hashlib.sha256(subprocess.check_output([tools['llvm-ar-18'],'p',path,n])).hexdigest() for n in members}
        member_before=member_hashes(baseline/ARCHIVE)
        def object_command(directory,src):
            command=compile_command.copy()
            for flag,value in (('-o',directory/MEMBER),('-MF',directory/'headers.d'),('-MT',directory/MEMBER)):
                command[command.index(flag)+1]=str(value)
            require(command[-2]=='-c' and pathlib.Path(command[-1])==source,'source selection changed');command[-1]=str(src)
            return command
        # Before-use header identity, including generated and system headers.
        noop=out/'noop';noop.mkdir();cmd=object_command(noop,source);pre=cmd.copy();pre[pre.index('-MD')]='-M';pre.remove('-c')
        run('lowering-dependencies',pre,baseline)
        for path in dependencies(noop/'headers.d'):bind(path)
        def build(directory,src):
            run(directory.name+'-compile',object_command(directory,src),baseline)
            archive=directory/'libLLVMAMDGPUCodeGen.a';shutil.copyfile(baseline/ARCHIVE,archive)
            require(archive.read_bytes().startswith(b'!<arch>\n'),'normal archive required')
            run(directory.name+'-archive',[tools['llvm-ar-18'],'r',archive,directory/MEMBER])
            require(run(directory.name+'-members',[tools['llvm-ar-18'],'t',archive]).splitlines()==members,'archive member order changed')
            hashes=member_hashes(archive)
            require(all(hashes[n]==member_before[n] for n in members if n!=MEMBER),'unrelated member changed')
            require(hashes[MEMBER]==sha(directory/MEMBER),'replacement member mismatch')
            command=link_command.copy();command[command.index(ARCHIVE)]=str(archive);command[command.index('-o')+1]=str(directory/'llc')
            flags=[i for i,x in enumerate(command) if x.startswith('--dependency-file=')];require(len(flags)==1,'link dep destination')
            command[flags[0]]='--dependency-file='+str(directory/'link.d')
            run(directory.name+'-link',command,baseline)
            return {'archive_members':hashes,'object_sha256':sha(directory/MEMBER),'llc_sha256':sha(directory/'llc')}
        report['noop']=build(noop,source)
        matrix={}
        for target in ('gfx1013','gfx1030'):
            for opt in ('O1','O2'):
                key=target+'-'+opt;matrix[key]={}
                for mode,compiler in (('baseline',baseline/'bin/llc'),('noop',noop/'llc')):
                    dest=out/(key+'-'+mode+'.s');run(key+'-'+mode,[compiler,'-march=amdgcn','-mcpu='+target,'-mattr=+wavefrontsize32,-wavefrontsize64','-'+opt,HERE/'exp-controls.ll','-o',dest])
                    matrix[key][mode]=dest.read_text()
                require(matrix[key]['baseline']==matrix[key]['noop'],'no-op matrix differs')
        print('NOOP_MATRIX_PASS',flush=True)
        patched=out/'patched';patched.mkdir();text=source.read_text();patched_source=patched/'AMDGPUISelLowering.cpp';new=patch_source(text);patched_source.write_text(new)
        (out/'natural-exp-cutoff.patch').write_text(''.join(difflib.unified_diff(text.splitlines(True),new.splitlines(True),fromfile='a/llvm/lib/Target/AMDGPU/AMDGPUISelLowering.cpp',tofile='b/llvm/lib/Target/AMDGPU/AMDGPUISelLowering.cpp')))
        report['patched']=build(patched,patched_source);report['matrix']={}
        for key,versions in matrix.items():
            target,opt=key.split('-');dest=out/(key+'-patched.s')
            run(key+'-patched',[patched/'llc','-march=amdgcn','-mcpu='+target,'-mattr=+wavefrontsize32,-wavefrontsize64','-'+opt,HERE/'exp-controls.ll','-o',dest])
            report['matrix'][key]=audit_matrix(versions['baseline'],versions['noop'],dest.read_text())
        require(all(sha(path)==digest for path,digest in inputs.items()),'consumed inputs changed')
        report.update(classification='passed-isolated-exp-cutoff-compiler-matrix',inputs=inputs,commands=commands,
            limits=['Supplied source-built LLVM objects/archives relinked on one host; not a clean hermetic build.',
                    'MPFR certifies reference rounding and cutoff; no native EXP/FLDEXP or GPU numerical execution.',
                    'Frozen six-operation device-object regeneration remains a separate next qualification.',
                    'No baseline/private compiler mutation; no oracle or hardware acceptance change.'])
        report['artifacts']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
        (out/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS '+sha(out/'receipt.json'),flush=True)
    except BaseException as error:
        report.update(classification='failed-preserved',error=repr(error),inputs=inputs,commands=commands)
        (out/'failure.json').write_text(json.dumps(report,indent=2)+'\n');raise
if __name__=='__main__':main()
