if not __debug__: raise RuntimeError("Validation requires ordinary Python")
"""Strict native adapter link from independently qualified real CLR and SDK objects."""
from pathlib import Path
import hashlib,json,subprocess,shlex,shutil,functools,concurrent.futures,threading
C=Path(__file__).resolve().parents[1];R=C.parents[1];O=C/'results/native-title-v1';O.mkdir(parents=True,exist_ok=False)
A=R/'.local/g5-clr-ps5-adapter';S=C/'results/source-v1/source'
P=R/'.local/g4-clr-cpu-lifetime/results/native-title-lifetime-v1';Q=R/'results/clr-adapter-native-compile-v1'
F=R/'.local/g4-native-workers/results/workers-compiler-v2'
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text());inputs={};commands=[]
@functools.lru_cache(None)
def resolved(p):return Path(p).resolve()
def bind(p,expected=None):
 p=resolved(str(p));digest=inputs.get(str(p))
 if digest is None:digest=h(p);inputs[str(p)]=digest
 assert expected is None or digest==expected,str(p);return p
bind(__file__)
for directory,digest in [(P,'d2d39cd2899c0e32caa90c40cfd6424c1b41ddc09a6507846e29a1d5402f8721'),(Q,'6c9ea0a19ff43c7c0a5849359ad0f1c49225a11f8e9681616533da486a43d5b8')]:
 proof=bind(directory/'receipt.json',digest);report=read(proof)
 for name,sha in report['inputs'].items():bind(name,sha)
 for name,sha in report['artifacts'].items():bind(directory/name,sha)
bind(F/'receipt.json','f6c57047764720fbf6b61c7eb2d03939a914dc50a0942099d2b33a4a572f2d33')
compiler=read(F/'receipt.json')
for name,sha in compiler['sources'].items():bind(F.parents[1]/name,sha)
for name,sha in compiler['inputs'].items():bind(name,sha)
for entry in compiler['tools'].values():bind(entry['path'],entry['sha256'])
for name,sha in compiler['artifacts'].items():bind(F/name,sha)
code=bind(F/'ps5/threads.hsaco','0484405bc2a5247de40c41105a170026be1ea1802d60930fb0df56099dfc3dc0')
bind(__file__)
for name in ('ps5-device.cpp','ps5-device.hpp','ps5-device-unsupported.inc','ps5-virtual-unsupported.inc'):
 assert h(C/'ports/clr'/name)==h(A/'ports/clr'/name);bind(C/'ports/clr'/name)
for name in ('client.cpp','client.h','watchdog.cpp','watchdog.h'):bind(C/'tools/clr_gpu'/name)
bind(C/'tools/clr_gpu_title.cpp')
# Every upstream and AOT unit is rebuilt against the combined portable source.
prepared=read(bind(C/'results/source-v1/receipt.json'))
for name,digest in prepared['inputs'].items():bind(name,digest)
for name,digest in prepared['generated'].items():bind(S/name,digest)
cpu=read(P/'receipt.json');objects=[];reused=[];upstream=[]
for i in range(28):
 cmd=next(c['argv'] for c in cpu['commands'] if c['label']=='compile-'+str(i));old=Path(cmd[cmd.index('-c')+1])
 marker='/results/client-asan-v4/source/'
 source=S/str(old).split(marker,1)[1] if marker in str(old) else C/'ports/clr/ps5-clr-tls.cpp'
 upstream.append(source)
assert len(set(upstream))==28
crt=(P/'app_crt.cpp').read_text();assert 'PS5HIP_CLR_CPU_TITLE' in crt
(O/'app_crt.cpp').write_text(crt.replace('PS5HIP_CLR_CPU_TITLE','PS5HIP_CLR_GPU_TITLE'))
shutil.copyfile(P/'ps5-pie.ld',O/'ps5-pie.ld')
blob=code.read_bytes();(O/'code.cpp').write_text('#include <cstddef>\nalignas(16) extern const unsigned char ps5hip_clr_code[] = {'+','.join(str(v) for v in blob)+'};\nextern const size_t ps5hip_clr_code_size = sizeof(ps5hip_clr_code);\n')
q=read(Q/'receipt.json');cmd=next(c['argv'] for c in q['commands'] if c['label']=='0');flags=cmd[:cmd.index('-c')]
flags=[v.replace('-std=c++17','-std=c++20').replace(str(A/'results/clr-source-v7/source'),str(S)).replace(str(A/'include'),str(C/'include')).replace(str(A/'ports/clr'),str(C/'ports/clr')) for v in flags]
E=R/'.local/g4-multi-reduction-integration/build/native/gfx1013-hip_multi_query-integration-v2/elevation'
flags += ['-I'+str(C/'include'),'-I'+str(C/'tools'),'-I'+str(E/'client'),'-DPS5HIP_WITH_ELEVATION=1','-DPS5HIP_SHELL_EXIT_PROBE=1','-DPS5HIP_CLR_GPU_TITLE=1']
run_lock=threading.Lock()
def run(label,argv):
 z=subprocess.run(argv,capture_output=True,text=True,timeout=180);(O/(label+'.stdout')).write_text(z.stdout);(O/(label+'.stderr')).write_text(z.stderr)
 with run_lock:
  commands.append(dict(label=label,argv=argv,exit_code=z.returncode));(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 assert z.returncode==0,(label,z.stderr);return z.stdout
sources=upstream+[C/'ports/clr/ps5-device.cpp',C/'tools/clr_gpu/client.cpp',C/'tools/clr_gpu/watchdog.cpp',C/'tools/clr_gpu_title.cpp',O/'app_crt.cpp',O/'code.cpp']
sources += [C/'src'/(''+n+'.cpp') for n in ('code_object','fat_binary','host_launch','launch','native_memory','native_data','native_launch','submission','hip_launch_profile','agc_backend')]
sources += [C/'tools/native_shell_exit_probe.cpp',E/'client/elevation.cpp']
for source in sources:bind(source)
objects=[O/(str(i)+'.o') for i in range(len(sources))]
def compile_one(pair):
 i,source=pair;dep=O/(str(i)+'.d');opts=flags+(['-fno-inline'] if source.name=='app_crt.cpp' else [])
 run('headers-'+str(i),opts+['-M',str(source),'-MF',str(dep)])
 for name in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):bind(name)
 run('compile-'+str(i),opts+['-c',str(source),'-o',str(objects[i])]);print('compile',i,source.name,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(compile_one,enumerate(sources)))
sdk=R/'.deps/sdk/target/lib';libs=['libc++.a','libc++abi.a','libunwind.a','libc.a','libps5platform.a','libdl.a','libpthread.a'];imports=['libkernel.so','libSceLibcInternal.so','libSceNet.so','libSceSystemService.so','libSceAgc.so','libSceAgcDriver.so']
N=R/'.local/g4-multi-reduction-integration/build/native/gfx1013-hip_multi_query-integration-v2'
np=bind(N/'receipt.json');nr=read(np)
for name in ('libSceAgc.so','libSceAgcDriver.so'):bind(N/name,nr['raw_artifacts'][name])
provider=lambda name:(N if name.startswith('libSceAgc') else sdk)/name
for name in libs+imports:bind(provider(name))
for name in ('ld.lld-18','llvm-readelf-18','llvm-objdump-18'):bind(shutil.which(name))
(O/'symbols.map').write_text('{ local: *; };\n')
for p in (O/'symbols.map',O/'ps5-pie.ld'):bind(p)
unwind=[v for n in ['eh_frame','eh_frame_hdr'] for v in ['--defsym','__'+n+'_start=ADDR(.'+n+')','--defsym','__'+n+'_end=ADDR(.'+n+')+SIZEOF(.'+n+')']]
run('native-title-link',['ld.lld-18','-pie','--no-undefined','-T',str(O/'ps5-pie.ld'),'--eh-frame-hdr',*unwind,'-z','max-page-size=0x4000','--hash-style=gnu','--version-script',str(O/'symbols.map'),'--exclude-libs=ALL','-L',str(sdk),'-e','_start','-o',str(O/'title.elf'),*map(str,objects),'--start-group',*[str(sdk/n) for n in libs],'--end-group','--as-needed',*[str(provider(n)) for n in imports]])
run('symbols',['llvm-readelf-18','-Ws',str(O/'title.elf')]);run('segments',['llvm-readelf-18','-lW',str(O/'title.elf')]);run('startup',['llvm-objdump-18','-d','--demangle','--no-show-raw-insn',str(O/'title.elf')])
for name,sha in inputs.items():assert h(name)==sha,name
r=dict(classification='linked-offline-native-clr-gpu-adapter',inputs=inputs,commands=commands,reused_cpu_units=reused,reused_aot_objects=[],new_units=len(sources),artifacts={p.name:h(p) for p in O.iterdir() if p.is_file()},gpu_executed=False,limits=['Raw native PIE only; final provider/startup/converted image admission pending.','Client has not executed on PS5; backend imports are genuine AGC. All upstream and AOT units freshly compiled with combined startup patches.'])
(O/'receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(h(O/'receipt.json'))
