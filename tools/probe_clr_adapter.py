from pathlib import Path
import argparse,json,subprocess,hashlib
p=argparse.ArgumentParser();p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[1];B=a.reference_root.resolve();O=a.output.resolve();O.mkdir(parents=True,exist_ok=False)
P=R/'results/clr-source-v1';old=B/'results/clr-native-combined-v6'
receipt=json.loads((B/'results/clr-combined-workload-v1/receipt.json').read_text())
argv=next(c['argv'] for c in receipt['commands'] if c['label']=='linux-compile-0');flags=argv[:argv.index('-c')]
flags=[x.replace(str(old/'source'),str(P/'source')) for x in flags]+['-DWITH_PS5_DEVICE','-I'+str(R/'include'),'-I'+str(R/'ports/clr')]
commands=[]
def run(label,args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=180);(O/(label+'.stdout')).write_text(r.stdout);(O/(label+'.stderr')).write_text(r.stderr);commands.append(dict(label=label,argv=args,exit_code=r.returncode));(O/'commands.json').write_text(json.dumps(commands,indent=2)+'\n');assert r.returncode==0,r.stderr[-6000:]
for label,source in [('adapter',R/'ports/clr/ps5-device.cpp'),('device-program',P/'source/rocclr/device/devprogram.cpp'),('platform-program',P/'source/rocclr/platform/program.cpp')]:
 run(label,flags+['-c',str(source),'-o',str(O/(label+'.o'))])
print('ADAPTER_OBJECTS_PASS')
native=json.loads((B/'results/clr-native-combined-v6/receipt.json').read_text())
objects=[]
for i,unit in enumerate(native['units']):
 suffix=unit['unit']
 if suffix.endswith('device/devprogram.cpp'):obj=O/'device-program.o'
 elif suffix.endswith('platform/program.cpp'):obj=O/'platform-program.o'
 elif suffix.endswith('os/os_posix.cpp'):obj=B/'results/clr-rwx-qualified-v7/os-linux.o'
 else:obj=B/'results/clr-combined-workload-v1'/f'{i}.o'
 objects.append(str(obj))
lock=json.loads((R/'dependencies.json').read_text());sdk=B/'.deps'/('sdk-source-'+lock['sdk']['revision'])/'platform/include'
flags+=['-I'+str(sdk)]
core=['code_object','fat_binary','host_launch','launch','native_memory','native_data','native_launch','submission','hip_launch_profile']
for name in core:
 run(name,flags+['-c',str(R/'src'/(name+'.cpp')),'-o',str(O/(name+'.o'))]);objects.append(str(O/(name+'.o')))
objects.append(str(O/'adapter.o'))
run('link',flags+['-fuse-ld=lld-18',str(R/'tests/clr_adapter_model.cpp')]+objects+['-Wl,-z,defs','-o',str(O/'model')])
artifact=B/'.local/g4-native-workers/results/workers-compiler-v2/ps5/threads.hsaco'
for mode in ['normal','submit','throw','kick','timeout','postflush','cleanup']:
 run(mode,[str(O/'model'),str(artifact),mode])
print('ADAPTER_MODEL_PASS')
