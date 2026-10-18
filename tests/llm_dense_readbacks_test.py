import hashlib,io,json,math,struct,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import verify_llm_dense_readbacks as m
class DenseReadbacks(unittest.TestCase):
 def test_strict_float_policy(self):
  f=lambda *x:struct.pack('<'+'f'*len(x),*x)
  for a,b,kwargs in [(f(2e-30),f(1e-30),{}),(f(1e-30),f(0),{}),(f(float('nan')),f(0),{}),(f(float('inf')),f(0),{}),(f(-0.),f(0.),{'exact':True}),(f(1,1e-7),f(1,0),{'probabilities':True})]:
   with self.assertRaises(ValueError):m.compare_values(a,b,**kwargs)
  with self.assertRaises(ValueError):m.compare_values(struct.pack('<2e',1,2**-24),struct.pack('<2e',1,0),half=True)
  m.compare_values(f(0),f(0));m.compare_values(f(1.00001),f(1))
 def test_attention_and_cache(self):
  state=m.Attention();state.query([0.]*2048);state.update([0.]*256,0,True);state.scores([0.]*8192,256)
  a=[0.]*8192;a[17]=1e-30
  with self.assertRaisesRegex(ValueError,'gamma64'):state.scores(a,256)
  state.update([1.]*256,2,True);state.checkpoint(bytes(state.k),True)
  wrong=bytearray(state.k);wrong[1025]=0
  with self.assertRaisesRegex(ValueError,'KV writes'):state.checkpoint(bytes(wrong),True)
  state.update([2.]*256,2,False);state.checkpoint(bytes(state.v),False);state.reset();self.assertEqual(state.k,bytes(262144))
 def test_stream_reset_noncheckpoint_and_integrity(self):
  def rows_for(records):
   rows=[];blob=b''
   for case,pos,stage,value in records:
    b=struct.pack('<f',value);rows.append(dict(case=case,position=pos,stage=stage,dtype='f32le',bytes=4,sha256=hashlib.sha256(b).hexdigest()));blob+=b
   return rows,blob
  records=[('sequence',0,f'out-{i:02}',1.) for i in range(21) if i not in (4,5,7,8)]+[('reset',0,f'out-{i:02}',1.) for i in range(21) if i not in (4,5,7,8)]
  rows,blob=rows_for(records);m.compare_stream(rows,io.BytesIO(blob),io.BytesIO(blob))
  changed=bytearray(blob);changed[-4:]=struct.pack('<f',1.00001)
  with self.assertRaisesRegex(ValueError,'reset bits'):m.compare_stream(rows,io.BytesIO(changed),io.BytesIO(blob))
  rows,blob=rows_for([('sequence',2,'out-03',1.)]);changed=struct.pack('<f',2.)
  with self.assertRaisesRegex(ValueError,'element budget'):m.compare_stream(rows,io.BytesIO(changed),io.BytesIO(blob))
  for a,b in [(blob[:-1],blob),(blob+b'x',blob),(blob,changed)]:
   with self.assertRaises(ValueError):m.compare_stream(rows,io.BytesIO(a),io.BytesIO(b))
 def test_exact_control_inventory(self):
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder);controls=p/'control.json';raw=p/'readback.bin';raw.touch();canonical=m.control_inventory()
   for mutate in [lambda d:d['records'][23].update(position=999),lambda d:d['records'][0].update(offset=4),lambda d:d['records'].pop(),lambda d:d.update(extra=1)]:
    value=json.loads(json.dumps(canonical));mutate(value);controls.write_text(json.dumps(value))
    with patch.object(m,'authenticate_reference',return_value={}):
     with self.assertRaisesRegex(ValueError,'control inventory'):m.compare(p,raw,controls,p/'out.json')
   controls.write_text('{"schema":1,"schema":1}')
   with self.assertRaisesRegex(ValueError,'duplicate'):m.read_json(controls)
 def test_identity_sweep(self):
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'file';p.write_bytes(b'a');bound={str(p):m.sha(p)};m.sweep(bound);p.write_bytes(b'b')
   with self.assertRaisesRegex(ValueError,'changed'):m.sweep(bound)
 def test_actual_reference_authentication(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);folder=root/'reference';folder.mkdir();cpu=root/'cpu';lib=cpu/'cmake/bin';lib.mkdir(parents=True)
   for n in m.SOURCES:
    path=root/n;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'source')
   cache=cpu/'cmake/CMakeCache.txt';cache.write_bytes(b'cache')
   for n in ('libggml-cpu.so','libggml-base.so'):(lib/n).write_bytes(b'library')
   compiler=str(Path('/usr/bin/clang++-18').resolve());original_sha=m.sha
   def test_sha(p):return 'a'*64 if str(p)==compiler else original_sha(p)
   inputs={str(root/n):test_sha(root/n) for n in m.SOURCES}
   inputs.update({str(p):test_sha(p) for p in (cache,Path(compiler),lib/'libggml-cpu.so',lib/'libggml-base.so')})
   flags=['-mavx2','-mfma','-mf16c','-std=c++17','-Wall','-Wextra','-Werror','-fno-fast-math','-ffp-contract=off','-D__HIP_PLATFORM_AMD__']
   flags+=['-I'+str(root/n) for n in ('ports/ggml','ports/hip','.deps/sources/hip-headers/include','.deps/sources/clr/hipamd/include','.deps/sources/llama-client/ggml/include')]
   commands=[]
   for name,source in [('default','tests/llm_dense_layer_default_test.cpp'),('checks','tests/llm_dense_layer_checks_test.cpp'),('failures','tests/llm_dense_layer_failure_test.cpp'),('admission','tests/llm_dense_layer_test.cpp'),('numeric','tests/llm_dense_layer_test.cpp')]:
    binary=folder/name;options=['-O2'] if name=='numeric' else ['-O1','-fsanitize=address,undefined','-fno-sanitize-recover=all','-fno-omit-frame-pointer']
    rows=[('compile-'+name,['/usr/bin/clang++-18',*flags,*options,str(root/source),'-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lggml-cpu','-lggml-base','-o',str(binary)]),('linkage-'+name,['/usr/bin/ldd',str(binary)]),(name,[str(binary),*(['admission'] if name=='admission' else ['numeric',str(folder/'reference.bin')] if name=='numeric' else [])])]
    for label,argv in rows:
     commands.append(dict(label=label,argv=argv,exit_code=0));(folder/(label+'.stderr')).touch()
     text=str(lib/'libggml-cpu.so')+' '+str(lib/'libggml-base.so') if label.startswith('linkage-') else 'DENSE_LAYER_NUMERIC_PASS requests=526 checked=11084 resident_f16_bytes=88080384 max_abs=0 GPU_execution=false' if label=='numeric' else ''
     (folder/(label+'.stdout')).write_text(text)
   raw=folder/'reference.bin';manifest=folder/'reference.json';raw.write_bytes(b'raw');manifest.write_bytes(b'{}')
   validated=dict(sha256=test_sha(raw),manifest_sha256=test_sha(manifest))
   data=dict(status='PASS',classification='synthetic-dense-layer-host-arithmetic',inputs=inputs,commands=commands,gpu_executed=False,native_executed=False,reference_export=dict(raw=str(raw),raw_sha256=validated['sha256'],manifest=str(manifest),manifest_sha256=validated['manifest_sha256'],records=11084,bytes=m.reference.BYTES))
   receipt=folder/'receipt.json'
   with patch.object(m,'ROOT',root),patch.object(m,'sha',side_effect=test_sha),patch.object(m.reference,'validate',return_value=validated) as validator:
    receipt.write_text(json.dumps(data));accepted=m.authenticate_reference(folder);self.assertIn(str(receipt),accepted['inputs']);self.assertIn(str(Path(m.__file__).resolve()),accepted['inputs'])
    for mutate in [lambda d:d['inputs'].pop(str(root/m.SOURCES[0])),lambda d:d['inputs'].update({str(root/m.SOURCES[0]):'b'*64}),lambda d:d['commands'][0].update(exit_code=1),lambda d:d['commands'][-1]['argv'].pop(),lambda d:d['reference_export'].update(raw_sha256='b'*64),lambda d:d.pop('reference_export')]:
     broken=json.loads(json.dumps(data));mutate(broken);receipt.write_text(json.dumps(broken))
     with self.assertRaises(ValueError):m.authenticate_reference(folder)
    receipt.write_text(json.dumps(data))
    def change_after_admission(*args):
     (root/m.SOURCES[0]).write_bytes(b'changed');return validated
    validator.side_effect=change_after_admission
    with self.assertRaisesRegex(ValueError,'changed'):m.authenticate_reference(folder)
if __name__=='__main__':unittest.main()

