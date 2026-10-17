import sys,json,shutil,tempfile
from pathlib import Path
import argparse
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'tools'))
import build_llm_full_graph_trace as trace
parser=argparse.ArgumentParser(description='Validate full-graph raw trace and reject isolated tampering.')
parser.add_argument('--raw',type=Path,required=True);source=parser.parse_args().raw.resolve();results=[]
trace.validate(source);results.append('actual inventory PASS')
for mode in ('node-id','source-id','compute-flag','out-id','position','reset-logit','extra-file'):
 with tempfile.TemporaryDirectory() as d:
  folder=Path(d)/'raw';shutil.copytree(source,folder);path=folder/'run-0-pos-0.graph.json';g=json.loads(path.read_text())
  if mode=='node-id':g['tensors'][0]['id']=9999
  elif mode=='source-id':g['tensors'][0]['sources'][0]=9999
  elif mode=='compute-flag':g['tensors'][1]['compute']=True
  elif mode=='out-id':
   t=next(t for t in g['tensors'] if t['name']=='out_ids');(folder/t['control_file']).write_bytes(b'\x01\0\0\0')
  elif mode=='position':
   rows=[json.loads(s) for s in (folder/'steps.jsonl').read_text().splitlines()];rows[256]['committed_position']=255;(folder/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
  elif mode=='reset-logit':
   q=folder/'run-1-pos-0.logits.f32le';raw=bytearray(q.read_bytes());raw[0]^=1;q.write_bytes(raw)
  else:(folder/'unexpected').write_bytes(b'x')
  path.write_text(json.dumps(g)+'\n')
  try:trace.validate(folder)
  except (ValueError,KeyError):results.append(mode+' rejected')
  else:raise AssertionError(mode+' accepted')
print(json.dumps({'status':'PASS','cases':results},indent=2))
