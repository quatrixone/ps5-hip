"""Exact bounded first-layer reference/readback file contract."""
import math,struct
from pathlib import Path
CHECKPOINTS=((0,(0,1,255,256,511)),(1,(0,1)))
def output_sizes(position):
 n=256 if position<256 else 512
 return [8192]*5+[1024]*3+[128*n]*2+[8192]*6+[22528]*3+[8192]*2

def inventory():
 files={}
 for run,positions in CHECKPOINTS:
  for pos in positions:
   for i,size in enumerate(output_sizes(pos)):files[f'run-{run}-pos-{pos}-out-{i:02d}.f32le']=size
   if pos==511 or (run==1 and pos==1):
    for name in ('k','v'):files[f'run-{run}-cache-{name}.f16le']=262144
 return files
FILES=inventory()
RAW_BYTES=sum(FILES.values())
assert len(FILES)==151

def read_raw(folder):
 folder=Path(folder)
 if {p.name for p in folder.iterdir()}!=set(FILES):raise ValueError('layer raw inventory')
 data={}
 for name,size in FILES.items():
  p=folder/name
  if p.is_symlink() or not p.is_file() or p.stat().st_size!=size:raise ValueError('layer raw kind/size')
  blob=p.read_bytes();fmt='f' if name.endswith('.f32le') else 'e';width=4 if fmt=='f' else 2
  if len(blob)!=size or not all(math.isfinite(x) for x in struct.unpack('<'+str(size//width)+fmt,blob)):raise ValueError('layer raw nonfinite/changed')
  data[name]=blob
 return data
