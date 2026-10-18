"""Locked six-file external original-model contract for native title provisioning."""
import hashlib,json,re
from pathlib import Path
NAMES=('config.json','tokenizer.model','tokenizer_config.json','tokenizer.json','special_tokens_map.json','model.safetensors')
def remote_directory(value):
 if not isinstance(value,str) or len(value)>240 or not value.startswith('/data/hip/models/') or not re.fullmatch(r'/[A-Za-z0-9_./-]+',value):raise ValueError('original model directory')
 if any(p in ('','.','..') for p in value.split('/')[1:]):raise ValueError('original model directory components')
 return value
def contract(directory,remote,lock):
 directory=Path(directory);remote_directory(remote)
 if directory.is_symlink() or not directory.is_dir():raise ValueError('original source directory')
 expected={n:lock['metadata'][n] for n in NAMES if n!='model.safetensors'}
 expected['model.safetensors']={'bytes':lock['weights']['file_bytes'],'sha256':lock['weights']['expected_file_sha256']}
 files={}
 for name in NAMES:
  p=directory/name;item=expected[name]
  if p.is_symlink() or not p.is_file() or p.stat().st_size!=item['bytes']:raise ValueError('original source file size/kind')
  with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
  if digest!=item['sha256']:raise ValueError('original source digest')
  files[name]={'bytes':item['bytes'],'sha256':digest}
 return {'schema':1,'format':'original_safetensors','directory':remote,'files':files}
def encode(value):
 remote_directory(value['directory'])
 if set(value)!= {'schema','format','directory','files'} or type(value['schema'])is not int or value['schema']!=1 or value['format']!='original_safetensors' or set(value['files'])!=set(NAMES):raise ValueError('original source contract')
 for item in value['files'].values():
  if set(item)!= {'bytes','sha256'} or type(item['bytes'])is not int or item['bytes']<=0 or re.fullmatch('[0-9a-f]{64}',item['sha256']) is None:raise ValueError('original source identity')
 return (json.dumps(value,sort_keys=True,separators=(',',':'))+'\n').encode()
def validate(value,lock):
 encode(value)
 expected={n:{k:lock['metadata'][n][k] for k in ('bytes','sha256')} for n in NAMES if n!='model.safetensors'}
 expected['model.safetensors']={'bytes':lock['weights']['file_bytes'],'sha256':lock['weights']['expected_file_sha256']}
 if value['files']!=expected:raise ValueError('original source locked inventory differs')
 return value
