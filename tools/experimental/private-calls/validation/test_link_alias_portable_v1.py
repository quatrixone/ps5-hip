"""Mutated copies only: actual linked bytes and matching disassembly operands."""
from pathlib import Path
import hashlib,json,re,struct
from link_alias_audit_v1 import parse,Elf
p=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def mutate(data,isa,needle,new_text,new_words):
 lines=isa.splitlines();matches=[(i,re.fullmatch(r'\s*(.*?)\s+//\s+([0-9A-Fa-f]+):\s+([0-9A-Fa-f ]+?)\s*',line)) for i,line in enumerate(lines) if needle in line]
 assert len(matches)==1;i,m=matches[0];assert m
 address=int(m[2],16);words=m[3].split();assert len(words)==len(new_words)
 elf=Elf(data);offsets=[offset+address-base for base,length,offset in elf.loads if base<=address and len(words)*4<=length-(address-base)];assert len(offsets)==1
 changed=bytearray(data);encoded=b''.join(struct.pack('<I',word) for word in new_words);changed[offsets[0]:offsets[0]+len(encoded)]=encoded
 lines[i]='\t'+new_text+' // '+m[2]+': '+' '.join(f'{word:08X}' for word in new_words)
 return bytes(changed),'\n'.join(lines)+'\n'
rows=[]
for target in ('gfx1013','gfx1030'):
 d=p.parents[1]/'linked-alias-audit'/target/'patched-on';data=(d/'linked.hsaco').read_bytes();isa=(d/'isa.txt').read_text();parse(data,isa)
 low=re.search(r's_add_u32 s4, s4, (0x[0-9a-f]+).*?// [0-9A-F]+: ([0-9A-F]+) ([0-9A-F]+)',isa);assert low
 wrong_delta=(int(low[1],16)+4)&0xffffffff
 specs=[('target-inside-safe','s_add_u32 s4, s4,',f's_add_u32 s4, s4, {hex(wrong_delta)}',[int(low[2],16),wrong_delta]),
 ('wrong-high-sign','s_addc_u32 s5, s5,','s_addc_u32 s5, s5, 0',[0x8205ff05,0]),
 ('wrong-return-pair','s_swappc_b64','s_swappc_b64 s[28:29], s[4:5]',[0xbe9c2104]),
 ('wrong-source-pair','s_swappc_b64','s_swappc_b64 s[30:31], s[6:7]',[0xbe9e2106])]
 for label,needle,new_text,words in specs:
  changed,listing=mutate(data,isa,needle,new_text,words)
  try:parse(changed,listing)
  except AssertionError:pass
  else:raise AssertionError((target,label,'matched-byte invalid relocation accepted'))
 rows.append({'target':target,'linked_sha256':sha(d/'linked.hsaco'),'isa_sha256':sha(d/'isa.txt'),'matched_byte_mutations_rejected':[spec[0] for spec in specs]})
out=p/'alias-linked-v1-mutation-tests.json';assert not out.exists();out.write_text(json.dumps({'cases':rows,'hardware_validated':False,'actual_llc_invocations':0,'script_sha256':sha(Path(__file__)),'checker_sha256':sha(p/'link_alias_audit_v1.py')},indent=2)+'\n');print('PASS8byteboundmutations',sha(out))
