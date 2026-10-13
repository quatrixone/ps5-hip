"""Exact plural compiler admission; full native ELF audit follows actual link."""
import argparse,pathlib,sys
R=pathlib.Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'tools'),str(R/'tests')]
from native_multi_query import PROFILE,validate_compiler
from native_vector_title import profile_for,provenance_files
def main():
 p=argparse.ArgumentParser();p.add_argument('--compiler',type=pathlib.Path,required=True);a=p.parse_args();proof=a.compiler.resolve()
 raw=(proof/'receipt.json').read_bytes();objects=[(proof/'ps5'/(m+'.o')).read_bytes() for m in 'ab']
 assert profile_for('hip_multi_query')==PROFILE and {'multi-a.o','multi-b.o','compiler-receipt.json'}<=provenance_files('hip_multi_query')
 validate_compiler(raw,objects,R)
 cases=[(raw,objects[:1]),(raw,objects[::-1]),(raw,[objects[0],objects[0]]),(raw,[objects[0][:-1],objects[1]]),(raw+b' ',objects)]
 for receipt,modules in cases:
  try:validate_compiler(receipt,modules,R)
  except ValueError:pass
  else:raise AssertionError('incorrect compiler identity accepted')
 print('PASS exact two-module compiler admission and five substitutions rejected')
if __name__=='__main__':main()
