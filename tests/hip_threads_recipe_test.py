#!/usr/bin/env python3
import argparse,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from probe_hip_threads import check,validate
class Boundaries(unittest.TestCase):
 def test_missing_or_false_completion(self):
  for text in ('','THREAD_MODEL_PASS context\n','THREAD_MODEL_PASS context\nTHREAD_MODEL_FINAL finalized=0 live=0\n'):
   with self.assertRaises(RuntimeError):validate(text,'context')
 def test_sanitizer_rejected(self):
  with self.assertRaises(RuntimeError):validate('THREAD_MODEL_PASS context\nTHREAD_MODEL_FINAL finalized=1 live=0\nruntime error: data race','context')
if __name__=='__main__':
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--check',type=pathlib.Path);a,rest=p.parse_known_args()
 if a.check:check(a.check.resolve())
 else:unittest.main(argv=[sys.argv[0],*rest])
