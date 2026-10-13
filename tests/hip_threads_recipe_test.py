#!/usr/bin/env python3
import argparse,copy,pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'tools'))
from probe_hip_threads import ROOT,MODES,check,check_inventory,source_files,validate
class Boundaries(unittest.TestCase):
 def test_missing_or_false_completion(self):
  for text in ('','THREAD_MODEL_PASS context\n','THREAD_MODEL_PASS context\nTHREAD_MODEL_FINAL finalized=0 live=0\n'):
   with self.assertRaises(RuntimeError):validate(text,'context')
 def test_sanitizer_rejected(self):
  with self.assertRaises(RuntimeError):validate('THREAD_MODEL_PASS context\nTHREAD_MODEL_FINAL finalized=1 live=0\nruntime error: data race','context')
 def test_inventory_omissions(self):
  labels=['hip-compile','metadata','isa','host-isa','runtime-compile','model-link',*('run-'+m for m in MODES)]
  artifacts={'commands.json','threads.o','threads.bundle','threads.hsaco','runtime.o','linux.ld','threads'}
  artifacts.update(label+'.'+ext for label in labels for ext in ('stdout','stderr'))
  r={'schema':1,'sanitizer':'address,undefined','modes':list(MODES),
     'sources':{str(p.relative_to(ROOT)):'hash' for p in source_files()},
     'inputs':{'compiler':'hash'},'artifacts':{p:'hash' for p in artifacts},
     'commands':[{'label':label,'exit_code':0} for label in labels]}
  check_inventory(r)
  for field in ('sources','inputs','artifacts'):
   bad=copy.deepcopy(r);bad[field]={}
   with self.subTest(empty=field),self.assertRaises(RuntimeError):check_inventory(bad)
  for field in ('sources','artifacts'):
   for key in r[field]:
    bad=copy.deepcopy(r);del bad[field][key]
    with self.subTest(field=field,missing=key),self.assertRaises(RuntimeError):check_inventory(bad)
  for index in range(len(labels)):
   bad=copy.deepcopy(r);bad['commands'][index]['exit_code']=1
   with self.subTest(failed=labels[index]),self.assertRaises(RuntimeError):check_inventory(bad)
if __name__=='__main__':
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--check',type=pathlib.Path);a,rest=p.parse_known_args()
 if a.check:check(a.check.resolve())
 else:unittest.main(argv=[sys.argv[0],*rest])
