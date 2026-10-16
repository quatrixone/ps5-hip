from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import qualify_public_hip_q4 as q

class Validation(unittest.TestCase):
 def test_profiles_reach_full_authentication(self):
  for kind,names,source in [('getrows',('getrows',),'getrows.cu'),('q4-q8',('q8','q4'),'mmvq.cu')]:
   r={'classification':'source-only-selected-upstream-'+kind+'-fixture','status':'PASS','native_executed':False,'gpu_executed':False,'profiles':{n:{} for n in names}}
   with tempfile.TemporaryDirectory() as tmp:
    p=Path(tmp);(p/'receipt.json').write_text(json.dumps(r))
    with patch.object(q,'checked_inputs',side_effect=RuntimeError('full-authentication')) as check:
     with self.assertRaisesRegex(RuntimeError,'full-authentication'):q.fixture_inputs(p)
     required=check.call_args.args[2]
     self.assertIn(q.ROOT/'.deps/sources/llama-client/ggml/src/ggml-cuda'/source,required)
     if kind=='getrows':self.assertIn(q.ROOT/'tests/llm_getrows_fixture_test.py',required)
    for field,value in [('status','FAIL'),('native_executed',True),('gpu_executed',True),('profiles',{}),('profiles',{'getrows':{},'q8':{}}),('classification','unknown')]:
     changed=dict(r);changed[field]=value;(p/'receipt.json').write_text(json.dumps(changed))
     with self.subTest(kind=kind,field=field,value=value),patch.object(q,'checked_inputs') as check:
      with self.assertRaises(ValueError):q.fixture_inputs(p)
      check.assert_not_called()

if __name__=='__main__':unittest.main()
