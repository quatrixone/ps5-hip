import sys,tempfile,unittest,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_llm_full_graph_trace as trace
class TraceValidation(unittest.TestCase):
 def test_missing_steps_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   (Path(d)/'steps.jsonl').write_text('')
   with self.assertRaisesRegex(ValueError,'step count'):trace.validate(d)
 def test_failed_commit_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   rows=[dict(run=r,position=p,token=(0,1,2,31999)[p%4],status=0,scheduler_calls=1,committed_position=p) for r,n in ((0,512),(1,2)) for p in range(n)]
   rows[256]['committed_position']=255;(Path(d)/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
   with self.assertRaisesRegex(ValueError,'committed progression'):trace.validate(d)
if __name__=='__main__':unittest.main()
