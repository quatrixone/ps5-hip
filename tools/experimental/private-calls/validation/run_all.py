"""Sequential acceptance preparations; never builds LLVM or accesses a console."""
from pathlib import Path
import argparse,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--compiler',required=True);p.add_argument('--mode',choices=['baseline','patched-off','patched-on'],required=True);args=p.parse_args()
root=Path(__file__).resolve().parent
cases=[('run_matrix_v2.py',[]),('run_negatives.py',[]),('run_stress.py',['--expect-proof','rejected']),('run_stress.py',['--fixture','late-callee-unbounded','--expect-proof','proven']),('run_reload.py',[]),('run_ineligible.py',[])]
for script,extra in cases:
 subprocess.run([sys.executable,str(root/script),'--compiler',args.compiler,'--mode',args.mode]+extra,check=True)
print('All selected controls completed; this is not complete compiler acceptance or hardware qualification.')
