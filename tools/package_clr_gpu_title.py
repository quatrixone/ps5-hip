if not __debug__: raise RuntimeError("Validation requires ordinary Python")
from pathlib import Path
import argparse,hashlib,json,subprocess,shutil
C=Path(__file__).resolve().parents[1];R=C.parents[1]
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--results-dir-source',type=Path,required=True);ap.add_argument('--review',type=Path,required=True);ap.add_argument('--review-sha256',required=True);a=ap.parse_args()
 assert h(a.review)==a.review_sha256 and read(a.review).get('findings')==[]
 assert read(a.review).get('classification')=='clear-offline-native-clr-gpu-startup-candidate'
 baseline=a.results_dir_source.resolve();baseline_stage=baseline.parent.parent
 baseline_manifest=baseline_stage/'manifest.json';baseline_package=baseline_stage/'package-receipt.json';baseline_evidence=R/'evidence/native-clr-cpu-path-success.json'
 assert read(baseline_package)['image']['sha256']==read(baseline_evidence)['image_sha256']
 assert read(baseline_package)['manifest_sha256']==h(baseline_manifest)
 assert h(baseline)==read(baseline_manifest)['files']['results-dir.txt']['sha256']
 rd=baseline.read_text();assert 2<=len(rd)<=240 and rd.startswith('/') and all(32<=ord(c)<127 and c!='\\' for c in rd) and all(x not in ('','.','..') for x in rd[1:].split('/'))
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=C).strip()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=C,text=True).strip()
 Q=C/'results/offline-candidate-v2/receipt.json';q=read(Q)
 assert read(a.review)['exact_bindings'][str(Q)]==h(Q)
 for n,z in read(a.review)['exact_bindings'].items():assert h(Path(n) if Path(n).is_absolute() else R/n)==z,n
 for n,z in q['sources'].items():assert h(C/n)==z,n
 for n,z in q['proofs'].items():assert h(C/n)==z,n
 V=C/'results/native-conversion-v1';assert h(V/'eboot.bin')==q['native']['fself_sha256']
 supplier=R/'.local/g4-multi-reduction-integration/build/native/gfx1013-hip_multi_query-integration-v2';s=read(supplier/'receipt.json')
 files={'eboot.bin':(V/'eboot.bin').read_bytes()};assert hashlib.sha256(files['eboot.bin']).hexdigest()==q['native']['fself_sha256']
 for n in ['lapy.elf','sce_module/libc.prx']:
  p=supplier/'candidate'/n;assert h(p)==s['artifacts'][n];files[n]=p.read_bytes();assert hashlib.sha256(files[n]).hexdigest()==h(p)
 # Reuse the already validated identity metadata and icon, bound to its stage manifest.
 stages=list((supplier.parents[2]/'dist/PPSA99014').glob('gfx1013-hip_multi_query-*'));assert len(stages)==1
 prior=stages[0];pm=read(prior/'manifest.json')
 for n in ['sce_sys/param.json','sce_sys/icon0.png']:
  p=prior/'app'/n;assert h(p)==pm['files'][n]['sha256'];files[n]=p.read_bytes();assert hashlib.sha256(files[n]).hexdigest()==h(p)
 assert json.loads(files['sce_sys/param.json'])['titleId']=='PPSA99014'
 files['results-dir.txt']=rd.encode('ascii')
 stage=C/'dist/PPSA99014'/('gfx1013-clr-gpu-'+q['native']['fself_sha256'][:12]);stage.mkdir(parents=True,exist_ok=False)
 for n,data in files.items():p=stage/'app'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
 shutil.copyfile(Q,stage/'build-receipt.json')
 inputs={str(p):h(p) for p in [Path(__file__),a.review,Q,baseline,baseline_manifest,baseline_package,baseline_evidence,supplier/'receipt.json',prior/'manifest.json',C/'tools/package_native_title.py',C/'dependencies.json',C/'title.json']}
 manifest=dict(schema=1,title_id='PPSA99014',target='gfx1013',kernel='clr_gpu',source_commit=commit,build_receipt_sha256=h(Q),hardware_validated=False,expected_log=rd+'/ps5-hip-clr-gpu.log',inputs=inputs,files={n:dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)) for n,data in files.items()})
 (stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 subprocess.run(['python3',str(C/'tools/package_native_title.py'),'--stage',str(stage)],check=True,cwd=C)
 for n,z in inputs.items():assert h(n)==z,n
 print(stage)
if __name__=='__main__':main()
