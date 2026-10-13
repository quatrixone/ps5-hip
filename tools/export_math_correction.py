#!/usr/bin/env python3
"""Export the one independently reviewed OCML correction; no compilation or GPU."""
import argparse,json,pathlib,shutil
import math_native_fixture as fixture
ROOT=pathlib.Path(__file__).resolve().parents[1]
EXPECTED={
 'results/readiness-v1/receipt.json':'ce50098d6d0e622b014565f6720da3919fd2b5d5135837960db5dd8efb71da5f',
 'results/freeze-v1/receipt.json':'2bbda06fab0dfb614400ab161bb14d5674814748a60fe2fac7bb78c0c4d360f0',
 'results/experiment-v3/receipt.json':'6a7d246a7e58f6f4ba2c1b5976f30dc1ec0260f581bf81af0b87c400f3080852',
 'results/experiment-v3-audit-supplement/receipt.json':'2e1da959d049eed1b2cbe562747f3ed5f3bc773ef9f986e17d4fbf927ae89508',
 'results/math-regeneration-v1/receipt.json':'f41e8499540ef8354ff4857d712ec251fed9e68b0ebcebb19234b6ccd7b3361f',
 'results/math-regeneration-v1-supplement-v2/receipt.json':'9b17c46bbeee5ba1af7837c512fdf70a827ee16813d413114eba9d6fdc817326',
}
REVIEW='0de001b46d8d1c8a8709a63a4e76590125e626fd4c4fad66483e05945bdedda3'
def require(yes,why):
 if not yes:raise ValueError(why)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment-root',type=pathlib.Path,required=True);p.add_argument('--original',type=pathlib.Path,required=True);p.add_argument('--reference-root',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--write-lock',action='store_true');args=p.parse_args()
 experiment=args.experiment_root.resolve();original=args.original.resolve();reference=args.reference_root.resolve();out=args.output.resolve()
 require(not out.exists() and out.is_relative_to(ROOT/'build'),'fresh candidate-local build output required')
 old=fixture.lock();require('original' not in old,'candidate lock already derived; preserve existing fixture')
 files={};reports={}
 def bind(path,expected=None):
  path=pathlib.Path(path).resolve();data=path.read_bytes();digest=fixture.sha(data);require(expected is None or digest==expected,'authority changed '+str(path));require(str(path) not in files or files[str(path)]==digest,'conflicting authority');files[str(path)]=digest;return data
 for name,digest in EXPECTED.items():reports[name]=json.loads(bind(experiment/name,digest))
 review_path=reference/'results/exp-cutoff-independent-review-v2/review.json';review=json.loads(bind(review_path,REVIEW))
 require(review['classification']=='clear-offline-exp-cutoff-experiment' and not review['findings'] and review['readiness_sha256']==EXPECTED['results/readiness-v1/receipt.json'],'review not clear')
 require(reports['results/freeze-v1/receipt.json']['commit']=='5e856b40cb97f5442c0b8932e5bf9aa54f37d776','wrong experimental source freeze')
 for name,digest in reports['results/readiness-v1/receipt.json']['source_sha256'].items():bind(experiment/name,digest)
 # Recheck exactly the consumed input/output maps of the two actual experiments.
 # Historical source paths refer to preserved sources after audit-only fixes.
 for folder,alias in (('experiment-v3',{'tools/experimental/exp-underflow/probe.py':'probe-preserved.py'}),('math-regeneration-v1',{'tools/experimental/exp-underflow/regenerate_math.py':'regenerate-preserved.py'})):
  report=reports['results/'+folder+'/receipt.json'];directory=experiment/'results'/folder
  for name,digest in report['inputs'].items():
   replacement=next((directory/v for k,v in alias.items() if pathlib.Path(name)==experiment/k),None)
   bind(replacement or pathlib.Path(name),digest)
  for name,digest in report['artifacts'].items():
   path=directory/name;require(path.resolve().is_relative_to(directory),'artifact escaped original output');bind(path,digest)
 raw=bind(original/'receipt.json',old['device_receipt_sha256']);basis=json.loads(raw);require(fixture.canonical(basis)==old['device_report_sha256'],'original report changed')
 ready=reports['results/readiness-v1/receipt.json'];profiles=[];offsets={};payloads={}
 for item in old['profiles']:
  key=item['target']+'-'+item['optimization'];measurement=ready['profiles'][key];source=experiment/'results/math-regeneration-v1'/key/'patched.hsaco';data=bind(source,measurement['patched_sha256'])
  prior=bind(original/(key+'.hsaco'),item['code_sha256']);locations=review['matrix'][key]['patched_only_two_immediates'];require(len(locations)==6 and locations[1:3]==[locations[0]+1,locations[0]+2] and locations[4:]==[locations[3]+1,locations[3]+2],'unexpected reviewed changes');offsets[key]=[locations[0],locations[3]]
  require(data==fixture.corrected_bytes(prior,offsets[key]),'object changed beyond reviewed immediate sites');payloads[key+'.hsaco']=data
  profiles.append(dict(item,code_sha256=fixture.sha(data),bytes=len(data)))
 sources=dict(basis['sources']);sources['tools/export_math_correction.py']=fixture.sha(pathlib.Path(__file__).read_bytes())
 report={'schema':2,'classification':'derived-ocml-natural-exp-correction','hardware_validated':False,'numeric_accuracy_qualified':False,'original_directory':str(original),'original_receipt_sha256':old['device_receipt_sha256'],'original_report_sha256':old['device_report_sha256'],'sources':sources,'profiles':profiles,'cutoff_offsets':offsets,'correction_files':files,'limits':['Original OCML compilation authority plus explicitly reviewed experimental LLVM cutoff correction.','No numerical hardware qualification or production compiler acceptance; frozen oracle unchanged.']}
 out.mkdir(parents=True)
 for name,data in payloads.items():(out/name).write_bytes(data)
 raw=(json.dumps(report,indent=2)+'\n').encode();(out/'receipt.json').write_bytes(raw)
 new=dict(old,original=old,profiles=profiles,device_receipt_sha256=fixture.sha(raw),device_report_sha256=fixture.canonical(report))
 require(all(fixture.sha(pathlib.Path(n).read_bytes())==h for n,h in files.items()),'authority drifted during export')
 require(fixture.lock()==old,'candidate original lock drifted')
 if args.write_lock:(ROOT/'tools/math_native_inputs.json').write_text(json.dumps(new,indent=2)+'\n')
 else:(out/'proposed-lock.json').write_text(json.dumps(new,indent=2)+'\n')
 print(json.dumps({'output':str(out),'receipt_sha256':fixture.sha(raw),'correction_files':len(files),'lock_updated':args.write_lock},indent=2))
if __name__=='__main__':main()
