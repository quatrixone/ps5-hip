"""Add genuine HIP profiling TLS writers to the preserved SVM prerequisite."""
from pathlib import Path
import argparse, difflib, hashlib, json, shutil
p=argparse.ArgumentParser();p.add_argument('--reference-root',type=Path,required=True);a=p.parse_args()
R=Path(__file__).resolve().parents[1];B=a.reference_root.resolve();old=R/'results/clr-source-v7';O=R/'results/clr-source-v8';O.mkdir(parents=True,exist_ok=False)
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
shutil.copytree(old/'source',O/'source');shutil.copyfile(old/'ps5-aot.patch',O/'ps5-aot.patch');metadata=json.loads((old/'preparation.json').read_text())
for name,digest in metadata['changed_sources'].items():assert H(O/name)==digest,name
up=B/'results/g4-runtime-architecture-v1/sources/clr-65d174c3e35423bf25c9c369513780c9d9ca760c'
for name in ['hip_internal.hpp','hip_prof_api.h']:shutil.copyfile(up/'hipamd/src'/name,O/'source/hipamd/src'/name)
relative='hipamd/src/hip_prof_api.h';q=O/'source'/relative;before=q.read_text();assert before.count('amd::activity_prof::correlation_id =')==2
after=before.replace('amd::activity_prof::correlation_id =','amd::activity_prof::correlationId() =');q.write_text(after)
patch=R/'ports/clr/ps5-hip-correlation.patch';patch.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative)))
metadata['changed_sources']['source/'+relative]=H(q);metadata['changed_sources']['source/hipamd/src/hip_internal.hpp']=H(O/'source/hipamd/src/hip_internal.hpp')
metadata['correlation_preparer_sha256']=H(__file__);metadata['correlation_patch_sha256']=H(patch);metadata['previous_preparation_sha256']=H(old/'preparation.json')
(O/'preparation.json').write_text(json.dumps(metadata,indent=2)+'\n')
m=R/'ports/clr/adapter-manifest.json';manifest=json.loads(m.read_text());manifest['patch_order'].append(patch.name);manifest['patches'][patch.name]=H(patch)
manifest['files'][relative]={'base_sha256':H(up/relative),'adapted_sha256':H(q)};m.write_text(json.dumps(manifest,indent=2)+'\n')
print(H(O/'preparation.json'))
