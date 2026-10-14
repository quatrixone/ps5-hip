"""Package the reviewed native ROCclr GEMM title without touching a console."""
if not __debug__: raise RuntimeError("Validation requires ordinary Python")
from pathlib import Path
import hashlib,json,shutil,subprocess

C=Path(__file__).resolve().parents[1]
R=C.parent/'ps5-hip'
G=['git','--git-dir='+str(C/'.git' if (C/'.git').is_dir() else R/'.git/worktrees'/C.name),'--work-tree='+str(C)]
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
J=lambda p:json.loads(Path(p).read_text())

def main():
    review=C/'results/gemm-independent-review-v1/review.json'
    assert H(review)=='8423e584b5414980dc1384dff2d97c8e2dbb9413f84f0a2f20089333ebe0a5cc'
    q=J(review);assert q['decision']=='CLEAR_FOR_OFFLINE_PACKAGING_ONLY'
    assert not subprocess.check_output(G+['status','--porcelain'],cwd=C).strip()
    subprocess.run(G+['merge-base','--is-ancestor',q['candidate_commit'],'HEAD'],cwd=C,check=True)
    evidence=C/'evidence/offline-clr-gemm-admission-repair.json';e=J(evidence)
    for key,label in [('address_model_39_units','admission_model'),('native_build_46_units','native_build'),
                      ('fself_conversion','conversion'),('native_admission_172_imports','native_admission')]:
        proof=C/e[label]['receipt_path']
        assert H(proof)==e[label]['receipt_sha256']==q['receipts'][key]
        p=J(proof)
        for name,digest in p['inputs'].items():assert H(name)==digest,name
        for name,digest in p.get('artifacts',{}).items():assert H(proof.parent/name)==digest,name
    image=C/'results/native-conversion-v2/eboot.bin'
    assert H(image)==e['conversion']['fself_sha256']==q['artifact_sha256']['fself']
    prior=R/'.local/g5-clr-gpu-startup/dist/PPSA99014/gfx1013-clr-gpu-b21b9c51dc42'
    manifest=J(prior/'manifest.json');package=J(prior/'package-receipt.json')
    assert manifest['title_id']=='PPSA99014' and manifest['kernel']=='clr_gpu'
    assert package['image']['sha256']=='84e52f366b9c7e598d81c3d37197ff8e71d548ebe6cfe933176639066abcc8da'
    assert package['manifest_sha256']==H(prior/'manifest.json')
    files={'eboot.bin':image.read_bytes()}
    for name,record in manifest['files'].items():
        if name=='eboot.bin':continue
        path=prior/'app'/name;data=path.read_bytes()
        assert H(path)==record['sha256'] and len(data)==record['bytes']
        files[name]=data
    assert set(files)=={'eboot.bin','lapy.elf','sce_module/libc.prx','sce_sys/param.json','sce_sys/icon0.png','results-dir.txt'}
    assert J(C/'title.json')['title_id']=='PPSA99014'
    assert json.loads(files['sce_sys/param.json'])['titleId']=='PPSA99014'
    result_dir=files['results-dir.txt'].decode('ascii')
    assert result_dir=='/data/ps5-hip/results'
    stage=C/'dist/PPSA99014'/('gfx1013-clr-gemm-'+H(image)[:12]);stage.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():
        path=stage/'app'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    shutil.copyfile(evidence,stage/'build-receipt.json')
    inputs={str(p):H(p) for p in (Path(__file__),review,evidence,image,prior/'manifest.json',
        prior/'package-receipt.json',R/'evidence/native-clr-gpu-dispatch-success.json',
        C/'tools/package_native_title.py',C/'dependencies.json',C/'title.json')}
    inputs.update({str(prior/'app'/name):H(prior/'app'/name) for name in files if name!='eboot.bin'})
    frozen=dict(schema=1,title_id='PPSA99014',target='gfx1013',kernel='clr_gemm',
        source_commit=q['candidate_commit'],hardware_validated=False,
        build_receipt_sha256=H(evidence),expected_log=result_dir+'/ps5-hip-clr-gemm.log',inputs=inputs,
        files={name:dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)) for name,data in files.items()})
    (stage/'manifest.json').write_text(json.dumps(frozen,indent=2)+'\n')
    subprocess.run(['python3',str(C/'tools/package_native_title.py'),'--stage',str(stage)],cwd=C,check=True)
    assert all(H(name)==digest for name,digest in inputs.items())
    print(stage)

if __name__=='__main__':main()
