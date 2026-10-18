"""Authenticate producers at freeze time and consumed artifacts at runtime."""
import copy, hashlib, json
from pathlib import Path
from admission import child, digest, inventory, load, require, BUDGETS, PROFILES

SCHEMA='ps5-hip-frozen-full-v1'

def runtime_identities(root):
    # Retain current broad imported execution/validator code until a smaller set is justified.
    from source_model_freeze import validator_identities
    values=validator_identities()
    for name in ('tools/frozen_full_admission.py','tools/frozen_full_readbacks.py','tools/admission.py','ports/ggml/generation_cases.def'):
        values[name]=digest(root/name)
    return values

def profile(manifest):
    require(manifest.get('fixture')=='llm_client' and manifest.get('llm_client',{}).get('phase')=='model-full','full model only')
    selected=manifest['model_full']['profile'];require(selected in PROFILES,'unsupported full profile')
    return selected

def contract(manifest):
    return {k:v for k,v in manifest.items() if k not in ('files','artifact_admission')}

def emit(candidate, manifest, root):
    """Issuer authenticates producer itself; no caller-provided bypass flag."""
    from source_model_freeze import validate_model_candidate
    incoming=dict(manifest['files'])
    validate_model_candidate(candidate,manifest)
    selected=profile(manifest);candidate=Path(candidate);root=Path(root)
    require(not (candidate/'artifact-admission.json').exists(),'fresh admission required')
    prefix=candidate.relative_to(root).as_posix()+'/'
    authenticated={name[len(prefix):]:value for name,value in incoming.items() if name.startswith(prefix)}
    current={p.relative_to(candidate).as_posix():digest(p) for p in candidate.rglob('*') if p.is_file()}
    require(current==authenticated,'candidate changed after producer authentication')
    runtime=runtime_identities(root)
    require(all(incoming.get(name)==value for name,value in runtime.items()),'runtime not bound to producer authentication')
    # Compact source receipts are copied for inspection; never interpreted as runtime paths.
    bound=manifest['llm_client'];receipts={'aggregate':bound['full'],'client':bound['native_client'],'model':bound['model'],'vocabulary':bound['vocabulary']}
    folder=candidate/'provenance';folder.mkdir(exist_ok=False)
    for name,binding in receipts.items():
        raw=Path(binding['receipt']).read_bytes();require(hashlib.sha256(raw).hexdigest()==binding['receipt_sha256'],'producer receipt changed')
        (folder/(name+'.json')).write_bytes(raw)
    frozen={p.relative_to(candidate).as_posix():dict(bytes=p.stat().st_size,sha256=digest(p)) for p in candidate.rglob('*') if p.is_file()}
    require(set(frozen)==set(authenticated)|{'provenance/'+name+'.json' for name in receipts},'candidate inventory changed during attestation')
    require('manifest.json' not in frozen and 'review.json' not in frozen,'emit before manifest/review')
    require(all(frozen[name]['sha256']==value for name,value in authenticated.items()),'candidate changed during attestation')
    require(runtime_identities(root)==runtime,'runtime changed during attestation')
    attestation=dict(schema=SCHEMA,profile=selected,budgets=BUDGETS,contract=contract(manifest),files=frozen,runtime=runtime,producer_authenticated=True)
    raw=(json.dumps(attestation,sort_keys=True,indent=2)+'\n').encode()
    temp=candidate/'artifact-admission.pending';temp.write_bytes(raw);temp.rename(candidate/'artifact-admission.json')
    result=copy.deepcopy(manifest);result['artifact_admission']={'schema':SCHEMA,'sha256':hashlib.sha256(raw).hexdigest()}
    result['files']={str((candidate/n).relative_to(root)):item['sha256'] for n,item in frozen.items()}
    result['files'][str((candidate/'artifact-admission.json').relative_to(root))]=result['artifact_admission']['sha256'];result['files'].update(runtime)
    validate(candidate,result,root)
    return result

def validate(candidate, manifest, root):
    candidate,root=Path(candidate),Path(root)
    require(set(manifest.get('artifact_admission',{}))=={'schema','sha256'} and manifest['artifact_admission']['schema']==SCHEMA,'artifact schema')
    selected=profile(manifest)
    for key in ('package','app','launch','close','snapshot','post_snapshot'):
        path=child(root,manifest[key]);require(path.resolve().is_relative_to(candidate.resolve()),'consumed path outside candidate')
    path=child(candidate,'artifact-admission.json')
    require(digest(path)==manifest['artifact_admission']['sha256'],'attestation changed')
    att=load(path.read_bytes())
    require(set(att)=={'schema','profile','budgets','contract','files','runtime','producer_authenticated'} and att['schema']==SCHEMA and att['producer_authenticated'] is True,'attestation fields')
    require(att['profile']==selected and att['budgets']==BUDGETS and att['contract']==contract(manifest),'attestation contract')
    require(att['runtime']==runtime_identities(root),'runtime source changed')
    expected={str((candidate/n).relative_to(root)):v['sha256'] for n,v in att['files'].items()}
    expected[str(path.relative_to(root))]=manifest['artifact_admission']['sha256'];expected.update(att['runtime'])
    require(manifest['files']==expected,'consumed closure differs')
    # Candidate lifecycle output directories are deliberately outside immutable frozen subtrees.
    for name,item in att['files'].items():
        require(set(item)=={'bytes','sha256'} and type(item['bytes']) is int and item['bytes']>=0,'artifact identity schema')
        p=child(candidate,name);require(p.is_file() and p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],'consumed artifact changed')
    for directory in ('package','controllers','snapshot','full-reference','provenance'):
        prefix=directory+'/';inventory(child(candidate,directory),{n[len(prefix):]:v for n,v in att['files'].items() if n.startswith(prefix)})
    receipt=load((candidate/'build-receipt.json').read_bytes());audit=load((candidate/'llm-client-native-audit.json').read_bytes());bound=manifest['llm_client']
    require(receipt.get('status')=='PASS' and receipt.get('classification')=='source-only-model-full-native-client-app' and receipt.get('phase')=='model-full' and receipt.get('fixture')=='llm_client','build receipt semantics')
    require(bound['full']['profile']==selected,'bound full profile')
    require(receipt.get('native_executed') is False and receipt.get('gpu_executed') is False and receipt['llm_client']==bound,'build receipt binding')
    from source_model_freeze import runtime_model
    runtime_key,runtime=runtime_model(manifest)
    original=runtime_key=='external_original_model'
    require(runtime_model(receipt)==(runtime_key,runtime) and receipt['package_sha256']==manifest['model_transfer']['sha256'] and receipt['package_bytes']==manifest['model_transfer']['bytes'] and receipt['package_files']==manifest['package_files'],'package/model contract')
    require(audit.get('status')=='PASS' and audit.get('classification')=='offline-public-hip-llm-client-native-image-audit' and audit.get('phase')=='model-full' and audit.get('full_profile','smoke')==selected,'audit semantics')
    if original:require(audit.get('original_model_call_chain_audited') is True and audit.get('model_format')=='original_safetensors','original model call chain')
    for field in ('wrapper_retained','wrapper_no_bypass','actual_client_call_chain_audited','actual_full_call_chain_audited'):
        require(audit.get(field) is True,'audit call chain')
    if selected!='smoke':require(audit.get('actual_generation_call_chain_audited') is True,'generation call chain')
    for field in ('actual_client_gate_executed','native_executed','gpu_executed'):require(audit.get(field) is False,'audit unearned execution')
    require(digest(candidate/'llm-client-native-audit.json')==bound['audit']['sha256'],'audit binding')
    for name,binding in {'aggregate':bound['full'],'client':bound['native_client'],'model':bound['model'],'vocabulary':bound['vocabulary']}.items():
        require(digest(candidate/'provenance'/(name+'.json'))==binding['receipt_sha256'],'provenance receipt binding')
    app=root/manifest['app'];package=root/manifest['package'];require(app.resolve()==(candidate/'package/app').resolve() and package.resolve()==(candidate/'package/PPSA99014.ffpkg').resolve(),'package location')
    require(digest(package)==receipt['package_sha256'] and package.stat().st_size==receipt['package_bytes'],'package identity')
    require({p.relative_to(app).as_posix():digest(p) for p in app.rglob('*') if p.is_file()}==receipt['package_files'],'staged package')
    require((app/'client-phase.txt').read_text()=='model-full\n' and (app/'results-dir.txt').read_text()==manifest['results_dir'],'package configuration')
    if original:
        from external_original_model import encode
        require((app/'original-model.json').read_bytes()==encode(runtime) and 'model-path.txt' not in receipt['package_files'] and 'model.gguf' not in receipt['package_files'],'original runtime configuration')
    else:require((app/'model-path.txt').read_text().strip()==runtime['path'] and 'original-model.json' not in receipt['package_files'],'GGUF runtime configuration')
    reference=root/manifest['model_full']['reference'];require(reference.resolve()==(candidate/'full-reference').resolve(),'reference location')
    require(digest(reference/'receipt.json')==manifest['model_full']['reference_receipt_sha256'],'reference receipt identity')
    oracle=load((reference/'receipt.json').read_bytes());require(oracle.get('status')=='PASS' and oracle.get('gpu_executed') is False,'oracle semantics')
    require(oracle.get('classification')==('successful-full-model-cpu-boundary-reference' if selected=='smoke' else 'successful-greedy-generation-cpu-reference'),'oracle classification')
    require(oracle.get('requests')==(514 if selected=='smoke' else PROFILES[selected][0]) and oracle.get('raw_files')==(168 if selected=='smoke' else PROFILES[selected][1]),'oracle counts')
    require(oracle['model']['bytes']==bound['model']['bytes'] and oracle['model']['sha256']==bound['model']['sha256'],'oracle model')
    require(oracle.get('profile','short')==('corpus' if selected=='corpus' else 'short') if selected!='smoke' else 'profile' not in oracle,'oracle profile')
    from verify_llm_full_readbacks import corpus_files,generation_files
    from build_llm_full_numeric_test import FILES
    files=corpus_files(reference) if selected=='corpus' else generation_files(reference) if selected=='generation' else FILES
    require(manifest['model_full']['raw_files']==files,'raw contract')
    from source_model_freeze import LAYER_MODULES
    modules=manifest['model_full']['module_sha256'];require(set(modules)==set(LAYER_MODULES),'module inventory')
    for name,h in modules.items():require(receipt['package_files'].get('model-layer/'+name+'.hsaco')==h,'module identity')
    from source_model_freeze import model_contract
    from source_public_hip_log import package_limit
    require(manifest['baseline_transfer']['bytes']<=package_limit(manifest.get('baseline_log_fixture','bootstrap')),'new artifact path does not support historical embedded baseline')
    model_contract(manifest)
    require(manifest['deadline_seconds']==(630 if original else 1830 if selected=='corpus' else 150) and manifest['control_phase_seconds']==(2400 if original else 4200 if selected=='corpus' else 1800),'phase budgets')
    require(manifest.get('raw_capture_seconds')==(600 if selected=='corpus' else None),'capture budget')
    require(att['runtime']==runtime_identities(root),'runtime changed during admission')
    require(digest(path)==manifest['artifact_admission']['sha256'],'attestation changed during admission')
    for name,item in att['files'].items():
        p=child(candidate,name);require(p.stat().st_size==item['bytes'] and digest(p)==item['sha256'],'artifact changed during admission')
    return manifest['llm_client']

class FrozenReference:
    def __init__(self,candidate,manifest,root):
        self.candidate,self.manifest,self.root=Path(candidate),manifest,Path(root);self.check()
    def check(self):
        raw=(self.candidate/'manifest.json').read_bytes();review=load((self.candidate/'review.json').read_bytes())
        require(load(raw)==self.manifest and review.get('approved') is True and review.get('manifest_sha256')==hashlib.sha256(raw).hexdigest(),'frozen reference requires independently reviewed manifest')
        validate(self.candidate,self.manifest,self.root)
    def binding(self):
        self.check();reference=self.root/self.manifest['model_full']['reference'];receipt=load((reference/'receipt.json').read_bytes())
        data={'receipt_sha256':digest(reference/'receipt.json'),'model':receipt['model'],'profile':receipt.get('profile','short')}
        inputs={str(p):digest(p) for p in reference.rglob('*') if p.is_file()}
        return reference,data,inputs
