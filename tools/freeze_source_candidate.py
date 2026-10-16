"""Copy a completed source build into a new, reviewable console candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from source_public_hip_log import LOG_NAMES, package_limit

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(65536),b''):h.update(block)
    return h.hexdigest()


def validate_source_inputs(receipt):
    inputs = receipt.get('input_sha256')
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError('candidate lacks source input identities')
    if 'tools/native_ccache.py' in inputs:
        import native_ccache as cache
        cache.verify(ROOT, receipt.get('compiler_cache'))
    for name, expected in inputs.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError('candidate source input changed: ' + name)


LLM_CLIENT_PACKAGE_FILES = {'eboot.bin','lapy.elf','sce_module/libc.prx','sce_sys/param.json',
                            'sce_sys/icon0.png','results-dir.txt','vocab.gguf','client-phase.txt'}

def validate_client_package(build, receipt):
    package = build/'package/PPSA99014.ffpkg'
    app = build/'package/app'
    if (not 0 < package.stat().st_size <= package_limit('llm_client') or
            digest(package) != receipt.get('package_sha256') or
            set(receipt.get('package_files',{})) != LLM_CLIENT_PACKAGE_FILES or
            {str(p.relative_to(app)) for p in app.rglob('*') if p.is_file()} != LLM_CLIENT_PACKAGE_FILES):
        raise ValueError('vocabulary package inventory/size/hash differs')
    for name, expected in receipt['package_files'].items():
        path=app/name
        if path.is_symlink() or digest(path)!=expected:
            raise ValueError('vocabulary package file differs')
    if (app/'client-phase.txt').read_bytes()!=b'vocab\n':
        raise ValueError('full model phase is not authorized')
    if (app/'results-dir.txt').read_text()!=receipt['results_dir']:
        raise ValueError('vocabulary results path differs')

def validate_client_units(original, receipt):
    inputs=receipt['inputs'];units=receipt.get('units',[]);expected_headers={}
    model_phase=receipt.get('phase')=='model-gate'
    count=149 if model_phase else 147
    if len(units)!=count:raise ValueError('client unit inventory differs')
    if model_phase and (ROOT/units[77]['source']).resolve()!=(ROOT/'.deps/sources/llama-client/ggml/src/ggml-opt.cpp').resolve():
        raise ValueError('missing upstream model cleanup implementation')
    if model_phase and (ROOT/units[78]['source']).resolve()!=(ROOT/'ports/ggml/native_host_allocator.cpp').resolve():
        raise ValueError('missing native model host allocator')
    if model_phase:
        path=ROOT/'tools/external_model.py'
        if inputs.get(str(path))!=digest(path):raise ValueError('external model contract input omitted or changed')
        for name in ('ports/ggml/native_host_allocator.cpp','ports/ggml/native_host_allocator.h','tests/llm_native_host_allocator_test.py','src/native_memory.cpp','include/ps5hip/native_memory.h'):
            path=ROOT/name
            if inputs.get(str(path))!=digest(path):raise ValueError('native host allocator input omitted or changed')
    objects=[(original/u['object']).resolve() for u in units]
    if len(set(objects))!=count:raise ValueError('duplicate client objects')
    for index,(unit,obj) in enumerate(zip(units,objects)):
        if inputs.get(str(obj))!=unit.get('object_sha256') or digest(obj)!=unit.get('object_sha256'):
            raise ValueError('client object identity omitted or changed')
        if index<count-70:
            if not obj.is_relative_to(original.resolve()):raise ValueError('runtime object outside build')
            source=(ROOT/unit['source']).resolve();headers=unit.get('headers',{})
            if (not headers or headers.get(str(source))!=unit.get('source_sha256') or
                    inputs.get(str(source))!=unit.get('source_sha256')):
                raise ValueError('runtime source/header identity omitted')
            for name,h in headers.items():
                if name in expected_headers and expected_headers[name]!=h:
                    raise ValueError('conflicting runtime header identities')
                expected_headers[name]=h
    for name,h in expected_headers.items():
        if not Path(name).is_absolute() or inputs.get(name)!=h or digest(Path(name))!=h:
            raise ValueError('runtime header identity omitted or changed')
    for name in ('tools/build_llm_native_app.py','tools/build_public_hip.py',
                 'tools/build_llm_native_client.py','tools/build_llm_vocab_fixture.py'):
        path=ROOT/name
        if inputs.get(str(path))!=digest(path):raise ValueError('required client producer identity omitted')

def validate_client_image_assets(original, receipt):
    for name,source in (('eboot.bin','eboot.bin'),('sce_module/libc.prx','libc.prx')):
        path=original/source;h=digest(path)
        if receipt['inputs'].get(str(path))!=h or receipt['package_files'].get(name)!=h:
            raise ValueError('packaged native image/provider differs')


def validate_llm_client(build, receipt, frozen=False):
    from build_llm_native_app import vocab_inputs
    from build_llm_native_client import app_objects
    from audit_public_hip_llm_client_native import SOURCES
    if (receipt.get('schema')!=1 or receipt.get('status')!='PASS' or
            receipt.get('classification')!='source-only-vocabulary-native-client-app' or
            receipt.get('fixture')!='llm_client' or receipt.get('phase')!='vocab' or
            receipt.get('native_executed') is not False or receipt.get('gpu_executed') is not False or
            receipt.get('object_count')!=147 or len(receipt.get('units',[]))!=147):
        raise ValueError('not a complete vocabulary-only native app')
    bound=receipt.get('llm_client',{})
    if bound.get('phase')!='vocab':raise ValueError('full model phase is not authorized')
    identities=receipt.get('inputs',{})
    if not identities:raise ValueError('client lacks input identities')
    for name,expected in identities.items():
        if not Path(name).is_absolute() or digest(Path(name))!=expected:
            raise ValueError('client input differs: '+name)
    for name,target in receipt.get('symlinks',{}).items():
        if str(Path(name).resolve())!=target:raise ValueError('client symlink differs')
    vocab=vocab_inputs(Path(bound['vocabulary']['receipt']).parent)
    client=app_objects(Path(bound['native_client']['receipt']).parent)
    if ({k:vocab[k] for k in ('receipt','receipt_sha256','asset')}!=bound['vocabulary'] or
            {k:client[k] for k in ('receipt','receipt_sha256')}!=bound['native_client']):
        raise ValueError('client prerequisite differs')
    for closure in (vocab['inputs'],client['inputs']):
        if any(identities.get(name)!=h for name,h in closure.items()):
            raise ValueError('client prerequisite closure omitted')
    if [u['object'] for u in receipt['units'][-70:]]!=client['objects']:
        raise ValueError('client object list differs')
    original=Path(bound['audit']['path']).parent
    validate_client_units(original,receipt)
    validate_client_image_assets(original,receipt)
    audit_path=build/'llm-client-native-audit.json'
    if digest(audit_path)!=bound['audit']['sha256']:raise ValueError('client audit identity differs')
    audit=json.loads(audit_path.read_text())
    if (audit.get('status')!='PASS' or audit.get('classification')!='offline-public-hip-llm-client-native-image-audit' or
            audit.get('phase')!='vocab' or audit.get('wrapper_retained') is not True or
            audit.get('wrapper_no_bypass') is not True or audit.get('actual_client_gate_executed') is not False or
            audit.get('native_executed') is not False or audit.get('gpu_executed') is not False):
        raise ValueError('client audit not accepted')
    audit_inputs=audit.get('inputs',{})
    original=Path(bound['audit']['path']).parent
    for path in [original/n for n in ('title.elf','eboot.elf','extracted.elf')]+[ROOT/n for n in SOURCES]:
        if audit_inputs.get(str(path))!=digest(path) or identities.get(str(path))!=digest(path):
            raise ValueError('client audit required identity differs')
    for name,h in audit_inputs.items():
        if digest(Path(name))!=h:raise ValueError('client audit input differs')
    validate_client_package(build,receipt)
    if receipt['package_files']['vocab.gguf']!=digest(Path(vocab['asset'])):
        raise ValueError('packaged vocabulary differs from authenticated asset')
    return bound


def validate_embedded_audit(build, receipt):
    path = build / 'embedded-native-audit.json'
    if (receipt.get('compiler_object_count') != 1 or
            receipt.get('total_linked_object_count') != 77 or
            digest(path) != receipt.get('embedded_audit_sha256')):
        raise ValueError('embedded candidate lacks the exact compiler/layout audit')
    audit = json.loads(path.read_text())
    if (audit.get('classification') != 'offline-public-hip-embedded-native-layout-audit' or
            audit.get('fixture', 'embedded') != receipt.get('fixture', 'embedded') or
            audit.get('native_executed') is not False or not audit.get('inputs') or
            any(digest(Path(name)) != expected for name, expected in audit['inputs'].items())):
        raise ValueError('embedded audit inputs changed or classification differs')
    roundtrip = receipt.get('fself_roundtrip', {})
    required = {str((build / 'title.elf').resolve()): receipt.get('title_elf_sha256'),
                str((build / 'eboot.elf').resolve()): roundtrip.get('elf_sha256'),
                str((build / 'extracted.elf').resolve()): roundtrip.get('extracted_sha256')}
    compiler = receipt.get('embedded_compiler', {})
    required[compiler.get('object', '')] = compiler.get('object_sha256')
    if any(expected is None or audit['inputs'].get(name) != expected
           for name, expected in required.items()):
        raise ValueError('embedded audit does not bind this build and compiler object')
    return path


def validate_rmsnorm_audit(build, receipt):
    path = build/'rmsnorm-native-audit.json'
    audit = json.loads(path.read_text())
    if (audit.get('status') != 'PASS' or
            audit.get('classification') != 'offline-public-hip-rmsnorm-native-image-audit' or
            audit.get('native_executed') is not False or not isinstance(audit.get('inputs'), dict) or
            not audit['inputs']):
        raise ValueError('RMSNorm native image audit missing or unqualified')
    for name, expected in audit['inputs'].items():
        if digest(Path(name)) != expected:
            raise ValueError('RMSNorm image audit input changed: '+name)
    roundtrip = receipt.get('fself_roundtrip', {})
    required = {str((build/'title.elf').resolve()): receipt.get('title_elf_sha256'),
                str((build/'eboot.elf').resolve()): roundtrip.get('elf_sha256'),
                str((build/'extracted.elf').resolve()): roundtrip.get('extracted_sha256')}
    for name in ('tools/audit_public_hip_rmsnorm_native.py', 'tools/audit_public_hip_embedded_native.py',
                 'tools/probe_host_registration.py', 'tools/native_multi_query.py',
                 'tests/fat_binary_fixture_test.py', 'tests/native_import_libraries_test.py',
                 'tools/build_native_vector_title.py', 'tools/public_hip_rmsnorm_console_title.cpp',
                 'tools/public_hip_app_crt.cpp', 'tests/public_hip_rmsnorm_native_audit_test.py'):
        required[str((ROOT/name).resolve())] = digest(ROOT/name)
    if any(expected is None or audit['inputs'].get(name) != expected for name, expected in required.items()):
        raise ValueError('RMSNorm image audit does not bind candidate images and implementation')
    return path


def freeze_rmsnorm_reference(build, receipt, output):
    from build_public_hip import rmsnorm_assets, rmsnorm_reference_inputs
    bound = receipt.get('rmsnorm', {})
    reference = bound.get('reference', {})
    original = Path(reference.get('receipt', '')).resolve()
    module = Path(bound.get('assets', {}).get('rmsnorm/norm.hsaco', '')).resolve()
    if (not original.is_file() or not module.is_file() or
            rmsnorm_assets(module.parent, original.parent) != bound):
        raise ValueError('RMSNorm build inputs no longer match')
    for name, source in bound['assets'].items():
        expected = digest(Path(source))
        if receipt['package_files'].get(name) != expected or digest(build/'package/app'/name) != expected:
            raise ValueError('RMSNorm packaged oracle or module differs')
    report = json.loads(original.read_text())
    destination = output/'rmsnorm-reference'
    destination.mkdir()
    shutil.copyfile(original, destination/'receipt.json')
    total = 0
    for name, expected in report['artifacts'].items():
        source = (original.parent/name).resolve()
        if Path(name).is_absolute() or not source.is_relative_to(original.parent):
            raise ValueError('invalid reference artifact path')
        total += source.stat().st_size
        if total > 16*1024*1024 or digest(source) != expected:
            raise ValueError('reference artifact bound or identity changed')
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    copied = rmsnorm_reference_inputs(destination)
    if copied['receipt_sha256'] != reference['receipt_sha256']:
        raise ValueError('frozen reference receipt differs')
    return {'reference': str(destination.relative_to(ROOT)),
            'reference_receipt_sha256': copied['receipt_sha256']}


Q4_REFERENCE_LIMIT = 16*1024*1024
Q4_VERIFY_SOURCES = ('tools/verify_public_hip_q4.py','tools/public_hip_q4_assets.py',
                     'tools/build_public_hip.py','tools/qualify_public_hip_q4.py',
                     'tools/build_llm_q4_reference.py','tools/llm_q4_reference.py',
                     'tests/public_hip_q4_assets_test.py','tests/q4_dump_test.py',
                     'tests/source_q4_freeze_test.py')


def validate_q4_audit(build, receipt):
    from audit_public_hip_q4_native import SOURCES
    path=build/'q4-native-audit.json'
    audit=json.loads(path.read_text())
    if (audit.get('status')!='PASS' or
            audit.get('classification')!='offline-public-hip-q4-native-image-audit' or
            audit.get('native_executed') is not False or not isinstance(audit.get('inputs'),dict) or
            not audit['inputs']):
        raise ValueError('Q4 native image audit missing or unqualified')
    for name,expected in audit['inputs'].items():
        if not Path(name).is_absolute() or digest(Path(name))!=expected:
            raise ValueError('Q4 audit input changed: '+name)
    roundtrip=receipt.get('fself_roundtrip',{})
    required={str((build/'title.elf').resolve()):receipt.get('title_elf_sha256'),
              str((build/'eboot.elf').resolve()):roundtrip.get('elf_sha256'),
              str((build/'extracted.elf').resolve()):roundtrip.get('extracted_sha256')}
    required.update({str((ROOT/name).resolve()):digest(ROOT/name) for name in SOURCES})
    if any(expected is None or audit['inputs'].get(name)!=expected for name,expected in required.items()):
        raise ValueError('Q4 audit does not bind candidate images and implementation')
    return path


def freeze_q4_reference(build, receipt, output):
    from public_hip_q4_assets import assets
    from verify_public_hip_q4 import ASSET_NAMES
    bound=receipt.get('q4',{})
    original=Path(bound.get('reference',{}).get('receipt','')).resolve()
    q8=Path(bound.get('assets',{}).get('q4/q8.hsaco','')).resolve()
    q4=Path(bound.get('assets',{}).get('q4/q4.hsaco','')).resolve()
    if (not original.is_file() or q8.name!='q8.hsaco' or q4.name!='q4.hsaco' or
            q8.parent!=q4.parent or not q8.parent.is_relative_to(ROOT) or
            assets(q8.parent,original.parent)!=bound):
        raise ValueError('Q4 build inputs no longer match')
    expected_names=ASSET_NAMES|{'q4/q8.hsaco','q4/q4.hsaco'}
    if set(bound.get('assets',{}))!=expected_names or len(expected_names)!=58:
        raise ValueError('Q4 exact58 package assets required')
    expected={name:digest(Path(source)) for name,source in bound['assets'].items()}
    if {name:value for name,value in receipt['package_files'].items() if name.startswith('q4/')}!=expected:
        raise ValueError('Q4 package asset inventory differs')
    if any(digest(build/'package/app'/name)!=value for name,value in expected.items()):
        raise ValueError('Q4 packaged asset differs')
    report=json.loads(original.read_text())
    planned=[];total=original.stat().st_size
    # Validate the entire copy set and bound before creating any destination files.
    for name,row in report['artifacts'].items():
        source=original.parent/name
        if (Path(name).is_absolute() or not source.resolve().is_relative_to(original.parent) or
                any(part.is_symlink() for part in (source,*source.parents)) or not source.is_file()):
            raise ValueError('unsafe Q4 reference artifact path')
        size=source.stat().st_size;total+=size
        if size!=row['bytes'] or digest(source)!=row['sha256']:
            raise ValueError('Q4 reference artifact identity changed')
        planned.append((name,source,row['sha256']))
    if total>Q4_REFERENCE_LIMIT:
        raise ValueError('Q4 reference exceeds16MiB frozen bound')
    destination=output/'q4-reference'
    destination.mkdir()
    shutil.copyfile(original,destination/'receipt.json')
    for name,source,expected_sha in planned:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        if digest(target)!=expected_sha:raise ValueError('Q4 reference changed during copy')
    copied=assets(q8.parent,destination)
    if copied['reference']['receipt_sha256']!=bound['reference']['receipt_sha256']:
        raise ValueError('frozen Q4 reference receipt differs')
    if assets(q8.parent,original.parent)!=bound:
        raise ValueError('Q4 inputs changed during freeze')
    return {'fixture':str(q8.parent.relative_to(ROOT)),
            'reference':str(destination.relative_to(ROOT)),
            'reference_receipt_sha256':copied['reference']['receipt_sha256'],
            'module_sha256':{name:expected['q4/'+name+'.hsaco'] for name in ('q8','q4')}}


WEIGHTED_QKV_REFERENCE_LIMIT=16*1024*1024
WEIGHTED_QKV_VERIFY_SOURCES=('tools/verify_weighted_qkv_readbacks.py','tools/public_hip_weighted_qkv_assets.py',
    'tools/build_llm_weighted_qkv_reference.py','tools/llm_q4_reference.py','tools/build_public_hip.py',
    'tests/weighted_qkv_readbacks_test.py','tests/source_weighted_qkv_controls_test.py')

def validate_weighted_qkv_audit(build,receipt):
    from build_public_hip import validate_weighted_qkv_audit as validate
    if not receipt.get("weighted_qkv_audit_sha256"):raise ValueError("missing weighted audit identity")
    validate(build,receipt["weighted_qkv_audit_sha256"])
    roundtrip=receipt.get("fself_roundtrip",{})
    for name,h in (("title.elf",receipt.get("title_elf_sha256")),("eboot.elf",roundtrip.get("elf_sha256")),("extracted.elf",roundtrip.get("extracted_sha256"))):
        if h is None or digest(build/name)!=h:raise ValueError("weighted audited image differs from receipt")
    return build/'weighted-qkv-native-audit.json'

def freeze_weighted_qkv_reference(build,receipt,output):
    from public_hip_weighted_qkv_assets import assets
    bound=receipt.get('weighted_qkv',{})
    original=Path(bound.get('reference',{}).get('receipt','')).resolve()
    module_paths={n:Path(bound.get('assets',{}).get('weighted-qkv/'+n+'.hsaco','')).resolve() for n in ('weighted','q8','q4')}
    weighted=module_paths['weighted'].parent;q4=module_paths['q4'].parent
    if (not original.is_file() or any(not p.is_file() or not p.is_relative_to(ROOT) or p.name!=n+'.hsaco' for n,p in module_paths.items()) or module_paths['q8'].parent!=q4 or assets(weighted,q4,original.parent)!=bound):
        raise ValueError('weighted QKV build inputs changed')
    expected={name:digest(Path(source)) for name,source in bound['assets'].items()}
    if len(expected)!=10 or {n:h for n,h in receipt['package_files'].items() if n.startswith('weighted-qkv/')}!=expected:
        raise ValueError('weighted QKV exact10 packaged assets required')
    if any(digest(build/'package/app'/n)!=h for n,h in expected.items()):raise ValueError('weighted QKV packaged asset differs')
    report=json.loads(original.read_text());planned=[];total=original.stat().st_size
    for name,row in report['artifacts'].items():
        source=original.parent/name
        if Path(name).is_absolute() or not source.resolve().is_relative_to(original.parent) or any(p.is_symlink() for p in (source,*source.parents)) or not source.is_file():raise ValueError('unsafe weighted reference path')
        size=source.stat().st_size;total+=size
        if size!=row['bytes'] or digest(source)!=row['sha256']:raise ValueError('weighted reference changed')
        planned.append((name,source,row['sha256']))
    if total>WEIGHTED_QKV_REFERENCE_LIMIT:raise ValueError('weighted reference exceeds16MiB bound')
    destination=output/'weighted-qkv-reference';destination.mkdir()
    shutil.copyfile(original,destination/'receipt.json')
    for name,source,h in planned:
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        if digest(target)!=h:raise ValueError('weighted reference copy changed')
    copied=assets(weighted,q4,destination)
    if copied['reference']['receipt_sha256']!=bound['reference']['receipt_sha256'] or assets(weighted,q4,original.parent)!=bound:raise ValueError('weighted QKV closure changed during freeze')
    return {'weighted_fixture':str(weighted.relative_to(ROOT)),'q4_fixture':str(q4.relative_to(ROOT)),
            'reference':str(destination.relative_to(ROOT)),'reference_receipt_sha256':copied['reference']['receipt_sha256'],
            'module_sha256':{n:expected['weighted-qkv/'+n+'.hsaco'] for n in module_paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshot-build', type=Path, default=ROOT / 'build/source-make-snapshot',
                        help='Offline teardown helper from make console-controls')
    parser.add_argument('--baseline-image-sha256', required=True)
    parser.add_argument('--baseline-log-sha256')
    parser.add_argument('--baseline-results-dir', default='/data/ps5-hip/results')
    parser.add_argument('--baseline-log-preserved', action='store_true',
                        help='Verify an existing SHA-suffixed preserved baseline log; never rename it again')
    parser.add_argument('--baseline-log-fixture', choices=tuple(LOG_NAMES), default='bootstrap')
    parser.add_argument('--client-phase', choices=('vocab','model-gate'), default='vocab')
    parser.add_argument('--baseline-image-bytes', type=int)
    parser.add_argument('--baseline-model-receipt', type=Path)
    parser.add_argument('--transfer-seconds', type=int, default=1800)
    args = parser.parse_args()
    from build_public_hip import validate_results_directory
    try:
        validate_results_directory(args.baseline_results_dir)
    except ValueError as error:
        parser.error(str(error))
    if args.baseline_log_preserved and not args.baseline_log_sha256:
        parser.error('--baseline-log-preserved requires --baseline-log-sha256')
    import re
    for value in (args.baseline_image_sha256, args.baseline_log_sha256):
        if value is not None and not re.fullmatch('[0-9a-f]{64}', value):
            parser.error('baseline digests must be lowercase SHA-256')
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'results'):
        parser.error('candidate must be under results/')
    receipt_path = args.build / 'link-receipt.json'
    if not receipt_path.exists():receipt_path=args.build/'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    fixture = receipt.get('fixture', 'bootstrap')
    if fixture not in LOG_NAMES:
        parser.error('unknown fixture')
    if not isinstance(receipt.get('results_dir'), str):
        parser.error('invalid results directory type')
    try:
        validate_results_directory(receipt['results_dir'])
    except ValueError as error:
        parser.error(str(error))
    if receipt['object_count'] != ((149 if args.client_phase=='model-gate' else 147) if fixture=='llm_client' else 76):
        parser.error('not the complete .40 candidate')
    model_phase=fixture=='llm_client' and args.client_phase=='model-gate'
    if args.client_phase=='model-gate' and fixture!='llm_client':parser.error('model phase requires llm_client')
    if fixture=='llm_client' and receipt.get('phase')!=args.client_phase:parser.error('explicit client phase differs from receipt')
    validators={}
    if model_phase:
        from source_model_freeze import validate_model_receipt, validator_identities, model_contract, validate_model_candidate
        validators=validator_identities()
        llm_client=validate_model_receipt(args.build,receipt)
        if args.baseline_image_bytes is None or not 0<args.transfer_seconds<=1800:parser.error('model phase requires baseline size and bounded transfer seconds')
    else:llm_client = validate_llm_client(args.build,receipt) if fixture=='llm_client' else None
    if fixture!='llm_client':validate_source_inputs(receipt)
    audit = (args.build/'llm-client-native-audit.json' if fixture=='llm_client' else
             validate_weighted_qkv_audit(args.build, receipt) if fixture == 'weighted_qkv' else
             validate_q4_audit(args.build, receipt) if fixture == 'q4' else
             validate_rmsnorm_audit(args.build, receipt) if fixture == 'rmsnorm' else
             validate_embedded_audit(args.build, receipt) if fixture in ('embedded', 'aggregate') else None)
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT):
        parser.error('commit source before freezing')
    model_transfer=baseline_transfer=None
    if model_phase:
        model_transfer={'bytes':receipt['package_bytes'],'sha256':receipt['package_sha256'],'deadline_seconds':args.transfer_seconds}
        baseline_transfer={'bytes':args.baseline_image_bytes,'sha256':args.baseline_image_sha256,'deadline_seconds':args.transfer_seconds}
        if args.baseline_model_receipt:
            baseline_transfer.update(receipt=str(args.baseline_model_receipt.resolve()),receipt_sha256=digest(args.baseline_model_receipt))
        model_contract({'fixture':fixture,'llm_client':llm_client,'external_model':receipt['external_model'],'model_transfer':model_transfer,'baseline_transfer':baseline_transfer,'baseline_image_sha256':args.baseline_image_sha256,'baseline_log_fixture':args.baseline_log_fixture})
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.build / 'package', output / 'package')
    shutil.copyfile(receipt_path, output / 'build-receipt.json')
    if audit is not None:
        shutil.copyfile(audit, output / audit.name)
    shutil.copytree(ROOT / 'build/controllers/PPSA99014', output / 'controllers')
    shutil.copytree(args.snapshot_build, output / 'snapshot')
    package = output / 'package/PPSA99014.ffpkg'
    if package.stat().st_size > package_limit(fixture) or digest(package) != receipt['package_sha256']:
        parser.error('package hash mismatch')
    for name, expected in receipt['package_files'].items():
        if digest(output / 'package/app' / name) != expected:
            parser.error('package stage hash mismatch: ' + name)
    rmsnorm = freeze_rmsnorm_reference(args.build, receipt, output) if fixture == 'rmsnorm' else None
    q4 = freeze_q4_reference(args.build, receipt, output) if fixture == 'q4' else None
    weighted_qkv = freeze_weighted_qkv_reference(args.build, receipt, output) if fixture == 'weighted_qkv' else None
    files = {str(path.relative_to(ROOT)): digest(path) for path in output.rglob('*') if path.is_file()}
    for name in ('tools/source_console.py', 'tools/source_public_hip_log.py', 'tools/freeze_source_candidate.py',
                 'dependencies.json', 'elevation-inputs.json', 'controller-inputs.json', 'title.json'):
        files[name] = digest(ROOT / name)
    if rmsnorm is not None:
        for name, expected in json.loads(audit.read_text())['inputs'].items():
            path = Path(name).resolve()
            if path.is_relative_to(ROOT):
                files[str(path.relative_to(ROOT))] = expected
        for name in ('tools/verify_public_hip_rmsnorm.py', 'tools/build_public_hip.py',
                     'tools/qualify_public_hip_rmsnorm.py', 'tools/build_llm_rmsnorm_reference.py',
                     'tests/public_hip_rmsnorm_readback_test.py'):
            files[name] = digest(ROOT/name)
        for name, expected in receipt['rmsnorm']['reference']['inputs'].items():
            path = Path(name).resolve()
            if path.is_relative_to(ROOT):
                files[str(path.relative_to(ROOT))] = expected
    if q4 is not None:
        for name in Q4_VERIFY_SOURCES:
            files[name] = digest(ROOT/name)
        for source_inputs in (json.loads(audit.read_text())['inputs'], receipt['q4']['inputs']):
            for name, expected in source_inputs.items():
                path = Path(name).resolve()
                if path.is_relative_to(ROOT):
                    files[str(path.relative_to(ROOT))] = expected
    if weighted_qkv is not None:
        for name in WEIGHTED_QKV_VERIFY_SOURCES:files[name]=digest(ROOT/name)
        for source_inputs in (json.loads(audit.read_text())['inputs'],receipt['weighted_qkv']['inputs']):
            for name,expected in source_inputs.items():
                path=Path(name).resolve()
                if path.is_relative_to(ROOT):files[str(path.relative_to(ROOT))]=expected
    if llm_client is not None:
        for name,h in receipt['inputs'].items():
            path=Path(name).resolve()
            if path.is_relative_to(ROOT):files[str(path.relative_to(ROOT))]=h
    if model_phase:
        if validator_identities()!=validators:parser.error('model validators changed during freeze')
        for name,h in validators.items():
            if name in files and files[name]!=h:parser.error('validator identity conflict')
            files[name]=h
    relative = str(output.relative_to(ROOT))
    manifest = {'host': '192.168.4.40', 'title_id': 'PPSA99014', 'results_dir': receipt['results_dir'],
                'baseline_results_dir': args.baseline_results_dir,
                'baseline_image_sha256': args.baseline_image_sha256,
                'baseline_log_sha256': args.baseline_log_sha256,
                'baseline_log_preserved': args.baseline_log_preserved,
                'baseline_log_fixture': args.baseline_log_fixture, 'fixture': fixture,
                'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'package': relative + '/package/PPSA99014.ffpkg', 'app': relative + '/package/app',
                'launch': relative + '/controllers/launch.elf', 'close': relative + '/controllers/close.elf',
                'snapshot': relative + '/snapshot/teardown-snapshot.elf',
                'post_snapshot': relative + '/snapshot/teardown-snapshot.elf',
                'package_files': receipt['package_files'], 'files': files,
                'hypothesis': ('Vocabulary-only upstream loader/tokenizer matches32cases236tokens; real publicHIP storage copies/clears16floats; unsupportedADD rejects beforeCPUcompute; no complete model/inference/GPUdispatch. Require1allocation/free and1retiredqueue with clean application lifecycle.' if fixture=='llm_client' else
                               'Weighted RMSNorm produces2048values consumed in-place by Q8_1 and full Q/K/V projections2048/256/256. Require30full raw files11972800bytes, frozen weighted budget then exact Q8 and unchanged Q4 bounds from actual weighted readback, unchanged shared intermediates/inputs/guards/tails,5launches6allocations/frees3moduleunloads1retiredqueue. Transport transcript alone never establishes numerical success.' if fixture == 'weighted_qkv' else
                               'Selected Q8_1 and nonfused Q4_0 kernels, including any diagnostic load-width/cache/scheduling changes declared in the bound fixture receipt, compute12cases over28stages/dispatches; compare592Q4values and80640Q8bytes,112fullrawfiles/9097984bytes, unchangedinputs/weights/guards/inactivebytes,4allocations/frees,2moduleunloads and1retiredqueue. Frozen and chained stages require independent authenticated raw verification; transcript alone is insufficient.' if fixture == 'q4' else
                               'Unchanged upstream RMSNorm computes six CPU-oracle cases with81920 finite values,192 outputguardbytes,507904 inactivebytes and835776 unchangedinputbytes. Require independently verified full raw readbacks, exact two allocation/frees and one retiredqueue; transcript alone is insufficient.' if fixture == 'rmsnorm' else
                               'Genuine compiler stubs pass 12/16/128-byte aggregate arguments to two changed-input GPU rounds, preserving320memberbytes64guardbytes, one allocation/free and one retired queue. Internal struct padding is unspecified; no nested-pointer lifetime claim. Audited compiler unregister precedes CLR teardown; final device backing checks do not prove full static-registry retirement.' if fixture == 'aggregate' else
                               'Genuine compiler registration and launch stubs execute two changed-input GPU rounds, preserving146values16guards, one allocation/free and one retired queue. Audited compiler unregister precedes CLR teardown; final device backing checks do not prove full static-registry retirement.' if fixture == 'embedded' else
                               'After the existing joined-copy recovery oracle, a two-kernel joined graph alternates eager/captured order over four warmup and sixteen further changed-input rounds. Validate 5840 added values and640guards,84totaldispatches,28callbacks,fourallocations/frees and seven retired queues. Require32post-sync backing samples at32768bytes/twoallocations,128balanced allocations/releases per path,98304accountedpeak and zero final backing. This measures NativeMemory backing only, not process heap, command lifetime or timing.' if fixture == 'copy_gpu' else
                               'Public HIP module kernel capture records a kernel and dependent host callback without dispatch; disabled and changed-argument replay, executable update and source graph destruction preserve146values16guards and clean3queue retirement.' if fixture == 'capture_gpu' else
                               'Explicit public HIP GPU graph owns arguments, updates the executable, survives source destruction, preserves146values16guards across2launches and2disabledreplays with4callbacks, then frees1buffer and retires3queues.' if fixture == 'graph_gpu' else
                               'Public HIP producer GPU launch records an event; a nonblocking consumer waits before D2D copy, preserving146outputs16guards and source over2changedrounds with event reuse and3retiredqueues.' if fixture == 'stream' else
                               'Public HIP explicit and captured host graphs replay32changed inputs, execute64callbacks, update callback/user data, survive source-graph destruction and recover6invalidations while retaining3caller-owned graphs, then retire5queues without GPU work.' if fixture == 'graph' else
                               'New source-built upstream hipInit and device discovery complete with count=1 and clean natural teardown.' if fixture == 'bootstrap' else
                               'Public hipMalloc allocates two distinct buffers, hipFree rejects interior/double frees and retires both, followed by clean natural teardown.' if fixture == 'memory' else
                               'Public HIP contiguous and pitched 2D/3D fills validate4096bytes across two changed-pattern rounds, reject16invalid requests, preserve guards and row/slice padding and finalize without GPU submission.' if fixture == 'fill' else
                               'Public HIP loads the owned AOT kernel and executes two changed-input dispatches, validates146values and16guards, rejects unsupported geometry, and finalizes with exact resource cleanup.' if fixture == 'launch' else
                               'Two independent public HIP modules load their owned AOT file, expose distinct functions, retain the surviving module after the other unloads, and finalize without queues or GPU work.' if fixture == 'module' else
                               'Public HIP contiguous and rectangular copies validate 7168 bytes and reject 16 invalid requests, then shared queued-free and parameter-migration cases validate 384 lifetime bytes with executable/source graphs still alive, eight allocations/frees and four retired queues without GPU submission.'),
                'expected': 'Exact ' + fixture + ' transcript, one ordered title start/stop, equal two-row mount snapshots, healthy services.',
                'reject': 'Any identity, log, mount, lifecycle, timeout, exception, or health failure; never retry automatically.',
                'deadline_seconds': 40, 'public_hip_gpu_execution_claimed': False}
    if rmsnorm is not None:
        manifest['rmsnorm'] = rmsnorm
    if llm_client is not None:
        manifest['llm_client']=llm_client
    if q4 is not None:
        manifest['q4'] = q4
    if weighted_qkv is not None:
        manifest['weighted_qkv']=weighted_qkv
    if model_phase:
        manifest.update(external_model=receipt['external_model'],model_transfer=model_transfer,baseline_transfer=baseline_transfer,deadline_seconds=150,control_phase_seconds=1800,
            hypothesis='Source-authenticated full model loads without mmap/lazy loading; two same-position actual llama_process calls are rejected by mandatory wrapped scheduler before CPU/GPU computation, with measured graph reuse and zero final allocation/queue counters. This is rejection readiness, not inference.')
        validate_model_candidate(output,manifest)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(output / 'manifest.json')


if __name__ == '__main__':
    main()
