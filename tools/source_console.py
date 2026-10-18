"""One-use .40 source-build deployment and startup control; review-bound, never automatic."""
import argparse
import ftplib
import hashlib
import io
import json
from pathlib import Path
import re
import socket
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from source_public_hip_log import LOG_NAMES, package_limit
from source_model_streaming import local_identity, readback, stage_upload
from source_model_freeze import model_contract, stream_digest, validate_model_candidate, MODEL_PACKAGE_LIMIT
from source_llm_model_log import check_model_log

ROOT = Path(__file__).resolve().parents[1]
HOST = '192.168.4.40'
TITLE = 'PPSA99014'
REMOTE = '/data/homebrew/PPSA99014.ffpkg'
MOUNT = '/mnt/shadowmnt/PPSA99014_b6d9b2e4'
LOG = '/data/ps5-hip/results/ps5-hip-public-bootstrap.log'
HISTORY = '/data/shadowmount/debug.log'
LOCK = Path('/mnt/c/Users/Denis/Documents/PS5/lock.txt')

RMSNORM_RAW_BYTES = 139296
RMSNORM_RAW_NAMES = tuple('rmsnorm-%s-%d.%s.raw' % (family, rows, kind)
    for family in ('stress', 'embedding') for rows in (1, 2, 17)
    for kind in ('output', 'input-readback'))

def native_fault(raw):
    lower = bytes(raw).lower()
    return any(marker in lower for marker in (b'# exception:', b'receives a fatal signal', b'kernel panic', b'panic:'))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def snapshot(raw):
    text = raw.decode('ascii', 'strict')
    require(text.count('PS5HIP TEARDOWN BEGIN') == 1 and text.count('PS5HIP TEARDOWN END') == 1,
            'incomplete snapshot')
    require('snapshots_stable=1 status_stable=1' in text, 'unstable snapshot')
    statuses = re.findall(r'PS5HIP TEARDOWN STATUS phase=(before|after).* running=0x([0-9a-f]{8}) title_app=0x([0-9a-f]{8})', text)
    require(len(statuses) == 2 and {x[0] for x in statuses} == {'before', 'after'} and
            all(x[1:] == ('ffffffff', 'ffffffff') for x in statuses), 'application queries not idle')
    rows = re.findall(r'PS5HIP TEARDOWN MOUNT snapshot=([12]) (.*)', text)
    first = sorted(row for index, row in rows if index == '1')
    second = sorted(row for index, row in rows if index == '2')
    require(first == second and len(first) == 2, 'expected stable two-row title mounts')
    decoded = []
    for row in first:
        fields = dict(item.split('=', 1) for item in row.split())
        decoded.append((fields['kind'], bytes.fromhex(fields['fstype_hex']).decode(),
                        bytes.fromhex(fields['to_hex']).decode(), bytes.fromhex(fields['from_hex']).decode()))
    require(any(kind == 'registered_image' and fs == 'ufs' and to == MOUNT and
                re.fullmatch('/dev/lvd[0-9]+', source) for kind, fs, to, source in decoded), 'missing UFS anchor')
    require(('runtime_layer', 'nullfs', '/system_ex/app/' + TITLE, MOUNT) in decoded, 'missing title anchor')
    return first


def lifecycle(raw):
    text = raw.decode('utf-8', 'replace')
    starts = re.findall(r'\[GAME\] started: ([A-Z0-9]+) pid=(\d+)\b', text)
    events = re.findall(r'\[(GAME|KSTUFF)\] (started|game stopped): ([A-Z0-9]+) pid=(\d+)\b', text)
    complete = (len(starts) == 1 and starts[0][0] == TITLE and
                events == [('GAME', 'started', *starts[0]), ('KSTUFF', 'game stopped', *starts[0])])
    return complete, starts


class PhaseFTP:
    def __init__(self, ftp, remaining): self.ftp,self.remaining=ftp,remaining
    def close(self):
        return self.ftp.close()
    def __getattr__(self,name):
        value=getattr(self.ftp,name)
        if not callable(value):return value
        def guarded(*args,**kwargs):
            self.remaining()
            result=value(*args,**kwargs)
            self.remaining()
            return result
        return guarded

class Control:
    def __init__(self, candidate, phase):
        self.candidate, self.phase = candidate, phase
        manifest_bytes = (candidate / 'manifest.json').read_bytes()
        self.manifest_sha = sha(manifest_bytes)
        self.manifest = json.loads(manifest_bytes)
        self.fixture = self.manifest.get('fixture', 'bootstrap')
        baseline_fixture = self.manifest.get('baseline_log_fixture', 'bootstrap')
        self.baseline_fixture = baseline_fixture
        require(self.fixture in LOG_NAMES and baseline_fixture in LOG_NAMES, 'unknown log fixture')
        from build_public_hip import validate_results_directory
        results_dir = self.manifest['results_dir']
        baseline_results_dir = self.manifest.get('baseline_results_dir', '/data/ps5-hip/results')
        for directory in (results_dir, baseline_results_dir):
            require(isinstance(directory, str), 'invalid results directory type')
            validate_results_directory(directory)
        self.log = results_dir + '/' + LOG_NAMES[self.fixture]
        self.baseline_log = baseline_results_dir + '/' + LOG_NAMES[baseline_fixture]
        self.baseline_log_preserved = self.manifest.get('baseline_log_preserved', False)
        require(type(self.baseline_log_preserved) is bool, 'invalid baseline log preservation mode')
        require(not self.baseline_log_preserved or
                re.fullmatch('[0-9a-f]{64}', self.manifest.get('baseline_log_sha256') or ''),
                'preserved baseline log requires digest')
        review = json.loads((candidate / 'review.json').read_text())
        require(review.get('approved') is True and review.get('manifest_sha256') == sha((candidate / 'manifest.json').read_bytes()),
                'independent review does not approve this manifest')
        require(self.manifest['host'] == HOST and self.manifest['title_id'] == TITLE,
                'wrong candidate environment')
        if phase == 'execute':
            require(review.get('deployment_receipt_sha256') == sha((candidate / 'deploy/receipt.json').read_bytes()),
                    'deployment receipt has not been independently reviewed')
        if self.prefix_phase():self.preload_prefix_validators()
        self.verify()
        require((ROOT/self.manifest['package']).stat().st_size <= (MODEL_PACKAGE_LIMIT if self.model_phase() and 'external_model' not in self.manifest else package_limit(self.fixture)), 'candidate package size exceeds fixture bound')
        if self.fixture == 'llm_client':
            self.authenticate_llm_client()
        if self.fixture == 'rmsnorm':
            from build_public_hip import rmsnorm_reference_inputs
            bound = self.manifest.get('rmsnorm', {})
            reference = (ROOT / bound.get('reference', '')).resolve()
            require(reference.is_relative_to(candidate.resolve()) and
                    self.manifest['files'].get(str((reference/'receipt.json').relative_to(ROOT))) ==
                    bound.get('reference_receipt_sha256'), 'RMSNorm reference not frozen')
            require(rmsnorm_reference_inputs(reference)['receipt_sha256'] ==
                    bound['reference_receipt_sha256'], 'RMSNorm reference identity changed')
        if self.fixture == 'q4':
            self.authenticate_q4()
        if self.fixture == 'weighted_qkv':
            self.authenticate_weighted_qkv()
        self.out = candidate / phase
        self.out.mkdir(exist_ok=False)
        self.record = {'phase': phase, 'host': HOST, 'title_id': TITLE, 'accepted': False, 'launched': False,
                       'manifest_sha256': self.manifest_sha,
                       'package_sha256': stream_digest(ROOT / self.manifest['package'])}
        self.token = 'source-hip-' + str(uuid.uuid4())
        self.owned = False
        self.ftp = None
        self.history = None
        self.stop = threading.Event()
        self.klog = bytearray()
        self.capture_errors = []

    def verify(self):
        require(sha((self.candidate / 'manifest.json').read_bytes()) == self.manifest_sha, 'manifest changed')
        def check(item):
            name, digest = item
            path = ROOT / name
            require(path.resolve().is_relative_to(ROOT) and stream_digest(path) == digest, 'frozen input changed: ' + name)
        # Bound open files and overlap WSL filesystem latency; every call still reads fresh bytes.
        with ThreadPoolExecutor(max_workers=4) as workers:
            for _ in workers.map(check, self.manifest['files'].items()):
                pass

    def phase_remaining(self, requested=1800):
        if not self.model_phase(): return requested
        if not hasattr(self, 'phase_deadline'): self.phase_deadline = time.monotonic() + (4200 if self.corpus_phase() else 1800)
        remaining = self.phase_deadline - time.monotonic()
        require(remaining > 0, 'whole model phase deadline exhausted')
        return min(requested, remaining)

    def model_readback(self, remote, size, digest, records, seconds, output=None):
        result=readback(self.ftp,remote,size,digest,records,self.phase_remaining(seconds),output,guard=self.phase_remaining)
        self.phase_remaining()
        return result

    def model_upload(self, package, temporary, size, digest, records, seconds):
        # stage_upload contains upload + readback: use one shared deadline for both via FTP guard.
        result=stage_upload(self.ftp,package,temporary,size,digest,records,self.phase_remaining(seconds),guard=self.phase_remaining)
        self.phase_remaining()
        return result

    def model_phase(self):
        return getattr(self,'fixture',None) == 'llm_client' and self.manifest.get('llm_client', {}).get('phase') in ('model-gate','model-prefix','model-layer','model-full')

    def prefix_phase(self):
        return self.model_phase() and self.manifest['llm_client']['phase'] in ('model-prefix','model-layer','model-full')

    def authenticate_llm_client(self):
        if self.model_phase():
            validate_model_candidate(self.candidate,self.manifest)
            return
        from freeze_source_candidate import validate_llm_client
        bound = self.manifest.get('llm_client', {})
        require(bound.get('phase') == 'vocab', 'full model phase is not authorized')
        receipt_path = self.candidate / 'build-receipt.json'
        receipt = json.loads(receipt_path.read_text())
        # Validate frozen package/audit and original authenticated producer closure again before network access.
        result = validate_llm_client(self.candidate, receipt, frozen=True)
        require(result == bound, 'frozen client provenance differs')

    def authenticate_q4(self):
        from public_hip_q4_assets import assets
        bound=self.manifest.get('q4', {})
        reference=(ROOT/bound.get('reference','')).resolve()
        fixture=(ROOT/bound.get('fixture','')).resolve()
        require(reference.is_relative_to(self.candidate.resolve()) and fixture.is_relative_to(ROOT),
                'Q4 reference or fixture outside frozen scope')
        require(self.manifest['files'].get(str((reference/'receipt.json').relative_to(ROOT)))==
                bound.get('reference_receipt_sha256'), 'Q4 reference not frozen')
        authenticated=assets(fixture,reference)
        require(authenticated['reference']['receipt_sha256']==bound.get('reference_receipt_sha256'),
                'Q4 reference identity changed')
        for name in ('q8','q4'):
            require(sha((fixture/(name+'.hsaco')).read_bytes())==bound.get('module_sha256',{}).get(name),
                    'Q4 module identity changed')
        app=self.candidate/'package/app'
        for name,path in authenticated['assets'].items():
            digest=sha(Path(path).read_bytes())
            require(self.manifest['package_files'].get(name)==digest and sha((app/name).read_bytes())==digest,
                    'Q4 packaged asset differs: '+name)
        require({n for n in self.manifest['package_files'] if n.startswith('q4/')}==set(authenticated['assets']),
                'Q4 packaged asset inventory differs')

    def authenticate_weighted_qkv(self):
        from public_hip_weighted_qkv_assets import assets
        bound=self.manifest.get('weighted_qkv',{})
        reference=(ROOT/bound.get('reference','')).resolve()
        weighted=(ROOT/bound.get('weighted_fixture','')).resolve();q4=(ROOT/bound.get('q4_fixture','')).resolve()
        require(reference.is_relative_to(self.candidate.resolve()) and weighted.is_relative_to(ROOT) and q4.is_relative_to(ROOT),'weighted QKV scope differs')
        require(self.manifest['files'].get(str((reference/'receipt.json').relative_to(ROOT)))==bound.get('reference_receipt_sha256'),'weighted reference not frozen')
        authenticated=assets(weighted,q4,reference)
        require(authenticated['reference']['receipt_sha256']==bound.get('reference_receipt_sha256'),'weighted reference changed')
        for name,folder in (('weighted',weighted),('q8',q4),('q4',q4)):
            require(sha((folder/(name+'.hsaco')).read_bytes())==bound.get('module_sha256',{}).get(name),'weighted QKV module changed')
        expected={n:sha(Path(p).read_bytes()) for n,p in authenticated['assets'].items()}
        require(len(expected)==10 and {n:h for n,h in self.manifest['package_files'].items() if n.startswith('weighted-qkv/')}==expected,'weighted QKV exact10 assets required')
        for n,h in expected.items():require(sha((self.candidate/'package/app'/n).read_bytes())==h,'weighted QKV packaged asset changed')

    def owned_lock(self):
        require(self.owned and LOCK.read_text() == self.token, 'console lock changed')

    def health(self):
        for port in (2121, 3232, 9021):
            with socket.create_connection((HOST, port), timeout=3):
                pass
        return True

    def fetch(self, path, bound=8388608, deadline_seconds=30):
        require(0 < deadline_seconds <= 90, 'invalid FTP read deadline')
        data = bytearray()
        started = time.monotonic()
        entry = {'path': path, 'byte_limit': bound, 'deadline_seconds': deadline_seconds,
                 'received_bytes': 0, 'complete': False}
        received_hash = hashlib.sha256()
        def append(block):
            entry['received_bytes'] += len(block)
            received_hash.update(block)
            require(entry['received_bytes'] <= bound, 'FTP read size bound: ' + path)
            require(time.monotonic() - started < deadline_seconds, 'FTP read deadline: ' + path)
            data.extend(block)
        try:
            self.ftp.retrbinary('RETR ' + path, append, blocksize=65536)
            require(time.monotonic() - started < deadline_seconds, 'FTP read deadline: ' + path)
            entry['complete'] = True
            return bytes(data)
        except Exception as error:
            entry['error'] = repr(error)
            raise
        finally:
            entry['elapsed_seconds'] = time.monotonic() - started
            entry['received_sha256'] = received_hash.hexdigest()
            self.record.setdefault('ftp_reads', []).append(entry)

    def fetch_image(self, path, bound):
        # A verified 9 MiB baseline transfer exceeded 30 s on a healthy console.
        # Keep small logs responsive and the 8 s socket inactivity timeout, but
        # allow a bounded 90 s for complete images; byte/hash checks are unchanged.
        return self.fetch(path, bound, deadline_seconds=90)

    def absent(self, path):
        require(not any(row.endswith(' ' + Path(path).name)
                        for row in self.listing(str(Path(path).parent))), 'path already exists: ' + path)

    def baseline_log_read(self):
        digest = self.manifest.get('baseline_log_sha256')
        if not digest:
            self.fresh_log()
            return None
        require(re.fullmatch('[0-9a-f]{64}', digest), 'invalid baseline log digest')
        backup = self.baseline_log + '.' + digest + '.preserved'
        if self.baseline_log_preserved:
            self.absent(self.baseline_log)
            self.fresh_log()
            source = backup
        else:
            self.absent(backup)
            source = self.baseline_log
        raw = self.fetch(source, 2*1024*1024)
        require(sha(raw) == digest, 'baseline log changed')
        (self.out / 'previous-application.log').write_bytes(raw)
        return raw, backup

    def preserve_baseline_log(self, previous):
        if previous is None:
            return
        raw, backup = previous
        if self.baseline_log_preserved:
            self.absent(self.baseline_log)
        else:
            require(self.fetch(self.baseline_log, 2*1024*1024) == raw, 'baseline log changed before preservation')
            self.absent(backup)
            self.ftp.rename(self.baseline_log, backup)
        require(self.fetch(backup, 2*1024*1024) == raw, 'preserved log mismatch')
        self.record['log_backup'] = backup
        self.record['log_already_preserved'] = self.baseline_log_preserved

    def listing(self, path):
        rows = []
        self.ftp.retrlines('LIST ' + path, rows.append)
        require(len(rows) < 4096, 'directory bound')
        return rows

    def active_titles(self):
        self.owned_lock()
        rows = self.listing('/mnt/sandbox')
        require(rows, 'unknown sandbox state')
        return {match[1] for row in rows if (match := re.search(r'\s((?!NPXS)[A-Z]{4}[0-9]{5})_', row))}

    def idle(self):
        require(not self.active_titles(), 'another title active')

    def history_read(self, name):
        raw = self.fetch(HISTORY)
        require(self.history is None or raw.startswith(self.history), 'ShadowMount history reset')
        self.history = raw
        (self.out / (name + '.log')).write_bytes(raw)
        return raw

    def full_phase(self):
        return self.model_phase() and self.manifest['llm_client']['phase']=='model-full'

    def generation_phase(self):
        return self.full_phase() and self.manifest.get('model_full',{}).get('profile','smoke') in ('generation','corpus')

    def corpus_phase(self):
        return self.full_phase() and self.manifest.get('model_full',{}).get('profile')=='corpus'

    def application_log_limit(self):
        return 2*1024*1024 if self.corpus_phase() else 65536

    def prefix_files(self):
        if self.generation_phase():return self.manifest['model_full']['raw_files']
        return self.compute_contract().FILES

    def prefix_dump_name(self,name):
        if self.corpus_phase():return name=='corpus.json' or '/' in name or name.startswith('run-')
        if self.generation_phase() and name in ('generation.json','steps.jsonl'):return True
        return name.startswith('run-') and name.endswith(('.f32le','.f16le')) if self.layer_phase() or self.full_phase() else name.startswith('token-') and name.endswith('.f32le')

    def generation_policy(self):
        from build_llm_generation_reference import loads
        binding=self.manifest['model_full'];reference=(ROOT/binding['reference']).resolve()
        require(reference.is_relative_to(self.candidate.resolve()),'generation reference outside candidate')
        values={}
        from build_llm_generation_reference import corpus_cases
        names=['corpus.json',*(case+'/'+name for case,_ in corpus_cases() for name in ('generation.json','steps.jsonl'))] if self.corpus_phase() else ('generation.json','steps.jsonl')
        for name in names:
            path=reference/'raw'/name
            require(path.resolve()==path and not path.is_symlink() and path.stat().st_size==binding['raw_files'][name],'generation metadata kind/size')
            raw=path.read_bytes()
            require(self.manifest['files'].get(str(path.relative_to(ROOT)))==sha(raw),'generation frozen metadata identity differs')
            values[name]=raw
        policy={case:loads(values[case+'/generation.json'].decode('utf-8')) for case,_ in corpus_cases()} if self.corpus_phase() else loads(values['generation.json'].decode('utf-8'))
        return policy,{name:len(raw) for name,raw in values.items()}

    def layer_phase(self):
        return self.model_phase() and self.manifest['llm_client']['phase']=='model-layer'

    def compute_contract(self):
        if self.generation_phase():
            import source_llm_generation_log as contract
        elif self.full_phase():
            import source_llm_full_log as contract
        elif self.layer_phase():
            import source_llm_layer_log as contract
        else:
            import source_llm_prefix_log as contract
        return contract

    def preload_prefix_validators(self):
        # All prefix-specific lazy imports must be resident before the first full authentication.
        import source_llm_prefix_log
        import verify_llm_prefix_readbacks
        import build_llm_prefix_reference
        import public_hip_prefix_assets
        if self.full_phase():
            import source_llm_full_log
            import verify_llm_full_readbacks
            import build_llm_full_model_reference
            import public_hip_full_assets
            import source_llm_generation_log
            import build_llm_generation_reference
            import build_llm_generation_numeric_test
        elif self.layer_phase():
            import source_llm_layer_log
            import verify_llm_layer_readbacks
            import build_llm_layer_reference
            import public_hip_layer_assets

    def frozen_controller_payload(self,name):
        require(name in ('snapshot','post_snapshot','launch','close'),'unknown controller payload')
        manifest_bytes=(self.candidate/'manifest.json').read_bytes()
        require(sha(manifest_bytes)==self.manifest_sha,'controller manifest changed')
        relative=self.manifest[name];path=(ROOT/relative).resolve()
        require(not Path(relative).is_absolute() and path.is_relative_to(self.candidate.resolve()),'controller payload outside candidate')
        payload=path.read_bytes()
        require(len(payload)<=262144 and self.manifest['files'].get(relative)==sha(payload),'controller payload differs')
        return payload

    def final_prefix_authentication(self):
        require(self.record.get('accepted') is True,'prefix phase has not passed')
        self.record['accepted']=False
        self.phase_remaining();started=time.monotonic()
        self.verify()
        self.authenticate_llm_client()
        self.phase_remaining();self.owned_lock()
        self.record['prefix_final_authentication']={'status':'PASS','elapsed_seconds':time.monotonic()-started}
        self.record['accepted']=True

    def verify_close_payload(self):
        require(sha((self.candidate/'manifest.json').read_bytes())==self.manifest_sha,'cleanup manifest changed')
        path=(ROOT/self.manifest['close']).resolve()
        require(path.is_relative_to(ROOT) and self.manifest['files'].get(self.manifest['close'])==stream_digest(path),'cleanup close payload differs')

    def send(self, name, bound=262144):
        if name != 'close':self.phase_remaining()
        prefix=self.prefix_phase()
        if prefix:
            payload=self.frozen_controller_payload(name)
        else:
            if name == 'close' and self.model_phase():self.verify_close_payload()
            else:self.verify()
        if name == 'close' and self.model_phase():self.cleanup_remaining()
        else:self.phase_remaining()
        self.owned_lock()
        if not prefix:
            payload = (ROOT / self.manifest[name]).read_bytes()
            if name == 'close' and self.model_phase():require(sha(payload)==self.manifest['files'].get(self.manifest['close']),'cleanup payload changed after verification')
        data = bytearray()
        with (self.out / (name + '-consumed')).open('x') as stream:
            stream.write(self.token)
        deadline = time.monotonic() + 10
        try:
            if name == 'close' and self.model_phase():self.cleanup_remaining()
            else:self.phase_remaining()
            with socket.create_connection((HOST, 9021), timeout=3) as sock:
                if name == 'close' and self.model_phase():self.cleanup_remaining()
                sock.settimeout(8)
                if name == 'close' and self.model_phase():self.cleanup_remaining()
                else:self.phase_remaining()
                if prefix:self.owned_lock()
                sock.sendall(payload)
                while True:
                    if name == 'close' and self.model_phase():self.cleanup_remaining()
                    else:self.phase_remaining()
                    require(time.monotonic() < deadline, 'controller deadline')
                    block = sock.recv(min(65536, bound + 1 - len(data)))
                    if name == 'close' and self.model_phase():self.cleanup_remaining()
                    else:self.phase_remaining()
                    if not block:
                        break
                    data.extend(block)
                    require(len(data) <= bound, 'controller response bound')
        finally:
            (self.out / (name + '.log')).write_bytes(data)
        return bytes(data)

    def installed(self):
        if self.model_phase():return self.installed_model()
        app = ROOT / self.manifest['app']
        image = (ROOT / self.manifest['package']).read_bytes()
        require(self.fetch_image(REMOTE, len(image)) == image, 'installed image mismatch')
        require(self.fetch('/user/app/' + TITLE + '/mount.lnk', 1024) == MOUNT.encode(), 'mount link changed')
        require(self.fetch('/user/appmeta/' + TITLE + '/param.json', 65536) ==
                (app / 'sce_sys/param.json').read_bytes(), 'registered metadata mismatch')
        for name in self.manifest['package_files']:
            expected = (app / name).read_bytes()
            require(self.fetch(MOUNT + '/' + name, len(expected)) == expected, 'mounted bytes mismatch: ' + name)

    def installed_model(self):
        transfer=self.manifest['model_transfer']
        self.model_readback(REMOTE,transfer['bytes'],transfer['sha256'],self.record.setdefault('ftp_reads',[]),transfer['deadline_seconds'])
        require(self.fetch('/user/app/'+TITLE+'/mount.lnk',1024)==MOUNT.encode(),'mount link changed')
        app=ROOT/self.manifest['app']
        require(self.fetch('/user/appmeta/'+TITLE+'/param.json',65536)==(app/'sce_sys/param.json').read_bytes(),'registered metadata mismatch')
        for name,digest in self.manifest['package_files'].items():
            path=app/name
            local_identity(path,path.stat().st_size,digest)
            self.model_readback(MOUNT+'/'+name,path.stat().st_size,digest,self.record.setdefault('ftp_reads',[]),transfer['deadline_seconds'])

    def external_model_entry(self, path):
        entries = [row.split(None, 8) for row in self.listing(str(Path(path).parent))]
        found = [row for row in entries if len(row) == 9 and
                 (row[8] == Path(path).name or row[8].startswith(Path(path).name + ' -> '))]
        require(len(found) <= 1, 'ambiguous external model entry')
        if found:
            require(found[0][0].startswith('-'), 'external model must be a regular file')
        return bool(found)

    def verify_external_model(self):
        if 'external_model' not in self.manifest:
            return
        from external_model import validate_external_model
        model = self.manifest['external_model']
        validate_external_model(model)
        require(self.external_model_entry(model['path']), 'external model is missing')
        self.model_readback(model['path'], model['bytes'], model['sha256'],
                            self.record.setdefault('external_model_reads', []), 1800)

    def provision_external_model(self):
        if 'external_model' not in self.manifest:
            return
        require(self.phase == 'deploy', 'model provisioning is deploy-only')
        from external_model import validate_external_model
        model = self.manifest['external_model']
        validate_external_model(model)
        self.ensure_directory(str(Path(model['path']).parent), 'model')
        evidence = self.record['external_model'] = {**model, 'complete': False}
        if self.external_model_entry(model['path']):
            evidence['action'] = 'reuse'
            self.verify_external_model()
        else:
            evidence['action'] = 'upload'
            temporary = str(Path(model['path']).parent) + '/.model-' + sha(self.token.encode())[:32] + '.upload'
            evidence['temporary'] = temporary
            require(not self.external_model_entry(temporary), 'model staging path already exists')
            source = Path(self.manifest['llm_client']['model']['asset'])
            self.model_upload(source, temporary, model['bytes'], model['sha256'],
                              self.record.setdefault('external_model_transfers', []), 1800)
            self.idle()
            require(not self.external_model_entry(model['path']), 'external model appeared before promotion')
            # Shared-console ownership and the final absence check guard FTP rename.
            self.phase_remaining()
            self.ftp.rename(temporary, model['path'])
            self.phase_remaining()
            evidence['promoted'] = True
            self.verify_external_model()
        evidence['complete'] = True

    def fresh_log(self):
        require(not any(row.endswith(' ' + Path(self.log).name) for row in self.listing(str(Path(self.log).parent))), 'result log already exists')

    def fresh_rmsnorm_dumps(self):
        if self.fixture != 'rmsnorm':
            return
        directory = self.manifest['results_dir']
        rows = self.listing(directory)
        for name in RMSNORM_RAW_NAMES:
            require(not any(row.endswith(' ' + name) for row in rows),
                    'RMSNorm dump already exists: ' + name)
        self.record['rmsnorm_raw_absent_before_launch'] = list(RMSNORM_RAW_NAMES)

    def collect_rmsnorm_dumps(self, partial=False):
        if self.fixture != 'rmsnorm':
            return
        folder = self.out/'raw'
        folder.mkdir(exist_ok=True)
        records = self.record.setdefault('rmsnorm_raw_files', {})
        for name in RMSNORM_RAW_NAMES:
            if name in records:
                require(sha((folder/name).read_bytes()) == records[name]['sha256'], 'collected dump changed')
                continue
            try:
                raw = self.fetch(self.manifest['results_dir']+'/'+name, RMSNORM_RAW_BYTES)
                with (folder/name).open('xb') as stream:
                    stream.write(raw)
                require(len(raw) == RMSNORM_RAW_BYTES, 'RMSNorm dump length: '+name)
                records[name] = {'bytes':len(raw), 'sha256':sha(raw)}
            except Exception as error:
                if not partial:
                    raise
                self.record.setdefault('rmsnorm_raw_capture_errors', {})[name] = repr(error)
        if not partial:
            require(set(records) == set(RMSNORM_RAW_NAMES), 'incomplete RMSNorm readbacks')

    def verify_rmsnorm_dumps(self):
        if self.fixture != 'rmsnorm':
            return
        self.verify()
        from verify_public_hip_rmsnorm import verify
        result = verify(ROOT/self.manifest['rmsnorm']['reference'], self.out/'raw', self.out/'numerical')
        require(result.get('status') == 'PASS', 'RMSNorm raw numerical verification failed')
        metrics = {'max_abs':float(format(max(c['maximum_absolute_error'] for c in result['cases']), '.9g')),
                   'max_budget_fraction':float(format(max(c['maximum_budget_fraction'] for c in result['cases']), '.9g'))}
        require(self.record.get('application', {}).get('reported_metrics') == metrics,
                'RMSNorm transcript metrics differ from raw readbacks')
        self.record['rmsnorm_numerical'] = {'status':'PASS',
            'receipt_sha256':sha((self.out/'numerical/receipt.json').read_bytes())}

    def fresh_q4_dumps(self):
        if self.fixture!='q4':return
        from verify_public_hip_q4 import DUMP_NAMES
        names={row.rsplit(' ',1)[-1] for row in self.listing(self.manifest['results_dir'])}
        require(not any(n.startswith('q4-') and n.endswith('.raw') for n in names), 'Q4 dump already exists')
        self.record['q4_raw_absent_before_launch']=sorted(DUMP_NAMES)

    def collect_q4_dumps(self,partial=False):
        if self.fixture!='q4':return
        from verify_public_hip_q4 import STAGES,CAPACITIES,DUMP_NAMES
        if not hasattr(self,'q4_capture_deadline'):
            self.q4_capture_deadline=time.monotonic()+120
            self.record['q4_raw_capture_limit_seconds']=120
        folder=self.out/'raw';folder.mkdir(exist_ok=True)
        records=self.record.setdefault('q4_raw_files',{})
        if not partial:
            names={row.rsplit(' ',1)[-1] for row in self.listing(self.manifest['results_dir'])}
            require({n for n in names if n.startswith('q4-') and n.endswith('.raw')}==DUMP_NAMES,
                    'Q4 remote raw inventory differs')
        for case,stage,_,_,_ in STAGES:
            for buffer,capacity in CAPACITIES.items():
                name=f'q4-{case}.{stage}.{buffer}.raw'
                if name in records:
                    require(sha((folder/name).read_bytes())==records[name]['sha256'],'collected Q4 dump changed')
                    continue
                remaining=self.q4_capture_deadline-time.monotonic()
                if remaining<=0:
                    self.record['q4_raw_capture_deadline_exceeded']=True
                    if partial:return
                    raise RuntimeError('Q4 total raw capture deadline')
                try:
                    raw=self.fetch(self.manifest['results_dir']+'/'+name,capacity+32,deadline_seconds=min(30,remaining))
                    with (folder/name).open('xb') as stream:stream.write(raw)
                    require(len(raw)==capacity+32,'Q4 dump length: '+name)
                    require(time.monotonic()<self.q4_capture_deadline,'Q4 total raw capture deadline')
                    records[name]={'bytes':len(raw),'sha256':sha(raw)}
                except Exception as error:
                    self.record.setdefault('q4_raw_capture_errors',{})[name]=repr(error)
                    if not partial:raise
        if not partial:require(set(records)==DUMP_NAMES,'incomplete Q4 readbacks')

    def verify_q4_dumps(self):
        if self.fixture!='q4':return
        self.verify()
        from verify_public_hip_q4 import verify
        bound=self.manifest['q4'];output=self.out/'numerical/receipt.json'
        result=verify(ROOT/bound['fixture'],ROOT/bound['reference'],self.out/'raw',output)
        require(result.get('status')=='PASS','Q4 raw numerical verification failed')
        stages=[dict(name=c['name'],stage=c['stage'],
                     max_abs=float(format(c['maximum_absolute_error'],'.9g')),
                     max_budget_fraction=float(format(c['maximum_budget_fraction'],'.9g'))) for c in result['records']]
        metrics=dict(max_abs=max(c['max_abs'] for c in stages),
                     max_budget_fraction=max(c['max_budget_fraction'] for c in stages),stages=stages)
        require(self.record.get('application',{}).get('reported_metrics')==metrics,
                'Q4 transcript metrics differ from raw readbacks')
        self.record['q4_numerical']={'status':'PASS','receipt_sha256':sha(output.read_bytes())}

    def fresh_weighted_qkv_dumps(self):
        if self.fixture!='weighted_qkv':return
        from verify_weighted_qkv_readbacks import DUMP_NAMES
        names={row.rsplit(' ',1)[-1] for row in self.listing(self.manifest['results_dir'])}
        require(not any(n.startswith('weighted-qkv-') and n.endswith('.raw') for n in names),'weighted QKV dump already exists')
        self.record['weighted_qkv_raw_absent_before_launch']=sorted(DUMP_NAMES)

    def collect_weighted_qkv_dumps(self,partial=False):
        if self.fixture!='weighted_qkv':return
        from verify_weighted_qkv_readbacks import DUMP_NAMES,CAPACITIES,STAGES
        if not hasattr(self,'weighted_qkv_capture_deadline'):
            self.weighted_qkv_capture_deadline=time.monotonic()+120
            self.record['weighted_qkv_raw_capture_limit_seconds']=120
        folder=self.out/'raw';folder.mkdir(exist_ok=True)
        records=self.record.setdefault('weighted_qkv_raw_files',{})
        names={row.rsplit(' ',1)[-1] for row in self.listing(self.manifest['results_dir'])}
        if not partial:require({n for n in names if n.startswith('weighted-qkv-') and n.endswith('.raw')}==DUMP_NAMES,'weighted QKV remote inventory differs')
        for stage in STAGES:
            for buffer,capacity in CAPACITIES.items():
                name=f'weighted-qkv-{stage}.{buffer}.raw'
                if name in records or (partial and name not in names):continue
                remaining=self.weighted_qkv_capture_deadline-time.monotonic()
                if remaining<=0:
                    self.record['weighted_qkv_raw_capture_deadline_exceeded']=True
                    if partial:return
                    raise RuntimeError('weighted QKV total raw capture deadline')
                try:
                    raw=self.fetch(self.manifest['results_dir']+'/'+name,capacity+32,deadline_seconds=min(30,remaining))
                    with (folder/name).open('xb') as stream:stream.write(raw)
                    require(len(raw)==capacity+32,'weighted QKV dump length: '+name)
                    require(time.monotonic()<self.weighted_qkv_capture_deadline,'weighted QKV total raw capture deadline')
                    records[name]={'bytes':len(raw),'sha256':sha(raw)}
                except Exception as error:
                    self.record.setdefault('weighted_qkv_raw_capture_errors',{})[name]=repr(error)
                    if not partial:raise
        if not partial:require(set(records)==DUMP_NAMES,'incomplete weighted QKV readbacks')

    def verify_weighted_qkv_dumps(self):
        if self.fixture!='weighted_qkv':return
        self.verify()
        from verify_weighted_qkv_readbacks import verify
        output=self.out/'numerical/receipt.json'
        result=verify(ROOT/self.manifest['weighted_qkv']['reference'],self.out/'raw',output)
        require(result.get('status')=='PASS' and result.get('raw_files')==30 and result.get('projection_values')==2560,'weighted QKV numerical verification failed')
        values=result['weighted_metrics'];metrics={key:float(format(values[field],'.9g')) for key,field in (('max_abs','maximum_absolute_error'),('max_budget_fraction','maximum_budget_fraction'))}
        require(self.record.get('application',{}).get('reported_metrics')==metrics,'weighted QKV transcript metrics differ')
        self.record['weighted_qkv_numerical']={'status':'PASS','receipt_sha256':sha(output.read_bytes())}

    def fresh_prefix_dumps(self):
        if not self.prefix_phase():return
        FILES=self.prefix_files()
        names={row.rsplit(' ',1)[-1] for row in self.listing(self.manifest['results_dir'])}
        if self.corpus_phase():
            from build_llm_generation_reference import corpus_cases
            require(not names.intersection(case for case,_ in corpus_cases()),'corpus case directories already exist')
            require(names <= {LOG_NAMES['llm_client']},'corpus unexpected files already exist')
        require(not any(self.prefix_dump_name(n) for n in names),'prefix raw files already exist')
        self.record['prefix_raw_absent_before_launch']=sorted(FILES)

    def collect_prefix_dumps(self,partial=False):
        if not self.prefix_phase():return
        FILES=self.prefix_files()
        if not hasattr(self,'prefix_capture_deadline'):
            limit=600 if self.corpus_phase() else 120
            self.prefix_capture_deadline=time.monotonic()+limit
            self.record['prefix_raw_capture_limit_seconds']=limit
        names={row.rsplit(' ',1)[-1] for row in self.listing(self.manifest['results_dir'])}
        if self.corpus_phase():
            from build_llm_generation_reference import corpus_cases
            require(names <= {'corpus.json',LOG_NAMES['llm_client'],*(case for case,_ in corpus_cases())},'corpus root inventory differs')
            for case,_ in corpus_cases():
                if case not in names:
                    require(partial,'corpus case directory missing')
                    continue
                names.remove(case)
                children={row.rsplit(' ',1)[-1] for row in self.listing(self.manifest['results_dir']+'/'+case)}
                require(all(n not in ('.','..') and '/' not in n and '\\' not in n for n in children),'corpus child name')
                names.update(case+'/'+n for n in children)
        records=self.record.setdefault('prefix_raw_files',{})
        folder=self.out/'raw';folder.mkdir(exist_ok=True)
        if not partial:require({n for n in names if self.prefix_dump_name(n)}==set(FILES),'prefix raw inventory differs')
        for name,size in FILES.items():
            if name in records or (partial and name not in names):continue
            remaining=self.prefix_capture_deadline-time.monotonic()
            if remaining<=0:
                self.record['prefix_raw_capture_deadline_exceeded']=True
                if partial:break
                raise RuntimeError('prefix total raw capture deadline')
            try:
                raw=self.fetch(self.manifest['results_dir']+'/'+name,size,self.phase_remaining(min(15,remaining)))
                require(len(raw)==size,'prefix raw byte count differs: '+name)
                require(time.monotonic()<self.prefix_capture_deadline,'prefix total raw capture deadline')
                path=folder/name
                if self.corpus_phase():
                    path.parent.mkdir(exist_ok=True)
                    require(path.parent.resolve()==path.parent and not path.parent.is_symlink(),'corpus local output directory')
                with path.open('xb') as stream:stream.write(raw)
                records[name]={'bytes':len(raw),'sha256':sha(raw)}
            except Exception as error:
                if not partial:raise
                self.record.setdefault('prefix_raw_capture_errors',{})[name]=repr(error)
        if not partial:require(set(records)==set(FILES),'incomplete prefix raw capture')

    def verify_prefix_dumps(self):
        if not self.prefix_phase():return
        if self.full_phase():
            from verify_llm_full_readbacks import verify
        elif self.layer_phase():
            from verify_llm_layer_readbacks import verify
        else:
            from verify_llm_prefix_readbacks import verify
        output=self.out/'prefix-numerical.json'
        binding=self.manifest['model_full' if self.full_phase() else 'model_layer' if self.layer_phase() else 'model_prefix']
        result=verify(ROOT/binding['reference'],self.out/'raw',output,**({'profile':binding['profile']} if self.generation_phase() else {}))
        require(result.get('status')=='PASS' and result.get('raw_files')==len(self.prefix_files()) and result.get('raw_bytes')==sum(self.prefix_files().values()),'prefix numerical verification failed')
        require(result.get('reference_receipt_sha256')==binding['reference_receipt_sha256'],'prefix numerical reference differs')
        require(result.get('raw_sha256')=={name:entry['sha256'] for name,entry in self.record['prefix_raw_files'].items()},'prefix compared bytes differ from captured bytes')
        self.record['prefix_numerical']={'status':'PASS','receipt_sha256':sha(output.read_bytes())}

    def ensure_results_directory(self):
        require(self.phase == 'deploy', 'results directory creation is deploy-only')
        from build_public_hip import validate_results_directory
        directory = self.manifest['results_dir']
        validate_results_directory(directory)
        self.ensure_directory(directory, 'results')

    def ensure_directory(self, directory, kind):
        require(self.phase == 'deploy', 'directory creation is deploy-only')
        parent = '/'
        for leaf in directory[1:].split('/'):
            path = parent.rstrip('/') + '/' + leaf
            def matches():
                entries = [row.split(None, 8) for row in self.listing(parent)]
                return [row for row in entries if len(row) == 9 and
                        (row[8] == leaf or row[8].startswith(leaf + ' -> '))]
            found = matches()
            require(len(found) <= 1, 'ambiguous results directory entry')
            if found:
                require(found[0][0].startswith('d'), 'results path is not a directory')
            else:
                self.ftp.mkd(path)
                self.record.setdefault(kind + '_directories_created', []).append(path)
                if path == directory:
                    self.record[kind + '_directory_created'] = directory
                found = matches()
                require(len(found) == 1 and found[0][0].startswith('d'),
                        'created results directory not visible')
            parent = path
        self.listing(directory)
        self.record[kind + '_directory_verified'] = directory

    def deploy(self):
        if self.model_phase():return self.deploy_model()
        self.idle()
        snapshot(self.send('snapshot'))
        baseline_log = self.manifest.get('baseline_log_sha256')
        old_sha = self.manifest['baseline_image_sha256']
        require(re.fullmatch('[0-9a-f]{64}', old_sha), 'invalid baseline image digest')
        old = self.fetch_image(REMOTE, package_limit(self.baseline_fixture))
        require(sha(old) == old_sha, 'historical baseline differs; replan')
        (self.out / 'old.ffpkg').write_bytes(old)
        self.ensure_results_directory()
        if self.baseline_log != self.log:
            self.fresh_log()
        previous = self.baseline_log_read()
        backup = REMOTE + '.source-rebuild-' + old_sha + '.preserved'
        temporary = '/data/homebrew/.PPSA99014-source-' + self.token + '.upload'
        entries = self.listing('/data/homebrew')
        require(not any(row.endswith(' ' + Path(backup).name) for row in entries), 'backup already exists')
        image = (ROOT / self.manifest['package']).read_bytes()
        self.idle(); self.verify()
        self.ftp.storbinary('STOR ' + temporary, io.BytesIO(image))
        require(self.fetch_image(temporary, len(image)) == image, 'temporary upload mismatch')
        self.idle()
        self.preserve_baseline_log(previous)
        self.fresh_log()
        start = self.history_read('before-preserve')
        require(self.fetch_image(REMOTE, len(old)) == old, 'baseline changed')
        self.record.update(backup=backup, temporary=temporary)
        self.ftp.rename(REMOTE, backup)
        require(self.fetch_image(backup, len(old)) == old, 'backup mismatch')
        deadline = time.monotonic() + 60
        while True:
            self.idle()
            current = self.history_read('release')
            if re.search(r'\[IMG\]\[LVD\] unmount complete: source=' + re.escape(REMOTE) + ' mount=' + re.escape(MOUNT) + r' unit=\d+', current[len(start):].decode()):
                break
            require(time.monotonic() < deadline, 'old image release timed out')
            time.sleep(3)
        self.idle(); self.verify()
        self.phase_remaining()
        self.ftp.rename(temporary, REMOTE)
        self.phase_remaining()
        self.record['promoted'] = True
        # Give ShadowMount and storage the full runbook settling interval.
        for _ in range(24):
            time.sleep(5)
            self.idle()
        self.installed()
        require(self.fetch_image(backup, len(old)) == old, 'preserved image changed')
        if baseline_log:
            previous_log, log_backup = previous
            require(self.fetch(log_backup, 2*1024*1024) == previous_log, 'preserved log changed')
        self.fresh_log()
        snapshot(self.send('post_snapshot'))
        self.record['accepted'] = True

    def deploy_model(self):
        self.phase_remaining()
        self.ftp = PhaseFTP(self.ftp, self.phase_remaining)
        model_contract(self.manifest)
        self.idle()
        snapshot(self.send('snapshot'))
        self.provision_external_model()
        baseline_log = self.manifest.get('baseline_log_sha256')
        old_sha = self.manifest['baseline_image_sha256']
        require(re.fullmatch('[0-9a-f]{64}', old_sha), 'invalid baseline image digest')
        baseline=self.manifest['baseline_transfer']
        old_size=baseline['bytes']
        self.model_readback(REMOTE,old_size,old_sha,self.record.setdefault('ftp_reads',[]),baseline['deadline_seconds'],self.out/'old.ffpkg')
        self.ensure_results_directory()
        if self.baseline_log != self.log:
            self.fresh_log()
        previous = self.baseline_log_read()
        backup = REMOTE + '.source-rebuild-' + old_sha + '.preserved'
        temporary = '/data/homebrew/.PPSA99014-source-' + self.token + '.upload'
        entries = self.listing('/data/homebrew')
        require(not any(row.endswith(' ' + Path(backup).name) for row in entries), 'backup already exists')
        package=ROOT/self.manifest['package']
        transfer=self.manifest['model_transfer']
        size=transfer['bytes']; digest=transfer['sha256']; seconds=transfer['deadline_seconds']
        self.idle(); self.verify()
        self.absent(temporary)
        self.model_upload(package,temporary,size,digest,self.record.setdefault('ftp_transfers',[]),seconds)
        self.idle()
        self.preserve_baseline_log(previous)
        self.fresh_log()
        start = self.history_read('before-preserve')
        self.model_readback(REMOTE,old_size,old_sha,self.record.setdefault('ftp_reads',[]),baseline['deadline_seconds'])
        self.record.update(backup=backup, temporary=temporary)
        self.ftp.rename(REMOTE, backup)
        self.model_readback(backup,old_size,old_sha,self.record.setdefault('ftp_reads',[]),baseline['deadline_seconds'])
        deadline = time.monotonic() + 60
        while True:
            self.idle()
            current = self.history_read('release')
            if re.search(r'\[IMG\]\[LVD\] unmount complete: source=' + re.escape(REMOTE) + ' mount=' + re.escape(MOUNT) + r' unit=\d+', current[len(start):].decode()):
                break
            require(time.monotonic() < deadline, 'old image release timed out')
            time.sleep(3)
        self.idle(); self.verify()
        self.phase_remaining()
        self.ftp.rename(temporary, REMOTE)
        self.phase_remaining()
        self.record['promoted'] = True
        # Give ShadowMount and storage the full runbook settling interval.
        for _ in range(24):
            time.sleep(5)
            self.idle()
        self.installed()
        self.model_readback(backup,old_size,old_sha,self.record.setdefault('ftp_reads',[]),baseline['deadline_seconds'])
        if baseline_log:
            previous_log, log_backup = previous
            require(self.fetch(log_backup, 2*1024*1024) == previous_log, 'preserved log changed')
        self.fresh_log()
        snapshot(self.send('post_snapshot'))
        self.record['accepted'] = True

    def cleanup_remaining(self):
        if not hasattr(self,'cleanup_deadline'): self.cleanup_deadline=time.monotonic()+60
        require(time.monotonic()<self.cleanup_deadline,'bounded cleanup deadline exhausted')

    def cleanup_ftp(self):
        ftp=self.ftp
        while isinstance(ftp,PhaseFTP):ftp=ftp.ftp
        return ftp

    def cleanup_fetch(self,path,bound):
        require(path in (HISTORY,self.log),'cleanup read path forbidden')
        self.cleanup_remaining();self.owned_lock();data=bytearray()
        def append(block):
            self.cleanup_remaining();require(len(data)+len(block)<=bound,'cleanup read bound');data.extend(block)
        self.cleanup_ftp().retrbinary('RETR '+path,append,blocksize=65536)
        self.cleanup_remaining();return bytes(data)

    def cleanup_history(self,name):
        raw=self.cleanup_fetch(HISTORY,8388608)
        require(self.history is None or raw.startswith(self.history),'ShadowMount history reset')
        self.history=raw;(self.out/(name+'.log')).write_bytes(raw);return raw

    def cleanup_active(self):
        self.cleanup_remaining();self.owned_lock();rows=[]
        def append(row):
            self.cleanup_remaining();require(len(rows)<4096,'cleanup directory bound');rows.append(row)
        self.cleanup_ftp().retrlines('LIST /mnt/sandbox',append)
        self.cleanup_remaining();require(rows,'unknown sandbox state')
        return {match[1] for row in rows if (match:=re.search(r'\s((?!NPXS)[A-Z]{4}[0-9]{5})_',row))}

    def emergency_close(self,baseline):
        current=self.cleanup_history('before-safety-close');_,starts=lifecycle(current[len(baseline):])
        require(len(starts)==1 and starts[0][0]==TITLE,'ambiguous launch; refuse close')
        active=self.cleanup_active();require(active in (set(),{TITLE}),'unrelated title; refuse close')
        if active:
            self.cleanup_remaining();self.send('close');self.record['safety_close_sent']=True;self.cleanup_remaining()
            deadline=time.monotonic()+8
            while self.cleanup_active()=={TITLE} and time.monotonic()<deadline:time.sleep(.5)
        self.record['idle_after_safety_close']=not self.cleanup_active();self.cleanup_history('after-safety-close')

    def observation_seconds(self):
        return 1830 if self.corpus_phase() else 150 if self.model_phase() else 40

    def check_application(self, raw):
        if self.corpus_phase():
            from source_llm_generation_log import check_corpus_full_log
            return check_corpus_full_log(raw,*self.generation_policy())
        if self.generation_phase():
            from source_llm_generation_log import check_full_log
            return check_full_log(raw,*self.generation_policy())
        if self.full_phase():
            from source_llm_full_log import check_full_log
            return check_full_log(raw)
        if self.layer_phase():
            from source_llm_layer_log import check_layer_log
            return check_layer_log(raw)
        if self.prefix_phase():
            from source_llm_prefix_log import check_prefix_log
            return check_prefix_log(raw)
        if self.model_phase(): return check_model_log(raw)
        from source_public_hip_log import check_log
        return check_log(raw, self.fixture)

    def execute(self):
        if self.model_phase():
            self.phase_remaining()
            if getattr(self,'ftp',None) is not None: self.ftp = PhaseFTP(self.ftp, self.phase_remaining)
        deployment = json.loads((self.candidate / 'deploy/receipt.json').read_text())
        require(deployment.get('accepted') is True and deployment.get('healthy_after') is True and
                deployment.get('lock_released') is True and deployment.get('phase') == 'deploy' and
                deployment.get('manifest_sha256') == self.manifest_sha and
                deployment.get('package_sha256') == self.record['package_sha256'], 'fresh deployment not complete')
        self.idle(); self.installed(); self.fresh_log(); self.fresh_rmsnorm_dumps(); self.fresh_q4_dumps(); self.fresh_weighted_qkv_dumps(); self.fresh_prefix_dumps()
        self.verify_external_model()
        before = snapshot(self.send('snapshot'))
        baseline = self.history_read('before-launch')
        sock = socket.create_connection((HOST, 3232), timeout=3)
        def capture():
            try:
                sock.settimeout(.5)
                while not self.stop.is_set():
                    try:
                        block = sock.recv(65536)
                    except socket.timeout:
                        continue
                    require(block and len(self.klog) + len(block) <= 4194304, 'klog EOF/bound')
                    self.klog.extend(block)
            except Exception as error:
                self.capture_errors.append(repr(error))
        worker = threading.Thread(target=capture)
        worker.start()
        launch_offset = 0
        try:
            self.idle(); self.fresh_log(); self.fresh_rmsnorm_dumps(); self.fresh_q4_dumps(); self.fresh_weighted_qkv_dumps(); self.fresh_prefix_dumps()
            require(not self.capture_errors and worker.is_alive(), 'klog unavailable before launch')
            launch_offset = len(self.klog)
            self.record['klog_launch_offset'] = launch_offset
            self.record['launched'] = True
            self.phase_remaining()
            self.send('launch')
            self.phase_remaining() # bounded blocking launch return  # This pinned controller prints no result; lifecycle is authoritative.
            self.record['observation_seconds'] = self.observation_seconds()
            deadline = time.monotonic() + self.record['observation_seconds']
            while True:
                current = self.history_read('observed')
                stopped, starts = lifecycle(current[len(baseline):])
                require(not self.capture_errors and worker.is_alive(), 'klog capture failed')
                require(not native_fault(self.klog[launch_offset:]), 'native fault: no automatic retry')
                if stopped:
                    self.record['ordered_stop'] = True
                    break
                require(time.monotonic() < deadline, 'title observation deadline')
                time.sleep(1)
            self.idle()
            application = self.fetch(self.log, self.application_log_limit())
            (self.out / 'application.log').write_bytes(application)
            self.collect_rmsnorm_dumps(); self.collect_q4_dumps(); self.collect_weighted_qkv_dumps(); self.collect_prefix_dumps()
            from source_public_hip_log import check_log
            self.record['application'] = self.check_application(application)
            self.verify_rmsnorm_dumps(); self.verify_q4_dumps(); self.verify_weighted_qkv_dumps(); self.verify_prefix_dumps()
            # Snapshot uses another marker name so pre/post are each one-use.
            after = snapshot(self.send('post_snapshot'))
            require(before == after, 'matching title mounts changed')
            self.installed()
            self.record['accepted'] = True
        finally:
            if self.record['launched'] and not self.record.get('ordered_stop'):
                try:
                    if self.model_phase():
                        self.emergency_close(baseline)
                    else:
                        current = self.history_read('before-safety-close')
                        _, starts = lifecycle(current[len(baseline):])
                        require(len(starts) == 1 and starts[0][0] == TITLE, 'ambiguous launch; refuse close')
                        active = self.active_titles()
                        require(active in (set(), {TITLE}), 'unrelated title; refuse close')
                        if active:
                            self.send('close')
                            self.record['safety_close_sent'] = True
                            close_deadline = time.monotonic() + 8
                            while self.active_titles() == {TITLE} and time.monotonic() < close_deadline:
                                time.sleep(.5)
                        self.record['idle_after_safety_close'] = not self.active_titles()
                        self.history_read('after-safety-close')
                except Exception as error:
                    self.record['safety_close_error'] = repr(error)
            self.stop.set(); worker.join(timeout=2); sock.close()
            if self.fixture == 'rmsnorm' and not self.record.get('rmsnorm_numerical'):
                try:
                    self.collect_rmsnorm_dumps(partial=True)
                except Exception as error:
                    self.record['rmsnorm_raw_capture_error'] = repr(error)
            if self.fixture == 'q4' and not self.record.get('q4_numerical'):
                try:
                    self.collect_q4_dumps(partial=True)
                except Exception as error:
                    self.record['q4_raw_capture_error'] = repr(error)
            if self.fixture == 'weighted_qkv' and not self.record.get('weighted_qkv_numerical'):
                try:
                    self.collect_weighted_qkv_dumps(partial=True)
                except Exception as error:
                    self.record['weighted_qkv_raw_capture_error']=repr(error)
            if self.prefix_phase() and not self.record.get('prefix_numerical'):
                try:self.collect_prefix_dumps(partial=True)
                except Exception as error:self.record['prefix_raw_capture_error']=repr(error)
            (self.out / 'klog.log').write_bytes(self.klog)
            if not (self.out / 'application.log').exists():
                try:
                    (self.out / 'application.log').write_bytes(self.cleanup_fetch(self.log,self.application_log_limit()) if self.model_phase() else self.fetch(self.log,self.application_log_limit()))
                except Exception as error:
                    self.record['application_capture_error'] = repr(error)
            require(not worker.is_alive(), 'klog collector did not stop')
            require(not self.capture_errors and not native_fault(self.klog[launch_offset:]), 'klog failed or native fault during postchecks')
        self.record['natural_stop'] = True

    def run(self):
        try:
            if self.model_phase(): self.phase_remaining()
            with LOCK.open('x') as stream:
                stream.write(self.token)
            self.owned = True
            self.health()
            self.ftp = ftplib.FTP()
            self.ftp.connect(HOST, 2121, timeout=8)
            self.ftp.login('anonymous', 'codex')
            if self.model_phase():self.phase_remaining();self.ftp=PhaseFTP(self.ftp,self.phase_remaining)
            require(self.ftp.sendcmd('SELF') == '226 SELF transfer mode disabled', 'raw SELF transfer mode unavailable')
            getattr(self, self.phase)()
            if self.prefix_phase():self.final_prefix_authentication()
        except Exception as error:
            self.record['accepted'] = False
            self.record['error'] = repr(error)
        finally:
            if self.ftp:
                try:self.ftp.close()
                except Exception as error:
                    self.record['ftp_close_error']=repr(error);self.record['accepted']=False
            if self.owned:
                try:
                    self.record['healthy_after'] = self.health()
                except Exception as error:
                    self.record['health_error'] = repr(error)
                    self.record['accepted'] = False
                try:
                    require(LOCK.read_text() == self.token, 'owned lock changed')
                    LOCK.unlink()
                    self.record['lock_released'] = True
                except Exception as error:
                    self.record['lock_error'] = repr(error)
                    self.record['accepted'] = False
            (self.out / 'receipt.json').write_text(json.dumps(self.record, indent=2) + '\n')
        print(json.dumps(self.record))
        return 0 if self.record['accepted'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('deploy', 'execute'))
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(Control(args.candidate.resolve(), args.phase).run())
