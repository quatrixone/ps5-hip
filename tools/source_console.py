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

ROOT = Path(__file__).resolve().parents[1]
HOST = '192.168.4.40'
TITLE = 'PPSA99014'
REMOTE = '/data/homebrew/PPSA99014.ffpkg'
MOUNT = '/mnt/shadowmnt/PPSA99014_b6d9b2e4'
LOG = '/data/ps5-hip/results/ps5-hip-public-bootstrap.log'
HISTORY = '/data/shadowmount/debug.log'
LOCK = Path('/mnt/c/Users/Denis/Documents/PS5/lock.txt')

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


class Control:
    def __init__(self, candidate, phase):
        self.candidate, self.phase = candidate, phase
        self.manifest_sha = sha((candidate / 'manifest.json').read_bytes())
        self.manifest = json.loads((candidate / 'manifest.json').read_text())
        review = json.loads((candidate / 'review.json').read_text())
        require(review.get('approved') is True and review.get('manifest_sha256') == sha((candidate / 'manifest.json').read_bytes()),
                'independent review does not approve this manifest')
        require(self.manifest['host'] == HOST and self.manifest['title_id'] == TITLE and
                self.manifest['results_dir'] == '/data/ps5-hip/results', 'wrong candidate environment')
        if phase == 'execute':
            require(review.get('deployment_receipt_sha256') == sha((candidate / 'deploy/receipt.json').read_bytes()),
                    'deployment receipt has not been independently reviewed')
        self.verify()
        self.out = candidate / phase
        self.out.mkdir(exist_ok=False)
        self.record = {'phase': phase, 'host': HOST, 'title_id': TITLE, 'accepted': False, 'launched': False,
                       'manifest_sha256': self.manifest_sha,
                       'package_sha256': sha((ROOT / self.manifest['package']).read_bytes())}
        self.token = 'source-hip-' + str(uuid.uuid4())
        self.owned = False
        self.ftp = None
        self.history = None
        self.stop = threading.Event()
        self.klog = bytearray()
        self.capture_errors = []

    def verify(self):
        require(sha((self.candidate / 'manifest.json').read_bytes()) == self.manifest_sha, 'manifest changed')
        for name, digest in self.manifest['files'].items():
            path = ROOT / name
            require(path.resolve().is_relative_to(ROOT) and sha(path.read_bytes()) == digest, 'frozen input changed: ' + name)

    def owned_lock(self):
        require(self.owned and LOCK.read_text() == self.token, 'console lock changed')

    def health(self):
        for port in (2121, 3232, 9021):
            with socket.create_connection((HOST, port), timeout=3):
                pass
        return True

    def fetch(self, path, bound=8388608):
        data = bytearray()
        deadline = time.monotonic() + 30
        def append(block):
            require(len(data) + len(block) <= bound and time.monotonic() < deadline, 'FTP read bound')
            data.extend(block)
        self.ftp.retrbinary('RETR ' + path, append, blocksize=65536)
        return bytes(data)

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

    def send(self, name, bound=262144):
        self.verify()
        self.owned_lock()
        payload = (ROOT / self.manifest[name]).read_bytes()
        data = bytearray()
        with (self.out / (name + '-consumed')).open('x') as stream:
            stream.write(self.token)
        deadline = time.monotonic() + 10
        try:
            with socket.create_connection((HOST, 9021), timeout=3) as sock:
                sock.settimeout(8)
                sock.sendall(payload)
                while True:
                    require(time.monotonic() < deadline, 'controller deadline')
                    block = sock.recv(min(65536, bound + 1 - len(data)))
                    if not block:
                        break
                    data.extend(block)
                    require(len(data) <= bound, 'controller response bound')
        finally:
            (self.out / (name + '.log')).write_bytes(data)
        return bytes(data)

    def installed(self):
        app = ROOT / self.manifest['app']
        image = (ROOT / self.manifest['package']).read_bytes()
        require(self.fetch(REMOTE, len(image)) == image, 'installed image mismatch')
        require(self.fetch('/user/app/' + TITLE + '/mount.lnk', 1024) == MOUNT.encode(), 'mount link changed')
        require(self.fetch('/user/appmeta/' + TITLE + '/param.json', 65536) ==
                (app / 'sce_sys/param.json').read_bytes(), 'registered metadata mismatch')
        for name in self.manifest['package_files']:
            expected = (app / name).read_bytes()
            require(self.fetch(MOUNT + '/' + name, len(expected)) == expected, 'mounted bytes mismatch: ' + name)

    def fresh_log(self):
        require(not any(row.endswith(' ' + Path(LOG).name) for row in self.listing(str(Path(LOG).parent))), 'result log already exists')

    def deploy(self):
        self.idle()
        snapshot(self.send('snapshot'))
        baseline_log = self.manifest.get('baseline_log_sha256')
        if baseline_log:
            previous_log = self.fetch(LOG, 65536)
            require(sha(previous_log) == baseline_log, 'baseline log changed')
            (self.out / 'previous-application.log').write_bytes(previous_log)
            log_backup = LOG + '.' + baseline_log + '.preserved'
            require(not any(row.endswith(' ' + Path(log_backup).name) for row in self.listing(str(Path(LOG).parent))),
                    'log backup already exists')
        else:
            self.fresh_log()
        old_sha = self.manifest['baseline_image_sha256']
        require(re.fullmatch('[0-9a-f]{64}', old_sha), 'invalid baseline image digest')
        old = self.fetch(REMOTE, 16777216)
        require(sha(old) == old_sha, 'historical baseline differs; replan')
        (self.out / 'old.ffpkg').write_bytes(old)
        backup = REMOTE + '.source-rebuild-' + old_sha + '.preserved'
        temporary = '/data/homebrew/.PPSA99014-source-' + self.token + '.upload'
        entries = self.listing('/data/homebrew')
        require(not any(row.endswith(' ' + Path(backup).name) for row in entries), 'backup already exists')
        image = (ROOT / self.manifest['package']).read_bytes()
        self.idle(); self.verify()
        self.ftp.storbinary('STOR ' + temporary, io.BytesIO(image))
        require(self.fetch(temporary, len(image)) == image, 'temporary upload mismatch')
        self.idle()
        if baseline_log:
            require(self.fetch(LOG, 65536) == previous_log, 'baseline log changed before preservation')
            self.ftp.rename(LOG, log_backup)
            require(self.fetch(log_backup, 65536) == previous_log, 'preserved log mismatch')
            self.record['log_backup'] = log_backup
        self.fresh_log()
        start = self.history_read('before-preserve')
        require(self.fetch(REMOTE, len(old)) == old, 'baseline changed')
        self.record.update(backup=backup, temporary=temporary)
        self.ftp.rename(REMOTE, backup)
        require(self.fetch(backup, len(old)) == old, 'backup mismatch')
        deadline = time.monotonic() + 60
        while True:
            self.idle()
            current = self.history_read('release')
            if re.search(r'\[IMG\]\[LVD\] unmount complete: source=' + re.escape(REMOTE) + ' mount=' + re.escape(MOUNT) + r' unit=\d+', current[len(start):].decode()):
                break
            require(time.monotonic() < deadline, 'old image release timed out')
            time.sleep(3)
        self.idle(); self.verify()
        self.ftp.rename(temporary, REMOTE)
        self.record['promoted'] = True
        # Give ShadowMount and storage the full runbook settling interval.
        for _ in range(24):
            time.sleep(5)
            self.idle()
        self.installed()
        require(self.fetch(backup, len(old)) == old, 'preserved image changed')
        if baseline_log:
            require(self.fetch(log_backup, 65536) == previous_log, 'preserved log changed')
        self.fresh_log()
        snapshot(self.send('post_snapshot'))
        self.record['accepted'] = True

    def execute(self):
        deployment = json.loads((self.candidate / 'deploy/receipt.json').read_text())
        require(deployment.get('accepted') is True and deployment.get('healthy_after') is True and
                deployment.get('lock_released') is True and deployment.get('phase') == 'deploy' and
                deployment.get('manifest_sha256') == self.manifest_sha and
                deployment.get('package_sha256') == self.record['package_sha256'], 'fresh deployment not complete')
        self.idle(); self.installed(); self.fresh_log()
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
            self.idle(); self.fresh_log()
            require(not self.capture_errors and worker.is_alive(), 'klog unavailable before launch')
            launch_offset = len(self.klog)
            self.record['klog_launch_offset'] = launch_offset
            self.record['launched'] = True
            self.send('launch')  # This pinned controller prints no result; lifecycle is authoritative.
            deadline = time.monotonic() + 40
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
            application = self.fetch(LOG, 65536)
            (self.out / 'application.log').write_bytes(application)
            from source_public_hip_log import check_log
            self.record['application'] = check_log(application)
            # Snapshot uses another marker name so pre/post are each one-use.
            after = snapshot(self.send('post_snapshot'))
            require(before == after, 'matching title mounts changed')
            self.installed()
            self.record['accepted'] = True
        finally:
            if self.record['launched'] and not self.record.get('ordered_stop'):
                try:
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
            (self.out / 'klog.log').write_bytes(self.klog)
            if not (self.out / 'application.log').exists():
                try:
                    (self.out / 'application.log').write_bytes(self.fetch(LOG, 65536))
                except Exception as error:
                    self.record['application_capture_error'] = repr(error)
            require(not worker.is_alive(), 'klog collector did not stop')
            require(not self.capture_errors and not native_fault(self.klog[launch_offset:]), 'klog failed or native fault during postchecks')
        self.record['natural_stop'] = True

    def run(self):
        try:
            with LOCK.open('x') as stream:
                stream.write(self.token)
            self.owned = True
            self.health()
            self.ftp = ftplib.FTP()
            self.ftp.connect(HOST, 2121, timeout=8)
            self.ftp.login('anonymous', 'codex')
            require(self.ftp.sendcmd('SELF') == '226 SELF transfer mode disabled', 'raw SELF transfer mode unavailable')
            getattr(self, self.phase)()
        except Exception as error:
            self.record['accepted'] = False
            self.record['error'] = repr(error)
        finally:
            if self.ftp:
                self.ftp.close()
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
