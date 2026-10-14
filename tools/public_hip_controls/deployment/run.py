"""Preserve the GPU image and log, then stage one public HIP package."""
import datetime,ftplib,hashlib,importlib.util,io,json,re,subprocess,sys,time,uuid
from pathlib import Path
if not __debug__:raise RuntimeError('optimized Python is forbidden')
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
HOST='192.168.4.40';TITLE='PPSA99014';MOUNT='/mnt/shadowmnt/PPSA99014_b6d9b2e4'
CANDIDATE=ROOT/'.local/g5-hip-native-link'
STAGE=CANDIDATE/'dist/PPSA99014/gfx1013-public-bootstrap-19107f507822'
PRIOR_APP=ROOT/'.local/g5-clr-gpu-startup/dist/PPSA99014/gfx1013-clr-gpu-b21b9c51dc42/app'
SOURCE='ea37055286903f680e43b049d70730743ffa2358'
IMAGE='836c10c6e66eb3d98245e23f96394ac8a7f51c34c750a4b18b35fc22210975b8'
OLD_IMAGE = '84e52f366b9c7e598d81c3d37197ff8e71d548ebe6cfe933176639066abcc8da'
LOG_SHA = '5c91d896f50ec94b7092b90ed30faa84e9a7819c5850f3772edc0e'
FINAL='/data/homebrew/'+TITLE+'.ffpkg'
OLD_LOG='/data/ps5-hip/results/ps5-hip-clr-gpu.log'
PRESERVED_BACKUPS = {'/data/homebrew/PPSA99014.ffpkg.atomic-current.sha256-14e07323b96c081c040ab65babc412da985fe36b10fd73dfe27ba4293f178498.preserved': {'bytes': 3637248, 'sha256': '14e07323b96c081c040ab65babc412da985fe36b10fd73dfe27ba4293f178498'}, '/data/ps5-hip/results/ps5-hip-gfx1013-atomic_suite.log.atomic-current-PID231-appe018.sha256-bccfb21af8a5de24f2bd1e88b20a941fb96bd629e5bf877ebe03c55b50de310c': {'bytes': 1904, 'sha256': 'bccfb21af8a5de24f2bd1e88b20a941fb96bd629e5bf877ebe03c55b50de310c'}, '/data/homebrew/PPSA99014.ffpkg.math-failed.sha256-54f5703f60d13c384725ed50cb3bae87f1286da518f50c024c8d1995054a7364.preserved': {'bytes': 3637248, 'sha256': '54f5703f60d13c384725ed50cb3bae87f1286da518f50c024c8d1995054a7364'}, '/data/ps5-hip/results/ps5-hip-gfx1013-fp32_math.log.math-failed-PID240-app18.sha256-5a90708f1bc3fbdd7bccc88cfede65f9bdea166ba6182ee1f045c29157329a5f': {'bytes': 373, 'sha256': '5a90708f1bc3fbdd7bccc88cfede65f9bdea166ba6182ee1f045c29157329a5f'}, '/data/homebrew/PPSA99014.ffpkg.hip-generic.sha256-c8de4c7a59d7f7d73b706dc1a367e1c278993312b9ee203ca9255c123b98a8b6.preserved': {'bytes': 3670016, 'sha256': 'c8de4c7a59d7f7d73b706dc1a367e1c278993312b9ee203ca9255c123b98a8b6'}, '/data/ps5-hip/results/ps5-hip-gfx1013-hip_generic.log.hip-generic-PID248-app2018.sha256-49118f7351f10dfee2a3cbd637bbe52a34dc72b4aee739fe0ae5977aa2a28e14': {'bytes': 1842, 'sha256': '49118f7351f10dfee2a3cbd637bbe52a34dc72b4aee739fe0ae5977aa2a28e14'}, '/data/homebrew/PPSA99014.ffpkg.math-exp-corrected.sha256-28581ca2a1ca2cdf94ebe902923195efffcf98f85128a013edcfc257fb307d82.preserved': {'bytes': 3637248, 'sha256': '28581ca2a1ca2cdf94ebe902923195efffcf98f85128a013edcfc257fb307d82'}, '/data/ps5-hip/results/ps5-hip-gfx1013-fp32_math.log.math-exp-corrected-PID257-app4018.sha256-6db4b87ea5fbf3afb22f99c7cc70ddce0b5ff4d8ce3572cd52efea743b996c33': {'bytes': 625, 'sha256': '6db4b87ea5fbf3afb22f99c7cc70ddce0b5ff4d8ce3572cd52efea743b996c33'}, '/data/homebrew/PPSA99014.ffpkg.hip-matmul.sha256-cb1b0c39d36a73ecdabd05ef3185ef5c17ba662c7ac361e69bab235b809d532e.preserved': {'bytes': 3670016, 'sha256': 'cb1b0c39d36a73ecdabd05ef3185ef5c17ba662c7ac361e69bab235b809d532e'}, '/data/ps5-hip/results/ps5-hip-gfx1013-hip_matmul.log.hip-matmul-PID266-app6018.sha256-d183ca0d03804c20a8d02005b64217f12caca886db68299d63a9ef632644190f': {'bytes': 2049, 'sha256': 'd183ca0d03804c20a8d02005b64217f12caca886db68299d63a9ef632644190f'}, '/data/homebrew/PPSA99014.ffpkg.hipify.sha256-ead66563443612b713afe668dfe76dd3b974fb8f5c9575dacb23b799f52f32cb.preserved': {'bytes': 3670016, 'sha256': 'ead66563443612b713afe668dfe76dd3b974fb8f5c9575dacb23b799f52f32cb'}, '/data/ps5-hip/results/ps5-hip-gfx1013-hipify.log.hipify-PID275-app8018.sha256-fc305204d5b2f1b36e5dd3a1c15b466c57b2ccffc9419318acba957e3d8fa5a4': {'bytes': 1253, 'sha256': 'fc305204d5b2f1b36e5dd3a1c15b466c57b2ccffc9419318acba957e3d8fa5a4'}, '/data/homebrew/PPSA99014.ffpkg.cpu-host-threads.sha256-28f76350b682a2c47d20dc05d459d0d3cdf3793b1b70488267e10986e88f826f.preserved': {'bytes': 3473408, 'sha256': '28f76350b682a2c47d20dc05d459d0d3cdf3793b1b70488267e10986e88f826f'}, '/data/ps5-hip/results/ps5-hip-host_threads.log.cpu-host-threads-PID300-appe018.sha256-ffc24ede0a06643676ad86655a2df2046d7a4161fe569de98e1a9bef28ca3bc2': {'bytes': 663, 'sha256': 'ffc24ede0a06643676ad86655a2df2046d7a4161fe569de98e1a9bef28ca3bc2'}, '/data/homebrew/PPSA99014.ffpkg.threaded-hipify.sha256-635ef1c675174a3e950763212e05084cc6f9c3dc5f552fc3a12f5e378ca2bcf6.preserved': {'bytes': 3670016, 'sha256': '635ef1c675174a3e950763212e05084cc6f9c3dc5f552fc3a12f5e378ca2bcf6'}, '/data/ps5-hip/results/ps5-hip-gfx1013-hipify.log.threaded-hipify-PID309-app18.sha256-fc305204d5b2f1b36e5dd3a1c15b466c57b2ccffc9419318acba957e3d8fa5a4': {'bytes': 1253, 'sha256': 'fc305204d5b2f1b36e5dd3a1c15b466c57b2ccffc9419318acba957e3d8fa5a4'}, '/data/homebrew/PPSA99014.ffpkg.workers-main.sha256-4a4d969be9cd8187eacc0081d328977f7ac1b0f95b5d6f919a0b21878771265f.preserved': {'bytes': 3670016, 'sha256': '4a4d969be9cd8187eacc0081d328977f7ac1b0f95b5d6f919a0b21878771265f'}, '/data/ps5-hip/results/ps5-hip-gfx1013-hip_workers_main.log.workers-main-PID318-app2018.sha256-7eff115bc3050e3833a71055ff3a928df06e4e78796d17ee5f1af8f15b3a426b': {'bytes': 2419, 'sha256': '7eff115bc3050e3833a71055ff3a928df06e4e78796d17ee5f1af8f15b3a426b'}, '/data/homebrew/PPSA99014.ffpkg.workers-init.sha256-330b3a7205e076b5b9f873627a7084d1e5c27939e5b4dbf8787bcbd7c6a03cb4.preserved': {'bytes': 3670016, 'sha256': '330b3a7205e076b5b9f873627a7084d1e5c27939e5b4dbf8787bcbd7c6a03cb4'}, '/data/ps5-hip/results/ps5-hip-gfx1013-hip_workers_init.log.workers-init-PID327-app4018.sha256-433c0f6cf2733994b0a54f2e714fc103e173cf77fb08acbbc54c4238a225e84e': {'bytes': 2206, 'sha256': '433c0f6cf2733994b0a54f2e714fc103e173cf77fb08acbbc54c4238a225e84e'}, '/data/homebrew/PPSA99014.ffpkg.contention.sha256-57c92907f9f01ef58b58bfab99849a9390db3a4cdbd3816fc9a3d1962cee932c.preserved': {'bytes': 3670016, 'sha256': '57c92907f9f01ef58b58bfab99849a9390db3a4cdbd3816fc9a3d1962cee932c'}, '/data/ps5-hip/results/ps5-hip-gfx1013-hip_contend_main.log.contention-PID336-app6018.sha256-460938f09a63357c222d2f609673b02f34911d39cde4235b37a3fd6eed15164c': {'bytes': 1901, 'sha256': '460938f09a63357c222d2f609673b02f34911d39cde4235b37a3fd6eed15164c'}, '/data/homebrew/PPSA99014.ffpkg.memory-fill.sha256-cca66888e3233a84c8ef72a1f0c0dd7eab3a9f66ad51b9a057682a94fee83caf.preserved': {'bytes': 3670016, 'sha256': 'cca66888e3233a84c8ef72a1f0c0dd7eab3a9f66ad51b9a057682a94fee83caf'}, '/data/ps5-hip/results/ps5-hip-gfx1013-hip_memory_fill.log.memory-fill-PID345-app8018.sha256-217bb46f3303c01cada00fa434e00a83724650ef9fb5d573bd6929c3cbe74105': {'bytes': 1843, 'sha256': '217bb46f3303c01cada00fa434e00a83724650ef9fb5d573bd6929c3cbe74105'}, '/data/homebrew/PPSA99014.ffpkg.device-copy.sha256-45311e46b536f7a65fb519ba45f44e1fbbd96492573df7483aa96f3e1d3a3134.preserved': {'bytes': 3670016, 'sha256': '45311e46b536f7a65fb519ba45f44e1fbbd96492573df7483aa96f3e1d3a3134'}, '/data/ps5-hip/results/ps5-hip-gfx1013-hip_device_copy.log.device-copy-PID354-appa018.sha256-10cdda626fdad484b1b7a5ef718d2156f7b61296b65fddb9f7b75d7b80515a94': {'bytes': 3602, 'sha256': '10cdda626fdad484b1b7a5ef718d2156f7b61296b65fddb9f7b75d7b80515a94'}, '/data/homebrew/PPSA99014.ffpkg.reduction.sha256-6b8487e6b4c66bdd0a8ae5aa79b5fa4627a5f7fd296ed6e89d96216c1340c2ff.preserved': {'bytes': 3702784, 'sha256': '6b8487e6b4c66bdd0a8ae5aa79b5fa4627a5f7fd296ed6e89d96216c1340c2ff'}, '/data/ps5-hip/results/ps5-hip-gfx1013-hip_reduction.log.reduction-PID363-appc018.sha256-8efabb57772a1524f55ab0de56b3bf847d9a6a7ce789a455b76c2717f5eed97a': {'bytes': 1448, 'sha256': '8efabb57772a1524f55ab0de56b3bf847d9a6a7ce789a455b76c2717f5eed97a'}, '/data/homebrew/PPSA99014.ffpkg.multi-query.sha256-5266e6b0c1c85029613e698639000ff53d6f889005d46d47800ec08063269151.preserved': {'bytes': 3702784, 'sha256': '5266e6b0c1c85029613e698639000ff53d6f889005d46d47800ec08063269151'}, '/data/ps5-hip/results/ps5-hip-gfx1013-hip_multi_query.log.multi-query-PID372-appe018.sha256-b055f5d3b5abc82c14869b9ff1e2b36d4983cfeeeced004537a0f13eb0bb3842': {'bytes': 1991, 'sha256': 'b055f5d3b5abc82c14869b9ff1e2b36d4983cfeeeced004537a0f13eb0bb3842'}, '/data/homebrew/PPSA99014.ffpkg.cpu-clr-unlaunched.sha256-e78490b0e0d58757aecbd0abc67157c80ba7db5d7ff119477e40f305c9b951ba.preserved': {'bytes': 5144576, 'sha256': 'e78490b0e0d58757aecbd0abc67157c80ba7db5d7ff119477e40f305c9b951ba'}, '/data/homebrew/PPSA99014.ffpkg.cpu-clr-failed-startup.sha256-8b2a8c8b1ce2e2d3713f88283b8402482aeebaf2210c555ed86cb1f16891fb91.preserved': {'bytes': 5144576, 'sha256': '8b2a8c8b1ce2e2d3713f88283b8402482aeebaf2210c555ed86cb1f16891fb91'}, '/data/homebrew/PPSA99014.ffpkg.cpu-clr-os-init-diagnostic.sha256-cb9a134795672010b128dfeab17cdfd1fd5095d5ccbf2f26be1acc5c620c4a77.preserved': {'bytes': 5144576, 'sha256': 'cb9a134795672010b128dfeab17cdfd1fd5095d5ccbf2f26be1acc5c620c4a77'}, '/data/homebrew/PPSA99014.ffpkg.cpu-clr-path-query-failure.sha256-f63a8f3922d69e0689b6a3d99e57fe3980fb91dc5a7fbc6dbd53d73716e42e66.preserved': {'sha256': 'f63a8f3922d69e0689b6a3d99e57fe3980fb91dc5a7fbc6dbd53d73716e42e66', 'bytes': 5144576}}
PHASES=('upload','preserve-image','preserve-log','promote')
RECOVERY_PHASES=('quarantine','restore-image','restore-log')

def sha(data):return hashlib.sha256(data).hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

_VERIFIED_MANIFEST_SHA=None

def verify_direct():
    raw=(HERE/'manifest.json').read_bytes()
    require(_VERIFIED_MANIFEST_SHA is not None and sha(raw)==_VERIFIED_MANIFEST_SHA,'manifest changed during operation')
    manifest=json.loads(raw)
    for name,digest in manifest['direct_sha256'].items():require(sha((ROOT/name).read_bytes())==digest,'direct input changed: '+name)
    require(sha((STAGE/'PPSA99014.ffpkg').read_bytes())==IMAGE,'candidate image changed')

def verify_closure():
    global _VERIFIED_MANIFEST_SHA
    raw=(HERE/'manifest.json').read_bytes()
    require(_VERIFIED_MANIFEST_SHA is None or sha(raw)==_VERIFIED_MANIFEST_SHA,'manifest changed between full verifications')
    _VERIFIED_MANIFEST_SHA=sha(raw)
    manifest=json.loads(raw)
    for name,digest in manifest['files_sha256'].items():require(sha((ROOT/name).read_bytes())==digest,'frozen input changed: '+name)
    verify_direct()
    require(subprocess.run(['git','merge-base','--is-ancestor',SOURCE,'HEAD'],cwd=CANDIDATE).returncode==0,'candidate source is not an ancestor')
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=CANDIDATE,text=True),'candidate dirty')
    package=json.loads((STAGE/'package-receipt.json').read_text());m=json.loads((STAGE/'manifest.json').read_text())
    require(package['manifest_sha256']==sha((STAGE/'manifest.json').read_bytes()) and m['source_commit']==SOURCE and m['target']=='gfx1013' and m['kernel']=='public_hip_bootstrap','package identity')
    require(package['image']=={'file':'PPSA99014.ffpkg','bytes':9437184,'sha256':IMAGE},'package image')
    actual={p.relative_to(STAGE/'app').as_posix():{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in (STAGE/'app').rglob('*') if p.is_file()}
    require(actual==m['files'] and not any(p.is_symlink() for p in (STAGE/'app').rglob('*')),'staged file inventory')
    control=json.loads((ROOT/'build/controllers/PPSA99014/receipt.json').read_text())
    require(control==json.loads((ROOT/'evidence/offline-controllers.json').read_text())['controller_build'] and control['title_id']==TITLE and set(control['artifacts'])=={'launch.elf','close.elf'},'controller authority')
    for name,expected in control['artifacts'].items():
        p=ROOT/'build/controllers'/TITLE/name;require({'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}==expected,'controller bytes')
    return {'local_artifacts_verified':True,'hardware_gate_passed':False,'title_id':TITLE,'target':'gfx1013','kernel':'public_hip_bootstrap','shutdown_mode':'shell-request-probe','source_commit':SOURCE,'expected_log':m['expected_log'],'image':package['image'],'image_bytes':package['image']['bytes'],'controllers':control['artifacts']}

def verify_local():
    case=verify_closure()
    helper=load('deploy_helpers',ROOT/'results/deploy-private-buffer-milestone-v2.py')
    guards=load('deploy_guards',ROOT/'results/teardown-console-control-v3-initial-focus/run.py')
    return helper,guards,case

def consume(phase,out,record):
    require(phase in PHASES+RECOVERY_PHASES,'unknown phase')
    with (HERE/(phase+'-consumed.json')).open('x') as stream:json.dump({'phase':phase,'output':str(out),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()},stream)
    record.setdefault('attempted',[]).append(phase)

def unmounted(delta):
    pattern=r'^.*\[IMG\]\[LVD\] unmount complete: source='+re.escape(FINAL)+r' mount='+re.escape(MOUNT)+r' unit=\d+\s*$'
    rows=list(re.finditer(pattern,delta,re.M))
    require(len(rows)<=1,'ambiguous old image unmount')
    if re.search(r'\b(?:failed|failure|panic|fatal|error)\b',delta,re.I):raise RuntimeError('failure during old image release')
    return len(rows)==1

def raw_self_mode(ftp,helper,out):
    # SELF has no argument and toggles exactly one connection's transfer state.
    response=None
    try:
        with helper.bounded(ftp,time.monotonic()+8):response=ftp.sendcmd('SELF')
    finally:(out/'self-mode-response.json').write_text(json.dumps({'command':'SELF','response':response},indent=2)+'\n')
    require(response=='226 SELF transfer mode disabled','raw SELF transfer mode not established; do not toggle again')

def fetch_self_evidence(ftp,path,expected_bytes,destination,helper):
    # Same absolute bounded FTP sequence, retaining at most the exact bound+1
    # on failure. The extra byte diagnoses oversize; it is never accepted.
    data=bytearray();deadline=time.monotonic()+10
    try:
        with helper.bounded(ftp,deadline):
            ftp.voidcmd('TYPE I')
            with ftp.transfercmd('RETR '+path) as transfer:
                while True:
                    transfer.settimeout(helper.remaining(deadline))
                    block=transfer.recv(min(65536,expected_bytes-len(data)+1))
                    if not block:break
                    data.extend(block)
                    require(len(data)<=expected_bytes,'mounted SELF response exceeded exact bound')
            ftp.sock.settimeout(helper.remaining(deadline));ftp.voidresp()
        return bytes(data)
    finally:destination.write_bytes(data)


def deploy(out,record,deps):
    helper,guards,case=deps;locked=False;owned=[];token='public-hip-bootstrap-deployment-v1|'+str(uuid.uuid4());ftp=None;history=None;broken=False
    old=log=payload=release_start=None
    backup=FINAL+'.clr-gpu-success.sha256-'+OLD_IMAGE+'.preserved'
    log_backup=OLD_LOG+'.clr-gpu-success-PID407-appa018.sha256-'+LOG_SHA
    temporary='/data/homebrew/.PPSA99014.public-hip-'+IMAGE+'.upload'
    record['preservation']={'package':backup,'package_sha256':OLD_IMAGE,'gpu_result':log_backup,'gpu_result_sha256':LOG_SHA}
    record['temporary']=temporary
    def fetch(path,limit=4194304,deadline=None,command='RETR'):
        nonlocal broken
        require(not broken,'FTP unavailable after uncertain I/O')
        try:return helper.fetch(ftp,path,limit,deadline,command)
        except Exception:broken=True;raise
    def listing(path,label,deadline=None):
        raw=fetch(path,65536,deadline,command='LIST');(out/(label+'.list')).write_bytes(raw);return helper.parse_listing(raw)
    def idle(label,deadline=None):
        nonlocal history
        raw=fetch('/data/shadowmount/debug.log',4194304,deadline);(out/(label+'.log')).write_bytes(raw)
        require(history is None or raw.startswith(history),'ShadowMount history reset')
        require(guards.idle(raw.decode('utf-8','strict')),'not idle FW12.70')
        history=raw
        require(owned==guards.LOCKS and all(p.read_text()==token for p in owned),'owned locks changed')
        return raw
    def mutate(phase,method,*args,seconds=10):
        nonlocal broken
        idle('before-'+phase);consume(phase,out,record)
        try:
            with helper.bounded(ftp,time.monotonic()+seconds):method(*args)
        except Exception:broken=True;raise
    def verify_prior_backups(label):
        actual={}
        for path,expected in PRESERVED_BACKUPS.items():
            data=fetch(path,expected['bytes'],time.monotonic()+30)
            require(len(data)==expected['bytes'] and sha(data)==expected['sha256'],'existing prior backup changed')
            actual[path]={'bytes':len(data),'sha256':sha(data)}
        record.setdefault('prior_backups_unchanged',{})[label]=actual
    def preconditions(label):
        entries=listing('/data/homebrew',label+'-homebrew');logs=listing('/data/ps5-hip/results',label+'-results')
        require(TITLE not in entries and TITLE+'.ffpfsc' not in entries,'conflicting install')
        require(entries.get(TITLE+'.ffpkg')=={'type':'-','bytes':record['baseline_image_bytes']},'baseline image entry changed')
        require(Path(backup).name not in entries and Path(temporary).name not in entries,'backup/upload conflict')
        require(logs.get(Path(OLD_LOG).name)=={'type':'-','bytes':870} and Path(log_backup).name not in logs,'baseline GPU log entry/conflict')
        require(Path(case['expected_log']).name not in logs,'public HIP result already exists')
        require(fetch('/user/appmeta/'+TITLE+'/param.json',65536)==(STAGE/'app/sce_sys/param.json').read_bytes(),'installed metadata mismatch')
        require(fetch('/user/app/'+TITLE+'/mount.lnk',1024)==MOUNT.encode(),'installed mount link mismatch')
        old=fetch(FINAL,record['baseline_image_bytes']);require(len(old)==record['baseline_image_bytes'] and sha(old)==OLD_IMAGE,'baseline image digest mismatch')
        log=fetch(OLD_LOG,870);require(len(log)==870 and sha(log)==LOG_SHA,'baseline GPU log digest mismatch')
        return old,log
    def await_release(start,label):
        deadline=time.monotonic()+45;poll=0
        while True:
            helper.remaining(deadline);poll+=1;current=idle(label+'-%03d'%poll,deadline)
            require(TITLE+'.ffpkg' not in listing('/data/homebrew',label+'-%03d-homebrew'%poll,deadline),'final path reappeared before release')
            if unmounted(current[len(start):].decode('utf-8','strict')):
                helper.remaining(deadline);return
            time.sleep(min(3,helper.remaining(deadline)))
    def await_registration(start,label):
        deadline=time.monotonic()+45;poll=0
        while True:
            helper.remaining(deadline);poll+=1;current=idle(label+'-%03d'%poll,deadline)
            if helper.title_registered(current[len(start):].decode('utf-8','strict')):
                helper.remaining(deadline);return
            time.sleep(min(3,helper.remaining(deadline)))
    def recover():
        # No mutation after uncertain FTP I/O or lost ownership. Keep the backups for read-only diagnosis.
        require(locked and owned==guards.LOCKS and all(p.read_text()==token for p in owned),'rollback locks uncertain')
        require(not broken and old is not None and log is not None,'rollback transport or baseline uncertain')
        idle('rollback-before')
        entries=listing('/data/homebrew','rollback-homebrew');logs=listing('/data/ps5-hip/results','rollback-results')
        image_backup=Path(backup).name in entries
        log_preserved=Path(log_backup).name in logs
        require(not image_backup or fetch(backup,len(old))==old,'rollback old-image backup changed')
        require(not log_preserved or fetch(log_backup,len(log))==log,'rollback old-log backup changed')
        require(image_backup or fetch(FINAL,len(old))==old,'rollback old image unavailable')
        require(log_preserved or fetch(OLD_LOG,len(log))==log,'rollback old log unavailable')
        if image_backup:
            if TITLE+'.ffpkg' in entries:
                require(fetch(FINAL,len(payload))==payload,'rollback final image ambiguous')
                require(Path(temporary).name not in entries,'rollback quarantine path occupied')
                quarantine_start=history
                mutate('quarantine',ftp.rename,FINAL,temporary)
                require(fetch(temporary,len(payload))==payload,'rollback quarantined image mismatch')
                await_release(quarantine_start,'rollback-candidate-release')
            else:
                require(release_start is not None,'rollback release baseline unavailable')
                await_release(release_start,'rollback-old-release')
            require(TITLE+'.ffpkg' not in listing('/data/homebrew','rollback-pre-restore'),'rollback final path conflict')
            restoration_start=idle('rollback-before-restore')
            mutate('restore-image',ftp.rename,backup,FINAL)
            require(fetch(FINAL,len(old))==old,'rollback restored image mismatch')
        else:restoration_start=None
        if log_preserved:
            require(Path(OLD_LOG).name not in listing('/data/ps5-hip/results','rollback-pre-log'),'rollback log path conflict')
            mutate('restore-log',ftp.rename,log_backup,OLD_LOG)
        require(fetch(OLD_LOG,len(log))==log,'rollback restored log mismatch')
        if restoration_start is not None:await_registration(restoration_start,'rollback-registration')
        prior=PRIOR_APP
        for name in ('eboot.bin','sce_module/libc.prx','lapy.elf','results-dir.txt'):
            expected=(prior/name).read_bytes()
            if name in ('eboot.bin','sce_module/libc.prx'):
                actual=fetch_self_evidence(ftp,MOUNT+'/'+name,len(expected),
                                           out/('rollback-mounted-'+name.replace('/','_')),helper)
            else:actual=fetch(MOUNT+'/'+name,len(expected))
            require(actual==expected,'rollback mounted old bytes mismatch: '+name)
        require(fetch('/user/appmeta/'+TITLE+'/param.json',65536)==(prior/'sce_sys/param.json').read_bytes(),'rollback metadata mismatch')
        require(fetch('/user/app/'+TITLE+'/mount.lnk',1024)==MOUNT.encode(),'rollback mount link mismatch')
        verify_prior_backups('rollback');idle('rollback-after')
        record['rollback']={'status':'old-gpu-restored','image_sha256':OLD_IMAGE,'log_sha256':LOG_SHA,
                            'registered':True,'mounted_old_bytes_verified':True}
    try:
        guards.acquire(owned,token)
        require(owned==guards.LOCKS and all(p.read_text()==token for p in owned),'complete exclusive locks absent')
        locked=True;record['health_before']=guards.health()
        require(record['health_before']=={str(p):'reachable' for p in (2121,3232,9021)},'preflight services unhealthy')
        ftp=ftplib.FTP()
        with helper.bounded(ftp,time.monotonic()+10):ftp.connect(HOST,2121,timeout=5);ftp.login('anonymous','codex')
        idle('before')
        try:raw_self_mode(ftp,helper,out)
        except Exception:broken=True;raise
        record['self_transfer_mode']='disabled on this connection only'
        verify_prior_backups('before')
        old,log=preconditions('before');(out/'preserved-gpu.ffpkg').write_bytes(old);(out/'preserved-gpu.log').write_bytes(log)
        payload=(STAGE/(TITLE+'.ffpkg')).read_bytes();require(len(payload)==case['image_bytes'] and sha(payload)==IMAGE,'candidate changed before upload')
        mutate('upload',ftp.storbinary,'STOR '+temporary,io.BytesIO(payload),seconds=30);record['stage']='temporary-uploaded'
        require(fetch(temporary,len(payload),time.monotonic()+30)==payload,'temporary upload mismatch');record['stage']='temporary-verified'
        # Recheck all old bytes/conflicts, allowing only our verified temporary file.
        idle('before-preservation');verify_direct()
        entries=listing('/data/homebrew','recheck-homebrew');logs=listing('/data/ps5-hip/results','recheck-results')
        require(Path(backup).name not in entries and Path(log_backup).name not in logs,'preservation conflict')
        require(Path(case['expected_log']).name not in logs,'public HIP result appeared')
        require(fetch(FINAL,len(old))==old and fetch(OLD_LOG,len(log))==log,'baseline changed during upload')
        require(fetch('/user/appmeta/'+TITLE+'/param.json',65536)==(STAGE/'app/sce_sys/param.json').read_bytes() and fetch('/user/app/'+TITLE+'/mount.lnk',1024)==MOUNT.encode(),'installation changed during upload')
        release_start=history
        mutate('preserve-image',ftp.rename,FINAL,backup);record['stage']='old-image-preserved'
        require(fetch(backup,len(old))==old,'preserved image mismatch')
        mutate('preserve-log',ftp.rename,OLD_LOG,log_backup);record['stage']='old-log-preserved'
        require(fetch(log_backup,len(log))==log,'preserved GPU log mismatch')
        # A cached 1.6 image is keyed by path. Leave that path absent until passive cleanup completes.
        await_release(release_start,'release')
        record['stage']='old-registration-released'
        verify_direct();promotion_start=idle('before-promotion')
        require(fetch(backup,len(old))==old and fetch(log_backup,len(log))==log,'preserved artifacts changed')
        require(TITLE+'.ffpkg' not in listing('/data/homebrew','pre-promote-homebrew'),'final path conflict')
        mutate('promote',ftp.rename,temporary,FINAL);record['stage']='promoted'
        require(fetch(FINAL,len(payload),time.monotonic()+30)==payload,'promoted image mismatch')
        await_registration(promotion_start,'registration')
        mounted={}
        for name in ('eboot.bin','sce_module/libc.prx','lapy.elf','results-dir.txt'):
            expected=(STAGE/'app'/name).read_bytes()
            if name in ('eboot.bin','sce_module/libc.prx'):
                try:actual=fetch_self_evidence(ftp,MOUNT+'/'+name,len(expected),out/('mounted-'+name.replace('/','_')),helper)
                except Exception:broken=True;raise
            else:actual=fetch(MOUNT+'/'+name,len(expected))
            require(actual==expected,'mounted candidate bytes mismatch: '+name)
            mounted[name]={'bytes':len(actual),'sha256':sha(actual)}
        record['mounted_files']=mounted
        require(fetch('/user/appmeta/'+TITLE+'/param.json',65536)==(STAGE/'app/sce_sys/param.json').read_bytes(),'registered metadata mismatch')
        require(fetch('/user/app/'+TITLE+'/mount.lnk',1024)==MOUNT.encode(),'registered link mismatch')
        require(fetch(FINAL,len(payload),time.monotonic()+30)==payload and fetch(backup,len(old))==old and fetch(log_backup,len(log))==log,'final package/preservation mismatch')
        logs=listing('/data/ps5-hip/results','final-results');require(Path(OLD_LOG).name not in logs and Path(case['expected_log']).name not in logs,'fresh result absent check failed')
        verify_prior_backups('after')
        idle('after');record['stage']='registered';record['registered']=True
    except BaseException as error:
        record['error']=repr(error)
        if 'preserve-image' in record.get('attempted',[]):
            try:recover()
            except BaseException as recovery_error:
                record['rollback']={'status':'unverified-hold','reason':repr(recovery_error),
                                    'backups_must_remain':True,'manual_read_only_diagnosis':True}
    finally:
        still_locked=locked and owned==guards.LOCKS and all(p.exists() and p.read_text()==token for p in owned)
        if still_locked and ftp and not broken:
            try:idle('final-idle')
            except Exception as error:record['idle_error']=repr(error)
        else:record['idle_error']='final idle unavailable'
        if still_locked:
            try:
                record['health_after']=guards.health()
                require(record['health_after']=={str(p):'reachable' for p in (2121,3232,9021)},'postflight services unhealthy')
            except Exception as error:record['health_error']=repr(error)
        else:record['health_error']='postchecks skipped without complete owned locks'
        if ftp:
            try:ftp.close()
            except Exception as error:record['ftp_close_error']=repr(error)
        record['lock_errors']=guards.release(owned,token);record['owned_locks_released']=not record['lock_errors']

def main():
    deps=verify_local();require(not any((HERE/(p+'-consumed.json')).exists() for p in PHASES),'deployment already attempted')
    out=HERE/('deployment-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'));out.mkdir()
    baseline=ROOT/'.local/g5-clr-gpu-startup/dist/PPSA99014/gfx1013-clr-gpu-b21b9c51dc42/PPSA99014.ffpkg'
    record={'host':HOST,'title_id':TITLE,'kernel':'public_hip_bootstrap','source_commit':SOURCE,'image_sha256':IMAGE,'image':deps[2]['image'],'remote':FINAL,'stage':'local-verified','registered':False,'launched':False,'hardware_gate_passed':False,'attempted':[],'baseline_image_bytes':baseline.stat().st_size,'script_sha256':sha(Path(__file__).read_bytes()),'manifest_sha256':sha((HERE/'manifest.json').read_bytes())}
    try:deploy(out,record,deps)
    finally:
        try:
            verify_local();require(sha(Path(__file__).read_bytes())==record['script_sha256'] and sha((HERE/'manifest.json').read_bytes())==record['manifest_sha256'],'control changed')
        except Exception as error:record['local_final_error']=repr(error)
        record['accepted']=record['registered'] and record['owned_locks_released'] and not any(record.get(k) for k in ('error','idle_error','health_error','ftp_close_error','lock_errors','local_final_error'))
        record['raw_file_sha256']={p.name:sha(p.read_bytes()) for p in out.iterdir() if p.is_file()}
        (out/'deployment.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'output':str(out),'registered':record['registered'],'accepted':record['accepted'],'launched':False,'error':record.get('error')}))
    return 0 if record['accepted'] else 1
if __name__=='__main__':raise SystemExit(main())
