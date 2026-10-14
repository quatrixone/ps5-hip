"""INERT DRAFT: one frozen GPU ROCclr workload deployment only: no payload, launch, close or force-unmount."""
import datetime,ftplib,hashlib,importlib.util,io,json,re,subprocess,sys,time,uuid
from pathlib import Path
if not __debug__:raise RuntimeError('optimized Python is forbidden')
HERE=Path(__file__).resolve().parent;CANDIDATE=HERE.parents[2];ROOT=CANDIDATE.parent/'ps5-hip'
HOST='192.168.4.40';TITLE='PPSA99014';MOUNT='/mnt/shadowmnt/PPSA99014_b6d9b2e4'
STAGE=CANDIDATE/'dist/PPSA99014/gfx1013-clr-gemm-832962dca512'
SOURCE='b00dac10adcc1ba45d9e2407b9a41f3c6c513c69'
IMAGE='9c229e4221570716ec1cc2ee9bea08d41299bd6b433aad46a30ebf121f5777e4'
OLD_IMAGE = '84e52f366b9c7e598d81c3d37197ff8e71d548ebe6cfe933176639066abcc8da'
LOG_SHA = '5c91d896f50ec94b7092b90ed30faa84e9b69f3c84e9a7819c5850f3772edc0e'
FINAL='/data/homebrew/'+TITLE+'.ffpkg'
OLD_LOG='/data/ps5-hip/results/ps5-hip-clr-gpu.log'
PRESERVED_BACKUPS = json.loads((HERE/'prior-backups.json').read_text())
PHASES=('upload','preserve-image','preserve-log','promote')

def sha(data):return hashlib.sha256(data).hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

_VERIFIED_MANIFEST_SHA=None

def verify_direct():
    control=load('gemm_control_guard',CANDIDATE/'tools/clr_gemm_control_guard.py')
    control.verify_control(CANDIDATE,HERE)

def verify_closure():
    verify_direct()
    package=json.loads((STAGE/'package-receipt.json').read_text())
    m=json.loads((STAGE/'manifest.json').read_text())
    control=json.loads((ROOT/'build/controllers/PPSA99014/receipt.json').read_text())
    require(control==json.loads((ROOT/'evidence/offline-controllers.json').read_text())['controller_build'] and control['title_id']==TITLE and set(control['artifacts'])=={'launch.elf','close.elf'},'controller authority')
    for name,expected in control['artifacts'].items():
        p=ROOT/'build/controllers'/TITLE/name
        require({'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}==expected,'controller bytes')
    return {'local_artifacts_verified':True,'hardware_gate_passed':False,'title_id':TITLE,'target':'gfx1013','kernel':'clr_gemm','shutdown_mode':'shell-request-probe','source_commit':SOURCE,'expected_log':m['expected_log'],'image':package['image'],'image_bytes':package['image']['bytes'],'controllers':control['artifacts']}
def verify_local():
    case=verify_closure()
    helper=load('deploy_helpers',ROOT/'results/deploy-private-buffer-milestone-v2.py')
    guards=load('deploy_guards',ROOT/'results/teardown-console-control-v3-initial-focus/run.py')
    return helper,guards,case

def consume(phase,out,record):
    require(phase in PHASES,'unknown phase')
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
    helper,guards,case=deps;locked=False;owned=[];token='clr-gemm-deployment-v1|'+str(uuid.uuid4());ftp=None;history=None;broken=False
    backup=FINAL+'.clr-gpu-success.sha256-'+OLD_IMAGE+'.preserved'
    log_backup=OLD_LOG+'.clr-gpu-success-PID407-appa018.sha256-'+LOG_SHA
    temporary='/data/homebrew/.PPSA99014.clr-gpu-'+IMAGE+'.upload'
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
        require(Path(case['expected_log']).name not in logs,'GEMM ROCclr result already exists')
        require(fetch('/user/appmeta/'+TITLE+'/param.json',65536)==(STAGE/'app/sce_sys/param.json').read_bytes(),'installed metadata mismatch')
        require(fetch('/user/app/'+TITLE+'/mount.lnk',1024)==MOUNT.encode(),'installed mount link mismatch')
        old=fetch(FINAL,record['baseline_image_bytes']);require(len(old)==record['baseline_image_bytes'] and sha(old)==OLD_IMAGE,'baseline image digest mismatch')
        log=fetch(OLD_LOG,870);require(len(log)==870 and sha(log)==LOG_SHA,'baseline GPU log digest mismatch')
        return old,log
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
        require(Path(case['expected_log']).name not in logs,'GEMM ROCclr result appeared')
        require(fetch(FINAL,len(old))==old and fetch(OLD_LOG,len(log))==log,'baseline changed during upload')
        require(fetch('/user/appmeta/'+TITLE+'/param.json',65536)==(STAGE/'app/sce_sys/param.json').read_bytes() and fetch('/user/app/'+TITLE+'/mount.lnk',1024)==MOUNT.encode(),'installation changed during upload')
        release_start=history
        mutate('preserve-image',ftp.rename,FINAL,backup);record['stage']='old-image-preserved'
        require(fetch(backup,len(old))==old,'preserved image mismatch')
        mutate('preserve-log',ftp.rename,OLD_LOG,log_backup);record['stage']='old-log-preserved'
        require(fetch(log_backup,len(log))==log,'preserved GPU log mismatch')
        # A cached 1.6 image is keyed by path. Leave that path absent until passive cleanup completes.
        deadline=time.monotonic()+45;poll=0
        while True:
            helper.remaining(deadline);poll+=1;current=idle('release-%03d'%poll,deadline)
            require(TITLE+'.ffpkg' not in listing('/data/homebrew','release-%03d-homebrew'%poll,deadline),'final path reappeared before release')
            if unmounted(current[len(release_start):].decode('utf-8','strict')):
                helper.remaining(deadline);break
            time.sleep(min(3,helper.remaining(deadline)))
        record['stage']='old-registration-released'
        verify_direct();promotion_start=idle('before-promotion')
        require(fetch(backup,len(old))==old and fetch(log_backup,len(log))==log,'preserved artifacts changed')
        require(TITLE+'.ffpkg' not in listing('/data/homebrew','pre-promote-homebrew'),'final path conflict')
        mutate('promote',ftp.rename,temporary,FINAL);record['stage']='promoted'
        require(fetch(FINAL,len(payload),time.monotonic()+30)==payload,'promoted image mismatch')
        deadline=time.monotonic()+45;poll=0
        while True:
            helper.remaining(deadline);poll+=1;current=idle('registration-%03d'%poll,deadline)
            if helper.title_registered(current[len(promotion_start):].decode('utf-8','strict')):
                helper.remaining(deadline);break
            time.sleep(min(3,helper.remaining(deadline)))
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
    except BaseException as error:record['error']=repr(error)
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
    raise RuntimeError('INERT: independent package/control review and fresh .40 state gate required')
    deps=verify_local();require(not any((HERE/(p+'-consumed.json')).exists() for p in PHASES),'deployment already attempted')
    out=HERE/('deployment-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'));out.mkdir()
    baseline=ROOT/'.local/g5-clr-gpu-startup/dist/PPSA99014/gfx1013-clr-gpu-b21b9c51dc42/PPSA99014.ffpkg'
    record={'host':HOST,'title_id':TITLE,'kernel':'clr_gemm','source_commit':SOURCE,'image_sha256':IMAGE,'image':deps[2]['image'],'remote':FINAL,'stage':'local-verified','registered':False,'launched':False,'hardware_gate_passed':False,'attempted':[],'baseline_image_bytes':baseline.stat().st_size,'script_sha256':sha(Path(__file__).read_bytes()),'manifest_sha256':sha((HERE/'manifest.json').read_bytes())}
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


