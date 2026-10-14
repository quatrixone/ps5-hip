"""One GPU ROCclr execution; inert until reviewed deployment receipt is bound."""
import datetime,ftplib,hashlib,importlib.util,json,re,socket,subprocess,sys,threading,time,uuid
from pathlib import Path
if not __debug__:raise RuntimeError('optimized Python is forbidden')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
HOST='192.168.4.40';TITLE='PPSA99014'
STAGE=ROOT/'.local/g5-hip-native-link/dist/PPSA99014/gfx1013-public-bootstrap-19107f507822'
CANDIDATE=ROOT/'.local/g5-hip-native-link'
SOURCE='ea37055286903f680e43b049d70730743ffa2358'
IMAGE_SHA='836c10c6e66eb3d98245e23f96394ac8a7f51c34c750a4b18b35fc22210975b8'
SNAPSHOT=ROOT/'.local/teardown-snapshot-v3/build/teardown-qualified-v3/teardown-snapshot.elf'
SNAPSHOT_SHA='e241976012341eac0a2045fc4bca1c2141a9dfca2d5a6111b69644d83948dd93'
PHASES=('pre-snapshot','launch','post-snapshot','close')
def sha(data):return hashlib.sha256(data).hexdigest()
def require(ok,message):
    if not ok:raise RuntimeError(message)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
_VERIFIED_MANIFEST_SHA=None

def verify_direct():
    raw=(HERE/'manifest.json').read_bytes()
    require(_VERIFIED_MANIFEST_SHA is not None and sha(raw)==_VERIFIED_MANIFEST_SHA,'manifest changed during operation')
    manifest=json.loads(raw)
    for name,digest in manifest['direct_sha256'].items():require(sha((ROOT/name).read_bytes())==digest,'direct input changed: '+name)
    require(sha((STAGE/'PPSA99014.ffpkg').read_bytes())==IMAGE_SHA,'candidate image changed')

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
    require(package['image']=={'file':'PPSA99014.ffpkg','bytes':9437184,'sha256':IMAGE_SHA},'package image')
    actual={p.relative_to(STAGE/'app').as_posix():{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in (STAGE/'app').rglob('*') if p.is_file()}
    require(actual==m['files'] and not any(p.is_symlink() for p in (STAGE/'app').rglob('*')),'staged file inventory')
    control=json.loads((ROOT/'build/controllers/PPSA99014/receipt.json').read_text())
    require(control==json.loads((ROOT/'evidence/offline-controllers.json').read_text())['controller_build'] and control['title_id']==TITLE and set(control['artifacts'])=={'launch.elf','close.elf'},'controller authority')
    for name,expected in control['artifacts'].items():
        p=ROOT/'build/controllers'/TITLE/name;require({'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size}==expected,'controller bytes')
    return {'local_artifacts_verified':True,'hardware_gate_passed':False,'title_id':TITLE,'target':'gfx1013','kernel':'public_hip_bootstrap','shutdown_mode':'shell-request-probe','source_commit':SOURCE,'expected_log':m['expected_log'],'image':package['image'],'controllers':control['artifacts']}

def verify_local():
    manifest=json.loads((HERE/'manifest.json').read_text())
    # This binding is supplied only after parent-reviewed actual deployment.
    binding=manifest.get('deployment_binding')
    require(isinstance(binding,dict) and set(binding)=={'receipt','review'},'reviewed deployment is not yet bound; offline preparation only')
    for key in ('receipt','review'):
        entry=binding[key]
        require(isinstance(entry,dict) and set(entry)=={'path','sha256'},'invalid deployment binding')
        relative=Path(entry['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,'deployment binding escapes repository')
        require(sha((ROOT/relative).read_bytes())==entry['sha256'],'deployment '+key+' changed')
    deployment=json.loads((ROOT/binding['receipt']['path']).read_text())
    review=json.loads((ROOT/binding['review']['path']).read_text())
    require(review.get('classification')=='clear' and review.get('deployment_receipt_sha256')==binding['receipt']['sha256'],'deployment review does not bind accepted receipt')
    case=verify_closure()
    helper=load('arithmetic_ftp',ROOT/'results/deploy-private-buffer-milestone-v2.py')
    snapshot=load('arithmetic_snapshot',ROOT/'results/teardown-console-control-v3-initial-focus/run.py')
    checker=load('cpu_checker',ROOT/'.local/g5-hip-native-link/tools/public_hip_controls/execution/check_log.py')
    acceptance=load('prospective_acceptance',HERE/'acceptance.py')
    require(sha(SNAPSHOT.read_bytes())==SNAPSHOT_SHA and SNAPSHOT.stat().st_size==110904,'snapshot changed')
    acceptance.validate_deployment(deployment)
    case['deployment_receipt']=deployment;case['deployment_binding']=binding
    return helper,snapshot,checker,acceptance,case

def consume(phase,out,record):
    require(phase in PHASES,'unknown phase')
    with (HERE/(phase+'-consumed.json')).open('x') as stream:
        json.dump({'phase':phase,'output':str(out),'time':datetime.datetime.now(datetime.timezone.utc).isoformat()},stream)
    record.setdefault('attempted',[]).append(phase)

def events(delta):
    active=set();starts=[];stops=[]
    for match in re.finditer(r'\[(GAME|KSTUFF)\] (started|game stopped): ([A-Z0-9]+) pid=(\d+)\b',delta):
        kind,action,title,pid=match.groups();key=(title,pid)
        if (kind,action)==('GAME','started'):active.add(key);starts.append(key)
        elif (kind,action)==('KSTUFF','game stopped'):active.discard(key);stops.append(key)
    own=[key for key in starts if key[0]==TITLE]
    stopped=len(own)==1 and stops.count(own[0])==1 and not active
    return active,own,stopped

def own_fatal_crash(shadow,klog):
    active,own,stopped=events(shadow)
    if len(own)!=1:return None
    text=klog.decode('utf-8',errors='replace')
    matches=re.findall(r'# exception: ([^\n]+)\n(?:(?!# exception:)[\s\S]){0,4096}?# proc ID: (\d+)\b',text)
    matching=[exception for exception,pid in matches if int(pid)==int(own[0][1])]
    if len(matching)!=1:return None
    return {'pid':int(own[0][1]),'exception':matching[0],'stopped':stopped,'acceptance':False}

def parse_launch_response(response):
    require(isinstance(response,bytes) and response.endswith(b'\n') and len(response)<=65536,'launch response incomplete/bound')
    lines=response.decode('utf-8','strict').splitlines()
    tagged=[line for line in lines if '[launch]' in line]
    require(len(tagged)==1,'launch response duplicate/missing')
    match=re.fullmatch(r'\[launch\] title=PPSA99014 result=0x([0-9a-fA-F]{8})',tagged[0])
    require(match is not None,'launch response malformed/wrong title/initialization failure')
    result=int(match[1],16)
    require(result<0x80000000,'launch reported negative signed32 error')
    return {'unsigned_result':result,'result_hex':f'0x{result:08x}','nonnegative':True}


def send_controller(kind,out,case,helper,record,deadline):
    payload=(ROOT/'build/controllers'/TITLE/(kind+'.elf')).read_bytes();expected=case['controllers'][kind+'.elf']
    require(len(payload)==expected['bytes'] and sha(payload)==expected['sha256'],'controller changed')
    consume(kind,out,record);response=bytearray()
    try:
        with socket.create_connection((HOST,9021),timeout=helper.remaining(deadline,3)) as sock:
            sock.settimeout(helper.remaining(deadline));sock.sendall(payload)
            while True:
                sock.settimeout(helper.remaining(deadline));block=sock.recv(min(4096,65537-len(response)))
                if not block:break
                response.extend(block);require(len(response)<=65536,'controller response bound')
            helper.remaining(deadline)
    finally:(out/(kind+'-response.log')).write_bytes(response)
    if kind=='launch':record['launch_result']=parse_launch_response(bytes(response))
    record.setdefault('controller_responses',{})[kind]={'sha256':sha(response),'bytes':len(response)}

def capture(sock,stop,data,errors):
    deadline=time.monotonic()+120
    try:
        while not stop.is_set():
            require(time.monotonic()<deadline,'klog absolute120s deadline')
            sock.settimeout(.5)
            try:block=sock.recv(min(65536,4194305-len(data)))
            except socket.timeout:continue
            require(bool(block),'klog EOF');data.extend(block);require(len(data)<=4194304,'klog4MiB bound')
    except Exception as error:errors.append(repr(error))

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

def verify_installed(ftp,helper,case,out,label='installed'):
    remote=helper.fetch(ftp,'/data/homebrew/'+TITLE+'.ffpkg',case['image']['bytes'])
    require(len(remote)==case['image']['bytes'] and sha(remote)==IMAGE_SHA,'installed image mismatch')
    param=helper.fetch(ftp,'/user/appmeta/'+TITLE+'/param.json',65536)
    require(param==(STAGE/'app/sce_sys/param.json').read_bytes(),'installed metadata mismatch')
    link=helper.fetch(ftp,'/user/app/'+TITLE+'/mount.lnk',1024)
    require(link==b'/mnt/shadowmnt/PPSA99014_b6d9b2e4','installed link mismatch')
    mounted={}
    for name in ('eboot.bin','sce_module/libc.prx','lapy.elf','results-dir.txt'):
        local=(STAGE/'app'/name).read_bytes()
        path='/mnt/shadowmnt/PPSA99014_b6d9b2e4/'+name
        value=fetch_self_evidence(ftp,path,len(local),out/(label+'-'+name.replace('/','_')),helper) if name in ('eboot.bin','sce_module/libc.prx') else helper.fetch(ftp,path,len(local))
        require(value==local,'mounted candidate '+name+' mismatch')
        mounted[name]={'sha256':sha(value),'bytes':len(value)}
    (out/(label+'.json')).write_text(json.dumps({'image_sha256':sha(remote),'metadata_sha256':sha(param),'mount_link':link.decode(),'mounted_files':mounted},indent=2)+'\n')

def verify_log_absent(ftp,helper,case,out,label='private-log-absence'):
    path=case['expected_log'];directory=str(Path(path).parent);name=Path(path).name
    raw=helper.fetch(ftp,directory,65536,command='LIST')
    (out/(label+'.list')).write_bytes(raw)
    entries=helper.parse_listing(raw)
    require(name not in entries,'GPU ROCclr result already exists; refuse overwrite or stale output')
    (out/(label+'.json')).write_text(json.dumps({'path':path,'absent':True,'remote_mutations':0},indent=2)+'\n')


def run_cycle(out,record,deps):
    helper,snapshot,checker,acceptance,case=deps
    owned=[];token='public-hip-bootstrap-execution-v1|'+str(uuid.uuid4());ftp=None;sock=None;worker=None
    stop=threading.Event();klog=bytearray();errors=[];history=None;launch_history=None;ftp_broken=False
    def installed(label='installed'):
        nonlocal ftp_broken
        try:verify_installed(ftp,helper,case,out,label)
        except Exception:
            ftp_broken=True;raise
    def history_read(label,require_idle=True,deadline=None):
        nonlocal history,ftp_broken
        require(not ftp_broken,'FTP history stream unavailable after failed transfer')
        try:data=helper.fetch(ftp,'/data/shadowmount/debug.log',4194304,deadline)
        except Exception:
            ftp_broken=True;raise
        (out/(label+'.log')).write_bytes(data)
        if history is not None and not data.startswith(history):
            record['history_failed']=True;raise RuntimeError('history reset/truncated')
        if require_idle:require(snapshot.idle(data.decode(errors='replace')),'not fresh idle FW12.70')
        record.setdefault('history_checks',{})[label]={'continuous':True,'idle_required':require_idle,'idle':snapshot.idle(data.decode(errors='replace'))}
        history=data;return data
    def collect(phase):
        history_read('before-'+phase)
        verify_direct()
        require(sha((HERE/'manifest.json').read_bytes())==record['manifest_sha256'],'manifest changed')
        require(worker.is_alive() and not errors,'klog unavailable before snapshot')
        require(owned==snapshot.LOCKS and all(p.read_text()==token for p in owned),'exclusive locks changed')
        consume(phase,out,record);response=bytearray()
        payload=SNAPSHOT.read_bytes();require(sha(payload)==SNAPSHOT_SHA,'snapshot changed before send')
        try:parsed=snapshot.exchange(payload,response)
        finally:(out/(phase+'.log')).write_bytes(response)
        record[phase]=parsed
        return bytes(response)
    def close_owned():
        if 'launch' not in record.get('attempted',[]) or 'close' in record.get('attempted',[]):return
        current=history_read('before-close',False,time.monotonic()+8)
        require(launch_history is not None and current.startswith(launch_history),'no launch baseline')
        delta=current[len(launch_history):].decode(errors='replace');active,own,stopped=events(delta)
        record['stopped_before_close']=stopped
        if snapshot.idle(current.decode(errors='replace')):
            require(stopped,'close refused: no uniquely completed own cycle')
        else:require(len(own)==1 and active=={own[0]},'close refused: unrelated/ambiguous active identity')
        require(owned==snapshot.LOCKS and all(p.read_text()==token for p in owned),'exclusive locks changed before close')
        send_controller('close',out,case,helper,record,time.monotonic()+8)
    try:
        snapshot.acquire(owned,token);record['exclusive_owned_locks']=owned==snapshot.LOCKS and all(p.read_text()==token for p in owned)
        record['health_before']=snapshot.health()
        ftp=ftplib.FTP()
        with helper.bounded(ftp,time.monotonic()+10):ftp.connect(HOST,2121,timeout=5);ftp.login('anonymous','codex')
        history_read('before')
        raw_self_mode(ftp,helper,out)
        record['self_mode']={'command':'SELF','response':'226 SELF transfer mode disabled','scope':'this fresh FTP connection only; no retry or persistent setting'}
        installed();record['installed_verified']=True
        sock=socket.create_connection((HOST,3232),timeout=3)
        worker=threading.Thread(target=capture,args=(sock,stop,klog,errors),daemon=True);worker.start()
        pre=collect('pre-snapshot');record['baseline']=acceptance.validate_pre(pre)
        history_read('before-log-check');verify_log_absent(ftp,helper,case,out);record['log_absent']=True
        installed('installed-before-launch');record['mounted_candidate_files_verified']=True
        verify_log_absent(ftp,helper,case,out,'private-log-absence-before-launch')
        launch_history=history_read('before-launch');verify_direct()
        require(sha((HERE/'manifest.json').read_bytes())==record['manifest_sha256'],'manifest changed')
        require(worker.is_alive() and not errors,'klog stopped before launch')
        require(owned==snapshot.LOCKS and all(p.read_text()==token for p in owned),'exclusive locks changed before launch')
        record['klog_launch_offset']=len(klog)
        deadline=time.monotonic()+35
        try:
            send_controller('launch',out,case,helper,record,min(deadline,time.monotonic()+8))
            while True:
                time.sleep(min(1,helper.remaining(deadline)))
                current=history_read('observed',False,deadline)
                delta=current[len(launch_history):]
                fatal=own_fatal_crash(delta.decode(errors='replace'),bytes(klog[record['klog_launch_offset']:]))
                if fatal:
                    record['fatal_crash']=fatal;record['fatal_observed_before_deadline']=time.monotonic()<deadline;break
                if events(delta.decode(errors='replace'))[2]:
                    try:record['observed_lifecycle']=acceptance.check_lifecycle(delta,bytes(klog[record['klog_launch_offset']:]))
                    except ValueError as error:record['pending_lifecycle']=str(error);continue
                    launch_id=record['launch_result']['unsigned_result']
                    require(int(record['observed_lifecycle']['app_id'],16)==launch_id,'launch return and lifecycle app ID mismatch')
                    helper.remaining(deadline)
                    if not snapshot.idle(current.decode(errors='replace')):
                        record['pending_idle']='complete own lifecycle; waiting for strict idle within observation deadline'
                        continue
                    break
        except Exception as error:record['observation_error']=repr(error)
        record['klog_launch_end']=len(klog)
        (out/'launch-klog.log').write_bytes(klog[record['klog_launch_offset']:record['klog_launch_end']])
        require(history is not None and history.startswith(launch_history),'launch history unavailable')
        (out/'launch-shadow-delta.log').write_bytes(history[len(launch_history):])
        if record.get('fatal_crash'):
            record['stopped_before_close']=record['fatal_crash']['stopped']
            if record['fatal_crash']['stopped'] and not ftp_broken and snapshot.idle(history.decode(errors='replace')):
                collect('post-snapshot');history_read('after-crash-post',True,time.monotonic()+8)
                record['crash_poststate_collected']=True
            raise RuntimeError('uniquely identified own fatal native exception; workload failed')
        require(not ftp_broken,'no post-observation operations after uncertain FTP stream')
        close_owned()
        time.sleep(1)
        history_read('after-cycle')
        application=helper.fetch(ftp,case['expected_log'],65536);(out/'application.log').write_bytes(application)
        record['application_check']=checker.check_log(application)
        post=collect('post-snapshot');history_read('after-post')
        record['health_after']=snapshot.health()
        delta=history[len(launch_history):];(out/'lifecycle-delta.log').write_bytes(delta)
        record['evidence_complete']=True
    except BaseException as error:record['error']=repr(error)
    finally:
        if ftp and 'launch' in record.get('attempted',[]) and 'close' not in record.get('attempted',[]) and not ftp_broken and not record.get('fatal_crash'):
            try:close_owned()
            except Exception as error:record['cleanup_close_error']=repr(error)
        if record.get('attempted') and not record.get('evidence_complete'):
            try:history_read('failure-after',True,time.monotonic()+8)
            except Exception as error:record['post_idle_error']=repr(error)
        if record.get('attempted') and 'health_after' not in record:
            try:record['health_after']=snapshot.health()
            except Exception as error:record['health_error']=repr(error)
        stop.set();record['cleanup_errors']=[]
        for action in [lambda: worker.join(1) if worker and worker.ident is not None else None,
                       lambda: sock.close() if sock else None,
                       lambda: worker.join(1) if worker and worker.ident is not None else None,
                       lambda: ftp.close() if ftp else None]:
            try:action()
            except Exception as error:record['cleanup_errors'].append(repr(error))
        record['klog_errors']=errors;record['klog_worker_stopped']=not worker or not worker.is_alive()
        try:(out/'klog.log').write_bytes(klog)
        finally:
            record['lock_errors']=snapshot.release(owned,token);record['owned_locks_released']=not record['lock_errors']

def build_prerequisites(record):
    expected_health={'2121':'reachable','3232':'reachable','9021':'reachable'}
    attempted=record.get('attempted',[])
    return {
        'frozen_installed_image_verified':record.get('installed_verified') is True,
        'fresh_application_log_created':record.get('log_absent') is True and 'application_check' in record,
        'fresh_pre_post_snapshots':all(attempted.count(p)==1 and p in record for p in ('pre-snapshot','post-snapshot')),
        'fresh_launch_to_preclose_capture':'klog_launch_offset' in record and 'klog_launch_end' in record and record.get('klog_worker_stopped') is True and not record.get('klog_errors'),
        'shadow_history_continuous':bool(record.get('history_checks')) and not record.get('history_failed') and not record.get('post_idle_error'),
        'exclusive_owned_locks':record.get('exclusive_owned_locks') is True,
        'services_healthy_before_after':record.get('health_before')==expected_health and record.get('health_after')==expected_health,
        'owned_locks_released':record.get('owned_locks_released') is True and not record.get('lock_errors'),
        'provenance_verified':not record.get('local_final_error'),
        'deployment_receipt_verified':record.get('deployment_receipt_verified') is True,
        'mounted_candidate_files_verified':record.get('mounted_candidate_files_verified') is True,
    }

def main():
    raise RuntimeError('INERT: public HIP launch requires reviewed deployment binding and activation')
    deps=verify_local()
    require(not any((HERE/(p+'-consumed.json')).exists() for p in PHASES),'prospective cycle already attempted')
    out=HERE/('observation-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'));out.mkdir()
    record={'host':HOST,'title_id':TITLE,'kernel':'public_hip_bootstrap','source_commit':SOURCE,'image_sha256':IMAGE_SHA,'hardware_gate_passed':False,'public_hip_bootstrap_verified':False,'accepted':False,'attempted':[],'manifest_sha256':sha((HERE/'manifest.json').read_bytes()),'script_sha256':sha(Path(__file__).read_bytes())}
    record['deployment_receipt_verified']=True
    record['deployment_binding']=deps[4]['deployment_binding']
    try:run_cycle(out,record,deps)
    finally:
        try:
            verify_local()
            require(record['manifest_sha256']==sha((HERE/'manifest.json').read_bytes()) and record['script_sha256']==sha(Path(__file__).read_bytes()),'control changed during operation')
        except Exception as error:record['local_final_error']=repr(error)
        if record.get('evidence_complete'):
            try:
                prerequisites=build_prerequisites(record)
                record['assessment']=deps[3].assess_cycle((out/'pre-snapshot.log').read_bytes(),(out/'post-snapshot.log').read_bytes(),
                    (out/'launch-shadow-delta.log').read_bytes(),(out/'launch-klog.log').read_bytes(),(out/'application.log').read_bytes(),record.get('stopped_before_close',False),prerequisites=prerequisites,deployment_receipt=deps[4]['deployment_receipt'])
            except Exception as error:record['assessment_error']=repr(error)
        record['accepted']=record.get('assessment',{}).get('accepted') is True and not any(record.get(k) for k in ('error','observation_error','cleanup_close_error','health_error','klog_errors','lock_errors','cleanup_errors','local_final_error','assessment_error','post_idle_error'))
        record['public_hip_bootstrap_verified']=record['accepted'] and record.get('assessment',{}).get('public_hip_bootstrap_verified') is True
        record['raw_sha256']={p.name:sha(p.read_bytes()) for p in out.iterdir() if p.is_file()}
        (out/'receipt.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'output':str(out),'error':record.get('error'),'hardware_gate_passed':False,'public_hip_bootstrap_verified':record['public_hip_bootstrap_verified']}))
    return 0 if record['accepted'] else 1
if __name__=='__main__':raise SystemExit(main())
