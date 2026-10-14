"""Offline deployment model using actual controller logic, no console I/O."""
import ast,contextlib,importlib.util,json,tempfile,unittest
from pathlib import Path,PurePosixPath
from types import SimpleNamespace
from unittest.mock import patch,Mock
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('deployment',HERE/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
helper=r.load('pinned_helpers',r.ROOT/'results/deploy-private-buffer-milestone-v2.py')

def activated_main():
    node=next(n for n in ast.parse((HERE/'run.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    if isinstance(node.body[0],ast.Raise):node.body.pop(0)
    namespace=dict(vars(r));exec(compile(ast.Module(body=[node],type_ignores=[]),'<mock activated main>','exec'),namespace)
    return namespace['main']()

class Clock:
    def __init__(self):self.now=0
    def sleep(self,n):self.now+=n
    def monotonic(self):return self.now
    def remaining(self,deadline,maximum=5):
        if self.now>=deadline:raise TimeoutError('deadline')
        return min(maximum,deadline-self.now)

class Tests(unittest.TestCase):
    def test_local_failure_precedes_remote_work(self):
        with patch.object(r,'verify_local',side_effect=RuntimeError('local rejection')) as local,patch.object(r,'deploy') as remote:
            with self.assertRaisesRegex(RuntimeError,'local rejection'):activated_main()
            local.assert_called_once();remote.assert_not_called()

    def test_actual_raw_self_helpers(self):
        h=SimpleNamespace(bounded=lambda *args:contextlib.nullcontext(),remaining=lambda deadline:1)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            for ack in ('226 SELF transfer mode disabled','226 SELF transfer mode enabled','200 OK',None):
                ftp=SimpleNamespace(sendcmd=Mock(return_value=ack))
                if ack=='226 SELF transfer mode disabled':r.raw_self_mode(ftp,h,out)
                else:
                    with self.assertRaisesRegex(RuntimeError,'not established'):r.raw_self_mode(ftp,h,out)
                ftp.sendcmd.assert_called_once_with('SELF')
                self.assertEqual(json.loads((out/'self-mode-response.json').read_text())['response'],ack)
            class Transfer:
                def __init__(self,data):self.data=data
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def settimeout(self,*args):pass
                def recv(self,n):data=self.data[:n];self.data=self.data[n:];return data
            for data in (b'abc',b'abcd',b'abcde',b'x'*100):
                ftp=SimpleNamespace(voidcmd=Mock(),transfercmd=lambda command:Transfer(data),sock=SimpleNamespace(settimeout=lambda *args:None),voidresp=Mock())
                dest=out/'partial'
                if len(data)>4:
                    with self.assertRaisesRegex(RuntimeError,'exceeded exact bound'):r.fetch_self_evidence(ftp,'/mounted/eboot.bin',4,dest,h)
                    ftp.voidresp.assert_not_called()
                else:self.assertEqual(r.fetch_self_evidence(ftp,'/mounted/eboot.bin',4,dest,h),data)
                self.assertEqual(dest.read_bytes(),data[:5])

    def test_oneuse_main_refuses_before_operation(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(r,'HERE',Path(temp)),patch.object(r,'verify_local',return_value=None),patch.object(r,'deploy') as op:
            r.consume('upload',Path(temp),{})
            with self.assertRaises(FileExistsError):r.consume('upload',Path(temp),{})
            # Test the unchanged post-activation one-use guard with remote work mocked.
            node=next(n for n in ast.parse((HERE/'run.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
            if isinstance(node.body[0],ast.Raise):node.body.pop(0)
            namespace=dict(vars(r));exec(compile(ast.Module(body=[node],type_ignores=[]),'<mock activated main>','exec'),namespace)
            with self.assertRaisesRegex(RuntimeError,'deployment already attempted'):namespace['main']()
            op.assert_not_called()

    def test_exact_release_parser(self):
        line='[22:00:00] [IMG][LVD] unmount complete: source='+r.FINAL+' mount='+r.MOUNT+' unit=2\n'
        self.assertTrue(r.unmounted(line))
        for bad in (line.replace('PPSA99014','PPSA990140'),line.replace('unit=2','unit=x'),line.replace('[LVD]','[MD]')):self.assertFalse(r.unmounted(bad))
        for bad in (line+line,line+'error: failed detach\n'):
            with self.assertRaises(RuntimeError):r.unmounted(bad)

    def model(self,mode):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);out=Path(tmp.name);stage=out/'stage';(stage/'app/sce_sys').mkdir(parents=True);(stage/'app/sce_module').mkdir()
        payload=b'n'*9437184;old=b'oldimage';log=b'L'*870;eboot=b'newelf';code=b'newcode';clock=Clock();ops=[];health=[]
        for name,data in [('PPSA99014.ffpkg',payload),('app/eboot.bin',eboot),('app/sce_module/libc.prx',code),('app/lapy.elf',b'helper'),('app/results-dir.txt',b'/results'),('app/sce_sys/param.json',b'param')]: (stage/name).write_bytes(data)
        old_sha=r.sha(old);log_sha=r.sha(log);new_sha=r.sha(payload)
        backup=r.FINAL+'.clr-gpu-success.sha256-'+old_sha+'.preserved';logbackup=r.OLD_LOG+'.clr-gpu-success-PID407-appa018.sha256-'+log_sha
        temporary='/data/homebrew/.PPSA99014.public-hip-'+new_sha+'.upload';private='/data/ps5-hip/results/private.log'
        historic=r.OLD_LOG+'.sha256-historic';files={r.FINAL:old,r.OLD_LOG:log,historic:b'untouched','/user/appmeta/'+r.TITLE+'/param.json':b'param','/user/app/'+r.TITLE+'/mount.lnk':r.MOUNT.encode(),r.MOUNT+'/eboot.bin':b'oldelf',r.MOUNT+'/sce_module/libc.prx':b'oldcode',r.MOUNT+'/lapy.elf':b'oldhelper',r.MOUNT+'/results-dir.txt':b'/oldresults'}
        atomic_path='/data/homebrew/atomic.preserved';files[atomic_path]=b'atomicbackup'
        atomic={atomic_path:{'bytes':12,'sha256':r.sha(b'atomicbackup')}}
        for index in range(29):
            p='/data/homebrew/prior-'+str(index);files[p]=b'priorbackup';atomic[p]={'bytes':11,'sha256':r.sha(b'priorbackup')}
        if mode=='atomic_changed':files[atomic_path]=b'wrong'
        if mode=='oldhash':files[r.FINAL]=b'badimage'
        if mode=='oldlog':files[r.OLD_LOG]=b'X'*870
        if mode=='conflict':files[backup]=b'conflict'
        if mode=='private_exists':files[private]=b'old private'
        history=b'IDLE\n';released=False;promoted=False
        def fetch(ftp,path,limit=4194304,deadline=None,command='RETR'):
            nonlocal history,released
            if command=='LIST':return ''.join(f'-rw-r--r-- 1 user group {len(v)} Oct 13 00:00 {Path(p).name}\n' for p,v in files.items() if str(PurePosixPath(p).parent)==path).encode()
            if path.endswith('debug.log'):
                if mode=='busy':return b'BUSY\n'
                if mode=='reset' and 'upload' in rec['attempted']:return b'RESET\n'
                if r.FINAL not in files and not released and mode!='release_timeout':
                    history+=('[IMG][LVD] unmount complete: source='+r.FINAL+' mount='+r.MOUNT+' unit=2\n').encode();released=True
                if promoted and not b'Mounting image' in history and mode!='stale_registration':
                    history+=('[IMG] Mounting image (ufs): '+r.FINAL+' -> '+r.MOUNT+'\n[IMG][LVD] Mounted (ufs) /dev/lvd2 -> '+r.MOUNT+'\n[LINK] mount.lnk created: /user/app/'+r.TITLE+'/mount.lnk -> '+r.MOUNT+'\n').encode()
                return history
            if mode=='upload_bad' and path==temporary:return b'corrupt'
            if path not in files:raise OSError('absent '+path)
            return files[path]
        def store(command,stream):
            ops.append(('upload',command));files[command[5:]]=stream.read()
            if mode=='upload_error':raise OSError('uncertain upload')
        def rename(source,dest):
            nonlocal promoted
            ops.append(('rename',source,dest))
            if source==temporary:
                self.assertTrue(released,'promotion cannot precede passive unmount')
                if mode=='promote_error':raise OSError('uncertain promotion')
                promoted=True
                if mode!='mounted_old':
                    files[r.MOUNT+'/eboot.bin']=eboot;files[r.MOUNT+'/sce_module/libc.prx']=code
                    files[r.MOUNT+'/lapy.elf']=b'helper';files[r.MOUNT+'/results-dir.txt']=b'/results'
                    if mode=='mounted_libc':files[r.MOUNT+'/sce_module/libc.prx']=b'wrong'
                    if mode=='mounted_helper':files[r.MOUNT+'/lapy.elf']=b'wrong'
                    if mode=='mounted_config':files[r.MOUNT+'/results-dir.txt']=b'wrong'
            self.assertNotIn(dest,files);files[dest]=files.pop(source)
        class Transfer:
            def __init__(self,data):self.data=data
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def settimeout(self,*args):pass
            def recv(self,n):result=self.data[:n];self.data=self.data[n:];return result
        ftp=SimpleNamespace(sendcmd=lambda command:'226 SELF transfer mode enabled' if mode=='wrong_self' else '226 SELF transfer mode disabled',voidcmd=lambda command:None,transfercmd=lambda command:Transfer(files[command[5:]]),sock=SimpleNamespace(settimeout=lambda *args:None),voidresp=lambda:None,connect=lambda *x,**k:None,login=lambda *x:None,close=lambda:None,storbinary=store,rename=rename)
        locks=[out/'lock1',out/'lock2']
        def acquire(owned,token):
            for p in locks:
                p.write_text(token);owned.append(p)
                if mode=='lockbusy':raise RuntimeError('second lock busy')
        def release(owned,token):
            for p in owned:self.assertEqual(p.read_text(),token);p.unlink()
            return []
        def healthcheck():
            health.append(1)
            result={str(p):'reachable' for p in (2121,3232,9021)}
            if (mode=='health_before' and len(health)==1) or (mode=='health_after' and len(health)==2):result['9021']='unreachable'
            return result
        guards=SimpleNamespace(LOCKS=locks,acquire=acquire,release=release,health=healthcheck,idle=lambda s:s.startswith('IDLE'))
        h=SimpleNamespace(fetch=fetch,parse_listing=helper.parse_listing,bounded=lambda *x:contextlib.nullcontext(),remaining=clock.remaining,title_registered=helper.title_registered)
        rec={'baseline_image_bytes':len(old),'stage':'local-verified','registered':False,'attempted':[]}
        with contextlib.ExitStack() as stack:
            for obj,name,value in [(r,'PRESERVED_BACKUPS',atomic),(r,'HERE',out),(r,'STAGE',stage),(r,'OLD_IMAGE',old_sha),(r,'LOG_SHA',log_sha),(r,'IMAGE',new_sha),(r,'verify_direct',lambda:None),(r.ftplib,'FTP',lambda:ftp),(r.time,'monotonic',clock.monotonic),(r.time,'sleep',clock.sleep)]:stack.enter_context(patch.object(obj,name,value))
            r.deploy(out,rec,(h,guards,{'expected_log':private,'image_bytes':len(payload)}))
        self.assertEqual(files[historic],b'untouched');self.assertEqual(files[atomic_path],b'wrong' if mode=='atomic_changed' else b'atomicbackup');self.assertEqual(len(health),0 if mode=='lockbusy' else 2);self.assertTrue(rec['owned_locks_released'])
        return rec,ops,files,backup,logbackup,payload,old,log

    def test_full_deploy_no_launch_exact_mount_bytes(self):
        rec,ops,files,backup,logbackup,payload,old,log=self.model('good')
        self.assertTrue(rec['registered']);self.assertNotIn('error',rec)
        self.assertEqual(len(rec['prior_backups_unchanged']['before']),30);self.assertEqual(rec['prior_backups_unchanged']['before'],rec['prior_backups_unchanged']['after'])
        self.assertEqual(rec['attempted'],list(r.PHASES));self.assertEqual(files[r.FINAL],payload);self.assertEqual(files[backup],old);self.assertEqual(files[logbackup],log)
        self.assertNotIn(r.OLD_LOG,files);self.assertEqual(len(ops),4);self.assertEqual(set(rec['mounted_files']),{'eboot.bin','sce_module/libc.prx','lapy.elf','results-dir.txt'})

    def test_preflight_no_mutations(self):
        for mode in ('lockbusy','busy','oldhash','oldlog','conflict','private_exists','wrong_self','health_before','atomic_changed'):
            with self.subTest(mode=mode):
                rec,ops,*_=self.model(mode);self.assertFalse(rec['registered']);self.assertIn('error',rec);self.assertEqual(ops,[])

    def test_bad_post_health_cannot_be_accepted(self):
        rec,*_=self.model('health_after')
        self.assertTrue(rec['registered']);self.assertIn('postflight services unhealthy',rec['health_error'])
        self.assertFalse(rec['registered'] and rec['owned_locks_released'] and not any(rec.get(k) for k in ('error','idle_error','health_error','ftp_close_error','lock_errors','local_final_error')))

    def test_uncertain_upload_hash_history_and_deadlines_stop(self):
        for mode in ('upload_error','upload_bad','reset','release_timeout','promote_error','stale_registration','mounted_old','mounted_libc','mounted_helper','mounted_config'):
            with self.subTest(mode=mode):
                rec,ops,*_=self.model(mode);self.assertFalse(rec['registered']);self.assertIn('error',rec)
                self.assertEqual(len(rec['attempted']),len(set(rec['attempted'])))
                if mode in ('upload_error','upload_bad','reset'):self.assertEqual(len(ops),1)
                if mode=='release_timeout':self.assertNotIn('promote',rec['attempted'])
                if mode.startswith('mounted_'):self.assertIn('mounted candidate bytes mismatch',rec['error'])



class PreservationAuthorityTests(unittest.TestCase):
    def test_exact_historical_inventory(self):
        actual=json.loads((r.ROOT/'results/clr-path-controls-v1/deployment/deployment-20261014T003557.771142Z/deployment.json').read_text())
        expected=dict(actual['prior_backups_unchanged']['after'])
        expected[actual['preservation']['package']]={'bytes':actual['baseline_image_bytes'],'sha256':actual['preservation']['package_sha256']}
        self.assertEqual(r.OLD_IMAGE,'84e52f366b9c7e598d81c3d37197ff8e71d548ebe6cfe933176639066abcc8da')
        self.assertNotEqual(r.OLD_IMAGE,r.IMAGE)
        self.assertEqual(len(expected),34);self.assertEqual(r.PRESERVED_BACKUPS,expected)
        self.assertEqual(json.loads((HERE/'manifest.json').read_text())['prior_backups'],expected)
if __name__=='__main__':unittest.main()
