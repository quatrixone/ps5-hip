"""Offline production-controller tests; sockets and console I/O are replaced."""
import ast, contextlib, importlib.util, json, socket, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cycle',HERE/'run.py');r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
OLD=(r.ROOT/'results/private-buffer-milestone-arithmetic_suite-20261012T134721.542681Z/application.log').read_bytes()

def activated_main():
    node=next(n for n in ast.parse((HERE/'run.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    if isinstance(node.body[0],ast.Raise):node.body.pop(0)
    namespace=dict(vars(r));exec(compile(ast.Module(body=[node],type_ignores=[]),'<mock activated main>','exec'),namespace)
    return namespace['main']()

class Clock:
    def __init__(self):self.now=0
    def time(self):return self.now
    def sleep(self,n):self.now+=n
    def remaining(self,deadline,maximum=5):
        if self.now>=deadline:raise TimeoutError('deadline')
        return min(maximum,deadline-self.now)

class Tests(unittest.TestCase):
    def test_dispatch_flag_receipt_and_summary_require_full_acceptance(self):
        import io
        for accepted,verified,cleanup in ((True,True,False),(True,False,False),(False,False,False),(True,True,True)):
            with self.subTest(accepted=accepted,verified=verified,cleanup=cleanup),tempfile.TemporaryDirectory() as temp:
                folder=Path(temp);(folder/'manifest.json').write_text('{}');output=io.StringIO()
                acceptance=SimpleNamespace(assess_cycle=lambda *args,**kw:{'accepted':accepted,'public_hip_bootstrap_verified':verified})
                deps=(None,None,None,acceptance,{'deployment_binding':{},'deployment_receipt':{}})
                def cycle(out,record,deps):
                    for name in ('pre-snapshot.log','post-snapshot.log','launch-shadow-delta.log','launch-klog.log','application.log'):(out/name).write_bytes(b'model')
                    record['evidence_complete']=True
                    if cleanup:record['cleanup_errors']=['injected cleanup failure']
                with patch.object(r,'HERE',folder),patch.object(r,'verify_local',return_value=deps),patch.object(r,'run_cycle',side_effect=cycle),contextlib.redirect_stdout(output):
                    status=activated_main()
                paths=list(folder.glob('observation-*/receipt.json'));self.assertEqual(len(paths),1)
                receipt=json.loads(paths[0].read_text());summary=json.loads(output.getvalue())
                expected=accepted and verified and not cleanup
                self.assertIs(receipt['public_hip_bootstrap_verified'],expected)
                self.assertIs(summary['public_hip_bootstrap_verified'],expected)
                self.assertIs(receipt['hardware_gate_passed'],False);self.assertIs(summary['hardware_gate_passed'],False)
                self.assertEqual(status,0 if accepted and not cleanup else 1)

    def test_exclusive_markers_and_original_untouched(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(r,'HERE',Path(temp)):
            old=Path(temp)/'consumed.json';old.write_bytes(b'original')
            for phase in r.PHASES:
                rec={};r.consume(phase,Path(temp),rec);self.assertEqual(rec['attempted'],[phase])
                with self.assertRaises(FileExistsError):r.consume(phase,Path(temp),rec)
            self.assertEqual(old.read_bytes(),b'original')
            with patch.object(r,'verify_local',return_value=None),patch.object(r,'run_cycle') as op:
                node=next(n for n in ast.parse((HERE/'run.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='main')
                while isinstance(node.body[0],ast.Raise):node.body.pop(0)
                namespace=dict(vars(r));exec(compile(ast.Module(body=[node],type_ignores=[]),'<mock activated draft>','exec'),namespace)
                with self.assertRaisesRegex(RuntimeError,'already attempted'):namespace['main']()
                op.assert_not_called()

    def test_private_log_absent_only_no_remote_mutations(self):
        for entries in ({'unrelated.log':{'type':'-','bytes':17}}, {'a.log':{'type':'-','bytes':1}}, {'a.log':{'type':'d','bytes':0}}):
            with tempfile.TemporaryDirectory() as temp:
                helper=SimpleNamespace(fetch=Mock(return_value=b'raw model listing'),parse_listing=lambda x:entries)
                ftp=Mock()
                if 'a.log' in entries:
                    with self.assertRaises(RuntimeError):r.verify_log_absent(ftp,helper,{'expected_log':'/results/a.log'},Path(temp))
                else:r.verify_log_absent(ftp,helper,{'expected_log':'/results/a.log'},Path(temp))
                self.assertFalse(ftp.method_calls)

    def test_unbound_deployment_refuses_before_locks_network(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(r,'HERE',Path(temp)),patch.object(r.socket,'create_connection') as network,patch.object(r.ftplib,'FTP') as ftp:
            (Path(temp)/'manifest.json').write_text('{"deployment_binding":null}')
            with patch.object(r,'verify_local',side_effect=RuntimeError('local rejection')) as local:
                with self.assertRaisesRegex(RuntimeError,'local rejection'):activated_main()
                local.assert_called_once()
            network.assert_not_called();ftp.assert_not_called()

    def test_actual_send_launch_return_contract(self):
        root=r.ROOT
        saved=(root/'results/arithmetic-prospective-cycle1/observation-20261013T021130.156360Z/launch-response.log').read_bytes()
        prior=(root/'results/private-buffer-milestone-arithmetic_suite-20261012T134721.542681Z/launch-response.log').read_bytes()
        prefix=b'[launch] title=PPSA99014 result=0x'
        good=[(saved,24),(prior,0xa018),(prefix+b'00000000\n',0),(prefix+b'7fffffff\n',0x7fffffff)]
        bad=[prefix+b'80000000\n',prefix+b'ffffffff\n',saved+saved,saved+b'[launch] rubbish\n',prefix+b'00000018',prefix+b'0000001\n',prefix+b'gggggggg\n',saved.replace(b'PPSA99014',b'PPSA99008'),b'[launch] title=PPSA99014 initialize=0x00000000\n']
        for response,want in good+[(b,None) for b in bad]:
            with self.subTest(response=response),tempfile.TemporaryDirectory() as temp,patch.object(r,'ROOT',Path(temp)),patch.object(r,'HERE',Path(temp)):
                folder=Path(temp)/'build/controllers'/r.TITLE;folder.mkdir(parents=True);(folder/'launch.elf').write_bytes(b'elf')
                sock=Mock();sock.__enter__=Mock(return_value=sock);sock.__exit__=Mock(return_value=False)
                # Duplicate records may arrive in a later packet, so consume through EOF.
                sock.recv.side_effect=[response[:len(response)//2],response[len(response)//2:],b'']
                record={};case={'controllers':{'launch.elf':{'bytes':3,'sha256':r.sha(b'elf')}}}
                with patch.object(r.socket,'create_connection',return_value=sock):
                    if want is None:
                        with self.assertRaises(RuntimeError):r.send_controller('launch',Path(temp),case,SimpleNamespace(remaining=lambda *x:1),record,8)
                    else:
                        r.send_controller('launch',Path(temp),case,SimpleNamespace(remaining=lambda *x:1),record,8)
                        self.assertEqual(record['launch_result']['unsigned_result'],want)
                self.assertEqual(record['attempted'],['launch']);self.assertEqual((Path(temp)/'launch-response.log').read_bytes(),response)

    def test_installed_stored_and_mounted_exact_bytes(self):
        names=('eboot.bin','sce_module/libc.prx','lapy.elf','results-dir.txt')
        with tempfile.TemporaryDirectory() as temp:
            stage=Path(temp);(stage/'app/sce_sys').mkdir(parents=True);(stage/'app/sce_module').mkdir()
            (stage/'app/sce_sys/param.json').write_bytes(b'param')
            for name in names:(stage/'app'/name).write_bytes(name.encode())
            for which in ('good','image','metadata','link',*names):
                values={'image':b'image','metadata':b'param','link':b'/mnt/shadowmnt/PPSA99014_b6d9b2e4',**{name:name.encode() for name in names}}
                if which!='good':values[which]+=b'x'
                def fetch(ftp,path,*args):
                    if path.endswith('.ffpkg'):return values['image']
                    if path.endswith('param.json'):return values['metadata']
                    if path.endswith('mount.lnk'):return values['link']
                    return values[path.split('/PPSA99014_b6d9b2e4/',1)[1]]
                signed=Mock(side_effect=lambda ftp,path,size,dest,helper:fetch(ftp,path))
                with patch.object(r,'STAGE',stage),patch.object(r,'IMAGE_SHA',r.sha(b'image')),patch.object(r,'fetch_self_evidence',signed):
                    if which=='good':
                        r.verify_installed(None,SimpleNamespace(fetch=fetch),{'image':{'bytes':5}},stage)
                        proof=json.loads((stage/'installed.json').read_text())
                        self.assertEqual(set(proof['mounted_files']),set(names));self.assertEqual(signed.call_count,2)
                        self.assertNotIn('canary.hsaco',proof['mounted_files'])
                    else:
                        with self.assertRaises(RuntimeError):r.verify_installed(None,SimpleNamespace(fetch=fetch),{'image':{'bytes':5}},stage)

    def test_raw_self_exact_ack_once_and_failure_capture(self):
        for response in ('226 SELF transfer mode disabled','226 SELF transfer mode enabled','200 okay','226 SELF transfer mode disabled extra',None):
            with self.subTest(response=response),tempfile.TemporaryDirectory() as temp:
                ftp=SimpleNamespace(sendcmd=Mock(return_value=response))
                helper=SimpleNamespace(bounded=lambda *x:contextlib.nullcontext())
                if response=='226 SELF transfer mode disabled':r.raw_self_mode(ftp,helper,Path(temp))
                else:
                    with self.assertRaises(RuntimeError):r.raw_self_mode(ftp,helper,Path(temp))
                ftp.sendcmd.assert_called_once_with('SELF')
                self.assertEqual(json.loads((Path(temp)/'self-mode-response.json').read_text()),{'command':'SELF','response':response})
        with tempfile.TemporaryDirectory() as temp:
            ftp=SimpleNamespace(sendcmd=Mock(side_effect=socket.timeout('uncertain ACK')))
            with self.assertRaises(socket.timeout):r.raw_self_mode(ftp,helper,Path(temp))
            ftp.sendcmd.assert_called_once_with('SELF')
            self.assertIsNone(json.loads((Path(temp)/'self-mode-response.json').read_text())['response'])

    def test_self_read_bound_and_uncertain_bytes_preserved(self):
        for mode in ('good','oversize','timeout'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                transfer=Mock();transfer.__enter__=Mock(return_value=transfer);transfer.__exit__=Mock(return_value=False)
                transfer.recv.side_effect=[b'ab',b''] if mode=='good' else ([b'abc'] if mode=='oversize' else [b'a',socket.timeout()])
                ftp=SimpleNamespace(voidcmd=Mock(),transfercmd=Mock(return_value=transfer),sock=Mock(),voidresp=Mock())
                helper=SimpleNamespace(bounded=lambda *x:contextlib.nullcontext(),remaining=lambda *x:1)
                destination=Path(temp)/'raw'
                if mode=='good':self.assertEqual(r.fetch_self_evidence(ftp,'/mounted/eboot.bin',2,destination,helper),b'ab')
                else:
                    with self.assertRaises((RuntimeError,socket.timeout)):r.fetch_self_evidence(ftp,'/mounted/eboot.bin',2,destination,helper)
                self.assertEqual(destination.read_bytes(),{'good':b'ab','oversize':b'abc','timeout':b'a'}[mode])
                self.assertEqual(ftp.voidresp.call_count,1 if mode=='good' else 0)
                ftp.transfercmd.assert_called_once_with('RETR /mounted/eboot.bin')

    def test_send_uncertain_preserves_raw_consumes_once(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(r,'ROOT',Path(temp)),patch.object(r,'HERE',Path(temp)):
            folder=Path(temp)/'build/controllers'/r.TITLE;folder.mkdir(parents=True);(folder/'launch.elf').write_bytes(b'elf')
            sock=Mock();sock.__enter__=Mock(return_value=sock);sock.__exit__=Mock(return_value=False);sock.recv.side_effect=[b'partial',socket.timeout()]
            rec={};case={'controllers':{'launch.elf':{'bytes':3,'sha256':r.sha(b'elf')}}}
            with patch.object(r.socket,'create_connection',return_value=sock) as connect:
                with self.assertRaises(socket.timeout):r.send_controller('launch',Path(temp),case,SimpleNamespace(remaining=lambda *x:1),rec,8)
                with self.assertRaises(FileExistsError):r.send_controller('launch',Path(temp),case,SimpleNamespace(remaining=lambda *x:1),rec,8)
                self.assertEqual(connect.call_count,1)
            self.assertEqual((Path(temp)/'launch-response.log').read_bytes(),b'partial');self.assertEqual(rec['attempted'],['launch'])

    def test_capture_eof_bound_deadline(self):
        for mode in ('eof','bytes','deadline'):
            sock=Mock();sock.recv.return_value=b'' if mode=='eof' else b'x'*4194305
            stop=Mock();stop.is_set.return_value=False;data=bytearray();errors=[]
            with patch.object(r.time,'monotonic',side_effect=[0,121] if mode=='deadline' else [0,0]):r.capture(sock,stop,data,errors)
            self.assertEqual(len(errors),1)

    def run_model(self,mode):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);out=Path(temp.name);(out/'manifest.json').write_bytes(b'{}')
        payload=out/'snapshot.elf';payload.write_bytes(b'payload');clock=Clock();record={'manifest_sha256':r.sha(b'{}'),'attempted':[],'deployment_receipt_verified':True}
        idle=b'FW: 12.70.\nidle\n';history=idle;dataref=None;exchange_count=0;sends=[];health_calls=[];labels=[]
        start=b'[GAME] started: PPSA99014 pid=77 app_id=0x123\n';stop=b'[KSTUFF] game stopped: PPSA99014 pid=77\n'
        locks=[out/'a.lock',out/'b.lock']
        def acquire(owned,token):
            if mode=='lock_busy':raise RuntimeError('busy shared locks')
            if mode=='partial_lock_busy':
                locks[0].write_text(token);owned.append(locks[0]);raise RuntimeError('second shared lock busy')
            for p in locks:p.write_text(token);owned.append(p)
        def release(owned,token):
            for p in owned:
                if p.read_text()!=token:return ['wrong token']
                p.unlink()
            return []
        def health():health_calls.append(1);return {str(x):'reachable' for x in (2121,3232,9021)}
        def fetch(ftp,path,*args,**kwargs):
            nonlocal history
            if path.endswith('debug.log'):
                labels.append(path)
                if 'launch' in sends:
                    if mode=='history_io':raise OSError('uncertain FTP history stream')
                    if mode=='busy':history=idle+start+b'[GAME] started: PPSA99008 pid=88 app_id=0x124\n'
                    elif mode=='reset':history=b'reset\n'
                    else:history=idle+start+stop
                return history
            return OLD
        def exchange(payload,response):
            nonlocal exchange_count
            exchange_count+=1;response.extend(b'snapshot\n')
            if exchange_count==1 and mode=='pre_io':raise OSError('uncertain snapshot')
            if dataref is not None:dataref.extend(b'PRE QUERY\n' if exchange_count==1 else b'POST QUERY\n')
            return {'parsed':True}
        def validate_pre(raw):
            if mode=='pre_invalid':raise ValueError('invalid baseline')
            return {'ready':True}
        def lifecycle(shadow,klog):
            if mode=='delayed' and clock.now<4:raise ValueError('not complete yet')
            if mode=='late':clock.now=35
            if mode in ('partial','busy','own_crash','foreign_crash'):raise ValueError('incomplete')
            return {'pid':77,'app_id':'0x00000123' if mode!='appid' else '0x00000124'}
        class Worker:
            ident=1
            def __init__(self,**kw):
                nonlocal dataref
                dataref=kw['args'][2];self.stop=kw['args'][1]
            def start(self):pass
            def is_alive(self):return not self.stop.is_set()
            def join(self,*x):pass
        def send(kind,*args):
            r.consume(kind,out,record);sends.append(kind);record['launch_result']={'unsigned_result':0x123};dataref.extend(b'LAUNCH\n' if kind=='launch' else b'CLOSE QUERY\n')
            if kind=='launch' and mode in ('own_crash','foreign_crash'):
                dataref.extend(('# exception: 0xa002030a (SYSTEM_ILLEGAL_FUNCTION_CALL)\n# thread ID: 1\n# proc ID: '+('77' if mode=='own_crash' else '88')+'\n').encode())
        helper=SimpleNamespace(fetch=fetch,bounded=lambda *x:contextlib.nullcontext(),remaining=clock.remaining)
        snapshot=SimpleNamespace(LOCKS=locks,acquire=acquire,release=release,health=health,idle=lambda s:'reset' not in s and (not '[GAME]' in s or '[KSTUFF]' in s) and 'PPSA99008' not in s and (mode not in ('idle_delayed','idle_timeout') or '[GAME]' not in s or (mode=='idle_delayed' and clock.now>=4)),exchange=exchange)
        acceptance=SimpleNamespace(validate_pre=validate_pre,check_lifecycle=lifecycle)
        checker=SimpleNamespace(check_log=lambda *x,**kw:{'application_result':'application-pass'})
        with contextlib.ExitStack() as stack:
            ftp=Mock();ftp.sendcmd.return_value='226 SELF transfer mode enabled' if mode=='raw_ack' else '226 SELF transfer mode disabled'
            for obj,name,value in [(r,'HERE',out),(r,'SNAPSHOT',payload),(r,'SNAPSHOT_SHA',r.sha(b'payload')),(r,'verify_direct',lambda:None),(r,'verify_installed',lambda *x:None),(r,'verify_log_absent',lambda *x:None),(r,'send_controller',send),(r.time,'monotonic',clock.time),(r.time,'sleep',clock.sleep),(r.threading,'Thread',Worker),(r.ftplib,'FTP',lambda:ftp),(r.socket,'create_connection',Mock())]:stack.enter_context(patch.object(obj,name,value))
            r.run_cycle(out,record,(helper,snapshot,checker,acceptance,{'expected_log':'/application.log'}))
        return out,record,sends,health_calls

    def test_own_fatal_crash_stops_promptly_preserves_and_never_passes(self):
        out,rec,sends,health=self.run_model('own_crash')
        self.assertEqual(sends,['launch']);self.assertEqual(rec['fatal_crash']['pid'],77)
        self.assertTrue(rec['fatal_observed_before_deadline']);self.assertTrue(rec['crash_poststate_collected'])
        self.assertIn('own fatal native exception',rec['error']);self.assertNotIn('observation_error',rec)
        self.assertFalse(rec.get('evidence_complete',False));self.assertTrue(rec['owned_locks_released'])
        self.assertIn(b'SYSTEM_ILLEGAL_FUNCTION_CALL',(out/'launch-klog.log').read_bytes())
        self.assertIn(b'pid=77',(out/'launch-shadow-delta.log').read_bytes());self.assertTrue((out/'post-snapshot.log').exists())
        self.assertEqual(len(health),2)

    def test_fatal_identity_rejects_foreign_and_ambiguous(self):
        log=b'# exception: fatal\n# proc ID: 77\n'
        self.assertIsNone(r.own_fatal_crash('[GAME] started: PPSA99008 pid=77\n',log))
        self.assertIsNone(r.own_fatal_crash('[GAME] started: PPSA99014 pid=88\n',log))
        self.assertIsNone(r.own_fatal_crash('[GAME] started: PPSA99014 pid=77\n[GAME] started: PPSA99014 pid=78\n',log))
        self.assertIsNone(r.own_fatal_crash('[GAME] started: PPSA99014 pid=77\n',log+log))

    def test_uncertain_ftp_after_launch_prevents_snapshot_and_close(self):
        out,rec,sends,health=self.run_model('history_io')
        self.assertEqual(sends,['launch']);self.assertEqual(rec['attempted'],['pre-snapshot','launch'])
        self.assertIn('uncertain FTP',rec['observation_error']);self.assertFalse(rec.get('evidence_complete',False))
        self.assertFalse((out/'post-snapshot.log').exists());self.assertTrue(rec['owned_locks_released'])

    def test_happy_window_close_once_and_delayed_convergence(self):
        for mode in ('good','delayed','idle_delayed'):
            out,rec,sends,health=self.run_model(mode)
            self.assertEqual(sends,['launch','close']);self.assertTrue(rec['evidence_complete']);self.assertTrue(rec['stopped_before_close'])
            self.assertEqual((out/'launch-klog.log').read_bytes(),b'LAUNCH\n');self.assertIn(b'POST QUERY',(out/'klog.log').read_bytes());self.assertTrue(all(r.build_prerequisites(rec).values()))
            self.assertEqual(len(health),2)
            if mode=='delayed':self.assertIn('pending_lifecycle',rec)
            if mode=='idle_delayed':self.assertIn('pending_idle',rec)

    def test_pre_snapshot_failure_postchecks_without_launch(self):
        for mode in ('pre_io','pre_invalid'):
            out,rec,sends,health=self.run_model(mode)
            self.assertEqual(sends,[]);self.assertEqual(rec['attempted'],['pre-snapshot']);self.assertEqual(len(health),2)
            self.assertTrue((out/'failure-after.log').exists());self.assertTrue(rec['owned_locks_released']);self.assertIn('error',rec)

    def test_busy_reset_partial_deadline_fail_closed(self):
        for mode in ('busy','reset','late','partial','appid'):
            out,rec,sends,health=self.run_model(mode)
            self.assertIn('observation_error',rec);self.assertEqual(sends.count('launch'),1)
            if mode in ('busy','reset'):self.assertEqual(sends,['launch']);self.assertIn('cleanup_close_error',rec)
            self.assertTrue(rec['owned_locks_released']);self.assertEqual(len(health),2)

    def test_busy_locks_make_zero_console_contacts(self):
        for mode in ('lock_busy','partial_lock_busy'):
            out,rec,sends,health=self.run_model(mode)
            self.assertEqual(sends,[]);self.assertEqual(health,[])
            self.assertEqual(rec['attempted'],[]);self.assertIn('error',rec)
            self.assertFalse((out/'before.log').exists());self.assertTrue(rec['owned_locks_released'])

    def test_wrong_raw_ack_prevents_reads_snapshots_and_launch(self):
        out,rec,sends,_=self.run_model('raw_ack')
        self.assertEqual(sends,[]);self.assertEqual(rec['attempted'],[])
        self.assertIn('raw SELF',rec['error']);self.assertTrue((out/'before.log').exists())
        self.assertTrue(rec['owned_locks_released'])
        self.assertEqual(json.loads((out/'self-mode-response.json').read_text())['response'],'226 SELF transfer mode enabled')

    def test_prerequisites_failure_conjunction(self):
        _,rec,_,_=self.run_model('good')
        for key,value in [('installed_verified',False),('log_absent',False),('deployment_receipt_verified',False),('mounted_candidate_files_verified',False),('klog_errors',['EOF']),('history_failed',True),('exclusive_owned_locks',False),('health_after',{}),('owned_locks_released',False),('local_final_error','changed')]:
            bad=dict(rec);bad[key]=value;self.assertFalse(all(r.build_prerequisites(bad).values()),key)

    def test_complete_lifecycle_never_idle_stays_bounded(self):
        out,rec,sends,health=self.run_model('idle_timeout')
        self.assertIn('pending_idle',rec)
        self.assertIn('observation_error',rec)
        self.assertIn('deadline',rec['observation_error'])
        self.assertEqual(sends,['launch'])
        self.assertFalse(rec.get('evidence_complete',False))
        self.assertTrue(rec['owned_locks_released'])
        self.assertEqual(len(health),2)

    def test_actual_saved_focus_lag_and_settle_predicate(self):
        snapshot=r.load('spill_idle_regression',r.ROOT/'results/teardown-console-control-v3-initial-focus/run.py')
        folder=r.ROOT/'results/private304-execution-v1/observation-20261013T025305.355455Z'
        receipt=json.loads((folder/'receipt.json').read_text())
        for name,want in [('before-close.log',False),('failure-after.log',True)]:
            data=(folder/name).read_bytes()
            self.assertEqual(r.sha(data),receipt['raw_sha256'][name])
            self.assertEqual(snapshot.idle(data.decode('utf-8')),want)

if __name__=='__main__':unittest.main()
