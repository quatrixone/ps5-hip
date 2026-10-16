from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import unittest,tempfile,json,threading
from pathlib import Path
from unittest.mock import patch
import source_console as c
from source_model_execute_test import Worker,Sock
class FTP:
 def __init__(self):self.active=True;self.closed=False
 def connect(self,*a,**k):pass
 def login(self,*a):pass
 def sendcmd(self,*a):return '226 SELF transfer mode disabled'
 def close(self):self.closed=True
 def retrbinary(self,cmd,callback,blocksize):
  callback(b'[GAME] started: PPSA99014 pid=1\n' if cmd=='RETR '+c.HISTORY else b'partial log')
 def retrlines(self,cmd,callback):callback('drwx PPSA99014_test' if self.active else 'drwx NPXS00001_test')
class Cleanup(unittest.TestCase):
 def test_expired_close_always_releases_socket(self):
  ftp=FTP();wrapped=c.PhaseFTP(ftp,lambda:(_ for _ in ()).throw(RuntimeError('expired')));wrapped.close();self.assertTrue(ftp.closed)
 def test_expired_after_launch_closes_unlocks_and_writes(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'deploy').mkdir();(root/'deploy/receipt.json').write_text(json.dumps(dict(accepted=True,healthy_after=True,lock_released=True,phase='deploy',manifest_sha256='m',package_sha256='p')))
   ftp=FTP();clock=[0];calls=[];health=[];o=c.Control.__new__(c.Control)
   o.fixture='llm_client';o.manifest={'llm_client':{'phase':'model-gate'}};o.phase='execute';o.phase_deadline=1800;o.token='owned';o.owned=False;o.ftp=None;o.candidate=o.out=root;o.manifest_sha='m';o.history=None;o.record={'accepted':False,'launched':False,'package_sha256':'p'};o.stop=threading.Event();o.klog=bytearray();o.capture_errors=[];o.log='/log'
   o.health=lambda:health.append(True) or True
   for name in ('idle','installed','fresh_log','fresh_rmsnorm_dumps','fresh_q4_dumps','fresh_weighted_qkv_dumps'):setattr(o,name,lambda:None)
   o.history_read=lambda name:b''
   def send(name):
    calls.append(name)
    if name=='launch':clock[0]=1801
    if name=='close':ftp.active=False
    return b''
   o.send=send
   with patch.object(c,'LOCK',root/'lock'),patch.object(c.ftplib,'FTP',return_value=ftp),patch.object(c.socket,'create_connection',return_value=Sock()),patch.object(c.threading,'Thread',Worker),patch.object(c,'snapshot',return_value=[]),patch.object(c.time,'monotonic',lambda:clock[0]):
    self.assertEqual(o.run(),1)
   self.assertEqual(calls,['snapshot','launch','close']);self.assertTrue(ftp.closed);self.assertEqual(len(health),2);self.assertFalse((root/'lock').exists())
   receipt=json.loads((root/'receipt.json').read_text());self.assertTrue(receipt['safety_close_sent']);self.assertTrue(receipt['lock_released']);self.assertTrue(receipt['healthy_after']);self.assertFalse(receipt['accepted']);self.assertIn('deadline',receipt['error']);self.assertEqual(o.phase_deadline,1800)
 def test_actual_send_verify_expiry_never_opens_socket(self):
  for name,deadline in [('close',60),('launch',1800)]:
   with self.subTest(name=name):
    clock=[0];o=c.Control.__new__(c.Control);o.fixture='llm_client';o.manifest={'llm_client':{'phase':'model-gate'}};o.cleanup_deadline=60;o.phase_deadline=1800
    def verify():clock[0]=deadline+1
    o.verify=verify;o.verify_close_payload=verify;o.owned_lock=lambda:None
    with patch.object(c.time,'monotonic',lambda:clock[0]),patch.object(c.socket,'create_connection') as connect:
     with self.assertRaisesRegex(RuntimeError,'deadline'):o.send(name)
     connect.assert_not_called()
 def test_actual_send_receive_expiry_preserves_evidence(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'close.elf').write_bytes(b'controller');clock=[0];o=c.Control.__new__(c.Control);o.fixture='llm_client';o.manifest={'llm_client':{'phase':'model-gate'},'close':'close.elf','files':{'close.elf':c.sha(b'controller')}};o.cleanup_deadline=60;o.out=root;o.token='owned';o.verify=lambda:None;o.verify_close_payload=lambda:None;o.owned_lock=lambda:None
   class Socket:
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def settimeout(self,n):pass
    def sendall(self,p):pass
    def recv(self,n):
     self.bound=n;clock[0]=61;return b'late'
   sock=Socket()
   with patch.object(c,'ROOT',root),patch.object(c.time,'monotonic',lambda:clock[0]),patch.object(c.socket,'create_connection',return_value=sock):
    with self.assertRaisesRegex(RuntimeError,'cleanup deadline'):o.send('close')
   self.assertLessEqual(sock.bound,65536);self.assertTrue((root/'close.log').exists());self.assertTrue((root/'close-consumed').exists())
 def test_narrow_close_binding(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'close.elf').write_bytes(b'close');(root/'model').write_bytes(b'model');o=c.Control.__new__(c.Control);o.candidate=root;o.manifest={'close':'close.elf','files':{'close.elf':c.sha(b'close'),'model':c.sha(b'model')}};(root/'manifest.json').write_text(json.dumps(o.manifest));o.manifest_sha=c.sha((root/'manifest.json').read_bytes())
   with patch.object(c,'ROOT',root):
    (root/'model').write_bytes(b'changed unrelated model');o.verify_close_payload()
    (root/'close.elf').write_bytes(b'wrong')
    with self.assertRaisesRegex(RuntimeError,'close payload'):o.verify_close_payload()
    (root/'close.elf').write_bytes(b'close');(root/'manifest.json').write_text('{}')
    with self.assertRaisesRegex(RuntimeError,'manifest'):o.verify_close_payload()
 def test_cleanup_refuses_unrelated_title(self):
  o=c.Control.__new__(c.Control);o.cleanup_history=lambda name:b'[GAME] started: PPSA99014 pid=1\n';o.cleanup_active=lambda:{'PPSA12345'};o.send=lambda name:self.fail('close sent')
  with self.assertRaisesRegex(RuntimeError,'unrelated title'):o.emergency_close(b'')
 def test_cleanup_cannot_read_arbitrary_remote(self):
  o=c.Control.__new__(c.Control);o.log='/log'
  with self.assertRaisesRegex(RuntimeError,'forbidden'):o.cleanup_fetch('/data/arbitrary',1)
if __name__=='__main__':unittest.main()
