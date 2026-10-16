"""Exercise production validation cost and rejection contracts without a native build."""
import ast,hashlib,tempfile,unittest,threading,sys
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(HERE))
import build_llm_native_app as app_driver
CLIENT=ast.parse((HERE/'build_llm_native_client.py').read_text())
APP=ast.parse((HERE/'build_llm_native_app.py').read_text())
FUNCTION=next(n for n in CLIENT.body if isinstance(n,ast.FunctionDef) and n.name=='app_objects')
def execute(node,env):exec(compile(ast.Module(body=[node],type_ignores=[]),'<production-validation>','exec'),env)
def loop(target):return next(n for n in FUNCTION.body if isinstance(n,ast.For) and ast.unparse(n.target)==target)
def nested(name):return next(n for n in FUNCTION.body if isinstance(n,ast.FunctionDef) and n.name==name)
class Cost(unittest.TestCase):
 def test_duplicate_headers_hash_once_each_epoch_and_changed_bytes_reject(self):
  with tempfile.TemporaryDirectory() as directory:
   p=Path(directory)/'header';p.write_bytes(b'one');calls=[]
   def sha(p):calls.append(str(p));return hashlib.sha256(Path(p).read_bytes()).hexdigest()
   env={'Path':Path,'inputs':{str(p):sha(p)},'sha':sha,'verified_inputs':set()};calls.clear()
   execute(loop('(name, digest)'),env);execute(nested('required'),env)
   for _ in range(100):env['required'](p)
   self.assertEqual(len(calls),1)
   execute(loop('(p, h)'),env);self.assertEqual(len(calls),2)
   p.write_bytes(b'two')
   with self.assertRaisesRegex(ValueError,'changed during validation'):execute(loop('(p, h)'),env)
   with self.assertRaisesRegex(ValueError,'missing'):env['required'](p.with_name('missing'))
 def test_duplicate_resolution_once_each_epoch_and_symlink_change_rejects(self):
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);a=folder/'a';b=folder/'b';link=folder/'link';a.write_bytes(b'same');b.write_bytes(b'same');link.symlink_to(a)
   env={'Path':Path,'resolved_dependencies':{}};execute(nested('resolve_dependency'),env)
   original=Path.resolve;calls=[]
   def resolve(p,*args,**kwargs):calls.append(str(p));return original(p,*args,**kwargs)
   with patch.object(Path,'resolve',resolve):
    for _ in range(100):self.assertEqual(env['resolve_dependency'](str(link)),a)
    self.assertEqual(len(calls),1);execute(loop('(name, target)'),env);self.assertEqual(len(calls),2)
    link.unlink();link.symlink_to(b)
    with self.assertRaisesRegex(ValueError,'resolution changed'):execute(loop('(name, target)'),env)
 def test_changed_depfile_pair_still_rejected(self):
  node=next(n for n in ast.walk(FUNCTION) if isinstance(n,ast.If) and 'paths[0] != paths[1]' in ast.unparse(n.test))
  with self.assertRaisesRegex(ValueError,'dependency closure'):execute(node,{'paths':[{Path('/a')},{Path('/b')}],'p':Path('/a')})
 def test_link_trace_resolves_only_consumed_inputs_once_each_epoch(self):
  node=next(n for n in ast.walk(APP) if isinstance(n,ast.If) and ast.unparse(n.test)=="label == 'link' or host_link")
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);a=folder/'a';b=folder/'b';a.write_bytes(b'a');b.write_bytes(b'b');(folder/'link.stdout').write_text((str(a)+'\n')*20)
   env={'Path':Path,'ROOT':folder,'label':'link','host_link':False,'inputs':{str(a):'a',str(b):'b'},'out':folder,'row':{},'bind':lambda p,expected:None}
   original=Path.resolve;calls=[]
   def resolve(p,*args,**kwargs):calls.append(str(p));return original(p,*args,**kwargs)
   with patch.object(Path,'resolve',resolve):execute(node,env)
   self.assertEqual(len(calls),2);self.assertEqual(len(env['row']['selected_inputs']),20)
 def test_link_trace_rejects_unknown_selected_input(self):
  node=next(n for n in ast.walk(APP) if isinstance(n,ast.If) and ast.unparse(n.test)=="label == 'link' or host_link")
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);known=folder/'known';unknown=folder/'unknown';known.write_bytes(b'k');unknown.write_bytes(b'u');(folder/'link.stdout').write_text(str(unknown)+'\n')
   bound=[];env={'Path':Path,'ROOT':folder,'label':'link','host_link':False,'inputs':{str(known):'k'},'out':folder,'row':{},'bind':lambda p,expected:bound.append(p)}
   with self.assertRaisesRegex(ValueError,'unbound selected linker input'):execute(node,env)
   self.assertFalse(bound);self.assertNotIn('selected_inputs',env['row'])
 def test_link_trace_rejects_changed_symlink_route_after_binding(self):
  node=next(n for n in ast.walk(APP) if isinstance(n,ast.If) and ast.unparse(n.test)=="label == 'link' or host_link")
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);a=folder/'a';b=folder/'b';link=folder/'link';a.write_bytes(b'same');b.write_bytes(b'same');link.symlink_to(a);(folder/'link.stdout').write_text(str(link)+'\n')
   bound=[]
   def bind(p,expected):
    bound.append(p);link.unlink();link.symlink_to(b)
   env={'Path':Path,'ROOT':folder,'label':'link','host_link':False,'inputs':{str(link):'same'},'out':folder,'row':{},'bind':bind}
   with self.assertRaisesRegex(ValueError,'link input resolution changed'):execute(node,env)
   self.assertEqual(bound,[link]);self.assertNotIn('selected_inputs',env['row'])
 def test_link_trace_accepts_bound_target_or_bound_alias_and_passes_expected_digest(self):
  node=next(n for n in ast.walk(APP) if isinstance(n,ast.If) and ast.unparse(n.test)=="label == 'link' or host_link")
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);target=folder/'target';alias=folder/'alias';target.write_bytes(b'content');alias.symlink_to(target)
   for known,selected in ((target,alias),(alias,target)):
    (folder/'link.stdout').write_text(str(selected)+'\n');calls=[]
    env={'Path':Path,'ROOT':folder,'label':'link','host_link':False,'inputs':{str(known):'expected'},'out':folder,'row':{},'bind':lambda p,h:calls.append((p,h))}
    execute(node,env);self.assertEqual(calls,[(selected,'expected')])
 def test_link_trace_checks_selected_digest_and_rejects_drift(self):
  node=next(n for n in ast.walk(APP) if isinstance(n,ast.If) and ast.unparse(n.test)=="label == 'link' or host_link")
  with tempfile.TemporaryDirectory() as directory:
   folder=Path(directory);p=folder/'input';p.write_bytes(b'old');digest=hashlib.sha256(p.read_bytes()).hexdigest();p.write_bytes(b'new');(folder/'link.stdout').write_text(str(p)+'\n')
   def bind(path,expected):
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('identity differs')
   env={'Path':Path,'ROOT':folder,'label':'link','host_link':False,'inputs':{str(p):digest},'out':folder,'row':{},'bind':bind}
   with self.assertRaisesRegex(ValueError,'identity differs'):execute(node,env)
   self.assertNotIn('selected_inputs',env['row'])
BUILD=next(n for n in APP.body if isinstance(n,ast.FunctionDef) and n.name=='build')
BIND=next(n for n in BUILD.body if isinstance(n,ast.FunctionDef) and n.name=='bind')
def binder():
 env={'Path':Path,'b':app_driver.b,'lock':threading.RLock(),'inputs':{},'links':{}}
 exec(compile(ast.Module(body=[BIND],type_ignores=[]),'<actual-bind>','exec'),env);return env['bind']
class Headers(unittest.TestCase):
 def test_duplicate_headers_hashed_once_at_merge_boundary(self):
  with tempfile.TemporaryDirectory() as name:
   p=Path(name)/'header';p.write_bytes(b'one');digest=app_driver.b.sha(p);units=[{'headers':{str(p):digest}} for _ in range(76)];bind=binder()
   with patch.object(app_driver.b,'sha',wraps=app_driver.b.sha) as sha:
    for path,h in app_driver.merged_unit_headers(units).items():bind(path,h)
   self.assertEqual(sha.call_count,1)
 def test_conflicting_compile_identity_rejected_before_hash(self):
  with patch.object(app_driver.b,'sha') as sha:
   with self.assertRaisesRegex(ValueError,'conflicting compiled header'):app_driver.merged_unit_headers([{'headers':{'header':'a'}},{'headers':{'header':'b'}}])
   sha.assert_not_called()
 def test_current_file_drift_still_rejected_by_original_bind(self):
  with tempfile.TemporaryDirectory() as name:
   p=Path(name)/'header';p.write_bytes(b'before');headers=app_driver.merged_unit_headers([{'headers':{str(p):app_driver.b.sha(p)}}]);p.write_bytes(b'after');bind=binder()
   with self.assertRaisesRegex(ValueError,'identity differs'):
    for path,h in headers.items():bind(path,h)

if __name__=='__main__':unittest.main()
