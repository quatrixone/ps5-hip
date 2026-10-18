"""Batch compiler input authentication with fake compiler I/O, no SDK build."""
from pathlib import Path
import collections,sys,tempfile,threading,time,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import build_public_hip as b

class HeaderBatch(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.out=self.root/'output';self.out.mkdir()
  self.sources=[self.root/f's{i}.cpp' for i in range(4)];self.headers=[self.root/f'h{i}.h' for i in range(3)]
  for p in self.sources+self.headers:p.write_text(p.name)
  self.commands=[];self.after_compile=lambda source:None;self.actual_extra=[];self.actual_missing=False;self.before_omit=False
  self.addCleanup(patch.stopall);patch.object(b,'ROOT',self.root).start();patch.object(b,'run',side_effect=self.compiler).start()
 def compiler(self,output,label,argv,**kwargs):
  source=Path(argv[argv.index('-c')+1]) if '-M' not in argv else Path(argv[argv.index('-M')+1]);dep=Path(argv[argv.index('-MF')+1]);source=source if source.is_absolute() else Path(kwargs.get('cwd',self.root))/source;names=[source,*self.headers]
  if '-M' in argv and self.before_omit:names.remove(source)
  if '-M' not in argv:
   if self.actual_missing:names.remove(self.headers[0])
   names+=self.actual_extra
   Path(argv[argv.index('-o')+1]).write_text('object '+source.name)
  dep.write_text('target: '+' '.join(map(str,names))+'\n');self.commands.append(label)
  if '-M' not in argv:self.after_compile(source)
 def plan(self,p):return b._plan_bound_unit(self.out,p,self.out/(p.stem+'.o'),['compiler'],'compile-'+p.stem)
 def test_same_receipts_and_count_reduction(self):
  counts=collections.Counter();sha=b.sha
  def observed(p):counts[str(p)]+=1;return sha(p)
  with patch.object(b,'sha',side_effect=observed):
   old=[b.compile_bound_unit(self.out,p,self.out/(p.stem+'.o'),['compiler'],'compile-'+p.stem) for p in self.sources]
   before=sum(counts[str(p)] for p in self.sources+self.headers);counts.clear()
   new=b._compile_bound_plans([self.plan(p) for p in self.sources],2)
   after=sum(counts[str(p)] for p in self.sources+self.headers)
  self.assertEqual(old,new);self.assertEqual((before,after),(32,14));self.assertEqual([counts[str(p)] for p in self.sources+self.headers],[2]*7)
 def test_after_initial_mutation_rejected(self):
  self.after_compile=lambda _:self.headers[0].write_text('changed')
  with self.assertRaisesRegex(RuntimeError,'input changed'):b._compile_bound_plans([self.plan(p) for p in self.sources],1)
 def test_header_removed_after_initial_rejected(self):
  self.after_compile=lambda _:self.headers[0].unlink(missing_ok=True)
  with self.assertRaises((RuntimeError,OSError)):b._compile_bound_plans([self.plan(self.sources[0])],1)
 def test_actual_missing_or_added_dependency_rejected(self):
  extra=self.root/'extra.h';extra.write_text('extra')
  for added in (False,True):
   self.actual_missing=not added;self.actual_extra=[extra] if added else []
   with self.subTest(added=added),self.assertRaisesRegex(RuntimeError,'actual compiler dependencies'):b._compile_bound_plans([self.plan(self.sources[0])],1)
 def test_source_omission_rejects_before_compile(self):
  self.before_omit=True
  with self.assertRaisesRegex(RuntimeError,'omits source'):self.plan(self.sources[0])
  self.assertTrue(all(label.endswith('-headers') for label in self.commands))
 def test_alias_changed_after_planning_rejected_before_compile(self):
  target=self.headers[0];alias=self.root/'alias.h';alias.symlink_to(target);self.headers=[alias];plans=[self.plan(p) for p in self.sources]
  replacement=self.root/'replacement.h';replacement.write_text('replacement');alias.unlink();alias.symlink_to(replacement)
  with self.assertRaisesRegex(RuntimeError,'resolution changed before'):b._compile_bound_plans(plans,2)
  self.assertTrue(all(label.endswith('-headers') for label in self.commands))
 def test_alias_changed_after_actual_dependency_check_rejected(self):
  target=self.headers[0];alias=self.root/'alias.h';alias.symlink_to(target);self.headers=[alias];plan=self.plan(self.sources[0]);paths,compile=plan
  replacement=self.root/'replacement.h';replacement.write_text('replacement')
  def wrapped(headers):
   result=compile(headers);alias.unlink();alias.symlink_to(replacement);return result
  with self.assertRaisesRegex(RuntimeError,'resolution changed during'):b._compile_bound_plans([(paths,wrapped)],1)
 def test_failed_worker_waits_for_other_worker(self):
  started=threading.Event();finished=threading.Event();plans=[self.plan(p) for p in self.sources[:2]]
  def fail(_):self.assertTrue(started.wait(2));raise RuntimeError('worker failed')
  def slow(headers):started.set();time.sleep(.04);result=plans[1][1](headers);finished.set();return result
  with self.assertRaisesRegex(RuntimeError,'worker failed'):b._compile_bound_plans([(plans[0][0],fail),(plans[1][0],slow)],2)
  self.assertTrue(finished.is_set())
 def test_standalone_still_rejects_mutation(self):
  self.after_compile=lambda _:self.headers[0].write_text('changed')
  with self.assertRaisesRegex(RuntimeError,'input changed'):b.compile_bound_unit(self.out,self.sources[0],self.out/'s.o',['compiler'],'single')

 def test_real_compile_units_uses_complete_initial_and_final_union(self):
  source=self.root/'public';(source/'hipamd/src').mkdir(parents=True)
  for i in range(30):(source/'hipamd/src'/f'hip{i}.cpp').write_text('hip')
  projects=('tools/public_hip_bootstrap_console_title.cpp',*(f'project{i}.cpp' for i in range(44)))
  for name in (*projects,'deps/elevation/client/elevation.cpp'):
   p=self.root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('source')
  counts=collections.Counter();sha=b.sha
  def observed(p):counts[str(p)]+=1;return sha(p)
  with patch.object(b,'SOURCE_UNITS',()),patch.object(b,'PROJECT_UNITS',projects),patch.object(b,'DEPS',self.root/'deps'),patch.object(b,'compiler_flags',return_value=['compiler']),patch.object(b,'build_jobs',return_value=4),patch.object(b,'sha',side_effect=observed):
   records=b.compile_units(self.out,source)
  self.assertEqual(len(records),76)
  self.assertEqual(self.commands[:2],['compile-00-headers','compile-00'])
  self.assertTrue(all(n.endswith('-headers') for n in self.commands[2:77]))
  self.assertTrue(all(not n.endswith('-headers') for n in self.commands[77:]))
  headers=set().union(*(row['headers'] for row in records))
  self.assertEqual(sum(counts[p] for p in headers),164) # standalone title8 + parallel union156

if __name__=='__main__':unittest.main()
