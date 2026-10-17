"""Reject changed or omitted compiler inputs before accepting a native object."""
from pathlib import Path
import sys
import tempfile
import os
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import build_public_hip as b


class CompileInputs(unittest.TestCase):
    def test_dependency_changes_and_input_mutation(self):
        for mode in ('valid', 'missing-source', 'new-header', 'mutated-header'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                source = out / 'source.cpp'
                header = out / 'header.h'
                source.write_text('#include "header.h"\n')
                header.write_text('original')
                obj = out / 'source.o'

                def run(output, label, args):
                    deps = [source, header]
                    if mode == 'missing-source':
                        deps = [header]
                    if '-MD' in args:
                        obj.write_bytes(b'object')
                        if mode == 'new-header':
                            deps.append(out / 'unexpected.h')
                        if mode == 'mutated-header':
                            header.write_text('changed')
                    Path(args[args.index('-MF') + 1]).write_text(
                        'object: ' + ' '.join(map(str, deps)) + '\n')

                with patch.object(b, 'run', side_effect=run):
                    if mode == 'valid':
                        result = b.compile_bound_unit(out, source, obj, ['compiler'], 'compile')
                        self.assertEqual(result['headers'][str(header.resolve())], b.sha(header))
                        self.assertEqual(result['object_sha256'], b.sha(obj))
                    else:
                        with self.assertRaises(RuntimeError):
                            b.compile_bound_unit(out, source, obj, ['compiler'], 'compile')


    def test_swapped_alias_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);source=out/'source.cpp';source.write_text('source')
            a=out/'a.h';a.write_text('a');z=out/'z.h';z.write_text('z')
            first=out/'first.h';first.symlink_to(a);second=out/'second.h';second.symlink_to(z)
            obj=out/'source.o'
            def run(output,label,args):
                if '-MD' in args:
                    first.unlink();second.unlink();first.symlink_to(z);second.symlink_to(a)
                    obj.write_bytes(b'object')
                Path(args[args.index('-MF')+1]).write_text('object: '+' '.join(map(str,(source,first,second)))+'\n')
            with patch.object(b,'run',side_effect=run),self.assertRaises(RuntimeError):
                b.compile_bound_unit(out,source,obj,['compiler'],'compile')

class Canonicalization(unittest.TestCase):
 def test_paths_symlinks_relative_and_dotdot(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'real').mkdir();(root/'real'/'sub').mkdir();(root/'real'/'a b.h').write_text('a')
   (root/'alias').symlink_to('real',target_is_directory=True);(root/'chain').symlink_to('alias',target_is_directory=True)
   (root/'leaf').symlink_to('chain/a b.h')
   names=[root/'real'/'a b.h',root/'alias'/'a b.h',root/'chain'/'sub'/'..'/'a b.h',root/'leaf']
   for p in names:self.assertEqual(b.canonicalize_dependencies([p]),{str(p):str(p.resolve(strict=True))})
   previous=os.getcwd()
   try:
    os.chdir(root);self.assertEqual(b.canonicalize_dependencies(['leaf'])[str(root/'leaf')],str((root/'leaf').resolve()))
   finally:os.chdir(previous)
 def test_invalid_and_loop(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'loop').symlink_to('loop')
   for name in ('missing','loop','.'):
    with self.subTest(name=name),self.assertRaises((OSError,RuntimeError,ValueError)):b.canonicalize_dependencies([p/name])
 def test_new_inventory_refreshes_alias(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for name in ('a','b'):(p/name).mkdir();(p/name/'x').write_text(name)
   link=p/'link';link.symlink_to('a',target_is_directory=True)
   first=b.canonicalize_dependencies([link/'x']);link.unlink();link.symlink_to('b',target_is_directory=True)
   self.assertNotEqual(first,b.canonicalize_dependencies([link/'x']))
 def test_directory_replacement_during_inventory_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'a').mkdir();(p/'a'/'x').write_text('a');real=Path.resolve;calls=0
   def resolve(path,*a,**kw):
    nonlocal calls
    result=real(path,*a,**kw)
    if path==p/'a':
     calls+=1
     if calls==1:(p/'a').rename(p/'old');(p/'a').mkdir();(p/'a'/'x').write_text('b')
    return result
   with patch.object(Path,'resolve',resolve),self.assertRaises(ValueError):b.canonicalize_dependencies([p/'a'/'x'])
 def test_symlink_retarget_during_inventory_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)
   for name in ('a','b'):(p/name).mkdir();(p/name/'x').write_text(name)
   link=p/'link';link.symlink_to('a',target_is_directory=True);real=Path.resolve;calls=0
   def resolve(path,*a,**kw):
    nonlocal calls
    result=real(path,*a,**kw)
    if path==link:
     calls+=1
     if calls==1:link.unlink();link.symlink_to('b',target_is_directory=True)
    return result
   with patch.object(Path,'resolve',resolve),self.assertRaises(ValueError):b.canonicalize_dependencies([link/'x'])
 def test_unrelated_sibling_creation_does_not_change_route(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'a').mkdir();(p/'a'/'x').write_text('a');real=Path.resolve;calls=0
   def resolve(path,*a,**kw):
    nonlocal calls
    result=real(path,*a,**kw)
    if path==p/'a':
     calls+=1
     if calls==1:(p/'new-log').write_text('unrelated output')
    return result
   with patch.object(Path,'resolve',resolve):self.assertEqual(b.canonicalize_dependencies([p/'a'/'x'])[str(p/'a'/'x')],str(p/'a'/'x'))
 def test_leaf_symlink_retarget_during_inventory_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'a').write_text('a');(p/'b').write_text('b');leaf=p/'leaf';leaf.symlink_to('a');real=Path.resolve
   def resolve(path,*a,**kw):
    result=real(path,*a,**kw)
    if path==leaf:leaf.unlink();leaf.symlink_to('b')
    return result
   with patch.object(Path,'resolve',resolve),self.assertRaises(ValueError):b.canonicalize_dependencies([leaf])
 def test_regular_file_replacement_during_inventory_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);leaf=p/'x';leaf.write_text('a');real=Path.resolve;calls=0
   def resolve(path,*a,**kw):
    nonlocal calls
    result=real(path,*a,**kw)
    if path==p:
     calls+=1
     if calls==2:leaf.rename(p/'old');leaf.write_text('b')
    return result
   with patch.object(Path,'resolve',resolve),self.assertRaises(ValueError):b.canonicalize_dependencies([leaf])


if __name__ == '__main__':
    unittest.main()
