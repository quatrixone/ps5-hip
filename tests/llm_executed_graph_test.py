import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import build_llm_executed_graph as graph

class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
    def rows(self):
        rows = []
        for pos in graph.POSITIONS:
            for op in sorted(graph.OPS):
                t = dict(name='kq-0' if op == 'MUL_MAT' else op, dtype='f32', op=op,
                    description=op, ne=[256 if pos < 256 else 512,1,32,1] if op == 'MUL_MAT' else [64,1,1,1],
                    nb=[4,256,256,256], view_offset=0, view_source=None, op_params_hex='00'*64)
                rows.append(dict(position=pos,phase='prefill' if pos<4 else 'decode',node=len(rows),tensor=t,sources=[]))
        return rows
    def inventory(self, rows):
        p=self.root/'trace.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows));return graph.inventory(p)
    def test_complete(self):
        value=self.inventory(self.rows());self.assertEqual(value['node_count'],140);self.assertEqual(set(value['ops']),graph.OPS)
    def test_missing_position(self):
        with self.assertRaises(ValueError):self.inventory(self.rows()[:-14])
    def test_order(self):
        rows=self.rows();rows[1]['node']=0
        with self.assertRaisesRegex(ValueError,'sequence'):self.inventory(rows)
    def test_missing_operator(self):
        rows=self.rows()
        for r in rows:
            if r['tensor']['description']=='SWIGLU':r['tensor']['description']='MUL'
        with self.assertRaisesRegex(ValueError,'operator'):self.inventory(rows)
    def test_bad_metadata(self):
        for field,value in [('ne',[0,1,1,1]),('nb',[4,-1,1,1]),('view_offset',-1),('op_params_hex','ff')]:
            rows=self.rows();rows[0]['tensor'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.inventory(rows)
    def test_cache_growth(self):
        rows=self.rows()
        for r in rows:
            if r['position']==256 and r['tensor']['name']=='kq-0':r['tensor']['ne'][0]=256
        with self.assertRaisesRegex(ValueError,'cache growth'):self.inventory(rows)
    def test_sources(self):
        rows=self.rows();t=copy.deepcopy(rows[0]['tensor']);rows[0]['sources']=[dict(index=0,tensor=t),dict(index=0,tensor=t)]
        with self.assertRaisesRegex(ValueError,'indices'):self.inventory(rows)
    def steps(self):
        rows=[dict(position=i,phase='prefill' if i<4 else 'decode',input_token=graph.PROMPT[i] if i<4 else 0,next_token=0,completed_nodes=14) for i in range(512)]
        p=self.root/'steps';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        f=self.root/'logits'
        with f.open('wb') as stream:stream.truncate(512*32000*4)
        return f,p,rows
    def test_full_steps(self):
        f,p,_=self.steps();out=graph.validate_steps(f,p,self.inventory(self.rows()));self.assertEqual(out['finite_logits'],16384000)
    def test_bad_steps_and_logits(self):
        f,p,rows=self.steps();rows[4]['input_token']=1;p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError,'token'):graph.validate_steps(f,p,self.inventory(self.rows()))
        f,p,_=self.steps()
        with f.open('r+b') as stream:stream.write(struct.pack('<f',float('nan')))
        with self.assertRaisesRegex(ValueError,'nonfinite'):graph.validate_steps(f,p,self.inventory(self.rows()))
    def test_changed_input(self):
        p=self.root/'input';p.write_text('original');identities={str(p):graph.sha(p)};p.write_text('changed')
        with self.assertRaisesRegex(ValueError,'changed'):graph.recheck(identities)
    def test_existing_output_refused(self):
        with self.assertRaisesRegex(ValueError,'fresh output'):graph.build(self.root,self.root,self.root)
    def test_dependency_paths(self):
        p=self.root/'headers.d';p.write_text('graph: /tmp/header\\ with\\ spaces.h /tmp/second.h\n')
        self.assertEqual(graph.dependencies(p),{'/tmp/header with spaces.h','/tmp/second.h'})
        p.write_text('graph: ')
        with self.assertRaisesRegex(ValueError,'empty'):graph.dependencies(p)
    def test_identity_conflict(self):
        identities={'input':'old'}
        with self.assertRaisesRegex(ValueError,'conflicting'):graph.merge(identities,'input','new')
        self.assertEqual(identities,{'input':'old'})
    def test_runtime_authentication(self):
        lib=self.root/'CPU libs';lib.mkdir();identities={};lines=[]
        for name in ['libllama','libggml','libggml-base','libggml-cpu']:
            p=lib/(name+'.so');p.write_bytes(name.encode());identities[str(p)]=graph.sha(p)
            lines.append(name+'.so => '+str(p)+' (0x123)')
        text='\n'.join(lines)
        self.assertEqual(graph.runtime_inputs(text,lib,identities),identities)
        with self.assertRaisesRegex(ValueError,'unresolved'):graph.runtime_inputs('libllama.so => not found',lib,identities)
        (lib/'libllama.so').write_text('changed')
        with self.assertRaisesRegex(ValueError,'unauthenticated'):graph.runtime_inputs(text,lib,identities)

class CpuInputsTests(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.cpu = self.root / 'cpu'; self.cpu.mkdir()
        self.model = self.root / 'model.gguf'; self.model.write_bytes(b'good-model')
        (self.root / 'evidence').mkdir()
        (self.root / 'llm-inputs.json').write_text(json.dumps({'client': {'revision': 'locked'}}))
        self.expected = graph.sha(self.model)
        (self.root / 'evidence/offline-llm-model-conversion.json').write_text(json.dumps(
            {'outputs': {'tinyllama-q4_0.gguf': {'bytes': 10, 'sha256': self.expected}}}))
        source = self.root / 'source'; source.write_bytes(b'source')
        tool = self.root / 'compiler'; tool.write_bytes(b'tool')
        artifacts = {}
        for name in ('libllama.so', 'libggml.so', 'libggml-base.so', 'libggml-cpu.so'):
            path = self.cpu / name; path.write_bytes(name.encode()); artifacts[name] = graph.sha(path)
        self.receipt = {'classification': 'host-cpu-reference-tools-built', 'source_revision': 'locked',
            'identity': {'inputs': {'source': graph.sha(source)},
                         'tools': {'compiler': {'path': str(tool), 'sha256': graph.sha(tool)}}},
            'artifacts': artifacts}
        self.save()
        self.addCleanup(patch.stopall)
        patch.object(graph, 'ROOT', self.root).start()
        patch.object(graph.subprocess, 'check_output', side_effect=lambda argv, **kw:
            'locked\n' if argv[-1] == 'HEAD' else '').start()
    def save(self):
        (self.cpu / 'receipt.json').write_text(json.dumps(self.receipt))
    def test_one_fresh_model_hash_per_invocation(self):
        from unittest.mock import patch
        original = graph.sha
        with patch.object(graph, 'sha', wraps=original) as hashed:
            identities = graph.cpu_inputs(self.cpu, self.model)
            self.assertEqual(identities[str(self.model)], self.expected)
            self.assertEqual(sum(Path(c.args[0]) == self.model for c in hashed.call_args_list), 1)
            graph.cpu_inputs(self.cpu, self.model)
            self.assertEqual(sum(Path(c.args[0]) == self.model for c in hashed.call_args_list), 2)
    def test_same_size_corruption_rejected(self):
        self.model.write_bytes(b'evil-model')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            graph.cpu_inputs(self.cpu, self.model)
    def test_same_size_change_during_closure_rejected(self):
        from unittest.mock import patch
        original = graph.merge
        def mutate(inputs, key, digest):
            original(inputs, key, digest)
            if Path(key).name == 'source': self.model.write_bytes(b'evil-model')
        with patch.object(graph, 'merge', side_effect=mutate), self.assertRaisesRegex(ValueError, 'input changed'):
            graph.cpu_inputs(self.cpu, self.model)
    def test_changed_source_tool_or_library_rejected(self):
        for path in (self.root / 'source', self.root / 'compiler', self.cpu / 'libllama.so'):
            content = path.read_bytes()
            with self.subTest(path=path):
                path.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'input changed'):
                    graph.cpu_inputs(self.cpu, self.model)
                path.write_bytes(content)
    def test_missing_library_and_unsafe_artifact_rejected(self):
        original = self.receipt['artifacts'].copy()
        del self.receipt['artifacts']['libllama.so']; self.save()
        with self.assertRaisesRegex(ValueError, 'libraries missing'):
            graph.cpu_inputs(self.cpu, self.model)
        self.receipt['artifacts'] = {**original, '../escape': '0' * 64}; self.save()
        with self.assertRaisesRegex(ValueError, 'unsafe CPU artifact'):
            graph.cpu_inputs(self.cpu, self.model)
    def test_conflicting_model_identity_rejected(self):
        self.receipt['identity']['inputs'][str(self.model)] = '0' * 64; self.save()
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            graph.cpu_inputs(self.cpu, self.model)

if __name__=='__main__':unittest.main()
