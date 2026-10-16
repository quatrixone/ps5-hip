import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import qualify_public_hip_source as producer
import qualify_public_hip_llm as q

class QualificationTests(unittest.TestCase):
    def test_admission_replaced_once(self):
        self.assertEqual(q.replace_admission([Path('a.cpp'),Path('hip_admission_model.cpp')],['a.o','old.o'],'new.o'),['a.o','new.o'])
        for units,objects in [([],[]),([Path('hip_admission_model.cpp')],[]),([Path('hip_admission_model.cpp')]*2,[1,2])]:
            with self.assertRaises(ValueError):q.replace_admission(units,objects,'new.o')

    def test_paired_arguments_fail_before_output(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fresh'
            with patch('sys.argv',['producer','--output',str(p),'--llm-model','missing']),self.assertRaises(SystemExit):producer.main()
            self.assertFalse(p.exists())

    def test_host_transcript(self):
        text='\n'.join(['LOG_POLICY_PASS controls=7 unexpected_loader_error_rejected=true','M0_APP STORAGE copied=16 cleared=16 synthetic_add_rejected=true cpu_compute_calls=0 hip_launches=0 client_graph_executed=false',*q.MARKERS,'PRODUCTION_GATE_HOST_PASS actual_llama_process=true cpu_bypass=false storage_bounds=3 allocation_limit=67108864 model_compute=0 gpu_submits=0'])+'\n'
        q.check_transcript(text)
        for bad in [text+'extra\n',text.replace('graph_reused=true','graph_reused=false'),text.replace('forwarded=0','forwarded=1')]:
            with self.assertRaises(ValueError):q.check_transcript(bad)

    def test_caller_wrap(self):
        name='_ZN13llama_context13graph_computeXX'
        good='<'+name+'>:\n call 10 <__wrap_ggml_backend_sched_graph_compute_async>'
        self.assertEqual(q.check_call_chain('T '+name,good),name)
        for bad in [good.replace('__wrap_',''),good+'\n call 20 <ggml_backend_sched_graph_compute_async@plt>']:
            with self.assertRaises(ValueError):q.check_call_chain('T '+name,bad)

    def test_environment(self):
        with patch.dict(q.os.environ,{'LD_PRELOAD':'bad','LLAMA_GRAPH_REUSE_DISABLE':'1'}):
            env=q.environment();self.assertNotIn('LD_PRELOAD',env);self.assertNotIn('LLAMA_GRAPH_REUSE_DISABLE',env)

    def test_shared_llama_rejected(self):
        with self.assertRaises(ValueError):q.runtime_inputs('libllama.so => /tmp/libllama.so (0x1)',Path('/tmp'),{})

    def test_pinned_inventory_generated_fresh(self):
        with tempfile.TemporaryDirectory() as d:
            units,models=q.host_sources(q.ROOT/'.deps/sources/llama-client',Path(d),'b76b13379d3b7d67ab0bc424516bb6fd2522bd8e')
            self.assertEqual(len(units),44);self.assertEqual(len(models),161)
            self.assertTrue((Path(d)/'llama-version.h').is_file())

if __name__=='__main__':unittest.main()
