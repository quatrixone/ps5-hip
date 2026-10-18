import unittest
from native_generation_transcript_test import fixture
from native_full_transcript_test import transcript
import source_llm_generation_log as log


def native_fixture():
 raw,policy,sizes=fixture(8)
 memory=[('load_begin','before_load',12882804736),('load_complete','after_load',12000002048),('context_complete','after_context',11000004608),('cleanup_begin','after_cleanup',12882804736)]
 for phase,name,free in memory:
  marker=f'M3_FULL PHASE {phase}\n'.encode();line=f'M0_APP MEMORY phase={name} pool_bytes=12882804736 largest_free_bytes={free} total_free_bytes=unknown alignment=16384\n'.encode();raw=raw.replace(marker,marker+line)
 baseline=transcript().decode().splitlines();footer=baseline[-9:];footer[1]=log.APP_COMPLETE
 return ('\n'.join(baseline[:2])+'\n').encode()+raw+('\n'.join(footer)+'\n').encode(),policy,sizes

class GenerationApplication(unittest.TestCase):
 def test_complete_app_transport_timing_and_memory(self):
  r=log.check_full_log(*native_fixture());self.assertEqual((r['requests'],r['raw_files'],r['gpu_submits']),(16,384,11728));self.assertEqual(r['profile'],'generation');self.assertEqual(len(r['memory']),4);self.assertTrue(r['finalization_transcript_complete']);self.assertFalse(r['numerical_verified']);self.assertFalse(r['gpu_origin_established'])
 def test_lifecycle_mutations(self):
  raw,p,s=native_fixture()
  for old,new in [(b'profile=generation watchdog',b'profile=smoke watchdog'),(b'releases=10',b'releases=9'),(b'frees=16',b'frees=15'),(b'retired=1',b'retired=2'),(b'phase=after_cleanup pool_bytes=12882804736 largest_free_bytes=12882804736',b'phase=after_cleanup pool_bytes=12882804736 largest_free_bytes=12000002048'),(b'phase=after_context',b'phase=after_load')]:
   self.assertIn(old,raw)
   with self.subTest(old=old),self.assertRaises(ValueError):log.check_full_log(raw.replace(old,new,1),p,s)
 def test_missing_native_memory_and_finalization(self):
  raw,p,s=native_fixture()
  for data in (raw[:-1],raw+b'extra\n',b'\n'.join(line for line in raw.split(b'\n') if not line.startswith(b'M0_APP MEMORY')),raw.replace(b'\n',b'\r\n')):
   with self.assertRaises(ValueError):log.check_full_log(data,p,s)
 def test_helper_is_not_native_app(self):
  with self.assertRaises(ValueError):log.check_full_log(*fixture(8))
  with self.assertRaisesRegex(ValueError,'request profile'):log.check_full_log(*fixture())

if __name__=='__main__':unittest.main()
