import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import source_llm_full_log as log


def transcript():
    lines = [*log.LLM_CLIENT_LINES[:2], 'M3_FULL PROFILE name=smoke requests=3 raw_files=72 raw_bytes=949248']
    for phase, memory, free in [('load_begin', 'before_load', 12882804736), ('load_complete', 'after_load', 12000002048), ('context_complete', 'after_context', 11000004608)]:
        lines += ['M3_FULL PHASE ' + phase, f'M0_APP MEMORY phase={memory} pool_bytes=12882804736 largest_free_bytes={free} total_free_bytes=unknown alignment=16384']
    for run, pos in ((0, 0), (0, 1), (1, 0)):
        lines += [f'M3_FULL RAW file={name} bytes={size} synced=true' for name, size in log.FILES.items() if name.startswith(f'run-{run}-pos-{pos}-')]
        lines += [f'M3_FULL CHECKPOINT run={run} position={pos} token={pos} committed_position={pos} layers=22 logits=32000']
    lines += ['M3_FULL PHASE cleanup_begin', 'M0_APP MEMORY phase=after_cleanup pool_bytes=12882804736 largest_free_bytes=12882804736 total_free_bytes=unknown alignment=16384', 'M3_FULL TRANSPORT_COMPLETE requests=3 layers=22 logits=32000 nodes=2394 launches=2199 raw_files=72 raw_bytes=949248 instances=1/1 numerical=pending inference=false finalization=pending', 'M0_APP HOST_STORAGE allocations=10 releases=10 live_allocations=0 live_requested_bytes=0 peak_requested_bytes=4096 failures=0 poisoned=false scope=ggml_aligned', log.APP_COMPLETE, *log.LINES[3:6], 'M0_APP FINAL allocations=16 frees=16 queues=1 retired=1 gate_destroyed=true application_fini=pending', *log.LINES[7:]]
    return ('\n'.join(lines) + '\n').encode()


class FullTranscript(unittest.TestCase):
    def test_transport_only(self):
        result = log.check_full_log(transcript())
        self.assertEqual(result['raw_files'], 72)
        self.assertFalse(result['numerical_verified'])
        self.assertFalse(result['gpu_origin_established'])

    def test_rejections(self):
        raw = transcript()
        for old, new in [(b'profile=smoke', b'profile=sequence'), (b'launches=2199', b'launches=2198'), (b'instances=1/1', b'instances=1/2'), (b'frees=16', b'frees=15'), (b'releases=10', b'releases=9'), (b'layer-21.f32le', b'layer-20.f32le'), (b'committed_position=1', b'committed_position=0'), (b'peak_requested_bytes=4096', b'peak_requested_bytes=0'), (b'phase=after_cleanup pool_bytes=12882804736 largest_free_bytes=12882804736', b'phase=after_cleanup pool_bytes=12882804736 largest_free_bytes=12000002048')]:
            with self.subTest(old=old), self.assertRaises(ValueError):
                log.check_full_log(raw.replace(old, new))
        for bad in (raw[:-1], raw + b'extra\n', raw.replace(b'\n', b'\r\n')):
            with self.assertRaises(ValueError):
                log.check_full_log(bad)


if __name__ == '__main__':
    unittest.main()
