import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from source_llm_generation_log import check_fragment


def fixture():
    policy = dict(max_new_tokens=4, runs=[dict(processed=[1, 2, 3, 4, 5], generated=[3, 4, 5, 6], stop='step_limit', text_hex='6162') for _ in range(2)])
    sizes = {'generation.json': 500, 'steps.jsonl': 700}
    lines = ['M4_GEN PROFILE name=greedy-short runs=2 max_new_tokens=4 context=512', 'M3_FULL PHASE load_begin']
    tick = 100

    def phase(scope, run=0):
        nonlocal tick
        lines.append(f'M4_GEN TIMING scope={scope} run={run} position=0 begin_ns={tick} end_ns={tick+2} elapsed_ns=2')
        tick += 3

    phase('load'); lines.append('M3_FULL PHASE load_complete')
    phase('context'); lines.append('M3_FULL PHASE context_complete')
    for run, item in enumerate(policy['runs']):
        phase('reset', run); began = tick
        for pos, token in enumerate(item['processed']):
            for stage in [*(f'layer-{i}' for i in range(22)), 'norm', 'logits']:
                size = 128000 if stage == 'logits' else 8192
                lines.append(f'M3_FULL RAW file=run-{run}-pos-{pos}-{stage}.f32le bytes={size} synced=true')
            lines.append(f'M4_GEN DECODE run={run} position={pos} token={token} committed_position={pos} layers=22 logits=32000 begin_ns={tick} end_ns={tick+10} elapsed_ns=10 readback_ns=3 inclusive_readback=true')
            tick += 12
        lines.append(f'M4_GEN RUN_TIME run={run} begin_ns={began} end_ns={tick} elapsed_ns={tick-began} decode_ns=50 scheduling_sampling_log_ns={tick-began-50}')
        lines.append(f'M4_GEN RESULT run={run} processed=5 generated=4 stop=step_limit text_hex=6162')
    lines += ['M4_GEN METADATA files=2 generation_bytes=500 steps_bytes=700 synced=true', 'M3_FULL PHASE cleanup_begin']
    phase('cleanup')
    lines.append('M4_GEN TRANSPORT_COMPLETE requests=10 layers=22 logits=32000 nodes=7980 launches=7330 raw_files=240 raw_bytes=3164160 metadata_files=2 metadata_bytes=1200 instances=1/1 numerical=pending finalization=pending')
    return ('\n'.join(lines)+'\n').encode(), policy, sizes


class GenerationTranscript(unittest.TestCase):
    def test_controls_timings_only(self):
        result = check_fragment(*fixture())
        self.assertEqual((result['requests'], result['raw_files']), (10, 240))
        self.assertFalse(result['gpu_origin_established'])
        self.assertFalse(result['numerical_verified'])
        self.assertFalse(result['finalization_verified'])

    def test_mutations_rejected(self):
        raw, policy, sizes = fixture()
        for old, new in [(b'token=1 committed', b'token=7 committed'), (b'committed_position=0', b'committed_position=1'),
                         (b'readback_ns=3', b'readback_ns=11'), (b'elapsed_ns=10 readback', b'elapsed_ns=11 readback'),
                         (b'begin_ns=100 end_ns=102', b'begin_ns=0 end_ns=102'),
                         (b'begin_ns=103 end_ns=105', b'begin_ns=99 end_ns=101'),
                         (b'decode_ns=50', b'decode_ns=51'), (b'stop=step_limit', b'stop=eos'),
                         (b'bytes=8192', b'bytes=8191'), (b'instances=1/1', b'instances=1/2'),
                         (b'metadata_bytes=1200', b'metadata_bytes=1201'),
                         (b'run-0-pos-1-layer', b'run-0-pos-0-layer')]:
            self.assertIn(old, raw)
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_fragment(raw.replace(old, new, 1), policy, sizes)

    def test_framing_and_missing_stage(self):
        raw, policy, sizes = fixture()
        for data in [raw[:-1], raw+b'extra\n', raw.replace(b'\n', b'\r\n'), raw+b'\0', b'x'*65537, b'\n'.join(raw.split(b'\n')[:20]+raw.split(b'\n')[21:])]:
            with self.subTest(data=data[:20]), self.assertRaises(ValueError):
                check_fragment(data, policy, sizes)

    def test_metadata_sizes(self):
        raw, policy, sizes = fixture()
        for bad in [{}, {**sizes, 'steps.jsonl': True}, {**sizes, 'generation.json': 65537}]:
            with self.assertRaises(ValueError):
                check_fragment(raw, policy, bad)


if __name__ == '__main__':
    unittest.main()
