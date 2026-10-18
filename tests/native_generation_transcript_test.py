import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from source_llm_generation_log import check_fragment, check_corpus_fragment
from build_llm_generation_reference import corpus_cases


def fixture(processed_count=5):
    policy = dict(max_new_tokens=4, runs=[dict(processed=list(range(1,processed_count+1)), generated=list(range(processed_count-2,processed_count+2)), stop='step_limit', text_hex='6162') for _ in range(2)])
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
        lines.append(f'M4_GEN RUN_TIME run={run} begin_ns={began} end_ns={tick} elapsed_ns={tick-began} decode_ns={processed_count*10} scheduling_sampling_log_ns={tick-began-processed_count*10}')
        lines.append(f'M4_GEN RESULT run={run} processed={processed_count} generated=4 stop=step_limit text_hex=6162')
    lines += ['M4_GEN METADATA files=2 generation_bytes=500 steps_bytes=700 synced=true', 'M3_FULL PHASE cleanup_begin']
    phase('cleanup')
    lines.append(f'M4_GEN TRANSPORT_COMPLETE requests={processed_count*2} layers=22 logits=32000 nodes={processed_count*2*798} launches={processed_count*2*733} raw_files={processed_count*48} raw_bytes={processed_count*2*316416} metadata_files=2 metadata_bytes=1200 instances=1/1 numerical=pending finalization=pending')
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


def corpus_fixture():
    policies = {}; sizes = {'corpus.json': 123}
    lines = ['M4_GEN PROFILE name=greedy-corpus cases=5 runs=2 max_new_tokens=16 context=512', 'M3_FULL PHASE load_begin']
    tick = 100
    def phase(scope, run=0, key=None):
        nonlocal tick
        tag = ' case=' + key if key else ''
        lines.append(f'M4_GEN TIMING{tag} scope={scope} run={run} position=0 begin_ns={tick} end_ns={tick+2} elapsed_ns=2'); tick += 3
    phase('load'); lines.append('M3_FULL PHASE load_complete')
    phase('context'); lines.append('M3_FULL PHASE context_complete')
    requests = 0
    for index, (key, prompt) in enumerate(corpus_cases()):
        tokens = [1, index + 3]; generated = list(range(100 + index, 116 + index))
        runs = [dict(run=run, processed=tokens + generated[:-1], generated=generated, stop='step_limit', text_hex='6162') for run in range(2)]
        policies[key] = dict(prompt=prompt, max_new_tokens=16, prompt_tokens=tokens, eog_tokens=[2], runs=runs)
        sizes[key + '/generation.json'] = 500; sizes[key + '/steps.jsonl'] = 700
        for run, row in enumerate(runs):
            phase('reset', run, key); began = tick
            for position, token in enumerate(row['processed']):
                for stage in [*(f'layer-{i}' for i in range(22)), 'norm', 'logits']:
                    size = 128000 if stage == 'logits' else 8192
                    lines.append(f'M3_FULL RAW file={key}/run-{run}-pos-{position}-{stage}.f32le bytes={size} synced=true')
                lines.append(f'M4_GEN DECODE case={key} run={run} position={position} token={token} committed_position={position} layers=22 logits=32000 begin_ns={tick} end_ns={tick+10} elapsed_ns=10 readback_ns=3 inclusive_readback=true')
                tick += 12; requests += 1
            elapsed = tick - began; decoded = len(row['processed']) * 10
            lines.append(f'M4_GEN RUN_TIME case={key} run={run} begin_ns={began} end_ns={tick} elapsed_ns={elapsed} decode_ns={decoded} scheduling_sampling_log_ns={elapsed-decoded}')
            lines.append(f'M4_GEN RESULT case={key} run={run} processed=17 generated=16 stop=step_limit text_hex=6162')
        lines.append(f'M4_GEN METADATA case={key} files=2 generation_bytes=500 steps_bytes=700 synced=true')
    lines.append('M4_GEN CORPUS cases=5 runs=2 max_new_tokens=16 index_bytes=123 synced=true')
    lines.append('M3_FULL PHASE cleanup_begin'); phase('cleanup')
    lines.append(f'M4_GEN TRANSPORT_COMPLETE requests={requests} layers=22 logits=32000 nodes={requests*798} launches={requests*733} raw_files={requests*24} raw_bytes={requests*316416} metadata_files=11 metadata_bytes={sum(sizes.values())} instances=1/1 numerical=pending finalization=pending')
    return ('\n'.join(lines)+'\n').encode(), policies, sizes


class CorpusTranscript(unittest.TestCase):
    def test_all_cases_global_counts(self):
        result = check_corpus_fragment(*corpus_fixture())
        self.assertEqual((result['requests'], result['raw_files'], result['metadata_files'], result['instances']), (170, 4080, 11, 1))
        self.assertEqual(len(result['timings']), 183)
        self.assertEqual({row['case'] for row in result['timings'] if 'case' in row}, {key for key, _ in corpus_cases()})
        for field in ('gpu_origin_established', 'numerical_verified', 'finalization_verified'):
            self.assertFalse(result[field])

    def test_cross_case_history_files_and_totals(self):
        raw, policy, sizes = corpus_fixture()
        for old, new in [(b'file=story/', b'file=science/'), (b'case=science run=0', b'case=story run=0'),
                         (b'case=science scope=reset', b'case=story scope=reset'),
                         (b'run=1 position=0 token=1', b'run=1 position=0 token=101'),
                         (b'committed_position=0', b'committed_position=1'),
                         (b'run-0-pos-1-layer', b'run-0-pos-0-layer'),
                         (b'generated=16', b'generated=15'), (b'stop=step_limit', b'stop=eos'),
                         (b'text_hex=6162', b'text_hex=6163'), (b'bytes=8192', b'bytes=8191'),
                         (b'requests=170', b'requests=171'), (b'nodes=135660', b'nodes=135661'),
                         (b'launches=124610', b'launches=124611'), (b'raw_files=4080', b'raw_files=4081'),
                         (b'raw_bytes=53790720', b'raw_bytes=53790721'),
                         (b'metadata_files=11', b'metadata_files=10'), (b'metadata_bytes=6123', b'metadata_bytes=6124'),
                         (b'index_bytes=123', b'index_bytes=124'), (b'instances=1/1', b'instances=2/2')]:
            self.assertIn(old, raw)
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_corpus_fragment(raw.replace(old, new, 1), policy, sizes)

    def test_clock_and_case_reset_cannot_restart(self):
        raw, policy, sizes = corpus_fixture()
        for old, new in [(b'readback_ns=3', b'readback_ns=11'), (b'elapsed_ns=10 readback', b'elapsed_ns=9 readback'),
                         (b'decode_ns=170', b'decode_ns=171'), (b'scheduling_sampling_log_ns=34', b'scheduling_sampling_log_ns=33')]:
            self.assertIn(old, raw)
            with self.subTest(old=old), self.assertRaises(ValueError):
                check_corpus_fragment(raw.replace(old, new, 1), policy, sizes)
        lines = raw.splitlines(); index = next(i for i, line in enumerate(lines) if b'TIMING case=science scope=reset' in line)
        lines[index] = b'M4_GEN TIMING case=science scope=reset run=0 position=0 begin_ns=100 end_ns=102 elapsed_ns=2'
        with self.assertRaises(ValueError): check_corpus_fragment(b'\n'.join(lines)+b'\n', policy, sizes)

    def test_framing_extra_and_missing(self):
        raw, policy, sizes = corpus_fixture()
        for data in [raw[:-1], raw+b'PASS made up\n', raw+b'\0', raw.replace(b'\n', b'\r\n'), b'x'*(2*1024*1024+1), b'\n'.join(raw.split(b'\n')[:20]+raw.split(b'\n')[21:])]:
            with self.subTest(data=data[:30]), self.assertRaises(ValueError): check_corpus_fragment(data, policy, sizes)

    def test_native_memory_is_required_but_not_origin_proof(self):
        raw, policy, sizes = corpus_fixture()
        with self.assertRaises(ValueError): check_corpus_fragment(raw, policy, sizes, native=True)
        for phase, name, free in [('load_begin','before_load',12882804736), ('load_complete','after_load',12000002048), ('context_complete','after_context',11000004608), ('cleanup_begin','after_cleanup',12882804736)]:
            marker = f'M3_FULL PHASE {phase}\n'.encode()
            row = f'M0_APP MEMORY phase={name} pool_bytes=12882804736 largest_free_bytes={free} total_free_bytes=unknown alignment=16384\n'.encode()
            raw = raw.replace(marker, marker+row)
        result = check_corpus_fragment(raw, policy, sizes, native=True)
        self.assertEqual(len(result['memory']), 4)
        self.assertFalse(result['gpu_origin_established'])
        with self.assertRaises(ValueError): check_corpus_fragment(raw.replace(b'phase=after_cleanup', b'phase=after_load'), policy, sizes, native=True)

    def test_policy_and_metadata_fail_closed(self):
        raw, policy, sizes = corpus_fixture()
        for name, value in [('max_new_tokens', 4), ('prompt', 'wrong'), ('prompt_tokens', [True]), ('eog_tokens', [100])]:
            bad = copy.deepcopy(policy); bad['story'][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError): check_corpus_fragment(raw, bad, sizes)
        for name, value in [('generated', [100]*15), ('processed', [1,3]+[101]*15), ('stop', 'eos'), ('text_hex', 'x'), ('run', True)]:
            bad = copy.deepcopy(policy); bad['story']['runs'][1][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError): check_corpus_fragment(raw, bad, sizes)
        bad = copy.deepcopy(policy); del bad['story']
        with self.assertRaises(ValueError): check_corpus_fragment(raw, bad, sizes)
        for bad_sizes in [{}, {**sizes, 'corpus.json': True}, {**sizes, 'corpus.json': 65537}, {**sizes, 'extra': 1}]:
            with self.assertRaises(ValueError): check_corpus_fragment(raw, policy, bad_sizes)



if __name__ == '__main__':
    unittest.main()
