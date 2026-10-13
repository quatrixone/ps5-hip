"""Strict model transcript qualification; does not execute target ISA."""
import argparse
import json
from pathlib import Path
import unittest

def validate_model(log,mode,code):
    def need(value):
        if not value:raise ValueError('invalid vector model transcript: '+mode)
    uncertain=mode in ('submit','kick','timeout','throw-submit','unknown-submit','throw-kick','postflush')
    cleanup=mode in ('init-unmap','init-release','preflush-unmap','preflush-release','marker-map-unmap','completed-unmap','completed-release')
    setup=mode in ('code-allocate','kernarg-map','command-map','marker-map')
    terminal=uncertain or cleanup
    corrupt=mode in ('missing','corrupt','input','guard')
    need(code==(20 if corrupt else 0))
    need(log.count('VECTOR_REGISTER owned=true gpu_calls=0')==1)
    need(log.count('VECTOR_FUNCTION bound=true')==(0 if mode=='partial' else 1))
    completed=0 if terminal or setup or mode in ('partial','init') else 1 if corrupt else 2
    need(log.count('VECTOR_COMPLETE sequence=')==completed)
    for i in range(1,completed+1):need(f'VECTOR_COMPLETE sequence={i} n=73 blocking=true' in log)
    fields=f'registered=1 removed={0 if terminal else 1} completed={completed} active={1 if terminal else 0} terminal={int(terminal)} main_finished={int(not terminal)}'
    retained=mode not in ('normal','preflush')
    final=(f'RUNTIME_RETAINED {fields} contexts=0 pending=0 abandoned={int(terminal)} registration_failed={int(mode=="partial")} hardware_validated=false'
           if retained else f'VECTOR_FINAL {fields} hardware_validated=false')
    need(log.count(final)==1)
    need(('VECTOR_FINAL ' if retained else 'RUNTIME_RETAINED ') not in log)
    if terminal:
        need(('VECTOR_CLIENT_QUARANTINE_PASS retained=7 no_successful_unregister=true model_only=true' if uncertain else
              'VECTOR_CLIENT_CLEANUP_FAILURE_PASS gpu_unresolved=false no_successful_unregister=true model_only=true') in log)
        need(log.count('VECTOR_UNREGISTER rejected=true')==1)
        need('VECTOR_UNREGISTER after_main=true' not in log)
    else:
        need(log.count('VECTOR_UNREGISTER after_main=true active=0')==1)
        need('VECTOR_UNREGISTER rejected=true' not in log)
    if mode=='partial':need('VECTOR_CLIENT_PARTIAL_PASS no_gpu=true' in log)
    elif mode=='init':need('VECTOR_CLIENT_INIT_FAILURE_PASS no_retry=true' in log)
    elif setup:need('VECTOR_CLIENT_SETUP_FAILURE_PASS clean_unwind=true' in log)
    elif not corrupt and not terminal:need('VECTOR_CLIENT_PASS exact_values=146 immutable_inputs=true full_guards=true blocking=true hardware_validated=false' in log)
    if corrupt or terminal:need('VECTOR_CLIENT_PASS ' not in log)
    backends=0 if mode in ('partial','code-allocate','kernarg-map') else 1
    submits=1 if uncertain or corrupt or mode in ('completed-unmap','completed-release') else 2 if mode in ('normal','preflush') else 0
    # Failed finish retains State; known-complete cleanup faults retain its three
    # data owners instead of allowing automatic State destruction to free them.
    prefix=f'VECTOR_MODEL_FINAL allocations={7 if uncertain else 3 if cleanup else 0} backends={backends} submits={submits} releases='
    need(log.count(prefix)==1)
    if terminal:
        released=1 if uncertain else 3 if mode in ('init-unmap','init-release') else 5
        need(log.count(prefix+str(released)+' model_only=true')==1)
    need(log.count(f'VECTOR_MODEL_CLEANUP failures={int(cleanup)}')==1)
    need(log.index('VECTOR_REGISTER')<log.index('VECTOR_UNREGISTER')<log.index(final)<log.index(prefix))

class Tests(unittest.TestCase):
    def test_actual_transcripts_and_mutations(self):
        if OUTPUT is None:self.skipTest('completed vector probe required')
        for path in sorted(OUTPUT.glob('model-*.stdout')):
            mode=path.stem[6:];code=20 if mode in ('missing','corrupt','input','guard') else 0
            log=path.read_text();validate_model(log,mode,code)
            finished='main_finished=0' if 'main_finished=0' in log else 'main_finished=1'
            for before,after in [('VECTOR_REGISTER','LOST_REGISTER'),('VECTOR_UNREGISTER','LOST_UNREGISTER'),
                                 (finished,'main_finished='+str(1-int(finished[-1]))),('hardware_validated=false','hardware_validated=true'),
                                 ('VECTOR_MODEL_FINAL allocations=','VECTOR_MODEL_FINAL allocations=9')]:
                with self.assertRaises(ValueError):validate_model(log.replace(before,after),mode,code)
            with self.assertRaises(ValueError):validate_model(log,mode,1 if code==0 else 0)
        self.assertEqual(len(list(OUTPUT.glob('model-*.stdout'))),26)

OUTPUT=None
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args,rest=parser.parse_known_args();OUTPUT=args.output
    unittest.main(argv=[__file__]+rest)
