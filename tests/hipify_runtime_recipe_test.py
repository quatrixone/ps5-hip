#!/usr/bin/env python3
"""Strict HIPIFY runtime result boundaries; optional saved actual proof check."""
import pathlib, sys, unittest
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from probe_hipify_runtime import MODES, TERMINAL, COUNTS, CLIENT_PASS, validate_model, check

def transcript(mode):
    text = 'VECTOR_REGISTER owned=true gpu_calls=0\nVECTOR_FUNCTION bound=true\n'+f'HIPIFY_MODEL_BEGIN mode={mode} model_only=true\n'
    if mode in TERMINAL:
        live, freed, unresolved = (4,3,0) if mode=='cleanup' else (7,0,1)
        return text+f'TRANSLATED_CLIENT_FAIL clean=0 error=injected\nHIPIFY_QUARANTINE live={live} releases={freed} unresolved={unresolved} submits=1 no_retry=true\n'
    backend,submitted,released,completed,finalized = COUNTS[mode]
    text += ''.join(f'RUNTIME_COMPLETE sequence={i} kernel=translated_add blocking=true\n' for i in range(1,completed+1))
    success = mode in ('normal','late-partial','no-finish')
    text += (CLIENT_PASS+'\n') if success else 'TRANSLATED_CLIENT_FAIL clean=1 error=injected\n'
    text += 'HIPIFY_RUNTIME_PASS launches=6 allocations=closed stale_rejected=true\n' if mode=='normal' else f'HIPIFY_EXPECTED_REJECT mode={mode} result={0 if success else 1} completed={completed}\n'
    unfinished = mode=='no-finish'
    text += ('VECTOR_UNREGISTER rejected=true\n' if unfinished else 'VECTOR_UNREGISTER after_main=true active=0\n')
    text += f'VECTOR_FINAL registered=1 removed={int(not unfinished)} completed={completed} active={int(unfinished)} terminal=0 main_finished={int(not unfinished)} hardware_validated=false\n'
    return text+f'HIPIFY_MODEL_FINAL live=0 backends={backend} submits={submitted} releases={released} completed={completed} modules={int(unfinished)} functions={int(unfinished)} finalized={finalized} model_only=true\n'

class Boundaries(unittest.TestCase):
    def reject(self, text, mode, code=None):
        with self.assertRaises(ValueError): validate_model(text,mode,code if code is not None else 30 if mode in TERMINAL else 0 if mode=='normal' else 20)
    def test_all_declared_modes(self):
        for mode in MODES: validate_model(transcript(mode),mode,30 if mode in TERMINAL else 0 if mode=='normal' else 20)
    def test_missing_duplicate_or_wrong_result(self):
        text=transcript('normal')
        for bad in (text+text,text.replace(CLIENT_PASS,''),text.replace('launches=6','launches=5'),text.replace('stale_rejected=true','stale_rejected=false')): self.reject(bad,'normal')
    def test_lifetime_and_exact_failure_point(self):
        for mode in COUNTS:
            text=transcript(mode)
            for bad in (text.replace('live=0','live=1'),text.replace('modules=','modules=9'),text.replace('functions=','functions=9'),text.replace('submits=','submits=9'),text.replace('releases=','releases=9')): self.reject(bad,mode)
    def test_partial_or_unfinished_never_finalized(self):
        for mode in ('partial','late-partial','no-finish','init'):self.reject(transcript(mode).replace('finalized=0','finalized=1'),mode)
    def test_unsafe_terminal_false_closure(self):
        for mode in TERMINAL:
            text=transcript(mode)
            for suffix in (CLIENT_PASS,'HIPIFY_MODEL_FINAL','VECTOR_FINAL','VECTOR_UNREGISTER','HIPIFY_RUNTIME_PASS'): self.reject(text+suffix,mode)
            self.reject(text.replace('unresolved=','unresolved=9'),mode)
            self.reject(text.replace('live=','live=9'),mode)
    def test_actual_compiler_lifecycle_required(self):
        for token in ('VECTOR_REGISTER','VECTOR_FUNCTION','VECTOR_UNREGISTER','VECTOR_FINAL'):
            self.reject(transcript('normal').replace(token,'WRONG'),'normal')
        self.reject(transcript('normal').replace('sequence=2','sequence=3'),'normal')
        text=transcript('normal');line='VECTOR_UNREGISTER after_main=true active=0\n'
        self.reject(line+text.replace(line,''),'normal')
        self.reject(transcript('normal').replace('mode=normal','mode=corrupt'),'normal')
    def test_reject_is_not_oracle_success(self):
        for mode in ('missing','corrupt','input','guard','stale-second','init','preflush','data-allocate'):
            text=transcript(mode);self.reject(text+CLIENT_PASS+'\n',mode);self.reject(text.replace('clean=1','clean=0'),mode)
            self.reject(text.replace('mode='+mode,'mode=unrelated'),mode)
    def test_diagnostics_and_exit(self):
        for fault in ('AddressSanitizer','UndefinedBehaviorSanitizer','runtime error:','HIPIFY_MODEL_FAILURE'):self.reject(transcript('normal')+fault,'normal')
        self.reject(transcript('normal'),'normal',20)

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--check': check(pathlib.Path(sys.argv[2]).resolve())
    else: unittest.main()
