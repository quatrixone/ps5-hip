"""Offline public HIP bootstrap assessment; synthetic log is not native evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
spec = importlib.util.spec_from_file_location('private_adapter', HERE / 'acceptance.py')
a = importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
OBS = ROOT / 'results/arithmetic-prospective-cycle2/observation-20261013T022218.879692Z'
from check_log import EXPECTED as APPLICATION


def deployment():
    return dict(host='192.168.4.40', title_id='PPSA99014', kernel='public_hip_bootstrap',
                source_commit=a.SOURCE, image_sha256=a.IMAGE_SHA,
                remote='/data/homebrew/PPSA99014.ffpkg', stage='registered',
                registered=True, launched=False, hardware_gate_passed=False,
                accepted=True, owned_locks_released=True, lock_errors=[],
                health_after={'2121':'reachable','3232':'reachable','9021':'reachable'},
                image={'file':'PPSA99014.ffpkg','bytes':9437184,'sha256':a.IMAGE_SHA})


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        receipt=json.loads((OBS/'receipt.json').read_text())
        def read(name):
            data=(OBS/name).read_bytes()
            if hashlib.sha256(data).hexdigest()!=receipt['raw_sha256'][name]:raise ValueError('raw fixture changed')
            return data
        cls.pre=read('pre-snapshot.log');cls.post=read('post-snapshot.log')
        cls.shadow=read('launch-shadow-delta.log');cls.klog=read('launch-klog.log')

    def assess(self, **changes):
        args=dict(pre_bytes=self.pre,post_bytes=self.post,shadow_delta_bytes=self.shadow,
                  klog_bytes=self.klog,application_log=APPLICATION,stopped_before_close=True,
                  prerequisites={key:True for key in a.PREREQUISITES},deployment_receipt=deployment())
        args.update(changes);result=a.assess_cycle(**args)
        self.assertIs(result['public_hip_bootstrap_verified'],result['accepted']);self.assertIs(result['hardware_gate_passed'],False)
        return result

    def test_model_conjunction_and_no_broad_gate_claim(self):
        result=self.assess()
        self.assertTrue(result['public_hip_bootstrap_cycle_accepted']);self.assertEqual(result['classification'],'pass')
        self.assertFalse(result['hardware_gate_passed'])
        self.assertEqual(result['lifecycle']['pid'],114)

    def test_new_lvd_baseline_but_exact_post_identity_required(self):
        before=self.pre.replace(b'from_hex=2f6465762f6c766432',b'from_hex=2f6465762f6c766433')
        after=self.post.replace(b'from_hex=2f6465762f6c766432',b'from_hex=2f6465762f6c766433')
        self.assertTrue(self.assess(pre_bytes=before,post_bytes=after)['accepted'])
        self.assertFalse(self.assess(pre_bytes=before)['accepted'])
        for invalid in (b'2f6465762f6c766478',b'2f6465762f6c7664322f',b'2f746d702f6c766432'):
            self.assertFalse(self.assess(pre_bytes=self.pre.replace(b'2f6465762f6c766432',invalid))['accepted'])

    def test_deployment_identity_and_completed_registration_required(self):
        for key,value in [('host','192.168.4.30'),('title_id','PPSA99008'),('kernel','arithmetic_suite'),
                          ('source_commit','wrong'),('image_sha256','0'*64),('stage','promoted'),
                          ('registered',1),('launched',True),('owned_locks_released',False),('error','failure'),
                          ('accepted',False),('health_after',{}),('idle_error','failure'),('health_error','failure'),
                          ('local_final_error','changed')]:
            with self.subTest(key=key):
                receipt=deployment();receipt[key]=value
                self.assertFalse(self.assess(deployment_receipt=receipt)['accepted'])
        receipt=deployment();receipt['image']['bytes']-=1
        self.assertFalse(self.assess(deployment_receipt=receipt)['accepted'])

    def test_each_control_prerequisite_required(self):
        for key in a.PREREQUISITES:
            p={item:True for item in a.PREREQUISITES};p[key]=False
            self.assertFalse(self.assess(prerequisites=p)['accepted'],key)
        self.assertFalse(self.assess(stopped_before_close=False)['accepted'])

    def test_exact_public_bootstrap_and_destructor_proof(self):
        for old,new in [(b'init=0',b'init=1'),(b'count=1',b'count=0'),(b'gate_destroyed=true',b'gate_destroyed=false'),(b'watchdog_joined=true',b'watchdog_joined=false'),(b'application-fini=completed',b'application-fini=pending')]:
            with self.subTest(old=old):
                self.assertIn(old,APPLICATION)
                self.assertFalse(self.assess(application_log=APPLICATION.replace(old,new))['accepted'])
        lines=APPLICATION.splitlines(keepends=True)
        bad=[b''.join(lines[:i]+lines[i+1:]) for i in range(len(lines))]
        bad += [b''.join(lines[:i]+[lines[i]]+lines[i:]) for i in range(len(lines))]
        changed=lines.copy();changed[2],changed[3]=changed[3],changed[2];bad.append(b''.join(changed))
        bad += [APPLICATION[:-1],APPLICATION+b'PS5HIP QUARANTINED\n',APPLICATION+b'PS5HIP FAIL\n',
                (OBS/'application.log').read_bytes()]
        for data in bad:self.assertFalse(self.assess(application_log=data)['accepted'])

    def test_lifecycle_other_pid_fault_missingphase(self):
        for changed in [self.klog.replace(b'pid=0x72',b'pid=0x73'),self.klog+b'SIGSEGV\n',
                        self.klog.replace(b'All processes exited',b'missingphase')]:
            self.assertFalse(self.assess(klog_bytes=changed)['accepted'])
        fault=self.assess(klog_bytes=self.klog+b'SIGSEGV\n')
        self.assertEqual(fault['classification'],'failed')
        # Retrospective transport fixture only, paired with synthetic matmul output:
        # proves this adapter really delegates to the reviewed prospective parser.
        saved=ROOT/'results/visibility-execution-v1/observation-20261013T043518.324248Z'
        receipt=json.loads((saved/'receipt.json').read_text())
        raw={name:(saved/name).read_bytes() for name in ('launch-shadow-delta.log','launch-klog.log')}
        for name,data in raw.items():self.assertEqual(hashlib.sha256(data).hexdigest(),receipt['raw_sha256'][name])
        args=dict(shadow_delta_bytes=raw['launch-shadow-delta.log'],klog_bytes=raw['launch-klog.log'])
        self.assertTrue(self.assess(**args)['accepted'])
        args['klog_bytes']+=b'SIGSEGV\n'
        self.assertEqual(self.assess(**args)['classification'],'failed')

    def test_complete_multiset_no_weakening(self):
        self.assertFalse(self.assess(pre_bytes=self.pre[:-1])['accepted'])
        self.assertFalse(self.assess(post_bytes=self.post.replace(b'flags=0x1000',b'flags=0x1001'))['accepted'])

    def test_frozen_checker_identity_fails_closed(self):
        old=a.CHECKER_SHA
        try:
            a.CHECKER_SHA='0'*64
            self.assertFalse(self.assess()['accepted'])
        finally:a.CHECKER_SHA=old


if __name__=='__main__':unittest.main(verbosity=2)
